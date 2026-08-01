# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, Optional
import typer
from typing_extensions import Annotated

from colab_cli.attach_models import AttachEnvelope, AttachError
from colab_cli.accelerators import (
    AcceleratorArgumentError,
    format_assignment_error,
    resolve_accelerator,
)
from colab_cli.client import ColabRequestError, PostAssignmentResponse
from colab_cli.observability import (
    DEFAULT_PROBE_TIMEOUT,
    collect_sessions,
    collect_status,
    emit_json,
    machine_diagnostics_to_stderr,
    redact_text,
    validate_probe_timeout,
)
from colab_cli.observability.collector import utc_now
from colab_cli.remote import open_remote_executor
from colab_cli.runtime import ColabRuntime
from colab_cli.state import SessionState
from colab_cli.utils import get_status_code


logger = logging.getLogger(__name__)
_ORPHAN_HISTORY_SESSION = "_orphan_assignments"


class InvalidAttachedSessionName(ValueError):
    pass


def _is_scope_error(e: Exception) -> bool:
    """True if a ColabRequestError's response body indicates a missing OAuth scope.

    The frontend returns a `google.rpc.Status` with `code=7` (PERMISSION_DENIED)
    and a `DebugInfo` payload mentioning `SCOPE_NOT_PERMITTED` /
    "insufficient authentication scopes". Match on either substring so we
    don't depend on the exact wording of one of them.
    """
    body = getattr(e, "response_body", None) or ""
    body_str = str(body)
    return (
        "SCOPE_NOT_PERMITTED" in body_str
        or "insufficient authentication scopes" in body_str
    )


def _scope_remediation_message(provider) -> str:
    """User-facing remediation hint, tailored per auth provider.

    Keep-alive is a Tunnel Frontend ping against the Colab session backend
    (colab.research.google.com), authenticated with the user's own Gaia bearer
    token — the same credential and host used to assign the VM. A missing-scope
    error here is rare (assignment would normally have failed first), but if it
    happens the fix is to re-authenticate with the standard Colab scopes.
    """
    # Importing locally to avoid a circular import at module load time.
    from colab_cli.auth import AuthProvider

    common = (
        "Keeping the session alive requires valid Colab credentials for "
        "colab.research.google.com."
    )
    if provider == AuthProvider.ADC:
        return (
            f"{common}\n"
            "Re-authenticate ADC with the standard Colab scopes (the "
            "cloud-platform and openid scopes are required by gcloud itself):\n"
            "  gcloud auth application-default login \\\n"
            "      --scopes=openid,"
            "https://www.googleapis.com/auth/cloud-platform,"
            "https://www.googleapis.com/auth/userinfo.email,"
            "https://www.googleapis.com/auth/colaboratory\n"
            "Then re-run `colab new`."
        )
    # OAuth2 (and any future provider) fallback.
    return (
        f"{common}\n"
        "Run `colab login --force` to replace the cached control-plane "
        "credentials, then re-run `colab new`."
    )


def _hardware_label(accelerator: str) -> str:
    """`NONE` -> `CPU`; everything else passes through."""
    return "CPU" if accelerator == "NONE" else accelerator


def _format_session_line(
    name: str,
    endpoint: str,
    accelerator: str,
    variant: str,
    status: Optional[str] = None,
) -> str:
    """Single source of truth for session display lines.

    Format: ``[name] endpoint | Hardware: X | Variant: Y[ | Status: Z]``.
    Use ``"?"`` as the name for orphaned server-side assignments with no local
    state.
    """
    parts = [
        f"[{name}] {endpoint}",
        f"Hardware: {_hardware_label(accelerator)}",
        f"Variant: {variant}",
    ]
    if status is not None:
        parts.append(f"Status: {status}")
    return " | ".join(parts)


def _rollback_allocated_session(
    state,
    *,
    name: str,
    endpoint: str,
    session_state: Optional[SessionState],
) -> bool:
    """Rollback setup without discarding the only recovery handle.

    Returns ``True`` only when the backend accepted the unassign request. If the
    result is ambiguous, a usable local SessionState is retained so ``colab
    stop -s NAME`` or a later reconciliation can retry cleanup.
    """

    if session_state is not None and session_state.keep_alive_pid:
        try:
            from colab_cli.common import kill_process

            kill_process(session_state.keep_alive_pid)
            session_state.keep_alive_pid = None
        except Exception as cleanup_error:
            typer.echo(
                f"[colab] Failed to stop keep-alive during rollback: {cleanup_error}",
                err=True,
            )

    try:
        state.client.unassign(endpoint)
    except Exception as cleanup_error:
        typer.echo(
            f"[colab] Could not confirm release of endpoint '{endpoint}' during "
            f"rollback: {cleanup_error}",
            err=True,
        )
        if session_state is not None:
            session_state.running = None
            try:
                state.store.add(session_state)
                typer.echo(
                    f"[colab] Retained session '{name}' locally; retry cleanup with "
                    f"`colab stop -s {name}`.",
                    err=True,
                )
            except Exception as state_error:
                typer.echo(
                    "[colab] Failed to retain rollback recovery state: "
                    f"{state_error}. Endpoint requiring manual inspection: {endpoint}",
                    err=True,
                )
        return False

    try:
        state.store.remove(name)
    except Exception as cleanup_error:
        typer.echo(
            "[colab] Endpoint was released, but local rollback state could not "
            f"be removed: {cleanup_error}",
            err=True,
        )
    return True


def new(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    tpu: Annotated[
        Optional[str],
        typer.Option(
            help="TPU accelerator variant. Supported: v5e1, v6e1.",
        ),
    ] = None,
    gpu: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "GPU accelerator variant. Supported: T4, L4, G4, H100, A100."
                "\n\nIf omitted (along with --tpu), a CPU runtime is created."
                "\n\nAvailability varies by Colab subscription tier."
            ),
        ),
    ] = None,
):
    """Create a new session"""
    from colab_cli.common import state

    name = session or uuid.uuid4().hex[:6]
    if state.store.get(name) is not None:
        typer.echo(
            f"[colab] Session '{name}' already exists. Stop it or choose a new name.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        variant, accelerator = resolve_accelerator(gpu=gpu, tpu=tpu)
    except AcceleratorArgumentError as error:
        typer.echo(f"[colab] {error}", err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"[colab] Creating session '{name}'...")
    try:
        res = state.client.assign(
            uuid.uuid4(), variant=variant, accelerator=accelerator
        )
    except ColabRequestError as error:
        typer.echo(format_assignment_error(error, accelerator), err=True)
        raise typer.Exit(code=1) from error

    endpoint = res.endpoint
    s = None
    try:
        if isinstance(res, PostAssignmentResponse):
            token = res.runtime_proxy_info.token
            url = res.runtime_proxy_info.url
        else:
            token = (
                res.runtime_proxy_info.token
                if hasattr(res, "runtime_proxy_info")
                else getattr(res, "runtime_proxy_token", "")
            )
            url = (
                res.runtime_proxy_info.url if hasattr(res, "runtime_proxy_info") else ""
            )

        s = SessionState(
            name=name,
            token=token,
            url=url,
            endpoint=endpoint,
            variant=variant.value,
            accelerator=accelerator.value,
        )

        # Pre-flight the keep-alive ping once. If it returns a 403 caused by
        # missing OAuth scopes we know the daemon will fail and the VM would be
        # idle-pruned. Catch it now so we surface actionable remediation instead
        # of a session that quietly disappears a few minutes later.
        try:
            state.client.keep_alive_assignment(endpoint)
        except ColabRequestError as e:
            if get_status_code(e) == 403 and _is_scope_error(e):
                typer.echo(
                    "[colab] Keep-alive pre-flight failed: your credentials "
                    "are missing an OAuth scope required by Colab.\n",
                    err=True,
                )
                typer.echo(_scope_remediation_message(state.auth_provider), err=True)
                raise typer.Exit(code=1)
            # Other failures do not block session creation; the daemon retries
            # and logs via the existing keep_alive_error event path.

        # Persist before spawning so the daemon's initial state lookup cannot
        # race. Persist again after spawn to record the PID.
        state.store.add(s)
        s.keep_alive_pid = spawn_keep_alive(
            endpoint,
            name,
            auth_provider=state.auth_provider,
            config_path=state.config_path,
            client_oauth_config=state.client_oauth_config,
        )
        state.store.add(s)
        state.history.log_event(
            name,
            "session_created",
            {
                "endpoint": endpoint,
                "variant": variant.value,
                "accelerator": accelerator.value,
            },
        )
    except BaseException:
        _rollback_allocated_session(
            state, name=name, endpoint=endpoint, session_state=s
        )
        raise

    typer.echo("[colab] Session READY.")


def _validate_attached_session_name(name: str) -> str:
    value = name.strip()
    if not value:
        raise InvalidAttachedSessionName("session name must not be empty")
    if len(value) > 128:
        raise InvalidAttachedSessionName("session name must be 128 characters or fewer")
    if any(ord(char) < 32 for char in value):
        raise InvalidAttachedSessionName("session name contains a control character")
    if any(char in value for char in '<>:"/\\|?*'):
        raise InvalidAttachedSessionName(
            "session name contains a Windows-unsafe filename character"
        )
    if value.endswith((" ", ".")):
        raise InvalidAttachedSessionName(
            "session name must not end with a space or dot"
        )
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if value.split(".", 1)[0].upper() in reserved:
        raise InvalidAttachedSessionName("session name is reserved on Windows")
    return value


def _emit_attach_result(
    envelope: AttachEnvelope,
    *,
    json_output: bool,
) -> None:
    if json_output:
        emit_json(envelope)
        return
    if envelope.ok:
        typer.echo(
            f"[colab] Attached '{envelope.session_name}' "
            f"to {envelope.endpoint} "
            f"({envelope.accelerator or 'UNKNOWN'}, "
            f"{envelope.variant or 'UNKNOWN'})."
        )
        for warning in envelope.warnings:
            typer.echo(
                f"[colab] Warning: {warning}",
                err=True,
            )
        return
    message = envelope.error.message if envelope.error else "unknown error"
    typer.echo(
        f"[colab] Attach failed: {message}",
        err=True,
    )


def attach(
    endpoint: Annotated[
        str,
        typer.Option(
            "--endpoint",
            help="Exact endpoint from `colab sessions --json`",
        ),
    ],
    session: Annotated[
        str,
        typer.Option(
            "-s",
            "--session",
            help="New local session name",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the stable colab.attach.v1 schema",
        ),
    ] = False,
    connect: Annotated[
        bool,
        typer.Option(
            "--connect/--no-connect",
            help=(
                "Establish and persist a control kernel before reporting attach success"
            ),
        ),
    ] = True,
):
    """Adopt an existing assignment without allocating or releasing it."""
    from colab_cli.common import kill_process, state

    name = session
    assignment = None
    local_state = None
    executor = None
    spawned_pid = None
    stored = False
    control_connected = False
    warnings: list[str] = []
    error_code = None
    error_message = None
    retryable = None

    try:
        name = _validate_attached_session_name(session)
        context = machine_diagnostics_to_stderr() if json_output else nullcontext()
        with context:
            local = state.store.list_strict()
            if name in local:
                raise RuntimeError(
                    "SESSION_NAME_CONFLICT: local session name already exists"
                )

            assignments = state.client.list_assignments(timeout=(5.0, 20.0))
            assignment = next(
                (item for item in assignments if item.endpoint == endpoint),
                None,
            )
            if assignment is None:
                raise LookupError(
                    "ASSIGNMENT_NOT_FOUND: endpoint is not "
                    "present in the current account assignment list"
                )

            conflict = next(
                (item for item in local.values() if item.endpoint == endpoint),
                None,
            )
            if conflict is not None:
                raise RuntimeError(
                    "ENDPOINT_ALREADY_ATTACHED: endpoint is "
                    f"already tracked as {conflict.name!r}"
                )

            # This verifies that the backend still accepts keep-alive for the
            # assignment. It does not allocate or release a runtime.
            state.client.keep_alive_assignment(endpoint)

            local_state = SessionState(
                name=name,
                token=assignment.runtime_proxy_info.token,
                url=assignment.runtime_proxy_info.url,
                endpoint=assignment.endpoint,
                variant=assignment.variant.name,
                accelerator=assignment.accelerator.value,
            )
            # Phase 1: persist recoverable endpoint/token state before any
            # local process or control-kernel setup can fail.
            state.store.claim_strict(local_state.model_copy())
            stored = True

            if connect:
                executor = open_remote_executor(
                    local_state,
                    state.store,
                    history=state.history,
                )
                try:
                    executor.execute_json(
                        "_colab_cli_result = {'attached': True}",
                        timeout=30.0,
                    )
                    control_connected = True
                finally:
                    try:
                        executor.close()
                    except Exception as close_error:
                        warnings.append(
                            "control connection closed with warning: "
                            + redact_text(
                                close_error,
                                secrets=(assignment.runtime_proxy_info.token,),
                            )
                        )
                    executor = None
            else:
                warnings.append(
                    "control connection deferred; the first "
                    "exec/jobs/monitor command will establish it"
                )

            spawned_pid = spawn_keep_alive(
                endpoint,
                name,
                auth_provider=state.auth_provider,
                config_path=state.config_path,
                client_oauth_config=state.client_oauth_config,
            )
            local_state.keep_alive_pid = spawned_pid

            # Phase 2: atomically publish the fully attached local state.
            state.store.update_claim_strict(local_state.model_copy())

            try:
                state.history.log_event(
                    name,
                    "session_attached",
                    {
                        "endpoint": endpoint,
                        "variant": assignment.variant.name,
                        "accelerator": (assignment.accelerator.value),
                        "machine_shape": (assignment.machine_shape.name),
                        "control_connected": control_connected,
                        "kernel_id": local_state.kernel_id,
                        "session_id": local_state.session_id,
                    },
                )
            except Exception as exc:
                warnings.append(
                    "assignment attached but history write failed: "
                    + redact_text(
                        exc,
                        secrets=(assignment.runtime_proxy_info.token,),
                    )
                )

    except Exception as exc:
        secret = assignment.runtime_proxy_info.token if assignment is not None else ""
        safe = redact_text(
            f"{type(exc).__name__}: {exc}",
            secrets=(secret,),
        )
        raw = str(exc)
        if "SESSION_NAME_CONFLICT:" in raw:
            error_code, retryable = (
                "SESSION_NAME_CONFLICT",
                False,
            )
        elif "ENDPOINT_ALREADY_ATTACHED:" in raw:
            error_code, retryable = (
                "ENDPOINT_ALREADY_ATTACHED",
                False,
            )
        elif "ASSIGNMENT_NOT_FOUND:" in raw:
            error_code, retryable = (
                "ASSIGNMENT_NOT_FOUND",
                True,
            )
        elif isinstance(exc, InvalidAttachedSessionName):
            error_code, retryable = (
                "INVALID_SESSION_NAME",
                False,
            )
        else:
            error_code, retryable = (
                "ATTACH_LOCAL_SETUP_FAILED",
                True,
            )
        error_message = safe.split(": ", 1)[-1]

        if executor is not None:
            try:
                executor.close()
            except Exception as cleanup_error:
                warnings.append(
                    "control connection rollback failed: "
                    + redact_text(
                        cleanup_error,
                        secrets=(secret,),
                    )
                )

        if spawned_pid is not None:
            try:
                kill_process(spawned_pid)
            except Exception as cleanup_error:
                warnings.append(
                    "keep-alive rollback failed: "
                    + redact_text(
                        cleanup_error,
                        secrets=(secret,),
                    )
                )

        if stored and local_state is not None:
            try:
                state.store.remove_claim_strict(
                    name,
                    local_state.endpoint,
                )
            except Exception as cleanup_error:
                warnings.append(
                    "local state rollback failed: "
                    + redact_text(
                        cleanup_error,
                        secrets=(secret,),
                    )
                )

    if error_code is not None:
        envelope = AttachEnvelope(
            ok=False,
            status="failed",
            session_name=name,
            endpoint=endpoint,
            warnings=warnings,
            error=AttachError(
                code=error_code,
                message=error_message or error_code,
                retryable=retryable,
            ),
        )
        _emit_attach_result(
            envelope,
            json_output=json_output,
        )
        raise typer.Exit(1)

    assert assignment is not None
    assert local_state is not None
    envelope = AttachEnvelope(
        ok=True,
        status="attached",
        session_name=name,
        endpoint=endpoint,
        accelerator=assignment.accelerator.value,
        variant=assignment.variant.name,
        machine_shape=assignment.machine_shape.name,
        keep_alive_pid=local_state.keep_alive_pid,
        control_connected=control_connected,
        kernel_id=local_state.kernel_id,
        session_id=local_state.session_id,
        attached_at=utc_now(),
        warnings=warnings,
    )
    _emit_attach_result(
        envelope,
        json_output=json_output,
    )


def restart_kernel(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
):
    """Restart a session's kernel"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)

    def on_started(kid):
        s.kernel_id = kid
        state.store.add(s)

    def on_sess_started(sid):
        s.session_id = sid
        state.store.add(s)

    runtime = ColabRuntime(
        s.url,
        s.token,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
        on_kernel_started=on_started,
        on_session_started=on_sess_started,
    )

    try:
        runtime.restart()
    finally:
        runtime.stop()


def sessions_command(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the stable colab.sessions.v1 schema")
    ] = False,
):
    """List all active sessions"""
    from colab_cli.common import state

    if json_output:
        with machine_diagnostics_to_stderr():
            envelope = collect_sessions(state)
        emit_json(envelope)
        if not envelope.ok:
            raise typer.Exit(1)
        return

    sessions, assignments = state.sync_sessions()
    if not assignments:
        typer.echo("[colab] No active sessions found on server.")
        return

    # Build endpoint -> local-name lookup so we can lead with the friendly name.
    name_by_endpoint = {s.endpoint: s.name for s in sessions.values()}
    for a in assignments:
        name = name_by_endpoint.get(a.endpoint, "?")
        # `a.variant` is an int-valued AssignmentVariant (DEFAULT=0/GPU=1/TPU=2);
        # its `.name` matches the user-facing string Variant enum, which is what
        # `status` shows for locally-tracked sessions.
        typer.echo(
            _format_session_line(
                name=name,
                endpoint=a.endpoint,
                accelerator=a.accelerator.value,
                variant=a.variant.name,
            )
        )


def _print_status_for(s: SessionState) -> None:
    """Print one session's status line plus optional last-execution detail."""
    status = f"BUSY ({s.running})" if s.running else "IDLE"
    typer.echo(
        _format_session_line(
            name=s.name,
            endpoint=s.endpoint,
            accelerator=s.accelerator,
            variant=s.variant,
            status=status,
        )
    )
    if s.last_execution:
        exec_file, exec_cell, exec_time = s.last_execution
        cell_str = f" | Cell: {exec_cell}" if exec_cell else ""
        typer.echo(f"  Last Execution: {exec_file}{cell_str} at {exec_time}")


def _probe_timeout_callback(value: float) -> float:
    try:
        return validate_probe_timeout(value)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def status(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the stable colab.status.v1 schema")
    ] = False,
    probe: Annotated[
        bool,
        typer.Option(
            "--probe",
            help="Probe one existing session without allocating or restarting it",
        ),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Total wall-clock budget for --probe in seconds",
            callback=_probe_timeout_callback,
        ),
    ] = DEFAULT_PROBE_TIMEOUT,
):
    """Show session status"""
    if probe and not json_output:
        raise typer.BadParameter("--probe requires --json", param_hint="--probe")
    if probe and session is None:
        raise typer.BadParameter(
            "--probe requires one explicit -s/--session", param_hint="--probe"
        )

    from colab_cli.common import state

    if json_output:
        with machine_diagnostics_to_stderr():
            envelope = collect_status(
                state, session_name=session, probe=probe, timeout=timeout
            )
        emit_json(envelope)
        if not envelope.ok:
            raise typer.Exit(1)
        return

    local_sessions, _ = state.sync_sessions()
    if session:
        s = state.store.get(session)
        if s:
            _print_status_for(s)
        else:
            typer.echo(f"[colab] Session '{session}' not found.")
        return

    if not local_sessions:
        typer.echo("[colab] No active sessions.")
        return
    for s in local_sessions.values():
        _print_status_for(s)


def stop(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    endpoint: Annotated[
        Optional[str],
        typer.Option(
            "--endpoint",
            help="Release an orphan server assignment by its exact endpoint",
        ),
    ] = None,
):
    """Stop a session"""
    from colab_cli.common import state

    if session is not None and endpoint is not None:
        typer.echo(
            "[colab] Error: --session and --endpoint are mutually exclusive.",
            err=True,
        )
        raise typer.Exit(2)

    if endpoint is not None:
        local_sessions = state.store.list()
        for local_name, local_session in local_sessions.items():
            if local_session.endpoint == endpoint:
                typer.echo(
                    "[colab] Error: endpoint is tracked by local session "
                    f"'{local_name}'. Use `colab stop -s {local_name}` so the "
                    "kernel and keep-alive process are also cleaned up.",
                    err=True,
                )
                raise typer.Exit(1)

        assignments = state.client.list_assignments()
        if endpoint not in {assignment.endpoint for assignment in assignments}:
            typer.echo(
                f"[colab] Error: endpoint '{endpoint}' is not active on the server.",
                err=True,
            )
            raise typer.Exit(1)

        state.client.unassign(endpoint)
        try:
            state.history.log_event(
                _ORPHAN_HISTORY_SESSION,
                "orphan_assignment_released",
                {"endpoint": endpoint},
            )
        except Exception as history_error:
            logger.warning(
                "Orphan assignment %s was released, but its history event "
                "could not be recorded: %s",
                endpoint,
                history_error,
            )
        typer.echo(f"[colab] Orphan assignment released: {endpoint}")
        return

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        return

    typer.echo(f"[colab] Stopping session '{name}'...")
    if s.keep_alive_pid:
        from colab_cli.common import kill_process

        kill_process(s.keep_alive_pid)
        s.keep_alive_pid = None
        try:
            state.store.add(s)
        except Exception as state_error:
            typer.echo(
                "[colab] Keep-alive stopped, but the cleared PID could not be "
                f"persisted: {state_error}",
                err=True,
            )

    try:
        runtime = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
        runtime.stop(shutdown_kernel=True)
    except Exception:
        pass

    try:
        state.client.unassign(s.endpoint)
    except Exception as cleanup_error:
        try:
            state.store.add(s)
        except Exception as state_error:
            typer.echo(
                "[colab] Runtime release was not confirmed and recovery state "
                f"could not be persisted: {state_error}. Endpoint: {s.endpoint}",
                err=True,
            )
        typer.echo(
            "[colab] Runtime release could not be confirmed; local session state "
            f"was retained. Retry with `colab stop -s {name}`.",
            err=True,
        )
        raise cleanup_error
    state.store.remove(name)
    state.history.log_event(name, "session_terminated", {"reason": "user_requested"})
    typer.echo("[colab] Session terminated.")


def spawn_keep_alive(
    endpoint: str,
    session_name: str,
    auth_provider=None,
    config_path=None,
    client_oauth_config=None,
):
    """Spawns a detached keep-alive process.

    Authentication and state paths are propagated as global flags so the
    detached child uses the same strategy and files as its parent.
    Without this, the child inherits Typer's defaults (`--auth=oauth2`,
    `--config=~/.config/colab-cli/sessions.json`), which causes:
      (a) wrong auth backend, and
      (b) the daemon's `state.store.get(session_name)` check finds nothing
          and exits with `reason=session_not_found` when the parent used
          `--config` to write to a non-default path.
    """
    cmd = [sys.executable, "-m", "colab_cli.entrypoint"]
    if auth_provider is not None:
        cmd.append(f"--auth={auth_provider.value}")
    if config_path is not None:
        cmd.extend(["--config", config_path])
    if client_oauth_config is not None:
        cmd.extend(["--client-oauth-config", client_oauth_config])
    cmd.extend(["keep-alive", endpoint, session_name])
    # Detach process
    kwargs = {}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    else:
        # Background-process behavior reference:
        # https://stackoverflow.com/questions/1356540/
        # how-can-i-make-a-python-script-run-in-the-background-as-a-service-on-windows
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **kwargs,
    )
    return p.pid


def keep_alive(
    endpoint: Annotated[str, typer.Argument(help="Endpoint ID")],
    session_name: Annotated[str, typer.Argument(help="Session name")],
):
    """Hidden command to run keep-alive loop. Terminate after 24h."""
    from colab_cli.common import state

    state.history.log_event(
        session_name,
        "keep_alive_started",
        {"endpoint": endpoint, "pid": os.getpid()},
    )

    start_time = time.time()
    # 24 hours limit
    max_duration = 24 * 3600
    consecutive_4xx = 0
    iterations = 0
    last_error: Optional[Dict[str, Any]] = None

    reason = "time_limit_reached"
    extra: Dict[str, Any] = {}
    while time.time() - start_time < max_duration:
        iterations += 1
        # Check if session still exists in local state
        s = state.store.get(session_name)
        if not s:
            reason = "session_not_found"
            break
        if s.endpoint != endpoint:
            reason = "endpoint_mismatch"
            extra["expected_endpoint"] = endpoint
            extra["actual_endpoint"] = s.endpoint
            break

        try:
            state.client.keep_alive_assignment(endpoint)
            consecutive_4xx = 0
            last_error = None
        except Exception as e:
            code = get_status_code(e)
            response_body = getattr(e, "response_body", None)
            err_info = {
                "status_code": code,
                "error_type": type(e).__name__,
                "error": str(e)[:500],
                "response_body": (str(response_body)[:1000] if response_body else None),
            }
            last_error = err_info
            state.history.log_event(
                session_name,
                "keep_alive_error",
                {
                    **err_info,
                    "iteration": iterations,
                    "consecutive_4xx": consecutive_4xx
                    + (1 if code is not None and 400 <= code < 500 else 0),
                },
            )
            if code is not None and 400 <= code < 500:
                consecutive_4xx += 1
                if consecutive_4xx >= 2:
                    reason = "consecutive_4xx_errors"
                    break
            else:
                # For other errors (network), we retry and don't count as 4xx
                pass

        time.sleep(60)

    payload: Dict[str, Any] = {
        "reason": reason,
        "iterations": iterations,
        "duration_seconds": round(time.time() - start_time, 2),
    }
    if last_error is not None:
        payload["last_error"] = last_error
    payload.update(extra)
    state.history.log_event(session_name, "keep_alive_stopped", payload)


def register(app: typer.Typer):
    app.command()(new)
    app.command()(attach)
    app.command(name="sessions")(sessions_command)
    app.command(name="restart-kernel")(restart_kernel)
    app.command()(status)
    app.command()(stop)
    app.command(hidden=True)(keep_alive)
