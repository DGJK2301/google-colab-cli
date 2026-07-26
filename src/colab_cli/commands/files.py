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

from __future__ import annotations

import click
from contextlib import nullcontext
import hashlib
import math
import os
import tempfile
from typing import Any, Callable, Optional

import typer
from typing_extensions import Annotated

from colab_cli.contents import ContentsClient
from colab_cli.observability import (
    SessionSelectionError,
    machine_diagnostics_to_stderr,
    redact_text,
    resolve_local_session_read_only,
)
from colab_cli.remote import (
    RemoteExecutionError,
    RemoteFileOps,
    open_remote_executor,
)
from colab_cli.transfer import (
    DEFAULT_CHUNK_SIZE,
    FileTransfer,
    TransferIntegrityError,
    TransferProgress,
    TransferResult,
)
from colab_cli.transfer_lease import (
    TransferLease,
    TransferLeaseBusy,
    TransferLeaseCorrupt,
    canonical_local_path,
    normalize_remote_path,
)
from colab_cli.transfer_output import (
    TransferTelemetry,
    build_resume_argv,
    emit_transfer_json,
    history_payload,
    progress_line,
)


_MIB = 1024 * 1024


class _TransferCommandError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class _TransferResources:
    def __init__(
        self,
        *,
        contents,
        executor,
        transfer,
    ) -> None:
        self.contents = contents
        self.executor = executor
        self.transfer = transfer
        self._closed = False

    def close(self) -> list[str]:
        if self._closed:
            return []
        self._closed = True
        errors = []
        for label, closer in (
            ("ContentsClient", self.contents.close),
            ("remote executor", self.executor.close),
        ):
            try:
                closer()
            except Exception as exc:
                errors.append(f"{label} close failed: {exc}")
        return errors


def _chunk_size_mib_to_bytes(value: float) -> int:
    if not math.isfinite(value):
        raise typer.BadParameter("must be a finite number")
    if value <= 0:
        raise typer.BadParameter("must be greater than 0")
    size_bytes = value * _MIB
    if not math.isfinite(size_bytes):
        raise typer.BadParameter("is too large")
    chunk_size = int(size_bytes)
    if chunk_size < 1:
        raise typer.BadParameter("must convert to at least 1 byte")
    return chunk_size


def _validate_chunk_size_mib(value: float) -> float:
    _chunk_size_mib_to_bytes(value)
    return value


def _legacy_progress(
    progress: TransferProgress,
) -> None:
    total = progress.total
    percent = 100.0 * progress.completed / total if total else 100.0
    typer.echo(
        f"[colab] {progress.direction} "
        f"{percent:5.1f}% "
        f"({progress.completed}/{total} bytes)",
        err=True,
    )


def _open_transfer(
    session,
    state,
    *,
    chunk_size_mib: float,
    progress=None,
) -> _TransferResources:
    chunk_size = _chunk_size_mib_to_bytes(chunk_size_mib)
    contents = ContentsClient(session)
    try:
        executor = open_remote_executor(
            session,
            state.store,
            history=state.history,
        )
    except BaseException:
        try:
            contents.close()
        except Exception:
            pass
        raise

    transfer = FileTransfer(
        contents,
        RemoteFileOps(executor),
        chunk_size=chunk_size,
        progress=progress or _legacy_progress,
    )
    return _TransferResources(
        contents=contents,
        executor=executor,
        transfer=transfer,
    )


def _select_session(
    state,
    requested: str | None,
    *,
    json_output: bool,
):
    if json_output:
        try:
            selected = resolve_local_session_read_only(
                state,
                requested,
            )
        except SessionSelectionError as exc:
            raise _TransferCommandError(
                exc.code,
                str(exc),
                retryable=False,
            ) from exc
        return selected.name, selected

    name = state.resolve_session(requested)
    selected = state.store.get(name)
    if not selected:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)
    return name, selected


def _best_effort_history(
    state,
    name: str,
    payload: dict[str, Any],
    *,
    telemetry: TransferTelemetry,
    secrets: tuple[str, ...],
) -> None:
    try:
        state.history.log_event(
            name,
            "file_operation",
            payload,
        )
    except Exception as exc:
        telemetry.warnings.append(
            "transfer history write failed: "
            + redact_text(
                exc,
                secrets=secrets,
            )
        )


def _error_identity(
    exc: Exception,
) -> tuple[str, bool | None]:
    if isinstance(exc, TransferLeaseBusy):
        return "TRANSFER_TARGET_BUSY", True
    if isinstance(exc, TransferLeaseCorrupt):
        return "TRANSFER_LEASE_UNSAFE", False
    if isinstance(exc, TransferIntegrityError):
        return "TRANSFER_INTEGRITY_ERROR", True
    if isinstance(exc, FileExistsError):
        return "TRANSFER_TARGET_EXISTS", False
    if isinstance(exc, FileNotFoundError):
        return "TRANSFER_FILE_NOT_FOUND", False
    if isinstance(exc, IsADirectoryError):
        return "TRANSFER_PATH_IS_DIRECTORY", False
    if isinstance(exc, PermissionError):
        return "TRANSFER_PERMISSION_DENIED", False
    if isinstance(exc, RemoteExecutionError):
        remote_codes = {
            "FileExistsError": (
                "TRANSFER_TARGET_EXISTS",
                False,
            ),
            "FileNotFoundError": (
                "TRANSFER_FILE_NOT_FOUND",
                False,
            ),
            "IsADirectoryError": (
                "TRANSFER_PATH_IS_DIRECTORY",
                False,
            ),
            "PermissionError": (
                "TRANSFER_PERMISSION_DENIED",
                False,
            ),
        }
        if exc.remote_name in remote_codes:
            return remote_codes[exc.remote_name]
    if isinstance(exc, _TransferCommandError):
        return exc.code, exc.retryable
    return "TRANSFER_FAILED", None


def _progress_callback(
    telemetry: TransferTelemetry,
    lease: TransferLease,
):
    def report(
        progress: TransferProgress,
    ) -> None:
        telemetry.update(progress)
        lease.heartbeat(
            completed_bytes=progress.completed,
            total_bytes=progress.total,
            resumed_from=progress.resumed_from,
            retry_count=progress.retry_count,
            sha256=progress.sha256,
            partial_path=progress.partial_path,
            force=progress.phase
            in {
                "prepared",
                "retrying",
                "completed",
            },
        )
        typer.echo(
            progress_line(telemetry),
            err=True,
        )

    return report


def _cleanup(
    resources: _TransferResources | None,
    lease: TransferLease | None,
    telemetry: TransferTelemetry,
    *,
    secrets: tuple[str, ...],
) -> None:
    if resources is not None:
        telemetry.warnings.extend(resources.close())
    if lease is not None:
        try:
            lease.heartbeat(
                completed_bytes=(telemetry.completed_bytes),
                total_bytes=telemetry.total_bytes,
                resumed_from=telemetry.resumed_from,
                retry_count=telemetry.retry_count,
                sha256=telemetry.sha256,
                partial_path=telemetry.partial_path,
                force=True,
            )
        except Exception as exc:
            telemetry.warnings.append(f"lease final heartbeat failed: {exc}")
        try:
            lease.release()
        except Exception as exc:
            telemetry.warnings.append(f"lease release failed: {exc}")
        telemetry.warnings.extend(
            getattr(
                lease,
                "cleanup_errors",
                [],
            )
        )

    telemetry.warnings = [
        redact_text(
            item,
            secrets=secrets,
        )
        for item in telemetry.warnings
    ]
    for warning in telemetry.warnings:
        typer.echo(
            f"[colab] Warning: {warning}",
            err=True,
        )


def _run_transfer(
    *,
    state,
    direction: str,
    requested_session: str | None,
    source_path: str,
    target_path: str,
    chunk_size_mib: float,
    overwrite: bool | None,
    json_output: bool,
    lease_factory: Callable[[Any], TransferLease],
    operation: Callable[
        [FileTransfer],
        TransferResult,
    ],
    success_prefix: str,
    interrupt_message: str,
) -> int:
    telemetry = TransferTelemetry(
        direction=direction,
        session=requested_session,
        endpoint=None,
        source_path=source_path,
        target_path=target_path,
        resume_argv=[],
    )
    resources = None
    lease = None
    result = None
    error: Exception | None = None
    status = "completed"
    name = None
    secrets: tuple[str, ...] = ()

    machine_context = machine_diagnostics_to_stderr() if json_output else nullcontext()
    try:
        with machine_context:
            name, selected = _select_session(
                state,
                requested_session,
                json_output=json_output,
            )
            endpoint = str(selected.endpoint)
            telemetry.session = name
            telemetry.endpoint = endpoint
            secrets = (str(selected.token),)
            telemetry.resume_argv = build_resume_argv(
                state=state,
                direction=direction,
                session_name=name,
                source_path=source_path,
                target_path=target_path,
                chunk_size_mib=chunk_size_mib,
                overwrite=overwrite,
                json_output=json_output,
            )
            lease = lease_factory(selected)
            lease.acquire()
            resources = _open_transfer(
                selected,
                state,
                chunk_size_mib=chunk_size_mib,
                progress=_progress_callback(
                    telemetry,
                    lease,
                ),
            )
            result = operation(resources.transfer)
            telemetry.finish(result)
            _best_effort_history(
                state,
                name,
                history_payload(
                    telemetry,
                    state="completed",
                    secrets=secrets,
                ),
                telemetry=telemetry,
                secrets=secrets,
            )
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        status = "interrupted"
        error = _TransferCommandError(
            "TRANSFER_INTERRUPTED",
            "Transfer interrupted by user",
            retryable=True,
        )
        if name is not None:
            _best_effort_history(
                state,
                name,
                history_payload(
                    telemetry,
                    state="interrupted",
                    secrets=secrets,
                ),
                telemetry=telemetry,
                secrets=secrets,
            )
    except TransferLeaseBusy as exc:
        status = "busy"
        error = exc
        if name is not None:
            _best_effort_history(
                state,
                name,
                history_payload(
                    telemetry,
                    state="busy",
                    error=str(exc),
                    secrets=secrets,
                ),
                telemetry=telemetry,
                secrets=secrets,
            )
    except Exception as exc:
        status = "failed"
        error = exc
        if name is not None:
            _best_effort_history(
                state,
                name,
                history_payload(
                    telemetry,
                    state="failed",
                    error=str(exc),
                    secrets=secrets,
                ),
                telemetry=telemetry,
                secrets=secrets,
            )
    finally:
        cleanup_context = (
            machine_diagnostics_to_stderr() if json_output else nullcontext()
        )
        with cleanup_context:
            _cleanup(
                resources,
                lease,
                telemetry,
                secrets=secrets,
            )

    error_code = None
    retryable = None
    error_message = None
    if error is not None:
        error_code, retryable = _error_identity(error)
        error_message = str(error)

    envelope = telemetry.envelope(
        status=status,
        lease=lease,
        error_code=error_code,
        error_message=error_message,
        retryable=retryable,
        secrets=secrets,
    )
    if json_output:
        emit_transfer_json(envelope)
    elif status == "completed":
        assert result is not None
        rate = telemetry.rate_mib_per_second()
        rate_text = f", {rate:.2f} MiB/s" if rate is not None else ""
        typer.echo(
            f"{success_prefix} "
            f"({result.size} bytes, "
            f"sha256={result.sha256}, "
            f"resumed_from={result.resumed_from}, "
            f"retries={result.retry_count}"
            f"{rate_text})"
        )
    elif status == "interrupted":
        typer.echo(
            interrupt_message,
            err=True,
        )
        typer.echo(
            f"[colab] Resume: {telemetry.resume_command()}",
            err=True,
        )
    else:
        label = "Upload" if direction == "upload" else "Download"
        safe_error = redact_text(
            error_message,
            secrets=secrets,
        )
        typer.echo(
            f"[colab] {label} failed: {safe_error}",
            err=True,
        )
        if telemetry.resume_argv:
            typer.echo(
                f"[colab] Resume: {telemetry.resume_command()}",
                err=True,
            )

    if status == "completed":
        return 0
    if status == "interrupted":
        return 130
    return 1


def _path_failure(
    *,
    direction: str,
    session: str | None,
    source_path: str,
    target_path: str,
    error: Exception,
    json_output: bool,
) -> None:
    telemetry = TransferTelemetry(
        direction=direction,
        session=session,
        endpoint=None,
        source_path=source_path,
        target_path=target_path,
        resume_argv=[],
    )
    if json_output:
        emit_transfer_json(
            telemetry.envelope(
                status="failed",
                lease=None,
                error_code="TRANSFER_INVALID_PATH",
                error_message=str(error),
                retryable=False,
            )
        )
    else:
        typer.echo(
            f"[colab] Invalid transfer path: {error}",
            err=True,
        )
    raise typer.Exit(2)


def ls(
    session: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--session",
            help="Session name",
        ),
    ] = None,
    path: Annotated[
        str,
        typer.Argument(help="Remote path to list"),
    ] = "content",
):
    """List files in a session."""
    from colab_cli.common import state

    name = state.resolve_session(session)
    selected = state.store.get(name)
    if not selected:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)

    contents = ContentsClient(selected)
    try:
        data = contents.list_dir(path)
        state.history.log_event(
            name,
            "file_operation",
            {
                "op": "ls",
                "path": path,
            },
        )
        if data.get("type") == "directory":
            items = data.get("content", [])
            for item in sorted(
                items,
                key=lambda value: (
                    value.get("type") != "directory",
                    value.get("name"),
                ),
            ):
                suffix = "/" if item.get("type") == "directory" else ""
                typer.echo(f"{item.get('name')}{suffix}")
        else:
            typer.echo(data.get("name"))
    except Exception as exc:
        typer.echo(f"[colab] Error: {exc}")
        raise typer.Exit(1)
    finally:
        contents.close()


def rm(
    session: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--session",
            help="Session name",
        ),
    ] = None,
    path: Annotated[
        str,
        typer.Argument(help="Remote path to remove"),
    ] = ...,
):
    """Remove a remote file."""
    from colab_cli.common import state

    name = state.resolve_session(session)
    selected = state.store.get(name)
    if not selected:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)

    contents = ContentsClient(selected)
    try:
        contents.rm(path)
        state.history.log_event(
            name,
            "file_operation",
            {
                "op": "rm",
                "path": path,
            },
        )
        typer.echo(f"[colab] Deleted {path}")
    except Exception as exc:
        typer.echo(f"[colab] Error: {exc}")
        raise typer.Exit(1)
    finally:
        contents.close()


def upload(
    session: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--session",
            help="Session name",
        ),
    ] = None,
    local_path: Annotated[
        str,
        typer.Argument(help="Local file to upload"),
    ] = ...,
    remote_path: Annotated[
        str,
        typer.Argument(help="Remote path to upload to"),
    ] = ...,
    chunk_size_mib: Annotated[
        float,
        typer.Option(
            "--chunk-size-mib",
            help=("Bounded transfer chunk size in MiB"),
            callback=_validate_chunk_size_mib,
        ),
    ] = DEFAULT_CHUNK_SIZE / _MIB,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help=("Resume a verified partial upload"),
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite",
            help=("Replace an existing remote file"),
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=("Emit the stable colab.transfer.v1 result"),
        ),
    ] = False,
):
    """Upload one verified file to a session."""
    from colab_cli.common import state

    canonical_source = canonical_local_path(local_path)
    try:
        target = normalize_remote_path(remote_path)
    except ValueError as exc:
        _path_failure(
            direction="upload",
            session=session,
            source_path=canonical_source,
            target_path=str(remote_path),
            error=exc,
            json_output=json_output,
        )

    if not os.path.isfile(canonical_source):
        telemetry = TransferTelemetry(
            direction="upload",
            session=session,
            endpoint=None,
            source_path=canonical_source,
            target_path=target,
            resume_argv=[],
        )
        envelope = telemetry.envelope(
            status="failed",
            lease=None,
            error_code="TRANSFER_FILE_NOT_FOUND",
            error_message=(f"Local file '{local_path}' not found"),
            retryable=False,
        )
        if json_output:
            emit_transfer_json(envelope)
        else:
            typer.echo(
                f"[colab] Local file '{local_path}' not found.",
                err=True,
            )
        raise typer.Exit(1)

    def lease_factory(selected):
        return TransferLease.for_upload(
            endpoint=str(selected.endpoint),
            local_path=canonical_source,
            remote_path=target,
        )

    exit_code = _run_transfer(
        state=state,
        direction="upload",
        requested_session=session,
        source_path=canonical_source,
        target_path=target,
        chunk_size_mib=chunk_size_mib,
        overwrite=overwrite,
        json_output=json_output,
        lease_factory=lease_factory,
        operation=lambda transfer: transfer.upload(
            local_path,
            target,
            overwrite=overwrite,
            resume=resume,
        ),
        success_prefix=(f"[colab] Uploaded '{local_path}' to '{target}'"),
        interrupt_message=(
            "[colab] Upload interrupted; verified remote partial preserved"
        ),
    )
    if exit_code:
        raise typer.Exit(exit_code)


def download(
    session: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--session",
            help="Session name",
        ),
    ] = None,
    remote_path: Annotated[
        str,
        typer.Argument(help="Remote path to download from"),
    ] = ...,
    local_path: Annotated[
        str,
        typer.Argument(help="Local path to save the file"),
    ] = ...,
    chunk_size_mib: Annotated[
        float,
        typer.Option(
            "--chunk-size-mib",
            help=("Bounded transfer chunk size in MiB"),
            callback=_validate_chunk_size_mib,
        ),
    ] = DEFAULT_CHUNK_SIZE / _MIB,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help=("Resume a verified partial download"),
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=("Emit the stable colab.transfer.v1 result"),
        ),
    ] = False,
):
    """Download one verified file from a session."""
    from colab_cli.common import state

    try:
        source = normalize_remote_path(remote_path)
    except ValueError as exc:
        _path_failure(
            direction="download",
            session=session,
            source_path=str(remote_path),
            target_path=canonical_local_path(local_path),
            error=exc,
            json_output=json_output,
        )
    canonical_target = canonical_local_path(local_path)

    def lease_factory(selected):
        return TransferLease.for_download(
            endpoint=str(selected.endpoint),
            remote_path=source,
            local_path=canonical_target,
        )

    exit_code = _run_transfer(
        state=state,
        direction="download",
        requested_session=session,
        source_path=source,
        target_path=canonical_target,
        chunk_size_mib=chunk_size_mib,
        overwrite=None,
        json_output=json_output,
        lease_factory=lease_factory,
        operation=lambda transfer: transfer.download(
            source,
            local_path,
            resume=resume,
        ),
        success_prefix=(f"[colab] Downloaded '{source}' to '{local_path}'"),
        interrupt_message=(
            "[colab] Download interrupted; verified local partial preserved"
        ),
    )
    if exit_code:
        raise typer.Exit(exit_code)


def edit(
    session: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--session",
            help="Session name",
        ),
    ] = None,
    remote_path: Annotated[
        str,
        typer.Argument(help="Remote path to edit"),
    ] = ...,
):
    """Edit a file on a running Colab session."""
    from colab_cli.common import state

    name, selected = _select_session(
        state,
        session,
        json_output=False,
    )
    target = normalize_remote_path(remote_path)
    _, extension = os.path.splitext(target)
    temp_file = tempfile.NamedTemporaryFile(
        suffix=extension,
        delete=False,
    )
    local_path = temp_file.name
    temp_file.close()

    lease = TransferLease.for_upload(
        endpoint=str(selected.endpoint),
        local_path=local_path,
        remote_path=target,
    )
    resources = None

    def file_hash(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as stream:
            return hashlib.file_digest(
                stream,
                "sha256",
            ).hexdigest()

    try:
        lease.acquire()
        resources = _open_transfer(
            selected,
            state,
            chunk_size_mib=1,
        )
        try:
            resources.transfer.download(
                target,
                local_path,
                resume=False,
            )
        except FileNotFoundError:
            open(local_path, "wb").close()

        before = file_hash(local_path)
        click.edit(filename=local_path)
        after = file_hash(local_path)
        if after != before:
            result = resources.transfer.upload(
                local_path,
                target,
                overwrite=True,
                resume=True,
            )
            state.history.log_event(
                name,
                "file_operation",
                {
                    "op": "edit",
                    "remote": target,
                    "size": result.size,
                    "sha256": result.sha256,
                },
            )
            typer.echo(f"[colab] Edited and uploaded '{target}'")
        else:
            typer.echo(f"[colab] No changes made to '{target}'")
    except (
        TransferLeaseBusy,
        TransferLeaseCorrupt,
    ) as exc:
        typer.echo(
            f"[colab] Edit refused: {exc}",
            err=True,
        )
        raise typer.Exit(1)
    finally:
        if resources is not None:
            for warning in resources.close():
                typer.echo(
                    f"[colab] Warning: {warning}",
                    err=True,
                )
        lease.release()
        for warning in lease.cleanup_errors:
            typer.echo(
                f"[colab] Warning: {warning}",
                err=True,
            )
        try:
            os.unlink(local_path)
        except OSError:
            pass


def register(app: typer.Typer):
    app.command()(ls)
    app.command()(rm)
    app.command()(upload)
    app.command()(download)
    app.command()(edit)
