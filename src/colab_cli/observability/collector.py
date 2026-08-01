# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Side-effect-free collection and JSON rendering for session observations."""

from __future__ import annotations

import contextlib
import ctypes
from datetime import datetime, timezone
import math
import os
import sys
from typing import Any, Literal

from pydantic import BaseModel

from colab_cli.auth import ReauthenticationRequiredError
from colab_cli.client import ListedAssignment
from colab_cli.observability.models import (
    AssignmentObservation,
    KeepAliveObservation,
    LastExecutionObservation,
    LocalSessionObservation,
    ObservationIssue,
    SessionObservation,
    SessionsEnvelope,
    StatusEnvelope,
)
from colab_cli.observability.redaction import redact_text
from colab_cli.state import SessionState


class SessionSelectionError(LookupError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@contextlib.contextmanager
def machine_diagnostics_to_stderr():
    with contextlib.redirect_stdout(sys.stderr):
        yield


def emit_json(model: BaseModel) -> None:
    sys.stdout.write(model.model_dump_json(indent=2, by_alias=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_probe_timeout(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("probe timeout must be a finite number greater than 0")
    return value


def collect_sessions(state: Any) -> SessionsEnvelope:
    local, local_issues = _read_local_sessions(state)
    assignments, assignment_issues = _read_assignments(state)
    sessions = _join_sessions(
        local,
        assignments,
        assignments_available=not assignment_issues,
        include_orphans=True,
    )
    issues = [*local_issues, *assignment_issues]
    fatal = any(i.severity == "error" for i in issues)
    status: Literal["ok", "partial", "error"]
    status = "error" if fatal and not sessions else ("partial" if issues else "ok")
    return SessionsEnvelope(
        ok=status != "error",
        status=status,
        generated_at=utc_now(),
        sessions=sessions,
        warnings=[i for i in issues if i.severity == "warning"],
        errors=[i for i in issues if i.severity == "error"],
    )


def collect_status(
    state: Any,
    *,
    session_name: str | None,
    probe: bool,
    timeout: float,
) -> StatusEnvelope:
    from colab_cli.observability.probes import probe_session

    local, local_issues = _read_local_sessions(state)
    if session_name is not None and session_name not in local:
        missing = ObservationIssue(
            code="SESSION_NOT_FOUND",
            message=f"Session '{session_name}' was not found in local state.",
            source="local_state",
            severity="error",
            retryable=False,
        )
        return StatusEnvelope(
            ok=False,
            status="error",
            generated_at=utc_now(),
            selected_session=session_name,
            sessions=[],
            warnings=[i for i in local_issues if i.severity == "warning"],
            errors=[*[i for i in local_issues if i.severity == "error"], missing],
        )

    assignments, assignment_issues = _read_assignments(state)
    selected = {session_name: local[session_name]} if session_name else local
    sessions = _join_sessions(
        selected,
        assignments,
        assignments_available=not assignment_issues,
        include_orphans=False,
    )
    if probe and session_name is not None and sessions:
        observed = sessions[0]
        observed.probe = probe_session(local[session_name], timeout=timeout)
        _add_probe_warning(observed)

    issues = [*local_issues, *assignment_issues]
    fatal = any(i.severity == "error" for i in issues)
    probe_partial = any(
        s.probe.status in {"partial", "timeout", "unavailable"} for s in sessions
    )
    status = (
        "error"
        if fatal and not sessions
        else ("partial" if issues or probe_partial else "ok")
    )
    return StatusEnvelope(
        ok=status != "error",
        status=status,
        generated_at=utc_now(),
        selected_session=session_name,
        sessions=sessions,
        warnings=[i for i in issues if i.severity == "warning"],
        errors=[i for i in issues if i.severity == "error"],
    )


def resolve_local_session_read_only(state: Any, name: str | None) -> SessionState:
    try:
        sessions = state.store.list()
    except Exception as exc:
        raise SessionSelectionError(
            "LOCAL_STATE_UNAVAILABLE", f"{type(exc).__name__}: {exc}"
        ) from exc
    if name is not None:
        if name not in sessions:
            raise SessionSelectionError(
                "SESSION_NOT_FOUND", f"Session '{name}' not found"
            )
        return sessions[name]
    if not sessions:
        raise SessionSelectionError("NO_LOCAL_SESSIONS", "No local sessions exist")
    if len(sessions) > 1:
        names = ", ".join(sorted(sessions))
        raise SessionSelectionError(
            "SESSION_SELECTION_REQUIRED",
            f"Multiple local sessions are available: {names}",
        )
    return next(iter(sessions.values()))


def _read_local_sessions(state):
    try:
        return state.store.list(), []
    except Exception as exc:
        return {}, [
            ObservationIssue(
                code="LOCAL_STATE_UNAVAILABLE",
                message=redact_text(f"{type(exc).__name__}: {exc}"),
                source="local_state",
                severity="error",
                retryable=True,
            )
        ]


def _read_assignments(state):
    try:
        return state.client.list_assignments(), []
    except SystemExit as exc:
        return [], [
            ObservationIssue(
                code="ASSIGNMENTS_AUTH_FAILED",
                message=f"Assignment query exited with code {exc.code}.",
                source="control_plane",
                retryable=True,
            )
        ]
    except ReauthenticationRequiredError as exc:
        return [], [
            ObservationIssue(
                code="AUTH_REAUTH_REQUIRED",
                message=redact_text(str(exc)),
                source="control_plane",
                retryable=True,
            )
        ]
    except Exception as exc:
        return [], [
            ObservationIssue(
                code="ASSIGNMENTS_UNAVAILABLE",
                message=redact_text(f"{type(exc).__name__}: {exc}"),
                source="control_plane",
                retryable=True,
            )
        ]


def _join_sessions(local, assignments, *, assignments_available, include_orphans):
    by_endpoint = {a.endpoint: a for a in assignments}
    result = []
    local_endpoints = set()
    for name in sorted(local):
        session = local[name]
        local_endpoints.add(session.endpoint)
        result.append(
            _one_session(
                name, session, by_endpoint.get(session.endpoint), assignments_available
            )
        )
    if include_orphans:
        for assignment in sorted(assignments, key=lambda a: a.endpoint):
            if assignment.endpoint not in local_endpoints:
                result.append(_one_session(None, None, assignment, True))
    return result


def _one_session(
    name: str | None,
    session: SessionState | None,
    assignment: ListedAssignment | None,
    assignments_available: bool,
) -> SessionObservation:
    if session is None and assignment is None:
        raise ValueError("session or assignment required")
    endpoint = session.endpoint if session else assignment.endpoint
    local = _local_observation(session)
    if assignment:
        assigned = AssignmentObservation(
            status="ok",
            endpoint=assignment.endpoint,
            accelerator=_enum_value(assignment.accelerator),
            variant=_enum_name(assignment.variant),
            machine_shape=_enum_name(assignment.machine_shape),
        )
    else:
        assigned = AssignmentObservation(
            status="missing" if assignments_available else "unavailable",
            endpoint=endpoint,
        )
    if session is None:
        lifecycle = "orphan_server"
    elif assignment is None and assignments_available:
        lifecycle = "stale_local"
    elif assignment is None:
        lifecycle = "unknown"
    elif session.running:
        lifecycle = "busy"
    else:
        lifecycle = "idle"
    observed = SessionObservation(
        name=name,
        endpoint=endpoint,
        lifecycle=lifecycle,
        local=local,
        assignment=assigned,
    )
    requested = local.requested_accelerator
    if requested and assigned.accelerator and requested != assigned.accelerator:
        observed.warnings.append(
            ObservationIssue(
                code="REQUESTED_ASSIGNED_ACCELERATOR_MISMATCH",
                message=(
                    f"Requested accelerator {requested} differs from server "
                    f"assignment {assigned.accelerator}."
                ),
                source="control_plane",
                retryable=False,
            )
        )
    return observed


def _local_observation(session: SessionState | None) -> LocalSessionObservation:
    if session is None:
        return LocalSessionObservation(
            tracked=False,
            kernel_status="not_recorded",
            keep_alive=KeepAliveObservation(
                status="not_recorded",
                unavailable_reason="server_orphan_has_no_local_keep_alive_record",
            ),
        )
    last = None
    if session.last_execution:
        file, cell, timestamp = session.last_execution
        last = LastExecutionObservation(
            file=redact_text(file), cell=cell, timestamp=timestamp
        )
    return LocalSessionObservation(
        tracked=True,
        requested_variant=session.variant,
        requested_accelerator=session.accelerator,
        running=session.running,
        kernel_id=session.kernel_id,
        kernel_status="recorded" if session.kernel_id else "not_recorded",
        session_id=session.session_id,
        keep_alive=_keep_alive_observation(session.keep_alive_pid),
        last_execution=last,
    )


def _add_probe_warning(observed: SessionObservation) -> None:
    requested = observed.local.requested_accelerator
    if requested and requested != "NONE" and not observed.probe.gpu.available:
        observed.warnings.append(
            ObservationIssue(
                code="REQUESTED_GPU_NOT_OBSERVED",
                message=f"Session requested {requested}, but no GPU was observed.",
                source="probe",
                retryable=True,
            )
        )


def _keep_alive_observation(pid: int | None) -> KeepAliveObservation:
    if pid is None:
        return KeepAliveObservation(
            status="not_recorded",
            unavailable_reason=(
                "keep_alive_heartbeat_is_not_persisted_by_current_protocol"
            ),
        )
    alive = _pid_alive(pid)
    return KeepAliveObservation(
        pid=pid,
        status="alive" if alive else "dead",
        last_heartbeat_at=None,
        unavailable_reason=(
            "keep_alive_heartbeat_is_not_persisted_by_current_protocol"
        ),
    )


def _pid_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(
                kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                and code.value == 259
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _enum_value(value):
    return None if value is None else str(getattr(value, "value", value))


def _enum_name(value):
    if value is None:
        return None
    return getattr(value, "name", None) or _enum_value(value)
