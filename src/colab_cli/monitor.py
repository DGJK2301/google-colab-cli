# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Foreground, resumable local evidence capture for detached jobs."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from colab_cli.jobs import DEFAULT_JOB_ROOT, RemoteJobClient
from colab_cli.monitor_models import MonitorState, MonitorSummary
from colab_cli.observability.probes import probe_session
from colab_cli.observability.redaction import redact_text
from colab_cli.remote import open_remote_executor
from colab_cli.state import SessionState


_TERMINAL_STATES = {
    "succeeded",
    "failed",
    "cancelled",
    "lost",
}
_MAX_TAIL_CHUNKS_PER_POLL = 64


class MonitorConfigurationError(ValueError):
    pass


class MonitorBindingError(RuntimeError):
    pass


class MonitorEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class MonitorConfig:
    output_dir: Path
    interval: float = 5.0
    probe_every: float = 60.0
    probe_timeout: float = 20.0
    control_timeout: float = 30.0
    max_bytes: int = 65536
    job_root: str = DEFAULT_JOB_ROOT
    timeout: float | None = None
    once: bool = False
    max_control_errors: int = 5

    def validated(self) -> "MonitorConfig":
        _finite_positive(self.interval, "interval")
        _finite_positive(
            self.probe_timeout,
            "probe-timeout",
        )
        _finite_positive(
            self.control_timeout,
            "control-timeout",
        )
        if not math.isfinite(self.probe_every) or self.probe_every < 0:
            raise MonitorConfigurationError(
                "probe-every must be finite and non-negative"
            )
        if self.max_bytes <= 0:
            raise MonitorConfigurationError("max-bytes must be positive")
        if self.timeout is not None:
            _finite_positive(self.timeout, "timeout")
        if self.max_control_errors <= 0:
            raise MonitorConfigurationError("max-control-errors must be positive")
        return self


@dataclass
class MonitorConnection:
    executor: Any
    client: RemoteJobClient

    def close(self) -> list[str]:
        try:
            self.executor.close()
            return []
        except Exception as exc:
            return [f"monitor executor close failed: {exc}"]


def run_monitor(
    state: Any,
    *,
    job_id: str,
    session: SessionState,
    config: MonitorConfig,
    connection_factory: Callable[
        [Any, SessionState, str],
        MonitorConnection,
    ]
    | None = None,
    probe_fn: Callable[..., Any] = probe_session,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> MonitorSummary:
    config = config.validated()
    output = config.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _restrict_directory(output)

    monitor_lock = FileLock(
        output / ".monitor.lock",
        timeout=0,
        blocking=False,
        is_singleton=False,
    )
    try:
        monitor_lock.acquire(blocking=False)
    except FileLockTimeout as exc:
        raise MonitorConfigurationError(f"another monitor owns {output}") from exc

    try:
        return _run_monitor_locked(
            state,
            job_id=job_id,
            session=session,
            config=config,
            output=output,
            connection_factory=connection_factory,
            probe_fn=probe_fn,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )
    finally:
        monitor_lock.release(force=True)


def _run_monitor_locked(
    state: Any,
    *,
    job_id: str,
    session: SessionState,
    config: MonitorConfig,
    output: Path,
    connection_factory: Callable[
        [Any, SessionState, str],
        MonitorConnection,
    ]
    | None,
    probe_fn: Callable[..., Any],
    sleep_fn: Callable[[float], None],
    monotonic_fn: Callable[[], float],
) -> MonitorSummary:
    paths = _paths(output)
    monitor_state, resumed, warnings = _load_or_create_state(
        paths["state"],
        job_id=job_id,
        session=session,
        job_root=config.job_root,
        output=output,
    )
    _reconcile_local_offsets(
        monitor_state,
        paths,
        warnings,
    )
    _write_state(
        paths["state"],
        monitor_state,
    )
    _append_event(
        paths["events"],
        ("monitor_resumed" if resumed else "monitor_started"),
        {
            "job_id": job_id,
            "session_name": session.name,
            "endpoint": session.endpoint,
            "stdout_offset": (monitor_state.stdout_offset),
            "stderr_offset": (monitor_state.stderr_offset),
            "monitor_runs": (monitor_state.monitor_runs),
        },
    )

    started_at = _utcnow()
    started = monotonic_fn()
    next_probe = started
    connection: MonitorConnection | None = None
    last_status: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    summary_status = "failed"
    exit_code = 1

    try:
        while True:
            if (
                config.timeout is not None
                and monotonic_fn() - started >= config.timeout
            ):
                summary_status = "timeout"
                exit_code = 124
                error_code = "MONITOR_TIMEOUT"
                error_message = (
                    "Local monitor deadline expired; remote job was not cancelled."
                )
                monitor_state.status = "timeout"
                break

            try:
                if connection is None:
                    factory = connection_factory or _default_connection
                    connection = factory(
                        state,
                        session,
                        config.job_root,
                    )
                    _append_event(
                        paths["events"],
                        "control_connected",
                        {},
                    )

                status = connection.client.status(
                    job_id,
                    timeout=config.control_timeout,
                )
                last_status = dict(status)
                monitor_state.consecutive_control_errors = 0
                _bind_runtime(
                    monitor_state,
                    status,
                )
                _append_jsonl(
                    paths["job"],
                    {
                        "observed_at": _utcnow(),
                        **_redact_record(
                            status,
                            session.token,
                        ),
                    },
                )
                monitor_state.last_job_state = _string(status.get("state"))
                monitor_state.last_returncode = _integer(status.get("returncode"))
                monitor_state.last_heartbeat_at = _string(status.get("heartbeat_at"))
                _validate_remote_log_sizes(
                    status,
                    monitor_state,
                )

                (
                    monitor_state.stdout_offset,
                    stdout_caught_up,
                ) = _drain(
                    connection.client,
                    job_id,
                    "stdout",
                    monitor_state.stdout_offset,
                    config.max_bytes,
                    paths["stdout"],
                    paths["events"],
                    config.control_timeout,
                    _integer(status.get("stdout_size")),
                )
                (
                    monitor_state.stderr_offset,
                    stderr_caught_up,
                ) = _drain(
                    connection.client,
                    job_id,
                    "stderr",
                    monitor_state.stderr_offset,
                    config.max_bytes,
                    paths["stderr"],
                    paths["events"],
                    config.control_timeout,
                    _integer(status.get("stderr_size")),
                )

                now = monotonic_fn()
                if config.probe_every > 0 and now >= next_probe:
                    next_probe = now + config.probe_every
                    _sample_resources(
                        session,
                        monitor_state,
                        paths,
                        probe_fn=probe_fn,
                        timeout=config.probe_timeout,
                    )

                monitor_state.updated_at = _utcnow()
                _write_state(
                    paths["state"],
                    monitor_state,
                )

                remote_state = status.get("state")
                if remote_state in _TERMINAL_STATES:
                    if not (stdout_caught_up and stderr_caught_up):
                        _append_event(
                            paths["events"],
                            "remote_terminal_logs_pending",
                            {
                                "state": remote_state,
                                "stdout_offset": (monitor_state.stdout_offset),
                                "stderr_offset": (monitor_state.stderr_offset),
                            },
                        )
                        sleep_fn(
                            min(
                                config.interval,
                                0.25,
                            )
                        )
                        continue

                    monitor_state.status = "terminal"
                    summary_status = "completed"
                    exit_code = _remote_exit_code(status)
                    if remote_state == "failed":
                        error_code = "REMOTE_JOB_FAILED"
                        error_message = _string(status.get("error")) or (
                            "Remote job failed with "
                            "return code "
                            f"{status.get('returncode')}."
                        )
                    elif remote_state == "lost":
                        error_code = "REMOTE_JOB_LOST"
                        error_message = _string(status.get("error")) or (
                            "Remote runtime or runner identity was lost."
                        )
                    elif remote_state == "cancelled":
                        error_code = "REMOTE_JOB_CANCELLED"
                        error_message = "Remote job was cancelled."
                    _append_event(
                        paths["events"],
                        "remote_terminal",
                        {
                            "state": remote_state,
                            "returncode": (status.get("returncode")),
                            "exit_code": exit_code,
                        },
                    )
                    break

                if config.once:
                    monitor_state.status = "snapshot"
                    summary_status = "snapshot"
                    exit_code = 0
                    break

            except MonitorBindingError as exc:
                error_code = "RUNTIME_IDENTITY_CHANGED"
                error_message = str(exc)
                monitor_state.status = "failed"
                _append_event(
                    paths["events"],
                    "runtime_identity_changed",
                    {
                        "error": redact_text(
                            exc,
                            secrets=(session.token,),
                        )
                    },
                )
                break

            except MonitorEvidenceError as exc:
                error_code = "LOCAL_EVIDENCE_DIVERGED"
                error_message = str(exc)
                monitor_state.status = "failed"
                _append_event(
                    paths["events"],
                    "local_evidence_diverged",
                    {
                        "error": redact_text(
                            exc,
                            secrets=(session.token,),
                        )
                    },
                )
                break

            except Exception as exc:
                monitor_state.consecutive_control_errors += 1
                safe_error = redact_text(
                    f"{type(exc).__name__}: {exc}",
                    secrets=(session.token,),
                )
                _append_event(
                    paths["events"],
                    "control_error",
                    {
                        "error": safe_error,
                        "consecutive": (monitor_state.consecutive_control_errors),
                    },
                )
                if connection is not None:
                    warnings.extend(connection.close())
                    connection = None

                assignment_state = _assignment_state(
                    state,
                    session,
                    config.probe_timeout,
                )
                if assignment_state == "missing":
                    error_code = "ASSIGNMENT_DISAPPEARED"
                    error_message = (
                        "The monitored endpoint is no "
                        "longer in the account "
                        "assignment list."
                    )
                    monitor_state.status = "failed"
                    break
                if (
                    monitor_state.consecutive_control_errors
                    >= config.max_control_errors
                ):
                    error_code = "MONITOR_CONTROL_UNAVAILABLE"
                    error_message = safe_error
                    monitor_state.status = "failed"
                    break

            monitor_state.updated_at = _utcnow()
            _write_state(
                paths["state"],
                monitor_state,
            )
            sleep_fn(config.interval)

    except KeyboardInterrupt:
        summary_status = "interrupted"
        exit_code = 130
        error_code = "MONITOR_INTERRUPTED"
        error_message = "Local monitor stopped; remote job was not cancelled."
        monitor_state.status = "interrupted"
        _append_event(
            paths["events"],
            "monitor_interrupted",
            {"remote_job_cancelled": False},
        )

    finally:
        if connection is not None:
            warnings.extend(connection.close())
        monitor_state.updated_at = _utcnow()
        _write_state(
            paths["state"],
            monitor_state,
        )

    summary = MonitorSummary(
        ok=(summary_status in {"completed", "snapshot"} and exit_code == 0),
        status=summary_status,
        job_id=job_id,
        session_name=session.name,
        endpoint=session.endpoint,
        job_root=config.job_root,
        output_dir=str(output),
        remote_state=_string((last_status or {}).get("state")),
        remote_returncode=_integer((last_status or {}).get("returncode")),
        exit_code=exit_code,
        started_at=started_at,
        finished_at=_utcnow(),
        elapsed_seconds=round(
            monotonic_fn() - started,
            6,
        ),
        stdout_offset=monitor_state.stdout_offset,
        stderr_offset=monitor_state.stderr_offset,
        remote_runtime_id=(monitor_state.remote_runtime_id),
        probe_boot_id=monitor_state.probe_boot_id,
        error_code=error_code,
        error=(
            redact_text(
                error_message,
                secrets=(session.token,),
            )
            if error_message is not None
            else None
        ),
        warnings=[
            redact_text(
                warning,
                secrets=(session.token,),
            )
            for warning in warnings
        ],
        files={name: str(path) for name, path in paths.items()},
    )
    _atomic_json(
        paths["summary"],
        summary.model_dump(
            mode="json",
            by_alias=True,
        ),
    )
    _append_event(
        paths["events"],
        "monitor_finished",
        {
            "status": summary.status,
            "exit_code": summary.exit_code,
            "remote_state": summary.remote_state,
        },
    )
    return summary


def _default_connection(
    state: Any,
    session: SessionState,
    job_root: str,
) -> MonitorConnection:
    executor = open_remote_executor(
        session,
        state.store,
        history=state.history,
    )
    return MonitorConnection(
        executor=executor,
        client=RemoteJobClient(
            executor,
            job_root=job_root,
        ),
    )


def _load_or_create_state(
    path: Path,
    *,
    job_id: str,
    session: SessionState,
    job_root: str,
    output: Path,
) -> tuple[MonitorState, bool, list[str]]:
    warnings: list[str] = []
    if path.exists():
        try:
            state = MonitorState.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MonitorConfigurationError(
                f"monitor_state.json is invalid: {exc}"
            ) from exc

        expected = {
            "job_id": job_id,
            "session_name": session.name,
            "endpoint": session.endpoint,
            "job_root": job_root,
        }
        for key, value in expected.items():
            if getattr(state, key) != value:
                raise MonitorBindingError(
                    "Existing monitor state binds "
                    f"{key}={getattr(state, key)!r}, "
                    f"not {value!r}."
                )
        state.monitor_runs += 1
        state.monitor_pid = os.getpid()
        state.status = "running"
        state.updated_at = _utcnow()
        return state, True, warnings

    for artifact in (
        "stdout.log",
        "stderr.log",
        "job.jsonl",
        "resources.jsonl",
    ):
        candidate = output / artifact
        if candidate.exists() and candidate.stat().st_size:
            raise MonitorConfigurationError(
                "Output directory contains "
                f"{artifact} but no "
                "monitor_state.json; refusing "
                "to mix unrelated evidence."
            )

    now = _utcnow()
    return (
        MonitorState(
            job_id=job_id,
            session_name=session.name,
            endpoint=session.endpoint,
            job_root=job_root,
            output_dir=str(output),
            created_at=now,
            updated_at=now,
            monitor_pid=os.getpid(),
        ),
        False,
        warnings,
    )


def _reconcile_local_offsets(
    state: MonitorState,
    paths: dict[str, Path],
    warnings: list[str],
) -> None:
    for stream_name in ("stdout", "stderr"):
        path = paths[stream_name]
        actual = path.stat().st_size if path.exists() else 0
        field = f"{stream_name}_offset"
        recorded = getattr(state, field)
        if actual != recorded:
            warnings.append(
                f"Reconciled {field} from {recorded} to local log size {actual}."
            )
            setattr(state, field, actual)


def _bind_runtime(
    state: MonitorState,
    status: dict[str, Any],
) -> None:
    runtime_id = _string(status.get("runtime_id"))
    if runtime_id is None:
        return
    if state.remote_runtime_id is None:
        state.remote_runtime_id = runtime_id
    elif state.remote_runtime_id != runtime_id:
        raise MonitorBindingError(
            "Remote job runtime changed from "
            f"{state.remote_runtime_id} "
            f"to {runtime_id}."
        )


def _validate_remote_log_sizes(
    status: dict[str, Any],
    state: MonitorState,
) -> None:
    for stream_name in ("stdout", "stderr"):
        remote_size = _integer(status.get(f"{stream_name}_size"))
        local_offset = getattr(
            state,
            f"{stream_name}_offset",
        )
        if remote_size is not None and local_offset > remote_size:
            raise MonitorEvidenceError(
                f"Local {stream_name} evidence "
                f"has {local_offset} bytes but "
                f"remote log reports "
                f"{remote_size}; refusing to "
                "rewind or duplicate data."
            )


def _drain(
    client: RemoteJobClient,
    job_id: str,
    stream_name: str,
    offset: int,
    max_bytes: int,
    log_path: Path,
    events_path: Path,
    control_timeout: float,
    remote_size_hint: int | None,
) -> tuple[int, bool]:
    if remote_size_hint is not None and offset == remote_size_hint:
        return offset, True

    start_offset = offset
    chunks = 0
    total_written = 0
    caught_up = False
    file_handle = None

    try:
        while chunks < _MAX_TAIL_CHUNKS_PER_POLL:
            tail = client.tail(
                job_id,
                stream=stream_name,
                offset=offset,
                max_bytes=max_bytes,
                timeout=control_timeout,
            )
            if tail.offset != offset:
                raise RuntimeError(
                    f"Remote tail returned offset {tail.offset}, expected {offset}."
                )

            if tail.data:
                if file_handle is None:
                    log_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    file_handle = log_path.open("ab")
                file_handle.write(tail.data)
                offset = tail.next_offset
                chunks += 1
                total_written += len(tail.data)

            caught_up = bool(
                tail.eof
                or (remote_size_hint is not None and offset >= remote_size_hint)
            )
            if caught_up or not tail.data:
                break

        if file_handle is not None:
            file_handle.flush()
            os.fsync(file_handle.fileno())
    finally:
        if file_handle is not None:
            file_handle.close()

    if total_written:
        _append_event(
            events_path,
            "log_drain",
            {
                "stream": stream_name,
                "offset": start_offset,
                "next_offset": offset,
                "bytes": total_written,
                "chunks": chunks,
                "remote_size": remote_size_hint,
                "caught_up": caught_up,
            },
        )
    return offset, caught_up


def _sample_resources(
    session: SessionState,
    state: MonitorState,
    paths: dict[str, Path],
    *,
    probe_fn: Callable[..., Any],
    timeout: float,
) -> None:
    try:
        probe = probe_fn(
            session,
            timeout=timeout,
        )
        payload = (
            probe.model_dump(mode="json")
            if hasattr(probe, "model_dump")
            else dict(probe)
        )
        payload["sampled_at"] = _utcnow()
        _append_jsonl(
            paths["resources"],
            payload,
        )

        boot_id = _string((payload.get("runtime") or {}).get("boot_id"))
        if boot_id is not None:
            if state.probe_boot_id is None:
                state.probe_boot_id = boot_id
            elif state.probe_boot_id != boot_id:
                raise MonitorBindingError(
                    f"Runtime boot ID changed from {state.probe_boot_id} to {boot_id}."
                )

        if payload.get("status") in {
            "partial",
            "timeout",
            "unavailable",
        }:
            _append_event(
                paths["events"],
                "resource_sample_partial",
                {
                    "status": payload.get("status"),
                    "issues": payload.get(
                        "issues",
                        [],
                    ),
                },
            )

    except MonitorBindingError:
        raise

    except Exception as exc:
        safe_error = redact_text(
            f"{type(exc).__name__}: {exc}",
            secrets=(session.token,),
        )
        _append_jsonl(
            paths["resources"],
            {
                "sampled_at": _utcnow(),
                "status": "unavailable",
                "error": safe_error,
            },
        )
        _append_event(
            paths["events"],
            "resource_sample_failed",
            {"error": safe_error},
        )


def _assignment_state(
    state: Any,
    session: SessionState,
    timeout: float,
) -> str:
    try:
        assignments = state.client.list_assignments(
            timeout=(min(3.0, timeout), timeout)
        )
    except Exception:
        return "unknown"
    return (
        "present"
        if any(item.endpoint == session.endpoint for item in assignments)
        else "missing"
    )


def _remote_exit_code(
    status: dict[str, Any],
) -> int:
    remote_state = status.get("state")
    if remote_state == "succeeded":
        return 0
    if remote_state == "cancelled":
        return 130
    if remote_state == "failed":
        returncode = _integer(status.get("returncode"))
        return returncode if returncode not in (None, 0) else 1
    return 1


def _paths(output: Path) -> dict[str, Path]:
    return {
        "stdout": output / "stdout.log",
        "stderr": output / "stderr.log",
        "job": output / "job.jsonl",
        "resources": output / "resources.jsonl",
        "events": output / "events.jsonl",
        "state": output / "monitor_state.json",
        "summary": output / "summary.json",
    }


def _append_jsonl(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    line = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _append_event(
    path: Path,
    event_type: str,
    data: dict[str, Any],
) -> None:
    _append_jsonl(
        path,
        {
            "timestamp": _utcnow(),
            "event_type": event_type,
            **data,
        },
    )


def _write_state(
    path: Path,
    state: MonitorState,
) -> None:
    _atomic_json(
        path,
        state.model_dump(
            mode="json",
            by_alias=True,
        ),
    )


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with temp.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _restrict_directory(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _redact_record(
    record: dict[str, Any],
    secret: str,
) -> dict[str, Any]:
    return {
        key: (
            redact_text(
                value,
                secrets=(secret,),
            )
            if isinstance(value, str)
            else value
        )
        for key, value in record.items()
    }


def _finite_positive(
    value: float,
    name: str,
) -> None:
    if not math.isfinite(value) or value <= 0:
        raise MonitorConfigurationError(
            f"{name} must be a finite number greater than 0"
        )


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _string(value: Any) -> str | None:
    return None if value is None else str(value)


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None
