# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Foreground local evidence capture for detached jobs."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from colab_cli.jobs import DEFAULT_JOB_ROOT
from colab_cli.monitor import (
    MonitorBindingError,
    MonitorConfig,
    MonitorConfigurationError,
    run_monitor,
)
from colab_cli.monitor_models import MonitorSummary
from colab_cli.observability import (
    emit_json,
    machine_diagnostics_to_stderr,
    redact_text,
)
from colab_cli.observability.collector import utc_now


def _select_monitor_session(
    state,
    name: str | None,
):
    sessions = state.store.list_strict()
    if name is not None:
        if name not in sessions:
            raise LookupError(f"Session {name!r} not found")
        return sessions[name]
    if not sessions:
        raise LookupError("No local sessions exist")
    if len(sessions) > 1:
        raise LookupError("Multiple sessions exist; specify -s/--session")
    return next(iter(sessions.values()))


def _emit_command_error(
    *,
    job_id: str,
    session_name: str,
    endpoint: str,
    job_root: str,
    output_dir: Path,
    exit_code: int,
    error_code: str,
    error: Exception,
    token: str,
) -> None:
    now = utc_now()
    emit_json(
        MonitorSummary(
            ok=False,
            status="failed",
            job_id=job_id,
            session_name=session_name,
            endpoint=endpoint,
            job_root=job_root,
            output_dir=str(output_dir.expanduser().resolve()),
            exit_code=exit_code,
            started_at=now,
            finished_at=now,
            elapsed_seconds=0.0,
            stdout_offset=0,
            stderr_offset=0,
            error_code=error_code,
            error=redact_text(
                f"{type(error).__name__}: {error}",
                secrets=(token,),
            ),
        )
    )


def monitor(
    job_id: Annotated[
        str,
        typer.Argument(help="Remote job id"),
    ],
    session: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--session",
            help="Session name",
        ),
    ] = None,
    interval: Annotated[
        float,
        typer.Option(
            "--interval",
            help="Job/log polling interval",
        ),
    ] = 5.0,
    probe_every: Annotated[
        float,
        typer.Option(
            "--probe-every",
            help="Resource sampling interval; 0 disables",
        ),
    ] = 60.0,
    probe_timeout: Annotated[
        float,
        typer.Option("--probe-timeout"),
    ] = 20.0,
    control_timeout: Annotated[
        float,
        typer.Option(
            "--control-timeout",
            help="Per job-control call deadline",
        ),
    ] = 30.0,
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            help="Local evidence directory",
        ),
    ] = None,
    max_bytes: Annotated[
        int,
        typer.Option(
            "--max-bytes",
            help="Maximum bytes per stream per poll",
        ),
    ] = 65536,
    job_root: Annotated[
        str,
        typer.Option("--job-root"),
    ] = DEFAULT_JOB_ROOT,
    timeout: Annotated[
        Optional[float],
        typer.Option(
            "--timeout",
            help=("Local monitor deadline; never cancels the remote job"),
        ),
    ] = None,
    once: Annotated[
        bool,
        typer.Option("--once"),
    ] = False,
    max_control_errors: Annotated[
        int,
        typer.Option("--max-control-errors"),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=("Emit the final colab.monitor.summary.v1 document on stdout"),
        ),
    ] = False,
):
    """Persist logs, job state, and resource samples locally."""
    from colab_cli.common import state

    output_dir = Path(output) if output else Path("runs") / job_id
    selected = None
    context = machine_diagnostics_to_stderr() if json_output else nullcontext()

    try:
        with context:
            selected = _select_monitor_session(
                state,
                session,
            )
            summary = run_monitor(
                state,
                job_id=job_id,
                session=selected,
                config=MonitorConfig(
                    output_dir=output_dir,
                    interval=interval,
                    probe_every=probe_every,
                    probe_timeout=probe_timeout,
                    control_timeout=control_timeout,
                    max_bytes=max_bytes,
                    job_root=job_root,
                    timeout=timeout,
                    once=once,
                    max_control_errors=(max_control_errors),
                ),
            )
    except (
        MonitorConfigurationError,
        MonitorBindingError,
        LookupError,
    ) as exc:
        exit_code = 2
        error_code = "MONITOR_CONFIGURATION_ERROR"
        if isinstance(exc, MonitorBindingError):
            error_code = "MONITOR_BINDING_ERROR"
        elif isinstance(exc, LookupError):
            error_code = "SESSION_SELECTION_ERROR"
        if json_output:
            _emit_command_error(
                job_id=job_id,
                session_name=(
                    selected.name if selected is not None else (session or "")
                ),
                endpoint=(selected.endpoint if selected is not None else ""),
                job_root=job_root,
                output_dir=output_dir,
                exit_code=exit_code,
                error_code=error_code,
                error=exc,
                token=(selected.token if selected is not None else ""),
            )
        else:
            typer.echo(
                f"[colab] Monitor refused: {exc}",
                err=True,
            )
        raise typer.Exit(exit_code) from exc
    except Exception as exc:
        exit_code = 1
        if json_output:
            _emit_command_error(
                job_id=job_id,
                session_name=(
                    selected.name if selected is not None else (session or "")
                ),
                endpoint=(selected.endpoint if selected is not None else ""),
                job_root=job_root,
                output_dir=output_dir,
                exit_code=exit_code,
                error_code="MONITOR_FAILED",
                error=exc,
                token=(selected.token if selected is not None else ""),
            )
        else:
            typer.echo(
                f"[colab] Monitor failed: {exc}",
                err=True,
            )
        raise typer.Exit(exit_code) from exc

    if json_output:
        emit_json(summary)
    else:
        typer.echo(
            f"[colab] Monitor {summary.status}: "
            f"remote={summary.remote_state}, "
            f"stdout={summary.stdout_offset}, "
            f"stderr={summary.stderr_offset}",
            err=True,
        )
        typer.echo(
            f"[colab] Evidence: {summary.output_dir}",
            err=True,
        )
        if summary.error:
            typer.echo(
                f"[colab] {summary.error_code}: {summary.error}",
                err=True,
            )
    raise typer.Exit(summary.exit_code)


def register(app: typer.Typer) -> None:
    app.command()(monitor)
