# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Local and opt-in network diagnostics."""

from __future__ import annotations

import typer
from typing_extensions import Annotated

from colab_cli.doctor import (
    collect_doctor,
    validate_doctor_timeout,
)
from colab_cli.observability import (
    emit_json,
    machine_diagnostics_to_stderr,
)


def _timeout_callback(value: float) -> float:
    try:
        return validate_doctor_timeout(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def doctor(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the stable colab.doctor.v1 schema",
        ),
    ] = False,
    network: Annotated[
        bool,
        typer.Option(
            "--network",
            help=("Run one bounded assignment query; never allocate a runtime"),
        ),
    ] = False,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="Network query timeout in seconds",
            callback=_timeout_callback,
        ),
    ] = 10.0,
):
    """Inspect installation, auth cache, state, and transfer leases."""
    from colab_cli.common import state

    if json_output:
        with machine_diagnostics_to_stderr():
            envelope = collect_doctor(
                state,
                network=network,
                timeout=timeout,
            )
        emit_json(envelope)
    else:
        envelope = collect_doctor(
            state,
            network=network,
            timeout=timeout,
        )
        typer.echo(f"Colab CLI doctor: {envelope.status.upper()}")
        typer.echo(f"Version: {envelope.runtime.cli_version}")
        typer.echo(
            f"Token: {envelope.token.parse_status} ({envelope.token.permission.status})"
        )
        typer.echo(
            f"Sessions: "
            f"{envelope.session_store.parse_status}, "
            f"{envelope.session_store.entry_count} entries"
        )
        typer.echo(f"Transfer leases: {len(envelope.transfer_leases.entries)} entries")
        typer.echo(f"Network: {envelope.network.status}")
        for issue in [
            *envelope.errors,
            *envelope.warnings,
        ]:
            typer.echo(
                f"[{issue.severity}] {issue.code}: {issue.message}",
                err=True,
            )
    if envelope.errors:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    app.command()(doctor)
