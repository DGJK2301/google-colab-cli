# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Transfer telemetry, resume commands, and JSON rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Literal

from colab_cli.observability.redaction import redact_text
from colab_cli.transfer import (
    TransferProgress,
    TransferResult,
)
from colab_cli.transfer_lease import TransferLease
from colab_cli.transfer_models import (
    TransferEnvelope,
    TransferErrorObservation,
    TransferLeaseObservation,
)


_MIB = 1024 * 1024


@dataclass
class TransferTelemetry:
    direction: Literal["upload", "download"]
    session: str | None
    endpoint: str | None
    source_path: str
    target_path: str
    resume_argv: list[str]
    started_at: str = field(default_factory=lambda: _utcnow())
    started_monotonic: float = field(default_factory=time.monotonic)
    completed_bytes: int = 0
    total_bytes: int | None = None
    resumed_from: int = 0
    retry_count: int = 0
    sha256: str | None = None
    partial_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def update(
        self,
        progress: TransferProgress,
    ) -> None:
        self.completed_bytes = int(progress.completed)
        self.total_bytes = int(progress.total)
        self.resumed_from = int(progress.resumed_from)
        self.retry_count = int(progress.retry_count)
        self.sha256 = progress.sha256 or self.sha256
        self.partial_path = progress.partial_path or self.partial_path

    def finish(
        self,
        result: TransferResult,
    ) -> None:
        self.completed_bytes = result.size
        self.total_bytes = result.size
        self.resumed_from = result.resumed_from
        self.retry_count = result.retry_count
        self.sha256 = result.sha256

    def elapsed_seconds(self) -> float:
        return max(
            0.0,
            time.monotonic() - self.started_monotonic,
        )

    def rate_mib_per_second(
        self,
    ) -> float | None:
        elapsed = self.elapsed_seconds()
        transferred = max(
            0,
            self.completed_bytes - self.resumed_from,
        )
        if elapsed <= 0 or transferred <= 0:
            return None
        return transferred / _MIB / elapsed

    def eta_seconds(self) -> float | None:
        if self.total_bytes is None:
            return None
        remaining = max(
            0,
            self.total_bytes - self.completed_bytes,
        )
        if remaining == 0:
            return 0.0
        rate = self.rate_mib_per_second()
        if not rate or rate <= 0:
            return None
        return remaining / _MIB / rate

    def resume_command(self) -> str:
        return render_argv(self.resume_argv)

    def envelope(
        self,
        *,
        status: Literal[
            "completed",
            "interrupted",
            "failed",
            "busy",
        ],
        lease: TransferLease | None,
        error_code: str | None = None,
        error_message: str | None = None,
        retryable: bool | None = None,
        secrets: tuple[str, ...] = (),
    ) -> TransferEnvelope:
        rate = self.rate_mib_per_second()
        eta = self.eta_seconds()
        error = None
        if error_code is not None:
            error = TransferErrorObservation(
                code=error_code,
                message=redact_text(
                    error_message or error_code,
                    secrets=secrets,
                ),
                retryable=retryable,
            )

        lease_info = TransferLeaseObservation()
        if lease is not None:
            lease_info = TransferLeaseObservation(
                lease_id=lease.lease_id,
                lock_key=lease.lock_key,
                stale_reclaimed=(lease.stale_reclaimed),
                stale_reclaimed_from=(lease.stale_reclaimed_from),
            )

        has_resume = status != "completed" and bool(self.resume_argv)
        return TransferEnvelope(
            ok=status == "completed",
            status=status,
            direction=self.direction,
            session=self.session,
            endpoint=self.endpoint,
            source_path=self.source_path,
            target_path=self.target_path,
            partial_path=self.partial_path,
            completed_bytes=self.completed_bytes,
            total_bytes=self.total_bytes,
            resumed_from=self.resumed_from,
            elapsed_seconds=round(
                self.elapsed_seconds(),
                6,
            ),
            mib_per_second=(round(rate, 6) if rate is not None else None),
            eta_seconds=(round(eta, 6) if eta is not None else None),
            retry_count=self.retry_count,
            sha256=self.sha256,
            resume_command=(self.resume_command() if has_resume else None),
            resume_argv=(self.resume_argv if has_resume else []),
            started_at=self.started_at,
            finished_at=_utcnow(),
            lease=lease_info,
            warnings=[
                redact_text(
                    item,
                    secrets=secrets,
                )
                for item in self.warnings
            ],
            error=error,
        )


def build_resume_argv(
    *,
    state,
    direction: Literal["upload", "download"],
    session_name: str,
    source_path: str,
    target_path: str,
    chunk_size_mib: float,
    overwrite: bool | None,
    json_output: bool,
) -> list[str]:
    raw_provider = getattr(
        state.auth_provider,
        "value",
        state.auth_provider,
    )
    provider = raw_provider if isinstance(raw_provider, str) else "oauth2"
    argv = [
        "colab",
        "--auth",
        provider,
    ]

    oauth_config = state.client_oauth_config
    if provider == "oauth2" and isinstance(
        oauth_config,
        (str, os.PathLike),
    ):
        argv += [
            "--client-oauth-config",
            str(Path(oauth_config).expanduser().resolve()),
        ]

    config_path = state.config_path
    if (
        isinstance(
            config_path,
            (str, os.PathLike),
        )
        and config_path
    ):
        argv += [
            "--config",
            str(Path(config_path).expanduser().resolve()),
        ]

    argv += [
        direction,
        "--session",
        session_name,
        "--chunk-size-mib",
        format(chunk_size_mib, ".12g"),
        "--resume",
    ]
    if direction == "upload":
        argv.append("--overwrite" if overwrite else "--no-overwrite")
    if json_output:
        argv.append("--json")
    return [
        *argv,
        source_path,
        target_path,
    ]


def emit_transfer_json(
    envelope: TransferEnvelope,
) -> None:
    print(
        envelope.model_dump_json(
            indent=2,
            by_alias=True,
        ),
        flush=True,
    )


def render_argv(argv: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def progress_line(
    telemetry: TransferTelemetry,
) -> str:
    total = telemetry.total_bytes or 0
    percent = 100.0 * telemetry.completed_bytes / total if total else 100.0
    rate = telemetry.rate_mib_per_second()
    eta = telemetry.eta_seconds()
    rate_text = f", {rate:.2f} MiB/s" if rate is not None else ""
    eta_text = f", ETA {eta:.1f}s" if eta is not None else ""
    return (
        f"[colab] {telemetry.direction} "
        f"{percent:5.1f}% "
        f"({telemetry.completed_bytes}/"
        f"{total} bytes, "
        f"retries={telemetry.retry_count}"
        f"{rate_text}{eta_text})"
    )


def history_payload(
    telemetry: TransferTelemetry,
    *,
    state: str,
    error: str | None = None,
    secrets: tuple[str, ...] = (),
) -> dict:
    payload = {
        "op": telemetry.direction,
        "state": state,
        "source": telemetry.source_path,
        "target": telemetry.target_path,
        "partial_path": telemetry.partial_path,
        "completed_bytes": (telemetry.completed_bytes),
        "total_bytes": telemetry.total_bytes,
        "resumed_from": telemetry.resumed_from,
        "retry_count": telemetry.retry_count,
        "sha256": telemetry.sha256,
        "elapsed_seconds": round(
            telemetry.elapsed_seconds(),
            6,
        ),
    }
    if error is not None:
        payload["error"] = redact_text(
            error,
            secrets=secrets,
        )
    return payload


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
