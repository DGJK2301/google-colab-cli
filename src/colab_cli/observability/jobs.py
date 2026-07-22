# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Normalize persisted remote-job state into colab.jobs.v1."""

from __future__ import annotations

from typing import Any

from colab_cli.observability.collector import utc_now
from colab_cli.observability.models import (
    JobLogObservation,
    JobObservation,
    JobsEnvelope,
    ObservationIssue,
)
from colab_cli.observability.redaction import redact_text
from colab_cli.state import SessionState


def normalize_jobs(*, session: SessionState, job_root: str, records):
    jobs = []
    for record in records:
        jobs.append(
            JobObservation(
                job_id=str(record.get("job_id", "")),
                state=str(record.get("state", "unknown")),
                heartbeat_at=_string(record.get("heartbeat_at")),
                returncode=_integer(record.get("returncode")),
                error=(
                    redact_text(record["error"], secrets=(session.token,))
                    if record.get("error") is not None
                    else None
                ),
                created_at=_string(record.get("created_at")),
                started_at=_string(record.get("started_at")),
                finished_at=_string(record.get("finished_at")),
                runner_alive=record.get("runner_alive")
                if isinstance(record.get("runner_alive"), bool)
                else None,
                runtime_id=_string(record.get("runtime_id")),
                stdout=JobLogObservation(
                    path=_string(record.get("stdout_path")),
                    size_bytes=_integer(record.get("stdout_size")),
                ),
                stderr=JobLogObservation(
                    path=_string(record.get("stderr_path")),
                    size_bytes=_integer(record.get("stderr_size")),
                ),
            )
        )
    return JobsEnvelope(
        ok=True,
        status="ok",
        generated_at=utc_now(),
        session=session.name,
        endpoint=session.endpoint,
        job_root=job_root,
        jobs=jobs,
    )


def jobs_error_envelope(*, session_name, job_root, code, message):
    return JobsEnvelope(
        ok=False,
        status="error",
        generated_at=utc_now(),
        session=session_name,
        job_root=job_root,
        jobs=[],
        errors=[
            ObservationIssue(
                code=code,
                message=message,
                source="remote_jobs",
                severity="error",
            )
        ],
    )


def _string(value: Any):
    return None if value is None else str(value)


def _integer(value: Any):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
