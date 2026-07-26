# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Stable machine-readable contracts for verified file transfers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SCHEMA_TRANSFER_V1 = "colab.transfer.v1"


class TransferErrorObservation(BaseModel):
    code: str
    message: str
    retryable: bool | None = None


class TransferLeaseObservation(BaseModel):
    lease_id: str | None = None
    lock_key: str | None = None
    stale_reclaimed: bool = False
    stale_reclaimed_from: str | None = None


class TransferEnvelope(BaseModel):
    schema_name: Literal["colab.transfer.v1"] = Field(
        SCHEMA_TRANSFER_V1,
        serialization_alias="schema",
    )
    ok: bool
    status: Literal[
        "completed",
        "interrupted",
        "failed",
        "busy",
    ]
    direction: Literal["upload", "download"]
    session: str | None = None
    endpoint: str | None = None
    source_path: str
    target_path: str
    partial_path: str | None = None
    completed_bytes: int
    total_bytes: int | None = None
    resumed_from: int
    elapsed_seconds: float
    mib_per_second: float | None = None
    eta_seconds: float | None = None
    retry_count: int
    sha256: str | None = None
    resume_command: str | None = None
    resume_argv: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str
    lease: TransferLeaseObservation = Field(default_factory=TransferLeaseObservation)
    warnings: list[str] = Field(default_factory=list)
    error: TransferErrorObservation | None = None
