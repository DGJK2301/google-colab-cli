# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Stable machine-readable contracts for local diagnostics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DoctorIssue(BaseModel):
    code: str
    message: str
    source: str
    severity: Literal["warning", "error"] = "warning"
    retryable: bool | None = None


class PackageObservation(BaseModel):
    name: str
    version: str | None = None
    status: Literal["ok", "missing", "error"]
    error: str | None = None


class RuntimeObservation(BaseModel):
    cli_version: str
    commit_sha: str | None = None
    commit_unavailable_reason: str | None = None
    install_path: str
    executable: str
    python_version: str
    platform: str


class PermissionObservation(BaseModel):
    status: Literal["secure", "insecure", "unknown", "not_applicable"]
    mode: str | None = None
    reason: str | None = None


class TokenObservation(BaseModel):
    provider: str
    path: str
    exists: bool
    readable: bool | None = None
    parse_status: Literal["ok", "missing", "invalid", "not_applicable"]
    expiry: str | None = None
    expired: bool | None = None
    scopes: list[str] = Field(default_factory=list)
    refresh_token_present: bool | None = None
    rapt_token_present: bool | None = None
    permission: PermissionObservation


class KeepAliveDiagnostic(BaseModel):
    session: str
    endpoint: str
    pid: int | None = None
    process_status: Literal["alive", "dead", "unknown", "not_recorded"]


class StoreObservation(BaseModel):
    path: str
    exists: bool
    parse_status: Literal["ok", "missing", "invalid"]
    entry_count: int = 0
    invalid_entries: list[str] = Field(default_factory=list)
    keep_alive: list[KeepAliveDiagnostic] = Field(default_factory=list)


class TransferLeaseDiagnostic(BaseModel):
    path: str
    lease_id: str | None = None
    direction: str | None = None
    target_path: str | None = None
    pid: int | None = None
    heartbeat_at: str | None = None
    metadata_state: str | None = None
    owner_status: Literal["alive", "dead", "unknown", "not_applicable"]
    diagnostic: Literal["active", "stale", "unsafe", "released", "invalid"]
    error: str | None = None


class TransferLeaseObservation(BaseModel):
    root: str
    exists: bool
    entries: list[TransferLeaseDiagnostic] = Field(default_factory=list)


class NetworkObservation(BaseModel):
    requested: bool
    status: Literal["not_requested", "ok", "error", "timeout"]
    assignment_count: int | None = None
    endpoints: list[str] = Field(default_factory=list)
    orphan_endpoints: list[str] = Field(default_factory=list)
    elapsed_seconds: float | None = None
    error: str | None = None


class DoctorEnvelope(BaseModel):
    schema_name: Literal["colab.doctor.v1"] = Field(
        "colab.doctor.v1",
        serialization_alias="schema",
    )
    ok: bool
    status: Literal["ok", "warning", "error"]
    generated_at: str
    runtime: RuntimeObservation
    dependencies: list[PackageObservation]
    token: TokenObservation
    session_store: StoreObservation
    settings_store: StoreObservation
    transfer_leases: TransferLeaseObservation
    network: NetworkObservation
    warnings: list[DoctorIssue] = Field(default_factory=list)
    errors: list[DoctorIssue] = Field(default_factory=list)
