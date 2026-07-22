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

"""Stable machine-readable contracts for Colab CLI observation commands."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SCHEMA_SESSIONS_V1 = "colab.sessions.v1"
SCHEMA_STATUS_V1 = "colab.status.v1"
SCHEMA_JOBS_V1 = "colab.jobs.v1"
DEFAULT_PROBE_TIMEOUT = 20.0


class ObservationIssue(BaseModel):
    code: str
    message: str
    source: str
    severity: Literal["warning", "error"] = "warning"
    retryable: bool | None = None


class LastExecutionObservation(BaseModel):
    file: str
    cell: str | None = None
    timestamp: str


class KeepAliveObservation(BaseModel):
    pid: int | None = None
    status: Literal["alive", "dead", "not_recorded", "unknown"]
    last_heartbeat_at: str | None = None
    unavailable_reason: str | None = None


class LocalSessionObservation(BaseModel):
    tracked: bool
    requested_variant: str | None = None
    requested_accelerator: str | None = None
    running: str | None = None
    kernel_id: str | None = None
    kernel_status: Literal["recorded", "not_recorded"]
    session_id: str | None = None
    keep_alive: KeepAliveObservation
    last_execution: LastExecutionObservation | None = None


class AssignmentObservation(BaseModel):
    status: Literal["ok", "missing", "unavailable", "not_queried"]
    endpoint: str
    accelerator: str | None = None
    variant: str | None = None
    machine_shape: str | None = None


class GpuObservation(BaseModel):
    available: bool = False
    name: str | None = None
    driver_version: str | None = None
    memory_total_bytes: int | None = None
    memory_used_bytes: int | None = None
    utilization_percent: float | None = None
    memory_utilization_percent: float | None = None
    temperature_c: float | None = None
    sources: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class MemoryObservation(BaseModel):
    total_bytes: int | None = None
    available_bytes: int | None = None
    used_bytes: int | None = None
    sources: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class DiskObservation(BaseModel):
    path: str = "/content"
    total_bytes: int | None = None
    used_bytes: int | None = None
    free_bytes: int | None = None
    sources: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class RuntimeObservation(BaseModel):
    boot_id: str | None = None
    python_version: str | None = None
    platform: str | None = None
    source: str | None = None
    unavailable_reason: str | None = None


class ProbeObservation(BaseModel):
    status: Literal["not_requested", "ok", "partial", "timeout", "unavailable"]
    observed_at: str | None = None
    duration_ms: int | None = None
    timeout_seconds: float | None = None
    gpu: GpuObservation = Field(default_factory=GpuObservation)
    memory: MemoryObservation = Field(default_factory=MemoryObservation)
    disk: DiskObservation = Field(default_factory=DiskObservation)
    runtime: RuntimeObservation = Field(default_factory=RuntimeObservation)
    issues: list[ObservationIssue] = Field(default_factory=list)


class ComputeUnitsObservation(BaseModel):
    balance: float | None = None
    consumption_rate_hourly: float | None = None
    unavailable_reason: str = "not_queried_by_structured_observability_v1"


class SessionObservation(BaseModel):
    name: str | None
    endpoint: str
    lifecycle: str
    local: LocalSessionObservation
    assignment: AssignmentObservation
    probe: ProbeObservation = Field(
        default_factory=lambda: ProbeObservation(status="not_requested")
    )
    compute_units: ComputeUnitsObservation = Field(
        default_factory=ComputeUnitsObservation
    )
    warnings: list[ObservationIssue] = Field(default_factory=list)


class SessionsEnvelope(BaseModel):
    schema_name: Literal["colab.sessions.v1"] = Field(
        SCHEMA_SESSIONS_V1, serialization_alias="schema"
    )
    ok: bool
    status: Literal["ok", "partial", "error"]
    generated_at: str
    sessions: list[SessionObservation]
    warnings: list[ObservationIssue] = Field(default_factory=list)
    errors: list[ObservationIssue] = Field(default_factory=list)


class StatusEnvelope(BaseModel):
    schema_name: Literal["colab.status.v1"] = Field(
        SCHEMA_STATUS_V1, serialization_alias="schema"
    )
    ok: bool
    status: Literal["ok", "partial", "error"]
    generated_at: str
    selected_session: str | None = None
    sessions: list[SessionObservation]
    warnings: list[ObservationIssue] = Field(default_factory=list)
    errors: list[ObservationIssue] = Field(default_factory=list)


class JobLogObservation(BaseModel):
    path: str | None = None
    size_bytes: int | None = None


class JobObservation(BaseModel):
    job_id: str
    state: str
    heartbeat_at: str | None = None
    returncode: int | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    runner_alive: bool | None = None
    runtime_id: str | None = None
    stdout: JobLogObservation = Field(default_factory=JobLogObservation)
    stderr: JobLogObservation = Field(default_factory=JobLogObservation)


class JobsEnvelope(BaseModel):
    schema_name: Literal["colab.jobs.v1"] = Field(
        SCHEMA_JOBS_V1, serialization_alias="schema"
    )
    ok: bool
    status: Literal["ok", "partial", "error"]
    generated_at: str
    session: str | None = None
    endpoint: str | None = None
    job_root: str
    jobs: list[JobObservation]
    warnings: list[ObservationIssue] = Field(default_factory=list)
    errors: list[ObservationIssue] = Field(default_factory=list)
