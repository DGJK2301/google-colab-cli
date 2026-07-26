# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Persistent evidence contracts for foreground monitoring."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MonitorState(BaseModel):
    schema_name: Literal["colab.monitor.state.v1"] = Field(
        "colab.monitor.state.v1",
        serialization_alias="schema",
    )
    job_id: str
    session_name: str
    endpoint: str
    job_root: str
    output_dir: str
    stdout_offset: int = 0
    stderr_offset: int = 0
    remote_runtime_id: str | None = None
    probe_boot_id: str | None = None
    created_at: str
    updated_at: str
    monitor_runs: int = 1
    monitor_pid: int
    last_job_state: str | None = None
    last_returncode: int | None = None
    last_heartbeat_at: str | None = None
    consecutive_control_errors: int = 0
    status: Literal[
        "running",
        "terminal",
        "snapshot",
        "interrupted",
        "timeout",
        "failed",
    ] = "running"


class MonitorSummary(BaseModel):
    schema_name: Literal["colab.monitor.summary.v1"] = Field(
        "colab.monitor.summary.v1",
        serialization_alias="schema",
    )
    ok: bool
    status: Literal[
        "completed",
        "snapshot",
        "interrupted",
        "timeout",
        "failed",
    ]
    job_id: str
    session_name: str
    endpoint: str
    job_root: str
    output_dir: str
    remote_state: str | None = None
    remote_returncode: int | None = None
    exit_code: int
    started_at: str
    finished_at: str
    elapsed_seconds: float
    stdout_offset: int
    stderr_offset: int
    remote_runtime_id: str | None = None
    probe_boot_id: str | None = None
    error_code: str | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)
