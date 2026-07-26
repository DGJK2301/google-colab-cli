# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Machine-readable result for adopting an existing assignment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AttachError(BaseModel):
    code: str
    message: str
    retryable: bool | None = None


class AttachEnvelope(BaseModel):
    schema_name: Literal["colab.attach.v1"] = Field(
        "colab.attach.v1",
        serialization_alias="schema",
    )
    ok: bool
    status: Literal["attached", "failed"]
    session_name: str
    endpoint: str
    accelerator: str | None = None
    variant: str | None = None
    machine_shape: str | None = None
    keep_alive_pid: int | None = None
    control_connected: bool = False
    kernel_id: str | None = None
    session_id: str | None = None
    attached_at: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: AttachError | None = None
