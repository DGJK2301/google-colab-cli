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

from __future__ import annotations

import base64
from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Any
import uuid

from colab_cli import remote_job_runtime
from colab_cli.remote import RemoteExecutor


DEFAULT_JOB_ROOT = "/content/.colab-cli/jobs"


@dataclass(frozen=True)
class JobTail:
    job_id: str
    stream: str
    offset: int
    next_offset: int
    size: int
    eof: bool
    data: bytes


class RemoteJobClient:
    def __init__(
        self, executor: RemoteExecutor, *, job_root: str = DEFAULT_JOB_ROOT
    ) -> None:
        self.executor = executor
        self.job_root = job_root
        self._helper_source = Path(remote_job_runtime.__file__).read_text(
            encoding="utf-8"
        )
        self._bootstrapped = False

    def submit(
        self,
        argv: list[str],
        *,
        job_id: str | None = None,
        cwd: str = "/content",
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not job_id:
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%d-%H%M%S"
            )
            job_id = f"job-{timestamp}-{uuid.uuid4().hex[:6]}"
        return self._call(
            "submit",
            {
                "root": self.job_root,
                "job_id": job_id,
                "argv": argv,
                "cwd": cwd,
                "env": env or {},
            },
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        return self._call("jobs", {"root": self.job_root})["jobs"]

    def status(self, job_id: str) -> dict[str, Any]:
        return self._call("status", {"root": self.job_root, "job_id": job_id})

    def tail(
        self,
        job_id: str,
        *,
        stream: str = "stdout",
        offset: int = 0,
        max_bytes: int = 65536,
    ) -> JobTail:
        result = self._call(
            "tail",
            {
                "root": self.job_root,
                "job_id": job_id,
                "stream": stream,
                "offset": offset,
                "max_bytes": max_bytes,
            },
        )
        return JobTail(
            job_id=result["job_id"],
            stream=result["stream"],
            offset=int(result["offset"]),
            next_offset=int(result["next_offset"]),
            size=int(result["size"]),
            eof=bool(result["eof"]),
            data=base64.b64decode(result["data"], validate=True),
        )

    def cancel(self, job_id: str, *, grace_seconds: float = 10.0) -> dict[str, Any]:
        return self._call(
            "cancel",
            {
                "root": self.job_root,
                "job_id": job_id,
                "grace_seconds": grace_seconds,
            },
            timeout=grace_seconds + 5.0,
        )

    def _call(
        self, operation: str, payload: dict[str, Any], *, timeout: float = 30.0
    ) -> dict[str, Any]:
        bootstrap = (
            f"{self._helper_source.rstrip()}\n\n" if not self._bootstrapped else ""
        )
        code = bootstrap + (
            f"_colab_cli_result = dispatch({operation!r}, {payload!r})\n"
        )
        result = self.executor.execute_json(code, timeout=timeout)
        self._bootstrapped = True
        return result
