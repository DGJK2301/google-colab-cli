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

import base64
import json
import os
import subprocess
import sys
import time

import pytest

from colab_cli import remote_job_runtime
from colab_cli.remote_job_runtime import _pid_alive, _process_start_token, dispatch


def _wait_for_terminal(root, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = dispatch("status", {"root": str(root), "job_id": job_id})
        if status["state"] in {"succeeded", "failed", "cancelled", "lost"}:
            return status
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {status}")


def test_pid_probe_does_not_terminate_the_process():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _pid_alive(process.pid)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_pid_probe_rejects_reused_pid_start_token():
    token = _process_start_token(os.getpid())
    if token is None:
        pytest.skip("process start tokens are unavailable on this platform")

    assert not _pid_alive(os.getpid(), expected_start_token=f"{token}-different")


def test_status_marks_foreign_runtime_lost_without_probing_pid(tmp_path, monkeypatch):
    job_dir = tmp_path / "old-job"
    job_dir.mkdir()
    (job_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": "old-job",
                "state": "running",
                "runner_pid": os.getpid(),
                "runtime_id": "previous-runtime",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(remote_job_runtime, "_runtime_id", lambda: "current-runtime")

    def fail_if_probed(*_args, **_kwargs):
        raise AssertionError("foreign-runtime PID must not be probed")

    monkeypatch.setattr(remote_job_runtime, "_pid_alive", fail_if_probed)

    status = dispatch("status", {"root": str(tmp_path), "job_id": "old-job"})

    assert status["state"] == "lost"
    assert "runtime identity changed" in status["error"]


def test_job_runtime_persists_status_logs_and_exit_code(tmp_path):
    result = dispatch(
        "submit",
        {
            "root": str(tmp_path),
            "job_id": "smoke-job",
            "argv": [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('hello'); print('warning', file=sys.stderr)",
            ],
            "cwd": str(tmp_path),
            "env": {"COLAB_CLI_TEST": "1"},
        },
    )

    assert result["job_id"] == "smoke-job"
    status = _wait_for_terminal(tmp_path, "smoke-job")
    assert status["state"] == "succeeded"
    assert status["returncode"] == 0
    assert status["runner_pid"] > 0
    stdout = dispatch(
        "tail",
        {
            "root": str(tmp_path),
            "job_id": "smoke-job",
            "stream": "stdout",
            "offset": 0,
            "max_bytes": 1024,
        },
    )
    stderr = dispatch(
        "tail",
        {
            "root": str(tmp_path),
            "job_id": "smoke-job",
            "stream": "stderr",
            "offset": 0,
            "max_bytes": 1024,
        },
    )
    assert (
        base64.b64decode(stdout["data"]) == b"hello\r\n"
        or base64.b64decode(stdout["data"]) == b"hello\n"
    )
    assert b"warning" in base64.b64decode(stderr["data"])
    assert stdout["next_offset"] == stdout["size"]
    assert (tmp_path / "smoke-job" / "spec.json").exists()
    assert (tmp_path / "smoke-job" / "status.json").exists()


def test_job_runtime_tail_resumes_from_byte_offset(tmp_path):
    job_dir = tmp_path / "existing"
    job_dir.mkdir()
    (job_dir / "stdout.log").write_bytes(b"first\nsecond\n")

    result = dispatch(
        "tail",
        {
            "root": str(tmp_path),
            "job_id": "existing",
            "stream": "stdout",
            "offset": 6,
            "max_bytes": 1024,
        },
    )

    assert base64.b64decode(result["data"]) == b"second\n"
    assert result["next_offset"] == len(b"first\nsecond\n")


def test_job_runtime_rejects_path_traversal_job_id(tmp_path):
    try:
        dispatch(
            "status",
            {"root": str(tmp_path), "job_id": "../outside"},
        )
    except ValueError as exc:
        assert "job_id" in str(exc)
    else:
        raise AssertionError("path-traversal job id was accepted")
