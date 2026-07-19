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

from colab_cli.jobs import RemoteJobClient


class FakeExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.codes = []

    def execute_json(self, code, *, timeout=30.0):
        self.codes.append((code, timeout))
        return self.results.pop(0)


def test_job_client_submits_argv_without_shell_interpolation():
    executor = FakeExecutor([{"job_id": "train", "state": "queued"}])
    jobs = RemoteJobClient(executor)

    result = jobs.submit(
        ["python", "-u", "train.py", "--name", "value with spaces"],
        job_id="train",
        cwd="/content/project",
        env={"RUN_ID": "r1"},
    )

    assert result["job_id"] == "train"
    code, timeout = executor.codes[0]
    assert "value with spaces" in code
    assert "dispatch('submit'" in code
    assert timeout == 30.0


def test_job_client_decodes_incremental_tail():
    executor = FakeExecutor(
        [
            {
                "job_id": "train",
                "stream": "stdout",
                "offset": 4,
                "next_offset": 7,
                "size": 7,
                "eof": True,
                "data": base64.b64encode(b"new").decode("ascii"),
            }
        ]
    )
    jobs = RemoteJobClient(executor)

    result = jobs.tail("train", stream="stdout", offset=4, max_bytes=100)

    assert result.data == b"new"
    assert result.next_offset == 7


def test_job_client_cancel_uses_bounded_control_timeout():
    executor = FakeExecutor([{"job_id": "train", "state": "cancelled"}])
    jobs = RemoteJobClient(executor)

    jobs.cancel("train", grace_seconds=3)

    assert executor.codes[0][1] == 8.0


def test_job_client_bootstraps_helper_once_per_connection():
    executor = FakeExecutor(
        [
            {"job_id": "train", "state": "running"},
            {"job_id": "train", "state": "succeeded", "returncode": 0},
        ]
    )
    jobs = RemoteJobClient(executor)

    jobs.status("train")
    jobs.status("train")

    first_code = executor.codes[0][0]
    second_code = executor.codes[1][0]
    assert "def dispatch(" in first_code
    assert "def dispatch(" not in second_code
    assert "dispatch('status'" in second_code
