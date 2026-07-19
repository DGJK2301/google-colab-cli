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

import pytest

from colab_cli.remote import RemoteExecutionError, RemoteExecutor, RemoteFileOps


class FakeRuntime:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []
        self.stopped = False

    def execute_code(self, code, timeout=None):
        self.calls.append((code, timeout))
        marker = code.split("_COLAB_CLI_RESULT_MARKER = ", 1)[1].splitlines()[0]
        marker = marker.strip().strip("'\"")
        materialized = []
        for output in self.outputs:
            if output == "RESULT":
                materialized.append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": f"noise\n{marker}{'{"ok": true}'}\n",
                    }
                )
            else:
                materialized.append(output)
        return materialized

    def stop(self):
        self.stopped = True


def test_remote_executor_returns_only_marked_json():
    runtime = FakeRuntime(["RESULT"])
    executor = RemoteExecutor(runtime)

    result = executor.execute_json("_colab_cli_result = {'ok': True}")

    assert result == {"ok": True}
    assert runtime.calls[0][1] == 30.0


def test_remote_executor_raises_on_jupyter_error():
    runtime = FakeRuntime(
        [{"output_type": "error", "ename": "RuntimeError", "evalue": "boom"}]
    )
    executor = RemoteExecutor(runtime)

    with pytest.raises(RemoteExecutionError, match="RuntimeError: boom"):
        executor.execute_json("raise RuntimeError('boom')")


def test_remote_executor_closes_runtime():
    runtime = FakeRuntime(["RESULT"])
    executor = RemoteExecutor(runtime)

    executor.close()

    assert runtime.stopped


class FakeExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.codes = []

    def execute_json(self, code, *, timeout=30.0):
        self.codes.append(code)
        return self.results.pop(0)


def test_remote_file_ops_decodes_read_chunk():
    executor = FakeExecutor(
        [{"data": base64.b64encode(b"abc").decode("ascii"), "size": 3}]
    )
    remote = RemoteFileOps(executor)

    assert remote.read_chunk("content/a.bin", offset=4, length=3) == b"abc"
    assert "offset = 4" in executor.codes[0]
    assert "length = 3" in executor.codes[0]


def test_remote_file_ops_finalize_requires_size_hash_and_atomic_replace():
    executor = FakeExecutor(
        [{"exists": True, "path": "content/a.bin", "size": 3, "sha256": "abc"}]
    )
    remote = RemoteFileOps(executor)

    result = remote.finalize_upload(
        "content/a.part",
        "content/a.bin",
        size=3,
        sha256="abc",
        overwrite=True,
    )

    assert result["sha256"] == "abc"
    code = executor.codes[0]
    assert "os.replace" in code
    assert "expected_size = 3" in code
    assert "expected_sha256 = 'abc'" in code
