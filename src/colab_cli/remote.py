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
import json
from typing import Any
import uuid

from colab_cli.runtime import ColabRuntime
from colab_cli.state import SessionState


class RemoteExecutionError(RuntimeError):
    pass


class RemoteExecutor:
    """Run bounded control code and return one marker-delimited JSON object."""

    def __init__(self, runtime: ColabRuntime):
        self.runtime = runtime

    def execute_json(self, code: str, *, timeout: float = 30.0) -> dict[str, Any]:
        marker = f"__COLAB_CLI_RESULT_{uuid.uuid4().hex}__"
        wrapped = (
            f"_COLAB_CLI_RESULT_MARKER = {marker!r}\n"
            "import json as _colab_cli_json\n"
            f"{code.rstrip()}\n"
            "print(_COLAB_CLI_RESULT_MARKER + "
            "_colab_cli_json.dumps(_colab_cli_result, sort_keys=True), flush=True)\n"
        )
        outputs = self.runtime.execute_code(wrapped, timeout=timeout)
        text_parts = []
        for output in outputs:
            if output.get("output_type") == "error":
                name = output.get("ename", "RemoteError")
                value = output.get("evalue", "")
                raise RemoteExecutionError(f"{name}: {value}".rstrip())
            text = output.get("text")
            if isinstance(text, list):
                text_parts.extend(str(part) for part in text)
            elif text is not None:
                text_parts.append(str(text))
            data = output.get("data")
            if isinstance(data, dict) and "text/plain" in data:
                plain = data["text/plain"]
                if isinstance(plain, list):
                    text_parts.extend(str(part) for part in plain)
                else:
                    text_parts.append(str(plain))

        combined = "".join(text_parts)
        marker_index = combined.find(marker)
        if marker_index < 0:
            raise RemoteExecutionError("Remote control call returned no result marker")
        payload = combined[marker_index + len(marker) :].lstrip()
        try:
            result, _ = json.JSONDecoder().raw_decode(payload)
        except (TypeError, ValueError) as exc:
            raise RemoteExecutionError(
                "Remote control result was not valid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise RemoteExecutionError("Remote control result must be a JSON object")
        return result

    def close(self) -> None:
        self.runtime.stop()


def open_remote_executor(session: SessionState, store, history=None) -> RemoteExecutor:
    def on_kernel_started(kernel_id: str) -> None:
        session.kernel_id = kernel_id
        store.add(session)

    def on_session_started(session_id: str) -> None:
        session.session_id = session_id
        store.add(session)

    runtime = ColabRuntime(
        session.url,
        session.token,
        session_name=session.name,
        history=history,
        kernel_id=session.kernel_id,
        session_id=session.session_id,
        on_kernel_started=on_kernel_started,
        on_session_started=on_session_started,
    )
    return RemoteExecutor(runtime)


class RemoteFileOps:
    def __init__(self, executor: RemoteExecutor):
        self.executor = executor

    def stat_file(self, path: str, *, hash_limit: int | None = None) -> dict:
        code = f"""
import hashlib
import os
path = {path!r}
hash_limit = {hash_limit!r}
runtime_path = '/' + path.lstrip('/')
if not os.path.exists(runtime_path):
    _colab_cli_result = {{'exists': False, 'path': path}}
elif not os.path.isfile(runtime_path):
    raise IsADirectoryError(runtime_path)
else:
    digest = hashlib.sha256()
    remaining = hash_limit
    with open(runtime_path, 'rb') as stream:
        while remaining is None or remaining > 0:
            block_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    _colab_cli_result = {{
        'exists': True,
        'path': path,
        'size': os.path.getsize(runtime_path),
        'sha256': digest.hexdigest(),
    }}
"""
        return self.executor.execute_json(code)

    def finalize_upload(
        self,
        temp_path: str,
        remote_path: str,
        *,
        size: int,
        sha256: str,
        overwrite: bool,
    ) -> dict:
        code = f"""
import hashlib
import os
temp_path = {temp_path!r}
remote_path = {remote_path!r}
expected_size = {size!r}
expected_sha256 = {sha256!r}
overwrite = {overwrite!r}
temp_runtime_path = '/' + temp_path.lstrip('/')
remote_runtime_path = '/' + remote_path.lstrip('/')
if not os.path.isfile(temp_runtime_path):
    raise FileNotFoundError(temp_runtime_path)
actual_size = os.path.getsize(temp_runtime_path)
digest = hashlib.sha256()
with open(temp_runtime_path, 'rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
actual_sha256 = digest.hexdigest()
if actual_size != expected_size or actual_sha256 != expected_sha256:
    raise IOError(
        f'Upload verification failed: expected {{expected_size}}/{{expected_sha256}}, '
        f'got {{actual_size}}/{{actual_sha256}}'
    )
if os.path.exists(remote_runtime_path) and not overwrite:
    raise FileExistsError(remote_runtime_path)
os.makedirs(os.path.dirname(remote_runtime_path) or '/', exist_ok=True)
os.replace(temp_runtime_path, remote_runtime_path)
_colab_cli_result = {{
    'exists': True,
    'path': remote_path,
    'size': actual_size,
    'sha256': actual_sha256,
}}
"""
        return self.executor.execute_json(code, timeout=120.0)

    def remove_file(self, path: str) -> None:
        code = f"""
import os
path = {path!r}
runtime_path = '/' + path.lstrip('/')
removed = False
if os.path.exists(runtime_path):
    if not os.path.isfile(runtime_path):
        raise IsADirectoryError(runtime_path)
    os.remove(runtime_path)
    removed = True
_colab_cli_result = {{'path': path, 'removed': removed}}
"""
        self.executor.execute_json(code)

    def read_chunk(self, path: str, *, offset: int, length: int) -> bytes:
        code = f"""
import base64
import os
path = {path!r}
offset = {offset!r}
length = {length!r}
runtime_path = '/' + path.lstrip('/')
if not os.path.isfile(runtime_path):
    raise FileNotFoundError(runtime_path)
with open(runtime_path, 'rb') as stream:
    stream.seek(offset)
    data = stream.read(length)
_colab_cli_result = {{
    'path': path,
    'offset': offset,
    'size': len(data),
    'data': base64.b64encode(data).decode('ascii'),
}}
"""
        result = self.executor.execute_json(code)
        return base64.b64decode(result["data"], validate=True)
