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
from urllib.parse import quote

import requests

from colab_cli.state import SessionState
from colab_cli.utils import get_status_code


class ContentsClient:
    def __init__(
        self,
        session_state: SessionState,
        request_timeout: tuple[float, float] = (10.0, 60.0),
        upload_request_timeout: tuple[float, float] = (60.0, 120.0),
    ):
        self.base_url = session_state.url.rstrip("/")
        self.token = session_state.token
        self.request_timeout = request_timeout
        self.upload_request_timeout = upload_request_timeout

    def _request(
        self,
        method: str,
        path: str,
        params: dict = None,
        json_data: dict = None,
        timeout: tuple[float, float] | None = None,
    ):
        # Quote the path, but don't encode slashes so directory paths stay intact
        quoted_path = quote(path.strip("/"), safe="/")
        url = f"{self.base_url}/api/contents/{quoted_path}"

        req_params = {"authuser": "0", "colab-runtime-proxy-token": self.token}
        if params:
            req_params.update(params)

        response = requests.request(
            method,
            url,
            params=req_params,
            json=json_data,
            timeout=self.request_timeout if timeout is None else timeout,
        )

        if get_status_code(response) == 404:
            raise FileNotFoundError(f"File or directory not found: {path}")

        response.raise_for_status()

        # DELETE doesn't return JSON
        if method == "DELETE":
            return None

        return response.json()

    def list_dir(self, path: str):
        return self._request("GET", path)

    def upload(self, local_path: str, remote_path: str):
        with open(local_path, "rb") as f:
            content = f.read()

        return self.upload_chunk(remote_path, content, chunk=1)

    def upload_chunk(self, remote_path: str, content: bytes, *, chunk: int):
        """Write one Jupyter LargeFileManager upload chunk.

        ``chunk=1`` truncates/creates the file, positive later values append,
        and ``chunk=-1`` appends the final chunk and runs post-save hooks.
        """

        b64_content = base64.b64encode(content).decode("ascii")
        filename = remote_path.split("/")[-1]

        payload = {
            "name": filename,
            "path": remote_path,
            "type": "file",
            "format": "base64",
            "content": b64_content,
            "chunk": chunk,
        }

        # Requests sends the request body before switching the socket to the
        # response read timeout. Slow Colab tunnels can therefore exhaust the
        # shorter connect-phase timeout while writing a base64 chunk. Keep
        # metadata operations responsive, but give bounded uploads their own
        # write/read budget.
        return self._request(
            "PUT",
            remote_path,
            json_data=payload,
            timeout=self.upload_request_timeout,
        )

    def download(self, remote_path: str, local_path: str):
        data = self._request("GET", remote_path, params={"content": "1"})

        if data.get("type") == "directory":
            raise IsADirectoryError(f"Cannot download a directory: {remote_path}")

        content = data.get("content", "")
        fmt = data.get("format")

        if fmt == "base64":
            content_bytes = base64.b64decode(content)
        else:
            # Assume text if it's not base64 explicitly encoded
            content_bytes = str(content).encode("utf-8")

        with open(local_path, "wb") as f:
            f.write(content_bytes)

    def rm(self, remote_path: str):
        self._request("DELETE", remote_path)
