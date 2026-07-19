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

import hashlib

from requests import ReadTimeout

from colab_cli.transfer import FileTransfer


class FakeRemoteFiles:
    def __init__(self):
        self.files = {}
        self.finalized = []
        self.stat_calls = []

    def stat_file(self, path, *, hash_limit=None):
        self.stat_calls.append((path, hash_limit))
        if path not in self.files:
            return {"exists": False, "path": path}
        data = self.files[path]
        if hash_limit is not None:
            data = data[:hash_limit]
        return {
            "exists": True,
            "path": path,
            "size": len(self.files[path]),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def finalize_upload(self, temp_path, remote_path, *, size, sha256, overwrite):
        data = self.files[temp_path]
        assert len(data) == size
        assert hashlib.sha256(data).hexdigest() == sha256
        if remote_path in self.files and not overwrite:
            raise FileExistsError(remote_path)
        self.files[remote_path] = data
        del self.files[temp_path]
        self.finalized.append((temp_path, remote_path))
        return self.stat_file(remote_path)

    def remove_file(self, path):
        self.files.pop(path, None)

    def read_chunk(self, path, *, offset, length):
        return self.files[path][offset : offset + length]


class FakeContents:
    def __init__(self, remote, fail_after_write_at=None):
        self.remote = remote
        self.calls = []
        self.fail_after_write_at = fail_after_write_at
        self.failed = False

    def upload_chunk(self, path, data, *, chunk):
        self.calls.append((path, data, chunk))
        if chunk == 1:
            self.remote.files[path] = data
        else:
            self.remote.files[path] = self.remote.files.get(path, b"") + data
        if (
            self.fail_after_write_at is not None
            and len(self.remote.files[path]) == self.fail_after_write_at
            and not self.failed
        ):
            self.failed = True
            raise ReadTimeout("response lost after server write")
        return {"type": "file", "size": len(self.remote.files[path])}


def test_upload_streams_chunks_and_atomically_finalizes(tmp_path):
    source = tmp_path / "archive.bundle"
    source.write_bytes(b"abcdefghij")
    remote = FakeRemoteFiles()
    contents = FakeContents(remote)
    progress = []
    transfer = FileTransfer(contents, remote, chunk_size=4, progress=progress.append)

    result = transfer.upload(source, "content/archive.bundle")

    assert [call[1] for call in contents.calls] == [b"abcd", b"efgh", b"ij", b""]
    assert [call[2] for call in contents.calls] == [1, 2, 2, -1]
    assert remote.files["content/archive.bundle"] == b"abcdefghij"
    assert result.sha256 == hashlib.sha256(b"abcdefghij").hexdigest()
    assert progress[-1].completed == 10
    assert not any(path.endswith(".part") for path in remote.files)
    assert any(limit == 0 for _path, limit in remote.stat_calls)


def test_upload_resumes_a_verified_remote_prefix(tmp_path):
    source = tmp_path / "archive.bundle"
    source.write_bytes(b"abcdefghij")
    remote = FakeRemoteFiles()
    contents = FakeContents(remote)
    transfer = FileTransfer(contents, remote, chunk_size=4)
    temp_path = transfer.remote_temp_path(
        "content/archive.bundle", hashlib.sha256(b"abcdefghij").hexdigest()
    )
    remote.files[temp_path] = b"abcdef"

    transfer.upload(source, "content/archive.bundle")

    assert [call[1] for call in contents.calls] == [b"ghij", b""]
    assert remote.files["content/archive.bundle"] == b"abcdefghij"


def test_upload_timeout_after_server_write_does_not_duplicate_chunk(tmp_path):
    source = tmp_path / "archive.bundle"
    source.write_bytes(b"abcdefgh")
    remote = FakeRemoteFiles()
    contents = FakeContents(remote, fail_after_write_at=4)
    transfer = FileTransfer(contents, remote, chunk_size=4)

    transfer.upload(source, "content/archive.bundle")

    assert remote.files["content/archive.bundle"] == b"abcdefgh"
    assert [call[1] for call in contents.calls] == [b"abcd", b"efgh", b""]


def test_download_resumes_verified_part_and_replaces_target(tmp_path):
    remote = FakeRemoteFiles()
    remote.files["content/model.ckpt"] = b"abcdefghij"
    contents = FakeContents(remote)
    transfer = FileTransfer(contents, remote, chunk_size=4)
    target = tmp_path / "model.ckpt"
    part = transfer.local_temp_path(target)
    part.write_bytes(b"abcde")

    result = transfer.download("content/model.ckpt", target)

    assert target.read_bytes() == b"abcdefghij"
    assert not part.exists()
    assert result.size == 10
    assert result.sha256 == hashlib.sha256(b"abcdefghij").hexdigest()
