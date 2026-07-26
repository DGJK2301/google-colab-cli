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

import pytest

from requests import ReadTimeout

from colab_cli.transfer import FileTransfer


class FakeRemoteFiles:
    def __init__(self):
        self.files = {}
        self.finalized = []
        self.stat_calls = []
        self.retry_count = 0

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


def test_upload_empty_file_creates_and_finalizes_remote_target(tmp_path):
    source = tmp_path / "empty.bundle"
    source.write_bytes(b"")
    remote = FakeRemoteFiles()
    contents = FakeContents(remote)
    transfer = FileTransfer(contents, remote, chunk_size=4)

    result = transfer.upload(source, "content/empty.bundle")

    assert contents.calls == [
        (
            transfer.remote_temp_path(
                "content/empty.bundle",
                hashlib.sha256(b"").hexdigest(),
            ),
            b"",
            -1,
        )
    ]
    assert remote.files["content/empty.bundle"] == b""
    assert result.size == 0
    assert result.sha256 == hashlib.sha256(b"").hexdigest()


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


def test_download_empty_file_atomically_replaces_target(tmp_path):
    remote = FakeRemoteFiles()
    remote.files["content/empty.ckpt"] = b""
    contents = FakeContents(remote)
    transfer = FileTransfer(contents, remote, chunk_size=4)
    target = tmp_path / "empty.ckpt"
    target.write_bytes(b"stale")

    result = transfer.download("content/empty.ckpt", target)

    assert target.read_bytes() == b""
    assert result.size == 0
    assert result.sha256 == hashlib.sha256(b"").hexdigest()


def test_actual_upload_retries_are_counted(tmp_path):
    source = tmp_path / "archive.bundle"
    source.write_bytes(b"abcdefgh")
    remote = FakeRemoteFiles()

    class FailBeforeWrite(FakeContents):
        def __init__(self, remote):
            super().__init__(remote)
            self.fail_once = True

        def upload_chunk(self, path, data, *, chunk):
            if self.fail_once and data:
                self.fail_once = False
                raise ReadTimeout("request failed before write")
            return super().upload_chunk(
                path,
                data,
                chunk=chunk,
            )

    transfer = FileTransfer(
        FailBeforeWrite(remote),
        remote,
        chunk_size=4,
    )

    result = transfer.upload(
        source,
        "content/archive.bundle",
    )

    assert result.retry_count == 1
    assert remote.files["content/archive.bundle"] == (b"abcdefgh")


def test_lost_response_reconciliation_is_not_counted_as_replay(
    tmp_path,
):
    source = tmp_path / "archive.bundle"
    source.write_bytes(b"abcdefgh")
    remote = FakeRemoteFiles()
    transfer = FileTransfer(
        FakeContents(
            remote,
            fail_after_write_at=4,
        ),
        remote,
        chunk_size=4,
    )

    result = transfer.upload(
        source,
        "content/archive.bundle",
    )

    assert result.retry_count == 0


def test_interrupted_upload_preserves_verified_partial_for_resume(
    tmp_path,
):
    source = tmp_path / "archive.bundle"
    source.write_bytes(b"abcdefghij")
    remote = FakeRemoteFiles()

    def interrupt(progress):
        if progress.phase == "transferring" and progress.completed >= 4:
            raise KeyboardInterrupt

    first = FileTransfer(
        FakeContents(remote),
        remote,
        chunk_size=4,
        progress=interrupt,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    temp_path = first.remote_temp_path(
        "content/archive.bundle",
        digest,
    )

    with pytest.raises(KeyboardInterrupt):
        first.upload(
            source,
            "content/archive.bundle",
        )

    assert remote.files[temp_path] == b"abcd"

    second = FileTransfer(
        FakeContents(remote),
        remote,
        chunk_size=4,
    )
    result = second.upload(
        source,
        "content/archive.bundle",
    )

    assert result.resumed_from == 4
    assert remote.files["content/archive.bundle"] == (source.read_bytes())
    assert result.sha256 == digest


def test_interrupted_download_preserves_verified_partial_for_resume(
    tmp_path,
):
    remote = FakeRemoteFiles()
    remote.files["content/model.ckpt"] = b"abcdefghij"
    target = tmp_path / "model.ckpt"

    def interrupt(progress):
        if progress.phase == "transferring" and progress.completed >= 4:
            raise KeyboardInterrupt

    first = FileTransfer(
        FakeContents(remote),
        remote,
        chunk_size=4,
        progress=interrupt,
    )
    part = first.local_temp_path(target)

    with pytest.raises(KeyboardInterrupt):
        first.download(
            "content/model.ckpt",
            target,
        )

    assert part.read_bytes() == b"abcd"

    second = FileTransfer(
        FakeContents(remote),
        remote,
        chunk_size=4,
    )
    result = second.download(
        "content/model.ckpt",
        target,
    )

    assert result.resumed_from == 4
    assert target.read_bytes() == b"abcdefghij"
    assert not part.exists()
    assert result.sha256 == hashlib.sha256(b"abcdefghij").hexdigest()


def test_upload_result_includes_remote_control_reconnects(
    tmp_path,
):
    source = tmp_path / "archive.bundle"
    source.write_bytes(b"abcdefgh")
    remote = FakeRemoteFiles()

    class RemoteWithReconnect(FakeRemoteFiles):
        def __init__(self):
            super().__init__()
            self.bumped = False

        def stat_file(self, path, *, hash_limit=None):
            if not self.bumped:
                self.retry_count += 1
                self.bumped = True
            return super().stat_file(
                path,
                hash_limit=hash_limit,
            )

    remote = RemoteWithReconnect()
    transfer = FileTransfer(
        FakeContents(remote),
        remote,
        chunk_size=4,
    )

    result = transfer.upload(
        source,
        "content/archive.bundle",
    )

    assert result.retry_count == 1


def test_download_result_includes_remote_control_reconnects(
    tmp_path,
):
    class RemoteWithReconnect(FakeRemoteFiles):
        def __init__(self):
            super().__init__()
            self.bumped = False

        def read_chunk(self, path, *, offset, length):
            if not self.bumped:
                self.retry_count += 1
                self.bumped = True
            return super().read_chunk(
                path,
                offset=offset,
                length=length,
            )

    remote = RemoteWithReconnect()
    remote.files["content/model.ckpt"] = b"abcdefgh"
    target = tmp_path / "model.ckpt"
    transfer = FileTransfer(
        FakeContents(remote),
        remote,
        chunk_size=4,
    )

    result = transfer.download(
        "content/model.ckpt",
        target,
    )

    assert result.retry_count == 1
    assert target.read_bytes() == b"abcdefgh"
