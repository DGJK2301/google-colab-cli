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

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable, Protocol

import requests

from colab_cli.contents import ContentsClient


DEFAULT_CHUNK_SIZE = 256 * 1024


class TransferIntegrityError(IOError):
    """Transferred bytes or verified partial state violated an invariant."""


@dataclass(frozen=True)
class TransferProgress:
    direction: str
    completed: int
    total: int
    resumed_from: int = 0
    retry_count: int = 0
    sha256: str | None = None
    partial_path: str | None = None
    phase: str = "transferring"


@dataclass(frozen=True)
class TransferResult:
    path: str
    size: int
    sha256: str
    resumed_from: int
    retry_count: int = 0


class RemoteFileOperations(Protocol):
    def stat_file(self, path: str, *, hash_limit: int | None = None) -> dict: ...

    def finalize_upload(
        self,
        temp_path: str,
        remote_path: str,
        *,
        size: int,
        sha256: str,
        overwrite: bool,
    ) -> dict: ...

    def remove_file(self, path: str) -> None: ...

    def read_chunk(self, path: str, *, offset: int, length: int) -> bytes: ...


class FileTransfer:
    """Resumable, verified file transfer over bounded remote operations."""

    def __init__(
        self,
        contents: ContentsClient,
        remote: RemoteFileOperations,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_attempts: int = 3,
        progress: Callable[[TransferProgress], None] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.contents = contents
        self.remote = remote
        self.chunk_size = chunk_size
        self.max_attempts = max_attempts
        self.progress = progress

    @staticmethod
    def remote_temp_path(remote_path: str, sha256: str) -> str:
        return f"{remote_path}.colab-upload-{sha256[:12]}.part"

    @staticmethod
    def local_temp_path(
        local_path: str | os.PathLike[str],
    ) -> Path:
        return Path(f"{Path(local_path)}.colab-download.part")

    def upload(
        self,
        local_path: str | os.PathLike[str],
        remote_path: str,
        *,
        overwrite: bool = True,
        resume: bool = True,
    ) -> TransferResult:
        remote_retry_start = _remote_retry_count(self.remote)
        source = Path(local_path)
        size, digest = _file_size_and_sha256(source)
        temp_path = self.remote_temp_path(
            remote_path,
            digest,
        )
        offset = self._verified_upload_offset(
            source,
            temp_path,
            size=size,
            resume=resume,
        )
        resumed_from = offset
        local_retry_count = 0

        def total_retry_count() -> int:
            return local_retry_count + max(
                0,
                _remote_retry_count(self.remote) - remote_retry_start,
            )

        def record_retry() -> None:
            nonlocal local_retry_count
            local_retry_count += 1
            self._notify(
                "upload",
                offset,
                size,
                resumed_from=resumed_from,
                retry_count=total_retry_count(),
                sha256=digest,
                partial_path=temp_path,
                phase="retrying",
            )

        self._notify(
            "upload",
            offset,
            size,
            resumed_from=resumed_from,
            retry_count=total_retry_count(),
            sha256=digest,
            partial_path=temp_path,
            phase="prepared",
        )

        with source.open("rb") as stream:
            stream.seek(offset)
            while offset < size:
                data = stream.read(
                    min(
                        self.chunk_size,
                        size - offset,
                    )
                )
                if not data:
                    break
                marker = 1 if offset == 0 else 2
                self._upload_chunk_with_recovery(
                    temp_path,
                    data,
                    chunk=marker,
                    before=offset,
                    on_retry=record_retry,
                )
                offset += len(data)
                self._notify(
                    "upload",
                    offset,
                    size,
                    resumed_from=resumed_from,
                    retry_count=total_retry_count(),
                    sha256=digest,
                    partial_path=temp_path,
                )

        self._upload_final_marker(
            temp_path,
            expected_size=size,
            on_retry=record_retry,
        )
        self.remote.finalize_upload(
            temp_path,
            remote_path,
            size=size,
            sha256=digest,
            overwrite=overwrite,
        )
        retries = total_retry_count()
        self._notify(
            "upload",
            size,
            size,
            resumed_from=resumed_from,
            retry_count=retries,
            sha256=digest,
            partial_path=temp_path,
            phase="completed",
        )
        return TransferResult(
            remote_path,
            size,
            digest,
            resumed_from,
            retries,
        )

    def download(
        self,
        remote_path: str,
        local_path: str | os.PathLike[str],
        *,
        resume: bool = True,
    ) -> TransferResult:
        remote_retry_start = _remote_retry_count(self.remote)

        def total_retry_count() -> int:
            return max(
                0,
                _remote_retry_count(self.remote) - remote_retry_start,
            )

        info = self.remote.stat_file(remote_path)
        if not info.get("exists"):
            raise FileNotFoundError(remote_path)
        size = int(info["size"])
        digest = str(info["sha256"])
        target = Path(local_path)
        part = self.local_temp_path(target)
        part.parent.mkdir(parents=True, exist_ok=True)
        offset = self._verified_download_offset(
            remote_path,
            part,
            size=size,
            resume=resume,
        )
        resumed_from = offset
        self._notify(
            "download",
            offset,
            size,
            resumed_from=resumed_from,
            retry_count=total_retry_count(),
            sha256=digest,
            partial_path=str(part),
            phase="prepared",
        )

        mode = "ab" if offset else "wb"
        with part.open(mode) as stream:
            while offset < size:
                data = self.remote.read_chunk(
                    remote_path,
                    offset=offset,
                    length=min(
                        self.chunk_size,
                        size - offset,
                    ),
                )
                if not data:
                    raise TransferIntegrityError(
                        f"Remote read returned no data at offset {offset} of {size}"
                    )
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                offset += len(data)
                self._notify(
                    "download",
                    offset,
                    size,
                    resumed_from=resumed_from,
                    retry_count=total_retry_count(),
                    sha256=digest,
                    partial_path=str(part),
                )

        actual_size, actual_digest = _file_size_and_sha256(part)
        if actual_size != size or actual_digest != digest:
            raise TransferIntegrityError(
                "Downloaded file failed verification: "
                f"expected {size} bytes/{digest}, "
                f"got {actual_size}/{actual_digest}"
            )
        os.replace(part, target)
        retries = total_retry_count()
        self._notify(
            "download",
            size,
            size,
            resumed_from=resumed_from,
            retry_count=retries,
            sha256=digest,
            partial_path=str(part),
            phase="completed",
        )
        return TransferResult(
            str(target),
            size,
            digest,
            resumed_from,
            retries,
        )

    def _verified_upload_offset(
        self,
        source: Path,
        temp_path: str,
        *,
        size: int,
        resume: bool,
    ) -> int:
        info = self.remote.stat_file(temp_path, hash_limit=0)
        if not info.get("exists"):
            return 0
        if not resume:
            self.remote.remove_file(temp_path)
            return 0
        remote_size = int(info["size"])
        if remote_size > size:
            self.remote.remove_file(temp_path)
            return 0
        remote_prefix = self.remote.stat_file(
            temp_path,
            hash_limit=remote_size,
        )
        local_prefix = _sha256_prefix(source, remote_size)
        if remote_prefix.get("sha256") != local_prefix:
            self.remote.remove_file(temp_path)
            return 0
        return remote_size

    def _verified_download_offset(
        self,
        remote_path: str,
        part: Path,
        *,
        size: int,
        resume: bool,
    ) -> int:
        if not part.exists():
            return 0
        if not resume:
            part.unlink()
            return 0
        local_size, local_digest = _file_size_and_sha256(part)
        if local_size > size:
            part.unlink()
            return 0
        remote_prefix = self.remote.stat_file(
            remote_path,
            hash_limit=local_size,
        )
        if remote_prefix.get("sha256") != local_digest:
            part.unlink()
            return 0
        return local_size

    def _upload_chunk_with_recovery(
        self,
        path: str,
        data: bytes,
        *,
        chunk: int,
        before: int,
        on_retry: Callable[[], None],
    ) -> None:
        expected = before + len(data)
        last_error: requests.RequestException | None = None
        for attempt in range(self.max_attempts):
            try:
                self.contents.upload_chunk(
                    path,
                    data,
                    chunk=chunk,
                )
                return
            except requests.RequestException as exc:
                last_error = exc
                info = self.remote.stat_file(
                    path,
                    hash_limit=0,
                )
                remote_size = int(info.get("size", -1)) if info.get("exists") else 0
                if remote_size == expected:
                    return
                if remote_size != before:
                    raise TransferIntegrityError(
                        "Remote upload size changed unexpectedly "
                        "after a failed request: expected "
                        f"{before} or {expected}, got {remote_size}"
                    ) from exc
                if attempt + 1 < self.max_attempts:
                    on_retry()
        assert last_error is not None
        raise last_error

    def _upload_final_marker(
        self,
        path: str,
        *,
        expected_size: int,
        on_retry: Callable[[], None],
    ) -> None:
        last_error: requests.RequestException | None = None
        for attempt in range(self.max_attempts):
            try:
                self.contents.upload_chunk(
                    path,
                    b"",
                    chunk=-1,
                )
                return
            except requests.RequestException as exc:
                last_error = exc
                info = self.remote.stat_file(
                    path,
                    hash_limit=0,
                )
                if not info.get("exists") or int(info["size"]) != expected_size:
                    raise TransferIntegrityError(
                        "Remote upload changed while committing the final chunk marker"
                    ) from exc
                if attempt + 1 < self.max_attempts:
                    on_retry()
        assert last_error is not None
        raise last_error

    def _notify(
        self,
        direction: str,
        completed: int,
        total: int,
        *,
        resumed_from: int,
        retry_count: int,
        sha256: str,
        partial_path: str,
        phase: str = "transferring",
    ) -> None:
        if self.progress is not None:
            self.progress(
                TransferProgress(
                    direction=direction,
                    completed=completed,
                    total=total,
                    resumed_from=resumed_from,
                    retry_count=retry_count,
                    sha256=sha256,
                    partial_path=partial_path,
                    phase=phase,
                )
            )


def _remote_retry_count(remote) -> int:
    value = getattr(remote, "retry_count", 0)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return 0


def _file_size_and_sha256(
    path: Path,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _sha256_prefix(path: Path, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()
