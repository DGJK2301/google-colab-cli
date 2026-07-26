# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Cross-process, fail-closed single-writer leases for transfer targets."""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import posixpath
import subprocess
import time
from typing import Any, Literal
import uuid

from filelock import FileLock
from filelock import Timeout as FileLockTimeout


DEFAULT_TRANSFER_LEASE_ROOT = "~/.config/colab-cli/transfer-leases"
TRANSFER_LEASE_ROOT_ENV = "COLAB_CLI_TRANSFER_LEASE_DIR"
_METADATA_SCHEMA = 1


class TransferLeaseError(RuntimeError):
    pass


class TransferLeaseBusy(TransferLeaseError):
    def __init__(
        self,
        message: str,
        *,
        owner: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.owner = owner or {}


class TransferLeaseCorrupt(TransferLeaseError):
    pass


class TransferLease:
    """Hold one OS lock across the complete transfer lifecycle."""

    def __init__(
        self,
        *,
        direction: Literal["upload", "download"],
        key_material: dict[str, str],
        source_path: str,
        target_path: str,
        endpoint: str | None,
        partial_path: str | None = None,
        root: str | os.PathLike[str] | None = None,
        heartbeat_interval: float = 1.0,
    ) -> None:
        self.direction = direction
        self.source_path = source_path
        self.target_path = target_path
        self.endpoint = endpoint
        self.partial_path = partial_path
        self.heartbeat_interval = heartbeat_interval
        self.lease_id = uuid.uuid4().hex
        self.cleanup_errors: list[str] = []
        self.stale_reclaimed = False
        self.stale_reclaimed_from: str | None = None
        self._acquired = False
        self._last_write_monotonic = 0.0

        encoded = json.dumps(
            key_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.lock_key = hashlib.sha256(encoded).hexdigest()
        prefix = f"{direction}-{self.lock_key}"
        self.root = _lease_root(root)
        self.lock_path = self.root / f"{prefix}.lock"
        self.metadata_path = self.root / f"{prefix}.json"
        self._lock = FileLock(
            self.lock_path,
            timeout=0,
            blocking=False,
            is_singleton=False,
        )
        self._metadata: dict[str, Any] = {}

    @classmethod
    def for_upload(
        cls,
        *,
        endpoint: str,
        local_path: str | os.PathLike[str],
        remote_path: str,
        root: str | os.PathLike[str] | None = None,
    ) -> "TransferLease":
        source = canonical_local_path(local_path)
        target = normalize_remote_path(remote_path)
        return cls(
            direction="upload",
            key_material={
                "endpoint": endpoint,
                "remote_path": target,
            },
            source_path=source,
            target_path=target,
            endpoint=endpoint,
            root=root,
        )

    @classmethod
    def for_download(
        cls,
        *,
        endpoint: str,
        remote_path: str,
        local_path: str | os.PathLike[str],
        root: str | os.PathLike[str] | None = None,
    ) -> "TransferLease":
        source = normalize_remote_path(remote_path)
        target = canonical_local_path(local_path)
        return cls(
            direction="download",
            key_material={"local_path": local_lock_identity(target)},
            source_path=source,
            target_path=target,
            endpoint=endpoint,
            partial_path=f"{target}.colab-download.part",
            root=root,
        )

    def acquire(self) -> "TransferLease":
        self.root.mkdir(parents=True, exist_ok=True)
        _restrict_directory(self.root)
        try:
            self._lock.acquire(blocking=False)
        except FileLockTimeout as exc:
            owner = _read_metadata_best_effort(self.metadata_path)
            raise TransferLeaseBusy(
                _busy_message(self.target_path, owner),
                owner=owner,
            ) from exc

        try:
            previous = self._read_existing_metadata()
            if previous and previous.get("state") == "active":
                identity = process_identity_state(
                    previous.get("pid"),
                    previous.get("process_start_token"),
                )
                if identity == "alive":
                    raise TransferLeaseBusy(
                        _busy_message(
                            self.target_path,
                            previous,
                        ),
                        owner=previous,
                    )
                if identity == "unknown":
                    raise TransferLeaseCorrupt(
                        "Existing transfer owner identity "
                        "cannot be verified; refusing to recycle "
                        f"{self.metadata_path}"
                    )
                self.stale_reclaimed = True
                self.stale_reclaimed_from = _string_or_none(previous.get("lease_id"))
            elif previous and previous.get("state") != "released":
                raise TransferLeaseCorrupt(
                    "Existing transfer lease metadata has "
                    "an unknown state; refusing to recycle "
                    f"{self.metadata_path}"
                )

            now = _utcnow()
            self._metadata = {
                "schema_version": _METADATA_SCHEMA,
                "lease_id": self.lease_id,
                "lock_key": self.lock_key,
                "state": "active",
                "direction": self.direction,
                "pid": os.getpid(),
                "process_start_token": process_start_token(os.getpid()),
                "created_at": now,
                "heartbeat_at": now,
                "finished_at": None,
                "source_path": self.source_path,
                "target_path": self.target_path,
                "partial_path": self.partial_path,
                "endpoint": self.endpoint,
                "source_size": None,
                "source_sha256": None,
                "completed_bytes": 0,
                "total_bytes": None,
                "resumed_from": 0,
                "retry_count": 0,
                "stale_reclaimed": self.stale_reclaimed,
                "stale_reclaimed_from": (self.stale_reclaimed_from),
            }
            self._write_metadata()
            self._acquired = True
            self._last_write_monotonic = time.monotonic()
            return self
        except BaseException:
            self._release_lock_best_effort()
            raise

    def heartbeat(
        self,
        *,
        completed_bytes: int | None = None,
        total_bytes: int | None = None,
        resumed_from: int | None = None,
        retry_count: int | None = None,
        sha256: str | None = None,
        partial_path: str | None = None,
        force: bool = False,
    ) -> None:
        if not self._acquired:
            return

        updates = {
            "completed_bytes": completed_bytes,
            "total_bytes": total_bytes,
            "resumed_from": resumed_from,
            "retry_count": retry_count,
            "source_sha256": sha256,
            "partial_path": partial_path,
        }
        for key, value in updates.items():
            if value is not None:
                self._metadata[key] = value
        if total_bytes is not None:
            self._metadata["source_size"] = total_bytes
        if partial_path is not None:
            self.partial_path = partial_path

        now = time.monotonic()
        if not force and now - self._last_write_monotonic < self.heartbeat_interval:
            return
        self._metadata["heartbeat_at"] = _utcnow()
        try:
            self._write_metadata()
            self._last_write_monotonic = now
        except OSError as exc:
            self.cleanup_errors.append(f"lease heartbeat update failed: {exc}")

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self._metadata["state"] = "released"
            self._metadata["heartbeat_at"] = _utcnow()
            self._metadata["finished_at"] = self._metadata["heartbeat_at"]
            try:
                self._write_metadata()
            except OSError as exc:
                self.cleanup_errors.append(f"lease release metadata failed: {exc}")
            try:
                self.metadata_path.unlink(missing_ok=True)
            except OSError as exc:
                self.cleanup_errors.append(f"lease metadata cleanup failed: {exc}")
        finally:
            self._release_lock_best_effort()
            self._acquired = False

    def __enter__(self) -> "TransferLease":
        return self.acquire()

    def __exit__(
        self,
        _exc_type,
        _exc,
        _traceback,
    ) -> None:
        self.release()

    def _read_existing_metadata(
        self,
    ) -> dict[str, Any] | None:
        if not self.metadata_path.exists():
            return None
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise TransferLeaseCorrupt(
                "Existing transfer lease metadata "
                "cannot be parsed; refusing to recycle "
                f"{self.metadata_path}: {exc}"
            ) from exc
        if not isinstance(data, dict) or data.get("schema_version") != _METADATA_SCHEMA:
            raise TransferLeaseCorrupt(
                "Existing transfer lease metadata has "
                "an unsupported schema; refusing to recycle "
                f"{self.metadata_path}"
            )
        return data

    def _write_metadata(self) -> None:
        _atomic_json(
            self.metadata_path,
            self._metadata,
        )

    def _release_lock_best_effort(self) -> None:
        try:
            self._lock.release(force=True)
        except Exception as exc:
            self.cleanup_errors.append(f"lease lock release failed: {exc}")


def normalize_remote_path(path: str) -> str:
    value = str(path).strip().replace("\\", "/")
    if not value:
        raise ValueError("remote path must not be empty")
    normalized = posixpath.normpath("/" + value.lstrip("/")).lstrip("/")
    if not normalized or normalized == ".":
        raise ValueError("remote path must identify a file")
    return normalized


def canonical_local_path(
    path: str | os.PathLike[str],
) -> str:
    return os.path.normpath(str(Path(path).expanduser().resolve(strict=False)))


def local_lock_identity(
    path: str | os.PathLike[str],
) -> str:
    """Return a platform-normalized identity without changing display paths."""

    return os.path.normcase(canonical_local_path(path))


def process_start_token(pid: int) -> str | None:
    token = _proc_start_token(pid)
    if token is not None:
        return f"proc:{token}"
    if os.name == "nt":
        token = _windows_start_token(pid)
        return f"win:{token}" if token is not None else None
    token = _ps_start_token(pid)
    return f"ps:{token}" if token is not None else None


def process_identity_state(
    pid: Any,
    expected_start_token: Any,
) -> Literal["alive", "dead", "unknown"]:
    try:
        numeric_pid = int(pid)
    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return "unknown"
    if numeric_pid <= 0:
        return "unknown"

    existence = _process_existence_state(numeric_pid)
    if existence != "alive":
        return existence
    if expected_start_token is None:
        return "unknown"

    actual = process_start_token(numeric_pid)
    if actual is None:
        return "unknown"
    return "alive" if actual == str(expected_start_token) else "dead"


def process_identity_alive(
    pid: Any,
    expected_start_token: Any,
) -> bool:
    return (
        process_identity_state(
            pid,
            expected_start_token,
        )
        == "alive"
    )


def _lease_root(
    root: str | os.PathLike[str] | None,
) -> Path:
    if root is not None:
        return Path(root).expanduser()
    configured = os.environ.get(TRANSFER_LEASE_ROOT_ENV)
    return Path(configured or DEFAULT_TRANSFER_LEASE_ROOT).expanduser()


def transfer_lease_root(
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the configured transfer-lease metadata root."""
    return _lease_root(root)


def process_existence_state(
    pid: int,
) -> Literal["alive", "dead", "unknown"]:
    """Public fail-closed process existence probe."""
    return _process_existence_state(pid)


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with temp.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _restrict_directory(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _read_metadata_best_effort(
    path: Path,
) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return None
    return data if isinstance(data, dict) else None


def _busy_message(
    target: str,
    owner: dict[str, Any] | None,
) -> str:
    if not owner:
        return f"Transfer target is already locked: {target}"
    return (
        "Transfer target is already locked: "
        f"{target}; owner_pid={owner.get('pid')}, "
        f"lease_id={owner.get('lease_id')}, "
        f"heartbeat_at={owner.get('heartbeat_at')}"
    )


def _process_existence_state(
    pid: int,
) -> Literal["alive", "dead", "unknown"]:
    if pid <= 0:
        return "dead"
    if os.name == "nt":
        return _windows_process_existence_state(pid)

    try:
        os.kill(pid, 0)
        return "alive"
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "alive"
    except OSError:
        return "unknown"


def _windows_process_existence_state(
    pid: int,
    *,
    kernel32=None,
) -> Literal["alive", "dead", "unknown"]:
    from ctypes import wintypes

    api = kernel32 if kernel32 is not None else ctypes.windll.kernel32
    open_process = _winapi_function(
        api.OpenProcess,
        argtypes=[wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
        restype=wintypes.HANDLE,
    )
    get_last_error = _winapi_function(
        api.GetLastError,
        argtypes=[],
        restype=wintypes.DWORD,
    )
    handle = open_process(
        0x1000,
        False,
        pid,
    )
    if not handle:
        error = int(get_last_error())
        if error == 87:
            return "dead"
        return "unknown"

    get_exit_code = _winapi_function(
        api.GetExitCodeProcess,
        argtypes=[wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)],
        restype=wintypes.BOOL,
    )
    close_handle = _winapi_function(
        api.CloseHandle,
        argtypes=[wintypes.HANDLE],
        restype=wintypes.BOOL,
    )

    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code(
            handle,
            ctypes.byref(exit_code),
        ):
            return "unknown"
        return "alive" if exit_code.value == 259 else "dead"
    finally:
        close_handle(handle)


def _pid_exists(pid: int) -> bool:
    """Compatibility helper: unknown existence is not treated as dead."""
    return _process_existence_state(pid) != "dead"


def _proc_start_token(pid: int) -> str | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    close = value.rfind(")")
    fields = value[close + 2 :].split() if close >= 0 else []
    return fields[19] if len(fields) > 19 else None


def _windows_start_token(pid: int) -> str | None:
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    open_process = _winapi_function(
        kernel32.OpenProcess,
        argtypes=[wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
        restype=wintypes.HANDLE,
    )
    get_process_times = _winapi_function(
        kernel32.GetProcessTimes,
        argtypes=[
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ],
        restype=wintypes.BOOL,
    )
    close_handle = _winapi_function(
        kernel32.CloseHandle,
        argtypes=[wintypes.HANDLE],
        restype=wintypes.BOOL,
    )
    handle = open_process(
        0x1000,
        False,
        pid,
    )
    if not handle:
        return None
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel_time = wintypes.FILETIME()
    user_time = wintypes.FILETIME()
    try:
        ok = get_process_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not ok:
            return None
        return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
    finally:
        close_handle(handle)


def _winapi_function(function, *, argtypes, restype):
    """Declare pointer-width-safe ctypes signatures for real Win32 calls."""

    try:
        function.argtypes = argtypes
        function.restype = restype
    except (AttributeError, TypeError):
        # Lightweight test doubles expose bound Python methods rather than
        # ctypes function pointers.
        pass
    return function


def _ps_start_token(pid: int) -> str | None:
    try:
        completed = subprocess.run(
            [
                "ps",
                "-o",
                "lstart=",
                "-p",
                str(pid),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return None
    if completed.returncode != 0:
        return None
    value = " ".join(completed.stdout.split())
    return value or None


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)
