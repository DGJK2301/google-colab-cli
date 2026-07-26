# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

import json
import multiprocessing
import os

import pytest

from colab_cli.transfer_lease import (
    TransferLease,
    TransferLeaseBusy,
    TransferLeaseCorrupt,
    canonical_local_path,
    normalize_remote_path,
    process_identity_state,
    process_start_token,
)


def _hold_lease(
    root,
    source,
    acquired,
    release,
):
    lease = TransferLease.for_upload(
        endpoint="runtime-a",
        local_path=source,
        remote_path="content/model.bin",
        root=root,
    )
    lease.acquire()
    acquired.set()
    release.wait(timeout=15)
    lease.release()


def _upload(
    tmp_path,
    *,
    remote="content/model.bin",
    endpoint="runtime-a",
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    return TransferLease.for_upload(
        endpoint=endpoint,
        local_path=source,
        remote_path=remote,
        root=tmp_path / "leases",
    )


def test_same_target_second_writer_fails_immediately(
    tmp_path,
):
    first = _upload(tmp_path)
    second = _upload(tmp_path)
    first.acquire()
    try:
        with pytest.raises(TransferLeaseBusy):
            second.acquire()
    finally:
        first.release()


def test_cross_process_writer_exclusion(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    root = tmp_path / "leases"
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(
        target=_hold_lease,
        args=(
            str(root),
            str(source),
            acquired,
            release,
        ),
    )
    holder.start()
    try:
        assert acquired.wait(timeout=10)
        contender = TransferLease.for_upload(
            endpoint="runtime-a",
            local_path=source,
            remote_path="content/model.bin",
            root=root,
        )
        with pytest.raises(TransferLeaseBusy):
            contender.acquire()
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


def test_different_targets_can_transfer_concurrently(
    tmp_path,
):
    first = _upload(
        tmp_path,
        remote="content/a.bin",
    )
    second = _upload(
        tmp_path,
        remote="content/b.bin",
    )
    first.acquire()
    second.acquire()
    second.release()
    first.release()


def test_upload_key_includes_endpoint_and_target(
    tmp_path,
):
    keys = {
        _upload(
            tmp_path,
            endpoint=endpoint,
            remote=remote,
        ).lock_key
        for endpoint, remote in (
            ("runtime-a", "content/model.bin"),
            ("runtime-b", "content/model.bin"),
            ("runtime-a", "content/other.bin"),
        )
    }
    assert len(keys) == 3


def test_download_key_is_canonical_local_target(
    tmp_path,
):
    first = TransferLease.for_download(
        endpoint="runtime-a",
        remote_path="content/a.bin",
        local_path=(tmp_path / "nested" / ".." / "model.bin"),
        root=tmp_path / "leases",
    )
    second = TransferLease.for_download(
        endpoint="runtime-b",
        remote_path="content/b.bin",
        local_path=tmp_path / "model.bin",
        root=tmp_path / "leases",
    )
    assert first.lock_key == second.lock_key
    assert first.target_path == canonical_local_path(tmp_path / "model.bin")


def test_dead_owner_metadata_is_reclaimed(
    tmp_path,
    monkeypatch,
):
    lease = _upload(tmp_path)
    lease.root.mkdir(parents=True)
    lease.metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lease_id": "old-lease",
                "state": "active",
                "pid": 987654321,
                "process_start_token": "proc:dead",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "colab_cli.transfer_lease.process_identity_state",
        lambda _pid, _token: "dead",
    )

    lease.acquire()
    try:
        assert lease.stale_reclaimed is True
        assert lease.stale_reclaimed_from == "old-lease"
    finally:
        lease.release()


def test_reused_pid_token_is_confirmed_dead(
    monkeypatch,
):
    monkeypatch.setattr(
        "colab_cli.transfer_lease._process_existence_state",
        lambda _pid: "alive",
    )
    monkeypatch.setattr(
        "colab_cli.transfer_lease.process_start_token",
        lambda _pid: "proc:new-owner",
    )

    assert (
        process_identity_state(
            123,
            "proc:old-owner",
        )
        == "dead"
    )


def test_unverifiable_process_existence_is_unknown(
    monkeypatch,
):
    monkeypatch.setattr(
        "colab_cli.transfer_lease._process_existence_state",
        lambda _pid: "unknown",
    )

    assert (
        process_identity_state(
            123,
            "proc:recorded",
        )
        == "unknown"
    )


def test_windows_access_denied_is_not_treated_as_dead():
    from colab_cli.transfer_lease import (
        _windows_process_existence_state,
    )

    class AccessDeniedKernel32:
        def OpenProcess(self, *_args):
            return 0

        def GetLastError(self):
            return 5

    assert (
        _windows_process_existence_state(
            123,
            kernel32=AccessDeniedKernel32(),
        )
        == "unknown"
    )


def test_windows_invalid_pid_is_confirmed_dead():
    from colab_cli.transfer_lease import (
        _windows_process_existence_state,
    )

    class InvalidPidKernel32:
        def OpenProcess(self, *_args):
            return 0

        def GetLastError(self):
            return 87

    assert (
        _windows_process_existence_state(
            123,
            kernel32=InvalidPidKernel32(),
        )
        == "dead"
    )


def test_windows_process_probe_preserves_pointer_width():
    from colab_cli.transfer_lease import (
        _windows_process_existence_state,
    )

    wide_handle = 0x1_0000_0001

    class WideHandleKernel32:
        def __init__(self):
            self.closed = None

        def OpenProcess(self, *_args):
            return wide_handle

        def GetLastError(self):
            return 0

        def GetExitCodeProcess(self, handle, exit_code):
            assert handle == wide_handle
            exit_code._obj.value = 259
            return 1

        def CloseHandle(self, handle):
            self.closed = handle
            return 1

    kernel32 = WideHandleKernel32()

    assert (
        _windows_process_existence_state(
            123,
            kernel32=kernel32,
        )
        == "alive"
    )
    assert kernel32.closed == wide_handle


def test_live_owner_metadata_fails_closed(
    tmp_path,
):
    lease = _upload(tmp_path)
    lease.root.mkdir(parents=True)
    token = process_start_token(os.getpid())
    if token is None:
        pytest.skip("process start identity unavailable")
    lease.metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lease_id": "live-owner",
                "state": "active",
                "pid": os.getpid(),
                "process_start_token": token,
                "heartbeat_at": "now",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TransferLeaseBusy):
        lease.acquire()


def test_unverifiable_live_pid_is_not_reclaimed(
    tmp_path,
):
    lease = _upload(tmp_path)
    lease.root.mkdir(parents=True)
    lease.metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "lease_id": "unknown-owner",
                "state": "active",
                "pid": os.getpid(),
                "process_start_token": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TransferLeaseCorrupt,
        match="cannot be verified",
    ):
        lease.acquire()


def test_corrupt_or_unknown_metadata_is_not_recycled(
    tmp_path,
):
    corrupt = _upload(
        tmp_path,
        remote="content/corrupt.bin",
    )
    corrupt.root.mkdir(parents=True)
    corrupt.metadata_path.write_text(
        "not json",
        encoding="utf-8",
    )
    with pytest.raises(TransferLeaseCorrupt):
        corrupt.acquire()

    unknown = _upload(
        tmp_path,
        remote="content/unknown.bin",
    )
    unknown.metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "mystery",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TransferLeaseCorrupt):
        unknown.acquire()


def test_heartbeat_records_integrity_and_retry_evidence(
    tmp_path,
):
    lease = _upload(tmp_path)
    lease.acquire()
    lease.heartbeat(
        completed_bytes=1024,
        total_bytes=4096,
        resumed_from=512,
        retry_count=2,
        sha256="abc123",
        partial_path="content/model.part",
        force=True,
    )
    metadata = json.loads(lease.metadata_path.read_text(encoding="utf-8"))

    assert metadata["source_size"] == 4096
    assert metadata["source_sha256"] == "abc123"
    assert metadata["completed_bytes"] == 1024
    assert metadata["resumed_from"] == 512
    assert metadata["retry_count"] == 2
    assert metadata["heartbeat_at"]
    assert metadata["pid"] == os.getpid()
    assert metadata["process_start_token"]

    lease.release()
    assert not lease.metadata_path.exists()


def test_remote_path_normalization_is_platform_independent():
    assert normalize_remote_path(r"\content\a\..\b.bin") == "content/b.bin"
    with pytest.raises(ValueError):
        normalize_remote_path("")


def test_release_cleanup_errors_do_not_raise(
    tmp_path,
    monkeypatch,
):
    lease = _upload(tmp_path)
    lease.acquire()
    monkeypatch.setattr(
        lease,
        "_write_metadata",
        lambda: (_ for _ in ()).throw(OSError("metadata failed")),
    )

    lease.release()

    assert any("metadata failed" in item for item in lease.cleanup_errors)
