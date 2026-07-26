# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
from types import SimpleNamespace

from filelock import FileLock
import pytest

from colab_cli.jobs import JobTail
from colab_cli.monitor import (
    MonitorConfig,
    MonitorConfigurationError,
    MonitorConnection,
    run_monitor,
)
from colab_cli.observability.models import (
    DiskObservation,
    GpuObservation,
    MemoryObservation,
    ProbeObservation,
    RuntimeObservation,
)
from colab_cli.state import SessionState


class FakeJobClient:
    def __init__(
        self,
        statuses,
        stdout=b"",
        stderr=b"",
    ):
        self.statuses = list(statuses)
        self.stdout = stdout
        self.stderr = stderr
        self.tail_calls = []
        self.cancel_calls = []

    def status(
        self,
        _job_id,
        *,
        timeout,
    ):
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def tail(
        self,
        job_id,
        *,
        stream,
        offset,
        max_bytes,
        timeout,
    ):
        self.tail_calls.append(
            (
                stream,
                offset,
                max_bytes,
                timeout,
            )
        )
        data = self.stdout if stream == "stdout" else self.stderr
        chunk = data[offset : offset + max_bytes]
        next_offset = offset + len(chunk)
        return JobTail(
            job_id=job_id,
            stream=stream,
            offset=offset,
            next_offset=next_offset,
            size=len(data),
            eof=next_offset >= len(data),
            data=chunk,
        )

    def cancel(self, *args, **kwargs):
        self.cancel_calls.append((args, kwargs))


class FakeExecutor:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _session():
    return SessionState(
        name="xoftr",
        endpoint="endpoint-1",
        token="runtime-secret",
        url="https://runtime.example.test/",
        kernel_id="kernel-1",
        session_id="jupyter-session-1",
        accelerator="G4",
        variant="GPU",
    )


def _state():
    return SimpleNamespace(
        client=SimpleNamespace(
            list_assignments=lambda **_kwargs: [SimpleNamespace(endpoint="endpoint-1")]
        ),
        store=SimpleNamespace(),
        history=SimpleNamespace(),
    )


def _connection(client):
    executor = FakeExecutor()
    return (
        MonitorConnection(
            executor=executor,
            client=client,
        ),
        executor,
    )


def _probe(boot_id="boot-1"):
    return ProbeObservation(
        status="ok",
        observed_at=("2026-07-26T00:00:00Z"),
        duration_ms=1,
        timeout_seconds=20,
        gpu=GpuObservation(
            available=True,
            name="NVIDIA G4",
            sources=["runtime_api"],
        ),
        memory=MemoryObservation(
            total_bytes=10,
        ),
        disk=DiskObservation(
            total_bytes=20,
        ),
        runtime=RuntimeObservation(
            boot_id=boot_id,
        ),
    )


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_failed_job_preserves_raw_logs_and_exit(
    tmp_path,
):
    status = {
        "state": "failed",
        "returncode": 7,
        "error": "trainer failed",
        "runtime_id": "runtime-1",
        "stdout_size": 7,
        "stderr_size": 30,
    }
    client = FakeJobClient(
        [status],
        stdout=b"step=1\n",
        stderr=(b"CUDA out of memory\nTraceback...\n"),
    )
    connection, executor = _connection(client)

    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            interval=0.01,
        ),
        connection_factory=lambda *_args: connection,
        probe_fn=lambda *_args, **_kwargs: _probe(),
    )

    assert summary.exit_code == 7
    assert summary.error_code == ("REMOTE_JOB_FAILED")
    assert (tmp_path / "stdout.log").read_bytes() == b"step=1\n"
    assert b"CUDA out of memory" in (tmp_path / "stderr.log").read_bytes()
    assert _lines(tmp_path / "resources.jsonl")[0]["gpu"]["name"] == "NVIDIA G4"
    assert executor.closed == 1
    assert client.cancel_calls == []


def test_restart_resumes_without_duplicate_bytes(
    tmp_path,
):
    first_client = FakeJobClient(
        [
            {
                "state": "running",
                "runtime_id": "runtime-1",
                "stdout_size": 3,
                "stderr_size": 0,
            }
        ],
        stdout=b"abc",
    )
    first_connection, _ = _connection(first_client)
    run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            once=True,
            probe_every=0,
        ),
        connection_factory=lambda *_args: first_connection,
    )
    first_state = json.loads(
        (tmp_path / "monitor_state.json").read_text(encoding="utf-8")
    )
    assert first_state["status"] == "snapshot"

    second_client = FakeJobClient(
        [
            {
                "state": "succeeded",
                "returncode": 0,
                "runtime_id": "runtime-1",
                "stdout_size": 6,
                "stderr_size": 0,
            }
        ],
        stdout=b"abcdef",
    )
    second_connection, _ = _connection(second_client)
    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            probe_every=0,
        ),
        connection_factory=lambda *_args: second_connection,
    )

    assert summary.exit_code == 0
    assert (tmp_path / "stdout.log").read_bytes() == b"abcdef"
    state = json.loads((tmp_path / "monitor_state.json").read_text(encoding="utf-8"))
    assert state["monitor_runs"] == 2


def test_idle_logs_skip_tail_calls(
    tmp_path,
):
    client = FakeJobClient(
        [
            {
                "state": "succeeded",
                "returncode": 0,
                "runtime_id": "runtime-1",
                "stdout_size": 0,
                "stderr_size": 0,
            }
        ]
    )
    connection, _ = _connection(client)

    run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            probe_every=0,
        ),
        connection_factory=lambda *_args: connection,
    )

    assert client.tail_calls == []


def test_ctrl_c_never_cancels_remote(
    tmp_path,
):
    client = FakeJobClient(
        [
            {
                "state": "running",
                "runtime_id": "runtime-1",
                "stdout_size": 0,
                "stderr_size": 0,
            }
        ]
    )
    connection, _ = _connection(client)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            interval=0.01,
            probe_every=0,
        ),
        connection_factory=lambda *_args: connection,
        sleep_fn=interrupt,
    )

    assert summary.exit_code == 130
    assert summary.error_code == ("MONITOR_INTERRUPTED")
    assert client.cancel_calls == []


def test_probe_failure_is_missing_sample(
    tmp_path,
):
    client = FakeJobClient(
        [
            {
                "state": "succeeded",
                "returncode": 0,
                "runtime_id": "runtime-1",
                "stdout_size": 0,
                "stderr_size": 0,
            }
        ]
    )
    connection, _ = _connection(client)

    def failed_probe(*_args, **_kwargs):
        raise TimeoutError("probe timeout")

    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
        ),
        connection_factory=lambda *_args: connection,
        probe_fn=failed_probe,
    )

    assert summary.exit_code == 0
    assert _lines(tmp_path / "resources.jsonl")[0]["status"] == "unavailable"


def test_runtime_identity_change_fails_closed(
    tmp_path,
):
    client = FakeJobClient(
        [
            {
                "state": "running",
                "runtime_id": "runtime-1",
                "stdout_size": 0,
                "stderr_size": 0,
            },
            {
                "state": "running",
                "runtime_id": "runtime-2",
                "stdout_size": 0,
                "stderr_size": 0,
            },
        ]
    )
    connection, _ = _connection(client)
    ticks = iter(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]
    )

    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            interval=0.01,
            probe_every=0,
        ),
        connection_factory=lambda *_args: connection,
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: next(ticks),
    )

    assert summary.error_code == ("RUNTIME_IDENTITY_CHANGED")


def test_local_evidence_beyond_remote_fails(
    tmp_path,
):
    first_client = FakeJobClient(
        [
            {
                "state": "running",
                "runtime_id": "runtime-1",
                "stdout_size": 6,
                "stderr_size": 0,
            }
        ],
        stdout=b"abcdef",
    )
    first_connection, _ = _connection(first_client)
    run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            once=True,
            probe_every=0,
        ),
        connection_factory=lambda *_args: first_connection,
    )

    second_client = FakeJobClient(
        [
            {
                "state": "running",
                "runtime_id": "runtime-1",
                "stdout_size": 3,
                "stderr_size": 0,
            }
        ],
        stdout=b"abc",
    )
    second_connection, _ = _connection(second_client)
    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            once=True,
            probe_every=0,
        ),
        connection_factory=lambda *_args: second_connection,
    )

    assert summary.error_code == ("LOCAL_EVIDENCE_DIVERGED")
    assert (tmp_path / "stdout.log").read_bytes() == b"abcdef"


def test_orphan_files_without_state_are_refused(
    tmp_path,
):
    (tmp_path / "stdout.log").write_bytes(b"x")
    client = FakeJobClient([{"state": "running"}])
    connection, _ = _connection(client)

    with pytest.raises(MonitorConfigurationError):
        run_monitor(
            _state(),
            job_id="train",
            session=_session(),
            config=MonitorConfig(
                output_dir=tmp_path,
                once=True,
            ),
            connection_factory=(lambda *_args: connection),
        )


def test_second_monitor_same_output_fails_fast(
    tmp_path,
):
    lock = FileLock(
        tmp_path / ".monitor.lock",
        timeout=0,
        blocking=False,
        is_singleton=False,
    )
    lock.acquire(blocking=False)
    try:
        with pytest.raises(
            MonitorConfigurationError,
            match="another monitor",
        ):
            run_monitor(
                _state(),
                job_id="train",
                session=_session(),
                config=MonitorConfig(
                    output_dir=tmp_path,
                    once=True,
                ),
            )
    finally:
        lock.release(force=True)


def test_configuration_precedes_connection(
    tmp_path,
):
    with pytest.raises(MonitorConfigurationError):
        run_monitor(
            _state(),
            job_id="train",
            session=_session(),
            config=MonitorConfig(
                output_dir=tmp_path,
                interval=0,
            ),
            connection_factory=(
                lambda *_args: (_ for _ in ()).throw(AssertionError("must not connect"))
            ),
        )


def test_terminal_monitor_drains_more_than_one_chunk(
    tmp_path,
):
    payload = b"abcdefghij"
    client = FakeJobClient(
        [
            {
                "state": "succeeded",
                "returncode": 0,
                "runtime_id": "runtime-1",
                "stdout_size": len(payload),
                "stderr_size": 0,
            }
        ],
        stdout=payload,
    )
    connection, _ = _connection(client)

    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            probe_every=0,
            max_bytes=3,
        ),
        connection_factory=lambda *_args: connection,
    )

    assert summary.exit_code == 0
    assert (tmp_path / "stdout.log").read_bytes() == payload
    assert len(client.tail_calls) == 4


def test_terminal_backlog_larger_than_one_poll_is_completed(
    tmp_path,
    monkeypatch,
):
    import colab_cli.monitor as monitor_module

    monkeypatch.setattr(
        monitor_module,
        "_MAX_TAIL_CHUNKS_PER_POLL",
        2,
    )
    payload = b"abcdefghij"
    client = FakeJobClient(
        [
            {
                "state": "succeeded",
                "returncode": 0,
                "runtime_id": "runtime-1",
                "stdout_size": len(payload),
                "stderr_size": 0,
            }
        ],
        stdout=payload,
    )
    connection, _ = _connection(client)

    summary = run_monitor(
        _state(),
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            interval=0.01,
            probe_every=0,
            max_bytes=3,
        ),
        connection_factory=lambda *_args: connection,
        sleep_fn=lambda _seconds: None,
    )

    assert summary.exit_code == 0
    assert (tmp_path / "stdout.log").read_bytes() == payload
    events = _lines(tmp_path / "events.jsonl")
    assert any(
        event["event_type"] == "remote_terminal_logs_pending" for event in events
    )


def test_assignment_disappearance_is_persisted_as_failure(
    tmp_path,
):
    class FailingStatusClient(FakeJobClient):
        def status(self, _job_id, *, timeout):
            raise ConnectionError("runtime unreachable")

    client = FailingStatusClient([])
    connection, _ = _connection(client)
    state = _state()
    state.client = SimpleNamespace(list_assignments=lambda **_kwargs: [])

    summary = run_monitor(
        state,
        job_id="train",
        session=_session(),
        config=MonitorConfig(
            output_dir=tmp_path,
            probe_every=0,
        ),
        connection_factory=lambda *_args: connection,
    )

    assert summary.exit_code == 1
    assert summary.error_code == ("ASSIGNMENT_DISAPPEARED")
    assert summary.status == "failed"
    assert (
        json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[
            "error_code"
        ]
        == "ASSIGNMENT_DISAPPEARED"
    )
