# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.observability.models import (
    DiskObservation,
    GpuObservation,
    MemoryObservation,
    ProbeObservation,
    RuntimeObservation,
)
from colab_cli.observability.probes import (
    ExistingKernelJsonExecutor,
    _RawDisk,
    _RawFilesystem,
    _RawGpu,
    _RawMemory,
    _RawResources,
    _fetch_runtime_resources,
    _merge_supplemental,
    probe_session,
)
from colab_cli.state import SessionState


runner = CliRunner()


def _session(
    *,
    name: str = "s1",
    endpoint: str = "endpoint-1",
    accelerator: str = "T4",
    variant: str = "GPU",
    kernel_id: str | None = "kernel-1",
) -> SessionState:
    return SessionState(
        name=name,
        endpoint=endpoint,
        accelerator=accelerator,
        variant=variant,
        token="secret-runtime-token",
        url="https://runtime.example.test/",
        kernel_id=kernel_id,
        session_id="jupyter-session-id",
    )


def _assignment(
    *,
    endpoint: str = "endpoint-1",
    accelerator: str = "L4",
    variant: str = "GPU",
    shape: str = "HIGH_RAM",
):
    return SimpleNamespace(
        endpoint=endpoint,
        accelerator=SimpleNamespace(value=accelerator),
        variant=SimpleNamespace(name=variant),
        machine_shape=SimpleNamespace(name=shape),
    )


def _json_stdout(result):
    assert result.stdout.strip().startswith("{"), result.output
    return json.loads(result.stdout)


def _resources(*, gpu: _RawGpu | None = None) -> _RawResources:
    return _RawResources(
        memory=_RawMemory(totalBytes=32_000, freeBytes=12_000),
        disks=[
            _RawDisk(
                filesystem=_RawFilesystem(
                    label="/content",
                    totalBytes=100_000,
                    usedBytes=40_000,
                )
            )
        ],
        gpus=[] if gpu is None else [gpu],
    )


def test_sessions_json_is_read_only_clean_and_redacted(
    mock_common_state, mocker, capsys
):
    session = _session()
    mock_common_state.store.list.return_value = {"s1": session}

    def noisy_assignments():
        print("dependency diagnostic that must not enter JSON")
        return [_assignment()]

    mock_common_state.client.list_assignments.side_effect = noisy_assignments
    update = mocker.patch("colab_cli.auto_update.run_background_check")

    result = runner.invoke(app, ["sessions", "--json"])

    assert result.exit_code == 0, result.output
    payload = _json_stdout(result)
    assert payload["schema"] == "colab.sessions.v1"
    assert payload["status"] == "ok"
    observed = payload["sessions"][0]
    assert observed["local"]["requested_accelerator"] == "T4"
    assert observed["assignment"]["accelerator"] == "L4"
    assert observed["assignment"]["machine_shape"] == "HIGH_RAM"
    assert observed["warnings"][0]["code"] == (
        "REQUESTED_ASSIGNED_ACCELERATOR_MISMATCH"
    )
    assert observed["local"]["keep_alive"]["status"] == "not_recorded"
    assert observed["local"]["keep_alive"]["last_heartbeat_at"] is None
    assert observed["local"]["keep_alive"]["unavailable_reason"]
    assert "secret-runtime-token" not in result.stdout
    assert "dependency diagnostic" not in result.stdout
    mock_common_state.sync_sessions.assert_not_called()
    mock_common_state.history.log_event.assert_not_called()
    update.assert_not_called()
    capsys.readouterr()


def test_sessions_json_includes_server_orphan_with_null_name(mock_common_state):
    mock_common_state.store.list.return_value = {}
    mock_common_state.client.list_assignments.return_value = [
        _assignment(endpoint="server-only", accelerator="NONE", variant="DEFAULT")
    ]

    result = runner.invoke(app, ["sessions", "--json"])

    assert result.exit_code == 0
    payload = _json_stdout(result)
    assert payload["sessions"][0]["name"] is None
    assert payload["sessions"][0]["lifecycle"] == "orphan_server"


def test_sessions_json_redacts_home_from_last_execution(
    mock_common_state, monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = _session()
    session.last_execution = (
        str(tmp_path / "work" / "train.py"),
        "cell-1",
        "2026-07-22T00:00:00Z",
    )
    mock_common_state.store.list.return_value = {"s1": session}
    mock_common_state.client.list_assignments.return_value = [_assignment()]

    result = runner.invoke(app, ["sessions", "--json"])

    assert result.exit_code == 0
    file_name = _json_stdout(result)["sessions"][0]["local"]["last_execution"]["file"]
    assert str(tmp_path) not in file_name
    assert file_name.startswith("~")


def test_human_sessions_keeps_existing_update_behavior(mock_common_state, mocker):
    mock_common_state.sync_sessions.return_value = ({}, [])
    update = mocker.patch("colab_cli.auto_update.run_background_check")

    result = runner.invoke(app, ["sessions"])

    assert result.exit_code == 0
    update.assert_called_once_with()
    mock_common_state.sync_sessions.assert_called_once_with()


def test_status_json_missing_session_does_not_query_assignments(mock_common_state):
    mock_common_state.store.list.return_value = {"s1": _session()}

    result = runner.invoke(app, ["status", "-s", "missing", "--json"])

    assert result.exit_code == 1
    payload = _json_stdout(result)
    assert payload["schema"] == "colab.status.v1"
    assert payload["errors"][0]["code"] == "SESSION_NOT_FOUND"
    mock_common_state.client.list_assignments.assert_not_called()
    mock_common_state.sync_sessions.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ["status", "--probe", "--json"],
        ["status", "-s", "s1", "--probe"],
    ],
)
def test_probe_requires_explicit_session_and_json(args, mock_common_state):
    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "probe" in result.output.lower()
    mock_common_state.store.list.assert_not_called()
    mock_common_state.client.list_assignments.assert_not_called()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_probe_timeout_is_validated_before_state_access(value, mock_common_state):
    result = runner.invoke(
        app,
        [
            "status",
            "-s",
            "s1",
            "--probe",
            "--json",
            "--timeout",
            value,
        ],
    )

    assert result.exit_code == 2
    normalized = " ".join(result.output.split())
    assert "finite number" in normalized
    assert "than 0" in normalized
    mock_common_state.store.list.assert_not_called()
    mock_common_state.client.list_assignments.assert_not_called()


def test_status_probe_json_reports_requested_actual_and_resources(
    mock_common_state, mocker
):
    mock_common_state.store.list.return_value = {"s1": _session()}
    mock_common_state.client.list_assignments.return_value = [_assignment()]
    mocker.patch(
        "colab_cli.observability.probes.probe_session",
        return_value=ProbeObservation(
            status="ok",
            observed_at="2026-07-22T00:00:00Z",
            duration_ms=123,
            timeout_seconds=20.0,
            gpu=GpuObservation(
                available=True,
                name="NVIDIA RTX PRO 6000",
                driver_version="580.00",
                memory_total_bytes=100,
                memory_used_bytes=25,
                utilization_percent=75,
                memory_utilization_percent=25,
                temperature_c=61,
                sources=["runtime_api", "existing_kernel"],
            ),
            memory=MemoryObservation(
                total_bytes=1000,
                available_bytes=400,
                used_bytes=600,
                sources=["runtime_api"],
            ),
            disk=DiskObservation(
                total_bytes=5000,
                used_bytes=2000,
                free_bytes=3000,
                sources=["runtime_api"],
            ),
            runtime=RuntimeObservation(
                boot_id="boot-1",
                python_version="3.12.0",
                platform="Linux",
                source="existing_kernel",
            ),
        ),
    )

    result = runner.invoke(
        app, ["status", "-s", "s1", "--probe", "--json", "--timeout", "20"]
    )

    assert result.exit_code == 0, result.output
    payload = _json_stdout(result)
    observed = payload["sessions"][0]
    assert observed["local"]["requested_accelerator"] == "T4"
    assert observed["assignment"]["accelerator"] == "L4"
    assert observed["probe"]["gpu"]["name"] == "NVIDIA RTX PRO 6000"
    assert observed["probe"]["runtime"]["boot_id"] == "boot-1"
    assert observed["probe"]["duration_ms"] == 123
    assert observed["compute_units"]["balance"] is None
    assert observed["compute_units"]["consumption_rate_hourly"] is None
    assert observed["compute_units"]["unavailable_reason"]


def test_cpu_probe_without_existing_kernel_returns_partial_not_failure(mocker):
    session = _session(accelerator="NONE", variant="DEFAULT", kernel_id=None)
    mocker.patch(
        "colab_cli.observability.probes._fetch_runtime_resources",
        return_value=_resources(),
    )

    result = probe_session(session, timeout=20)

    assert result.status == "partial"
    assert result.gpu.available is False
    assert result.gpu.unavailable_reason == "no_gpu_reported"
    assert result.memory.total_bytes == 32_000
    assert result.disk.free_bytes == 60_000
    assert result.runtime.boot_id is None
    assert {issue.code for issue in result.issues} == {"EXISTING_KERNEL_NOT_RECORDED"}


def test_runtime_resource_api_preserves_zero_usage_values(mocker):
    session = _session(kernel_id=None)
    mocker.patch(
        "colab_cli.observability.probes._fetch_runtime_resources",
        return_value=_RawResources(
            memory=_RawMemory(totalBytes=32_000, freeBytes=32_000),
            disks=[
                _RawDisk(
                    filesystem=_RawFilesystem(
                        label="/content",
                        totalBytes=100_000,
                        usedBytes=0,
                    )
                )
            ],
            gpus=[
                _RawGpu(
                    name="Tesla T4",
                    memoryUsedBytes=0,
                    memoryTotalBytes=16_000,
                    gpuUtilization=0,
                    memoryUtilization=0,
                )
            ],
        ),
    )

    result = probe_session(session, timeout=20)

    assert result.memory.used_bytes == 0
    assert result.disk.used_bytes == 0
    assert result.disk.free_bytes == 100_000
    assert result.gpu.memory_used_bytes == 0
    assert result.gpu.utilization_percent == 0
    assert result.gpu.memory_utilization_percent == 0


def test_resource_api_gpu_survives_nvidia_smi_failure(mocker):
    session = _session()
    mocker.patch(
        "colab_cli.observability.probes._fetch_runtime_resources",
        return_value=_resources(
            gpu=_RawGpu(
                name="Tesla T4",
                memoryUsedBytes=4_000,
                memoryTotalBytes=16_000,
                gpuUtilization=0.5,
                memoryUtilization=0.25,
            )
        ),
    )
    executor = MagicMock()
    executor.execute_json.return_value = {
        "gpu": {
            "available": False,
            "unavailable_reason": "nvidia_smi_failed",
        },
        "memory": {},
        "disk": {},
        "runtime": {},
        "issues": [
            {
                "code": "NVIDIA_SMI_FAILED",
                "message": "nvidia-smi failed",
                "retryable": True,
            }
        ],
    }

    @contextlib.contextmanager
    def existing(*_args, **_kwargs):
        yield executor

    mocker.patch(
        "colab_cli.observability.probes.open_existing_kernel_executor",
        existing,
    )

    result = probe_session(session, timeout=20)

    assert result.status == "partial"
    assert result.gpu.available is True
    assert result.gpu.name == "Tesla T4"
    assert result.gpu.memory_total_bytes == 16_000
    assert result.gpu.utilization_percent == 50
    assert "runtime_api" in result.gpu.sources
    assert "NVIDIA_SMI_FAILED" in {issue.code for issue in result.issues}


def test_partial_kernel_metrics_preserve_other_source_values_and_accept_zero():
    gpu = GpuObservation(
        available=True,
        name="Tesla T4",
        memory_total_bytes=16_000,
        memory_used_bytes=4_000,
        utilization_percent=50,
        memory_utilization_percent=25,
        sources=["runtime_api"],
    )
    memory = MemoryObservation(
        total_bytes=32_000,
        available_bytes=12_000,
        used_bytes=20_000,
        sources=["runtime_api"],
    )
    disk = DiskObservation(
        total_bytes=100_000,
        used_bytes=40_000,
        free_bytes=60_000,
        sources=["runtime_api"],
    )

    _merge_supplemental(
        {
            "gpu": {
                "available": True,
                "name": "Tesla T4",
                "memory_total_bytes": None,
                "memory_used_bytes": 0,
                "utilization_percent": None,
                "memory_utilization_percent": None,
            },
            "memory": {
                "total_bytes": 32_000,
                "available_bytes": None,
                "used_bytes": None,
            },
            "disk": {
                "total_bytes": 100_000,
                "used_bytes": None,
                "free_bytes": None,
            },
            "runtime": {},
            "issues": [],
        },
        gpu=gpu,
        memory=memory,
        disk=disk,
        runtime=RuntimeObservation(),
        issues=[],
    )

    assert gpu.memory_total_bytes == 16_000
    assert gpu.memory_used_bytes == 0
    assert gpu.utilization_percent == 50
    assert gpu.memory_utilization_percent == 25
    assert memory.available_bytes == 12_000
    assert memory.used_bytes == 20_000
    assert disk.used_bytes == 40_000
    assert disk.free_bytes == 60_000
    assert "existing_kernel" in gpu.sources
    assert "existing_kernel" in memory.sources
    assert "existing_kernel" in disk.sources


def test_resource_and_kernel_timeouts_are_structured(mocker):
    session = _session()
    mocker.patch(
        "colab_cli.observability.probes._fetch_runtime_resources",
        side_effect=requests.Timeout("resource deadline"),
    )

    @contextlib.contextmanager
    def timed_out(*_args, **_kwargs):
        raise TimeoutError("kernel deadline")
        yield  # pragma: no cover

    mocker.patch(
        "colab_cli.observability.probes.open_existing_kernel_executor",
        timed_out,
    )

    result = probe_session(session, timeout=20)

    assert result.status == "timeout"
    assert {issue.code for issue in result.issues} == {
        "RUNTIME_RESOURCE_API_TIMEOUT",
        "EXISTING_KERNEL_PROBE_TIMEOUT",
    }


def test_existing_kernel_executor_builds_exact_endpoint_without_creation(mocker):
    session = _session(kernel_id="kernel id/with slash")
    low_level = MagicMock()
    low_level.channels_running = True
    low_level.session = SimpleNamespace(session=None)
    constructor = mocker.patch(
        "colab_cli.observability.probes.KernelWebSocketClient",
        return_value=low_level,
    )

    executor = ExistingKernelJsonExecutor(session, connect_timeout=4)
    connected = executor._connect()

    assert connected is low_level
    kwargs = constructor.call_args.kwargs
    assert "/api/kernels/kernel%20id%2Fwith%20slash/channels" in kwargs["endpoint"]
    assert kwargs["timeout"] == 4
    low_level.start_channels.assert_called_once_with(stdin=False)
    executor.close()
    low_level.stop_channels.assert_called_once_with()


def test_existing_kernel_connection_failure_uses_short_cleanup_budget(mocker):
    session = _session()
    low_level = MagicMock()
    low_level.channels_running = False
    low_level.session = SimpleNamespace(session=None)
    mocker.patch(
        "colab_cli.observability.probes.KernelWebSocketClient",
        return_value=low_level,
    )

    executor = ExistingKernelJsonExecutor(session, connect_timeout=5)
    with pytest.raises(TimeoutError, match="did not become ready"):
        executor._connect()

    assert low_level.timeout == 1.0
    low_level.stop_channels.assert_called_once_with()


def test_runtime_resource_api_uses_bounded_authenticated_read(mocker):
    session = _session()
    response = MagicMock()
    response.text = ")]}'\n" + json.dumps(
        {
            "memory": {"totalBytes": 100, "freeBytes": 40},
            "disks": [],
            "gpus": [],
        }
    )
    get = mocker.patch(
        "colab_cli.observability.probes.requests.get",
        return_value=response,
    )

    resources = _fetch_runtime_resources(session, timeout=4)

    assert resources.memory.total_bytes == 100
    kwargs = get.call_args.kwargs
    assert kwargs["timeout"] == (2.0, 2.0)
    assert "params" not in kwargs
    assert kwargs["headers"]["X-Colab-Runtime-Proxy-Token"] == session.token
    response.raise_for_status.assert_called_once_with()


def test_jobs_json_reports_state_error_and_log_sizes(mock_common_state, mocker):
    session = _session()
    mock_common_state.store.list.return_value = {"s1": session}
    executor = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = executor
    context_manager.__exit__.return_value = False
    existing = mocker.patch(
        "colab_cli.commands.jobs.open_existing_kernel_executor",
        return_value=context_manager,
    )
    client = mocker.patch("colab_cli.commands.jobs.RemoteJobClient")
    client.return_value.list_jobs.return_value = [
        {
            "job_id": "train",
            "state": "failed",
            "heartbeat_at": "2026-07-22T00:00:00Z",
            "returncode": 1,
            "error": "CUDA out of memory",
            "stdout_path": "/content/jobs/train/stdout.log",
            "stdout_size": 123,
            "stderr_path": "/content/jobs/train/stderr.log",
            "stderr_size": 456,
        }
    ]
    high_level = mocker.patch("colab_cli.commands.jobs.open_remote_executor")

    result = runner.invoke(app, ["jobs", "-s", "s1", "--json"])

    assert result.exit_code == 0, result.output
    payload = _json_stdout(result)
    assert payload["schema"] == "colab.jobs.v1"
    job = payload["jobs"][0]
    assert job["state"] == "failed"
    assert job["heartbeat_at"] == "2026-07-22T00:00:00Z"
    assert job["returncode"] == 1
    assert job["error"] == "CUDA out of memory"
    assert job["stdout"]["size_bytes"] == 123
    assert job["stderr"]["size_bytes"] == 456
    existing.assert_called_once_with(session, connect_timeout=5.0)
    client.return_value.list_jobs.assert_called_once_with(timeout=120.0)
    high_level.assert_not_called()
    mock_common_state.resolve_session.assert_not_called()


def test_jobs_json_requires_selection_when_multiple_local_sessions(
    mock_common_state, mocker
):
    mock_common_state.store.list.return_value = {
        "s1": _session(name="s1", endpoint="e1"),
        "s2": _session(name="s2", endpoint="e2"),
    }
    existing = mocker.patch("colab_cli.commands.jobs.open_existing_kernel_executor")

    result = runner.invoke(app, ["jobs", "--json"])

    assert result.exit_code == 1
    payload = _json_stdout(result)
    assert payload["errors"][0]["code"] == "SESSION_SELECTION_REQUIRED"
    existing.assert_not_called()


def test_jobs_json_refuses_to_create_kernel_when_id_is_missing(
    mock_common_state, mocker
):
    mock_common_state.store.list.return_value = {"s1": _session(kernel_id=None)}
    high_level = mocker.patch("colab_cli.commands.jobs.open_remote_executor")

    result = runner.invoke(app, ["jobs", "-s", "s1", "--json"])

    assert result.exit_code == 1
    payload = _json_stdout(result)
    assert payload["errors"][0]["code"] == "EXISTING_KERNEL_NOT_RECORDED"
    high_level.assert_not_called()


def test_probe_redacts_runtime_token_from_source_errors(mocker):
    session = _session(kernel_id=None)
    mocker.patch(
        "colab_cli.observability.probes._fetch_runtime_resources",
        side_effect=requests.ConnectionError(
            "GET https://runtime.test/api?"
            f"colab-runtime-proxy-token={session.token}&authuser=0"
        ),
    )

    result = probe_session(session, timeout=20)

    message = next(
        issue.message
        for issue in result.issues
        if issue.code == "RUNTIME_RESOURCE_API_UNAVAILABLE"
    )
    assert session.token not in message
    assert "<redacted>" in message


def test_kernel_probe_redacts_runtime_token_from_remote_issues():
    session = _session()
    issues = []

    _merge_supplemental(
        {
            "gpu": {},
            "memory": {},
            "disk": {},
            "runtime": {},
            "issues": [
                {
                    "code": "REMOTE_PROBE_WARNING",
                    "message": f"TOKEN={session.token}",
                    "retryable": True,
                }
            ],
        },
        gpu=GpuObservation(),
        memory=MemoryObservation(),
        disk=DiskObservation(),
        runtime=RuntimeObservation(),
        issues=issues,
        secrets=(session.token,),
    )

    assert session.token not in issues[0].message
    assert "<redacted>" in issues[0].message


def test_jobs_json_redacts_runtime_token_from_failure(mock_common_state, mocker):
    session = _session()
    mock_common_state.store.list.return_value = {"s1": session}
    context_manager = MagicMock()
    context_manager.__enter__.side_effect = TimeoutError(
        f"wss://runtime.test/channels?token={session.token}"
    )
    context_manager.__exit__.return_value = False
    mocker.patch(
        "colab_cli.commands.jobs.open_existing_kernel_executor",
        return_value=context_manager,
    )

    result = runner.invoke(app, ["jobs", "-s", "s1", "--json"])

    assert result.exit_code == 1
    payload = _json_stdout(result)
    message = payload["errors"][0]["message"]
    assert session.token not in message
    assert "<redacted>" in message


def test_jobs_json_redacts_secret_like_job_error(mock_common_state, mocker):
    session = _session()
    mock_common_state.store.list.return_value = {"s1": session}
    executor = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = executor
    context_manager.__exit__.return_value = False
    mocker.patch(
        "colab_cli.commands.jobs.open_existing_kernel_executor",
        return_value=context_manager,
    )
    client = mocker.patch("colab_cli.commands.jobs.RemoteJobClient")
    client.return_value.list_jobs.return_value = [
        {
            "job_id": "train",
            "state": "failed",
            "error": "HF_TOKEN=super-secret-value",
        }
    ]

    result = runner.invoke(app, ["jobs", "-s", "s1", "--json"])

    assert result.exit_code == 0
    payload = _json_stdout(result)
    error = payload["jobs"][0]["error"]
    assert "super-secret-value" not in error
    assert "<redacted>" in error


def test_jobs_json_redacts_bare_runtime_token_from_job_error(mock_common_state, mocker):
    session = _session()
    mock_common_state.store.list.return_value = {"s1": session}
    executor = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = executor
    context_manager.__exit__.return_value = False
    mocker.patch(
        "colab_cli.commands.jobs.open_existing_kernel_executor",
        return_value=context_manager,
    )
    client = mocker.patch("colab_cli.commands.jobs.RemoteJobClient")
    client.return_value.list_jobs.return_value = [
        {
            "job_id": "train",
            "state": "failed",
            "error": f"remote failure: {session.token}",
        }
    ]

    result = runner.invoke(app, ["jobs", "-s", "s1", "--json"])

    assert result.exit_code == 0
    error = _json_stdout(result)["jobs"][0]["error"]
    assert session.token not in error
    assert "<redacted>" in error


def test_structured_redaction_hides_home_and_common_secret_forms(monkeypatch, tmp_path):
    from colab_cli.observability.redaction import redact_text

    monkeypatch.setenv("HOME", str(tmp_path))
    message = redact_text(
        f"{tmp_path}/config TOKEN=abc "
        "Authorization: Bearer bearer-value "
        "?colab-runtime-proxy-token=proxy-value&authuser=0"
    )

    assert str(tmp_path) not in message
    assert "abc" not in message
    assert "bearer-value" not in message
    assert "proxy-value" not in message
    assert message.count("<redacted>") >= 3


def test_structured_redaction_hides_quoted_and_multiline_private_keys():
    from colab_cli.observability.redaction import redact_text

    message = redact_text(
        'API_KEY="value with spaces" '
        "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\n"
        "key-material\n"
        "-----END PRIVATE KEY-----"
    )

    assert "value with spaces" not in message
    assert "key-material" not in message
    assert message.count("<redacted>") == 2
