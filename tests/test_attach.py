# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.client import (
    Accelerator,
    AssignmentVariant,
    ListedAssignment,
    RuntimeProxyInfo,
    Shape,
)


runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_control_executor(mocker):
    executor = mocker.patch(
        "colab_cli.commands.session.open_remote_executor"
    ).return_value
    executor.execute_json.return_value = {"attached": True}
    return executor


def _assignment(
    endpoint="endpoint-1",
    token="runtime-secret",
):
    return ListedAssignment(
        accelerator=Accelerator.G4,
        endpoint=endpoint,
        variant=AssignmentVariant.GPU,
        machineShape=Shape.HIGH_RAM,
        runtimeProxyInfo=RuntimeProxyInfo(
            token=token,
            tokenExpiresInSeconds=3600,
            url="https://runtime.example.test/",
        ),
    )


def _prepare(state, tmp_path):
    state.store.list_strict.return_value = {}
    state.auth_provider = SimpleNamespace(value="oauth2")
    state.config_path = str(tmp_path / "sessions.json")
    return state


def _json_result(result):
    assert result.stdout.strip().startswith("{"), result.output
    return json.loads(result.stdout)


def test_attach_success_is_transactional(
    mock_common_state,
    mock_control_executor,
    mocker,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )
    assignment = _assignment()
    state.client.list_assignments.return_value = [assignment]
    spawn = mocker.patch(
        "colab_cli.commands.session.spawn_keep_alive",
        return_value=4321,
    )
    update = mocker.patch("colab_cli.auto_update.run_background_check")

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            assignment.endpoint,
            "-s",
            "xoftr",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _json_result(result)
    assert payload["schema"] == "colab.attach.v1"
    assert payload["machine_shape"] == "HIGH_RAM"
    assert payload["keep_alive_pid"] == 4321
    assert payload["control_connected"] is True
    assert assignment.runtime_proxy_info.token not in result.stdout
    state.store.claim_strict.assert_called_once()
    state.store.update_claim_strict.assert_called_once()
    first = state.store.claim_strict.call_args.args[0]
    final = state.store.update_claim_strict.call_args.args[0]
    assert first.keep_alive_pid is None
    assert final.keep_alive_pid == 4321
    state.client.keep_alive_assignment.assert_called_once_with(assignment.endpoint)
    state.client.unassign.assert_not_called()
    mock_control_executor.execute_json.assert_called_once_with(
        "_colab_cli_result = {'attached': True}",
        timeout=30.0,
    )
    mock_control_executor.close.assert_called_once_with()
    spawn.assert_called_once()
    update.assert_not_called()


def test_name_conflict_precedes_network(
    mock_common_state,
    mock_control_executor,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )
    state.store.list_strict.return_value = {"xoftr": SimpleNamespace(endpoint="old")}

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "endpoint-1",
            "-s",
            "xoftr",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert _json_result(result)["error"]["code"] == "SESSION_NAME_CONFLICT"
    state.client.list_assignments.assert_not_called()
    mock_control_executor.execute_json.assert_not_called()


def test_endpoint_conflict_is_refused(
    mock_common_state,
    mock_control_executor,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )
    state.store.list_strict.return_value = {
        "existing": SimpleNamespace(
            name="existing",
            endpoint="endpoint-1",
        )
    }
    state.client.list_assignments.return_value = [_assignment()]

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "endpoint-1",
            "-s",
            "xoftr",
            "--json",
        ],
    )

    assert _json_result(result)["error"]["code"] == "ENDPOINT_ALREADY_ATTACHED"
    state.client.unassign.assert_not_called()
    mock_control_executor.execute_json.assert_not_called()


def test_spawn_failure_rolls_back_local_only(
    mock_common_state,
    mock_control_executor,
    mocker,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )
    state.client.list_assignments.return_value = [_assignment()]
    mocker.patch(
        "colab_cli.commands.session.spawn_keep_alive",
        side_effect=OSError("spawn failed"),
    )

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "endpoint-1",
            "-s",
            "xoftr",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert _json_result(result)["error"]["code"] == "ATTACH_LOCAL_SETUP_FAILED"
    state.store.remove_claim_strict.assert_called_once_with(
        "xoftr",
        "endpoint-1",
    )
    state.client.unassign.assert_not_called()
    mock_control_executor.close.assert_called_once_with()


def test_control_connection_failure_rolls_back_local_only(
    mock_common_state,
    mock_control_executor,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )
    state.client.list_assignments.return_value = [_assignment()]
    mock_control_executor.execute_json.side_effect = TimeoutError("kernel unavailable")

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "endpoint-1",
            "-s",
            "xoftr",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert _json_result(result)["error"]["code"] == "ATTACH_LOCAL_SETUP_FAILED"
    state.store.remove_claim_strict.assert_called_once_with(
        "xoftr",
        "endpoint-1",
    )
    state.client.unassign.assert_not_called()
    mock_control_executor.close.assert_called_once_with()


def test_no_connect_defers_control_kernel(
    mock_common_state,
    mock_control_executor,
    mocker,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )
    state.client.list_assignments.return_value = [_assignment()]
    mocker.patch(
        "colab_cli.commands.session.spawn_keep_alive",
        return_value=12,
    )

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "endpoint-1",
            "-s",
            "xoftr",
            "--no-connect",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _json_result(result)
    assert payload["control_connected"] is False
    assert any("deferred" in warning for warning in payload["warnings"])
    mock_control_executor.execute_json.assert_not_called()


def test_unknown_endpoint_does_not_write_state(
    mock_common_state,
    mock_control_executor,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )
    state.client.list_assignments.return_value = []

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "missing",
            "-s",
            "xoftr",
            "--json",
        ],
    )

    assert _json_result(result)["error"]["code"] == "ASSIGNMENT_NOT_FOUND"
    state.store.claim_strict.assert_not_called()
    state.client.unassign.assert_not_called()
    mock_control_executor.execute_json.assert_not_called()


def test_store_error_is_not_misreported_as_invalid_session_name(
    mock_common_state,
    mock_control_executor,
    tmp_path,
):
    state = _prepare(mock_common_state, tmp_path)
    state.store.list_strict.side_effect = ValueError(
        "session store root must be a JSON object"
    )

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "endpoint-1",
            "-s",
            "xoftr",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert _json_result(result)["error"]["code"] == "ATTACH_LOCAL_SETUP_FAILED"
    state.client.list_assignments.assert_not_called()
    mock_control_executor.execute_json.assert_not_called()


@pytest.mark.parametrize(
    "name",
    [
        "bad?name",
        "CON",
        "trailing.",
    ],
)
def test_invalid_windows_names_precede_network(
    name,
    mock_common_state,
    mock_control_executor,
    tmp_path,
):
    state = _prepare(
        mock_common_state,
        tmp_path,
    )

    result = runner.invoke(
        app,
        [
            "attach",
            "--endpoint",
            "endpoint-1",
            "-s",
            name,
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert _json_result(result)["error"]["code"] == "INVALID_SESSION_NAME"
    state.client.list_assignments.assert_not_called()
    mock_control_executor.execute_json.assert_not_called()
