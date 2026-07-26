# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.doctor import (
    _windows_permission_observation,
    collect_doctor,
)
from colab_cli.state import SettingsStore, StateStore


runner = CliRunner()


def _state(tmp_path):
    return SimpleNamespace(
        store=StateStore(str(tmp_path / "sessions.json")),
        settings_store=SettingsStore(str(tmp_path / "settings.json")),
        client=MagicMock(),
        auth_provider=SimpleNamespace(value="oauth2"),
    )


def _local(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "colab_cli.doctor.TOKEN_CONFIG_PATH",
        str(tmp_path / "token.json"),
    )
    monkeypatch.setenv(
        "COLAB_CLI_TRANSFER_LEASE_DIR",
        str(tmp_path / "leases"),
    )


def _json_result(result):
    assert result.stdout.strip().startswith("{"), result.output
    return json.loads(result.stdout)


def test_default_doctor_is_local_only(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    _local(monkeypatch, tmp_path)

    result = collect_doctor(
        state,
        network=False,
        timeout=7,
    )

    assert result.network.status == ("not_requested")
    state.client.list_assignments.assert_not_called()


def test_token_values_never_serialize(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    token = tmp_path / "token.json"
    token.write_text(
        json.dumps(
            {
                "token": "access-secret",
                "refresh_token": ("refresh-secret"),
                "rapt_token": "rapt-secret",
                "expiry": ("2099-01-01T00:00:00Z"),
                "scopes": [
                    "scope-b",
                    "scope-a",
                ],
            }
        ),
        encoding="utf-8",
    )
    token.chmod(0o600)
    monkeypatch.setattr(
        "colab_cli.doctor.TOKEN_CONFIG_PATH",
        str(token),
    )
    monkeypatch.setenv(
        "COLAB_CLI_TRANSFER_LEASE_DIR",
        str(tmp_path / "leases"),
    )

    result = collect_doctor(
        state,
        network=False,
        timeout=7,
    )
    serialized = result.model_dump_json(by_alias=True)

    assert result.token.refresh_token_present
    assert result.token.rapt_token_present
    assert result.token.scopes == [
        "scope-a",
        "scope-b",
    ]
    assert "access-secret" not in serialized
    assert "refresh-secret" not in serialized
    assert "rapt-secret" not in serialized


def test_adc_ignores_unused_oauth_token_cache(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    state.auth_provider = SimpleNamespace(value="adc")
    token = tmp_path / "token.json"
    token.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(
        "colab_cli.doctor.TOKEN_CONFIG_PATH",
        str(token),
    )
    monkeypatch.setenv(
        "COLAB_CLI_TRANSFER_LEASE_DIR",
        str(tmp_path / "leases"),
    )

    result = collect_doctor(
        state,
        network=False,
        timeout=7,
    )

    assert result.token.parse_status == "not_applicable"
    assert result.token.permission.status == "not_applicable"
    assert "TOKEN_CACHE_INVALID" not in {issue.code for issue in result.errors}


def test_invalid_session_store_is_error(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    Path(state.store.path).write_text(
        "not json",
        encoding="utf-8",
    )
    _local(monkeypatch, tmp_path)

    result = collect_doctor(
        state,
        network=False,
        timeout=7,
    )

    assert not result.ok
    assert result.status == "error"
    assert "SESSION_STORE_INVALID" in {issue.code for issue in result.errors}


def test_network_query_is_opt_in_and_bounded(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    state.client.list_assignments.return_value = [SimpleNamespace(endpoint="orphan")]
    _local(monkeypatch, tmp_path)

    result = collect_doctor(
        state,
        network=True,
        timeout=7,
    )

    assert result.network.orphan_endpoints == ["orphan"]
    state.client.list_assignments.assert_called_once_with(timeout=(5.0, 7.0))
    state.client.assign.assert_not_called()


def test_stale_transfer_lease_is_reported(
    tmp_path,
    monkeypatch,
):
    state = _state(tmp_path)
    root = tmp_path / "leases"
    root.mkdir()
    (root / "upload.json").write_text(
        json.dumps(
            {
                "state": "active",
                "lease_id": "lease-1",
                "pid": 999,
                "process_start_token": ("proc:old"),
            }
        ),
        encoding="utf-8",
    )
    _local(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "colab_cli.doctor.process_identity_state",
        lambda *_args: "dead",
    )

    result = collect_doctor(
        state,
        network=False,
        timeout=7,
    )

    assert result.transfer_leases.entries[0].diagnostic == "stale"
    assert "STALE_TRANSFER_LEASE" in {issue.code for issue in result.warnings}


def test_doctor_json_is_clean(
    mock_common_state,
    mocker,
    tmp_path,
    monkeypatch,
):
    mock_common_state.store.path = str(tmp_path / "sessions.json")
    mock_common_state.settings_store.path = str(tmp_path / "settings.json")
    mock_common_state.auth_provider = SimpleNamespace(value="oauth2")
    _local(monkeypatch, tmp_path)
    update = mocker.patch("colab_cli.auto_update.run_background_check")

    result = runner.invoke(
        app,
        ["doctor", "--json"],
    )

    assert result.exit_code == 0
    payload = _json_result(result)
    assert payload["schema"] == ("colab.doctor.v1")
    assert payload["network"]["status"] == ("not_requested")
    update.assert_not_called()
    (mock_common_state.client.list_assignments.assert_not_called())


def test_timeout_validation_precedes_network(
    mock_common_state,
):
    result = runner.invoke(
        app,
        [
            "doctor",
            "--json",
            "--network",
            "--timeout",
            "nan",
        ],
    )

    assert result.exit_code == 2
    (mock_common_state.client.list_assignments.assert_not_called())


def test_windows_acl_recognizes_stable_broad_principal_sid(
    tmp_path,
    monkeypatch,
):
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "colab_cli.doctor.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(f"{token} S-1-5-11:(I)(RX)\nSuccessfully processed 1 files"),
        ),
    )

    result = _windows_permission_observation(token)

    assert result.status == "insecure"
