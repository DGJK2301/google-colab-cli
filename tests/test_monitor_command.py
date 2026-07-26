# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json

from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.monitor_models import MonitorSummary
from colab_cli.state import SessionState


runner = CliRunner()


def _payload(result):
    assert result.stdout.strip().startswith("{"), result.output
    return json.loads(result.stdout)


def _session():
    return SessionState(
        name="xoftr",
        endpoint="endpoint-1",
        token="runtime-secret",
        url="https://runtime.example.test/",
    )


def test_monitor_json_configuration_failure_is_structured(
    mock_common_state,
    mocker,
):
    mock_common_state.store.list_strict.return_value = {}
    update = mocker.patch("colab_cli.auto_update.run_background_check")

    result = runner.invoke(
        app,
        ["monitor", "train", "--json"],
    )

    assert result.exit_code == 2
    payload = _payload(result)
    assert payload["schema"] == ("colab.monitor.summary.v1")
    assert payload["error_code"] == ("SESSION_SELECTION_ERROR")
    update.assert_not_called()


def test_monitor_json_success_is_one_document(
    mock_common_state,
    mocker,
    tmp_path,
):
    session = _session()
    mock_common_state.store.list_strict.return_value = {session.name: session}
    summary = MonitorSummary(
        ok=True,
        status="completed",
        job_id="train",
        session_name=session.name,
        endpoint=session.endpoint,
        job_root="/content/.colab-cli/jobs",
        output_dir=str(tmp_path),
        remote_state="succeeded",
        remote_returncode=0,
        exit_code=0,
        started_at="2026-07-26T00:00:00Z",
        finished_at="2026-07-26T00:00:01Z",
        elapsed_seconds=1.0,
        stdout_offset=10,
        stderr_offset=5,
    )
    run_monitor = mocker.patch(
        "colab_cli.commands.monitor.run_monitor",
        return_value=summary,
    )

    result = runner.invoke(
        app,
        [
            "monitor",
            "train",
            "-s",
            session.name,
            "--output",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = _payload(result)
    assert payload["status"] == "completed"
    assert payload["stdout_offset"] == 10
    assert session.token not in result.stdout
    run_monitor.assert_called_once()
