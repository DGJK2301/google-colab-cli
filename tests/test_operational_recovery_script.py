# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integration" / "repro_operational_recovery" / "test.ps1"


def test_script_uses_installed_cli_and_json_contracts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "uv run colab" not in text
    assert '[string]$ColabCommand = "colab"' in text
    assert '"colab.doctor.v1"' in text
    assert '"colab.attach.v1"' in text
    assert '"colab.monitor.summary.v1"' in text
    assert "ConvertFrom-Json" in text


def test_script_checks_durable_monitor_files():
    text = SCRIPT.read_text(encoding="utf-8")

    for name in (
        "stdout.log",
        "stderr.log",
        "job.jsonl",
        "resources.jsonl",
        "events.jsonl",
        "monitor_state.json",
        "summary.json",
    ):
        assert name in text


def test_script_releases_session_in_finally():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "finally {" in text
    assert "Invoke-Colab stop --session $sessionName" in text
    assert "LIVE_CPU_OPERATIONAL_RECOVERY_OK" in text
