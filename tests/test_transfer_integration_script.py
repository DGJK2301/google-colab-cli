# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "integration" / "repro_resumable_transfer_jobs" / "test.ps1"


def test_transfer_live_script_uses_installed_cli_and_json():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "uv run colab" not in text
    assert '[string]$ColabCommand = "colab"' in text
    assert '"--json"' in text
    assert '"colab.transfer.v1"' in text
    assert "ConvertFrom-Json" in text
    assert "Assert-TransferResult" in text


def test_transfer_live_script_has_required_benchmark_matrix():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "[ValidateSet(8, 64)]" in text
    assert "[double[]]$ChunkSizesMiB" in text
    assert "chunk_size_mib" in text
    assert "upload_mib_per_second" in text
    assert "download_mib_per_second" in text
    assert "colab.transfer.benchmark.v1" in text


def test_transfer_live_script_checks_local_process_cleanup():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Get-CimInstance Win32_Process" in text
    assert "Assert-NoNewColabProcesses" in text
    assert "colab.exe" in text
    assert "colab_cli" in text


def test_transfer_live_script_releases_session_in_finally():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "finally {" in text
    assert "Invoke-Colab stop --session $sessionName" in text
    assert "active Colab assignments remain after cleanup" in text
