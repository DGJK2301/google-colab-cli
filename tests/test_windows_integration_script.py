# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SMOKE = REPO_ROOT / "integration" / "repro_windows_exec_control" / "test.ps1"


def _write_fake_uv(path: Path, *, new_code: int, stop_code: int) -> None:
    path.write_text(
        '@echo %*>>"%UV_CALL_LOG%"\n'
        f'@if /I "%3"=="new" exit /b {new_code}\n'
        '@if /I "%3"=="exec" (\n'
        '  @echo %* | findstr /C:"Selected Cell" >nul\n'
        "  @if not errorlevel 1 (\n"
        "    @echo WINDOWS_CELL_SELECTION_OK\n"
        "    @exit /b 0\n"
        "  )\n"
        "  @echo EXPECTED_FAIL_ON_ERROR\n"
        "  @exit /b 1\n"
        ")\n"
        f'@if /I "%3"=="stop" exit /b {stop_code}\n'
        "@exit /b 0\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell cleanup regression")
def test_windows_smoke_stops_session_when_new_returns_nonzero(tmp_path):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is not installed")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "uv-calls.txt"
    _write_fake_uv(fake_bin / "uv.cmd", new_code=7, stop_code=0)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["UV_CALL_LOG"] = str(call_log)

    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-File",
            str(WINDOWS_SMOKE),
            "-Session",
            "cleanup-regression",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    calls = call_log.read_text(encoding="utf-8").splitlines()
    new_call = next(
        call for call in calls if call.startswith("run colab new --session ")
    )
    session_name = new_call.removeprefix("run colab new --session ")
    assert session_name.startswith("cleanup-regression-")
    assert len(session_name) == len("cleanup-regression-") + 8
    assert f"run colab stop --session {session_name}" in calls


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell cleanup regression")
def test_windows_smoke_fails_when_session_cleanup_fails(tmp_path):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is not installed")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "uv-calls.txt"
    _write_fake_uv(fake_bin / "uv.cmd", new_code=0, stop_code=9)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["UV_CALL_LOG"] = str(call_log)

    result = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(WINDOWS_SMOKE), "-Session", "cleanup"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert any(call.startswith("run colab stop --session cleanup-") for call in calls)
