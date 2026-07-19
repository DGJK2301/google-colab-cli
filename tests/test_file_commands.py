# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.transfer import TransferResult


runner = CliRunner()


def _session():
    session = MagicMock()
    session.name = "s1"
    session.url = "https://runtime"
    session.token = "token"
    session.kernel_id = None
    session.session_id = None
    return session


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_upload_uses_verified_transfer_and_closes_executor(
    mock_open_executor, mock_transfer_class, mock_common_state, tmp_path
):
    source = tmp_path / "repo.bundle"
    source.write_bytes(b"bundle")
    session = _session()
    mock_common_state.resolve_session.return_value = "s1"
    mock_common_state.store.get.return_value = session
    mock_transfer_class.return_value.upload.return_value = TransferResult(
        "content/repo.bundle", 6, "abc", 0
    )

    result = runner.invoke(
        app,
        ["upload", "-s", "s1", str(source), "content/repo.bundle"],
    )

    assert result.exit_code == 0
    mock_transfer_class.return_value.upload.assert_called_once_with(
        str(source), "content/repo.bundle", overwrite=True, resume=True
    )
    mock_open_executor.return_value.close.assert_called_once_with()
    assert "6 bytes" in result.output
    assert "sha256=abc" in result.output


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_download_uses_verified_transfer_and_closes_on_failure(
    mock_open_executor, mock_transfer_class, mock_common_state, tmp_path
):
    target = tmp_path / "model.ckpt"
    session = _session()
    mock_common_state.resolve_session.return_value = "s1"
    mock_common_state.store.get.return_value = session
    mock_transfer_class.return_value.download.side_effect = RuntimeError("broken")

    result = runner.invoke(
        app,
        ["download", "-s", "s1", "content/model.ckpt", str(target)],
    )

    assert result.exit_code == 1
    assert "broken" in result.output
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_upload_no_resume_and_no_overwrite_are_explicit(
    mock_open_executor, mock_transfer_class, mock_common_state, tmp_path
):
    source = tmp_path / "repo.bundle"
    source.write_bytes(b"bundle")
    session = _session()
    mock_common_state.resolve_session.return_value = "s1"
    mock_common_state.store.get.return_value = session
    mock_transfer_class.return_value.upload.return_value = TransferResult(
        "content/repo.bundle", 6, "abc", 0
    )

    result = runner.invoke(
        app,
        [
            "upload",
            "-s",
            "s1",
            "--no-resume",
            "--no-overwrite",
            str(source),
            "content/repo.bundle",
        ],
    )

    assert result.exit_code == 0
    mock_transfer_class.return_value.upload.assert_called_once_with(
        str(source), "content/repo.bundle", overwrite=False, resume=False
    )
