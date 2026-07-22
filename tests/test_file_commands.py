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

import pytest
import typer
from click import unstyle
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.commands.files import _chunk_size_mib_to_bytes, _open_transfer
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
    assert mock_transfer_class.call_args.kwargs["chunk_size"] == 256 * 1024
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


@pytest.mark.parametrize(
    "raw_value", ["nan", "inf", "-inf", "1e308", "0", "-1", "0.0000001", ""]
)
@pytest.mark.parametrize("command", ["upload", "download"])
def test_transfer_commands_reject_invalid_chunk_size_before_side_effects(
    command, raw_value, mock_common_state, mocker, tmp_path
):
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    target = tmp_path / "target.bin"
    mock_open_executor = mocker.patch("colab_cli.commands.files.open_remote_executor")
    mock_contents = mocker.patch("colab_cli.commands.files.ContentsClient")

    if command == "upload":
        args = [
            "upload",
            "--chunk-size-mib",
            raw_value,
            str(source),
            "content/source.bin",
        ]
    else:
        args = [
            "download",
            "--chunk-size-mib",
            raw_value,
            "content/source.bin",
            str(target),
        ]

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "chunk-size-mib" in unstyle(result.output)
    assert "Traceback" not in result.output
    mock_common_state.resolve_session.assert_not_called()
    mock_open_executor.assert_not_called()
    mock_contents.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        1e308,
        0.0,
        -1.0,
        0.5 / (1024 * 1024),
    ],
)
def test_open_transfer_defensively_rejects_invalid_chunk_size_before_executor(
    value, mock_common_state, mocker
):
    mock_open_executor = mocker.patch("colab_cli.commands.files.open_remote_executor")

    with pytest.raises(typer.BadParameter):
        _open_transfer(MagicMock(), mock_common_state, chunk_size_mib=value)

    mock_open_executor.assert_not_called()


def test_one_byte_chunk_size_is_valid():
    assert _chunk_size_mib_to_bytes(1 / (1024 * 1024)) == 1
