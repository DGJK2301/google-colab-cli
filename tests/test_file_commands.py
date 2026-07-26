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
    session.endpoint = "runtime-endpoint"
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


@pytest.fixture(autouse=True)
def isolated_transfer_leases(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "COLAB_CLI_TRANSFER_LEASE_DIR",
        str(tmp_path / "transfer-leases"),
    )


def _real_session(*, token="runtime-secret"):
    from colab_cli.state import SessionState

    return SessionState(
        name="s1",
        token=token,
        url="https://runtime.example.test",
        endpoint="runtime-endpoint",
        kernel_id="kernel-id",
        session_id="jupyter-session-id",
    )


def _json_payload(result):
    import json

    assert result.stdout.strip().startswith("{"), result.output
    return json.loads(result.stdout)


def _prepare_json_session(
    mock_common_state,
    session=None,
):
    selected = session or _real_session()
    mock_common_state.store.list.return_value = {selected.name: selected}
    mock_common_state.auth_provider = "oauth2"
    mock_common_state.client_oauth_config = None
    mock_common_state.config_path = None
    return selected


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_upload_json_emits_one_stable_document(
    mock_open_executor,
    mock_transfer_class,
    mock_common_state,
    mocker,
    tmp_path,
):
    source = tmp_path / "repo.bundle"
    source.write_bytes(b"bundle")
    selected = _prepare_json_session(mock_common_state)
    mock_transfer_class.return_value.upload.return_value = TransferResult(
        "content/repo.bundle",
        6,
        "abc",
        2,
        1,
    )
    update = mocker.patch("colab_cli.auto_update.run_background_check")

    result = runner.invoke(
        app,
        [
            "upload",
            "-s",
            selected.name,
            "--json",
            str(source),
            "content/repo.bundle",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _json_payload(result)
    assert payload["schema"] == "colab.transfer.v1"
    assert payload["status"] == "completed"
    assert payload["completed_bytes"] == 6
    assert payload["total_bytes"] == 6
    assert payload["resumed_from"] == 2
    assert payload["retry_count"] == 1
    assert payload["sha256"] == "abc"
    assert payload["resume_command"] is None
    assert payload["resume_argv"] == []
    assert payload["lease"]["lease_id"]
    assert selected.token not in result.stdout
    mock_common_state.sync_sessions.assert_not_called()
    mock_common_state.resolve_session.assert_not_called()
    update.assert_not_called()
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_download_json_reports_transfer_metrics(
    mock_open_executor,
    mock_transfer_class,
    mock_common_state,
    tmp_path,
):
    target = tmp_path / "model.ckpt"
    selected = _prepare_json_session(mock_common_state)
    mock_transfer_class.return_value.download.return_value = TransferResult(
        str(target),
        4096,
        "digest",
        1024,
        2,
    )

    result = runner.invoke(
        app,
        [
            "download",
            "-s",
            selected.name,
            "--json",
            "content/model.ckpt",
            str(target),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _json_payload(result)
    assert payload["direction"] == "download"
    assert payload["completed_bytes"] == 4096
    assert payload["resumed_from"] == 1024
    assert payload["retry_count"] == 2
    assert payload["sha256"] == "digest"
    assert payload["mib_per_second"] is not None
    assert payload["eta_seconds"] == 0.0
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_upload_interrupt_returns_130_and_exact_resume(
    mock_open_executor,
    mock_transfer_class,
    mock_common_state,
    tmp_path,
):
    from colab_cli.transfer import TransferProgress

    source = tmp_path / "repo bundle.bin"
    source.write_bytes(b"abcdefgh")
    selected = _prepare_json_session(mock_common_state)

    def interrupted(*_args, **_kwargs):
        progress = mock_transfer_class.call_args.kwargs["progress"]
        progress(
            TransferProgress(
                direction="upload",
                completed=4,
                total=8,
                resumed_from=0,
                retry_count=1,
                sha256="partial-digest",
                partial_path="content/repo.part",
            )
        )
        raise KeyboardInterrupt

    mock_transfer_class.return_value.upload.side_effect = interrupted

    result = runner.invoke(
        app,
        [
            "upload",
            "-s",
            selected.name,
            "--json",
            "--chunk-size-mib",
            "0.25",
            str(source),
            "content/repo bundle.bin",
        ],
    )

    assert result.exit_code == 130, result.output
    payload = _json_payload(result)
    assert payload["status"] == "interrupted"
    assert payload["error"]["code"] == ("TRANSFER_INTERRUPTED")
    assert payload["completed_bytes"] == 4
    assert payload["total_bytes"] == 8
    assert payload["partial_path"] == ("content/repo.part")
    assert payload["resume_argv"][:3] == [
        "colab",
        "--auth",
        "oauth2",
    ]
    assert "--resume" in payload["resume_argv"]
    assert "--json" in payload["resume_argv"]
    assert payload["resume_argv"][-2:] == [
        str(source.resolve()),
        "content/repo bundle.bin",
    ]
    assert payload["resume_command"]
    history = mock_common_state.history.log_event.call_args.args[2]
    assert history["state"] == "interrupted"
    assert history["completed_bytes"] == 4
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.ContentsClient")
@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_primary_error_survives_cleanup_and_is_redacted(
    mock_open_executor,
    mock_transfer_class,
    mock_contents,
    mock_common_state,
    tmp_path,
):
    source = tmp_path / "repo.bundle"
    source.write_bytes(b"bundle")
    selected = _prepare_json_session(mock_common_state)
    mock_transfer_class.return_value.upload.side_effect = RuntimeError(
        f"request failed token={selected.token}"
    )
    mock_contents.return_value.close.side_effect = OSError("contents cleanup failed")
    mock_open_executor.return_value.close.side_effect = OSError(
        "executor cleanup failed"
    )

    result = runner.invoke(
        app,
        [
            "upload",
            "-s",
            selected.name,
            "--json",
            str(source),
            "content/repo.bundle",
        ],
    )

    assert result.exit_code == 1
    payload = _json_payload(result)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == ("TRANSFER_FAILED")
    assert selected.token not in payload["error"]["message"]
    assert "request failed" in payload["error"]["message"]
    assert any("contents cleanup failed" in item for item in payload["warnings"])
    assert any("executor cleanup failed" in item for item in payload["warnings"])
    history = mock_common_state.history.log_event.call_args.args[2]
    assert history["state"] == "failed"
    assert selected.token not in history["error"]
    assert "<redacted>" in history["error"]


@patch("colab_cli.commands.files.ContentsClient")
@patch("colab_cli.commands.files.open_remote_executor")
def test_held_upload_lease_fails_before_remote_work(
    mock_open_executor,
    mock_contents,
    mock_common_state,
    tmp_path,
):
    from colab_cli.transfer_lease import (
        TransferLease,
    )

    source = tmp_path / "repo.bundle"
    source.write_bytes(b"bundle")
    selected = _prepare_json_session(mock_common_state)
    held = TransferLease.for_upload(
        endpoint=selected.endpoint,
        local_path=source,
        remote_path="content/repo.bundle",
    )
    held.acquire()
    try:
        result = runner.invoke(
            app,
            [
                "upload",
                "-s",
                selected.name,
                "--json",
                str(source),
                "content/repo.bundle",
            ],
        )
    finally:
        held.release()

    assert result.exit_code == 1
    payload = _json_payload(result)
    assert payload["status"] == "busy"
    assert payload["error"]["code"] == ("TRANSFER_TARGET_BUSY")
    assert payload["error"]["retryable"] is True
    mock_open_executor.assert_not_called()
    mock_contents.assert_not_called()


def test_missing_upload_source_json_precedes_session_access(
    mock_common_state,
    tmp_path,
):
    missing = tmp_path / "missing.bin"

    result = runner.invoke(
        app,
        [
            "upload",
            "-s",
            "s1",
            "--json",
            str(missing),
            "content/missing.bin",
        ],
    )

    assert result.exit_code == 1
    payload = _json_payload(result)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == ("TRANSFER_FILE_NOT_FOUND")
    mock_common_state.store.list.assert_not_called()
    mock_common_state.resolve_session.assert_not_called()


@pytest.mark.parametrize(
    "command,args",
    [
        (
            "upload",
            [
                "source.bin",
                "",
            ],
        ),
        (
            "download",
            [
                "",
                "target.bin",
            ],
        ),
    ],
)
def test_invalid_remote_path_json_is_structured(
    command,
    args,
    mock_common_state,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    if command == "upload":
        (tmp_path / "source.bin").write_bytes(b"x")

    result = runner.invoke(
        app,
        [
            command,
            "--json",
            *args,
        ],
    )

    assert result.exit_code == 2
    payload = _json_payload(result)
    assert payload["error"]["code"] == ("TRANSFER_INVALID_PATH")
    mock_common_state.store.list.assert_not_called()


@patch("colab_cli.commands.files.TransferLease.for_upload")
@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_lease_cleanup_failure_does_not_replace_primary_error(
    mock_open_executor,
    mock_transfer_class,
    mock_lease_factory,
    mock_common_state,
    tmp_path,
):
    source = tmp_path / "repo.bundle"
    source.write_bytes(b"bundle")
    selected = _prepare_json_session(mock_common_state)
    lease = MagicMock()
    lease.lease_id = "lease-id"
    lease.lock_key = "lock-key"
    lease.stale_reclaimed = False
    lease.stale_reclaimed_from = None
    lease.cleanup_errors = []
    lease.release.side_effect = RuntimeError("lease cleanup failed")
    mock_lease_factory.return_value = lease
    mock_transfer_class.return_value.upload.side_effect = RuntimeError(
        "primary transfer failure"
    )

    result = runner.invoke(
        app,
        [
            "upload",
            "-s",
            selected.name,
            "--json",
            str(source),
            "content/repo.bundle",
        ],
    )

    assert result.exit_code == 1
    payload = _json_payload(result)
    assert "primary transfer failure" in (payload["error"]["message"])
    assert any("lease cleanup failed" in warning for warning in payload["warnings"])
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_existing_target_has_stable_error_code(
    mock_open_executor,
    mock_transfer_class,
    mock_common_state,
    tmp_path,
):
    source = tmp_path / "repo.bundle"
    source.write_bytes(b"bundle")
    selected = _prepare_json_session(mock_common_state)
    from colab_cli.remote import RemoteExecutionError

    mock_transfer_class.return_value.upload.side_effect = RemoteExecutionError(
        "FileExistsError",
        "content/repo.bundle",
    )

    result = runner.invoke(
        app,
        [
            "upload",
            "-s",
            selected.name,
            "--json",
            "--no-overwrite",
            str(source),
            "content/repo.bundle",
        ],
    )

    assert result.exit_code == 1
    payload = _json_payload(result)
    assert payload["error"]["code"] == ("TRANSFER_TARGET_EXISTS")
    assert payload["error"]["retryable"] is False
