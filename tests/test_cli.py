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

import time
from unittest.mock import MagicMock, patch

import pytest
from click import unstyle
from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.client import (
    Assignment,
    ColabRequestError,
    PostAssignmentResponse,
)
from colab_cli.transfer import TransferResult

runner = CliRunner()


@pytest.fixture
def mock_client(mock_common_state):
    return mock_common_state.client


@pytest.fixture
def mock_store(mock_common_state):
    mock_common_state.store.get.return_value = None
    return mock_common_state.store


@pytest.fixture
def mock_history(mock_common_state):
    return mock_common_state.history


def test_cli_new_tpu(mock_client, mock_store):
    mock_res = MagicMock()
    mock_res.__class__ = PostAssignmentResponse
    mock_res.runtime_proxy_info.token = "t1"
    mock_res.runtime_proxy_info.url = "u1"
    mock_res.endpoint = "e1"
    mock_client.assign.return_value = mock_res

    result = runner.invoke(app, ["new", "-s", "my-session", "--tpu", "v5e1"])
    assert result.exit_code == 0

    added_state = mock_store.add.call_args[0][0]
    assert added_state.name == "my-session"
    assert added_state.variant == "TPU"
    assert added_state.accelerator == "V5E1"


def test_cli_new_gpu(mock_client, mock_store):
    mock_res = MagicMock()
    mock_res.__class__ = Assignment
    mock_res.runtime_proxy_token = "t2"
    mock_res.endpoint = "e2"
    del mock_res.runtime_proxy_info
    mock_client.assign.return_value = mock_res

    result = runner.invoke(app, ["new", "-s", "gpu-sess", "--gpu", "A100"])
    assert result.exit_code == 0

    added_state = mock_store.add.call_args[0][0]
    assert added_state.name == "gpu-sess"
    assert added_state.variant == "GPU"
    assert added_state.accelerator == "A100"
    assert added_state.token == "t2"


@pytest.mark.parametrize(
    "gpu_flag,expected_acc",
    [
        ("H100", "H100"),
        ("l4", "L4"),
        ("t4", "T4"),
        ("g4", "G4"),
    ],
)
def test_cli_new_gpu_variants(mock_client, mock_store, gpu_flag, expected_acc):
    mock_res = MagicMock()
    mock_res.__class__ = PostAssignmentResponse
    mock_res.runtime_proxy_info.token = "t1"
    mock_res.runtime_proxy_info.url = "u1"
    mock_res.endpoint = "e1"
    mock_client.assign.return_value = mock_res

    result = runner.invoke(app, ["new", "-s", "s", "--gpu", gpu_flag])
    assert result.exit_code == 0

    added_state = mock_store.add.call_args[0][0]
    assert added_state.accelerator == expected_acc


def test_cli_sessions_unified_format(mock_client, mock_common_state):
    """`sessions` should lead each line with the local name when known:
    `[name] endpoint | Hardware: X | Variant: Y`.
    """
    mock_assignment = MagicMock()
    mock_assignment.endpoint = "e1"
    mock_assignment.variant.name = "GPU"
    mock_assignment.accelerator.value = "T4"

    mock_session_state = MagicMock()
    mock_session_state.name = "s1"
    mock_session_state.endpoint = "e1"
    mock_session_state.running = None

    mock_common_state.sync_sessions.return_value = (
        {"s1": mock_session_state},
        [mock_assignment],
    )

    result = runner.invoke(app, ["sessions"])
    assert result.exit_code == 0
    assert "[s1] e1 | Hardware: T4 | Variant: GPU" in result.output


def test_cli_sessions_orphaned_assignment_marked(mock_client, mock_common_state):
    """Server-side assignments without a local session should be marked `[?]`."""
    mock_assignment = MagicMock()
    mock_assignment.endpoint = "orphan-ep"
    mock_assignment.variant.name = "DEFAULT"
    mock_assignment.accelerator.value = "NONE"

    mock_common_state.sync_sessions.return_value = ({}, [mock_assignment])

    result = runner.invoke(app, ["sessions"])
    assert result.exit_code == 0
    # CPU is the alias for accelerator NONE
    assert "[?] orphan-ep | Hardware: CPU | Variant: DEFAULT" in result.output


def test_cli_sessions_no_assignments(mock_client, mock_common_state):
    mock_common_state.sync_sessions.return_value = ({}, [])
    result = runner.invoke(app, ["sessions"])
    assert result.exit_code == 0
    assert "No active sessions found on server." in result.output


def test_cli_status(mock_store, mock_common_state):
    mock_session_state = MagicMock()
    mock_session_state.name = "s1"
    mock_session_state.endpoint = "e1"
    mock_session_state.accelerator = "NONE"
    mock_session_state.variant = "DEFAULT"
    mock_session_state.running = None
    mock_session_state.last_execution = (
        "my_notebook.ipynb",
        "cell_1",
        "2023-10-27 12:00:00",
    )
    mock_store.get.return_value = mock_session_state

    mock_common_state.sync_sessions.return_value = ({"s1": mock_session_state}, [])

    # Test with explicit session: uses unified format including endpoint and Status
    result = runner.invoke(app, ["status", "-s", "s1"])
    assert result.exit_code == 0
    assert "[s1] e1 | Hardware: CPU | Variant: DEFAULT | Status: IDLE" in result.output
    assert (
        "Last Execution: my_notebook.ipynb | Cell: cell_1 at 2023-10-27 12:00:00"
        in result.output
    )
    mock_store.get.assert_called_with("s1")

    # Test with missing session
    mock_store.get.return_value = None
    result = runner.invoke(app, ["status", "-s", "missing"])
    assert result.exit_code == 0
    assert "Session 'missing' not found" in result.output

    # Test list all sessions: same unified format
    mock_store.get.return_value = mock_session_state
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "[s1] e1 | Hardware: CPU | Variant: DEFAULT | Status: IDLE" in result.output

    # Test without execution metadata
    mock_session_state.last_execution = None
    mock_store.get.return_value = mock_session_state
    result = runner.invoke(app, ["status", "-s", "s1"])
    assert result.exit_code == 0
    assert "Last Execution" not in result.output


def test_cli_status_running_shows_busy(mock_store, mock_common_state):
    mock_session_state = MagicMock()
    mock_session_state.name = "s1"
    mock_session_state.endpoint = "e1"
    mock_session_state.accelerator = "T4"
    mock_session_state.variant = "GPU"
    mock_session_state.running = "exec.py"
    mock_session_state.last_execution = None
    mock_store.get.return_value = mock_session_state
    mock_common_state.sync_sessions.return_value = ({"s1": mock_session_state}, [])

    result = runner.invoke(app, ["status", "-s", "s1"])
    assert result.exit_code == 0
    assert (
        "[s1] e1 | Hardware: T4 | Variant: GPU | Status: BUSY (exec.py)"
        in result.output
    )


def test_cli_session_resolution(mock_store, mock_common_state):
    mock_session_state = MagicMock()
    mock_session_state.name = "unique-session"
    mock_session_state.endpoint = "e1"
    mock_session_state.url = "http://url"
    mock_session_state.token = "token"
    mock_session_state.kernel_id = None

    # Setup for resolve_session
    mock_common_state.resolve_session.return_value = "unique-session"
    mock_store.get.return_value = mock_session_state

    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0
    mock_store.remove.assert_called_with("unique-session")


def test_cli_stop(mock_client, mock_store, mock_common_state):
    mock_session_state = MagicMock()
    mock_session_state.endpoint = "e1"
    mock_session_state.name = "s1"
    mock_session_state.url = "http://url"
    mock_session_state.token = "token"
    mock_session_state.kernel_id = None
    mock_store.get.return_value = mock_session_state

    mock_common_state.resolve_session.return_value = "s1"
    result = runner.invoke(app, ["stop", "-s", "s1"])
    assert result.exit_code == 0

    mock_client.unassign.assert_called_with("e1")
    mock_store.remove.assert_called_with("s1")


def test_cli_stop_orphan_by_exact_endpoint(mock_client, mock_store, mock_history):
    assignment = MagicMock(endpoint="orphan-ep")
    mock_client.list_assignments.return_value = [assignment]
    mock_store.list.return_value = {}

    result = runner.invoke(app, ["stop", "--endpoint", "orphan-ep"])

    assert result.exit_code == 0
    mock_client.unassign.assert_called_once_with("orphan-ep")
    mock_history.log_event.assert_called_once_with(
        "_orphan_assignments",
        "orphan_assignment_released",
        {"endpoint": "orphan-ep"},
    )
    history_key = mock_history.log_event.call_args.args[0]
    assert not set('<>:"/\\|?*') & set(history_key)
    assert "Orphan assignment released" in result.output


def test_cli_stop_orphan_history_failure_is_best_effort(
    mock_client, mock_store, mock_history
):
    assignment = MagicMock(endpoint="orphan-ep")
    mock_client.list_assignments.return_value = [assignment]
    mock_store.list.return_value = {}

    def fail_history_write(*_args, **_kwargs):
        mock_client.unassign.assert_called_once_with("orphan-ep")
        raise OSError("history write failed")

    mock_history.log_event.side_effect = fail_history_write
    result = runner.invoke(app, ["stop", "--endpoint", "orphan-ep"])

    assert result.exit_code == 0
    mock_client.unassign.assert_called_once_with("orphan-ep")
    assert "Orphan assignment released: orphan-ep" in result.output


def test_cli_stop_endpoint_rejects_locally_tracked_assignment(mock_client, mock_store):
    assignment = MagicMock(endpoint="tracked-ep")
    mock_client.list_assignments.return_value = [assignment]
    mock_store.list.return_value = {
        "tracked": MagicMock(name="tracked", endpoint="tracked-ep")
    }

    result = runner.invoke(app, ["stop", "--endpoint", "tracked-ep"])

    assert result.exit_code == 1
    assert "colab stop -s tracked" in result.output
    mock_client.unassign.assert_not_called()


def test_cli_stop_endpoint_requires_exact_server_assignment(mock_client, mock_store):
    mock_client.list_assignments.return_value = []
    mock_store.list.return_value = {}

    result = runner.invoke(app, ["stop", "--endpoint", "missing-ep"])

    assert result.exit_code == 1
    assert "not active on the server" in result.output
    mock_client.unassign.assert_not_called()


def test_cli_stop_rejects_session_and_endpoint_together(mock_client, mock_store):
    result = runner.invoke(app, ["stop", "-s", "s1", "--endpoint", "e1"])

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_cli_sessions_prune(mock_common_state):
    mock_assignment = MagicMock()
    mock_session_state1 = MagicMock()

    mock_common_state.sync_sessions.return_value = (
        {"s1": mock_session_state1},
        [mock_assignment],
    )
    result = runner.invoke(app, ["sessions"])
    assert result.exit_code == 0


def test_cli_new_no_name(mock_client, mock_store):
    mock_res = MagicMock()
    mock_res.__class__ = PostAssignmentResponse
    mock_res.runtime_proxy_info.token = "t1"
    mock_res.runtime_proxy_info.url = "u1"
    mock_res.endpoint = "e1"
    mock_client.assign.return_value = mock_res

    result = runner.invoke(app, ["new"])
    assert result.exit_code == 0

    added_state = mock_store.add.call_args[0][0]
    assert len(added_state.name) == 6


def test_cli_new_default_is_cpu(mock_client, mock_store):
    """`colab new` with no flags must request a CPU runtime (no accelerator).
    A GPU/TPU should only be requested when --gpu or --tpu is explicitly set.
    """
    from colab_cli.client import Accelerator, Variant

    mock_res = MagicMock()
    mock_res.__class__ = PostAssignmentResponse
    mock_res.runtime_proxy_info.token = "t1"
    mock_res.runtime_proxy_info.url = "u1"
    mock_res.endpoint = "e1"
    mock_client.assign.return_value = mock_res

    result = runner.invoke(app, ["new"])
    assert result.exit_code == 0

    # The assign call must have used the DEFAULT (CPU) variant + NONE accelerator.
    _, kwargs = mock_client.assign.call_args
    assert kwargs["variant"] is Variant.DEFAULT
    assert kwargs["accelerator"] is Accelerator.NONE

    # The persisted SessionState should reflect the same.
    added_state = mock_store.add.call_args[0][0]
    assert added_state.variant == "DEFAULT"
    assert added_state.accelerator == "NONE"


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Options" in result.output
    assert "Commands" in result.output


def _extract_command_names(help_output: str) -> list[str]:
    """Parse the command list out of a Typer/Click help output rendered
    inside a rich box (either rounded ╭/╰ or square ┌/└ styles — Rich picks
    the latter on some Windows environments). Returns names in order.
    """
    lines = unstyle(help_output).splitlines()
    in_commands = False
    names = []
    for line in lines:
        if not in_commands:
            if "Commands" in line and ("─" in line or "-" in line):
                in_commands = True
            continue
        stripped = line.strip()
        # Bottom border of the commands box (rounded or square style).
        if stripped[:1] in ("╰", "└", "`"):
            break
        # Strip the left/right box borders ("│").
        inner = stripped.strip("│")
        if not inner.strip():
            continue
        # Command lines look like "│ console  Connect to raw TTY console │":
        # the name sits one space after the left border. Wrapped description
        # continuation lines are indented to the description column
        # ("│                 VM") — skip those so wrapped words aren't
        # mistaken for command names.
        if not inner.startswith(" ") or inner[1:2].isspace():
            continue
        tok = inner[1:].split()[0]
        names.append(tok)
    return names


def test_extract_command_names_handles_ansi_styling():
    help_output = (
        "\x1b[2m┌─\x1b[0m\x1b[2m Commands ─┐\x1b[0m\n"
        "\x1b[2m│\x1b[0m \x1b[1;36mexec\x1b[0m Execute code \x1b[2m│\x1b[0m\n"
        "\x1b[2m└───────────────┘\x1b[0m\n"
    )

    assert _extract_command_names(help_output) == ["exec"]


def test_cli_help_commands_sorted_alphabetically():
    """`colab --help` should list subcommands in alphabetical order so that
    users (and docs) can find them deterministically."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    names = _extract_command_names(result.output)
    assert names, f"Could not parse command names from help output:\n{result.output}"
    assert names == sorted(names), (
        f"Commands are not alphabetically sorted.\nGot:    {names}\n"
        f"Wanted: {sorted(names)}"
    )


def test_cli_help_subcommand_commands_sorted_alphabetically():
    """`colab help` (the help subcommand, no argument) should also list
    subcommands alphabetically — it shares the parent group's renderer."""
    result = runner.invoke(app, ["help"])
    assert result.exit_code == 0
    names = _extract_command_names(result.output)
    assert names, f"Could not parse command names from help output:\n{result.output}"
    assert names == sorted(names), (
        f"`colab help` commands are not alphabetically sorted.\nGot:    {names}\n"
        f"Wanted: {sorted(names)}"
    )


def test_cli_no_args():
    result = runner.invoke(app, [])
    # Typer with no_args_is_help=True might return 0 or 2 depending on version/config
    assert result.exit_code in [0, 2]


def test_cli_console(mock_store, mock_common_state):
    mock_session_state = MagicMock()
    mock_session_state.name = "s1"
    mock_session_state.token = "t1"
    mock_session_state.url = "http://test.com"
    mock_store.get.return_value = mock_session_state

    mock_common_state.resolve_session.return_value = "s1"
    with patch("colab_cli.commands.execution.connect_console") as mock_connect:
        result = runner.invoke(app, ["console", "-s", "s1"])
        assert result.exit_code == 0
        mock_connect.assert_called_once_with(mock_session_state)


@patch("colab_cli.commands.files.ContentsClient")
def test_cli_ls(mock_contents_class, mock_store, mock_common_state):
    mock_session_state = MagicMock()
    mock_store.get.return_value = mock_session_state

    mock_contents = mock_contents_class.return_value
    mock_contents.list_dir.return_value = {
        "type": "directory",
        "content": [
            {"name": "a_dir", "type": "directory"},
            {"name": "b_file", "type": "file"},
        ],
    }

    mock_common_state.resolve_session.return_value = "s1"
    result = runner.invoke(app, ["ls", "-s", "s1", "content"])
    assert result.exit_code == 0

    assert "a_dir/" in result.output
    assert "b_file" in result.output


@patch("colab_cli.commands.files.ContentsClient")
def test_cli_rm(mock_contents_class, mock_store, mock_common_state):
    mock_session_state = MagicMock()
    mock_store.get.return_value = mock_session_state

    mock_common_state.resolve_session.return_value = "s1"
    result = runner.invoke(app, ["rm", "-s", "s1", "content/file.txt"])
    assert result.exit_code == 0

    mock_contents_class.return_value.rm.assert_called_once_with("content/file.txt")
    assert "Deleted content/file.txt" in result.output


@patch("colab_cli.commands.files.os.path.isfile")
@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_cli_upload(
    mock_open_executor,
    mock_transfer_class,
    mock_isfile,
    mock_store,
    mock_common_state,
):
    mock_session_state = MagicMock()
    mock_store.get.return_value = mock_session_state
    mock_isfile.return_value = True
    mock_transfer_class.return_value.upload.return_value = TransferResult(
        "remote.txt", 5, "abc", 0
    )

    mock_common_state.resolve_session.return_value = "s1"
    result = runner.invoke(app, ["upload", "-s", "s1", "local.txt", "remote.txt"])
    assert result.exit_code == 0

    mock_transfer_class.return_value.upload.assert_called_once_with(
        "local.txt", "remote.txt", overwrite=True, resume=True
    )
    assert "Uploaded 'local.txt' to 'remote.txt'" in result.output
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
def test_cli_download(
    mock_open_executor, mock_transfer_class, mock_store, mock_common_state
):
    mock_session_state = MagicMock()
    mock_store.get.return_value = mock_session_state
    mock_transfer_class.return_value.download.return_value = TransferResult(
        "local.txt", 5, "abc", 0
    )

    mock_common_state.resolve_session.return_value = "s1"
    result = runner.invoke(app, ["download", "-s", "s1", "remote.txt", "local.txt"])
    assert result.exit_code == 0

    mock_transfer_class.return_value.download.assert_called_once_with(
        "remote.txt", "local.txt", resume=True
    )
    assert "Downloaded 'remote.txt' to 'local.txt'" in result.output
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
@patch("click.edit")
def test_cli_edit_no_changes(
    mock_edit,
    mock_open_executor,
    mock_transfer_class,
    mock_store,
    mock_common_state,
):
    mock_session_state = MagicMock()
    mock_store.get.return_value = mock_session_state

    mock_common_state.resolve_session.return_value = "s1"

    def mock_download(_, local_path, **kwargs):
        with open(local_path, "wb") as stream:
            stream.write(b"old content")
        return TransferResult(local_path, 11, "abc", 0)

    mock_transfer_class.return_value.download.side_effect = mock_download

    # Simulate editor making no changes by not modifying the file
    def mock_edit_side_effect(filename, **kwargs):
        pass

    mock_edit.side_effect = mock_edit_side_effect

    result = runner.invoke(app, ["edit", "-s", "s1", "remote.txt"])

    assert result.exit_code == 0
    mock_transfer_class.return_value.download.assert_called_once()
    mock_transfer_class.return_value.upload.assert_not_called()
    assert "No changes made to 'remote.txt'" in result.output
    mock_open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.files.FileTransfer")
@patch("colab_cli.commands.files.open_remote_executor")
@patch("click.edit")
def test_cli_edit_with_changes(
    mock_edit,
    mock_open_executor,
    mock_transfer_class,
    mock_store,
    mock_common_state,
):
    mock_session_state = MagicMock()
    mock_store.get.return_value = mock_session_state

    mock_common_state.resolve_session.return_value = "s1"

    def mock_download(_, local_path, **kwargs):
        with open(local_path, "wb") as stream:
            stream.write(b"old content")
        return TransferResult(local_path, 11, "abc", 0)

    mock_transfer_class.return_value.download.side_effect = mock_download
    mock_transfer_class.return_value.upload.return_value = TransferResult(
        "remote.txt", 22, "def", 0
    )

    # Simulate editor modifying the file
    def mock_edit_side_effect(filename, **kwargs):
        time.sleep(0.01)  # Ensure mtime differs if checking by mtime
        with open(filename, "a") as f:
            f.write("new content")

    mock_edit.side_effect = mock_edit_side_effect

    result = runner.invoke(app, ["edit", "-s", "s1", "remote.txt"])

    assert result.exit_code == 0
    mock_transfer_class.return_value.download.assert_called_once()
    mock_transfer_class.return_value.upload.assert_called_once()
    assert "Edited and uploaded 'remote.txt'" in result.output
    mock_open_executor.return_value.close.assert_called_once_with()


def _make_400_error(message="Bad Request"):
    """Build a ColabRequestError shaped like a 400 from the assign endpoint."""
    response = MagicMock()
    response.status_code = 400
    response.reason = "Bad Request"
    return ColabRequestError(message, request=MagicMock(), response=response)


def test_cli_new_400_with_gpu_shows_friendly_error(mock_client, mock_store):
    """A 400 from `assign` when a GPU was requested should surface a friendly
    message naming the accelerator and exit non-zero, NOT raise a traceback."""
    mock_client.assign.side_effect = _make_400_error()

    result = runner.invoke(app, ["new", "--gpu", "A100"])

    assert result.exit_code != 0
    # Friendly message should mention the accelerator we asked for
    assert "A100" in result.output
    # And give actionable hints
    assert "quota" in result.output.lower() or "entitle" in result.output.lower()
    # No partial state should be saved
    mock_store.add.assert_not_called()


def test_cli_new_400_with_tpu_shows_friendly_error(mock_client, mock_store):
    mock_client.assign.side_effect = _make_400_error()

    result = runner.invoke(app, ["new", "--tpu", "v5e1"])

    assert result.exit_code != 0
    assert "V5E1" in result.output
    mock_store.add.assert_not_called()


def test_cli_new_400_without_accelerator_propagates(mock_client, mock_store):
    """If a 400 happens for a default (CPU) request, we cannot blame an
    accelerator. The error should propagate so the user sees the real cause
    rather than a misleading 'no quota' message.
    """
    mock_client.assign.side_effect = _make_400_error()

    # Default `colab new` requests CPU (no --gpu, no --tpu).
    result = runner.invoke(app, ["new"])

    assert result.exit_code != 0
    # The error message should NOT pretend it was an accelerator quota issue.
    assert "quota" not in result.output.lower()


def test_cli_new_non_400_error_propagates(mock_client, mock_store):
    """Errors with non-400 status should NOT be caught by the friendly
    accelerator handler."""
    response = MagicMock()
    response.status_code = 500
    response.reason = "Internal Server Error"
    mock_client.assign.side_effect = ColabRequestError(
        "boom", request=MagicMock(), response=response
    )

    result = runner.invoke(app, ["new", "--gpu", "A100"])

    assert result.exit_code != 0
    # Should not present the 400-specific friendly text
    assert "quota" not in result.output.lower()
    mock_store.add.assert_not_called()


def _cli_assignment_error(status, reason="Error"):
    response = MagicMock(status_code=status, reason=reason)
    return ColabRequestError(
        "assignment failed", request=MagicMock(), response=response
    )


def test_cli_new_invalid_gpu_fails_before_assign(mock_client, mock_store):
    result = runner.invoke(app, ["new", "-s", "bad", "--gpu", "T44"])
    assert result.exit_code == 2
    assert "Unsupported GPU" in result.stderr
    mock_client.assign.assert_not_called()


def test_cli_new_gpu_and_tpu_fail_before_assign(mock_client, mock_store):
    result = runner.invoke(app, ["new", "-s", "bad", "--gpu", "T4", "--tpu", "v5e1"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
    mock_client.assign.assert_not_called()


def test_cli_new_http_412_has_honest_message(mock_client, mock_store):
    mock_client.assign.side_effect = _cli_assignment_error(412, "Precondition Failed")
    result = runner.invoke(app, ["new", "-s", "gpu", "--gpu", "T4"])
    assert result.exit_code == 1
    assert "HTTP 412" in result.stderr
    assert "does not reliably mean too many" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_new_http_503_is_retryable_capacity_message(mock_client, mock_store):
    mock_client.assign.side_effect = _cli_assignment_error(503, "Service Unavailable")
    result = runner.invoke(app, ["new", "-s", "gpu", "--gpu", "G4"])
    assert result.exit_code == 1
    assert "temporarily unavailable" in result.stderr
    assert "bounded backoff" in result.stderr


def test_cli_auth_default_matches_public_oauth_contract():
    import inspect

    from colab_cli.auth import AuthProvider
    from colab_cli.cli import callback

    default = inspect.signature(callback).parameters["auth"].default
    assert default is AuthProvider.OAUTH2
