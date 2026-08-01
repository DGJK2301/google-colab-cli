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

import re
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner
from colab_cli.cli import app
from colab_cli.state import SessionState

runner = CliRunner()


@pytest.fixture
def mock_session():
    return SessionState(
        name="test-session",
        token="test-token",
        url="https://test.url",
        endpoint="e1",
    )


@patch("colab_cli.commands.automation.ColabRuntime")
@patch("colab_cli.common.state")
def test_cli_auth(mock_state, mock_runtime_class, mock_session):
    mock_state.store.get.return_value = mock_session
    mock_state.resolve_session.return_value = "test-session"

    mock_runtime = mock_runtime_class.return_value
    mock_runtime.execute_code.return_value = [{"text": "Success"}]

    result = runner.invoke(app, ["auth", "-s", "test-session"])
    assert result.exit_code == 0

    assert mock_session.last_execution[0] == "automation:auth"
    assert mock_session.last_execution[1] is None
    assert mock_session.last_execution[2] is not None
    mock_state.store.add.assert_called_with(mock_session)

    # Verify ColabRuntime was invoked with the correct code
    mock_runtime.execute_code.assert_called_once()
    called_code = mock_runtime.execute_code.call_args[0][0]

    assert "os.environ['USE_AUTH_EPHEM'] = '0'" in called_code
    assert "auth.authenticate_user()" in called_code


@patch("colab_cli.commands.automation.ColabRuntime")
@patch("colab_cli.common.state")
def test_cli_install(mock_state, mock_runtime_class, mock_session):
    mock_state.store.get.return_value = mock_session
    mock_state.resolve_session.return_value = "test-session"

    mock_runtime = mock_runtime_class.return_value
    mock_runtime.execute_code.return_value = [{"text": "Installed"}]

    result = runner.invoke(app, ["install", "-s", "test-session", "pandas", "numpy"])
    assert result.exit_code == 0
    assert mock_session.last_execution[0] == "automation:install"
    assert mock_session.last_execution[2] is not None
    mock_state.store.add.assert_called_with(mock_session)

    mock_runtime.execute_code.assert_called_once()
    called_code = mock_runtime.execute_code.call_args[0][0]

    assert "subprocess" in called_code
    assert "pip" in called_code
    assert "pandas" in called_code
    assert "numpy" in called_code


@patch("colab_cli.commands.automation.ContentsClient")
@patch("colab_cli.commands.automation.ColabRuntime")
@patch("colab_cli.common.state")
def test_cli_install_closes_requirement_upload_client(
    mock_state,
    mock_runtime_class,
    mock_contents_class,
    mock_session,
    tmp_path,
):
    requirement = tmp_path / "requirements.txt"
    requirement.write_text("numpy\n", encoding="utf-8")
    mock_state.store.get.return_value = mock_session
    mock_state.resolve_session.return_value = "test-session"
    mock_runtime_class.return_value.execute_code.return_value = [{"text": "Installed"}]

    result = runner.invoke(
        app,
        [
            "install",
            "-s",
            "test-session",
            "-r",
            str(requirement),
        ],
    )

    assert result.exit_code == 0, result.output
    client = mock_contents_class.return_value.__enter__.return_value
    client.upload.assert_called_once_with(
        str(requirement),
        "content/requirements.txt",
    )
    mock_contents_class.return_value.__exit__.assert_called_once()


@patch("colab_cli.commands.automation.ColabRuntime")
@patch("colab_cli.common.state")
def test_cli_drivemount(mock_state, mock_runtime_class, mock_session):
    mock_session.kernel_id = "kernel-1"
    mock_session.session_id = "session-1"
    mock_state.store.get.return_value = mock_session
    mock_state.resolve_session.return_value = "test-session"

    mock_runtime = mock_runtime_class.return_value
    mock_runtime.execute_code.return_value = [{"text": "Mounted"}]

    result = runner.invoke(app, ["drivemount", "-s", "test-session", "/foo/bar"])
    assert result.exit_code == 0

    # Verify ColabRuntime was invoked with the correct code
    mock_runtime.execute_code.assert_called_once()
    called_code = mock_runtime.execute_code.call_args[0][0]

    assert "drive.mount('/foo/bar'" in called_code
    assert mock_runtime.colab_request_hook is not None
    # Drivemount waits for the user to OAuth in their browser; the kernel
    # goes silent during that wait and the default 10s execute() timeout
    # would raise TimeoutError mid-flow. Insist on a generous timeout
    # (>= 5 minutes) being forwarded to runtime.execute_code.
    _, kwargs = mock_runtime.execute_code.call_args
    assert kwargs.get("timeout") is not None and kwargs["timeout"] >= 300

    remote_timeout = re.search(r"timeout_ms=(\d+)", called_code)
    assert remote_timeout is not None
    assert kwargs["timeout"] >= int(remote_timeout.group(1)) / 1000 + 30

    _, runtime_kwargs = mock_runtime_class.call_args
    assert runtime_kwargs["kernel_id"] == "kernel-1"
    assert runtime_kwargs["session_id"] == "session-1"

    runtime_kwargs["on_kernel_started"]("kernel-2")
    runtime_kwargs["on_session_started"]("session-2")
    assert mock_session.kernel_id == "kernel-2"
    assert mock_session.session_id == "session-2"


@patch("colab_cli.commands.automation.get_credentials")
@patch("colab_cli.commands.automation.ColabRuntime")
@patch("colab_cli.common.state")
def test_drive_auth_does_not_block_the_websocket_callback(
    mock_state,
    mock_runtime_class,
    mock_get_credentials,
    mock_session,
):
    mock_state.store.get.return_value = mock_session
    mock_state.resolve_session.return_value = "test-session"
    mock_state.client.colab_domain = "https://colab.research.google.com"
    mock_runtime = mock_runtime_class.return_value
    mock_runtime.execute_code.return_value = [{"text": "Mounted"}]

    result = runner.invoke(app, ["drivemount", "-s", "test-session"])
    assert result.exit_code == 0, result.output

    gate = Event()
    responses = iter(
        [
            SimpleNamespace(status_code=200, text='{}\n{"token":"token"}'),
            SimpleNamespace(status_code=200, text='{}\n{"success":true}'),
            SimpleNamespace(status_code=200, text='{}\n{"success":true}'),
        ]
    )

    def request(*_args, **_kwargs):
        gate.wait(timeout=2)
        return next(responses)

    mock_get_credentials.return_value.request.side_effect = request
    wsclient = MagicMock()
    wsclient.session.msg.return_value = {"header": {}, "content": {}}
    reply_sent = Event()
    wsclient.stdin_channel.send.side_effect = lambda _reply: reply_sent.set()
    request_message = {
        "header": {"msg_type": "colab_request"},
        "content": {"request": {"authType": "dfs_ephemeral"}},
        "metadata": {"colab_msg_id": 42},
    }

    callback_returned = Event()

    def invoke_hook():
        mock_runtime.colab_request_hook(request_message, wsclient)
        callback_returned.set()

    caller = Thread(target=invoke_hook)
    caller.start()
    returned_without_waiting_for_auth = callback_returned.wait(timeout=0.2)
    gate.set()
    caller.join(timeout=2)
    reply_sent.wait(timeout=2)

    assert returned_without_waiting_for_auth
    assert reply_sent.is_set()


@patch("colab_cli.commands.automation._read_line_from_controlling_tty")
@patch("colab_cli.commands.automation.webbrowser.open")
@patch("colab_cli.commands.automation.get_credentials")
def test_drive_auth_opens_the_consent_url_and_preserves_a_manual_fallback(
    mock_get_credentials,
    mock_browser_open,
    mock_read_line,
    mock_session,
    capsys,
):
    from colab_cli.commands.automation import _handle_drivefs_auth

    consent_url = "https://accounts.google.test/drive-consent"
    mock_get_credentials.return_value.request.side_effect = [
        SimpleNamespace(status_code=200, text='{}\n{"token":"token"}'),
        SimpleNamespace(
            status_code=200,
            text=(
                f'{{}}\n{{"success":false,"unauthorized_redirect_uri":"{consent_url}"}}'
            ),
        ),
        SimpleNamespace(status_code=200, text='{}\n{"success":true}'),
    ]
    state = MagicMock()
    state.client.colab_domain = "https://colab.research.google.com"
    wsclient = MagicMock()
    wsclient.session.msg.side_effect = lambda _kind, content: {
        "header": {},
        "content": content,
    }
    request_message = {
        "header": {"msg_type": "colab_request"},
        "content": {"request": {"authType": "dfs_ephemeral"}},
        "metadata": {"colab_msg_id": 42},
    }

    _handle_drivefs_auth(state, mock_session, request_message, wsclient)

    mock_browser_open.assert_called_once_with(consent_url, new=2)
    mock_read_line.assert_called_once_with()
    assert consent_url in capsys.readouterr().out
    assert consent_url not in repr(state.history.log_event.call_args_list)
    reply = wsclient.stdin_channel.send.call_args.args[0]
    assert reply["content"]["value"] == {
        "type": "colab_reply",
        "colab_msg_id": 42,
    }


@patch("colab_cli.commands.automation.get_credentials")
def test_drive_auth_failure_replies_to_the_kernel(
    mock_get_credentials,
    mock_session,
):
    from colab_cli.commands.automation import _handle_drivefs_auth

    mock_get_credentials.return_value.request.side_effect = [
        SimpleNamespace(status_code=200, text='{}\n{"token":"token"}'),
        SimpleNamespace(status_code=200, text='{}\n{"success":true}'),
        SimpleNamespace(status_code=200, text='{}\n{"success":false}'),
    ]
    state = MagicMock()
    state.client.colab_domain = "https://colab.research.google.com"
    wsclient = MagicMock()
    wsclient.session.msg.side_effect = lambda _kind, content: {
        "header": {},
        "content": content,
    }
    request_message = {
        "header": {"msg_type": "colab_request"},
        "content": {"request": {"authType": "dfs_ephemeral"}},
        "metadata": {"colab_msg_id": 42},
    }

    _handle_drivefs_auth(state, mock_session, request_message, wsclient)

    reply = wsclient.stdin_channel.send.call_args.args[0]
    value = reply["content"]["value"]
    assert value["colab_msg_id"] == 42
    assert value["type"] == "colab_reply"
    assert value["error"].startswith("RuntimeError:")
    state.history.log_event.assert_any_call(
        mock_session.name,
        "drive_auth_error",
        {"error_type": "RuntimeError"},
    )


@patch("colab_cli.commands.automation.get_credentials")
def test_drive_auth_history_failure_does_not_block_kernel_reply(
    mock_get_credentials,
    mock_session,
):
    from colab_cli.commands.automation import _handle_drivefs_auth

    mock_get_credentials.return_value.request.side_effect = [
        SimpleNamespace(status_code=200, text='{}\n{"token":"token"}'),
        SimpleNamespace(status_code=200, text='{}\n{"success":true}'),
        SimpleNamespace(status_code=200, text='{}\n{"success":true}'),
    ]
    state = MagicMock()
    state.client.colab_domain = "https://colab.research.google.com"
    state.history.log_event.side_effect = OSError("history unavailable")
    wsclient = MagicMock()
    wsclient.session.msg.side_effect = lambda _kind, content: {
        "header": {},
        "content": content,
    }
    request_message = {
        "header": {"msg_type": "colab_request"},
        "content": {"request": {"authType": "dfs_ephemeral"}},
        "metadata": {"colab_msg_id": 42},
    }

    _handle_drivefs_auth(state, mock_session, request_message, wsclient)

    reply = wsclient.stdin_channel.send.call_args.args[0]
    assert reply["content"]["value"] == {
        "type": "colab_reply",
        "colab_msg_id": 42,
    }


@patch("colab_cli.commands.automation.ColabRuntime")
@patch("colab_cli.common.state")
def test_cli_auth_uses_long_timeout(mock_state, mock_runtime_class, mock_session):
    """`colab auth` walks the user through a paste-the-code flow that
    routinely takes >10s, so it must pass a generous timeout to
    runtime.execute_code or the call will TimeoutError mid-flow."""
    mock_state.store.get.return_value = mock_session
    mock_state.resolve_session.return_value = "test-session"

    mock_runtime = mock_runtime_class.return_value
    mock_runtime.execute_code.return_value = [{"text": "Authenticated"}]

    result = runner.invoke(app, ["auth", "-s", "test-session"])
    assert result.exit_code == 0

    _, kwargs = mock_runtime.execute_code.call_args
    assert kwargs.get("timeout") is not None and kwargs["timeout"] >= 300


def test_read_line_from_controlling_tty_uses_dev_tty():
    """On POSIX-like environments with /dev/tty, the helper reads from it."""
    from colab_cli.commands.automation import _read_line_from_controlling_tty
    from unittest.mock import mock_open

    with patch("colab_cli.commands.automation.open", mock_open(read_data="ok\n")):
        assert _read_line_from_controlling_tty() == "ok\n"


def test_read_line_from_controlling_tty_falls_back_to_stdin():
    """When /dev/tty is unavailable (Windows), the helper falls back to stdin."""
    import io
    from colab_cli.commands.automation import _read_line_from_controlling_tty

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/dev/tty":
            raise OSError("No /dev/tty on Windows")
        return real_open(path, *args, **kwargs)

    with (
        patch("colab_cli.commands.automation.open", side_effect=fake_open),
        patch("colab_cli.commands.automation.sys.stdin", io.StringIO("entered\n")),
    ):
        assert _read_line_from_controlling_tty() == "entered\n"
