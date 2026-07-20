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

import jupyter_kernel_client
import pytest

from colab_cli.runtime import ColabRuntime


@patch("colab_cli.runtime.jupyter_kernel_client.KernelClient")
def test_colab_runtime_kernel_client(mock_kc_cls):
    mock_kc = mock_kc_cls.return_value

    runtime = ColabRuntime("http://url", "token123")

    assert runtime._kernel_client is None

    kc = runtime.kernel_client

    mock_kc_cls.assert_called_once_with(
        server_url="http://url",
        token="token123",
        kernel_id=None,
        client_kwargs={
            "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
            "extra_params": {"colab-runtime-proxy-token": "token123"},
        },
        headers={
            "X-Colab-Client-Agent": "colab-cli",
            "X-Colab-Runtime-Proxy-Token": "token123",
        },
    )
    mock_kc.start.assert_called_once()
    assert kc == mock_kc


@patch("colab_cli.runtime.time.sleep")
@patch("colab_cli.runtime.jupyter_kernel_client.KernelClient")
def test_colab_runtime_retries_when_websocket_is_not_ready(mock_kc_cls, mock_sleep):
    first = MagicMock()
    first.id = "kernel-123"
    first._manager.client.channels_running = False

    second = MagicMock()
    second.id = "kernel-123"
    second._manager.client.channels_running = True

    mock_kc_cls.side_effect = [first, second]
    on_kernel_started = MagicMock()
    runtime = ColabRuntime(
        "http://url",
        "token123",
        on_kernel_started=on_kernel_started,
    )

    assert runtime.kernel_client is second
    assert mock_kc_cls.call_count == 2
    assert mock_kc_cls.call_args_list[1].kwargs["kernel_id"] == "kernel-123"
    first.stop.assert_called_once_with(shutdown_kernel=False)
    mock_sleep.assert_called_once_with(2)
    on_kernel_started.assert_called_once_with("kernel-123")


@patch("colab_cli.runtime.jupyter_kernel_client.KernelClient")
def test_colab_runtime_discards_cached_client_when_session_callback_fails(mock_kc_cls):
    first = MagicMock()
    first.id = "kernel-123"
    first._manager.client.channels_running = True
    first._manager.client.session.session = "session-123"

    second = MagicMock()
    second.id = "kernel-123"
    second._manager.client.channels_running = True

    mock_kc_cls.side_effect = [first, second]
    runtime = ColabRuntime(
        "http://url",
        "token123",
        on_session_started=MagicMock(side_effect=OSError("state write failed")),
    )

    with pytest.raises(OSError, match="state write failed"):
        runtime.kernel_client

    assert runtime._kernel_client is None
    first.stop.assert_called_once_with(shutdown_kernel=False)
    assert runtime.kernel_client is second


def test_colab_runtime_execute_code():
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    # Test empty reply
    mock_kc.execute.return_value = {}
    assert runtime.execute_code("print(1)") == []

    # Test normal reply
    mock_kc.execute.return_value = {"outputs": [{"text": "1\n"}]}
    assert runtime.execute_code("print(1)") == [{"text": "1\n"}]

    # Test error status without error output
    mock_kc.execute.return_value = {
        "status": "error",
        "ename": "ValueError",
        "evalue": "bad",
        "outputs": [{"text": "partial"}],
    }
    outputs = runtime.execute_code("raise ValueError")
    assert len(outputs) == 2
    assert outputs[0] == {"text": "partial"}
    assert outputs[1] == {
        "output_type": "error",
        "ename": "ValueError",
        "evalue": "bad",
        "traceback": [],
    }


def test_colab_runtime_execute_code_default_no_timeout():
    """By default, execute_code should NOT pass a timeout (relies on jupyter
    kernel client default), preserving existing behavior for fast / streaming
    workloads."""
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    mock_kc.execute.return_value = {"outputs": []}
    runtime.execute_code("print(1)")

    _, kwargs = mock_kc.execute.call_args
    assert "timeout" not in kwargs


def test_colab_runtime_execute_code_with_timeout():
    """When a timeout is supplied, it must be forwarded to kernel_client.execute."""
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    mock_kc.execute.return_value = {"outputs": []}
    runtime.execute_code("print(1)", timeout=600)

    _, kwargs = mock_kc.execute.call_args
    assert kwargs.get("timeout") == 600


def test_colab_runtime_execute_interactive_with_timeout():
    """timeout must also be plumbed through the execute_interactive branch
    (used when an output_hook is supplied)."""
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    mock_kc.execute_interactive.return_value = {"content": {"status": "ok"}}
    runtime.execute_code("print(1)", output_hook=lambda o: None, timeout=600)

    _, kwargs = mock_kc.execute_interactive.call_args
    assert kwargs.get("timeout") == 600


def test_colab_runtime_stop():
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    runtime._kernel_client = mock_kc

    runtime.stop()
    mock_kc._manager.client.stop_channels.assert_called_once()


def test_colab_runtime_stop_exception(caplog):
    runtime = ColabRuntime("http://url", "token123")
    mock_kc = MagicMock()
    mock_kc._manager.client.stop_channels.side_effect = Exception("Stop failed")
    runtime._kernel_client = mock_kc

    runtime.stop()  # Should not raise
    assert "Error stopping kernel client" in caplog.text


def _stdin_runtime(history=None):
    runtime = ColabRuntime(
        "http://url", "token", session_name="test-s", history=history
    )
    kernel_client = MagicMock()
    kernel_client._manager.client.input = MagicMock()
    runtime._kernel_client = kernel_client
    return runtime, kernel_client


def _execute_with_stdin_request(request):
    def execute(code, allow_stdin=False, stdin_hook=None, **kwargs):
        assert allow_stdin is True
        assert stdin_hook is not None
        stdin_hook(request)
        return {"outputs": []}

    return execute


def test_colab_runtime_sends_canonical_input_reply_and_logs_plain_value():
    history = MagicMock()
    runtime, kernel_client = _stdin_runtime(history)
    request = {"content": {"prompt": "Enter something: ", "password": False}}
    kernel_client.execute.side_effect = _execute_with_stdin_request(request)

    with patch("colab_cli.runtime.input", return_value="user input"):
        assert runtime.execute_code("code", allow_stdin=True) == []

    kernel_client._manager.client.input.assert_called_once_with("user input")
    history.log_event.assert_any_call(
        "test-s", "stdin_request", {"prompt": "Enter something: ", "password": False}
    )
    history.log_event.assert_any_call(
        "test-s", "input_reply", {"value": "user input", "password": False}
    )


def test_colab_runtime_uses_getpass_and_redacts_password_history():
    history = MagicMock()
    runtime, kernel_client = _stdin_runtime(history)
    request = {"content": {"prompt": "Password: ", "password": True}}
    kernel_client.execute.side_effect = _execute_with_stdin_request(request)

    with patch("colab_cli.runtime.getpass", return_value="super-secret"):
        runtime.execute_code("code", allow_stdin=True)

    kernel_client._manager.client.input.assert_called_once_with("super-secret")
    history.log_event.assert_any_call(
        "test-s", "input_reply", {"value": "<redacted>", "password": True}
    )
    assert "super-secret" not in repr(history.log_event.call_args_list)


def test_colab_runtime_custom_stdin_hook_receives_request_and_none_becomes_empty():
    runtime, kernel_client = _stdin_runtime()
    request = {"content": {"prompt": "Continue? ", "password": False}}
    kernel_client.execute.side_effect = _execute_with_stdin_request(request)
    hook = MagicMock(return_value=None)

    runtime.execute_code("code", allow_stdin=True, stdin_hook=hook)

    hook.assert_called_once_with(request)
    kernel_client._manager.client.input.assert_called_once_with("")


def test_colab_runtime_missing_input_reply_api_fails_explicitly():
    runtime, kernel_client = _stdin_runtime()
    kernel_client._manager.client.input = None
    request = {"content": {"prompt": "Value: ", "password": False}}
    kernel_client.execute.side_effect = _execute_with_stdin_request(request)

    with patch("colab_cli.runtime.input", return_value="value"):
        with pytest.raises(RuntimeError, match="does not expose input_reply"):
            runtime.execute_code("code", allow_stdin=True)


def test_colab_runtime_rejects_invalid_timeout_before_connecting():
    runtime = ColabRuntime("http://url", "token")

    with pytest.raises(ValueError, match="finite number greater than zero"):
        runtime.execute_code("pass", timeout=0)

    assert runtime._kernel_client is None
