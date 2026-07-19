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

from unittest.mock import MagicMock, call, patch

from typer.testing import CliRunner

from colab_cli.cli import app
from colab_cli.jobs import JobTail


runner = CliRunner()


def _session(mock_common_state):
    session = MagicMock(name="session")
    mock_common_state.resolve_session.return_value = "s1"
    mock_common_state.store.get.return_value = session
    return session


@patch("colab_cli.commands.jobs.RemoteJobClient")
@patch("colab_cli.commands.jobs.open_remote_executor")
def test_submit_passes_argv_without_shell_and_closes_executor(
    open_executor, job_client, mock_common_state
):
    session = _session(mock_common_state)
    executor = open_executor.return_value
    job_client.return_value.submit.return_value = {
        "job_id": "train",
        "state": "queued",
        "job_dir": "/content/.colab-cli/jobs/train",
    }

    result = runner.invoke(
        app,
        [
            "submit",
            "-s",
            "s1",
            "--name",
            "train",
            "--cwd",
            "/content/XoFTR",
            "--env",
            "RUN_ID=r1",
            "--",
            "python",
            "-u",
            "train.py",
            "--label",
            "value with spaces",
        ],
    )

    assert result.exit_code == 0, result.output
    open_executor.assert_called_once_with(
        session, mock_common_state.store, history=mock_common_state.history
    )
    job_client.return_value.submit.assert_called_once_with(
        ["python", "-u", "train.py", "--label", "value with spaces"],
        job_id="train",
        cwd="/content/XoFTR",
        env={"RUN_ID": "r1"},
    )
    executor.close.assert_called_once_with()
    assert "train" in result.output
    mock_common_state.history.log_event.assert_called_once()


@patch("colab_cli.commands.jobs.RemoteJobClient")
@patch("colab_cli.commands.jobs.open_remote_executor")
def test_jobs_lists_remote_persisted_state(
    open_executor, job_client, mock_common_state
):
    _session(mock_common_state)
    job_client.return_value.list_jobs.return_value = [
        {"job_id": "done", "state": "succeeded", "returncode": 0},
        {"job_id": "train", "state": "running", "returncode": None},
    ]

    result = runner.invoke(app, ["jobs", "-s", "s1"])

    assert result.exit_code == 0, result.output
    assert "done\tsucceeded\t0" in result.output
    assert "train\trunning\t-" in result.output
    open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.jobs.RemoteJobClient")
@patch("colab_cli.commands.jobs.open_remote_executor")
def test_tail_prints_bytes_and_next_offset(
    open_executor, job_client, mock_common_state
):
    _session(mock_common_state)
    job_client.return_value.tail.return_value = JobTail(
        job_id="train",
        stream="stdout",
        offset=4,
        next_offset=7,
        size=7,
        eof=True,
        data=b"new",
    )

    result = runner.invoke(
        app,
        ["tail", "train", "-s", "s1", "--offset", "4", "--stream", "stdout"],
    )

    assert result.exit_code == 0, result.output
    assert "new" in result.output
    assert "next_offset=7" in result.output
    job_client.return_value.tail.assert_called_once_with(
        "train", stream="stdout", offset=4, max_bytes=65536
    )
    open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.jobs.time.sleep")
@patch("colab_cli.commands.jobs.RemoteJobClient")
@patch("colab_cli.commands.jobs.open_remote_executor")
def test_wait_reconnects_tails_logs_and_returns_remote_exit_code(
    open_executor, job_client, sleep, mock_common_state
):
    _session(mock_common_state)
    client = job_client.return_value
    client.status.side_effect = [
        {"job_id": "train", "state": "running"},
        {"job_id": "train", "state": "failed", "returncode": 7},
    ]
    client.tail.side_effect = [
        JobTail("train", "stdout", 10, 14, 14, True, b"out\n"),
        JobTail("train", "stderr", 20, 24, 24, True, b"err\n"),
        JobTail("train", "stdout", 14, 14, 14, True, b""),
        JobTail("train", "stderr", 24, 24, 24, True, b""),
    ]

    result = runner.invoke(
        app,
        [
            "wait",
            "train",
            "-s",
            "s1",
            "--poll-seconds",
            "0.01",
            "--stdout-offset",
            "10",
            "--stderr-offset",
            "20",
        ],
    )

    assert result.exit_code == 7, (result.output, result.exception)
    assert "out" in result.output
    assert "err" in result.output
    assert client.tail.call_args_list == [
        call("train", stream="stdout", offset=10, max_bytes=65536),
        call("train", stream="stderr", offset=20, max_bytes=65536),
        call("train", stream="stdout", offset=14, max_bytes=65536),
        call("train", stream="stderr", offset=24, max_bytes=65536),
    ]
    sleep.assert_called_once_with(0.01)
    open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.jobs.time.monotonic", side_effect=[10.0, 10.0])
@patch("colab_cli.commands.jobs.RemoteJobClient")
@patch("colab_cli.commands.jobs.open_remote_executor")
def test_wait_timeout_does_not_cancel_job(
    open_executor, job_client, monotonic, mock_common_state
):
    _session(mock_common_state)
    job_client.return_value.status.return_value = {
        "job_id": "train",
        "state": "running",
    }
    job_client.return_value.tail.side_effect = [
        JobTail("train", "stdout", 0, 0, 0, True, b""),
        JobTail("train", "stderr", 0, 0, 0, True, b""),
    ]

    result = runner.invoke(app, ["wait", "train", "-s", "s1", "--timeout", "0"])

    assert result.exit_code == 124
    job_client.return_value.cancel.assert_not_called()
    open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.jobs.RemoteJobClient")
@patch("colab_cli.commands.jobs.open_remote_executor")
def test_cancel_uses_bounded_grace_and_closes_executor(
    open_executor, job_client, mock_common_state
):
    _session(mock_common_state)
    job_client.return_value.cancel.return_value = {
        "job_id": "train",
        "state": "cancelled",
    }

    result = runner.invoke(
        app,
        ["cancel", "train", "-s", "s1", "--grace-seconds", "3"],
    )

    assert result.exit_code == 0, result.output
    job_client.return_value.cancel.assert_called_once_with("train", grace_seconds=3.0)
    open_executor.return_value.close.assert_called_once_with()


@patch("colab_cli.commands.jobs.RemoteJobClient")
@patch("colab_cli.commands.jobs.open_remote_executor")
def test_submit_rejects_malformed_environment_before_remote_call(
    open_executor, job_client, mock_common_state
):
    _session(mock_common_state)

    result = runner.invoke(
        app,
        ["submit", "-s", "s1", "--env", "MISSING_EQUALS", "--", "python"],
    )

    assert result.exit_code == 2
    job_client.return_value.submit.assert_not_called()
    open_executor.assert_not_called()
