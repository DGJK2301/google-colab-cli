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

from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from colab_cli._jupyter_compat import (
    _DeadlineAwareMessageEvent,
    guard_interactive_timeout,
)


def _client(*, iopub_ready=False, stdin_ready=False):
    return SimpleNamespace(
        iopub_channel=SimpleNamespace(msg_ready=MagicMock(return_value=iopub_ready)),
        stdin_channel=SimpleNamespace(msg_ready=MagicMock(return_value=stdin_ready)),
    )


def test_finite_wait_raises_instead_of_spinning():
    event = MagicMock()
    event.wait.return_value = False
    event.is_set.return_value = False
    guarded = _DeadlineAwareMessageEvent(event, _client(), allow_stdin=False)

    with pytest.raises(TimeoutError, match="Timeout waiting for output"):
        guarded.wait(timeout=0.1)

    event.wait.assert_called_once_with(timeout=0.1)


def test_queued_message_wins_before_wait():
    event = MagicMock()
    guarded = _DeadlineAwareMessageEvent(
        event, _client(iopub_ready=True), allow_stdin=False
    )

    assert guarded.wait(timeout=0.1) is True
    event.wait.assert_not_called()


def test_message_arriving_at_deadline_is_not_a_false_timeout():
    wsclient = _client()
    event = MagicMock()

    def wait_at_boundary(timeout):
        wsclient.iopub_channel.msg_ready.return_value = True
        return False

    event.wait.side_effect = wait_at_boundary
    event.is_set.return_value = False
    guarded = _DeadlineAwareMessageEvent(event, wsclient, allow_stdin=False)

    assert guarded.wait(timeout=0.1) is True


def test_event_set_at_deadline_is_not_a_false_timeout():
    event = MagicMock()
    event.wait.return_value = False
    event.is_set.return_value = True
    guarded = _DeadlineAwareMessageEvent(event, _client(), allow_stdin=False)

    assert guarded.wait(timeout=0.1) is True


def test_guard_restores_original_event():
    original = MagicMock()
    wsclient = _client()
    wsclient._message_received = original
    kernel_client = SimpleNamespace(_manager=SimpleNamespace(client=wsclient))

    with guard_interactive_timeout(kernel_client, allow_stdin=True):
        assert wsclient._message_received is not original

    assert wsclient._message_received is original


def test_guard_serializes_concurrent_install_and_restore():
    original = MagicMock()
    wsclient = _client()
    wsclient._message_received = original
    kernel_client = SimpleNamespace(_manager=SimpleNamespace(client=wsclient))
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def first():
        with guard_interactive_timeout(kernel_client, allow_stdin=False):
            first_entered.set()
            release_first.wait(timeout=2)

    def second():
        first_entered.wait(timeout=2)
        with guard_interactive_timeout(kernel_client, allow_stdin=True):
            second_entered.set()

    first_thread = Thread(target=first)
    second_thread = Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=1)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first_thread.join(timeout=1)
    second_thread.join(timeout=1)

    assert second_entered.is_set()
    assert wsclient._message_received is original


def test_channel_readiness_error_is_not_misreported_as_timeout():
    event = MagicMock()
    wsclient = _client()
    wsclient.iopub_channel.msg_ready.side_effect = RuntimeError("channel failed")
    guarded = _DeadlineAwareMessageEvent(event, wsclient, allow_stdin=False)

    with pytest.raises(RuntimeError, match="channel failed"):
        guarded.wait(timeout=0.1)
