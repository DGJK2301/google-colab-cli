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

"""Narrow compatibility guards for the pinned Jupyter WebSocket client.

The affected ``execute_interactive`` receive loop ignores a finite
``Event.wait`` timeout and then repeatedly calls ``wait(0)``, consuming a CPU
core indefinitely. It can also clear the event after a message has already been
queued, losing the wake-up. This module fixes both failure modes without
copying or replacing the upstream receive loop.
"""

from contextlib import contextmanager
from threading import Lock, RLock
from typing import Any, Iterator


_TIMEOUT_MESSAGE = "Timeout waiting for output"
_GUARD_LOCK_ATTRIBUTE = "_colab_cli_timeout_guard_lock"
_GUARD_LOCK_CREATION = Lock()


class _DeadlineAwareMessageEvent:
    """Proxy an Event while making queue state authoritative at a deadline."""

    def __init__(self, event: Any, wsclient: Any, *, allow_stdin: bool):
        self._event = event
        self._wsclient = wsclient
        self._allow_stdin = allow_stdin

    def _channel_ready(self, name: str) -> bool:
        channel = getattr(self._wsclient, name, None)
        ready = getattr(channel, "msg_ready", None)
        if not callable(ready):
            return False
        return bool(ready())

    def _message_ready(self) -> bool:
        return self._channel_ready("iopub_channel") or (
            self._allow_stdin and self._channel_ready("stdin_channel")
        )

    def wait(self, timeout: float | None = None) -> bool:
        if self._message_ready():
            return True

        signaled = self._event.wait(timeout=timeout)
        if signaled:
            return True

        # A wake-up can race the timeout boundary. Treat either the event bit
        # or a queued stdin/IOPub message as authoritative before declaring the
        # deadline expired. The next receive-loop iteration will drain it.
        if self._event.is_set() or self._message_ready():
            return True

        if timeout is not None:
            raise TimeoutError(_TIMEOUT_MESSAGE)
        return False

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def is_set(self) -> bool:
        return bool(self._event.is_set())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._event, name)


def _guard_lock(wsclient: Any) -> RLock:
    """Return one re-entrant compatibility lock per WebSocket client."""

    lock = getattr(wsclient, _GUARD_LOCK_ATTRIBUTE, None)
    if lock is not None:
        return lock

    # Attribute creation itself must be serialized. Holding this RLock across
    # the context also closes a subtle race: the upstream interactive lock is
    # acquired *inside* execute_interactive(), so a second caller could otherwise
    # observe the first proxy and then execute after the first caller restored it.
    with _GUARD_LOCK_CREATION:
        lock = getattr(wsclient, _GUARD_LOCK_ATTRIBUTE, None)
        if lock is None:
            lock = RLock()
            setattr(wsclient, _GUARD_LOCK_ATTRIBUTE, lock)
    return lock


@contextmanager
def guard_interactive_timeout(
    kernel_client: Any, *, allow_stdin: bool
) -> Iterator[None]:
    """Temporarily harden a kernel client's interactive receive event."""

    manager = getattr(kernel_client, "_manager", None)
    wsclient = getattr(manager, "client", None)
    if wsclient is None:
        yield
        return

    with _guard_lock(wsclient):
        original_event = getattr(wsclient, "_message_received", None)
        if original_event is None:
            yield
            return

        # A nested call in the same thread is already protected by the outer
        # proxy. The per-client RLock prevents a different thread from reaching
        # this branch and later running after the proxy has been restored.
        if isinstance(original_event, _DeadlineAwareMessageEvent):
            yield
            return

        guarded_event = _DeadlineAwareMessageEvent(
            original_event, wsclient, allow_stdin=allow_stdin
        )
        wsclient._message_received = guarded_event
        try:
            yield
        finally:
            if getattr(wsclient, "_message_received", None) is guarded_event:
                wsclient._message_received = original_event
