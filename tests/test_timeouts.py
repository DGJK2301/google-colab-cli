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

import math

import pytest

from colab_cli.timeouts import (
    TIMEOUT_EXIT_CODE,
    format_execution_timeout,
    validate_execution_timeout,
)


def test_timeout_exit_code_matches_shell_convention():
    assert TIMEOUT_EXIT_CODE == 124


@pytest.mark.parametrize(
    "value", [0, -1, math.inf, -math.inf, math.nan, True, "10", object()]
)
def test_invalid_timeout_is_rejected(value):
    with pytest.raises(ValueError, match="finite number greater than zero"):
        validate_execution_timeout(value)


def test_none_and_positive_timeout_are_preserved():
    assert validate_execution_timeout(None) is None
    assert validate_execution_timeout(1) == 1.0
    assert validate_execution_timeout(0.25) == 0.25


def test_existing_session_timeout_states_remote_uncertainty():
    message = format_execution_timeout(10, remote_may_continue=True)
    assert "remote kernel may still be running" in message
    assert "submit`/`colab wait" in message


def test_ephemeral_timeout_states_cleanup_attempt():
    message = format_execution_timeout(10, remote_may_continue=False)
    assert "cleanup will attempt to release" in message
