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

"""Execution timeout validation and user-facing timeout semantics."""

import math
from numbers import Real
from typing import Optional


TIMEOUT_EXIT_CODE = 124


def validate_execution_timeout(timeout: Optional[float]) -> Optional[float]:
    """Validate a finite positive execution wait timeout."""

    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, Real):
        raise ValueError("--timeout must be a finite number greater than zero.")

    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--timeout must be a finite number greater than zero.")
    return value


def format_execution_timeout(
    timeout: Optional[float], *, remote_may_continue: bool
) -> str:
    """Explain that a local wait deadline is not proof of remote cancellation."""

    duration = "the configured deadline" if timeout is None else f"{timeout:g} seconds"
    message = f"[colab] Execution timed out after {duration}. The local wait ended"
    if remote_may_continue:
        message += "; the remote kernel may still be running"
    else:
        message += "; cleanup will attempt to release the ephemeral runtime"
    return (
        message + ". Use `colab submit`/`colab wait` for long tasks that must survive "
        "a local CLI disconnect."
    )
