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

from unittest.mock import MagicMock

import pytest

from colab_cli.accelerators import (
    AcceleratorArgumentError,
    format_assignment_error,
    resolve_accelerator,
)
from colab_cli.client import Accelerator, ColabRequestError, Variant


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        ("T4", Accelerator.T4),
        ("l4", Accelerator.L4),
        ("G4", Accelerator.G4),
        ("a100", Accelerator.A100),
        ("H100", Accelerator.H100),
    ],
)
def test_resolve_gpu_is_case_insensitive_and_exact(gpu, expected):
    assert resolve_accelerator(gpu=gpu) == (Variant.GPU, expected)


@pytest.mark.parametrize(
    ("tpu", "expected"),
    [("v5e1", Accelerator.V5E1), ("V6E1", Accelerator.V6E1)],
)
def test_resolve_tpu_is_case_insensitive_and_exact(tpu, expected):
    assert resolve_accelerator(tpu=tpu) == (Variant.TPU, expected)


def test_resolve_default_is_cpu():
    assert resolve_accelerator() == (Variant.DEFAULT, Accelerator.NONE)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gpu": "T44"}, "Unsupported GPU"),
        ({"gpu": ""}, "Unsupported GPU"),
        ({"gpu": "   "}, "Unsupported GPU"),
        ({"tpu": "v7"}, "Unsupported TPU"),
        ({"tpu": ""}, "Unsupported TPU"),
        ({"gpu": "T4", "tpu": "v5e1"}, "mutually exclusive"),
    ],
)
def test_invalid_accelerator_fails_closed(kwargs, message):
    with pytest.raises(AcceleratorArgumentError, match=message):
        resolve_accelerator(**kwargs)


def _request_error(status: int, reason: str = "Error") -> ColabRequestError:
    response = MagicMock(status_code=status, reason=reason)
    return ColabRequestError(
        "assignment failed", request=MagicMock(), response=response
    )


def test_http_412_is_not_relabelled_as_too_many_assignments():
    message = format_assignment_error(_request_error(412), Accelerator.T4)
    assert "HTTP 412" in message
    assert "usage limit" in message
    assert "does not reliably mean too many" in message


def test_http_503_is_presented_as_temporary_capacity():
    message = format_assignment_error(
        _request_error(503, "Service Unavailable"), Accelerator.G4
    )
    assert "temporarily unavailable" in message
    assert "bounded backoff" in message


def test_cpu_412_advice_does_not_tell_user_to_fall_back_to_cpu():
    message = format_assignment_error(_request_error(412), Accelerator.NONE)

    assert "use CPU" not in message
    assert "active-session limits" in message
