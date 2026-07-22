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

"""Deterministic accelerator parsing and allocation error presentation."""

from typing import Optional

from colab_cli.client import Accelerator, ColabRequestError, Variant
from colab_cli.utils import get_status_code


_GPU_ACCELERATORS = {
    "a100": Accelerator.A100,
    "h100": Accelerator.H100,
    "l4": Accelerator.L4,
    "t4": Accelerator.T4,
    "g4": Accelerator.G4,
}
_TPU_ACCELERATORS = {
    "v5e1": Accelerator.V5E1,
    "v6e1": Accelerator.V6E1,
}

SUPPORTED_GPUS = tuple(accelerator.value for accelerator in _GPU_ACCELERATORS.values())
SUPPORTED_TPUS = tuple(
    accelerator.value.lower() for accelerator in _TPU_ACCELERATORS.values()
)


class AcceleratorArgumentError(ValueError):
    """A local accelerator request is invalid and must not reach the backend."""


def resolve_accelerator(
    *, gpu: Optional[str] = None, tpu: Optional[str] = None
) -> tuple[Variant, Accelerator]:
    """Resolve user accelerator flags without silent fallback."""

    gpu_requested = gpu is not None
    tpu_requested = tpu is not None
    if gpu_requested and tpu_requested:
        raise AcceleratorArgumentError("--gpu and --tpu are mutually exclusive.")

    if gpu_requested:
        accelerator = _GPU_ACCELERATORS.get(gpu.strip().lower())
        if accelerator is None:
            supported = ", ".join(SUPPORTED_GPUS)
            raise AcceleratorArgumentError(
                f"Unsupported GPU {gpu!r}. Supported values: {supported}."
            )
        return Variant.GPU, accelerator

    if tpu_requested:
        accelerator = _TPU_ACCELERATORS.get(tpu.strip().lower())
        if accelerator is None:
            supported = ", ".join(SUPPORTED_TPUS)
            raise AcceleratorArgumentError(
                f"Unsupported TPU {tpu!r}. Supported values: {supported}."
            )
        return Variant.TPU, accelerator

    return Variant.DEFAULT, Accelerator.NONE


def _target_label(accelerator: Accelerator) -> str:
    return "CPU runtime" if accelerator is Accelerator.NONE else accelerator.value


def _retry_advice(accelerator: Accelerator) -> str:
    if accelerator is Accelerator.NONE:
        return "Retry later and inspect account or active-session limits."
    return "Retry later, choose another accelerator, or use CPU."


def format_assignment_error(error: ColabRequestError, accelerator: Accelerator) -> str:
    """Return a concise, non-misleading allocation failure message."""

    status = get_status_code(error)
    reason = getattr(getattr(error, "response", None), "reason", None)
    target = _target_label(accelerator)
    status_text = f"HTTP {status}" if status is not None else "unknown HTTP status"
    if reason:
        status_text = f"{status_text} {reason}"

    if status == 412:
        return (
            f"[colab] Colab rejected the {target} request ({status_text}). "
            "HTTP 412 may indicate an account usage limit, missing entitlement, "
            "or current capacity; it does not reliably mean too many active "
            f"sessions. {_retry_advice(accelerator)}"
        )

    if status == 503:
        return (
            f"[colab] The requested {target} is temporarily unavailable "
            f"({status_text}). Retry with bounded backoff. "
            f"{_retry_advice(accelerator)}"
        )

    if status == 429:
        return (
            f"[colab] Colab rate or usage limits rejected the {target} request "
            f"({status_text}). Retry later; repeated immediate retries are not "
            "recommended."
        )

    if status in {401, 403}:
        return (
            f"[colab] Colab did not authorize the {target} request "
            f"({status_text}). Re-authenticate and verify account entitlement."
        )

    if status == 400 and accelerator is not Accelerator.NONE:
        return (
            f"[colab] Colab rejected accelerator {accelerator.value!r} "
            f"({status_text}). The account may not have entitlement for this "
            "accelerator; try a different accelerator or CPU."
        )

    return (
        f"[colab] Colab could not allocate the {target} ({status_text}). "
        "Re-run with --logtostderr for request diagnostics."
    )
