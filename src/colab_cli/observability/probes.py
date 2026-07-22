# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Bounded, read-only probes for one explicitly selected Colab runtime."""

from __future__ import annotations

import contextlib
from functools import partial
import json
import math
import os
import time
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from jupyter_kernel_client.client import output_hook as capture_output
from jupyter_kernel_client.wsclient import JupyterSubprotocol, KernelWebSocketClient
from pydantic import BaseModel, ConfigDict, Field
import requests

from colab_cli._jupyter_compat import guard_interactive_timeout
from colab_cli.client import ACCEPT_JSON_HEADER, COLAB_CLIENT_AGENT_HEADER
from colab_cli.observability.models import (
    DiskObservation,
    GpuObservation,
    MemoryObservation,
    ObservationIssue,
    ProbeObservation,
    RuntimeObservation,
)
from colab_cli.observability.redaction import redact_text
from colab_cli.state import SessionState


_RUNTIME_PROXY_TOKEN_HEADER = "X-Colab-Runtime-Proxy-Token"
_MIB = 1024 * 1024


class _RawMemory(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    total_bytes: int | None = Field(None, alias="totalBytes")
    free_bytes: int | None = Field(None, alias="freeBytes")


class _RawGpu(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    name: str | None = None
    memory_used_bytes: int | None = Field(None, alias="memoryUsedBytes")
    memory_total_bytes: int | None = Field(None, alias="memoryTotalBytes")
    gpu_utilization: float | None = Field(None, alias="gpuUtilization")
    memory_utilization: float | None = Field(None, alias="memoryUtilization")


class _RawFilesystem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    label: str | None = None
    total_bytes: int | None = Field(None, alias="totalBytes")
    used_bytes: int | None = Field(None, alias="usedBytes")


class _RawDisk(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    filesystem: _RawFilesystem = Field(default_factory=_RawFilesystem)


class _RawResources(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    memory: _RawMemory = Field(default_factory=_RawMemory)
    disks: list[_RawDisk] = Field(default_factory=list)
    gpus: list[_RawGpu] = Field(default_factory=list)


class ExistingKernelJsonExecutor:
    """Connect to the recorded kernel websocket without a kernel-create path."""

    def __init__(self, session: SessionState, *, connect_timeout: float) -> None:
        if not session.kernel_id:
            raise LookupError("No existing kernel ID is recorded for this session.")
        self.session = session
        self.connect_timeout = connect_timeout
        self._client: KernelWebSocketClient | None = None

    def _connect(self) -> KernelWebSocketClient:
        if self._client is not None:
            return self._client
        parsed = urlparse(self.session.url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        kernel_id = quote(self.session.kernel_id, safe="")
        endpoint = (
            f"{scheme}://{parsed.netloc}{base_path}/api/kernels/{kernel_id}/channels"
        )
        client = KernelWebSocketClient(
            endpoint=endpoint,
            token=self.session.token,
            timeout=self.connect_timeout,
            subprotocol=JupyterSubprotocol.DEFAULT,
            headers={
                COLAB_CLIENT_AGENT_HEADER["key"]: COLAB_CLIENT_AGENT_HEADER["value"],
                _RUNTIME_PROXY_TOKEN_HEADER: self.session.token,
            },
        )
        if self.session.session_id:
            client.session.session = self.session.session_id
        try:
            client.start_channels(stdin=False)
            if not client.channels_running:
                raise TimeoutError("Existing kernel websocket did not become ready.")
        except Exception:
            client.timeout = min(1.0, self.connect_timeout)
            with contextlib.suppress(Exception):
                client.stop_channels()
            raise
        client.timeout = min(1.0, self.connect_timeout)
        self._client = client
        return client

    def execute_json(self, code: str, *, timeout: float) -> dict[str, Any]:
        client = self._connect()
        marker = f"__COLAB_CLI_OBSERVE_{os.urandom(16).hex()}__"
        wrapped = (
            f"_COLAB_CLI_RESULT_MARKER = {marker!r}\n"
            "import json as _colab_cli_json\n"
            f"{code.rstrip()}\n"
            "print(_COLAB_CLI_RESULT_MARKER + "
            "_colab_cli_json.dumps(_colab_cli_result, sort_keys=True), "
            "flush=True)\n"
        )
        outputs: list[dict[str, Any]] = []
        with guard_interactive_timeout(client, allow_stdin=False):
            reply = client.execute_interactive(
                wrapped,
                allow_stdin=False,
                timeout=timeout,
                output_hook=partial(capture_output, outputs),
            )
        if reply and reply.get("content", {}).get("status") == "error":
            content = reply["content"]
            raise RuntimeError(
                f"{content.get('ename', 'RemoteError')}: "
                f"{content.get('evalue', '')}".rstrip()
            )
        return _decode_result(outputs, marker)

    def close(self) -> None:
        if self._client is not None:
            self._client.stop_channels()
            self._client = None


@contextlib.contextmanager
def open_existing_kernel_executor(session: SessionState, *, connect_timeout: float):
    executor = ExistingKernelJsonExecutor(session, connect_timeout=connect_timeout)
    try:
        yield executor
    finally:
        executor.close()


def probe_session(session: SessionState, *, timeout: float) -> ProbeObservation:
    from colab_cli.observability.collector import utc_now, validate_probe_timeout

    validate_probe_timeout(timeout)
    started = time.monotonic()
    deadline = started + timeout
    issues: list[ObservationIssue] = []
    gpu = GpuObservation()
    memory = MemoryObservation()
    disk = DiskObservation()
    runtime = RuntimeObservation()

    resource_budget = min(5.0, _remaining(deadline))
    if resource_budget > 0:
        try:
            _merge_resource_api(
                _fetch_runtime_resources(session, timeout=resource_budget),
                gpu=gpu,
                memory=memory,
                disk=disk,
            )
        except requests.Timeout as exc:
            issues.append(
                ObservationIssue(
                    code="RUNTIME_RESOURCE_API_TIMEOUT",
                    message=redact_text(
                        str(exc) or "Runtime resource API timed out.",
                        secrets=(session.token,),
                    ),
                    source="runtime_api",
                    retryable=True,
                )
            )
        except Exception as exc:
            issues.append(
                ObservationIssue(
                    code="RUNTIME_RESOURCE_API_UNAVAILABLE",
                    message=redact_text(
                        f"{type(exc).__name__}: {exc}",
                        secrets=(session.token,),
                    ),
                    source="runtime_api",
                    retryable=True,
                )
            )

    remaining = _remaining(deadline)
    if session.kernel_id and remaining > 0:
        try:
            with open_existing_kernel_executor(
                session, connect_timeout=min(5.0, remaining)
            ) as executor:
                remaining = _remaining(deadline)
                if remaining <= 0:
                    raise TimeoutError("Probe deadline expired before execution.")
                supplemental = executor.execute_json(
                    _supplemental_code(min(5.0, remaining)),
                    timeout=remaining,
                )
            _merge_supplemental(
                supplemental,
                gpu=gpu,
                memory=memory,
                disk=disk,
                runtime=runtime,
                issues=issues,
                secrets=(session.token,),
            )
        except TimeoutError as exc:
            issues.append(
                ObservationIssue(
                    code="EXISTING_KERNEL_PROBE_TIMEOUT",
                    message=redact_text(
                        str(exc) or "Existing kernel probe timed out.",
                        secrets=(session.token,),
                    ),
                    source="existing_kernel",
                    retryable=True,
                )
            )
        except Exception as exc:
            issues.append(
                ObservationIssue(
                    code="EXISTING_KERNEL_PROBE_UNAVAILABLE",
                    message=redact_text(
                        f"{type(exc).__name__}: {exc}",
                        secrets=(session.token,),
                    ),
                    source="existing_kernel",
                    retryable=True,
                )
            )
    elif not session.kernel_id:
        issues.append(
            ObservationIssue(
                code="EXISTING_KERNEL_NOT_RECORDED",
                message=(
                    "No existing kernel ID is recorded; driver, temperature, "
                    "and runtime boot identity were not probed."
                ),
                source="existing_kernel",
                retryable=False,
            )
        )
    else:
        issues.append(
            ObservationIssue(
                code="PROBE_DEADLINE_EXHAUSTED",
                message="Probe deadline expired before the existing-kernel stage.",
                source="probe",
                retryable=True,
            )
        )

    _finalize(gpu, memory, disk, runtime)
    useful = any(
        value is not None
        for value in (
            gpu.name,
            memory.total_bytes,
            disk.total_bytes,
            runtime.boot_id,
        )
    )
    timed_out = any(
        "TIMEOUT" in issue.code or "DEADLINE" in issue.code for issue in issues
    )
    status = (
        "ok"
        if not issues
        else ("partial" if useful else ("timeout" if timed_out else "unavailable"))
    )
    return ProbeObservation(
        status=status,
        observed_at=utc_now(),
        duration_ms=round((time.monotonic() - started) * 1000),
        timeout_seconds=timeout,
        gpu=gpu,
        memory=memory,
        disk=disk,
        runtime=runtime,
        issues=issues,
    )


def _fetch_runtime_resources(session: SessionState, *, timeout: float) -> _RawResources:
    response = requests.get(
        urljoin(session.url.rstrip("/") + "/", "api/colab/resources"),
        headers={
            ACCEPT_JSON_HEADER["key"]: ACCEPT_JSON_HEADER["value"],
            COLAB_CLIENT_AGENT_HEADER["key"]: COLAB_CLIENT_AGENT_HEADER["value"],
            _RUNTIME_PROXY_TOKEN_HEADER: session.token,
        },
        timeout=(timeout / 2.0, timeout / 2.0),
    )
    response.raise_for_status()
    text = response.text
    if text.startswith(")]}'"):
        _, separator, text = text.partition("\n")
        if not separator:
            raise ValueError("Runtime resource response contained no JSON body.")
    return _RawResources.model_validate(json.loads(text))


def _merge_resource_api(resources, *, gpu, memory, disk):
    if (
        resources.memory.total_bytes is not None
        or resources.memory.free_bytes is not None
    ):
        memory.total_bytes = resources.memory.total_bytes
        memory.available_bytes = resources.memory.free_bytes
        if memory.total_bytes is not None and memory.available_bytes is not None:
            memory.used_bytes = max(0, memory.total_bytes - memory.available_bytes)
        memory.sources.append("runtime_api")
    selected = _select_disk(resources.disks)
    if selected is not None:
        fs = selected.filesystem
        disk.total_bytes = fs.total_bytes
        disk.used_bytes = fs.used_bytes
        if disk.total_bytes is not None and disk.used_bytes is not None:
            disk.free_bytes = max(0, disk.total_bytes - disk.used_bytes)
        disk.sources.append("runtime_api")
    if resources.gpus:
        raw = resources.gpus[0]
        gpu.available = True
        gpu.name = raw.name
        gpu.memory_total_bytes = raw.memory_total_bytes
        gpu.memory_used_bytes = raw.memory_used_bytes
        gpu.utilization_percent = _percent(raw.gpu_utilization)
        gpu.memory_utilization_percent = _percent(raw.memory_utilization)
        gpu.sources.append("runtime_api")


def _merge_supplemental(data, *, gpu, memory, disk, runtime, issues, secrets=()):
    raw = data.get("memory") or {}
    memory_values = {
        field: _integer(raw.get(field))
        for field in ("total_bytes", "available_bytes", "used_bytes")
    }
    if any(value is not None for value in memory_values.values()):
        for field, value in memory_values.items():
            if value is not None:
                setattr(memory, field, value)
        _source(memory.sources, "existing_kernel")
    raw = data.get("disk") or {}
    disk_values = {
        field: _integer(raw.get(field))
        for field in ("total_bytes", "used_bytes", "free_bytes")
    }
    if any(value is not None for value in disk_values.values()):
        for field, value in disk_values.items():
            if value is not None:
                setattr(disk, field, value)
        _source(disk.sources, "existing_kernel")
    raw = data.get("runtime") or {}
    runtime.boot_id = _string(raw.get("boot_id"))
    runtime.python_version = _string(raw.get("python_version"))
    runtime.platform = _string(raw.get("platform"))
    if any((runtime.boot_id, runtime.python_version, runtime.platform)):
        runtime.source = "existing_kernel"
    raw = data.get("gpu") or {}
    if raw.get("available"):
        gpu.available = True
        gpu.name = _string(raw.get("name")) or gpu.name
        gpu_values = {
            "driver_version": _string(raw.get("driver_version")),
            "memory_total_bytes": _integer(raw.get("memory_total_bytes")),
            "memory_used_bytes": _integer(raw.get("memory_used_bytes")),
            "utilization_percent": _number(raw.get("utilization_percent")),
            "memory_utilization_percent": _number(
                raw.get("memory_utilization_percent")
            ),
            "temperature_c": _number(raw.get("temperature_c")),
        }
        for field, value in gpu_values.items():
            if value is not None:
                setattr(gpu, field, value)
        _source(gpu.sources, "existing_kernel")
    elif not gpu.available:
        gpu.unavailable_reason = _string(raw.get("unavailable_reason"))
    for raw_issue in data.get("issues") or []:
        issues.append(
            ObservationIssue(
                code=str(raw_issue.get("code", "REMOTE_PROBE_WARNING")),
                message=redact_text(
                    raw_issue.get("message", "Remote probe warning."),
                    secrets=secrets,
                ),
                source="existing_kernel",
                retryable=raw_issue.get("retryable"),
            )
        )


def _supplemental_code(command_timeout: float) -> str:
    return f"""
import csv
import platform
import shutil
import subprocess


def _number(value):
    value = value.strip()
    if not value or value.upper() in {{'N/A', '[N/A]'}}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _memory():
    values = {{}}
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as stream:
            for line in stream:
                key, raw = line.split(':', 1)
                parts = raw.strip().split()
                if parts:
                    values[key] = int(parts[0]) * 1024
    except Exception:
        pass
    total = values.get('MemTotal')
    available = values.get('MemAvailable')
    return {{
        'total_bytes': total,
        'available_bytes': available,
        'used_bytes': (
            max(0, total - available)
            if total is not None and available is not None
            else None
        ),
    }}


_issues = []
_gpu = {{'available': False, 'unavailable_reason': None}}
_smi = shutil.which('nvidia-smi')
if _smi is None:
    _gpu['unavailable_reason'] = 'nvidia_smi_not_found'
else:
    try:
        _done = subprocess.run(
            [
                _smi,
                '--query-gpu=name,driver_version,memory.total,memory.used,'
                'utilization.gpu,utilization.memory,temperature.gpu',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout={command_timeout!r},
            check=False,
        )
        if _done.returncode != 0:
            _gpu['unavailable_reason'] = 'nvidia_smi_failed'
            _issues.append({{
                'code': 'NVIDIA_SMI_FAILED',
                'message': (_done.stderr or _done.stdout or 'nvidia-smi failed')[:500],
                'retryable': True,
            }})
        else:
            _rows = list(csv.reader(_done.stdout.splitlines()))
            if _rows:
                _row = _rows[0] + [''] * max(0, 7 - len(_rows[0]))
                _total = _number(_row[2])
                _used = _number(_row[3])
                _gpu = {{
                    'available': True,
                    'name': _row[0].strip() or None,
                    'driver_version': _row[1].strip() or None,
                    'memory_total_bytes': int(_total * {_MIB}) if _total is not None else None,
                    'memory_used_bytes': int(_used * {_MIB}) if _used is not None else None,
                    'utilization_percent': _number(_row[4]),
                    'memory_utilization_percent': _number(_row[5]),
                    'temperature_c': _number(_row[6]),
                    'unavailable_reason': None,
                }}
            else:
                _gpu['unavailable_reason'] = 'no_gpu_reported'
    except subprocess.TimeoutExpired:
        _gpu['unavailable_reason'] = 'nvidia_smi_timeout'
        _issues.append({{
            'code': 'NVIDIA_SMI_TIMEOUT',
            'message': 'nvidia-smi exceeded its bounded timeout.',
            'retryable': True,
        }})
    except Exception as _exc:
        _gpu['unavailable_reason'] = 'nvidia_smi_error'
        _issues.append({{
            'code': 'NVIDIA_SMI_ERROR',
            'message': f'{{type(_exc).__name__}}: {{_exc}}'[:500],
            'retryable': True,
        }})

try:
    _usage = shutil.disk_usage('/content')
    _disk = {{
        'path': '/content',
        'total_bytes': _usage.total,
        'used_bytes': _usage.used,
        'free_bytes': _usage.free,
    }}
except Exception as _exc:
    _disk = {{'path': '/content'}}
    _issues.append({{
        'code': 'CONTENT_DISK_UNAVAILABLE',
        'message': f'{{type(_exc).__name__}}: {{_exc}}'[:500],
        'retryable': True,
    }})

try:
    with open('/proc/sys/kernel/random/boot_id', 'r', encoding='utf-8') as stream:
        _boot_id = stream.read().strip() or None
except Exception:
    _boot_id = None

_colab_cli_result = {{
    'gpu': _gpu,
    'memory': _memory(),
    'disk': _disk,
    'runtime': {{
        'boot_id': _boot_id,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
    }},
    'issues': _issues,
}}
"""


def _decode_result(outputs, marker):
    parts = []
    for output in outputs:
        if output.get("output_type") == "error":
            raise RuntimeError(
                f"{output.get('ename', 'RemoteError')}: "
                f"{output.get('evalue', '')}".rstrip()
            )
        text = output.get("text")
        if isinstance(text, list):
            parts.extend(str(item) for item in text)
        elif text is not None:
            parts.append(str(text))
        data = output.get("data")
        if isinstance(data, dict) and "text/plain" in data:
            value = data["text/plain"]
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            else:
                parts.append(str(value))
    combined = "".join(parts)
    index = combined.find(marker)
    if index < 0:
        raise RuntimeError("Existing-kernel probe returned no result marker.")
    result, _ = json.JSONDecoder().raw_decode(combined[index + len(marker) :].lstrip())
    if not isinstance(result, dict):
        raise RuntimeError("Existing-kernel probe result must be a JSON object.")
    return result


def _select_disk(disks):
    for disk in disks:
        label = (disk.filesystem.label or "").lower()
        if "content" in label or label == "/":
            return disk
    return disks[0] if disks else None


def _finalize(gpu, memory, disk, runtime):
    if not gpu.available and gpu.unavailable_reason is None:
        gpu.unavailable_reason = "no_gpu_reported"
    if memory.total_bytes is None:
        memory.unavailable_reason = "memory_metrics_unavailable"
    if disk.total_bytes is None:
        disk.unavailable_reason = "content_disk_metrics_unavailable"
    if runtime.boot_id is None:
        runtime.unavailable_reason = "existing_kernel_probe_unavailable"


def _source(sources, value):
    if value not in sources:
        sources.append(value)


def _remaining(deadline):
    return max(0.0, deadline - time.monotonic())


def _percent(value):
    if value is None:
        return None
    return round(value * 100, 3) if 0 <= value <= 1 else round(value, 3)


def _string(value):
    return None if value is None else str(value)


def _integer(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None
