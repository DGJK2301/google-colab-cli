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

"""Self-contained runtime helper executed inside a Colab kernel.

Keep this module standard-library-only. ``RemoteJobClient`` sends its source to
the runtime, so a reconnected local CLI does not depend on a package install in
the VM and can recover jobs solely from their remote directory.
"""

import base64
import datetime
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import uuid


SCHEMA_VERSION = 1
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "lost"}
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


_RUNNER_SOURCE = r"""import datetime
import json
import os
import platform
import signal
import subprocess
import sys
import time


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, payload):
    temp = path + '.tmp-' + str(os.getpid())
    with open(temp, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(50):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 49:
                raise
            time.sleep(0.01)


def runtime_id():
    path = '/proc/sys/kernel/random/boot_id'
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            value = stream.read().strip()
        if value:
            return value
    except OSError:
        pass
    return f'{sys.platform}:{platform.node()}'


def process_start_token(pid):
    path = f'/proc/{int(pid)}/stat'
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            value = stream.read()
        fields = value[value.rfind(')') + 2:].split()
        return fields[19] if len(fields) > 19 else None
    except (OSError, ValueError):
        return None


job_dir = os.path.abspath(sys.argv[1])
spec_path = os.path.join(job_dir, 'spec.json')
status_path = os.path.join(job_dir, 'status.json')
with open(spec_path, 'r', encoding='utf-8') as stream:
    spec = json.load(stream)

cancel_requested = False
cancel_signal = None
child = None


def handle_signal(signum, _frame):
    global cancel_requested, cancel_signal
    cancel_requested = True
    cancel_signal = signum
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signum)
        except OSError:
            pass


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, handle_signal)

started_at = utcnow()
base_status = {
    'schema_version': 1,
    'job_id': spec['job_id'],
    'created_at': spec['created_at'],
    'started_at': started_at,
    'runner_pid': os.getpid(),
    'runner_start_token': process_start_token(os.getpid()),
    'runtime_id': runtime_id(),
    'state': 'running',
    'heartbeat_at': started_at,
    'stdout_path': os.path.join(job_dir, 'stdout.log'),
    'stderr_path': os.path.join(job_dir, 'stderr.log'),
}

try:
    env = os.environ.copy()
    env.update(spec.get('env') or {})
    with open(base_status['stdout_path'], 'ab', buffering=0) as stdout_stream, open(
        base_status['stderr_path'], 'ab', buffering=0
    ) as stderr_stream:
        child = subprocess.Popen(
            spec['argv'],
            cwd=spec['cwd'],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
        status = dict(base_status, process_pid=child.pid)
        atomic_json(status_path, status)
        while True:
            returncode = child.poll()
            if returncode is not None:
                break
            status['heartbeat_at'] = utcnow()
            atomic_json(status_path, status)
            time.sleep(1.0)

    finished_at = utcnow()
    if cancel_requested:
        state = 'cancelled'
    else:
        state = 'succeeded' if returncode == 0 else 'failed'
    status.update(
        {
            'state': state,
            'returncode': returncode,
            'finished_at': finished_at,
            'heartbeat_at': finished_at,
            'cancel_signal': cancel_signal,
        }
    )
    atomic_json(status_path, status)
except BaseException as exc:
    failed_at = utcnow()
    status = dict(
        base_status,
        state='cancelled' if cancel_requested else 'failed',
        returncode=None,
        finished_at=failed_at,
        heartbeat_at=failed_at,
        error=f'{type(exc).__name__}: {exc}',
        cancel_signal=cancel_signal,
    )
    atomic_json(status_path, status)
    raise
"""


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _validate_job_id(job_id):
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("job_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    return job_id


def _job_dir(root, job_id):
    return os.path.join(os.path.abspath(root), _validate_job_id(job_id))


def _atomic_json(path, payload):
    temp = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(50):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 49:
                raise
            time.sleep(0.01)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _runtime_id():
    path = "/proc/sys/kernel/random/boot_id"
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = stream.read().strip()
        if value:
            return value
    except OSError:
        pass
    return f"{sys.platform}:{platform.node()}"


def _process_start_token(pid):
    path = f"/proc/{int(pid)}/stat"
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = stream.read()
        fields = value[value.rfind(")") + 2 :].split()
        return fields[19] if len(fields) > 19 else None
    except (OSError, ValueError):
        return None


def _pid_alive(pid, expected_start_token=None):
    if not pid:
        return False
    if expected_start_token is not None:
        actual_start_token = _process_start_token(pid)
        if actual_start_token != str(expected_start_token):
            return False
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a harmless existence probe on Windows:
        # CPython can route it through TerminateProcess.  Use the Win32 query
        # API so local tests and Windows clients never mutate the target.
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    proc_stat = f"/proc/{int(pid)}/stat"
    if os.path.exists(proc_stat):
        try:
            with open(proc_stat, "r", encoding="utf-8") as stream:
                fields = stream.read().split()
            if len(fields) > 2 and fields[2] == "Z":
                return False
        except OSError:
            pass
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _status(root, job_id):
    job_dir = _job_dir(root, job_id)
    status_path = os.path.join(job_dir, "status.json")
    if not os.path.isfile(status_path):
        raise FileNotFoundError(f"Job status not found: {job_id}")
    status = _read_json(status_path)
    runner_pid = status.get("runner_pid")
    launcher_path = os.path.join(job_dir, "launcher.json")
    launcher = _read_json(launcher_path) if os.path.isfile(launcher_path) else {}
    if not runner_pid and launcher:
        runner_pid = launcher.get("runner_pid")
        status["runner_pid"] = runner_pid
    runtime_id = status.get("runtime_id") or launcher.get("runtime_id")
    if runtime_id != _runtime_id():
        status.update(
            {
                "state": "lost",
                "finished_at": _utcnow(),
                "error": "runtime identity changed or is missing",
                "runner_alive": False,
            }
        )
        _atomic_json(status_path, status)
        status["job_dir"] = job_dir
        return status
    runner_start_token = status.get("runner_start_token") or launcher.get(
        "runner_start_token"
    )
    alive = _pid_alive(runner_pid, expected_start_token=runner_start_token)
    status["runner_alive"] = alive
    if status.get("state") not in TERMINAL_STATES and not alive:
        # Re-read once in case the runner committed its terminal state between
        # our first read and process check.
        status = _read_json(status_path)
        if status.get("state") not in TERMINAL_STATES:
            status.update(
                {
                    "state": "lost",
                    "finished_at": _utcnow(),
                    "error": "runner process exited without a terminal status",
                    "runner_alive": False,
                }
            )
            _atomic_json(status_path, status)
    status["job_dir"] = job_dir
    return status


def _submit(payload):
    root = os.path.abspath(payload["root"])
    job_id = _validate_job_id(payload["job_id"])
    argv = payload.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) and value for value in argv)
    ):
        raise ValueError("argv must be a non-empty list of non-empty strings")
    cwd = os.path.abspath(payload.get("cwd") or "/content")
    if not os.path.isdir(cwd):
        raise FileNotFoundError(f"Working directory not found: {cwd}")
    env = payload.get("env") or {}
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise ValueError("env must map strings to strings")

    os.makedirs(root, exist_ok=True)
    job_dir = _job_dir(root, job_id)
    if os.path.exists(job_dir):
        raise FileExistsError(f"Job already exists: {job_id}")
    os.mkdir(job_dir)
    created_at = _utcnow()
    runtime_id = _runtime_id()
    spec = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "argv": argv,
        "cwd": cwd,
        "env": env,
        "created_at": created_at,
        "runtime_id": runtime_id,
    }
    _atomic_json(os.path.join(job_dir, "spec.json"), spec)
    runner_path = os.path.join(job_dir, "runner.py")
    with open(runner_path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(_RUNNER_SOURCE)
        stream.flush()
        os.fsync(stream.fileno())
    queued = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "state": "queued",
        "created_at": created_at,
        "heartbeat_at": created_at,
        "runtime_id": runtime_id,
        "stdout_path": os.path.join(job_dir, "stdout.log"),
        "stderr_path": os.path.join(job_dir, "stderr.log"),
    }
    _atomic_json(os.path.join(job_dir, "status.json"), queued)
    process = subprocess.Popen(
        [sys.executable, runner_path, job_dir],
        cwd=job_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _atomic_json(
        os.path.join(job_dir, "launcher.json"),
        {
            "runner_pid": process.pid,
            "runner_start_token": _process_start_token(process.pid),
            "runtime_id": runtime_id,
            "launched_at": _utcnow(),
        },
    )
    return dict(queued, runner_pid=process.pid, job_dir=job_dir)


def _list_jobs(payload):
    root = os.path.abspath(payload["root"])
    if not os.path.isdir(root):
        return {"root": root, "jobs": []}
    jobs = []
    for name in sorted(os.listdir(root)):
        if not _JOB_ID.fullmatch(name):
            continue
        status_path = os.path.join(root, name, "status.json")
        if not os.path.isfile(status_path):
            continue
        jobs.append(_status(root, name))
    return {"root": root, "jobs": jobs}


def _tail(payload):
    root = payload["root"]
    job_id = _validate_job_id(payload["job_id"])
    stream_name = payload.get("stream", "stdout")
    if stream_name not in {"stdout", "stderr"}:
        raise ValueError("stream must be stdout or stderr")
    offset = int(payload.get("offset", 0))
    max_bytes = int(payload.get("max_bytes", 65536))
    if offset < 0 or max_bytes <= 0:
        raise ValueError("offset must be non-negative and max_bytes must be positive")
    path = os.path.join(_job_dir(root, job_id), f"{stream_name}.log")
    if not os.path.exists(path):
        size = 0
        data = b""
    else:
        size = os.path.getsize(path)
        if offset > size:
            raise ValueError(f"offset {offset} exceeds log size {size}")
        with open(path, "rb") as stream:
            stream.seek(offset)
            data = stream.read(max_bytes)
    return {
        "job_id": job_id,
        "stream": stream_name,
        "offset": offset,
        "next_offset": offset + len(data),
        "size": size,
        "eof": offset + len(data) >= size,
        "data": base64.b64encode(data).decode("ascii"),
    }


def _signal_group(pid, sig):
    if os.name == "posix":
        os.killpg(int(pid), sig)
    else:
        os.kill(int(pid), sig)


def _cancel(payload):
    root = payload["root"]
    job_id = _validate_job_id(payload["job_id"])
    grace = float(payload.get("grace_seconds", 10.0))
    if grace < 0:
        raise ValueError("grace_seconds must be non-negative")
    status = _status(root, job_id)
    if status.get("state") in TERMINAL_STATES:
        return status
    pid = status.get("runner_pid")
    status.update({"state": "cancelling", "cancel_requested_at": _utcnow()})
    _atomic_json(os.path.join(_job_dir(root, job_id), "status.json"), status)
    if pid and _pid_alive(pid):
        try:
            _signal_group(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.1)
        if _pid_alive(pid):
            try:
                _signal_group(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
            kill_deadline = time.monotonic() + 2.0
            while time.monotonic() < kill_deadline and _pid_alive(pid):
                time.sleep(0.1)
    current = _read_json(os.path.join(_job_dir(root, job_id), "status.json"))
    if current.get("state") not in TERMINAL_STATES:
        finished_at = _utcnow()
        current.update(
            {
                "state": "cancelled",
                "finished_at": finished_at,
                "heartbeat_at": finished_at,
                "runner_alive": False,
            }
        )
        _atomic_json(os.path.join(_job_dir(root, job_id), "status.json"), current)
    return _status(root, job_id)


def dispatch(operation, payload):
    if operation == "submit":
        return _submit(payload)
    if operation == "jobs":
        return _list_jobs(payload)
    if operation == "status":
        return _status(payload["root"], payload["job_id"])
    if operation == "tail":
        return _tail(payload)
    if operation == "cancel":
        return _cancel(payload)
    raise ValueError(f"Unsupported remote job operation: {operation}")
