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

"""Reconnectable remote-job commands.

The commands use short kernel control requests.  The submitted process, logs,
and status files live in the Colab VM, so a later CLI process can inspect or
wait for the same job without keeping the original Jupyter request open.
"""

import re
import time
from typing import List, Optional

import typer
from typing_extensions import Annotated

from colab_cli.jobs import (
    DEFAULT_JOB_CONTROL_TIMEOUT,
    DEFAULT_JOB_ROOT,
    JobTail,
    RemoteJobClient,
)
from colab_cli.remote import open_remote_executor


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TERMINAL_STATES = {"succeeded", "failed", "cancelled", "lost"}


def _parse_env(values: Optional[List[str]]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid --env value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not _ENV_NAME.fullmatch(key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
        if key in parsed:
            raise ValueError(f"Duplicate environment variable: {key}")
        parsed[key] = value
    return parsed


def _open_client(session: Optional[str], job_root: str):
    from colab_cli.common import state

    name = state.resolve_session(session)
    remote_session = state.store.get(name)
    if not remote_session:
        raise LookupError(f"Session '{name}' not found")
    executor = open_remote_executor(remote_session, state.store, history=state.history)
    return name, executor, RemoteJobClient(executor, job_root=job_root)


def _echo_tail(tail: JobTail) -> None:
    if tail.data:
        typer.echo(
            tail.data.decode("utf-8", errors="replace"),
            nl=False,
            err=tail.stream == "stderr",
        )


def _drain_stream(
    client: RemoteJobClient,
    job_id: str,
    stream: str,
    offset: int,
    max_bytes: int,
    *,
    started: float | None = None,
    timeout: float | None = None,
) -> int:
    while True:
        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = _remaining_wait_timeout(started, timeout)
        tail = client.tail(
            job_id,
            stream=stream,
            offset=offset,
            max_bytes=max_bytes,
            **kwargs,
        )
        _echo_tail(tail)
        offset = tail.next_offset
        if tail.eof or not tail.data:
            return offset


def _remaining_wait_timeout(started: float | None, timeout: float) -> float:
    if started is None:
        raise ValueError("wait start time is required for a finite timeout")
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("local wait deadline exceeded")
    return min(DEFAULT_JOB_CONTROL_TIMEOUT, remaining)


def _remote_exit_code(status: dict) -> int:
    if status.get("state") == "succeeded":
        return 0
    if status.get("state") == "failed":
        returncode = status.get("returncode")
        if isinstance(returncode, int) and returncode != 0:
            return returncode
        return 1
    if status.get("state") == "cancelled":
        return 130
    return 1


def submit(
    command: Annotated[
        Optional[List[str]],
        typer.Argument(
            help=(
                "Command argv to run remotely. Use `--` before command options; "
                "shell parsing is never implicit."
            )
        ),
    ] = None,
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    name: Annotated[
        Optional[str], typer.Option("--name", help="Stable remote job id")
    ] = None,
    cwd: Annotated[
        str, typer.Option("--cwd", help="Remote working directory")
    ] = "/content",
    env: Annotated[
        Optional[List[str]],
        typer.Option("--env", help="Environment entry KEY=VALUE; repeatable"),
    ] = None,
    job_root: Annotated[
        str, typer.Option("--job-root", help="Remote job state directory")
    ] = DEFAULT_JOB_ROOT,
):
    """Submit a detached command whose state and logs can be reattached"""
    from colab_cli.common import state

    command = command or []
    if not command:
        typer.echo("[colab] Error: a command is required after `--`.", err=True)
        raise typer.Exit(2)
    try:
        parsed_env = _parse_env(env)
    except ValueError as exc:
        typer.echo(f"[colab] Error: {exc}", err=True)
        raise typer.Exit(2)

    executor = None
    try:
        session_name, executor, client = _open_client(session, job_root)
        result = client.submit(command, job_id=name, cwd=cwd, env=parsed_env)
        state.history.log_event(
            session_name,
            "remote_job",
            {
                "op": "submit",
                "job_id": result["job_id"],
                "argv": command,
                "cwd": cwd,
                "job_root": job_root,
            },
        )
        typer.echo(
            f"[colab] Submitted {result['job_id']} "
            f"(state={result['state']}, dir={result['job_dir']})"
        )
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"[colab] Submit failed: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        if executor is not None:
            executor.close()


def list_jobs(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    job_root: Annotated[
        str, typer.Option("--job-root", help="Remote job state directory")
    ] = DEFAULT_JOB_ROOT,
):
    """List persistent jobs in a session"""
    executor = None
    try:
        _, executor, client = _open_client(session, job_root)
        records = client.list_jobs()
        if not records:
            typer.echo("[colab] No remote jobs found.")
            return
        for record in records:
            returncode = record.get("returncode")
            rendered_returncode = "-" if returncode is None else str(returncode)
            typer.echo(f"{record['job_id']}\t{record['state']}\t{rendered_returncode}")
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"[colab] Jobs query failed: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        if executor is not None:
            executor.close()


def tail(
    job_id: Annotated[str, typer.Argument(help="Remote job id")],
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    stream: Annotated[
        str, typer.Option("--stream", help="stdout or stderr")
    ] = "stdout",
    offset: Annotated[
        int, typer.Option("--offset", help="Byte offset to resume from")
    ] = 0,
    max_bytes: Annotated[
        int, typer.Option("--max-bytes", help="Maximum bytes to read")
    ] = 65536,
    job_root: Annotated[
        str, typer.Option("--job-root", help="Remote job state directory")
    ] = DEFAULT_JOB_ROOT,
):
    """Read one bounded chunk from a persisted job log"""
    if stream not in {"stdout", "stderr"} or offset < 0 or max_bytes <= 0:
        typer.echo(
            "[colab] Error: stream must be stdout/stderr, offset >= 0, "
            "and max-bytes > 0.",
            err=True,
        )
        raise typer.Exit(2)
    executor = None
    try:
        _, executor, client = _open_client(session, job_root)
        result = client.tail(job_id, stream=stream, offset=offset, max_bytes=max_bytes)
        _echo_tail(result)
        typer.echo(
            f"[colab] next_offset={result.next_offset} "
            f"size={result.size} eof={str(result.eof).lower()}",
            err=True,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"[colab] Tail failed: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        if executor is not None:
            executor.close()


def wait(
    job_id: Annotated[str, typer.Argument(help="Remote job id")],
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    poll_seconds: Annotated[
        float, typer.Option("--poll-seconds", help="Status polling interval")
    ] = 2.0,
    timeout: Annotated[
        Optional[float],
        typer.Option(
            "--timeout",
            help="Local wait timeout; the remote job is not cancelled",
        ),
    ] = None,
    max_bytes: Annotated[
        int, typer.Option("--max-bytes", help="Maximum bytes per log request")
    ] = 65536,
    stdout_offset: Annotated[
        int, typer.Option("--stdout-offset", help="Initial stdout byte offset")
    ] = 0,
    stderr_offset: Annotated[
        int, typer.Option("--stderr-offset", help="Initial stderr byte offset")
    ] = 0,
    job_root: Annotated[
        str, typer.Option("--job-root", help="Remote job state directory")
    ] = DEFAULT_JOB_ROOT,
):
    """Reconnect, stream persisted logs, and wait for terminal job state"""
    from colab_cli.common import state

    if (
        poll_seconds <= 0
        or max_bytes <= 0
        or stdout_offset < 0
        or stderr_offset < 0
        or (timeout is not None and timeout < 0)
    ):
        typer.echo(
            "[colab] Error: poll-seconds/max-bytes must be positive and offsets/"
            "timeout must be non-negative.",
            err=True,
        )
        raise typer.Exit(2)

    executor = None
    started = time.monotonic()
    try:
        session_name, executor, client = _open_client(session, job_root)
        while True:
            status_kwargs = {}
            if timeout is not None:
                status_kwargs["timeout"] = _remaining_wait_timeout(started, timeout)
            status = client.status(job_id, **status_kwargs)
            stdout_offset = _drain_stream(
                client,
                job_id,
                "stdout",
                stdout_offset,
                max_bytes,
                started=started,
                timeout=timeout,
            )
            stderr_offset = _drain_stream(
                client,
                job_id,
                "stderr",
                stderr_offset,
                max_bytes,
                started=started,
                timeout=timeout,
            )
            if status.get("state") in _TERMINAL_STATES:
                exit_code = _remote_exit_code(status)
                state.history.log_event(
                    session_name,
                    "remote_job",
                    {
                        "op": "wait",
                        "job_id": job_id,
                        "state": status.get("state"),
                        "returncode": status.get("returncode"),
                    },
                )
                typer.echo(
                    f"[colab] Job {job_id} finished: {status.get('state')}",
                    err=True,
                )
                raise typer.Exit(exit_code)
            if timeout is not None and time.monotonic() - started >= timeout:
                typer.echo(
                    f"[colab] Wait timed out; remote job {job_id} is still running.",
                    err=True,
                )
                raise typer.Exit(124)
            time.sleep(poll_seconds)
    except typer.Exit:
        raise
    except TimeoutError as exc:
        if timeout is not None:
            typer.echo(
                f"[colab] Wait timed out; remote job {job_id} is still running.",
                err=True,
            )
            raise typer.Exit(124)
        typer.echo(f"[colab] Wait failed: {exc}", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"[colab] Wait failed: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        if executor is not None:
            executor.close()


def cancel(
    job_id: Annotated[str, typer.Argument(help="Remote job id")],
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    grace_seconds: Annotated[
        float, typer.Option("--grace-seconds", help="Seconds before force kill")
    ] = 10.0,
    job_root: Annotated[
        str, typer.Option("--job-root", help="Remote job state directory")
    ] = DEFAULT_JOB_ROOT,
):
    """Cancel a persistent job"""
    from colab_cli.common import state

    if grace_seconds < 0:
        typer.echo("[colab] Error: grace-seconds must be non-negative.", err=True)
        raise typer.Exit(2)
    executor = None
    try:
        session_name, executor, client = _open_client(session, job_root)
        status = client.cancel(job_id, grace_seconds=grace_seconds)
        state.history.log_event(
            session_name,
            "remote_job",
            {"op": "cancel", "job_id": job_id, "state": status.get("state")},
        )
        typer.echo(f"[colab] Job {job_id}: {status.get('state')}")
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"[colab] Cancel failed: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        if executor is not None:
            executor.close()


def register(app: typer.Typer):
    app.command(
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
    )(submit)
    app.command(name="jobs")(list_jobs)
    app.command()(tail)
    app.command()(wait)
    app.command()(cancel)
