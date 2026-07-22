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

import click
import hashlib
import math
import os
import tempfile
import typer
from typing import Optional
from typing_extensions import Annotated

from colab_cli.contents import ContentsClient
from colab_cli.remote import RemoteFileOps, open_remote_executor
from colab_cli.transfer import DEFAULT_CHUNK_SIZE, FileTransfer, TransferProgress


_MIB = 1024 * 1024


def _chunk_size_mib_to_bytes(value: float) -> int:
    """Validate a MiB chunk size and return its integral byte count.

    Validation happens before session lookup or executor construction so bad
    input cannot open a remote control channel as a side effect.
    """
    if not math.isfinite(value):
        raise typer.BadParameter("must be a finite number")
    if value <= 0:
        raise typer.BadParameter("must be greater than 0")

    size_in_bytes = value * _MIB
    if not math.isfinite(size_in_bytes):
        raise typer.BadParameter("is too large")

    chunk_size = int(size_in_bytes)
    if chunk_size < 1:
        raise typer.BadParameter("must convert to at least 1 byte")
    return chunk_size


def _validate_chunk_size_mib(value: float) -> float:
    _chunk_size_mib_to_bytes(value)
    return value


def _progress(progress: TransferProgress) -> None:
    if progress.total:
        percent = 100.0 * progress.completed / progress.total
    else:
        percent = 100.0
    typer.echo(
        f"[colab] {progress.direction} {percent:5.1f}% "
        f"({progress.completed}/{progress.total} bytes)",
        err=True,
    )


def _open_transfer(session, state, *, chunk_size_mib: float):
    chunk_size = _chunk_size_mib_to_bytes(chunk_size_mib)
    executor = open_remote_executor(session, state.store, history=state.history)
    transfer = FileTransfer(
        ContentsClient(session),
        RemoteFileOps(executor),
        chunk_size=chunk_size,
        progress=_progress,
    )
    return executor, transfer


def ls(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    path: Annotated[str, typer.Argument(help="Remote path to list")] = "content",
):
    """List files in a session"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)
    contents = ContentsClient(s)
    try:
        data = contents.list_dir(path)
        state.history.log_event(name, "file_operation", {"op": "ls", "path": path})
        if data.get("type") == "directory":
            items = data.get("content", [])
            for item in sorted(
                items, key=lambda x: (x.get("type") != "directory", x.get("name"))
            ):
                suffix = "/" if item.get("type") == "directory" else ""
                typer.echo(f"{item.get('name')}{suffix}")
        else:
            typer.echo(data.get("name"))
    except Exception as e:
        typer.echo(f"[colab] Error: {e}")
        raise typer.Exit(1)


def rm(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    path: Annotated[str, typer.Argument(help="Remote path to remove")] = ...,
):
    """Remove a remote file"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)
    contents = ContentsClient(s)
    try:
        contents.rm(path)
        state.history.log_event(name, "file_operation", {"op": "rm", "path": path})
        typer.echo(f"[colab] Deleted {path}")
    except Exception as e:
        typer.echo(f"[colab] Error: {e}")
        raise typer.Exit(1)


def upload(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    local_path: Annotated[str, typer.Argument(help="Local file to upload")] = ...,
    remote_path: Annotated[str, typer.Argument(help="Remote path to upload to")] = ...,
    chunk_size_mib: Annotated[
        float,
        typer.Option(
            "--chunk-size-mib",
            help="Bounded transfer chunk size in MiB",
            callback=_validate_chunk_size_mib,
        ),
    ] = DEFAULT_CHUNK_SIZE / (1024 * 1024),
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume a verified partial upload"),
    ] = True,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite/--no-overwrite", help="Replace an existing remote file"
        ),
    ] = True,
):
    """Upload a file to a session"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)
    if not os.path.isfile(local_path):
        typer.echo(f"[colab] Local file '{local_path}' not found.")
        raise typer.Exit(1)
    executor, transfer = _open_transfer(s, state, chunk_size_mib=chunk_size_mib)
    try:
        result = transfer.upload(
            local_path, remote_path, overwrite=overwrite, resume=resume
        )
        state.history.log_event(
            name,
            "file_operation",
            {
                "op": "upload",
                "local": local_path,
                "remote": remote_path,
                "size": result.size,
                "sha256": result.sha256,
                "resumed_from": result.resumed_from,
            },
        )
        typer.echo(
            f"[colab] Uploaded '{local_path}' to '{remote_path}' "
            f"({result.size} bytes, sha256={result.sha256})"
        )
    except Exception as e:
        typer.echo(f"[colab] Upload failed: {e}")
        raise typer.Exit(1)
    finally:
        executor.close()


def download(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    remote_path: Annotated[
        str, typer.Argument(help="Remote path to download from")
    ] = ...,
    local_path: Annotated[
        str, typer.Argument(help="Local path to save the file")
    ] = ...,
    chunk_size_mib: Annotated[
        float,
        typer.Option(
            "--chunk-size-mib",
            help="Bounded transfer chunk size in MiB",
            callback=_validate_chunk_size_mib,
        ),
    ] = DEFAULT_CHUNK_SIZE / (1024 * 1024),
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume a verified partial download"),
    ] = True,
):
    """Download a file from a session"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)
    executor, transfer = _open_transfer(s, state, chunk_size_mib=chunk_size_mib)
    try:
        result = transfer.download(remote_path, local_path, resume=resume)
        state.history.log_event(
            name,
            "file_operation",
            {
                "op": "download",
                "remote": remote_path,
                "local": local_path,
                "size": result.size,
                "sha256": result.sha256,
                "resumed_from": result.resumed_from,
            },
        )
        typer.echo(
            f"[colab] Downloaded '{remote_path}' to '{local_path}' "
            f"({result.size} bytes, sha256={result.sha256})"
        )
    except Exception as e:
        typer.echo(f"[colab] Download failed: {e}")
        raise typer.Exit(1)
    finally:
        executor.close()


def edit(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    remote_path: Annotated[str, typer.Argument(help="Remote path to edit")] = ...,
):
    """Edit a file on a running Colab session"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)

    executor, transfer = _open_transfer(s, state, chunk_size_mib=1)

    def get_file_hash(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    _, ext = os.path.splitext(remote_path)

    # `delete=False` + close-before-reopen: on Windows, NamedTemporaryFile
    # holds an exclusive handle that makes `open(path, "rb")` (for hashing)
    # and the editor fail with PermissionError. Close it and clean up manually.
    tf = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    local_path = tf.name
    tf.close()
    try:
        try:
            transfer.download(remote_path, local_path, resume=False)
        except FileNotFoundError:
            # A missing remote path is the documented create-new-file case.
            open(local_path, "wb").close()

        hash_before = get_file_hash(local_path)

        click.edit(filename=local_path)

        hash_after = get_file_hash(local_path)

        if hash_after != hash_before:
            result = transfer.upload(
                local_path, remote_path, overwrite=True, resume=True
            )
            state.history.log_event(
                name,
                "file_operation",
                {
                    "op": "edit",
                    "remote": remote_path,
                    "size": result.size,
                    "sha256": result.sha256,
                },
            )
            typer.echo(f"[colab] Edited and uploaded '{remote_path}'")
        else:
            typer.echo(f"[colab] No changes made to '{remote_path}'")
    finally:
        executor.close()
        try:
            os.unlink(local_path)
        except OSError:
            pass


def register(app: typer.Typer):
    app.command()(ls)
    app.command()(rm)
    app.command()(upload)
    app.command()(download)
    app.command()(edit)
