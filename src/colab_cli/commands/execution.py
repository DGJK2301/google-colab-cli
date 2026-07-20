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

import datetime
import logging
import nbformat
import os
import re
import sys
import typer
import uuid
from nbformat.v4 import new_output
from rich.console import Console
from typing import Optional
from typing_extensions import Annotated

from colab_cli.runtime import ColabRuntime
from colab_cli.timeouts import (
    TIMEOUT_EXIT_CODE,
    format_execution_timeout,
    validate_execution_timeout,
)
from colab_cli.utils import handle_image, is_terminal_error, render_display_data
from colab_cli.console import connect_console

_console = Console()

TITLE_REGEX = re.compile(r"^\s*#\s*@title\s+(.*)", re.MULTILINE)


def is_stdin_tty():
    return sys.stdin.isatty()


def save_output(outputs, cell):
    if cell is None:
        return

    if not hasattr(cell, "outputs"):
        cell.outputs = []
    else:
        cell.outputs.clear()

    for out in outputs:
        if out.get("output_type") == "stream":
            cell.outputs.append(
                new_output(
                    output_type="stream",
                    name=out.get("name", "stdout"),
                    text=out.get("text", ""),
                )
            )
        elif "data" in out:
            output_type = out.get("output_type", "display_data")
            cell.outputs.append(
                new_output(
                    output_type=output_type,
                    data=out["data"],
                    metadata=out.get("metadata", {}),
                )
            )
        elif out.get("output_type") == "error":
            cell.outputs.append(
                new_output(
                    output_type="error",
                    ename=out.get("ename", "Error"),
                    evalue=out.get("evalue", ""),
                    traceback=out.get("traceback", []),
                )
            )


def display_output(out, output_image=None):
    if out.get("output_type") == "stream":
        stream = sys.stderr if out.get("name") == "stderr" else sys.stdout
        stream.write(out.get("text", ""))
        stream.flush()
    elif "data" in out:
        data = out["data"]
        text = render_display_data(data)
        if text is not None:
            _console.print(text)
        if png := data.get("image/png"):
            handle_image(png, "image/png", target_path=output_image)
        elif jpeg := data.get("image/jpeg"):
            handle_image(jpeg, "image/jpeg", target_path=output_image)
    elif out.get("output_type") == "error":
        tb = out.get("traceback", [])
        if tb:
            sys.stderr.write("".join(tb) + "\n")
        else:
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            sys.stderr.write(f"{ename}: {evalue}\n")
    else:
        # Ignore silent outputs like metadata or clear_output for streaming
        pass


def select_notebook_code_blocks(code_blocks, requested_titles):
    if not requested_titles:
        return code_blocks

    if len(requested_titles) != len(set(requested_titles)):
        typer.echo(
            "[colab] Error: Duplicate --cell-title values are not allowed.", err=True
        )
        raise typer.Exit(2)

    title_counts = {}
    for block in code_blocks:
        title = block.get("title")
        if title:
            title_counts[title] = title_counts.get(title, 0) + 1

    for title in requested_titles:
        count = title_counts.get(title, 0)
        if count == 0:
            typer.echo(
                f"[colab] Error: Notebook cell title '{title}' was not found.",
                err=True,
            )
            raise typer.Exit(2)
        if count > 1:
            typer.echo(
                f"[colab] Error: Notebook cell title '{title}' is ambiguous "
                f"({count} matches).",
                err=True,
            )
            raise typer.Exit(2)

    requested = set(requested_titles)
    return [block for block in code_blocks if block.get("title") in requested]


def exec_command(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    file: Annotated[
        Optional[str], typer.Option("-f", "--file", help="File to execute")
    ] = None,
    output_image: Annotated[
        Optional[str], typer.Option("--output-image", help="Path to save plot")
    ] = None,
    timeout: Annotated[
        Optional[float],
        typer.Option("--timeout", help="Timeout in seconds for code execution"),
    ] = 30.0,
    cell_title: Annotated[
        Optional[list[str]],
        typer.Option(
            "--cell-title",
            help="Execute the uniquely titled notebook cell; repeat to select more",
        ),
    ] = None,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error",
            help="Stop at the first Jupyter error output and return a nonzero exit code",
        ),
    ] = False,
):
    """Execute code in a session"""
    from colab_cli.common import state

    try:
        timeout = validate_execution_timeout(timeout)
    except ValueError as error:
        typer.echo(f"[colab] {error}", err=True)
        raise typer.Exit(code=2) from error

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)

    code_blocks = []
    if file:
        if file.endswith(".ipynb"):
            typer.echo(f"[colab] Parsing notebook '{file}'...")
            with open(file, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
                for cell in nb.cells:
                    # nbformat v4.5+ requires 'id' at the top level
                    if not hasattr(cell, "id") or not cell.id:
                        cell.id = str(uuid.uuid4())

                    if cell.cell_type == "code":
                        title_match = TITLE_REGEX.search(cell.source)
                        code_blocks.append(
                            {
                                "code": cell.source,
                                "id": cell.id,
                                "cell": cell,
                                "title": (
                                    title_match.group(1).strip()
                                    if title_match
                                    else None
                                ),
                            }
                        )
            code_blocks = select_notebook_code_blocks(code_blocks, cell_title)
        else:
            if cell_title:
                typer.echo(
                    "[colab] Error: --cell-title is only valid for .ipynb files.",
                    err=True,
                )
                raise typer.Exit(2)
            with open(file, "r") as f:
                code_blocks.append({"code": f.read(), "id": None})
    else:
        if cell_title:
            typer.echo("[colab] Error: --cell-title requires a .ipynb file.", err=True)
            raise typer.Exit(2)
        if is_stdin_tty():
            typer.echo("[colab] Error: No input provided. Pipe code or provide a file.")
            raise typer.Exit(1)
        code_blocks.append({"code": sys.stdin.read(), "id": None})

    if not any(b["code"].strip() for b in code_blocks):
        raise typer.Exit(0)

    def on_started(kid):
        s.kernel_id = kid
        state.store.add(s)

    def on_sess_started(sid):
        s.session_id = sid
        state.store.add(s)

    runtime = ColabRuntime(
        s.url,
        s.token,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
        on_kernel_started=on_started,
        on_session_started=on_sess_started,
    )
    try:
        # Ensure we are in /content which is the standard Colab working directory
        runtime.execute_code(
            "import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')",
            timeout=timeout,
        )
    except TimeoutError:
        try:
            state.history.log_event(
                name,
                "execution_timeout",
                {"phase": "prelude", "timeout": timeout},
            )
        except Exception as history_error:
            logging.debug("Failed to record execution timeout: %s", history_error)
        typer.echo(
            format_execution_timeout(timeout, remote_may_continue=True), err=True
        )
        runtime.stop()
        raise typer.Exit(code=TIMEOUT_EXIT_CODE) from None
    except Exception as e:
        runtime.stop()
        if is_terminal_error(e):
            typer.echo(
                f"[colab] Session '{name}' appears to be lost (404/401). Cleaning up."
            )
            state.prune_session(name)
            raise typer.Exit(1)
        raise

    try:
        is_nb = file and file.endswith(".ipynb")
        s.running = f"exec({file or 'stdin'})"
        state.store.add(s)

        for i, block in enumerate(code_blocks):
            code = block["code"]
            identifier = None
            if is_nb:
                if block.get("title"):
                    identifier = block["title"]
                elif block.get("id"):
                    identifier = block["id"]
                else:
                    identifier = ""

                identifier_str = f" - {identifier}" if identifier else ""
                typer.echo(
                    f"[colab] Executing cell {i + 1}/{len(code_blocks)}{identifier_str}..."
                )

            s.last_execution = (
                file or "stdin",
                identifier,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            state.store.add(s)

            try:
                outputs = runtime.execute_code(
                    code,
                    output_hook=lambda o: display_output(o, output_image),
                    timeout=timeout,
                )
            except TimeoutError:
                try:
                    state.history.log_event(
                        name,
                        "execution_timeout",
                        {
                            "phase": "body",
                            "timeout": timeout,
                            "cell_index": i if len(code_blocks) > 1 else None,
                            "cell_id": block.get("id"),
                        },
                    )
                except Exception as history_error:
                    logging.debug(
                        "Failed to record execution timeout: %s", history_error
                    )
                typer.echo(
                    format_execution_timeout(timeout, remote_may_continue=True),
                    err=True,
                )
                raise typer.Exit(code=TIMEOUT_EXIT_CODE) from None
            if "cell" in block:
                save_output(outputs, block["cell"])
            state.history.log_event(
                name,
                "execution",
                {
                    "code": code,
                    "outputs": outputs,
                    "cell_index": i if len(code_blocks) > 1 else None,
                    "cell_id": block.get("id"),
                },
            )
            if fail_on_error and any(
                output.get("output_type") == "error" for output in outputs
            ):
                failed_at = identifier or file or "stdin"
                typer.echo(
                    f"[colab] Error: Cell execution failed ({failed_at}).",
                    err=True,
                )
                raise typer.Exit(1)
    finally:
        primary_error_active = sys.exc_info()[0] is not None
        cleanup_error = None
        s.running = None
        try:
            state.store.add(s)
        except Exception as error:
            cleanup_error = error
            typer.echo(
                f"[colab] Failed to persist final session state: {error}", err=True
            )
        runtime.stop()
        if file and file.endswith(".ipynb"):
            output_file = os.path.splitext(file)[0] + "_output.ipynb"
            typer.echo(f"[colab] Saving notebook with outputs to '{output_file}'...")
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    nbformat.write(nb, f)
            except Exception as error:
                if not primary_error_active and cleanup_error is None:
                    raise
                typer.echo(
                    f"[colab] Failed to save notebook outputs: {error}", err=True
                )
        if cleanup_error is not None and not primary_error_active:
            raise cleanup_error


def repl(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    output_image: Annotated[
        Optional[str], typer.Option("--output-image", help="Path to save plot")
    ] = None,
):
    """Start an interactive REPL"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)

    def on_started(kid):
        s.kernel_id = kid
        state.store.add(s)

    def on_sess_started(sid):
        s.session_id = sid
        state.store.add(s)

    runtime = ColabRuntime(
        s.url,
        s.token,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
        on_kernel_started=on_started,
        on_session_started=on_sess_started,
    )
    try:
        # Ensure we are in /content which is the standard Colab working directory
        runtime.execute_code(
            "import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')"
        )
    except Exception as e:
        if is_terminal_error(e):
            typer.echo(
                f"[colab] Session '{name}' appears to be lost (404/401). Cleaning up."
            )
            state.prune_session(name)
            raise typer.Exit(1)
        raise e

    if not is_stdin_tty():
        code = sys.stdin.read()
        if not code.strip():
            raise typer.Exit(0)

        s.last_execution = (
            "stdin",
            None,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        s.running = "repl(stdin)"
        state.store.add(s)
        try:
            outputs = runtime.execute_code(
                code, output_hook=lambda o: display_output(o, output_image)
            )
            state.history.log_event(
                name, "execution", {"code": code, "outputs": outputs, "source": "piped"}
            )
        finally:
            s.running = None
            state.store.add(s)
            runtime.stop()
    else:
        from colab_cli.repl import ColabREPL

        s.running = "repl"
        state.store.add(s)
        try:
            repl_inst = ColabREPL(
                runtime,
                session_name=s.name,
                history_logger=state.history,
                output_image=output_image,
            )
            state.history.log_event(name, "repl_started", {})
            repl_inst.run()
        finally:
            s.running = None
            state.store.add(s)


def console(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
):
    """Connect to raw TTY console"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    s = state.store.get(name)
    if not s:
        typer.echo(f"[colab] Session '{name}' not found.")
        raise typer.Exit(1)
    state.history.log_event(s.name, "console_started", {})
    s.running = "console"
    state.store.add(s)
    try:
        connect_console(s)
    except Exception as e:
        if is_terminal_error(e):
            typer.echo(
                f"[colab] Session '{name}' appears to be lost (404/401). Cleaning up."
            )
            state.prune_session(name)
            raise typer.Exit(1)
        raise e
    finally:
        s.running = None
        state.store.add(s)


def register(app: typer.Typer):
    app.command(name="exec")(exec_command)
    app.command()(repl)
    app.command()(console)
