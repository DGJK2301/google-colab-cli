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
import json
import logging
import os
import sys
import threading
import webbrowser
from typing import List, Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from colab_cli.auth import get_credentials
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
from colab_cli.utils import get_status_code, render_display_data

_console = Console()
_logger = logging.getLogger(__name__)


def _read_line_from_controlling_tty() -> str:
    """Read a single line from the controlling terminal, cross-platform.

    On POSIX we open ``/dev/tty`` directly so the prompt is answered even when
    stdin is piped/redirected (e.g. ``colab drivemount`` under a non-interactive
    harness). Windows has no ``/dev/tty``; fall back to ``sys.stdin``.
    """
    try:
        with open("/dev/tty") as tty:
            return tty.readline()
    except OSError:
        # No /dev/tty (Windows) or not readable — use the regular stdin.
        return sys.stdin.readline()


# Default execute() timeout for human-in-the-loop automations (auth /
# drivemount). The kernel goes silent while the user completes a browser
# OAuth flow, which can routinely take 30s+; the upstream 10s default
# raises ``TimeoutError`` mid-flow even though the mount actually succeeds.
# The remote Drive helper otherwise stops waiting for auth after 120 seconds,
# even if this CLI is still waiting. Keep the remote and local deadlines
# aligned, with one minute for DriveFS startup and reply delivery.
DRIVE_MOUNT_TIMEOUT_MS = 10 * 60 * 1000
INTERACTIVE_AUTOMATION_TIMEOUT_SEC = DRIVE_MOUNT_TIMEOUT_MS // 1000 + 60


def _send_drivefs_reply(deserialize_msg, wsclient, error: Optional[str] = None):
    msg_id = deserialize_msg.get("metadata", {}).get("colab_msg_id")
    value = {"type": "colab_reply", "colab_msg_id": msg_id}
    if error:
        value["error"] = error
    reply = wsclient.session.msg("input_reply", {"value": value})
    if "header" in deserialize_msg:
        reply["parent_header"] = deserialize_msg["header"]
    wsclient.stdin_channel.send(reply)


def _log_drive_event(state, session_name, event_type, details):
    try:
        state.history.log_event(session_name, event_type, details)
    except Exception:
        _logger.debug("Failed to write Drive authorization history", exc_info=True)


def _handle_drivefs_auth(state, session, deserialize_msg, wsclient):
    """Complete one Drive consent request outside the WebSocket callback."""

    try:
        msg_id = deserialize_msg.get("metadata", {}).get("colab_msg_id")
        _log_drive_event(
            state,
            session.name,
            "colab_request",
            {"type": "dfs_ephemeral", "colab_msg_id": msg_id},
        )
        url = (
            f"{state.client.colab_domain}/tun/m/credentials-propagation/"
            f"{session.endpoint}"
        )
        params = {
            "authuser": "0",
            "authtype": "dfs_ephemeral",
            "version": "2",
            "dryrun": "true",
            "propagate": "true",
            "record": "false",
        }
        typer.echo(
            "\n[colab] Intercepted Drive Auth Request. Connecting to "
            f"{state.client.colab_domain}..."
        )

        creds = get_credentials(state.client_oauth_config, provider=state.auth_provider)
        resp = creds.request("GET", url, params=params)
        token = (
            json.loads(resp.text.split("\n", 1)[-1]).get("token")
            if get_status_code(resp) == 200
            else None
        )
        if not token:
            raise RuntimeError("Drive credential propagation token is unavailable")

        headers = {"x-goog-colab-token": token}
        resp = creds.request(
            "POST",
            url,
            params=params,
            headers=headers,
            files={"file_id": (None, "empty.ipynb")},
        )
        data = json.loads(resp.text.split("\n", 1)[-1])

        if not data.get("success"):
            uri = data.get("unauthorized_redirect_uri")
            if not uri:
                raise RuntimeError("Drive authorization URL is unavailable")
            typer.echo(
                "\n[colab] REQUIRED: Google Drive Authorization needed.\n"
                f"Please visit:\n\n{uri}\n"
            )
            # The consent URL can contain short-lived authorization state. It
            # belongs on the interactive console, not in durable history.
            _log_drive_event(state, session.name, "drive_auth_needed", {})
            try:
                webbrowser.open(uri, new=2)
            except (OSError, webbrowser.Error):
                pass
            sys.stdout.write("Press Enter after you have granted access... ")
            sys.stdout.flush()
            _read_line_from_controlling_tty()

        typer.echo("[colab] Authorizing VM...")
        params["dryrun"] = "false"
        resp = creds.request(
            "POST",
            url,
            params=params,
            headers=headers,
            files={"file_id": (None, "empty.ipynb")},
        )
        data = json.loads(resp.text.split("\n", 1)[-1])
        if get_status_code(resp) != 200 or not data.get("success"):
            raise RuntimeError("Drive credential propagation was unsuccessful")

        typer.echo("[colab] Credentials propagated. Resuming mount...")
        _log_drive_event(state, session.name, "drive_auth_success", {})
        _send_drivefs_reply(deserialize_msg, wsclient)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        typer.echo(f"[colab] Drive authorization failed: {error}", err=True)
        _log_drive_event(
            state,
            session.name,
            "drive_auth_error",
            {"error_type": type(exc).__name__},
        )
        _send_drivefs_reply(deserialize_msg, wsclient, error=error)


def run_automation(
    name: str,
    op: str,
    code: str,
    allow_stdin: bool = False,
    path: str = None,
    timeout: Optional[float] = None,
):
    from colab_cli.common import state

    s = state.store.get(name)

    def on_started(kernel_id):
        s.kernel_id = kernel_id
        state.store.add(s)

    def on_session_started(session_id):
        s.session_id = session_id
        state.store.add(s)

    runtime = ColabRuntime(
        s.url,
        s.token,
        session_name=s.name,
        history=state.history,
        kernel_id=s.kernel_id,
        session_id=s.session_id,
        on_kernel_started=on_started,
        on_session_started=on_session_started,
    )

    def drivefs_hook(deserialize_msg, wsclient):
        content = deserialize_msg.get("content", {})
        if content.get("request", {}).get("authType") == "dfs_ephemeral":
            threading.Thread(
                target=_handle_drivefs_auth,
                args=(state, s, deserialize_msg, wsclient),
                name=f"colab-drive-auth-{s.name}",
                daemon=True,
            ).start()
            return True
        return False

    runtime.colab_request_hook = drivefs_hook
    try:
        s.running = f"automation({op})"
        s.last_execution = (
            f"automation:{op}",
            None,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        state.store.add(s)

        if op == "drivemount":
            state.history.log_event(
                name, "automation", {"op": "drivemount", "path": path, "code": code}
            )
        else:
            state.history.log_event(name, "automation", {"op": op, "code": code})

        outputs = runtime.execute_code(code, allow_stdin=allow_stdin, timeout=timeout)
        state.history.log_event(
            name, "automation_result", {"op": op, "outputs": outputs}
        )

        for out in outputs:
            if "text" in out:
                sys.stdout.write(out["text"])
            elif "data" in out:
                text = render_display_data(out["data"])
                if text is not None:
                    _console.print(text)
            elif out.get("output_type") == "error":
                ename = out.get("ename", "Error")
                evalue = out.get("evalue", "")
                tb = out.get("traceback", [])
                if tb:
                    sys.stderr.write("".join(tb) + "\n")
                else:
                    sys.stderr.write(f"{ename}: {evalue}\n")
    finally:
        s.running = None
        state.store.add(s)
        runtime.stop()


def auth(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
):
    """Authenticate with Google on the VM"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    code = "import os\nos.environ['USE_AUTH_EPHEM'] = '0'\nfrom google.colab import auth\nauth.authenticate_user()"
    typer.echo(f"[colab] Starting Google Auth flow on {name}...")
    run_automation(
        name,
        "auth",
        code,
        allow_stdin=True,
        timeout=INTERACTIVE_AUTOMATION_TIMEOUT_SEC,
    )


def drivemount(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    path: Annotated[str, typer.Argument(help="Mount path")] = "/content/drive",
):
    """Mount Google Drive at path"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    code = (
        "from google.colab import drive\n"
        f"drive.mount({path!r}, timeout_ms={DRIVE_MOUNT_TIMEOUT_MS})"
    )
    typer.echo(f"[colab] Mounting Google Drive to '{path}' on {name}...")
    run_automation(
        name,
        "drivemount",
        code,
        allow_stdin=True,
        path=path,
        timeout=INTERACTIVE_AUTOMATION_TIMEOUT_SEC,
    )


def install(
    session: Annotated[
        Optional[str], typer.Option("-s", "--session", help="Session name")
    ] = None,
    packages: Annotated[
        Optional[List[str]], typer.Argument(help="Packages to install")
    ] = None,
    requirement: Annotated[
        Optional[str], typer.Option("-r", "--requirement", help="Requirements file")
    ] = None,
):
    """Install python packages on the VM"""
    from colab_cli.common import state

    name = state.resolve_session(session)
    if not packages and not requirement:
        typer.echo("[colab] No packages or requirements specified.")
        raise typer.Exit(1)

    commands = []
    if requirement:
        if not os.path.isfile(requirement):
            typer.echo(f"[colab] Requirements file '{requirement}' not found locally.")
            raise typer.Exit(1)
        remote_path = f"content/{os.path.basename(requirement)}"
        with ContentsClient(state.store.get(name)) as contents:
            contents.upload(requirement, remote_path)
        commands.extend(["-r", f"/{remote_path}"])
    if packages:
        commands.extend(packages)

    cmd_str = ", ".join(f"'{c}'" for c in commands)
    code = f"""
import subprocess, sys
def install():
    packages = [{cmd_str}]
    try:
        subprocess.check_call(['uv', 'pip', 'install', '--system'] + packages)
        print('Installation Complete (via uv)!')
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print('Installation Complete (via pip)!')
install()
"""
    typer.echo(f"[colab] Installing packages on {name} (preferring uv)...")
    run_automation(name, "install", code)


def register(app: typer.Typer):
    app.command(hidden=True)(auth)
    app.command()(drivemount)
    app.command()(install)
