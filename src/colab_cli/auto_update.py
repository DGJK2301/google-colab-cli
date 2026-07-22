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

"""Auto-update subsystem.

Owns version detection, the release update probe, the on-disk
``latest_version`` cache, and the upgrade-banner UX. The CLI's global
callback (``cli.py``) calls ``check_for_updates`` once per day and
``maybe_show_cached_banner`` on every other invocation; the
``colab update`` Typer command (``commands/utility.py``) delegates to
``check_for_updates``.
"""

import json
import platform
import subprocess
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as installed_version
from packaging.version import InvalidVersion, Version
from typing import Optional

import typer

from colab_cli.common import state
from colab_cli.state import (
    DEFAULT_UPDATE_URL,
    LEGACY_UPSTREAM_UPDATE_URL,
    Settings,
)

DISTRIBUTION_NAME = "google-colab-cli"
FORK_REPOSITORY_URL = "https://github.com/DGJK2301/google-colab-cli.git"


# ---------- Version detection -------------------------------------------


def is_self_install_supported() -> bool:
    """Return True if self-install (--install) is supported on the current platform."""
    return platform.system() in ("Linux", "Darwin")


def get_app_version() -> str:
    """Return the installed package version, falling back to the git short hash."""
    try:
        return installed_version("google-colab-cli")
    except (PackageNotFoundError, InvalidVersion):
        pass

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "unknown"


# ---------- Source fetchers ---------------------------------------------


def _parse_version(payload: Optional[dict]) -> Optional[str]:
    """Return a validated PEP 440 version from PyPI or GitHub release JSON."""
    payload = payload or {}
    raw = payload.get("info", {}).get("version") or payload.get("tag_name")
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    if candidate.startswith("v"):
        candidate = candidate[1:]
    try:
        return str(Version(candidate))
    except InvalidVersion:
        return None


def _fetch_release(url: str, quiet: bool) -> Optional[dict]:
    """Fetch and parse a release JSON document at ``url``."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        if not quiet:
            typer.echo(f"[colab] Warning: Failed to fetch update info: {e}")
        return None


# ---------- Version comparison ------------------------------------------


def _is_newer(candidate: Optional[str], current: str) -> bool:
    """True when ``candidate`` strictly exceeds ``current`` (PEP 440)."""
    if not candidate:
        return False
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return candidate != current


# ---------- UX ----------------------------------------------------------


def announce_upgrade(
    latest: str,
    current: str,
    install_cmd: str,
    *,
    show_disable_hint: bool = False,
) -> None:
    """Print the upgrade banner.

    ``show_disable_hint`` controls whether the trailing line that explains
    how to silence the auto-check is included. It is only added when the
    banner is shown unsolicited (the daily background fetch and the cached
    banner on subsequent invocations); explicit ``colab update`` calls
    omit it because the user already opted in to seeing the result.
    """
    typer.echo(
        f"\n[colab] A new version of Colab CLI is available: {latest} (current: {current})"
    )
    if is_self_install_supported() and ("pip" in install_cmd or "uv" in install_cmd):
        typer.echo("[colab] You can run 'colab update --install' to upgrade in place.")
    typer.echo(f"[colab] Run '{install_cmd}' to update.")
    if show_disable_hint:
        typer.echo(
            "[colab] To silence this check, set "
            '"enable_update_check": false in '
            "~/.config/colab-cli/settings.json"
        )
    typer.echo("")


# ---------- Orchestration -----------------------------------------------


def _release_install_spec(version: str) -> str:
    """Return the exact audited fork Git reference for a validated version."""
    normalized = str(Version(version))
    return f"git+{FORK_REPOSITORY_URL}@v{normalized}"


def _get_install_command(version: str) -> str:
    """Return an exact fork installation command for the environment."""
    import sys

    spec = _release_install_spec(version)
    if is_self_install_supported() and "/uv/tools/" in sys.executable:
        return f'uv tool install --force "{spec}"'
    return f'pip install --force-reinstall "{DISTRIBUTION_NAME} @ {spec}"'


def check_for_updates(quiet: bool = False) -> Optional[str]:
    """Check audited fork releases and return this check's verified version.

    The disable-hint is appended to the banner only when ``quiet`` is True
    (the daily background fetch); explicit ``colab update`` invocations
    (``quiet=False``) omit it because the user asked for the check.
    """
    settings = state.settings_store.load()
    current = get_app_version()

    try:
        # Existing installs may have persisted the former upstream PyPI URL.
        # Migrate only that exact legacy default; explicit custom URLs remain
        # under user control.
        if settings.update_url == LEGACY_UPSTREAM_UPDATE_URL:
            settings.update_url = DEFAULT_UPDATE_URL
            # A cached version is meaningful only relative to its source. Do
            # not reinterpret an upstream PyPI version as an audited fork tag.
            settings.latest_version = None
        release = _fetch_release(settings.update_url, quiet)
        release_v = _parse_version(release)

        if _is_newer(release_v, current):
            announce_upgrade(
                release_v,
                current,
                _get_install_command(release_v),
                show_disable_hint=quiet,
            )
        elif not quiet:
            suffix = f", latest: {release_v}" if release_v else ""
            typer.echo(f"[colab] Colab CLI is up to date (version: {current}{suffix}).")

        # Cache the highest observed version; never downgrade.
        cached = settings.latest_version or "0"
        if _is_newer(release_v, cached):
            settings.latest_version = release_v

        settings.last_check = datetime.now(timezone.utc)
        state.settings_store.save(settings)
        return release_v

    except Exception as e:
        if not quiet:
            typer.echo(f"[colab] Failed to check for updates: {e}")
        return None


# ---------- Background hooks (called from cli.py) -----------------------


def _is_throttled(settings: Settings, *, now: Optional[datetime] = None) -> bool:
    """True if the once-per-day fetch should be skipped."""
    if settings.last_check is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - settings.last_check).days < 1


def maybe_show_cached_banner(settings: Settings) -> None:
    """Print the cached upgrade banner if the cache reports a newer version.

    Called from the global CLI callback when the daily fetch is throttled.
    The banner uses a generic ``colab update`` install hint because the
    cache does not record which source supplied the version; the disable
    hint is shown because this is unsolicited output.
    """
    if not settings.latest_version:
        return
    current = get_app_version()
    if not _is_newer(settings.latest_version, current):
        return
    announce_upgrade(
        settings.latest_version,
        current,
        "colab update",
        show_disable_hint=True,
    )


def run_background_check() -> None:
    """Entry point for the global CLI callback.

    Performs either the daily fetch (which writes the cache) or, if
    throttled, surfaces the cached banner. Honors the
    ``enable_update_check`` master switch.
    """
    settings = state.settings_store.load()
    if not settings.enable_update_check:
        return
    if _is_throttled(settings):
        maybe_show_cached_banner(settings)
    else:
        check_for_updates(quiet=True)


# ---------- Self-install ------------------------------------------------


def self_install(version: str) -> None:
    """Install one version verified by the current fork release check."""
    import sys

    spec = _release_install_spec(version)

    # If the executable path contains "/uv/", we assume it was installed via
    # `uv tool install` and use `uv` to upgrade it.
    if "/uv/tools/" in sys.executable:
        cmd = ["uv", "tool", "install", "--force", spec]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            f"{DISTRIBUTION_NAME} @ {spec}",
        ]

    typer.echo(f"[colab] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
