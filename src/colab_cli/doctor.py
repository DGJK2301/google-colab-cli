# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Local-first diagnostics for installation, auth, state, and leases."""

from __future__ import annotations

import datetime
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Any

from colab_cli.auth import TOKEN_CONFIG_PATH
from colab_cli.auto_update import get_app_version
from colab_cli.doctor_models import (
    DoctorEnvelope,
    DoctorIssue,
    KeepAliveDiagnostic,
    NetworkObservation,
    PackageObservation,
    PermissionObservation,
    RuntimeObservation,
    StoreObservation,
    TokenObservation,
    TransferLeaseDiagnostic,
    TransferLeaseObservation,
)
from colab_cli.observability.collector import utc_now
from colab_cli.observability.redaction import redact_text
from colab_cli.state import SessionState, Settings
from colab_cli.transfer_lease import (
    process_existence_state,
    process_identity_state,
    transfer_lease_root,
)


_DEPENDENCIES = (
    "google-auth",
    "google-auth-oauthlib",
    "jupyter-kernel-client",
    "pyzmq",
    "requests",
    "filelock",
    "pydantic",
    "typer",
    "click",
    "nbformat",
)


def validate_doctor_timeout(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("network timeout must be a finite number greater than 0")
    return float(value)


def collect_doctor(
    state: Any,
    *,
    network: bool,
    timeout: float,
) -> DoctorEnvelope:
    timeout = validate_doctor_timeout(timeout)
    warnings: list[DoctorIssue] = []
    errors: list[DoctorIssue] = []
    runtime = _runtime_observation()
    dependencies = [_package_observation(name) for name in _DEPENDENCIES]
    for package in dependencies:
        if package.status != "ok":
            warnings.append(
                DoctorIssue(
                    code="DEPENDENCY_UNAVAILABLE",
                    message=(f"{package.name}: {package.error or package.status}"),
                    source="dependencies",
                )
            )

    token = _token_observation(state)
    if token.parse_status == "invalid":
        errors.append(
            DoctorIssue(
                code="TOKEN_CACHE_INVALID",
                message="OAuth token cache exists but is invalid.",
                source="authentication",
                severity="error",
                retryable=False,
            )
        )
    if token.permission.status == "insecure":
        warnings.append(
            DoctorIssue(
                code="TOKEN_CACHE_PERMISSIONS_INSECURE",
                message=(
                    token.permission.reason or "Token cache permissions are broad."
                ),
                source="authentication",
                retryable=False,
            )
        )
    if token.expired and not token.refresh_token_present:
        warnings.append(
            DoctorIssue(
                code="TOKEN_EXPIRED_WITHOUT_REFRESH_TOKEN",
                message=("Cached OAuth token is expired and has no refresh token."),
                source="authentication",
                retryable=False,
            )
        )

    session_store = _strict_session_store(state.store.path)
    settings_store = _strict_settings_store(state.settings_store.path)
    if session_store.parse_status == "invalid":
        errors.append(
            DoctorIssue(
                code="SESSION_STORE_INVALID",
                message="Session state file failed strict parsing.",
                source="session_store",
                severity="error",
                retryable=False,
            )
        )
    if settings_store.parse_status == "invalid":
        warnings.append(
            DoctorIssue(
                code="SETTINGS_STORE_INVALID",
                message="Settings file failed strict parsing.",
                source="settings_store",
                retryable=False,
            )
        )
    for item in session_store.keep_alive:
        if item.process_status == "dead":
            warnings.append(
                DoctorIssue(
                    code="STALE_KEEP_ALIVE_PID",
                    message=(
                        f"Session {item.session!r} records dead "
                        f"keep-alive PID {item.pid}."
                    ),
                    source="session_store",
                    retryable=False,
                )
            )
        elif item.process_status == "unknown":
            warnings.append(
                DoctorIssue(
                    code="KEEP_ALIVE_PID_UNVERIFIED",
                    message=(
                        f"Session {item.session!r} keep-alive PID "
                        f"{item.pid} could not be verified."
                    ),
                    source="session_store",
                )
            )

    leases = _transfer_lease_observation()
    for lease in leases.entries:
        if lease.diagnostic == "stale":
            warnings.append(
                DoctorIssue(
                    code="STALE_TRANSFER_LEASE",
                    message=f"Stale transfer lease: {lease.path}",
                    source="transfer_leases",
                    retryable=False,
                )
            )
        elif lease.diagnostic in {"unsafe", "invalid"}:
            warnings.append(
                DoctorIssue(
                    code="UNSAFE_TRANSFER_LEASE",
                    message=(f"Transfer lease needs manual inspection: {lease.path}"),
                    source="transfer_leases",
                    retryable=False,
                )
            )

    network_result = _network_observation(
        state,
        requested=network,
        timeout=timeout,
        local_endpoints={item.endpoint for item in session_store.keep_alive},
    )
    if network_result.status in {"error", "timeout"}:
        warnings.append(
            DoctorIssue(
                code="NETWORK_ASSIGNMENT_QUERY_FAILED",
                message=(network_result.error or "Assignment query failed."),
                source="network",
                retryable=True,
            )
        )

    status = "error" if errors else ("warning" if warnings else "ok")
    return DoctorEnvelope(
        ok=not errors,
        status=status,
        generated_at=utc_now(),
        runtime=runtime,
        dependencies=dependencies,
        token=token,
        session_store=session_store,
        settings_store=settings_store,
        transfer_leases=leases,
        network=network_result,
        warnings=warnings,
        errors=errors,
    )


def _runtime_observation() -> RuntimeObservation:
    install = Path(__file__).resolve().parent
    commit, reason = _find_git_commit(install)
    return RuntimeObservation(
        cli_version=get_app_version(),
        commit_sha=commit,
        commit_unavailable_reason=reason,
        install_path=str(install),
        executable=sys.executable,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def _find_git_commit(
    start: Path,
) -> tuple[str | None, str | None]:
    for candidate in (start, *start.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(candidate),
                    "rev-parse",
                    "HEAD",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"git inspection failed: {exc}"
        value = completed.stdout.strip()
        if completed.returncode == 0 and len(value) == 40:
            return value, None
        return (
            None,
            "git worktree found but HEAD could not be resolved",
        )
    return None, "installed artifact has no Git metadata"


def _package_observation(name: str) -> PackageObservation:
    try:
        return PackageObservation(
            name=name,
            version=metadata.version(name),
            status="ok",
        )
    except metadata.PackageNotFoundError:
        return PackageObservation(
            name=name,
            status="missing",
            error="not installed",
        )
    except Exception as exc:
        return PackageObservation(
            name=name,
            status="error",
            error=redact_text(f"{type(exc).__name__}: {exc}"),
        )


def _token_observation(state: Any) -> TokenObservation:
    provider = str(
        getattr(
            state.auth_provider,
            "value",
            state.auth_provider,
        )
    )
    path = Path(TOKEN_CONFIG_PATH).expanduser()
    permission = _permission_observation(path)
    if provider != "oauth2":
        return TokenObservation(
            provider=provider,
            path=str(path),
            exists=path.exists(),
            readable=None,
            parse_status="not_applicable",
            permission=PermissionObservation(
                status="not_applicable",
                reason=(f"OAuth token cache is not used by the {provider!r} provider"),
            ),
        )
    if not path.exists():
        return TokenObservation(
            provider=provider,
            path=str(path),
            exists=False,
            parse_status="missing",
            permission=permission,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("token root must be an object")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ):
        return TokenObservation(
            provider=provider,
            path=str(path),
            exists=True,
            readable=os.access(path, os.R_OK),
            parse_status="invalid",
            permission=permission,
        )
    scopes = payload.get("scopes") or []
    if not isinstance(scopes, list):
        scopes = []
    expiry = payload.get("expiry")
    return TokenObservation(
        provider=provider,
        path=str(path),
        exists=True,
        readable=os.access(path, os.R_OK),
        parse_status="ok",
        expiry=(str(expiry) if expiry is not None else None),
        expired=_is_expired(expiry),
        scopes=sorted(str(scope) for scope in scopes),
        refresh_token_present=bool(payload.get("refresh_token")),
        rapt_token_present=bool(payload.get("rapt_token")),
        permission=permission,
    )


def _is_expired(value: Any) -> bool | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed <= datetime.datetime.now(datetime.timezone.utc)


def _permission_observation(
    path: Path,
) -> PermissionObservation:
    if not path.exists():
        return PermissionObservation(
            status="not_applicable",
            reason="file is absent",
        )
    if os.name == "nt":
        return _windows_permission_observation(path)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        return PermissionObservation(
            status="unknown",
            reason=redact_text(exc),
        )
    broad = mode & 0o077
    return PermissionObservation(
        status="secure" if broad == 0 else "insecure",
        mode=f"{mode:04o}",
        reason=(
            None
            if broad == 0
            else ("group/other permission bits are set on the token cache")
        ),
    )


def _windows_permission_observation(
    path: Path,
) -> PermissionObservation:
    try:
        completed = subprocess.run(
            ["icacls", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PermissionObservation(
            status="unknown",
            reason=("icacls unavailable: " + redact_text(exc)),
        )
    if completed.returncode != 0:
        return PermissionObservation(
            status="unknown",
            reason="icacls could not read token ACL",
        )
    text = completed.stdout.lower()
    broad_principals = (
        "everyone:",
        "builtin\\users:",
        "authenticated users:",
        "s-1-1-0:",
        "s-1-5-32-545:",
        "s-1-5-11:",
    )
    exposed = any(principal in text for principal in broad_principals)
    return PermissionObservation(
        status="insecure" if exposed else "secure",
        reason=(
            "token ACL grants a broad Windows principal"
            if exposed
            else "icacls found no broad principal entry"
        ),
    )


def _strict_session_store(
    path_value: str,
) -> StoreObservation:
    path = Path(path_value).expanduser()
    if not path.exists():
        return StoreObservation(
            path=str(path),
            exists=False,
            parse_status="missing",
        )
    invalid: list[str] = []
    keep_alive: list[KeepAliveDiagnostic] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("session store root must be an object")
        for key, value in payload.items():
            try:
                session = SessionState.model_validate(value)
                if session.name != key:
                    invalid.append(f"{key}: embedded name is {session.name!r}")
                process_status = (
                    "not_recorded"
                    if session.keep_alive_pid is None
                    else process_existence_state(session.keep_alive_pid)
                )
                keep_alive.append(
                    KeepAliveDiagnostic(
                        session=session.name,
                        endpoint=session.endpoint,
                        pid=session.keep_alive_pid,
                        process_status=process_status,
                    )
                )
            except Exception as exc:
                invalid.append(f"{key}: {type(exc).__name__}: {exc}")
        return StoreObservation(
            path=str(path),
            exists=True,
            parse_status=("invalid" if invalid else "ok"),
            entry_count=len(payload),
            invalid_entries=invalid,
            keep_alive=keep_alive,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        return StoreObservation(
            path=str(path),
            exists=True,
            parse_status="invalid",
            invalid_entries=[redact_text(f"{type(exc).__name__}: {exc}")],
        )


def _strict_settings_store(
    path_value: str,
) -> StoreObservation:
    path = Path(path_value).expanduser()
    if not path.exists():
        return StoreObservation(
            path=str(path),
            exists=False,
            parse_status="missing",
        )
    try:
        Settings.model_validate_json(path.read_text(encoding="utf-8"))
        return StoreObservation(
            path=str(path),
            exists=True,
            parse_status="ok",
            entry_count=1,
        )
    except Exception as exc:
        return StoreObservation(
            path=str(path),
            exists=True,
            parse_status="invalid",
            invalid_entries=[redact_text(f"{type(exc).__name__}: {exc}")],
        )


def _transfer_lease_observation() -> TransferLeaseObservation:
    root = transfer_lease_root()
    if not root.exists():
        return TransferLeaseObservation(
            root=str(root),
            exists=False,
        )
    entries: list[TransferLeaseDiagnostic] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("lease metadata root must be an object")
            metadata_state = str(payload.get("state") or "unknown")
            if metadata_state == "released":
                owner_status = "not_applicable"
                diagnostic = "released"
            elif metadata_state != "active":
                owner_status = "unknown"
                diagnostic = "unsafe"
            else:
                owner_status = process_identity_state(
                    payload.get("pid"),
                    payload.get("process_start_token"),
                )
                diagnostic = {
                    "alive": "active",
                    "dead": "stale",
                    "unknown": "unsafe",
                }[owner_status]
            entries.append(
                TransferLeaseDiagnostic(
                    path=str(path),
                    lease_id=_string(payload.get("lease_id")),
                    direction=_string(payload.get("direction")),
                    target_path=_string(payload.get("target_path")),
                    pid=_integer(payload.get("pid")),
                    heartbeat_at=_string(payload.get("heartbeat_at")),
                    metadata_state=metadata_state,
                    owner_status=owner_status,
                    diagnostic=diagnostic,
                )
            )
        except Exception as exc:
            entries.append(
                TransferLeaseDiagnostic(
                    path=str(path),
                    owner_status="unknown",
                    diagnostic="invalid",
                    error=redact_text(f"{type(exc).__name__}: {exc}"),
                )
            )
    return TransferLeaseObservation(
        root=str(root),
        exists=True,
        entries=entries,
    )


def _network_observation(
    state: Any,
    *,
    requested: bool,
    timeout: float,
    local_endpoints: set[str],
) -> NetworkObservation:
    if not requested:
        return NetworkObservation(
            requested=False,
            status="not_requested",
        )
    started = time.monotonic()
    try:
        assignments = state.client.list_assignments(
            timeout=(min(5.0, timeout), timeout)
        )
        endpoints = sorted(item.endpoint for item in assignments)
        return NetworkObservation(
            requested=True,
            status="ok",
            assignment_count=len(assignments),
            endpoints=endpoints,
            orphan_endpoints=[
                endpoint for endpoint in endpoints if endpoint not in local_endpoints
            ],
            elapsed_seconds=round(
                time.monotonic() - started,
                6,
            ),
        )
    except Exception as exc:
        status = "timeout" if "timeout" in type(exc).__name__.lower() else "error"
        return NetworkObservation(
            requested=True,
            status=status,
            elapsed_seconds=round(
                time.monotonic() - started,
                6,
            ),
            error=redact_text(f"{type(exc).__name__}: {exc}"),
        )


def _string(value: Any) -> str | None:
    return None if value is None else str(value)


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
