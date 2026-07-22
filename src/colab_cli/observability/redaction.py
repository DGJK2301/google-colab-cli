# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0.

"""Conservative redaction for machine-readable observation errors."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable


_REDACTION_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(colab-runtime-proxy-token=)[^&\s]+"),
    re.compile(r"(?i)([?&]token=)[^&\s]+"),
    re.compile(
        r"""(?ix)
        (
          ["']?
          [A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY)
          ["']?
          \s*[:=]\s*
        )
        (["']?)[^"',\s;&]+
        """
    ),
)
_QUOTED_SECRET_PATTERN = re.compile(
    r"""(?isx)
    (
      ["']?
      [A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY)
      ["']?
      \s*[:=]\s*
    )
    (["'])(.*?)\2
    """
)
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?"
    r"-----END [^-\r\n]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)


def redact_text(value: Any, *, secrets: Iterable[str] = ()) -> str:
    """Return a bounded error string with credentials replaced."""
    text = str(value)
    for home in _home_variants():
        text = text.replace(home, "~")
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "<redacted>")
    text = _PRIVATE_KEY_PATTERN.sub("<redacted>", text)
    text = _QUOTED_SECRET_PATTERN.sub(_replace_quoted_secret, text)
    for pattern in _REDACTION_PATTERNS:
        text = pattern.sub(_replace_match, text)
    return text[:2000]


def _home_variants() -> tuple[str, ...]:
    candidates = {
        os.path.expanduser("~"),
        os.environ.get("HOME"),
        os.environ.get("USERPROFILE"),
    }
    drive = os.environ.get("HOMEDRIVE")
    path = os.environ.get("HOMEPATH")
    if drive and path:
        candidates.add(drive + path)
    variants = set()
    for candidate in candidates:
        if not candidate or candidate == "~":
            continue
        normalized = os.path.normpath(candidate)
        if not os.path.isabs(normalized) or normalized in {os.path.sep, "\\", "/"}:
            continue
        variants.update(
            {
                normalized,
                normalized.replace("\\", "/"),
                normalized.replace("/", "\\"),
            }
        )
    return tuple(sorted(variants, key=len, reverse=True))


def _replace_match(match: re.Match[str]) -> str:
    prefix = match.group(1)
    quote = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
    return f"{prefix}{quote}<redacted>{quote}"


def _replace_quoted_secret(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}<redacted>{match.group(2)}"
