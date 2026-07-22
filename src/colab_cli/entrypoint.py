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

"""Minimal console entry point with dependency-corruption diagnostics."""

import importlib
import sys
from typing import Optional


_KNOWN_MODULE_PREFIXES = (
    "google.auth",
    "google.oauth2",
    "google_auth_oauthlib",
    "html2text",
    "jsonschema",
    "jupyter_kernel_client",
    "referencing",
    "rpds",
    "zmq",
)
_KNOWN_ERROR_MARKERS = (
    "zmq.backend.cython",
    "partially initialized module 'zmq",
    'partially initialized module "zmq',
    "No module named 'rpds.rpds'",
    'No module named "rpds.rpds"',
)


def dependency_import_diagnostic(error: BaseException) -> Optional[str]:
    """Return an actionable message for a known broken dependency stack."""

    missing_name = getattr(error, "name", None) or ""
    error_text = str(error)
    is_known_module = any(
        missing_name == prefix or missing_name.startswith(f"{prefix}.")
        for prefix in _KNOWN_MODULE_PREFIXES
    )
    is_known_abi_error = any(marker in error_text for marker in _KNOWN_ERROR_MARKERS)
    if not (is_known_module or is_known_abi_error):
        return None

    return (
        "[colab] The local Python environment contains a missing or incompatible "
        "CLI dependency. Reinstall this exact release from its published wheel "
        "or Git tag in an isolated `uv tool` environment. When repairing a "
        "shared Jupyter environment, the verified compatibility floors are "
        "pyzmq>=26.0.0, jsonschema>=4.26.0, and rpds-py>=0.25.0. "
        "Original import error: "
        f"{error_text}"
    )


def main() -> None:
    """Import the full CLI lazily so startup dependency failures stay readable."""

    try:
        cli = importlib.import_module("colab_cli.cli")
    except (ImportError, ModuleNotFoundError) as error:
        diagnostic = dependency_import_diagnostic(error)
        if diagnostic is None:
            raise
        print(diagnostic, file=sys.stderr)
        raise SystemExit(1) from None

    cli.main()


if __name__ == "__main__":
    main()
