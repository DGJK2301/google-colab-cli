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

from types import SimpleNamespace

import pytest

from colab_cli import entrypoint


def test_known_rpds_import_failure_gets_actionable_diagnostic():
    error = ModuleNotFoundError("No module named 'rpds.rpds'", name="rpds.rpds")
    message = entrypoint.dependency_import_diagnostic(error)
    assert message is not None
    assert "rpds-py>=0.25.0" in message
    assert "isolated `uv tool` environment" in message


def test_known_google_auth_import_failure_gets_reinstall_diagnostic():
    error = ModuleNotFoundError("No module named 'google.oauth2'", name="google.oauth2")
    message = entrypoint.dependency_import_diagnostic(error)
    assert message is not None
    assert "Reinstall this exact release" in message


def test_unknown_import_failure_is_not_hidden():
    error = ModuleNotFoundError(
        "No module named 'project_private'", name="project_private"
    )
    assert entrypoint.dependency_import_diagnostic(error) is None


def test_main_prints_known_dependency_failure_without_traceback(monkeypatch, capsys):
    error = ImportError(
        "cannot import name '_device' from partially initialized module "
        "'zmq.backend.cython'"
    )

    def fail_import(_):
        raise error

    monkeypatch.setattr(entrypoint.importlib, "import_module", fail_import)

    with pytest.raises(SystemExit) as raised:
        entrypoint.main()

    assert raised.value.code == 1
    assert "pyzmq>=26.0.0" in capsys.readouterr().err


def test_main_delegates_to_cli(monkeypatch):
    cli = SimpleNamespace(called=False)

    def call_main():
        cli.called = True

    cli.main = call_main
    monkeypatch.setattr(entrypoint.importlib, "import_module", lambda _: cli)

    entrypoint.main()

    assert cli.called is True
