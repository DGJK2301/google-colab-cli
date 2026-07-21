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

import tomllib
from pathlib import Path

from colab_cli.state import DEFAULT_UPDATE_URL, Settings


ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_COMMIT = "f18e982c3265df5e923aa9def101ab3fd737e139"
FORK_RELEASE_SPEC = "git+https://github.com/DGJK2301/google-colab-cli.git@v0.6.0.post1"


def test_published_metadata_pins_transport_and_dependency_floors():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert any(
        dependency.startswith("jupyter-kernel-client @ git+")
        and TRANSPORT_COMMIT in dependency
        for dependency in dependencies
    )
    assert "jsonschema>=4.26.0" in dependencies
    assert "pyzmq>=26.0.0" in dependencies
    assert "rpds-py>=0.25.0" in dependencies
    assert project["project"]["scripts"]["colab"] == "colab_cli.entrypoint:main"
    assert project["tool"]["hatch"]["version"]["fallback-version"] == ("0.6.0.post1")
    assert project["tool"]["hatch"]["metadata"]["allow-direct-references"] is True
    assert "sources" not in project["tool"]["uv"]


def test_lock_records_requested_and_resolved_transport_commit():
    lock = (ROOT / "uv.lock").read_text()
    requested = (
        "https://github.com/googlecolab/jupyter-kernel-client.git?rev="
        + TRANSPORT_COMMIT
    )
    assert requested in lock
    assert requested + "#" + TRANSPORT_COMMIT in lock


def test_windows_install_docs_are_single_line_and_pin_the_fork_release():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f'uv tool install --force "{FORK_RELEASE_SPEC}"' in readme
    assert (
        f'pip install --force-reinstall "google-colab-cli @ {FORK_RELEASE_SPEC}"'
        in readme
    )
    install_lines = [line.rstrip() for line in readme.splitlines() if "install" in line]
    assert all(not line.endswith("\\") for line in install_lines)


def test_bundled_skill_describes_fail_closed_accelerators_only():
    skill = (ROOT / "skills" / "colab-operator" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "silently falls back" not in skill
    assert "Accelerator requests fail closed" in skill


def test_default_update_source_is_the_audited_fork():
    assert Settings().update_url == DEFAULT_UPDATE_URL
    assert DEFAULT_UPDATE_URL == (
        "https://api.github.com/repos/DGJK2301/google-colab-cli/releases/latest"
    )
