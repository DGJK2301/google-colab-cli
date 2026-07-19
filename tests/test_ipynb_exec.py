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

import json
import nbformat
import os
import pytest
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, ANY
from colab_cli.cli import main


class TestIpynbExec(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.nb_path = os.path.join(self.temp_dir, "test.ipynb")

        # Create a simple v4 notebook
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "cell1",
                    "metadata": {},
                    "outputs": [],
                    "source": "print('cell 1')",
                },
                {
                    "cell_type": "markdown",
                    "id": "cell2",
                    "metadata": {},
                    "source": "# Markdown cell",
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "cell3",
                    "metadata": {},
                    "outputs": [],
                    "source": "print('cell 2')",
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(self.nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def write_titled_notebook(self, name="titled.ipynb", duplicate=False):
        path = os.path.join(self.temp_dir, name)
        second_title = "First" if duplicate else "Second"
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "first",
                    "metadata": {},
                    "outputs": [],
                    "source": "# @title First\nprint('first')",
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "second",
                    "metadata": {},
                    "outputs": [],
                    "source": f"# @title {second_title}\nprint('second')",
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "third",
                    "metadata": {},
                    "outputs": [],
                    "source": "# @title Third\nprint('third')",
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f)
        return path

    @patch("colab_cli.commands.execution.ColabRuntime")
    @patch("colab_cli.state.StateStore")
    def test_exec_ipynb(
        self,
        mock_store_class,
        mock_runtime_class,
    ):
        with patch.object(
            sys, "argv", ["colab", "exec", "-s", "test-s", "-f", self.nb_path]
        ):
            mock_store = mock_store_class.return_value
            mock_store.get.return_value = MagicMock(
                name="test-s", url="http://url", token="token"
            )

            mock_runtime = mock_runtime_class.return_value
            mock_runtime.execute_code.side_effect = [
                [],  # os.makedirs and os.chdir setup
                [{"text": "cell 1\n"}],
                [{"text": "cell 2\n"}],
            ]

            with patch("builtins.print"), pytest.raises(SystemExit) as error:
                main()

            assert error.value.code == 0

            # Verify both code cells were executed (plus the setup cell)
            self.assertEqual(mock_runtime.execute_code.call_count, 3)
            self.assertIn(
                "os.chdir", mock_runtime.execute_code.call_args_list[0].args[0]
            )
            mock_runtime.execute_code.assert_any_call(
                "print('cell 1')", output_hook=ANY, timeout=30.0
            )
            mock_runtime.execute_code.assert_any_call(
                "print('cell 2')", output_hook=ANY, timeout=30.0
            )

    @patch("colab_cli.commands.execution.ColabRuntime")
    @patch("colab_cli.state.StateStore")
    @patch("colab_cli.commands.execution.typer.echo")
    def test_exec_ipynb_output_format(
        self,
        mock_echo,
        mock_store_class,
        mock_runtime_class,
    ):
        nb_path = os.path.join(self.temp_dir, "test_format.ipynb")
        nb = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "my-cell-id-123",
                    "metadata": {},
                    "outputs": [],
                    "source": "# @title My Special Cell\nprint('hello')",
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "fallback-id-456",
                    "metadata": {},
                    "outputs": [],
                    "source": "print('world')",
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f)

        with patch.object(
            sys, "argv", ["colab", "exec", "-s", "test-s", "-f", nb_path]
        ):
            mock_store = mock_store_class.return_value
            mock_store.get.return_value = MagicMock(
                name="test-s", url="http://url", token="token"
            )

            mock_runtime = mock_runtime_class.return_value
            mock_runtime.execute_code.side_effect = [
                [],  # setup
                [{"text": "hello\n"}],
                [{"text": "world\n"}],
            ]

            with patch("builtins.print"), pytest.raises(SystemExit) as error:
                main()

            assert error.value.code == 0

            mock_echo.assert_any_call("[colab] Executing cell 1/2 - My Special Cell...")
            mock_echo.assert_any_call("[colab] Executing cell 2/2 - fallback-id-456...")

    @patch("colab_cli.commands.execution.ColabRuntime")
    @patch("colab_cli.state.StateStore")
    @patch("colab_cli.commands.execution.typer.echo")
    def test_exec_ipynb_creates_output_file(
        self,
        mock_echo,
        mock_store_class,
        mock_runtime_class,
    ):
        with patch.object(
            sys, "argv", ["colab", "exec", "-s", "test-s", "-f", self.nb_path]
        ):
            mock_store = mock_store_class.return_value
            mock_store.get.return_value = MagicMock(
                name="test-s", url="http://url", token="token"
            )

            mock_runtime = mock_runtime_class.return_value
            mock_runtime.execute_code.side_effect = [
                [],
                [{"output_type": "stream", "name": "stdout", "text": "cell 1\n"}],
                [{"output_type": "stream", "name": "stdout", "text": "cell 2\n"}],
            ]

            with patch("builtins.print"), pytest.raises(SystemExit) as error:
                main()

            assert error.value.code == 0

            output_nb_path = self.nb_path.replace(".ipynb", "_output.ipynb")
            self.assertTrue(os.path.exists(output_nb_path))
            with open(output_nb_path, "r", encoding="utf-8") as f:
                output_nb = nbformat.read(f, as_version=4)

            self.assertEqual(len(output_nb.cells), 3)
            # cell 1 outputs
            self.assertEqual(len(output_nb.cells[0].outputs), 1)
            self.assertEqual(output_nb.cells[0].outputs[0].text, "cell 1\n")
            # cell 2 is markdown
            self.assertEqual(output_nb.cells[1].cell_type, "markdown")
            self.assertFalse(
                hasattr(output_nb.cells[1], "outputs")
                and len(output_nb.cells[1].outputs) > 0
            )
            # cell 3 outputs
            self.assertEqual(len(output_nb.cells[2].outputs), 1)
            self.assertEqual(output_nb.cells[2].outputs[0].text, "cell 2\n")

    @patch("colab_cli.commands.execution.ColabRuntime")
    @patch("colab_cli.state.StateStore")
    def test_exec_ipynb_selects_titles_in_notebook_order(
        self,
        mock_store_class,
        mock_runtime_class,
    ):
        nb_path = self.write_titled_notebook()
        with patch.object(
            sys,
            "argv",
            [
                "colab",
                "exec",
                "-s",
                "test-s",
                "-f",
                nb_path,
                "--cell-title",
                "Third",
                "--cell-title",
                "First",
            ],
        ):
            mock_store = mock_store_class.return_value
            mock_store.get.return_value = MagicMock(
                name="test-s", url="http://url", token="token"
            )
            mock_runtime = mock_runtime_class.return_value
            mock_runtime.execute_code.side_effect = [[], [], []]

            with patch("builtins.print"), pytest.raises(SystemExit) as error:
                main()

            assert error.value.code == 0
            executed = [
                call.args[0] for call in mock_runtime.execute_code.call_args_list[1:]
            ]
            self.assertEqual(
                executed,
                [
                    "# @title First\nprint('first')",
                    "# @title Third\nprint('third')",
                ],
            )

    @patch("colab_cli.commands.execution.ColabRuntime")
    @patch("colab_cli.state.StateStore")
    @patch("colab_cli.commands.execution.typer.echo")
    def test_exec_ipynb_missing_title_fails_before_runtime(
        self,
        mock_echo,
        mock_store_class,
        mock_runtime_class,
    ):
        nb_path = self.write_titled_notebook()
        with patch.object(
            sys,
            "argv",
            [
                "colab",
                "exec",
                "-s",
                "test-s",
                "-f",
                nb_path,
                "--cell-title",
                "Missing",
            ],
        ):
            mock_store_class.return_value.get.return_value = MagicMock(
                name="test-s", url="http://url", token="token"
            )

            with patch("builtins.print"), pytest.raises(SystemExit) as error:
                main()

            assert error.value.code == 2
            mock_echo.assert_any_call(
                "[colab] Error: Notebook cell title 'Missing' was not found.",
                err=True,
            )
            mock_runtime_class.assert_not_called()

    @patch("colab_cli.commands.execution.ColabRuntime")
    @patch("colab_cli.state.StateStore")
    @patch("colab_cli.commands.execution.typer.echo")
    def test_exec_ipynb_ambiguous_title_fails_before_runtime(
        self,
        mock_echo,
        mock_store_class,
        mock_runtime_class,
    ):
        nb_path = self.write_titled_notebook(duplicate=True)
        with patch.object(
            sys,
            "argv",
            [
                "colab",
                "exec",
                "-s",
                "test-s",
                "-f",
                nb_path,
                "--cell-title",
                "First",
            ],
        ):
            mock_store_class.return_value.get.return_value = MagicMock(
                name="test-s", url="http://url", token="token"
            )

            with patch("builtins.print"), pytest.raises(SystemExit) as error:
                main()

            assert error.value.code == 2
            mock_echo.assert_any_call(
                "[colab] Error: Notebook cell title 'First' is ambiguous (2 matches).",
                err=True,
            )
            mock_runtime_class.assert_not_called()

    @patch("colab_cli.commands.execution.ColabRuntime")
    @patch("colab_cli.state.StateStore")
    def test_exec_ipynb_fail_on_error_stops_and_saves_output(
        self,
        mock_store_class,
        mock_runtime_class,
    ):
        nb_path = self.write_titled_notebook()
        with patch.object(
            sys,
            "argv",
            [
                "colab",
                "exec",
                "-s",
                "test-s",
                "-f",
                nb_path,
                "--fail-on-error",
            ],
        ):
            mock_store_class.return_value.get.return_value = MagicMock(
                name="test-s", url="http://url", token="token"
            )
            mock_runtime = mock_runtime_class.return_value
            mock_runtime.execute_code.side_effect = [
                [],
                [
                    {
                        "output_type": "error",
                        "ename": "RuntimeError",
                        "evalue": "boom",
                        "traceback": [],
                    }
                ],
            ]

            with patch("builtins.print"), pytest.raises(SystemExit) as error:
                main()

            assert error.value.code == 1
            self.assertEqual(mock_runtime.execute_code.call_count, 2)
            mock_runtime.stop.assert_called_once_with()

            output_nb_path = nb_path.replace(".ipynb", "_output.ipynb")
            with open(output_nb_path, "r", encoding="utf-8") as f:
                output_nb = nbformat.read(f, as_version=4)
            self.assertEqual(output_nb.cells[0].outputs[0].output_type, "error")
            self.assertEqual(output_nb.cells[0].outputs[0].evalue, "boom")
            self.assertEqual(output_nb.cells[1].outputs, [])


if __name__ == "__main__":
    unittest.main()
