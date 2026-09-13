"""The `moments` command: dispatch, pass-through arguments and exit codes."""
import contextlib
import io
import importlib.util
import sys
import unittest
from unittest.mock import patch

from moments import __version__, cli


class CommandLineTest(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_help_lists_every_command_and_version_is_reported(self):
        code, out, _ = self.run_cli("--help")
        self.assertEqual(code, 0)
        for name in cli.COMMANDS:
            self.assertIn(f"  {name}", out)
        self.assertEqual(self.run_cli("--version")[:2], (0, __version__ + "\n"))

    def test_every_command_points_at_a_module_that_exists(self):
        for name, (module, group, _) in cli.COMMANDS.items():
            self.assertIsNotNone(importlib.util.find_spec(module), f"{name} -> {module}")
            self.assertIn(group, dict(cli.GROUPS))

    def test_a_command_runs_its_module_with_the_remaining_arguments(self):
        seen = []

        def fake_run(module, run_name):
            seen.append((module, run_name, list(sys.argv)))

        with patch.object(cli.runpy, "run_module", side_effect=fake_run), patch.object(sys, "argv", ["moments"]):
            code, _, _ = self.run_cli("tune", "--trials", "5", "--seed", "3")
        self.assertEqual(code, 0)
        self.assertEqual(seen, [("bench.tune", "__main__", ["moments tune", "--trials", "5", "--seed", "3"])])

    def test_exit_codes_survive_the_dispatch(self):
        self.assertEqual(self.run_cli("no-such-command")[0], 2)
        with patch.object(cli.runpy, "run_module", side_effect=SystemExit("Not enough data yet.")), \
                patch.object(sys, "argv", ["moments"]):
            code, _, err = self.run_cli("rank")
        self.assertEqual(code, 1)
        self.assertIn("Not enough data yet.", err)
        with patch.object(cli.runpy, "run_module", side_effect=SystemExit(0)), patch.object(sys, "argv", ["moments"]):
            self.assertEqual(self.run_cli("rank")[0], 0)


if __name__ == "__main__":
    unittest.main()
