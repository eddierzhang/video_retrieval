"""The `moments` command: one entry point for the app and every benchmark and trainer.

Each subcommand runs an existing module as if it had been started with `python -m`, with the
remaining arguments passed through untouched, so `moments tune --help` shows that tool's own
options. Nothing heavy is imported until a subcommand is chosen.
"""
from __future__ import annotations

import runpy
import sys

from moments import __version__

# name -> (module, group, one-line summary)
COMMANDS = {
    "serve": ("webapp", "app", "Run the web app on http://127.0.0.1:8765"),
    "benchmark": ("bench.run", "measure", "Score end-to-end retrieval against a labeled dataset"),
    "routing": ("bench.routing", "measure", "Score how often the planner picks the right executor"),
    "runs": ("bench.tracking", "measure", "List, show and compare recorded experiment runs"),
    "construct": ("bench.construct", "data", "Build benchmark timelines whose answers are known"),
    "synthesize": ("bench.synthesize", "data", "Turn scene descriptions into pseudo-queries"),
    "tune": ("bench.tune", "learn", "Cross-validated search over how quick mode picks a moment"),
    "boundaries": ("bench.boundaries", "learn", "Train the clip boundary model"),
    "rank": ("bench.rank", "learn", "Train the candidate ranker and its conformal set"),
    "distill": ("bench.distill", "learn", "Train the pointwise candidate pre-filter"),
    "adapt": ("bench.adapt", "learn", "Train the self-supervised query adapter"),
    "replay": ("bench.replay", "learn", "Replay logged searches to learn when to stop verifying"),
}
GROUPS = (("app", "App"), ("measure", "Measure"), ("data", "Build data"), ("learn", "Learn"))


def usage():
    lines = [f"moments {__version__} - natural-language video retrieval on local models", "",
             "usage: moments <command> [options]", "       moments <command> --help", ""]
    for group, title in GROUPS:
        lines.append(f"{title}:")
        for name, (_, member, summary) in COMMANDS.items():
            if member == group:
                lines.append(f"  {name:<12}{summary}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"moments: unknown command {name!r}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2
    module = COMMANDS[name][0]
    sys.argv = [f"moments {name}", *rest]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as stop:
        # Tools exit with a message for expected conditions ("not enough data yet"); keep its code.
        if isinstance(stop.code, str):
            print(stop.code, file=sys.stderr)
            return 1
        return stop.code or 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
