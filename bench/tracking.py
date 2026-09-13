"""Record every experiment well enough to reproduce it, and compare any two.

    python -m bench.tracking list                    recent runs, newest first
    python -m bench.tracking list --kind rank        only one kind
    python -m bench.tracking show <run>              everything one run recorded
    python -m bench.tracking compare <run> <run>     what differed, and what it changed

A number from a trainer is only worth something if you can say what produced it. Each run
writes a folder under bench/runs/ holding:

  run.json       the command, every argument, the seed, the git commit and whether the working
                 tree had uncommitted changes (and a hash of them), library versions, the GPU,
                 and a SHA-256 of every input file - so "the ranker got better" can be checked
                 against "the data changed" before anyone believes it
  metrics.jsonl  one line per logged step, for curves and per-row results
  artifacts      the hash of every model or result file the run wrote, so a model on disk can
                 be traced back to the run that produced it

Nothing here needs a server or an account, in keeping with the rest of the project. A run that
raises is still recorded, with status "failed" and the error, because failed runs are part of
the record too.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PROJECT = ROOT.parent


def seed_everything(seed):
    """One seed for every source of randomness a trainer touches."""
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True,
                              timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state():
    """The commit, and whether what ran differs from it."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    diff = _git("diff", "HEAD") or ""
    untracked = _git("ls-files", "--others", "--exclude-standard") or ""
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(diff or untracked),
        # A dirty tree is not reproducible from the commit alone; the hash at least tells two
        # dirty runs apart.
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
        "untracked": untracked.splitlines()[:50],
    }


def environment():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in ("numpy", "torch", "transformers", "faiss"):
        try:
            info[module] = __import__(module).__version__
        except (ImportError, AttributeError):
            continue
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def _plain(value):
    """Arguments as JSON: paths become strings, anything else unknown becomes its repr."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return repr(value)


class Run:
    """One experiment. Use as a context manager so a crash is still written down."""

    def __init__(self, kind, args=None, seed=None, inputs=(), root=None, **extra):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.id = f"{stamp}-{kind}-{uuid.uuid4().hex[:4]}"
        self.folder = Path(root or RUNS) / self.id
        self.folder.mkdir(parents=True, exist_ok=True)
        self._started = time.time()
        arguments = vars(args) if isinstance(args, argparse.Namespace) else (args or {})
        self.record = {
            "id": self.id,
            "kind": kind,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv,
            "args": _plain(arguments),
            "seed": seed,
            "git": git_state(),
            "environment": environment(),
            "inputs": [],
            "summary": {},
            "artifacts": [],
            **_plain(extra),
        }
        for path in inputs:
            self.input(path)
        self._write()

    def input(self, path, **details):
        path = Path(path)
        entry = {"path": str(path), "exists": path.is_file(), **_plain(details)}
        if path.is_file():
            entry.update({"sha256": file_digest(path), "bytes": path.stat().st_size})
        self.record["inputs"].append(entry)
        self._write()
        return entry

    def note(self, **fields):
        """Top-level facts learned after the run started, such as the models it loaded."""
        self.record.update(_plain(fields))
        self._write()

    def log(self, values, step=None):
        line = {"time": round(time.time() - self._started, 3), **({"step": step} if step is not None else {}),
                **_plain(values)}
        with open(self.folder / "metrics.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps(line) + "\n")

    def summarize(self, **values):
        self.record["summary"].update(_plain(values))
        self._write()

    def artifact(self, path, name=None):
        path = Path(path)
        if not path.is_file():
            return None
        entry = {"name": name or path.name, "path": str(path), "sha256": file_digest(path),
                 "bytes": path.stat().st_size}
        self.record["artifacts"].append(entry)
        self._write()
        return entry

    def finish(self, status="done", error=None):
        self.record.update({
            "status": status,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - self._started, 1),
        })
        if error is not None:
            self.record["error"] = error
        self._write()

    def _write(self):
        temporary = self.folder / "run.json.part"
        temporary.write_text(json.dumps(self.record, indent=2), encoding="utf-8")
        os.replace(temporary, self.folder / "run.json")

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.finish("done")
        elif issubclass(kind, SystemExit):
            # Trainers exit early on purpose ("not enough data yet"); that is not a crash.
            self.finish("stopped", error=str(value))
        elif issubclass(kind, KeyboardInterrupt):
            self.finish("interrupted")
        else:
            self.finish("failed", error=f"{kind.__name__}: {value}")
        return False


# ------------------------------------------------------------------- reading

def load_runs(root=None, kind=None):
    runs = []
    for path in sorted(Path(root or RUNS).glob("*/run.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if kind is None or record.get("kind") == kind:
            runs.append(record)
    return runs


def find_run(prefix, root=None):
    matches = [run for run in load_runs(root) if run["id"].startswith(prefix) or run["id"].endswith(prefix)]
    if not matches:
        raise SystemExit(f"No run matches {prefix!r}.")
    if len(matches) > 1:
        raise SystemExit(f"{prefix!r} matches {len(matches)} runs; give more of the id.")
    return matches[0]


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix[:-1]: value}


def difference(first, second):
    """Everything two runs disagree on, split into what went in and what came out."""
    report = {}
    for section in ("args", "summary"):
        left, right = _flatten(first.get(section) or {}), _flatten(second.get(section) or {})
        report[section] = {key: (left.get(key), right.get(key))
                           for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)}
    report["seed"] = (first.get("seed"), second.get("seed")) if first.get("seed") != second.get("seed") else None
    commits = ((first.get("git") or {}).get("commit"), (second.get("git") or {}).get("commit"))
    report["commit"] = commits if commits[0] != commits[1] else None
    report["dirty"] = ((first.get("git") or {}).get("dirty"), (second.get("git") or {}).get("dirty"))
    inputs_left = {entry["path"]: entry.get("sha256") for entry in first.get("inputs", [])}
    inputs_right = {entry["path"]: entry.get("sha256") for entry in second.get("inputs", [])}
    report["inputs"] = {path: (inputs_left.get(path), inputs_right.get(path))
                        for path in sorted(set(inputs_left) | set(inputs_right))
                        if inputs_left.get(path) != inputs_right.get(path)}
    return report


def _short(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--kind")
    listing.add_argument("-n", type=int, default=20)
    commands.add_parser("show").add_argument("run")
    comparing = commands.add_parser("compare")
    comparing.add_argument("first")
    comparing.add_argument("second")
    args = parser.parse_args()

    if args.command == "list":
        runs = load_runs(kind=args.kind)[:args.n]
        if not runs:
            print("No runs recorded yet.")
        for run in runs:
            dirty = "*" if (run.get("git") or {}).get("dirty") else " "
            commit = ((run.get("git") or {}).get("commit") or "")[:7]
            headline = ", ".join(f"{key} {_short(value)}" for key, value in list(_flatten(run.get("summary") or {}).items())[:3])
            print(f"{run['id']:<36} {run.get('status', '?'):<11} {commit}{dirty} {headline}")
        if runs:
            print("\n* = run from a working tree with uncommitted changes")
    elif args.command == "show":
        print(json.dumps(find_run(args.run), indent=2))
    else:
        first, second = find_run(args.first), find_run(args.second)
        report = difference(first, second)
        print(f"{first['id']}  vs  {second['id']}")
        if report["commit"]:
            print(f"\ncommit     {report['commit'][0][:10]} -> {report['commit'][1][:10]}")
        if any(report["dirty"]):
            print(f"uncommitted changes: {report['dirty'][0]} -> {report['dirty'][1]}")
        if report["seed"]:
            print(f"seed       {report['seed'][0]} -> {report['seed'][1]}")
        for title, key in (("inputs changed", "inputs"), ("arguments", "args"), ("results", "summary")):
            if report[key]:
                print(f"\n{title}")
                for name, (left, right) in report[key].items():
                    if key == "inputs":
                        left, right = (left or "missing")[:12], (right or "missing")[:12]
                    print(f"   {name:<34} {_short(left):>14} -> {_short(right)}")
        if report["inputs"]:
            print("\nThe inputs differ, so a change in results may be the data rather than the method.")


if __name__ == "__main__":
    main()
