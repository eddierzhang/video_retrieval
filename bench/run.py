"""Measure retrieval accuracy against hand-labeled query -> interval pairs.

    python -m bench.run                                    every row, writes a result file
    python -m bench.run --only code-scroll                 one row
    python -m bench.run --mode quick                       force a mode
    python -m bench.run --compare bench/results/<file>     diff against an earlier run

Rows live in bench/dataset.json and are labeled by hand: watch the video in the app,
note the true interval, add a row. Every number here is only as good as those labels,
so treat a small dataset as a smoke test rather than a benchmark.

Videos must already be indexed in the library; the harness never re-indexes.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import re
import statistics
import tempfile
import time

from bench.tracking import Run
from video_retrieval.local_backend import LocalModels
from video_retrieval.local_indexing import IndexVersionMismatch, load_index
from webapp.library import Library

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset.json"
RESULTS = ROOT / "results"
THRESHOLDS = (0.3, 0.5)


def iou(a_start, a_end, b_start, b_end):
    """Temporal intersection over union of two intervals."""
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def load_pipelines(rows, library, models):
    """One pipeline per video, loaded once and shared by that video's rows."""
    records = {}
    for record in library.list():
        records[record["id"]] = record
        records[record["name"]] = record
    pipelines = {}
    for name in sorted({row["video"] for row in rows}):
        record = records.get(name)
        if record is None:
            raise SystemExit(f"No video called {name!r} in the library. Add it, or fix {DATASET.name}.")
        if not record.get("index"):
            raise SystemExit(f"{name!r} has no index yet. Index it in the app first.")
        folder = library.video_dir(record["id"])
        try:
            pipelines[name] = load_index(folder / record["index"]["dir"], models, folder / record["source"])
        except IndexVersionMismatch as exc:
            raise SystemExit(f"{name!r}: {exc}")
    return pipelines


def run_row(pipeline, row, output_root):
    started = time.time()
    result = pipeline.retrieve(
        row["query"],
        run_verification=row.get("mode", "verified") == "verified",
        output_root=str(output_root / row["id"]),
        final_frame_fps=1.0,
        max_frames_per_match=1,
    )
    # Matches come back in time order. "Top 1" has to mean the one the system trusts most, not the
    # earliest, or a correct but late answer is scored as a rank-2 miss.
    matches = sorted(result.get("matches", []), key=lambda match: -float(match.get("confidence", 0.0)))
    measured = {
        "id": row["id"],
        "video": row["video"],
        "query": row["query"],
        "mode": row.get("mode", "verified"),
        "elapsed": round(time.time() - started, 1),
        "executor": (result.get("plan") or {}).get("executor"),
        "num_matches": len(matches),
        # The intervals themselves, so bench.boundaries can learn from the proposals retrieval made.
        "matches": [[round(float(m["start"]), 3), round(float(m["end"]), 3)] for m in matches[:5]],
    }
    expect = row["expect"]
    if "text" in expect:
        found = [match.get("extracted_text") or "" for match in matches]
        target = normalize(expect["text"])
        measured["found"] = found[:3]
        measured["exact"] = any(normalize(item) == target for item in found)
        measured["contains"] = any(target and target in normalize(item) for item in found)
        measured["correct"] = measured["exact"]
    else:
        scores = [iou(expect["start"], expect["end"], float(m["start"]), float(m["end"])) for m in matches]
        best = max(scores, default=0.0)
        measured["best_iou"] = round(best, 3)
        measured["top1_iou"] = round(scores[0], 3) if scores else 0.0
        measured["best_rank"] = scores.index(best) + 1 if scores else None
        for threshold in THRESHOLDS:
            measured[f"hit@{threshold}"] = best >= threshold
        measured["correct"] = best >= THRESHOLDS[0]
    diagnostics = result.get("diagnostics") or {}
    measured["funnel"] = {key: value for key, value in diagnostics.items() if key.startswith("num_")}
    return measured


def summarize(rows):
    if not rows:
        return {}
    temporal = [row for row in rows if "best_iou" in row]
    text = [row for row in rows if "exact" in row]
    summary = {
        "rows": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 3),
        "mean_seconds": round(statistics.mean([row["elapsed"] for row in rows]), 1),
    }
    if temporal:
        summary["mean_iou"] = round(statistics.mean([row["best_iou"] for row in temporal]), 3)
        for threshold in THRESHOLDS:
            summary[f"recall@{threshold}"] = round(sum(row[f"hit@{threshold}"] for row in temporal) / len(temporal), 3)
    if text:
        summary["text_exact"] = round(sum(row["exact"] for row in text) / len(text), 3)
    return summary


def format_row(row):
    mark = "ok  " if row["correct"] else "MISS"
    if "best_iou" in row:
        detail = f"iou {row['best_iou']:.2f} (top1 {row['top1_iou']:.2f}, rank {row['best_rank']})"
    else:
        detail = f"exact={row['exact']} contains={row['contains']} found={row['found'][:1]}"
    return f"[{mark}] {row['id']:<18} {row['mode']:<8} {detail}  {row['elapsed']:>5.1f}s  {row['executor']}"


def compare(current, earlier_path):
    earlier = json.loads(Path(earlier_path).read_text(encoding="utf-8"))
    before = {row["id"]: row for row in earlier["rows"]}
    print(f"\nagainst {Path(earlier_path).name} ({earlier.get('created_at', '?')})")
    for row in current["rows"]:
        previous = before.get(row["id"])
        if previous is None:
            print(f"  {row['id']:<18} new")
            continue
        if "best_iou" in row and "best_iou" in previous:
            delta = row["best_iou"] - previous["best_iou"]
            arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            print(f"  {row['id']:<18} iou {previous['best_iou']:.2f} -> {row['best_iou']:.2f}  {arrow}{abs(delta):.2f}")
        else:
            print(f"  {row['id']:<18} correct {previous.get('correct')} -> {row.get('correct')}")
    for key, value in current["summary"].items():
        old = earlier.get("summary", {}).get(key)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)) and old != value:
            print(f"  summary {key}: {old} -> {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--only", action="append", help="row id; repeatable")
    parser.add_argument("--sample", type=int, help="run a random but repeatable subset of this size")
    parser.add_argument("--video", action="append", help="restrict to these library videos")
    parser.add_argument("--mode", choices=("verified", "quick"), help="override the mode of every row")
    parser.add_argument("--out", help="where to write the result JSON")
    parser.add_argument("--compare", help="an earlier result file to diff against")
    parser.add_argument("--seed", type=int, default=0, help="which subset --sample draws")
    args = parser.parse_args()

    with Run("benchmark", args, seed=args.seed) as run:
        print(f"run {run.id}")
        benchmark(args, run)


def benchmark(args, run):
    run.input(args.dataset)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    rows = [row for row in dataset["rows"] if not args.only or row["id"] in args.only]
    if args.video:
        rows = [row for row in rows if row["video"] in set(args.video)]
    if not rows:
        raise SystemExit("No rows selected.")
    if args.sample and args.sample < len(rows):
        rows = random.Random(args.seed).sample(rows, args.sample)
    if args.mode:
        rows = [{**row, "mode": args.mode} for row in rows]

    models = LocalModels()
    run.note(models=models.signature(), rows_selected=[row["id"] for row in rows])
    pipelines = load_pipelines(rows, Library(), models)
    measured, failed = [], []
    with tempfile.TemporaryDirectory() as temp:
        for row in rows:
            try:
                measured.append(run_row(pipelines[row["video"]], row, Path(temp)))
            except Exception as exc:
                # A local model returning unusable JSON is a property of the row, not a
                # reason to throw away every row after it - these runs take hours.
                failed.append({"id": row["id"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"[fail] {row['id']:<18} {type(exc).__name__}: {exc}", flush=True)
                run.log({"id": row["id"], "failed": failed[-1]["error"]})
                continue
            print(format_row(measured[-1]), flush=True)
            run.log(measured[-1])

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "models": models.signature(),
        "summary": summarize(measured),
        "rows": measured,
        "failed": failed,
    }
    print("\nsummary:", json.dumps(result["summary"]))
    if failed:
        print(f"{len(failed)} row(s) failed and were skipped: " +
              ", ".join(row["id"] for row in failed))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("written to", out)
    run.artifact(out, "benchmark_result")
    run.summarize(**result["summary"], failed=len(failed))
    if args.compare:
        compare(result, args.compare)


if __name__ == "__main__":
    main()
