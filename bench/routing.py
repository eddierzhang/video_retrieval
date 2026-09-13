"""Measure how often the planner sends a query to the right executor.

    python -m bench.routing

Routing is the first decision a search makes and the most expensive one to get wrong: a moment
sent to the text reader comes back empty after a minute or more of OCR, and a request to read
characters sent to event search returns a clip instead of an answer. Each row in routing.json is
labeled with the kind of answer wanted, so a prompt change can be scored instead of eyeballed.
The planner runs at temperature 0, so one call per query is a fair measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from bench.tracking import Run
from video_retrieval.local_backend import LocalModels, use_models
from video_retrieval.retrieval import plan_query

ROOT = Path(__file__).resolve().parent
ROUTING = ROOT / "routing.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROUTING))
    args = parser.parse_args()

    with Run("routing", args) as run:
        print(f"run {run.id}")
        run.input(args.dataset)
        rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["rows"]
        models = LocalModels()
        run.note(models=models.signature())
        wrong, seconds = [], []
        with use_models(models):
            for row in rows:
                started = time.time()
                plan = plan_query(row["query"])
                seconds.append(time.time() - started)
                got = plan.get("executor")
                ok = got == row["executor"]
                if not ok:
                    wrong.append(row)
                print(f"[{'ok  ' if ok else 'MISS'}] {got:<24} {row['query']}")
                run.log({"query": row["query"], "expected": row["executor"], "got": got, "correct": ok,
                         "seconds": round(seconds[-1], 2), "rerouted": plan.get("rerouted")})

        by_kind = {}
        for row in rows:
            kind = row["executor"]
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][1] += 1
            by_kind[kind][0] += row not in wrong
        accuracy = 1 - len(wrong) / len(rows)
        print(f"\nrouted correctly: {len(rows) - len(wrong)}/{len(rows)} ({accuracy:.0%})")
        for kind, (right, total) in by_kind.items():
            print(f"   {kind:<24} {right}/{total}")
        benchmark = [row for row in rows if row.get("from_benchmark")]
        if benchmark:
            fixed = sum(row not in wrong for row in benchmark)
            print(f"   the {len(benchmark)} benchmark misroutes: {fixed} now routed correctly")
        # Prompt examples copied from this file make it easy to score well on it; held-out rows are
        # the honest number.
        held = [row for row in rows if row.get("held_out")]
        if held:
            right = sum(row not in wrong for row in held)
            print(f"   held-out (not echoed in the prompt): {right}/{len(held)}")
            run.summarize(held_out_accuracy=right / len(held))
        echoed = [row for row in rows if row.get("in_prompt")]
        if echoed:
            print(f"   echoed in the prompt, so flattering: {sum(row not in wrong for row in echoed)}/{len(echoed)}")
        print(f"mean planning time {sum(seconds) / len(seconds):.1f}s")
        run.summarize(accuracy=accuracy, wrong=[row["query"] for row in wrong],
                      by_kind={kind: right / total for kind, (right, total) in by_kind.items()},
                      mean_seconds=sum(seconds) / len(seconds))


if __name__ == "__main__":
    main()
