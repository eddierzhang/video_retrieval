"""Train the boundary model on intervals whose truth is known.

    python -m bench.boundaries                                  train on bench/constructed.json
    python -m bench.boundaries --with-cuts                      save the model that also uses cuts
    python -m bench.boundaries --proposals bench/results/<run>.json   real proposals, not simulated

For each labeled row, frame embeddings and the query give two per-second curves (see
video_retrieval/boundaries.py). A proposal - where retrieval put the interval - is scored at
every second within a window of each of its boundaries, and a linear model is fitted so the
true boundary scores highest: a softmax over positions, one per boundary.

Proposals come from one of two places:

  simulated  the true interval pushed outward and inward at random, mostly wider, because a
             retrieval region is padded by design. Cheap, and plenty of them.
  real       the top match from a bench.run result over the same rows, so the model is measured
             against the errors retrieval actually makes rather than the ones we imagined.

Every run reports the same table: the proposal as it was, two hand-written rules (snap to the
biggest visual change; sit on the biggest similarity step), and the learned model with and
without the cut features.

Why "without cuts" is there. Constructed timelines are spliced, so every true boundary is a hard
cut. A model that learns "snap to the cut" is nearly perfect on them and may be useless on real
footage, where people get out of cars without the camera cutting. The gap between the two
learned rows is how much of the score came from that shortcut - if it is most of it, trust the
no-cuts number. That is why the no-cuts model is the one saved unless --with-cuts asks otherwise.

Rows are held out by whole video, so a model is never scored on a timeline it trained on.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np

from bench.tracking import Run, seed_everything
from video_retrieval.boundaries import (
    BOUNDARY_MODEL,
    CUT_FEATURES,
    FEATURES,
    candidate_matrix,
    frame_index,
    refine_interval,
    signals,
)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "constructed.json"


def iou(a_start, a_end, b_start, b_end):
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def load_rows(paths, include_text=False):
    """Labeled intervals, each with the video file it lives in."""
    from webapp.library import Library

    library = None
    rows = []
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        timelines = {timeline["name"]: timeline["file"] for timeline in document.get("timelines", [])}
        for row in document.get("rows", []):
            expect = row.get("expect") or {}
            if expect.get("end") is None or ("text" in expect and not include_text):
                continue
            source = timelines.get(row["video"])
            if source is None:
                library = library or Library()
                record = next((item for item in library.list() if item.get("name") == row["video"]), None)
                if record is None:
                    continue
                source = str(library.video_dir(record["id"]) / record["source"])
            rows.append({**row, "file": source, "dataset": str(path)})
    return rows


def attach_signals(rows):
    from video_retrieval.local_backend import embed_text, frame_embeddings

    cache = {}
    for row in rows:
        if row["file"] not in cache:
            cache[row["file"]] = frame_embeddings(row["file"])
        times, vectors = cache[row["file"]]
        row["signal"] = signals(times, vectors, embed_text(row["query"]))
    return rows


def simulated_proposals(row, count, window, rng):
    """The truth pushed around, mostly outward - retrieval pads its regions."""
    start, end = float(row["expect"]["start"]), float(row["expect"]["end"])
    proposals = []
    for _ in range(count):
        proposed_start = start - rng.uniform(-0.4 * window, 0.9 * window)
        proposed_end = end + rng.uniform(-0.4 * window, 0.9 * window)
        if proposed_end - proposed_start >= 2.0:
            proposals.append((max(0.0, proposed_start), proposed_end))
    return proposals


def real_proposals(results_path):
    """The top match per row from a bench.run result file."""
    document = json.loads(Path(results_path).read_text(encoding="utf-8"))
    found = {}
    for row in document.get("rows", []):
        if row.get("matches"):
            found[row["id"]] = [tuple(row["matches"][0])]
    return found


def examples(row, proposal, window, context):
    """Two softmax problems per proposal - one per boundary - with the index of the truth."""
    signal = row["signal"]
    start, end = proposal
    truth = {"start": float(row["expect"]["start"]), "end": float(row["expect"]["end"])}
    items = []
    for side, where, other in (("start", start, end), ("end", end, start)):
        spots, matrix = candidate_matrix(signal, side, where, other, window, context)
        if not spots:
            continue
        target_frame = frame_index(signal, truth[side])
        if target_frame < spots[0] or target_frame > spots[-1]:
            continue  # the truth is out of reach from here; nothing to learn from it
        items.append((matrix, spots.index(target_frame)))
    return items


def fit(items, mask, epochs=600, learning_rate=0.2, l2=1e-3):
    stacked = np.concatenate([matrix for matrix, _ in items])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    scaled = [((matrix - mean) / std * mask, target) for matrix, target in items]
    weights = np.zeros(len(FEATURES))
    for epoch in range(epochs):
        gradient = l2 * weights
        for matrix, target in scaled:
            logits = matrix @ weights
            logits -= logits.max()
            probability = np.exp(logits) / np.exp(logits).sum()
            probability[target] -= 1.0
            gradient += matrix.T @ probability / len(scaled)
        weights -= learning_rate * gradient
    return {"weights": weights.tolist(), "mean": mean.tolist(), "std": std.tolist(), "mask": mask.tolist()}


def rule(feature, window, context):
    """A hand-written baseline: the argmax of one feature, nudged toward not moving."""
    weights = np.zeros(len(FEATURES))
    weights[FEATURES.index(feature)] = 1.0
    weights[FEATURES.index("distance")] = -0.05
    return {"weights": weights.tolist(), "mean": [0.0] * len(FEATURES), "std": [1.0] * len(FEATURES),
            "window": window, "context": context}


def evaluate(cases, methods):
    """Mean IoU and mean boundary error per method, over (row, proposal) pairs."""
    report = {}
    for name, method in methods.items():
        ious, errors = [], []
        for row, proposal in cases:
            start, end = method(row, proposal)
            truth_start, truth_end = float(row["expect"]["start"]), float(row["expect"]["end"])
            ious.append(iou(truth_start, truth_end, start, end))
            errors.append((abs(start - truth_start) + abs(end - truth_end)) / 2.0)
        report[name] = {"iou": float(np.mean(ious)) if ious else float("nan"),
                        "error_seconds": float(np.mean(errors)) if errors else float("nan"),
                        "cases": len(ious)}
    return report


def split_by_video(rows, seed, holdout=0.34):
    videos = sorted({row["file"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(videos)
    if len(videos) >= 2:
        held = set(videos[:max(1, round(len(videos) * holdout))])
        return [row for row in rows if row["file"] not in held], [row for row in rows if row["file"] in held], "video"
    rows = rows[:]
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * (1 - holdout)))
    return rows[:cut], rows[cut:], "row"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", help="labeled rows with exact intervals; repeatable")
    parser.add_argument("--proposals", help="a bench.run result over the same rows, for real proposals")
    parser.add_argument("--per-row", type=int, default=12, help="simulated proposals per row")
    parser.add_argument("--window", type=int, default=8, help="seconds a boundary may move")
    parser.add_argument("--context", type=int, default=4, help="seconds either side a feature averages over")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--with-cuts", action="store_true",
                        help="save the model that uses cut features (constructed data flatters them)")
    parser.add_argument("--out", default=str(BOUNDARY_MODEL))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    with Run("boundaries", args, seed=args.seed) as run:
        print(f"run {run.id}")
        datasets = args.dataset or [str(DATASET)]
        for path in datasets:
            run.input(path)
        rows = load_rows(datasets)
        if len(rows) < 6:
            raise SystemExit(f"Only {len(rows)} labeled intervals. Build some with python -m bench.construct.")
        print(f"{len(rows)} labeled intervals across {len({row['file'] for row in rows})} videos - embedding")
        attach_signals(rows)

        rng = random.Random(args.seed)
        if args.proposals:
            run.input(args.proposals)
            real = real_proposals(args.proposals)
            proposals = {row["id"]: real.get(row["id"], []) for row in rows}
            source = "real"
        else:
            proposals = {row["id"]: simulated_proposals(row, args.per_row, args.window, rng) for row in rows}
            source = "simulated"
        fit_rows, test_rows, split = split_by_video(rows, args.seed)
        print(f"{source} proposals; split by {split}: {len(fit_rows)} rows fit, {len(test_rows)} rows test")

        items = [item for row in fit_rows for proposal in proposals[row["id"]]
                 for item in examples(row, proposal, args.window, args.context)]
        if len(items) < 10:
            raise SystemExit(f"Only {len(items)} reachable boundaries to learn from.")
        full = np.ones(len(FEATURES))
        no_cuts = full.copy()
        for name in CUT_FEATURES:
            no_cuts[FEATURES.index(name)] = 0.0

        models = {}
        for label, mask in (("learned", full), ("learned, no cuts", no_cuts)):
            model = fit(items, mask, epochs=args.epochs)
            model.update({"window": args.window, "context": args.context})
            models[label] = model

        methods = {"proposal as given": lambda row, proposal: proposal}
        methods["snap to biggest cut"] = lambda row, proposal: refine_interval(
            row["signal"], *proposal, rule("cut_near", args.window, args.context))
        methods["sit on similarity step"] = lambda row, proposal: refine_interval(
            row["signal"], *proposal, rule("edge", args.window, args.context))
        for label, model in models.items():
            methods[label] = lambda row, proposal, model=model: refine_interval(row["signal"], *proposal, model)

        cases = [(row, proposal) for row in test_rows for proposal in proposals[row["id"]]]
        report = evaluate(cases, methods)
        print(f"\nheld-out: {len(cases)} proposals on {len(test_rows)} rows")
        print(f"{'method':<26}{'mean IoU':>10}{'boundary error':>17}")
        for name, values in report.items():
            print(f"{name:<26}{values['iou']:>10.3f}{values['error_seconds']:>15.1f}s")
            run.log({"method": name, **values})
        shortcut = report["learned"]["iou"] - report["learned, no cuts"]["iou"]
        if shortcut > 0:
            print(f"\ncut features are worth {shortcut:+.3f} IoU here. Every constructed boundary is a hard cut, "
                  f"so that is an upper bound on what they are worth on real footage.")
        else:
            print(f"\ncut features cost {shortcut:+.3f} IoU here, even though every constructed boundary is a hard "
                  f"cut - they fire on cuts inside the source footage too.")
        print("weights:", {name: round(weight, 2) for name, weight in zip(FEATURES, models["learned"]["weights"])})
        run.summarize(proposals=source, split=split, fit_rows=len(fit_rows), test_rows=len(test_rows),
                      results=report, cut_shortcut=shortcut)

        chosen = "learned" if args.with_cuts else "learned, no cuts"
        if not report[chosen]["iou"] > report["proposal as given"]["iou"]:
            print(f"\nThe {chosen} model did not beat the proposals it was given, so nothing is saved.")
            run.summarize(saved=False)
            return
        model = models[chosen]
        model.update({"trained_at": datetime.now().isoformat(timespec="seconds"), "run": run.id,
                      "features": list(FEATURES), "proposals": source, "seed": args.seed,
                      "metrics": {name: report[name] for name in ("proposal as given", chosen)}})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
        run.artifact(args.out, "boundary_model")
        run.summarize(saved=True, saved_variant=chosen)
        print(f"\nsaved the {chosen} model to {args.out}")


if __name__ == "__main__":
    main()
