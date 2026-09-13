"""Tune how the pipeline picks a moment, against rows whose answers are known.

    python -m bench.tune                         cache what is missing, then search
    python -m bench.tune --trials 300 --refine 150 --folds 5
    python -m bench.tune --dry-run               cache and score the defaults only

On the constructed benchmark, quick mode puts its answer in the wrong place about half the time,
and tighter boundaries cannot fix an answer in the wrong place. Which window wins is decided by
about twenty numbers - eight scoring weights (retrieval.DEFAULT_SCORING) and the evidence-map,
candidate, refinement and NMS settings retrieve_video takes - all chosen by hand.

Searching them through the real pipeline would cost a planner call per row per trial. But
everything after retrieval calls no model, so each row's plan and retrieval hits are cached once
(local_data/tuning_cache/), and a trial replays only pipeline.locate_candidates, NMS and the
learned boundaries - the same functions a quick search runs, in milliseconds.

Twenty settings and a few dozen rows is a recipe for fitting the benchmark instead of the task, so:

  cross-validation   rows are split into folds by timeline; each fold's settings are searched on
                     the other timelines and scored on its own. The number that decides anything
                     is the held-out one.
  a gate             settings are saved only if, across folds, they beat the defaults on
                     timelines they never saw.
  one mode           they are tuned for quick mode and only ever applied to quick searches.

The cache is keyed by the query, the video's index, the planner model and the planner's own
source, so changing the planning prompt invalidates it rather than silently replaying old plans.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np

from bench.tracking import Run, seed_everything
from video_retrieval import boundaries, retrieval
from video_retrieval.config import DATA_DIR
from video_retrieval.pipeline import TUNED_SETTINGS, locate_candidates, retrieval_instances, retrieve_video
from video_retrieval.verification import temporal_nms

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "constructed.json"
CACHE = DATA_DIR / "tuning_cache"

PIPELINE_SPACE = {
    "evidence_bin_size": ("choice", [1.0, 2.0, 3.0, 4.0]),
    "evidence_smoothing_bins": ("choice", [0, 1, 2]),
    "candidate_max_gap": ("float", 2.0, 20.0),
    "candidate_padding": ("float", 0.0, 12.0),
    "candidate_relative_score_floor": ("float", 0.02, 0.6),
    "recursive_candidate_search": ("choice", [True, False]),
    "recursive_search_depth": ("choice", [1, 2, 3, 4]),
    "recursive_shrink_factor": ("float", 0.3, 0.8),
    "recursive_child_relative_score_floor": ("float", 0.3, 0.95),
    "recursive_min_window_seconds": ("float", 4.0, 30.0),
    "recursive_context_padding": ("float", 0.0, 8.0),
    "nms_iou_threshold": ("float", 0.3, 0.8),
}
SCORING_SPACE = {
    "negative_evidence_weight": ("float", 0.0, 1.5),
    "ordering_satisfied_boost": ("float", 1.0, 1.5),
    "ordering_violated_penalty": ("float", 0.4, 1.0),
    "predicate_floor": ("float", 0.2, 1.0),
    "peak_share": ("float", 0.4, 1.0),
    "window_peak": ("float", 0.0, 1.0),
    "window_top_mean": ("float", 0.0, 1.0),
    "window_mean": ("float", 0.0, 1.0),
}


def defaults():
    """The settings a search uses today, read from the code so they cannot drift from it."""
    signature = inspect.signature(retrieve_video).parameters
    return {"pipeline": {name: signature[name].default for name in PIPELINE_SPACE},
            "scoring": dict(retrieval.DEFAULT_SCORING)}


def iou(a_start, a_end, b_start, b_end):
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


# ------------------------------------------------------------------- caching

def planner_fingerprint(models):
    source = inspect.getsource(retrieval.plan_query) + inspect.getsource(retrieval.confirm_text_route)
    return hashlib.sha256((source + json.dumps(models.signature(), sort_keys=True)).encode()).hexdigest()[:16]


def cache_rows(rows, library, models, run):
    """Plan and retrieve each row once; later trials replay only the model-free stages."""
    from video_retrieval.local_backend import embed_text, use_models
    from video_retrieval.local_indexing import load_index

    CACHE.mkdir(parents=True, exist_ok=True)
    fingerprint = planner_fingerprint(models)
    records = {record["name"]: record for record in library.list()}
    pipelines, cached, fresh = {}, [], 0
    started = time.time()
    with use_models(models):
        for row in rows:
            record = records.get(row["video"])
            if not record or not record.get("index"):
                print(f"   {row['id']}: {row['video']} is not indexed; skipped")
                continue
            key = hashlib.sha256(json.dumps([row["query"], record["id"], record["index"]["dir"], fingerprint]).encode()).hexdigest()[:24]
            path = CACHE / f"{key}.pkl"
            if path.is_file():
                with open(path, "rb") as file:
                    entry = pickle.load(file)
            else:
                if row["video"] not in pipelines:
                    folder = library.video_dir(record["id"])
                    pipelines[row["video"]] = load_index(folder / record["index"]["dir"], models, folder / record["source"])
                resources = pipelines[row["video"]].resources
                plan = retrieval.plan_query(row["query"])
                results = None
                if plan.get("executor") == "temporal_grounding":
                    results = retrieval.run_retrieval_plan(
                        plan, resources.video_index, resources.video_metadata, resources.metadata_index,
                        resources.metadata_records, resources.transcript_index, resources.transcript_bm25,
                        resources.transcript_metadata, top_k=200)
                entry = {"plan": plan, "results": results, "duration": float(resources.manifest["video"]["duration"]),
                         "source": str(library.video_dir(record["id"]) / record["source"]),
                         "query_vector": np.asarray(embed_text(row["query"]), dtype=np.float32)}
                with open(path, "wb") as file:
                    pickle.dump(entry, file)
                fresh += 1
                print(f"   cached {row['id']} ({plan.get('executor')})")
            cached.append({**row, **entry})
    print(f"{len(cached)} rows ready, {fresh} planned and retrieved now in {time.time() - started:.0f}s")
    run.summarize(rows_cached=len(cached), rows_planned_now=fresh)
    return cached


def attach_signals(rows):
    from video_retrieval.local_backend import frame_embeddings

    for row in rows:
        times, vectors = frame_embeddings(row["source"])
        row["signal"] = boundaries.signals(times, vectors, row["query_vector"])


# ---------------------------------------------------------------- replaying

def replay(row, settings, boundary_model):
    """What a quick search would answer for this row under `settings`, most confident first."""
    pipeline = settings["pipeline"]
    locate = {name: value for name, value in pipeline.items() if name != "nms_iou_threshold"}
    with retrieval.scoring_override(settings["scoring"]):
        located = locate_candidates(row["results"], row["plan"], row["duration"], **locate)
    if located["empty"]:
        return []
    instances = temporal_nms(retrieval_instances(located["candidates"]),
                             iou_threshold=pipeline["nms_iou_threshold"], preserve_distinct_actors=True)
    if boundary_model:
        refined = []
        for instance in instances:
            start, end = boundaries.refine_interval(row["signal"], instance["start"], instance["end"], boundary_model)
            refined.append({**instance, "start": max(0.0, start), "end": min(row["duration"], end)})
        instances = refined
    instances = sorted(instances, key=lambda instance: -float(instance.get("confidence", 0.0)))
    if row["plan"].get("return_mode", "all") == "best":
        instances = instances[:1]
    return instances


def evaluate(rows, settings, boundary_model):
    top, best, hits = [], [], []
    for row in rows:
        answers = replay(row, settings, boundary_model)
        truth = (float(row["expect"]["start"]), float(row["expect"]["end"]))
        scores = [iou(*truth, float(answer["start"]), float(answer["end"])) for answer in answers[:5]]
        top.append(scores[0] if scores else 0.0)
        best.append(max(scores, default=0.0))
        hits.append(float(top[-1] >= 0.3))
    return {"top1_iou": float(np.mean(top)) if top else float("nan"),
            "best5_iou": float(np.mean(best)) if best else float("nan"),
            "top1_accuracy": float(np.mean(hits)) if hits else float("nan")}


# ------------------------------------------------------------------ searching

def sample(space, rng):
    values = {}
    for name, spec in space.items():
        values[name] = rng.choice(spec[1]) if spec[0] == "choice" else rng.uniform(spec[1], spec[2])
    return values


def perturb(values, space, rng, scale):
    changed = dict(values)
    for name, spec in space.items():
        if rng.random() > 0.35:
            continue  # move a few settings at a time, so a step can be told apart from noise
        if spec[0] == "choice":
            changed[name] = rng.choice(spec[1])
        else:
            low, high = spec[1], spec[2]
            current = changed[name] if changed[name] is not None else (low + high) / 2
            changed[name] = min(high, max(low, current + rng.gauss(0.0, scale * (high - low))))
    return changed


def search(rows, trials, refine, rng, boundary_model, log=None):
    """Random search from the defaults, then hill climbing around the best settings found."""
    best = defaults()
    best_score = evaluate(rows, best, boundary_model)["top1_iou"]
    for trial in range(trials):
        candidate = {"pipeline": sample(PIPELINE_SPACE, rng), "scoring": sample(SCORING_SPACE, rng)}
        score = evaluate(rows, candidate, boundary_model)["top1_iou"]
        if score > best_score:
            best, best_score = candidate, score
            if log:
                log(trial, score)
    for step in range(refine):
        scale = 0.15 * (1.0 - step / max(1, refine)) + 0.02
        candidate = {"pipeline": perturb(best["pipeline"], PIPELINE_SPACE, rng, scale),
                     "scoring": perturb(best["scoring"], SCORING_SPACE, rng, scale)}
        score = evaluate(rows, candidate, boundary_model)["top1_iou"]
        if score > best_score:
            best, best_score = candidate, score
            if log:
                log(trials + step, score)
    return best, best_score


def timeline_of(row):
    return row["timeline"] if "timeline" in row else row["video"]


def folds_by_timeline(rows, count, seed):
    timelines = sorted({timeline_of(row) for row in rows}, key=str)
    random.Random(seed).shuffle(timelines)
    count = max(2, min(count, len(timelines)))
    groups = [timelines[index::count] for index in range(count)]
    return [[row for row in rows if timeline_of(row) in group] for group in groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--trials", type=int, default=200, help="random settings tried per search")
    parser.add_argument("--refine", type=int, default=100, help="hill-climbing steps after that")
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds, by timeline")
    parser.add_argument("--no-boundaries", action="store_true", help="tune without the learned boundary model")
    parser.add_argument("--out", default=str(TUNED_SETTINGS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    with Run("tune", args, seed=args.seed) as run:
        print(f"run {run.id}")
        from video_retrieval.local_backend import LocalModels
        from webapp.library import Library

        run.input(args.dataset)
        document = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        rows = [row for row in document["rows"] if row["expect"].get("end") is not None and "text" not in row["expect"]]
        print(f"{len(rows)} event rows across {len({row.get('timeline') for row in rows})} timelines")
        models = LocalModels()
        run.note(models=models.signature())
        rows = cache_rows(rows, Library(), models, run)
        text_routed = [row for row in rows if row["results"] is None]
        rows = [row for row in rows if row["results"] is not None]
        if text_routed:
            print(f"{len(text_routed)} rows were routed to the text reader and are left out: "
                  + ", ".join(row["id"] for row in text_routed))
        attach_signals(rows)
        boundary_model = None if args.no_boundaries else boundaries.load_boundary_model()
        print(f"learned boundaries: {'on' if boundary_model else 'off'}")

        started = time.time()
        default_score = evaluate(rows, defaults(), boundary_model)
        per_trial = time.time() - started
        print(f"\ndefaults on all {len(rows)} rows: top-1 IoU {default_score['top1_iou']:.3f}   "
              f"top-1 accuracy {default_score['top1_accuracy']:.0%}   best-of-5 IoU {default_score['best5_iou']:.3f}")
        print(f"one trial replays every row in {per_trial * 1000:.0f} ms")
        run.summarize(defaults=default_score, trial_seconds=per_trial)
        if args.dry_run:
            return

        rng = random.Random(args.seed)
        folds = folds_by_timeline(rows, args.folds, args.seed)
        print(f"\ncross-validation: {len(folds)} folds by timeline, "
              f"{args.trials} random + {args.refine} refining trials each")
        held_default, held_tuned = [], []
        for number, held in enumerate(folds, 1):
            fit = [row for row in rows if row not in held]
            settings, fit_score = search(fit, args.trials, args.refine, rng, boundary_model)
            before = evaluate(held, defaults(), boundary_model)
            after = evaluate(held, settings, boundary_model)
            held_default.append((before["top1_iou"], len(held)))
            held_tuned.append((after["top1_iou"], len(held)))
            print(f"   fold {number}: {len(fit)} fit / {len(held)} held out   fit top-1 {fit_score:.3f}   "
                  f"held-out top-1 {before['top1_iou']:.3f} -> {after['top1_iou']:.3f}   "
                  f"accuracy {before['top1_accuracy']:.0%} -> {after['top1_accuracy']:.0%}")
            run.log({"fold": number, "fit_rows": len(fit), "held_rows": len(held), "fit_top1": fit_score,
                     "held_default": before, "held_tuned": after})

        weigh = lambda pairs: sum(score * count for score, count in pairs) / sum(count for _, count in pairs)
        cv_default, cv_tuned = weigh(held_default), weigh(held_tuned)
        wins = sum(after > before for (before, _), (after, _) in zip(held_default, held_tuned))
        print(f"\nheld-out top-1 IoU across folds: defaults {cv_default:.3f}   tuned {cv_tuned:.3f}   "
              f"({cv_tuned - cv_default:+.3f}; better in {wins} of {len(folds)} folds)")
        run.summarize(cv_default_top1=cv_default, cv_tuned_top1=cv_tuned, folds_improved=wins, folds=len(folds))

        print("\nfinal search on every row")
        settings, full_score = search(rows, args.trials, args.refine, rng, boundary_model,
                                      log=lambda trial, score: print(f"   trial {trial:>4}: top-1 IoU {score:.3f}"))
        final = evaluate(rows, settings, boundary_model)
        changed = {section: {name: value for name, value in settings[section].items()
                             if value != defaults()[section][name]} for section in settings}
        print(f"all rows: top-1 IoU {default_score['top1_iou']:.3f} -> {final['top1_iou']:.3f}   "
              f"(fitted on these rows - the held-out number above is the honest one)")
        for section, values in changed.items():
            for name, value in values.items():
                before = defaults()[section][name]
                shown = lambda item: f"{item:.3g}" if isinstance(item, float) else str(item)
                print(f"   {name:<38} {shown(before):>8} -> {shown(value)}")
        run.summarize(final_fit=final, settings=settings)

        if not (cv_tuned > cv_default and wins > len(folds) / 2):
            print("\nTuned settings did not beat the defaults on held-out timelines in most folds, "
                  "so nothing is saved.")
            run.summarize(saved=False)
            return
        document = {
            "mode": "quick",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "run": run.id,
            "rows": len(rows),
            "boundaries": bool(boundary_model),
            "metrics": {"cv_default_top1": cv_default, "cv_tuned_top1": cv_tuned, "folds_improved": wins,
                        "folds": len(folds), "defaults_all_rows": default_score, "tuned_all_rows": final},
            "pipeline": settings["pipeline"],
            "scoring": settings["scoring"],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(document, indent=2), encoding="utf-8")
        run.artifact(args.out, "retrieval_settings")
        run.summarize(saved=True)
        print(f"\nsaved to {args.out}; quick searches use it from now on. Delete the file to go back.")


if __name__ == "__main__":
    main()
