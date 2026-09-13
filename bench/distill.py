"""Train the candidate pre-filter from the verifier's own accept/reject decisions.

    python -m bench.distill              train on everything collected so far
    python -m bench.distill --dry-run    just describe the collected data

Every verified search appends one row per candidate to local_data/learning/candidates.jsonl:
the features retrieval had already computed, and whether that candidate went on to produce a
confirmed match. Learning to predict that lets the pipeline skip candidates the vision model
would have rejected, which is where a verified search spends nearly all of its time.

No hand-labeling is involved - the 4B vision model is the teacher and a logistic regression
is the student.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json

import numpy as np

from bench.tracking import Run, seed_everything
from video_retrieval.learning import CANDIDATE_EXAMPLES, FEATURE_NAMES, PREFILTER_MODEL, load_examples, vectorize

MIN_EXAMPLES = 40


def auc(labels, scores):
    """Probability that a random positive outranks a random negative."""
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def fit(features, labels, epochs=800, learning_rate=0.5, l2=1e-3):
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    for epoch in range(epochs):
        probability = 1.0 / (1.0 + np.exp(-(features @ weights + bias)))
        error = probability - labels
        step = learning_rate * (1.0 - epoch / (2 * epochs))
        weights -= step * (features.T @ error / len(labels) + l2 * weights)
        bias -= step * float(error.mean())
    return weights, bias


def choose_threshold(labels, scores, keep_recall=0.95):
    """The highest cutoff that still keeps `keep_recall` of the true positives."""
    positives = scores[labels == 1]
    if not len(positives):
        return 0.0
    return float(np.quantile(positives, 1.0 - keep_recall))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", default=str(CANDIDATE_EXAMPLES))
    parser.add_argument("--out", default=str(PREFILTER_MODEL))
    parser.add_argument("--keep-recall", type=float, default=0.95, help="fraction of confirmed candidates the filter must keep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    with Run("distill", args, seed=args.seed, inputs=[args.examples]) as run:
        print(f"run {run.id}")
        distill(args, run)


def distill(args, run):
    rows = load_examples(args.examples)
    if not rows:
        raise SystemExit(f"No examples yet at {args.examples}. Run some verified searches first.")
    features = np.stack([vectorize(row["features"]) for row in rows])
    labels = np.array([float(row["label"]) for row in rows])
    print(f"{len(rows)} candidates from {len({row['query'] for row in rows})} queries, "
          f"{int(labels.sum())} of them confirmed ({labels.mean():.1%})")

    if args.dry_run:
        for i, name in enumerate(FEATURE_NAMES):
            positive = features[labels == 1, i].mean() if labels.sum() else float("nan")
            negative = features[labels == 0, i].mean() if (1 - labels).sum() else float("nan")
            print(f"   {name:<18} confirmed {positive:8.3f}   rejected {negative:8.3f}")
        return
    if len(rows) < MIN_EXAMPLES or labels.sum() < 5 or (1 - labels).sum() < 5:
        raise SystemExit(f"Not enough data yet: need {MIN_EXAMPLES}+ candidates with at least 5 of each outcome.")

    order = np.random.RandomState(args.seed).permutation(len(labels))
    split = int(len(order) * 0.8)
    train, holdout = order[:split], order[split:]
    mean, std = features[train].mean(axis=0), features[train].std(axis=0)
    scaled = (features - mean) / np.where(std > 0, std, 1.0)

    weights, bias = fit(scaled[train], labels[train])
    scores = 1.0 / (1.0 + np.exp(-(scaled @ weights + bias)))
    threshold = choose_threshold(labels[holdout], scores[holdout], args.keep_recall)
    kept = scores >= threshold

    print(f"holdout AUC {auc(labels[holdout], scores[holdout]):.3f}   train AUC {auc(labels[train], scores[train]):.3f}")
    print(f"threshold {threshold:.3f} keeps {kept.mean():.0%} of candidates "
          f"and {labels[kept == 1].sum() / max(1.0, labels.sum()):.0%} of the confirmed ones")
    print("weights:", {name: round(float(w), 2) for name, w in zip(FEATURE_NAMES, weights)})

    model = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "examples": len(rows),
        "positives": int(labels.sum()),
        "features": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "threshold": threshold,
        "holdout_auc": auc(labels[holdout], scores[holdout]),
    }
    PREFILTER_MODEL.parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(json.dumps(model, indent=2))
    print("written to", args.out)
    run.artifact(args.out, "candidate_prefilter")
    run.summarize(examples=len(rows), threshold=threshold, holdout_auc=model["holdout_auc"],
                  kept=float(kept.mean()))


if __name__ == "__main__":
    main()
