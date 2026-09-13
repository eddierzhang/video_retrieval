"""Train the candidate ranker and calibrate a conformal prediction set.

    python -m bench.rank --dry-run           describe what has been collected
    python -m bench.rank                     train every loss and keep the best
    python -m bench.rank --loss listwise     train one
    python -m bench.rank --alpha 0.2         accept 80% coverage for smaller sets
    python -m bench.rank --learning-curve    is it short of data, or short of model?
    python -m bench.rank --ablate            which features carry the signal
    python -m bench.rank --seed 3            a different split and initialisation

Every invocation is recorded under bench/runs/ (see bench.tracking), including the hash of the
candidate log it read, so two results can be checked for "the data changed" before comparing.

Every verified search appends one row per candidate to local_data/learning/candidates.jsonl:
the features retrieval had already computed, and whether the vision model's verdict confirmed
that candidate. `bench.distill` fits a pointwise model to those rows - it asks "is this
candidate good?" one candidate at a time. But nothing consumes that answer one candidate at a
time. Verification walks the list from the top and stops, so what matters is the *order*, and
ordering is what a ranking loss optimises directly.

Three losses over the same network, so the comparison is an ablation rather than a claim:

  pointwise   binary cross-entropy per candidate            (what bench.distill does)
  pairwise    RankNet on positive/negative pairs, weighted
              by the NDCG each swap would change            (LambdaRank)
  listwise    softmax cross-entropy over a whole search     (ListNet)

Then split conformal turns the scores into a set with a coverage guarantee: on held-out
searches, the set it keeps contained a confirmed candidate at least `1 - alpha` of the time.
That is what tells the pipeline how many candidates are worth a vision call, which is the
expensive question - a verified search spends nearly all its time there.

The guarantee needs three disjoint splits - fit, calibrate, test - and whole searches on one
side of each line, never individual candidates.

Rows that the live ranker rejected but verified anyway are marked `explored`, and carry the
probability with which they were sampled. Pointwise training divides by that probability, so a
rare explored row counts for as much as the many confident ones it was sampled against; the
ranking losses see it as one more member of its search, which is what it is.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from bench.tracking import Run, seed_everything
from video_retrieval.learning import (
    CANDIDATE_EXAMPLES,
    FEATURE_NAMES,
    RANKER_MODEL,
    load_examples,
    ranker_probabilities,
    ranker_scores,
    vectorize,
)

LOSSES = ("pointwise", "pairwise", "listwise")
MAX_WEIGHT = 20.0  # a propensity of 1/20 is as much as one row is ever allowed to count

FEATURE_GROUPS = {
    "score": ("score", "relative_score"),
    "evidence": ("evidence_mass", "supporting_bins"),
    "duration": ("duration", "duration_ratio"),
    "rank": ("rank",),
    "channels": ("channel_video", "channel_metadata", "channel_speech", "channel_negative"),
    "ordering": ("ordering",),
}


def group_searches(rows):
    """One entry per search, because ranking is a per-search problem."""
    searches = {}
    for row in rows:
        # Rows written before searches carried an id fall back to the query and second.
        key = row.get("search") or f"{row.get('query')}@{int(float(row.get('at', 0)))}"
        searches.setdefault(key, []).append(row)
    groups = []
    for key, entries in searches.items():
        propensity = np.array([float(entry.get("propensity", 1.0) or 1.0) for entry in entries])
        groups.append({
            "search": key,
            "query": entries[0].get("query"),
            "executor": entries[0].get("executor"),
            "features": np.stack([vectorize(entry["features"]) for entry in entries]),
            "labels": np.array([float(entry["label"]) for entry in entries]),
            "weights": np.clip(1.0 / np.clip(propensity, 1e-6, None), 0.0, MAX_WEIGHT),
            "explored": np.array([bool(entry.get("explored", False)) for entry in entries]),
        })
    return groups


def split_searches(groups, seed=0, fractions=(0.5, 0.25)):
    """Fit / calibrate / test, split by whole search."""
    order = np.random.RandomState(seed).permutation(len(groups))
    first = max(1, int(len(groups) * fractions[0]))
    second = first + max(1, int(len(groups) * fractions[1]))
    pick = lambda indices: [groups[index] for index in indices]
    return pick(order[:first]), pick(order[first:second]), pick(order[second:])


def feature_mask(dropped=()):
    """A 1/0 vector over features, so an ablation is a multiply rather than a reshape."""
    mask = np.ones(len(FEATURE_NAMES))
    for group in dropped:
        for name in FEATURE_GROUPS[group]:
            mask[FEATURE_NAMES.index(name)] = 0.0
    return mask


# ------------------------------------------------------------------ metrics

def dcg(labels):
    return float(sum(label / math.log2(position + 2) for position, label in enumerate(labels)))


def ndcg(labels, order, k=5):
    ranked = [labels[index] for index in order[:k]]
    ideal = sorted(labels, reverse=True)[:k]
    best = dcg(ideal)
    return dcg(ranked) / best if best > 0 else float("nan")


def reciprocal_rank(labels, order):
    for position, index in enumerate(order):
        if labels[index] > 0:
            return 1.0 / (position + 1)
    return 0.0


def measure(groups, score_of, k=5):
    """Ranking quality over whole searches. Searches with nothing to find are skipped."""
    scores = {"ndcg": [], "mrr": [], "recall@3": []}
    for group in groups:
        labels = group["labels"]
        if labels.sum() == 0:
            continue
        order = list(np.argsort(-score_of(group)))
        scores["ndcg"].append(ndcg(labels, order, k))
        scores["mrr"].append(reciprocal_rank(labels, order))
        scores["recall@3"].append(float(labels[order[:3]].sum() / labels.sum()))
    return {name: float(np.mean(values)) if values else float("nan") for name, values in scores.items()}


# ----------------------------------------------------------------- training

def train(groups, loss_name, hidden=32, epochs=400, seed=0, mask=None, verbose=False):
    import torch

    torch.manual_seed(seed)
    mask = np.ones(len(FEATURE_NAMES)) if mask is None else np.asarray(mask, dtype=np.float64)
    features = np.concatenate([group["features"] for group in groups])
    mean, std = features.mean(axis=0), features.std(axis=0)
    std = np.where(std > 0, std, 1.0)

    usable = [group for group in groups
              if loss_name == "pointwise" or (0 < group["labels"].sum() < len(group["labels"]))]
    if not usable:
        raise SystemExit(f"No search has both a confirmed and a rejected candidate, so a "
                         f"{loss_name} loss has nothing to learn from.")
    tensors = [(torch.tensor(((group["features"] - mean) / std) * mask, dtype=torch.float32),
                torch.tensor(group["labels"], dtype=torch.float32),
                torch.tensor(group["weights"], dtype=torch.float32)) for group in usable]

    sizes = [len(FEATURE_NAMES)] + ([hidden] if hidden else []) + [1]
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    for epoch in range(epochs):
        total = 0.0
        optimizer.zero_grad()
        for inputs, labels, weights in tensors:
            scores = model(inputs)[:, 0]
            if loss_name == "pointwise":
                # Inverse propensity: a row that was sampled one time in ten stands for ten.
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    scores, labels, weight=weights)
            elif loss_name == "listwise":
                # Softmax over the search; the target mass sits on the confirmed candidates.
                log_probabilities = torch.log_softmax(scores, dim=0)
                loss = -torch.logsumexp(log_probabilities[labels > 0], dim=0)
            else:
                loss = _lambda_rank(scores, labels)
            total += float(loss.detach())
            (loss / len(tensors)).backward()
        optimizer.step()
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"      epoch {epoch:>4}  {loss_name} loss {total / len(tensors):.4f}")

    weights = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            weights.append({"w": layer.weight.detach().numpy().T.tolist(),
                            "b": layer.bias.detach().numpy().tolist()})
    return {"mean": mean.tolist(), "std": std.tolist(), "mask": mask.tolist(), "layers": weights}


def _lambda_rank(scores, labels):
    """RankNet on every positive/negative pair, weighted by the NDCG the swap would change."""
    import torch

    positives = torch.nonzero(labels > 0)[:, 0]
    negatives = torch.nonzero(labels <= 0)[:, 0]
    order = torch.argsort(scores, descending=True)
    positions = torch.empty_like(order)
    positions[order] = torch.arange(len(order), device=scores.device)
    discount = 1.0 / torch.log2(positions.double() + 2.0)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(int(labels.sum().item())))

    difference = scores[positives][:, None] - scores[negatives][None, :]
    delta = (discount[positives][:, None] - discount[negatives][None, :]).abs() / max(ideal, 1e-8)
    pairs = torch.nn.functional.softplus(-difference) * delta.float()
    return pairs.mean()


def fit_platt(scores, labels, epochs=3000):
    """A one-dimensional logistic fit, so a score reads as a probability."""
    a, b = 1.0, 0.0
    for _ in range(epochs):
        probability = 1.0 / (1.0 + np.exp(-(a * scores + b)))
        error = probability - labels
        a -= 0.5 * float(np.mean(error * scores))
        b -= 0.5 * float(np.mean(error))
    return {"a": float(a), "b": float(b)}


# ---------------------------------------------------------------- conformal

def calibrate_conformal(groups, probability_of, alpha=0.1):
    """Split conformal: the largest cut that still covers `1 - alpha` of calibration searches.

    The nonconformity of a search is how badly its *best* confirmed candidate scored. Keeping
    everything at least that good then contains a confirmed candidate on a new search with
    probability at least 1 - alpha, over searches, provided they are exchangeable.
    """
    nonconformity = []
    for group in groups:
        if group["labels"].sum() == 0:
            continue  # a search with nothing to find says nothing about coverage
        nonconformity.append(1.0 - float(probability_of(group)[group["labels"] > 0].max()))
    if not nonconformity:
        return None
    n = len(nonconformity)
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    qhat = float(np.quantile(nonconformity, level, method="higher"))
    return {"alpha": alpha, "coverage": 1.0 - alpha, "threshold": 1.0 - qhat,
            "calibration_searches": n}


def evaluate_sets(groups, probability_of, threshold, keep_min=3):
    """What the conformal set costs and what it keeps, on searches it has never seen."""
    covered, kept_fraction, positives_kept = [], [], []
    for group in groups:
        probabilities = probability_of(group)
        order = list(np.argsort(-probabilities))
        kept = [index for index in order if probabilities[index] >= threshold]
        if len(kept) < keep_min:
            kept = order[:keep_min]
        kept_fraction.append(len(kept) / len(order))
        if group["labels"].sum() > 0:
            covered.append(float(group["labels"][kept].sum() > 0))
            positives_kept.append(float(group["labels"][kept].sum() / group["labels"].sum()))
    return {
        "coverage": float(np.mean(covered)) if covered else float("nan"),
        "candidates_kept": float(np.mean(kept_fraction)),
        "confirmed_kept": float(np.mean(positives_kept)) if positives_kept else float("nan"),
        "searches": len(groups),
    }


# ------------------------------------------------------------------- driver

def build(fit, calibration, test, loss_name, hidden, epochs, alpha, mask=None, verbose=False, seed=0):
    """Train, calibrate on the middle split, and measure on the last one."""
    model = train(fit, loss_name, hidden=hidden, epochs=epochs, seed=seed, mask=mask, verbose=verbose)
    score_of = lambda group: ranker_scores(model, group["features"])
    model["platt"] = fit_platt(np.concatenate([score_of(group) for group in fit]),
                               np.concatenate([group["labels"] for group in fit]))
    probability_of = lambda group: ranker_probabilities(model, score_of(group))
    conformal = calibrate_conformal(calibration, probability_of, alpha)
    if not conformal:
        raise SystemExit("No calibration search has a confirmed candidate; cannot calibrate.")
    model["conformal"] = conformal
    return {"model": model, "quality": measure(test, score_of),
            "sets": evaluate_sets(test, probability_of, conformal["threshold"])}


def describe(groups):
    sizes = [len(group["labels"]) for group in groups]
    positives = [int(group["labels"].sum()) for group in groups]
    explored = sum(int(group["explored"].sum()) for group in groups)
    print(f"{sum(sizes)} candidates across {len(groups)} searches "
          f"({np.mean(sizes):.1f} per search, {sum(positives)} confirmed, {explored} explored)")
    both = sum(1 for group in groups if 0 < group["labels"].sum() < len(group["labels"]))
    print(f"{both} searches have both a confirmed and a rejected candidate - "
          f"only those teach an ordering")
    return both




def learning_curve(fit, calibration, test, loss_name, args, run=None):
    """Train on a growing slice of the fit split: is this short of data, or short of model?"""
    print(f"\nlearning curve ({loss_name}, held-out ndcg@5)")
    order = np.random.RandomState(args.seed).permutation(len(fit))
    rows = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        take = max(2, int(len(fit) * fraction))
        subset = [fit[index] for index in order[:take]]
        try:
            result = build(subset, calibration, test, loss_name, args.hidden, args.epochs, args.alpha,
                           seed=args.seed)
        except SystemExit as stop:
            print(f"   {take:>4} searches   {stop}")
            continue
        rows.append((take, result["quality"]["ndcg"]))
        print(f"   {take:>4} searches   ndcg@5 {result['quality']['ndcg']:.3f}")
        if run:
            run.log({"curve_searches": take, "ndcg": result["quality"]["ndcg"], "loss": loss_name}, step=take)
    if len(rows) >= 2:
        slope = rows[-1][1] - rows[-2][1]
        verdict = "more data should still help" if slope > 0.01 else "more data is not the bottleneck"
        print(f"   the last {rows[-1][0] - rows[-2][0]} searches moved ndcg@5 by {slope:+.3f} - {verdict}")
        if run:
            run.summarize(learning_curve={"points": rows, "last_slope": slope, "verdict": verdict})


def ablate(fit, calibration, test, loss_name, args, full, run=None):
    """Drop each group of features and see what the ordering loses without them."""
    print(f"\nfeature ablation ({loss_name}, held-out ndcg@5, full model {full:.3f})")
    drops = {}
    for name in FEATURE_GROUPS:
        result = build(fit, calibration, test, loss_name, args.hidden, args.epochs, args.alpha,
                       mask=feature_mask([name]), seed=args.seed)
        score = result["quality"]["ndcg"]
        drops[name] = score - full
        print(f"   without {name:<10} {score:.3f}   {score - full:+.3f}")
        if run:
            run.log({"ablated": name, "ndcg": score, "change": score - full, "loss": loss_name})
    if run:
        run.summarize(ablation=drops)


def run_ranking(args, run):
    examples = Path(args.examples) if args.examples else CANDIDATE_EXAMPLES
    run.input(examples)
    rows = load_examples(examples)
    if not rows:
        raise SystemExit("No examples yet. Run some verified searches first.")
    groups = [group for group in group_searches(rows) if len(group["labels"]) >= 2]
    usable = describe(groups)
    run.summarize(searches=len(groups), usable_searches=usable,
                  candidates=sum(len(group["labels"]) for group in groups))
    if args.dry_run:
        return
    if len(groups) < args.min_searches or usable < 5:
        raise SystemExit(f"Not enough data yet: need {args.min_searches}+ searches, at least 5 of "
                         f"them with both outcomes. Run `python -m bench.run --mode verified`.")

    fit, calibration, test = split_searches(groups, seed=args.seed)
    print(f"searches: {len(fit)} fit, {len(calibration)} calibrate, {len(test)} test")
    raw = lambda group: group["features"][:, FEATURE_NAMES.index("score")]
    baseline = measure(test, raw)
    run.summarize(baseline=baseline)
    print("\nbaseline - candidates in retrieval's own order")
    print(f"   ndcg@5 {baseline['ndcg']:.3f}   mrr {baseline['mrr']:.3f}   recall@3 {baseline['recall@3']:.3f}")

    results = {}
    for loss_name in (LOSSES if args.loss == "all" else (args.loss,)):
        print(f"\n{loss_name}")
        result = build(fit, calibration, test, loss_name, args.hidden, args.epochs, args.alpha,
                       verbose=True, seed=args.seed)
        conformal, quality, sets = result["model"]["conformal"], result["quality"], result["sets"]
        print(f"   ndcg@5 {quality['ndcg']:.3f}   mrr {quality['mrr']:.3f}   recall@3 {quality['recall@3']:.3f}")
        print(f"   conformal threshold {conformal['threshold']:.3f} from {conformal['calibration_searches']} searches")
        print(f"   test coverage {sets['coverage']:.1%} (target {conformal['coverage']:.0%})   "
              f"verifies {sets['candidates_kept']:.1%} of candidates   "
              f"keeps {sets['confirmed_kept']:.1%} of confirmed ones")
        run.log({"loss": loss_name, **quality, **{f"set_{key}": value for key, value in sets.items()},
                 "threshold": conformal["threshold"]})
        results[loss_name] = result

    print("\n" + "-" * 78)
    print(f"{'loss':<12}{'ndcg@5':>9}{'mrr':>9}{'recall@3':>11}{'coverage':>11}{'verified':>11}")
    print(f"{'baseline':<12}{baseline['ndcg']:>9.3f}{baseline['mrr']:>9.3f}{baseline['recall@3']:>11.3f}"
          f"{'-':>11}{'100.0%':>11}")
    for loss_name, result in results.items():
        print(f"{loss_name:<12}{result['quality']['ndcg']:>9.3f}{result['quality']['mrr']:>9.3f}"
              f"{result['quality']['recall@3']:>11.3f}{result['sets']['coverage']:>10.1%}"
              f"{result['sets']['candidates_kept']:>11.1%}")

    best = max(results, key=lambda name: results[name]["quality"]["ndcg"])
    run.summarize(losses={name: {**result["quality"], **result["sets"]} for name, result in results.items()},
                  best_loss=best)
    if args.learning_curve:
        learning_curve(fit, calibration, test, best, args, run)
    if args.ablate:
        ablate(fit, calibration, test, best, args, results[best]["quality"]["ndcg"], run)

    if not results[best]["quality"]["ndcg"] > baseline["ndcg"]:
        print("\nNo loss beat retrieval's own ordering on held-out searches, so nothing is saved.")
        print("Collect more verified searches and try again.")
        run.summarize(saved=False)
        return

    model = results[best]["model"]
    model.update({
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "run": run.id,
        "loss": best,
        "features": list(FEATURE_NAMES),
        "seed": args.seed,
        "searches": len(groups),
        "candidates": sum(len(group["labels"]) for group in groups),
        "metrics": {"baseline": baseline, "test": results[best]["quality"], "sets": results[best]["sets"]},
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
    run.artifact(args.out, "candidate_ranker")
    run.summarize(saved=True)
    print(f"\nkept the {best} model, written to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", default=None)
    parser.add_argument("--loss", choices=LOSSES + ("all",), default="all")
    parser.add_argument("--alpha", type=float, default=0.1, help="1 - alpha is the coverage target")
    parser.add_argument("--hidden", type=int, default=32, help="0 for a linear model")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--min-searches", type=int, default=20)
    parser.add_argument("--learning-curve", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(RANKER_MODEL))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    with Run("rank", args, seed=args.seed) as run:
        print(f"run {run.id}")
        run_ranking(args, run)


if __name__ == "__main__":
    main()
