"""Replay logged searches without the vision model, and learn when to stop verifying.

    python -m bench.replay                          compare stopping policies on logged searches
    python -m bench.replay --goal first             only the first confirmed match is worth anything
    python -m bench.replay --value-seconds 120      a confirmed match is worth two minutes of calls
    python -m bench.replay --ranker                 order and price candidates with the trained ranker

A verified search spends nearly all its time asking the vision model about candidates, one after
another. Deciding when to stop asking is a sequential decision problem, and the obvious way to
learn one - try policies and watch what happens - costs a minute or more per attempt. That is
what makes reinforcement learning impractical here, and this is what fixes it.

Every verified search already logged, for each candidate, whether the verifier confirmed it and
how long that took. So a search can be *replayed*: a policy asks for candidates in its own order,
the replay answers from the log and charges the logged seconds, and an episode costs microseconds
instead of a minute. Thousands of episodes run in the time one real search takes.

The reward is what a user would care about: `value-seconds` per confirmed match found, minus the
seconds spent finding it. With `--goal all` every confirmed match counts; with `--goal first` only
the first does, so there is no reason to keep looking after it.

Policies compared, all ordering candidates by their probability of being confirmed:

  verify all          what the pipeline does today
  top k               a fixed budget of k calls
  threshold           verify while the probability clears a cut chosen on the fit searches -
                      a static rule, like the conformal set
  optimal stopping    backward induction over the ordered list: verify the next candidate while
                      the expected value of continuing, including everything after it, beats
                      stopping now. Exactly optimal if the probabilities are calibrated and
                      candidates are independent - which they are not quite, and which is why
  fitted Q            learns the value of verifying from replayed outcomes instead of trusting
                      the probabilities: value-based RL (fitted Q iteration) over a small state -
                      the next candidate's probability, its cost, whether anything has been found
                      yet, how far down the list it is. If it beats optimal stopping, the
                      probabilities were not telling the whole truth.

What the replay cannot do: answer for a candidate that was never verified. While every logged
candidate was verified - true until a ranker is trained - every policy can be replayed exactly.
After that, only the explored rows (see select_candidates) keep that true, which is one more
reason they exist.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from bench.rank import group_searches, split_searches
from bench.tracking import Run, seed_everything
from video_retrieval.learning import (
    CANDIDATE_EXAMPLES,
    FEATURE_NAMES,
    load_examples,
    load_ranker,
    ranker_probabilities,
    ranker_scores,
)


# ------------------------------------------------------------------- episodes

def episodes(rows, default_cost):
    """One episode per search: its candidates' outcomes and what each one cost to verify."""
    groups = [group for group in group_searches(rows) if len(group["labels"]) >= 1]
    by_search = {}
    for row in rows:
        key = row.get("search") or f"{row.get('query')}@{int(float(row.get('at', 0)))}"
        by_search.setdefault(key, []).append(row)
    for group in groups:
        entries = by_search[group["search"]]
        seconds = [entry.get("verify_seconds") for entry in entries]
        total = entries[0].get("search_seconds")
        if all(value is not None for value in seconds):
            costs = np.array(seconds, dtype=float)
            if total and costs.sum() > 0:
                # Per-candidate timing covers the first pass; the second pass and boundary refinement
                # are shared out in proportion, so an episode costs what the search really took.
                costs = costs * float(total) / costs.sum()
        elif total:
            costs = np.full(len(entries), float(total) / len(entries))
        else:
            costs = np.full(len(entries), float(default_cost))
        group["costs"] = np.maximum(costs, 1e-3)
        group["measured_costs"] = bool(total) or all(value is not None for value in seconds)
    return groups


def fit_platt_on_score(groups):
    """A calibrated probability from retrieval's own score, when no ranker has been trained."""
    column = FEATURE_NAMES.index("score")
    scores = np.concatenate([group["features"][:, column] for group in groups])
    labels = np.concatenate([group["labels"] for group in groups])
    a, b = 1.0, 0.0
    for _ in range(4000):
        probability = 1.0 / (1.0 + np.exp(-(a * scores + b)))
        error = probability - labels
        a -= 0.5 * float(np.mean(error * scores))
        b -= 0.5 * float(np.mean(error))
    return lambda group: 1.0 / (1.0 + np.exp(-(a * group["features"][:, column] + b)))


def ranker_probability():
    model = load_ranker()
    if not model:
        raise SystemExit("No trained ranker. Train one with python -m bench.rank, or drop --ranker.")
    return lambda group: ranker_probabilities(model, ranker_scores(model, group["features"]))


# ------------------------------------------------------------------ policies
#
# A policy is called once per candidate, in order, with everything a live policy could know at
# that point - never the outcome of the candidate it is deciding about - and returns True to verify
# it or False to stop for good.

def verify_all(context):
    return True


def top_k(k):
    return lambda context: context["position"] < k


def threshold(cut):
    return lambda context: context["probability"] >= cut


def continuation_value(probabilities, costs, value, goal, start):
    """J(start): the expected reward of acting optimally from `start` on, by backward induction.

    J(i) = max(0, p_i * V - c_i + J(i+1))            every match counts
    J(i) = max(0, p_i * V - c_i + (1 - p_i) J(i+1))  only the first does: a hit ends the search
    """
    worth = 0.0
    for index in range(len(probabilities) - 1, start - 1, -1):
        gain = probabilities[index] * value - costs[index]
        carry = (1.0 - probabilities[index]) if goal == "first" else 1.0
        worth = max(0.0, gain + carry * worth)
    return worth


def optimal_stopping(context):
    """Verify the next candidate exactly when continuing optimally is worth more than stopping."""
    if context["goal"] == "first" and context["found"]:
        return False
    return continuation_value(context["probabilities"], context["costs"], context["value"],
                              context["goal"], context["position"]) > 0.0


def state_features(context):
    """What fitted Q sees: a handful of numbers any live policy would have."""
    probabilities, costs = context["probabilities"], context["costs"]
    position, count = context["position"], len(probabilities)
    cost = costs[position]
    return np.array([
        1.0,
        probabilities[position],
        min(10.0, probabilities[position] * context["value"] / max(cost, 1e-3)),
        float(context["found"] > 0),
        position / max(1, count - 1),
        float(probabilities[position:].max()),
        cost / max(float(np.mean(costs)), 1e-3),
        probabilities[position] * float(context["found"] > 0),
    ])


def fitted_q(weights):
    return lambda context: float(state_features(context) @ weights) > 0.0


# ------------------------------------------------------------------- replay

def replay(group, policy, probability_of, value, goal):
    """Run one policy over one logged search, answering every question from the log."""
    probabilities = np.asarray(probability_of(group), dtype=float)
    order = np.argsort(-probabilities, kind="stable")
    probabilities, labels, costs = probabilities[order], group["labels"][order], group["costs"][order]
    found, calls, seconds, reward = 0, 0, 0.0, 0.0
    for position in range(len(order)):
        context = {"position": position, "probability": probabilities[position], "probabilities": probabilities,
                   "costs": costs, "found": found, "value": value, "goal": goal}
        if not policy(context):
            break
        calls += 1
        seconds += costs[position]
        reward -= costs[position]
        if labels[position] > 0:
            if goal == "all" or found == 0:
                reward += value
            found += 1
    total = int(labels.sum())
    return {"reward": reward, "calls": calls, "seconds": seconds, "found": found, "confirmed": total,
            "success": float(found > 0) if total else float("nan"),
            "recall": found / total if total else float("nan"), "candidates": len(order)}


def evaluate(groups, policy, probability_of, value, goal):
    results = [replay(group, policy, probability_of, value, goal) for group in groups]
    mean = lambda key: float(np.nanmean([result[key] for result in results])) if results else float("nan")
    return {"reward": mean("reward"), "calls": mean("calls"), "seconds": mean("seconds"),
            "recall": mean("recall"), "success": mean("success"),
            "calls_fraction": float(np.mean([result["calls"] / max(1, result["candidates"]) for result in results]))}


def choose_threshold(groups, probability_of, value, goal):
    """The static cut that earned the most on the fit searches."""
    cuts = np.unique(np.round(np.concatenate([probability_of(group) for group in groups]), 3))
    best_cut, best_reward = 0.0, -math.inf
    for cut in np.concatenate([[0.0], cuts]):
        reward = evaluate(groups, threshold(float(cut)), probability_of, value, goal)["reward"]
        if reward > best_reward:
            best_cut, best_reward = float(cut), reward
    return best_cut


def train_fitted_q(groups, probability_of, value, goal, iterations=40, ridge=1e-2):
    """Fitted Q iteration on replayed transitions.

    Stopping is terminal and worth nothing, so only Q(state, verify) is learned. Reaching position
    i means every earlier candidate was verified, so walking each logged list end to end yields
    every state any policy could reach - which is what makes the log a complete simulator here.
    """
    transitions = []
    for group in groups:
        probabilities = np.asarray(probability_of(group), dtype=float)
        order = np.argsort(-probabilities, kind="stable")
        probabilities, labels, costs = probabilities[order], group["labels"][order], group["costs"][order]
        found = 0
        for position in range(len(order)):
            context = {"position": position, "probabilities": probabilities, "costs": costs,
                       "found": found, "value": value, "goal": goal}
            credited = labels[position] > 0 and (goal == "all" or found == 0)
            reward = -costs[position] + (value if credited else 0.0)
            found += int(labels[position] > 0)
            following = None
            if position + 1 < len(order):
                following = state_features({**context, "position": position + 1, "found": found})
            transitions.append((state_features(context), reward, following))
    states = np.stack([state for state, _, _ in transitions])
    rewards = np.array([reward for _, reward, _ in transitions])
    weights = np.zeros(states.shape[1])
    for _ in range(iterations):
        targets = rewards.copy()
        for index, (_, _, following) in enumerate(transitions):
            if following is not None:
                targets[index] += max(0.0, float(following @ weights))  # continue only if it pays
        gram = states.T @ states + ridge * np.eye(states.shape[1])
        weights = np.linalg.solve(gram, states.T @ targets)
    return weights


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", default=None)
    parser.add_argument("--goal", choices=("all", "first"), default="all")
    parser.add_argument("--value-seconds", type=float, default=60.0,
                        help="how many seconds of vision calls one confirmed match is worth")
    parser.add_argument("--cost-seconds", type=float, default=20.0,
                        help="assumed cost per candidate for rows logged before timing was recorded")
    parser.add_argument("--ranker", action="store_true", help="use the trained ranker's probabilities")
    parser.add_argument("--min-searches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    with Run("replay", args, seed=args.seed) as run:
        print(f"run {run.id}")
        examples = Path(args.examples) if args.examples else CANDIDATE_EXAMPLES
        run.input(examples)
        rows = load_examples(examples)
        groups = episodes(rows, args.cost_seconds)
        if len(groups) < args.min_searches:
            raise SystemExit(f"Only {len(groups)} logged searches; need {args.min_searches}. "
                             f"Run python -m bench.run --mode verified.")
        measured = sum(group["measured_costs"] for group in groups)
        print(f"{len(groups)} searches, {sum(len(group['labels']) for group in groups)} candidates; "
              f"{measured} with measured costs, the rest at {args.cost_seconds:.0f}s per candidate")

        fit, calibration, test = split_searches(groups, seed=args.seed, fractions=(0.6, 0.0))
        test = calibration + test  # no calibration split needed here; everything unseen is test
        probability_of = ranker_probability() if args.ranker else fit_platt_on_score(fit)
        value, goal = args.value_seconds, args.goal

        cut = choose_threshold(fit, probability_of, value, goal)
        weights = train_fitted_q(fit, probability_of, value, goal)
        policies = {
            "verify all": verify_all,
            "top 1": top_k(1),
            "top 3": top_k(3),
            f"threshold {cut:.2f}": threshold(cut),
            "optimal stopping": optimal_stopping,
            "fitted Q": fitted_q(weights),
        }
        baseline = evaluate(test, verify_all, probability_of, value, goal)
        print(f"\nheld-out: {len(test)} searches   goal: {goal}   a confirmed match is worth {value:.0f}s")
        success = "found any" if goal == "first" else "recall"
        print(f"{'policy':<20}{'reward':>9}{'calls':>8}{'of list':>9}{'seconds':>9}{'saved':>8}{success:>11}")
        results = {}
        for name, policy in policies.items():
            result = evaluate(test, policy, probability_of, value, goal)
            results[name] = result
            saved = 1.0 - result["seconds"] / baseline["seconds"] if baseline["seconds"] else float("nan")
            shown = result["success"] if goal == "first" else result["recall"]
            print(f"{name:<20}{result['reward']:>9.1f}{result['calls']:>8.2f}{result['calls_fraction']:>9.0%}"
                  f"{result['seconds']:>9.1f}{saved:>8.0%}{shown:>11.0%}")
            run.log({"policy": name, **result, "saved": saved})

        best = max(results, key=lambda name: results[name]["reward"])
        print(f"\nbest by reward: {best}")
        gap = results["fitted Q"]["reward"] - results["optimal stopping"]["reward"]
        if gap > 0.5:
            print(f"fitted Q beat optimal stopping by {gap:.1f} reward per search - the probabilities are "
                  f"not the whole story (miscalibrated, or candidates in a search are not independent).")
        elif gap < -0.5:
            print(f"optimal stopping beat fitted Q by {-gap:.1f} - with calibrated probabilities the "
                  f"closed-form policy is hard to improve on, and fitted Q has little data to learn from.")
        else:
            print("fitted Q and optimal stopping agree to within half a second of reward per search.")

        print("\nhow the answer moves with what a match is worth (optimal stopping):")
        sweep = {}
        for worth in (15.0, 30.0, 60.0, 120.0, 300.0):
            result = evaluate(test, optimal_stopping, probability_of, worth, goal)
            everything = evaluate(test, verify_all, probability_of, worth, goal)
            saved = 1.0 - result["seconds"] / everything["seconds"] if everything["seconds"] else float("nan")
            shown = result["success"] if goal == "first" else result["recall"]
            sweep[worth] = {"saved": saved, "kept": shown}
            print(f"   worth {worth:>4.0f}s   verifies {result['calls_fraction']:>4.0%} of candidates, "
                  f"saves {saved:>4.0%} of the time, keeps {shown:>4.0%} of what verify-all finds")
        run.summarize(goal=goal, value_seconds=value, searches=len(groups), test_searches=len(test),
                      threshold=cut, fitted_q_weights=weights.tolist(), results=results, best=best, sweep=sweep)


if __name__ == "__main__":
    main()
