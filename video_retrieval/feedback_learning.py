"""Learning from right/wrong marks on Detect results, across every search and video.

Each mark is kept as a training example: the result's measurements (how confident the detector
was, how well crops matched the description, how long the object stayed in view, how far motion
rose above chance...) and the average embedding of what was inside it. Two things are learned:

  scorer  a logistic regression over those measurements that predicts whether a result is right.
          The measurements mean the same for every query, so marks on "a fluffy dog" teach it what
          glare, edge slivers and one-sample flickers look like on "police officers" too. It is
          retrained after every mark, evaluated on videos it was not trained on, and only replaces
          the hand-set thresholds once it beats them there.
  memory  the marked embeddings themselves, reused as examples by later searches for the same thing.

Nothing here touches the large models' weights; with hundreds rather than thousands of labels, a
small model over their outputs is what can be learned without memorising the videos.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
import threading
import time

import numpy as np

# The measurements the scorer reads, in order. Every one is defined for any query.
FEATURES = [
    "rule_confidence",     # the confidence the hand-set rules gave
    "rule_pass",           # 1 when the hand-set thresholds kept it, 0 for a near miss
    "appearance",          # best track's mean crop probability for the description
    "peak_probability",    # highest single crop probability inside the result
    "detector_score",      # mean detector confidence of its boxes
    "samples",             # how many sampled frames it spans (log)
    "duration",            # seconds (log)
    "shot_share",          # share of its shot that it covers
    "box_area",            # median box size as a share of the frame (log)
    "edge",                # share of boxes touching the frame border
    "count",               # most objects counted at once (log)
    "identities",          # distinct tracked things inside it (log)
    "track_length",        # longest track's observations (log)
    "action_lift",         # motion score as a multiple of chance (0 without motion)
    "has_object",
    "has_target",
    "has_action",
    "shot_priority",       # how well the shot matched the query before detection
    "vision_checked",      # 1 when the vision model was asked about it
    "vision_yes",          # 1 when the vision model said it shows the request
]
LOG_FEATURES = {"samples", "duration", "box_area", "count", "identities", "track_length"}

MIN_EXAMPLES = 40          # below this a model over 20 measurements is guessing
MIN_PER_CLASS = 10         # it needs to have seen both right and wrong results
MIN_VIDEOS = 2             # evaluation holds out whole videos
MARGIN = 0.02              # how much better than the rules it must be on held-out videos
NEAR_MISS_BAR = 0.7        # near misses need more confidence until enough of them are labelled
MIN_NEAR_MISS_LABELS = 10
MEMORY_PER_LABEL = 50      # most recent examples per label reused as memory for a description


def vectorize(features):
    values = []
    for name in FEATURES:
        value = float(features.get(name, 0.0) or 0.0)
        if name in LOG_FEATURES:
            value = math.log(max(value, 0.0) + (1e-3 if name == "box_area" else 1.0))
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def fit_logistic(X, y, l2=1.0, iterations=30):
    """L2-regularised logistic regression by Newton's method on standardised features."""
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0          # a constant measurement carries no information; leave it at zero weight
    Z = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
    penalty = np.full(Z.shape[1], l2)
    penalty[-1] = 0.0              # the bias is not shrunk
    w = np.zeros(Z.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        gradient = Z.T @ (p - y) + penalty * w
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mean": mean.tolist(), "std": std.tolist(), "weights": w[:-1].tolist(), "bias": float(w[-1])}


def predict(model, X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    z = (X - np.asarray(model["mean"])) / np.asarray(model["std"]) @ np.asarray(model["weights"]) + model["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def evaluate(examples, l2=1.0):
    """Leave one video out: train on the rest, predict its marks, and compare with the rules' decisions."""
    videos = sorted({e["video_id"] for e in examples})
    learned_right = rules_right = evaluated = 0
    log_loss = 0.0
    for video in videos:
        train = [e for e in examples if e["video_id"] != video]
        test = [e for e in examples if e["video_id"] == video]
        labels = {e["label"] for e in train}
        if len(labels) < 2:
            continue
        model = fit_logistic([vectorize(e["features"]) for e in train], [e["label"] for e in train], l2=l2)
        probabilities = predict(model, [vectorize(e["features"]) for e in test])
        for example, p in zip(test, probabilities):
            label = example["label"]
            learned_right += int((p >= 0.5) == bool(label))
            rules_right += int(bool(example["features"].get("rule_pass", 1)) == bool(label))
            log_loss -= math.log(max(p if label else 1 - p, 1e-6))
            evaluated += 1
    if not evaluated:
        return None
    return {"evaluated": evaluated, "learned_accuracy": learned_right / evaluated,
            "rules_accuracy": rules_right / evaluated, "log_loss": log_loss / evaluated}


class Scorer:
    """The trained model as searches use it."""

    def __init__(self, model, near_miss_labels):
        self.model = model
        self.version = model["version"]
        self.near_miss_bar = 0.5 if near_miss_labels >= MIN_NEAR_MISS_LABELS else NEAR_MISS_BAR

    def probability(self, features):
        return float(predict(self.model, vectorize(features))[0])

    def keeps(self, features, probability):
        return probability >= (0.5 if features.get("rule_pass", 1) else self.near_miss_bar)


class Learner:
    """The examples, the current model and its history, kept in one folder and shared by all searches."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.examples = self._load_examples()
        self.model = self._load_json("model.json")
        self.history = self._load_json("history.json") or []
        if self.model and self.model.get("features") and self.model["features"] != FEATURES:
            # Saved by a version that measured different things; relearn from the same marks.
            self._retrain()
            self._save()

    # ------------------------------------------------------------ storage

    def _load_examples(self):
        try:
            with open(self.folder / "examples.pkl", "rb") as file:
                return pickle.load(file)
        except (OSError, pickle.UnpicklingError, EOFError):
            return {}

    def _load_json(self, name):
        try:
            return json.loads((self.folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save(self):
        temporary = self.folder / "examples.pkl.tmp"
        with open(temporary, "wb") as file:
            pickle.dump(self.examples, file)
        temporary.replace(self.folder / "examples.pkl")
        for name, value in (("model.json", self.model), ("history.json", self.history)):
            if value is not None:
                (self.folder / name).write_text(json.dumps(value, indent=1), encoding="utf-8")

    # ----------------------------------------------------------- learning

    def record(self, video_id, search_id, match_key, label, example=None):
        """Add, change or (label None) remove one mark, then retrain. Returns the new status."""
        key = f"{video_id}/{search_id}/{match_key}"
        with self._lock:
            if label is None or example is None:
                self.examples.pop(key, None)
            else:
                self.examples[key] = {**example, "video_id": video_id, "search_id": search_id,
                                      "match_key": match_key, "label": int(label == "positive"),
                                      "marked_at": self.examples.get(key, {}).get("marked_at", time.time())}
            self._retrain()
            self._save()
            return self._status()

    def _retrain(self):
        examples = list(self.examples.values())
        positives = sum(e["label"] for e in examples)
        counts = {"examples": len(examples), "right": positives, "wrong": len(examples) - positives,
                  "videos": len({e["video_id"] for e in examples}),
                  "near_misses": sum(1 for e in examples if not e["features"].get("rule_pass", 1))}
        previous = self.model or {}
        if (counts["examples"] < MIN_EXAMPLES or min(counts["right"], counts["wrong"]) < MIN_PER_CLASS
                or counts["videos"] < MIN_VIDEOS):
            self.model = {**counts, "version": previous.get("version", 0), "active": False, "state": "collecting",
                          "trained_at": None, "evaluation": None}
            return
        evaluation = evaluate(examples)
        fitted = fit_logistic([vectorize(e["features"]) for e in examples], [e["label"] for e in examples])
        active = bool(evaluation and evaluation["learned_accuracy"] >= evaluation["rules_accuracy"] + MARGIN)
        self.model = {**counts, **fitted, "features": FEATURES, "version": previous.get("version", 0) + 1,
                      "active": active, "state": "active" if active else "not_better",
                      "trained_at": time.time(), "evaluation": evaluation}
        self.history = (self.history + [{"version": self.model["version"], "trained_at": self.model["trained_at"],
                                         "examples": counts["examples"], "active": active,
                                         **(evaluation or {})}])[-200:]

    # -------------------------------------------------------------- using

    def scorer(self):
        with self._lock:
            model = self.model
            if not model or not model.get("active"):
                return None
            return Scorer(model, model.get("near_misses", 0))

    def memory(self, plan, exclude_search=None):
        """Embeddings of marked results for the same description in other searches: (positives, negatives)."""
        concept = concept_key(plan)
        with self._lock:
            matching = sorted((e for e in self.examples.values()
                               if e.get("concept") == concept and e["search_id"] != exclude_search
                               and e.get("embedding") is not None),
                              key=lambda e: -e["marked_at"])
        pick = lambda label: [np.asarray(e["embedding"], dtype=np.float32)
                              for e in matching if e["label"] == label][:MEMORY_PER_LABEL]
        return np.asarray(pick(1), dtype=np.float32), np.asarray(pick(0), dtype=np.float32)

    def status(self):
        with self._lock:
            return self._status()

    def _status(self):
        model = self.model or {"examples": 0, "right": 0, "wrong": 0, "videos": 0, "near_misses": 0,
                               "version": 0, "active": False, "state": "collecting", "evaluation": None}
        keys = ("examples", "right", "wrong", "videos", "near_misses", "version", "active", "state",
                "trained_at", "evaluation")
        return {**{key: model.get(key) for key in keys},
                "needs": {"examples": MIN_EXAMPLES, "per_label": MIN_PER_CLASS, "videos": MIN_VIDEOS},
                "history": self.history[-50:]}


def concept_key(plan):
    """What a mark is about: the searched-for description, ignoring case and spacing."""
    parts = [plan.get("object", ""), plan.get("target", ""), plan.get("with_object", ""),
             str(plan.get("with_count") or ""), plan.get("action", "")]
    return " | ".join(" ".join(str(part).lower().split()) for part in parts)
