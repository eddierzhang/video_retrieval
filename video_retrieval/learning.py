"""Optional learned components, trained from the pipeline's own output.

Two models live here, and both are absent until trained:

  candidate pre-filter  predicts whether a candidate is worth a vision call, learned
                        from the verifier's own accept/reject decisions
  query adapter         a linear map on visual query embeddings, learned from the
                        scene descriptions and transcript already in each index

Neither needs hand-labeled data, and when no model file exists every function here is
a no-op, so the pipeline behaves exactly as it did before. Train them with
`python -m bench.distill` and `python -m bench.adapt`.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .config import DATA_DIR

LEARNING_DIR = DATA_DIR / "learning"
CANDIDATE_EXAMPLES = LEARNING_DIR / "candidates.jsonl"
PREFILTER_MODEL = LEARNING_DIR / "candidate_prefilter.json"
RANKER_MODEL = LEARNING_DIR / "candidate_ranker.json"
ADAPTER_MODEL = LEARNING_DIR / "query_adapter.npz"

# How often a candidate the ranker rejected is verified anyway, so that the next generation
# is trained on more than its own opinions. See `select_candidates`.
EXPLORE_FRACTION = 0.10

FEATURE_NAMES = (
    "score",
    "relative_score",
    "evidence_mass",
    "supporting_bins",
    "duration",
    "duration_ratio",
    "rank",
    "channel_video",
    "channel_metadata",
    "channel_speech",
    "channel_negative",
    "ordering",
)


def candidate_features(candidate, rank, candidates, evidence_map, plan, video_duration):
    """Signals already computed by retrieval, shaped into a fixed feature vector."""
    start, end = float(candidate["start"]), float(candidate["end"])
    duration = max(0.0, end - start)
    best = max((float(row.get("score", 0.0)) for row in candidates), default=0.0)
    expected = (plan or {}).get("expected_duration") or {}
    expected_max = max(1.0, float(expected.get("max_seconds") or 0.0) or 1.0)

    inside = [row for row in evidence_map if float(row["end"]) > start and float(row["start"]) < end]
    def channel(name):
        values = [float((row.get("channel_scores") or {}).get(name, 0.0)) for row in inside]
        return max(values) if values else 0.0

    speech = max(channel("transcript_semantic"), channel("transcript_bm25"))
    return {
        "score": float(candidate.get("score", 0.0)),
        "relative_score": float(candidate.get("score", 0.0)) / best if best > 0 else 0.0,
        "evidence_mass": float(candidate.get("evidence_mass", 0.0)),
        "supporting_bins": float(candidate.get("num_supporting_bins") or candidate.get("recursive_support_bins") or 0),
        "duration": duration,
        "duration_ratio": duration / expected_max,
        "rank": float(rank) / max(1, len(candidates) - 1),
        "channel_video": channel("video"),
        "channel_metadata": channel("metadata"),
        "channel_speech": speech,
        "channel_negative": channel("negative"),
        "ordering": float(candidate.get("ordering_satisfied", 0)) - float(candidate.get("ordering_violated", 0)),
    }


def vectorize(features):
    return np.array([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float64)


# ------------------------------------------------------------ collecting

def log_candidates(candidates, evidence_map, plan, video_duration, query, survivors, search_seconds=None):
    """Record which candidates actually yielded a confirmed match, for later training.

    Every row of one search shares a `search` id, because ranking is a per-search problem:
    a listwise loss needs to know which candidates competed against each other, and the
    train/holdout split has to keep a whole search on one side.
    """
    if not candidates:
        return
    survivors = {int(x) for x in survivors if x is not None}
    search = uuid.uuid4().hex[:12]
    at = time.time()
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATE_EXAMPLES, "a", encoding="utf-8") as file:
        for rank, candidate in enumerate(candidates):
            features = candidate_features(candidate, rank, candidates, evidence_map, plan, video_duration)
            file.write(json.dumps({
                "at": at,
                "search": search,
                "query": query,
                "executor": (plan or {}).get("executor"),
                "label": int(candidate.get("candidate_id") in survivors),
                # Explored rows were sampled, not chosen, so training has to correct for the
                # probability that brought them here rather than treat them as ordinary rows.
                "explored": bool(candidate.get("explored", False)),
                "propensity": float(candidate.get("propensity", 1.0)),
                # What verifying this candidate cost, and the whole search's verification, so a
                # replay can charge a policy for the calls it makes (see bench.replay).
                "verify_seconds": candidate.get("verify_seconds"),
                "search_seconds": search_seconds,
                "features": features,
            }) + "\n")


def load_examples(path=CANDIDATE_EXAMPLES):
    rows = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


# -------------------------------------------------------------- pre-filter

@lru_cache(maxsize=4)
def _load_json_model(path, stamp):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prefilter():
    """The trained candidate pre-filter, or None when it has not been trained."""
    path = Path(PREFILTER_MODEL)
    if not path.is_file():
        return None
    try:
        return _load_json_model(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError):
        return None


def prefilter_probability(model, features):
    vector = vectorize(features)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    z = float(np.dot((vector - mean) / np.where(std > 0, std, 1.0), weights) + model["bias"])
    return 1.0 / (1.0 + np.exp(-z))


def prefilter_candidates(candidates, evidence_map, plan, video_duration, keep_min=5):
    """Drop candidates the student model thinks the vision model would reject.

    The top `keep_min` by predicted probability are always kept, so a badly fitted
    model can slow things down but cannot empty the candidate list.
    """
    model = load_prefilter()
    if not model or len(candidates) <= keep_min:
        return candidates, None
    scored = []
    for rank, candidate in enumerate(candidates):
        features = candidate_features(candidate, rank, candidates, evidence_map, plan, video_duration)
        scored.append((prefilter_probability(model, features), candidate))
    threshold = float(model.get("threshold", 0.0))
    ordered = sorted(scored, key=lambda pair: pair[0], reverse=True)
    kept = [candidate for probability, candidate in ordered if probability >= threshold]
    if len(kept) < keep_min:
        kept = [candidate for _, candidate in ordered[:keep_min]]
    kept.sort(key=lambda candidate: float(candidate.get("score", 0.0)), reverse=True)
    return kept, {
        "model_trained_at": model.get("trained_at"),
        "threshold": threshold,
        "before": len(candidates),
        "after": len(kept),
        "skipped": len(candidates) - len(kept),
    }


# ------------------------------------------------- ranker and conformal set

def load_ranker():
    """The trained candidate ranker, or None when it has not been trained."""
    path = Path(RANKER_MODEL)
    if not path.is_file():
        return None
    try:
        return _load_json_model(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError):
        return None


def ranker_scores(model, matrix):
    """Forward pass of the small ranking network, in numpy so search needs no torch."""
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    values = (np.asarray(matrix, dtype=np.float64) - mean) / np.where(std > 0, std, 1.0)
    if model.get("mask") is not None:
        # An ablated model was fitted with some features held at zero; keep them there.
        values = values * np.asarray(model["mask"], dtype=np.float64)
    layers = model["layers"]
    for index, layer in enumerate(layers):
        values = values @ np.asarray(layer["w"], dtype=np.float64) + np.asarray(layer["b"], dtype=np.float64)
        if index < len(layers) - 1:
            values = np.maximum(values, 0.0)
    return values[:, 0]


def ranker_probabilities(model, scores):
    """Platt-scaled scores. Conformal validity does not depend on these being calibrated."""
    platt = model.get("platt") or {}
    a = float(platt.get("a", 1.0))
    b = float(platt.get("b", 0.0))
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(scores, dtype=np.float64) + b)))


def select_candidates(candidates, evidence_map, plan, video_duration, keep_min=3,
                      explore=EXPLORE_FRACTION, rng=None):
    """Order candidates by the learned ranker and keep a conformal prediction set.

    The threshold comes from split conformal calibration: on held-out searches, the set it
    produces contained a confirmed candidate at least `1 - alpha` of the time. That is a
    statement about coverage across searches, not about any single candidate being right.

    A fraction of the *rejected* candidates is verified anyway. Without that, the only rows
    this system ever collects again are candidates it already approved of: it would learn
    that candidates like X are worthless, stop sending them to the vision model, and never
    see evidence that it was wrong. Exploration costs a little time per search and is the
    only thing keeping the next generation's training data honest. Explored rows are marked,
    and carry the probability with which they were sampled so training can correct for it.

    Falls back to the pointwise pre-filter, and then to doing nothing at all, so an untrained
    system behaves exactly as it did before.
    """
    model = load_ranker()
    if not model:
        return prefilter_candidates(candidates, evidence_map, plan, video_duration)
    if len(candidates) <= keep_min:
        return candidates, None

    matrix = np.stack([
        vectorize(candidate_features(candidate, rank, candidates, evidence_map, plan, video_duration))
        for rank, candidate in enumerate(candidates)
    ])
    scores = ranker_scores(model, matrix)
    probabilities = ranker_probabilities(model, scores)
    conformal = model.get("conformal") or {}
    threshold = float(conformal.get("threshold", 0.0))

    order = [int(index) for index in np.argsort(-scores)]
    kept = [index for index in order if probabilities[index] >= threshold]
    if len(kept) < keep_min:
        kept = order[:keep_min]  # the floor: a bad model can cost time, never empty the list
    chosen = set(kept)
    generator = rng if rng is not None else np.random
    explore = max(0.0, min(1.0, float(explore)))
    explored = [index for index in order
                if index not in chosen and generator.uniform() < explore]

    selected = []
    for position, index in enumerate(kept + explored):  # explored go last: they are extra work
        candidate = dict(candidates[index])
        candidate["ranker_score"] = float(scores[index])
        candidate["ranker_probability"] = float(probabilities[index])
        candidate["ranker_position"] = position
        candidate["explored"] = index not in chosen
        candidate["propensity"] = explore if index not in chosen else 1.0
        selected.append(candidate)
    return selected, {
        "model": "ranker",
        "model_trained_at": model.get("trained_at"),
        "loss": model.get("loss"),
        "threshold": threshold,
        "coverage_target": conformal.get("coverage"),
        "before": len(candidates),
        "after": len(selected),
        "skipped": len(candidates) - len(selected),
        "explored": len(explored),
        "reordered": kept != sorted(kept),
    }


# ----------------------------------------------------------- query adapter

@lru_cache(maxsize=2)
def _load_adapter(path, stamp):
    with np.load(path) as data:
        return data["weights"].astype(np.float32)


def load_adapter():
    """The trained query adapter matrix, or None when it has not been trained."""
    path = Path(ADAPTER_MODEL)
    if not path.is_file():
        return None
    try:
        return _load_adapter(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError):
        return None


def apply_query_adapter(vector):
    """Map a visual query embedding into the space the indexed frames actually occupy."""
    weights = load_adapter()
    if weights is None or weights.shape[0] != vector.shape[-1]:
        return vector
    adapted = np.asarray(vector, dtype=np.float32) @ weights
    return (adapted / max(float(np.linalg.norm(adapted)), 1e-8)).astype(np.float32)
