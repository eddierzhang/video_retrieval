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

import numpy as np

from .config import DATA_DIR

LEARNING_DIR = DATA_DIR / "learning"
CANDIDATE_EXAMPLES = LEARNING_DIR / "candidates.jsonl"
PREFILTER_MODEL = LEARNING_DIR / "candidate_prefilter.json"
ADAPTER_MODEL = LEARNING_DIR / "query_adapter.npz"

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

def log_candidates(candidates, evidence_map, plan, video_duration, query, survivors):
    """Record which candidates actually yielded a confirmed match, for later training."""
    if not candidates:
        return
    survivors = {int(x) for x in survivors if x is not None}
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATE_EXAMPLES, "a", encoding="utf-8") as file:
        for rank, candidate in enumerate(candidates):
            features = candidate_features(candidate, rank, candidates, evidence_map, plan, video_duration)
            file.write(json.dumps({
                "at": time.time(),
                "query": query,
                "executor": (plan or {}).get("executor"),
                "label": int(candidate.get("candidate_id") in survivors),
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
