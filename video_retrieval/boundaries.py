"""Learned clip boundaries, from signals the index already holds.

Retrieval finds roughly where something happens and pads it: a quick-mode answer is a region of
the evidence map with ten seconds added either side, which is most of why its IoU is low. The
per-second frame embeddings already say more than that. Two curves come out of them for free:

  similarity   how well each second matches the query, by its best view
  cut          how different each second looks from the one before

A boundary is a position where the first curve steps up (a start) or down (an end), often where
the second one spikes. This scores every second near a proposed boundary with a small linear
model over features of both curves and moves the boundary to the best one - a softmax over
positions, trained on intervals whose truth is known (bench.boundaries).

Nothing here calls a model except the text embedder, so it costs milliseconds, not the vision
calls the verifier's own refinement spends. Without a trained model every function is a no-op.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .config import DATA_DIR

BOUNDARY_MODEL = DATA_DIR / "learning" / "boundary_model.json"

FEATURES = (
    "inside",        # mean similarity over the seconds the boundary would include
    "outside",       # mean similarity over the seconds it would exclude
    "edge",          # inside - outside: the step a boundary should sit on
    "similarity",    # similarity at the boundary second itself
    "body",          # mean similarity across the whole proposed interval from this boundary
    "cut",           # visual change at the boundary
    "cut_near",      # largest visual change within a second of it
    "distance",      # how far the boundary moved, as a fraction of the search window
)
CUT_FEATURES = ("cut", "cut_near")


def signals(times, vectors, query_vector):
    """Per-second similarity to the query, and per-second visual change, from frame embeddings.

    Similarity is standardised within the video so a feature means the same thing whether the
    query is an easy match or a weak one.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 2:
        vectors = vectors[:, None, :]
    query = np.asarray(query_vector, dtype=np.float32)
    query = query / max(float(np.linalg.norm(query)), 1e-8)
    similarity = (vectors @ query).max(axis=1)
    spread = float(similarity.std())
    similarity = (similarity - float(similarity.mean())) / (spread if spread > 1e-6 else 1.0)

    whole = vectors[:, 0, :]
    whole = whole / np.clip(np.linalg.norm(whole, axis=1, keepdims=True), 1e-8, None)
    cut = np.zeros(len(whole), dtype=np.float32)
    if len(whole) > 1:
        cut[1:] = 1.0 - np.sum(whole[1:] * whole[:-1], axis=1)
    times = np.asarray(times, dtype=np.float64)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    return {"times": times, "step": step if step > 0 else 1.0, "similarity": similarity, "cut": cut}


def _mean(values, start, end):
    start, end = max(0, start), min(len(values), end)
    return float(values[start:end].mean()) if end > start else 0.0


def frame_index(signal, seconds):
    return int(round((float(seconds) - float(signal["times"][0])) / signal["step"])) if len(signal["times"]) else 0


def frame_time(signal, index):
    return float(signal["times"][0]) + index * signal["step"] if len(signal["times"]) else 0.0


def positions(signal, proposal, window):
    """Frame indices a boundary may move to, within `window` seconds of where retrieval put it."""
    count = len(signal["similarity"])
    centre = frame_index(signal, proposal)
    reach = max(1, int(round(window / signal["step"])))
    return list(range(max(0, centre - reach), min(count, centre + reach) + 1))


def boundary_features(signal, position, side, proposal, other, window, context):
    """Features for putting the `side` boundary at `position`, given the other boundary."""
    similarity, cut = signal["similarity"], signal["cut"]
    other = frame_index(signal, other)
    if side == "start":
        inside = _mean(similarity, position, position + context)
        outside = _mean(similarity, position - context, position)
        body = _mean(similarity, position, other)
    else:
        inside = _mean(similarity, position - context, position)
        outside = _mean(similarity, position, position + context)
        body = _mean(similarity, other, position)
    at = min(max(position, 0), len(similarity) - 1)
    near = cut[max(0, position - 1):min(len(cut), position + 2)]
    return [
        inside,
        outside,
        inside - outside,
        float(similarity[at]),
        body,
        float(cut[at]) if position < len(cut) else 0.0,
        float(near.max()) if len(near) else 0.0,
        abs(frame_time(signal, position) - float(proposal)) / max(1.0, float(window)),
    ]


def candidate_matrix(signal, side, proposal, other, window, context):
    spots = positions(signal, proposal, window)
    matrix = np.array([boundary_features(signal, spot, side, proposal, other, window, context) for spot in spots],
                      dtype=np.float64)
    return spots, matrix


def score(model, matrix):
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    mask = np.asarray(model.get("mask") or [1.0] * len(FEATURES), dtype=np.float64)
    scaled = (matrix - mean) / np.where(std > 0, std, 1.0) * mask
    return scaled @ np.asarray(model["weights"], dtype=np.float64)


def refine_interval(signal, start, end, model):
    """Move both boundaries to the best-scoring nearby seconds. Never returns an empty interval."""
    window, context = int(model.get("window", 8)), int(model.get("context", 4))
    new_start, new_end = float(start), float(end)
    spots, matrix = candidate_matrix(signal, "start", start, end, window, context)
    if spots:
        new_start = frame_time(signal, spots[int(np.argmax(score(model, matrix)))])
    spots, matrix = candidate_matrix(signal, "end", end, new_start, window, context)
    if spots:
        new_end = frame_time(signal, spots[int(np.argmax(score(model, matrix)))])
    if new_end - new_start < 1.0:
        return float(start), float(end)  # the two boundaries crossed; keep what retrieval said
    return new_start, new_end


@lru_cache(maxsize=2)
def _load(path, stamp):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_boundary_model(path=BOUNDARY_MODEL):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return _load(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError):
        return None


def refine_instances(manifest, query, instances):
    """Refine every instance's boundaries in place of the padded ones retrieval produced."""
    model = load_boundary_model()
    if not model or not instances:
        return instances, None
    from . import local_backend

    try:
        times, vectors = local_backend.frame_embeddings(manifest["video"]["path"])
        signal = signals(times, vectors, local_backend.embed_text(query))
    except Exception as exc:  # a missing cache or model must never cost the search its answer
        return instances, {"model_trained_at": model.get("trained_at"), "error": str(exc)}
    duration = float(manifest["video"]["duration"])
    refined = []
    for instance in instances:
        start, end = refine_interval(signal, float(instance["start"]), float(instance["end"]), model)
        refined.append({**instance, "start": max(0.0, start), "end": min(duration, end),
                        "unrefined": [float(instance["start"]), float(instance["end"])],
                        "boundaries": "learned"})
    return refined, {"model_trained_at": model.get("trained_at"), "refined": len(refined),
                     "uses_cuts": uses_cuts(model)}


def uses_cuts(model):
    mask = model.get("mask") or [1.0] * len(FEATURES)
    return all(mask[FEATURES.index(name)] > 0 for name in CUT_FEATURES)
