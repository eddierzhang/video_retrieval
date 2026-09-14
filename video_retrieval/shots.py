"""Shot boundaries, and the frame embeddings pooled per shot.

Edited video is made of shots, and people mean shots when they ask for "every clip with X": the
eggs frying is one three-second shot, not a stretch of a thirty-second chunk. Detect search uses
the shot as its unit, so every answer starts and ends on a real cut.

A cut is found where consecutive frames change abruptly in both colour distribution and pixels,
relative to how much this video usually changes around that moment - a handheld shot of traffic
moves constantly without ever cutting, and a fixed absolute threshold would split it apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import local_backend
from .config import CACHE_DIR

SHOT_FPS = 12.0


def frame_signatures(frames):
    """A colour histogram and a tiny grey image per RGB frame, for comparing neighbours."""
    import cv2

    histograms, greys = [], []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        histograms.append(histogram / max(float(histogram.sum()), 1.0))
        greys.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (64, 36), interpolation=cv2.INTER_AREA))
    return np.asarray(histograms, dtype=np.float32), np.asarray(greys, dtype=np.float32)


def change_scores(histograms, greys):
    """How different each frame is from the one before it, from 0 to 1. The first frame scores 0."""
    scores = np.zeros(len(histograms), dtype=np.float32)
    if len(histograms) < 2:
        return scores
    colour = 1.0 - np.minimum(histograms[1:], histograms[:-1]).sum(axis=1)
    pixels = np.abs(greys[1:] - greys[:-1]).mean(axis=(1, 2)) / 255.0
    scores[1:] = np.clip(0.6 * colour + 0.4 * np.clip(pixels * 3.0, 0.0, 1.0), 0.0, 1.0)
    return scores


def find_cuts(scores, times, threshold=0.3, ratio=3.0, min_shot=0.5, fps=SHOT_FPS):
    """Times where a frame starts a new shot: a local peak that is large on its own and next to its surroundings."""
    reach = max(1, int(round(fps)))
    cuts, last = [], float(times[0]) if len(times) else 0.0
    for index in range(1, len(scores)):
        score = float(scores[index])
        if score < threshold:
            continue
        neighbourhood = np.concatenate([scores[max(1, index - reach):index], scores[index + 1:index + 1 + reach]])
        typical = float(np.median(neighbourhood)) if len(neighbourhood) else 0.0
        if score < ratio * max(typical, 0.02):
            continue  # constant motion, not a cut
        if score < float(scores[max(1, index - 2):index + 3].max()):
            continue  # a neighbour is the actual peak of this transition
        if float(times[index]) - last < min_shot:
            continue
        cuts.append(float(times[index]))
        last = float(times[index])
    return cuts


def changed_fraction(before, after, grid=4, level=0.15):
    """The share of a grid of regions that changed between two small grey frames, and the median change."""
    height, width = before.shape
    changes = []
    for row in range(grid):
        for col in range(grid):
            a = before[row * height // grid:(row + 1) * height // grid, col * width // grid:(col + 1) * width // grid]
            b = after[row * height // grid:(row + 1) * height // grid, col * width // grid:(col + 1) * width // grid]
            changes.append(abs(float(a.mean()) - float(b.mean())) / 255.0 + float(np.abs(a - b).mean()) / 255.0)
    changes = np.asarray(changes)
    return float((changes > level).mean()), float(np.median(changes))


def is_real_cut(before, after):
    """False for something passing the camera: part of the frame changes while the rest stays put.

    A cut replaces the whole picture. A motorcycle leaving the foreground changes a third of it.
    Transitions through near-black frames are kept - both sides are dark, so few regions differ,
    but the change detector only fired because the picture did change. A vehicle that fills most
    of the frame still passes this test; tracks that continue across the boundary are joined
    later, so a false cut can split an answer but not lose one.
    """
    fraction, median = changed_fraction(before, after)
    dark = max(float(before.mean()), float(after.mean())) < 40.0
    return dark or fraction >= 0.5 or median >= 0.1


def detect_shots(video_path, fps=SHOT_FPS):
    """[(start, end)] covering the whole video, cached by file identity."""
    key = local_backend._file_key(video_path, "shots-v2", fps)
    cache = CACHE_DIR / "shots" / f"{key}.json"
    if cache.is_file():
        return [tuple(shot) for shot in json.loads(cache.read_text(encoding="utf-8"))]
    from .video import probe_video

    duration = float(probe_video(video_path)["duration"])
    decoded = local_backend.decode_frames(video_path, 0.0, duration, fps=fps, max_side=160)
    if not decoded:
        return [(0.0, duration)]
    times = np.asarray([time for time, _ in decoded], dtype=np.float32)
    histograms, greys = frame_signatures([frame for _, frame in decoded])
    cuts = []
    for cut in find_cuts(change_scores(histograms, greys), times, fps=fps):
        index = int(np.searchsorted(times, cut))
        # Compare a frame just before the transition with one just after, skipping the frame it lands on.
        before, after = greys[max(0, index - 2)], greys[min(len(greys) - 1, index + 1)]
        if is_real_cut(before, after):
            cuts.append(cut)
    edges = [0.0, *cuts, duration]
    shots = [(round(edges[i], 3), round(edges[i + 1], 3)) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(shots), encoding="utf-8")
    return shots


def shot_index(video_path):
    """Every shot with its per-view embedding pooled over the frames inside it.

    Returns {"shots": [{"id", "start", "end", "views": (views, dim)}], "times", "vectors"} so later
    stages can also read the per-second embeddings each shot was pooled from.
    """
    shots = detect_shots(video_path)
    times, vectors = local_backend.frame_embeddings(video_path)
    rows = []
    for number, (start, end) in enumerate(shots):
        inside = (times >= start) & (times < end)
        if inside.any():
            views = vectors[inside].mean(axis=0)
        else:  # a shot shorter than the one-second embedding interval borrows its nearest frame
            views = vectors[int(np.argmin(np.abs(times - (start + end) / 2)))]
        views = views / np.clip(np.linalg.norm(views, axis=-1, keepdims=True), 1e-8, None)
        rows.append({"id": number, "start": float(start), "end": float(end), "views": views.astype(np.float32)})
    return {"shots": rows, "times": times, "vectors": vectors}


def save_keyframes(video_path, shots, target_dir, max_side=320):
    """One middle frame per shot, for inspecting what the shot index found."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    paths = []
    for shot in shots:
        middle = (shot["start"] + shot["end"]) / 2
        frames = local_backend.decode_frames(video_path, middle, middle + 0.2, fps=5, max_side=max_side)
        if frames:
            path = target_dir / f"shot_{shot['id']:03d}.jpg"
            Image.fromarray(frames[0][1]).save(path, quality=85)
            paths.append(path)
    return paths
