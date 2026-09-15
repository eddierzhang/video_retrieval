"""Detect search: find every clip where something appears by checking frames, not asking about them.

Verified search asks a small vision model whether a handful of frames contain the target and
trusts the confidence it writes, which in practice is 0.95 for right and wrong answers alike. This
executor looks instead:

  1  shots are ranked by how well their per-second frame embeddings match the query, and detection
     effort is spent in that order. The ranking only decides order - a shot is never rejected for
     scoring low, only left unexamined when the frame budget runs out, and results say so.
  2  the shot is the unit: frames are pooled and results bounded per shot, so answers start and end
     on real cuts rather than on thirty-second chunk edges.
  3  an open-vocabulary detector finds every instance of the generic category ("person") in every
     sampled frame, and each crop is scored against the distinguishing description ("a uniformed
     police officer") versus competing ones ("a person in casual clothes"). Detection supplies
     recall; the crop softmax supplies precision.
  4  results can be marked right or wrong. Those marks become visual exemplars that pull similar
     crops up and push similar ones down, and the search is re-scored from its saved state.
  5  detections are linked into tracks within a shot, and tracks into identities across shots by
     their crop embeddings; a track that continues across a boundary joins the clips either side.
  6  motion is scored over short clips by a video-text model, for queries about what happens.
"""
from __future__ import annotations

import math
from pathlib import Path
import pickle
import re

import numpy as np

from . import local_backend
from .config import ACTION_CLIP_FRAMES

DETECT_STAGES = [
    "Planning detection",
    "Indexing shots",
    "Ranking shots",
    "Detecting objects",
    "Scoring motion",
    "Assembling results",
    "Checking results with the vision model",
    "Extracting matching clips",
]

VISION_SCHEMA = {
    "type": "object",
    "properties": {"matches": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["matches", "reason"],
    "additionalProperties": False,
}

# The vision model is asked about at most this many results per search, most confident first.
MAX_VISION_CHECKS = 12

GENERIC_ACTION_CONTRASTS = [
    "people standing still",
    "a still scene with nothing happening",
    "a person talking to the camera",
    "a close-up of an object",
]

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "target": {"type": "string"},
        "contrasts": {"type": "array", "items": {"type": "string"}},
        "min_count": {"type": "integer"},
        "with_object": {"type": "string"},
        "with_count": {"type": "integer"},
        "action": {"type": "string"},
        "action_contrasts": {"type": "array", "items": {"type": "string"}},
        "reads_text": {"type": "boolean"},
    },
    "required": ["object", "target", "contrasts", "min_count", "with_object", "with_count", "action", "action_contrasts",
                 "reads_text"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------ planning

def plan_detection(query):
    """Split a request into what a detector can find and what tells the target apart."""
    prompt = f"""You turn a video search request into instructions for an object detector.

REQUEST: {query}

Fill in:
- object: the plain, generic category a detector can find in a single frame - one or two common
  nouns such as "person", "dog", "cat", "car", "motorcycle", "bus", "tricycle", "egg", "pan",
  "knife", "sign", "lizard". Use "person" for any kind of person (officer, chef, girl, crowd).
  Leave it empty only when the request names no visible thing at all ("when does it get dark").
- target: what distinguishes the requested thing from other members of that category, written as a
  description of a cropped image of ONE of them, e.g. "a uniformed police officer", "a girl with red
  hair", "a person wearing a helmet". Leave it empty when every member counts ("clips with dogs" ->
  object "dog", target "").
- contrasts: 2-4 descriptions of the same category that are clearly NOT what the user wants, for
  comparison, e.g. for "a uniformed police officer": "a person in casual clothes", "a person in a
  business suit". Never list anything the user might plausibly include. Empty if target is empty.
- min_count: how many must be visible at once (1 unless the request asks for a number or a group).
- with_object / with_count: when what distinguishes the target is ANOTHER detectable thing on, in or
  held by it, name that thing and how many are needed, and leave target and contrasts empty.
  "a motorcycle with two riders" -> object "motorcycle", with_object "person", with_count 2.
  "a boy holding a plastic bag" -> object "person", with_object "plastic bag", with_count 1.
  Image-text scoring cannot count, so any number of things belongs here, never in target.
  Otherwise with_object is "" and with_count is 0.
- action: the motion or event the request is about, if any, as a short description of a moving clip
  ("a cat swatting with its paw", "slicing meat with a knife"). Empty for pure appearance requests.
  When the request is about a motion, do NOT also describe that motion as a still pose in target -
  a single frame mid-swat rarely shows it. Only set target alongside an action when the request
  names a separate appearance ("a girl in a red dress running" -> target "a girl in a red dress").
- action_contrasts: 2-4 clearly different motions in a similar setting. Empty if action is empty.
- reads_text: true when the request asks to read, transcribe or identify written text or numbers (a
  sign, a title, a plate, a jersey number); false when text is only part of a description.
"""
    try:
        plan = local_backend.chat_json(prompt, PLAN_SCHEMA, role="planner")
    except Exception as exc:  # the detector can still run on the user's own words
        plan = {"object": query, "target": "", "contrasts": [], "min_count": 1, "with_object": "", "with_count": 0, "action": "",
                "reads_text": bool(TEXT_REQUEST.search(query)),
                "action_contrasts": [], "planning_error": str(exc)}
    return normalize_plan(plan, query)


NOUN_STARTS = ("a ", "an ", "the ", "one ", "two ", "three ", "some ", "several ")

# Used only when the planner is unavailable: requests that plainly ask for text to be read.
TEXT_REQUEST = re.compile(r"^\s*(read|transcribe)\b|what (does|do) .+ say|what is written|\b(number|title|name) (on|of)\b",
                          re.IGNORECASE)


def with_subject(description, category):
    """Put the category in front of a description that does not start with a noun phrase of its own."""
    text = description.strip()
    lowered = text.lower()
    category_words = {word for word in category.lower().split() if len(word) > 2}
    if lowered.startswith(NOUN_STARTS) or any(word in lowered.split() for word in category_words):
        return text
    return f"a {category} {text}"


def normalize_plan(plan, query):
    clean = lambda values: [" ".join(str(value).split()) for value in values or [] if str(value).strip()]
    plan = {
        "object": " ".join(str(plan.get("object", "")).split()),
        "target": " ".join(str(plan.get("target", "")).split()),
        "contrasts": clean(plan.get("contrasts"))[:4],
        "min_count": max(1, min(20, int(plan.get("min_count") or 1))),
        "with_object": " ".join(str(plan.get("with_object", "")).split()),
        "with_count": max(0, min(20, int(plan.get("with_count") or 0))),
        "action": " ".join(str(plan.get("action", "")).split()),
        "action_contrasts": clean(plan.get("action_contrasts"))[:4],
        "reads_text": bool(plan.get("reads_text")),
        **({"planning_error": plan["planning_error"]} if plan.get("planning_error") else {}),
        **({"edited": True} if plan.get("edited") else {}),
    }
    if plan["target"] and plan["object"]:
        # "carrying two riders" scores poorly on its own; it describes a motorcycle, so say so.
        plan["target"] = with_subject(plan["target"], plan["object"])
        plan["contrasts"] = [with_subject(text, plan["object"]) for text in plan["contrasts"]]
    if plan["target"] and not plan["contrasts"]:
        # A softmax over one option is always 1; without contrasts the target cannot be told apart.
        plan["contrasts"] = [f"a {plan['object'] or 'thing'} that does not match: {plan['target']}"]
    if not plan["object"] and not plan["action"]:
        plan["action"] = query
    return plan


# ---------------------------------------------------------- 1, 2: ranking shots

def rank_shots(shots, times, vectors, text_vectors):
    """(shot, priority) best first: the strongest match to any query text in any view of any second."""
    ranked = []
    for shot in shots:
        inside = (times >= shot["start"]) & (times < shot["end"])
        views = vectors[inside] if inside.any() else shot["views"][None]
        similarity = np.einsum("fvd,td->fvt", views, text_vectors)
        ranked.append((shot, float(similarity.max())))
    return sorted(ranked, key=lambda item: -item[1])


def frames_for(shot, detect_fps, min_frames=3, max_fps=8.0):
    """How often to sample a shot: `detect_fps`, but never fewer than `min_frames` in a short shot."""
    duration = max(1e-3, shot["end"] - shot["start"])
    fps = min(max_fps, max(detect_fps, min_frames / duration))
    return fps, max(1, int(math.ceil(duration * fps)))


def budget_shots(ranked, detect_fps, max_frames):
    """Examine shots best first until the frame budget runs out; the rest are unexamined, not rejected."""
    examined, unexamined, used = [], [], 0
    for shot, priority in ranked:
        fps, count = frames_for(shot, detect_fps)
        if examined and used + count > max_frames:
            unexamined.append({"id": shot["id"], "start": shot["start"], "end": shot["end"], "priority": priority})
            continue
        examined.append((shot, priority, fps))
        used += count
    return examined, unexamined, used


# ------------------------------------------------------------ 3: crops

def contrast_probability(similarities, scale):
    """Softmax over [target, contrast...] similarities: the share that belongs to the target."""
    logits = scale * np.asarray(similarities, dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    return float(weights[0] / weights.sum())


def label_matches(label, phrase):
    """Whether a detector label belongs to a prompt phrase; the detector returns the words it grounded."""
    label_words, phrase_words = set(str(label).lower().split()), set(str(phrase).lower().split())
    return bool(label_words & phrase_words)


def dedupe_boxes(boxes, iou=0.6):
    """One box per object: the detector sometimes boxes the same rider twice."""
    kept = []
    for box in boxes:
        if all(box_iou(box, other) < iou for other in kept):
            kept.append(box)
    return kept


def attached_count(box, others, pad=0.15, share=0.5):
    """How many `others` sit on or in `box`: at least `share` of each lies inside the box grown by `pad`."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    grown = (x0 - dx, y0 - dy, x1 + dx, y1 + dy)
    count = 0
    for other in others:
        ox0, oy0, ox1, oy1 = other
        area = max(1e-9, (ox1 - ox0) * (oy1 - oy0))
        inside = max(0.0, min(ox1, grown[2]) - max(ox0, grown[0])) * max(0.0, min(oy1, grown[3]) - max(oy0, grown[1]))
        if inside / area >= share:
            count += 1
    return count


# Boxes smaller than this share of the frame are ignored: at a few hundred pixels, detections on glare
# and distant clutter outnumber real objects.
MIN_BOX_AREA = 0.002

# How far a context crop extends beyond the detector's box, as a fraction of the box's size.
CONTEXT_PAD = 0.45


def crop(frame, box, pad=0.05):
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    left, top = int(max(0, (x0 - dx) * width)), int(max(0, (y0 - dy) * height))
    right, bottom = int(min(width, (x1 + dx) * width)), int(min(height, (y1 + dy) * height))
    if right - left < 4 or bottom - top < 4:
        return None
    return frame[top:bottom, left:right]


# --------------------------------------------------------- 5: tracks, identities

def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_centre_distance(a, b):
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)


def build_tracks(frame_detections, shot_id, iou_threshold=0.2, embedding_threshold=0.85, max_gap=2, max_jump=0.25):
    """Link one shot's detections frame to frame, greedily, by overlap or by looking alike.

    `frame_detections` is [(time, [observation])] in time order; an observation carries "box",
    "score", "probability" and a unit "embedding". Returns tracks with their observations.
    """
    tracks, active = [], []
    for step, (time, observations) in enumerate(frame_detections):
        pairs = []
        for t, track in enumerate(active):
            last = track["observations"][-1]
            for o, observation in enumerate(observations):
                overlap = box_iou(last["box"], observation["box"])
                alike = float(np.dot(last["embedding"], observation["embedding"]))
                # Fast objects barely overlap between samples half a second apart, so something that looks
                # the same and has not moved far is the same object too.
                near = box_centre_distance(last["box"], observation["box"]) <= max_jump
                if overlap >= iou_threshold or (alike >= embedding_threshold and near):
                    pairs.append((overlap + alike, t, o))
        taken_tracks, taken_observations = set(), set()
        for _, t, o in sorted(pairs, reverse=True):
            if t in taken_tracks or o in taken_observations:
                continue
            active[t]["observations"].append({**observations[o], "time": time})
            active[t]["last_step"] = step
            taken_tracks.add(t)
            taken_observations.add(o)
        for o, observation in enumerate(observations):
            if o not in taken_observations:
                track = {"id": len(tracks), "shot": shot_id, "observations": [{**observation, "time": time}],
                         "last_step": step}
                tracks.append(track)
                active.append(track)
        active = [track for track in active if step - track["last_step"] <= max_gap]
    for track in tracks:
        track.pop("last_step", None)
    return tracks


def track_embedding(track):
    vector = np.mean([o["embedding"] for o in track["observations"]], axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def assign_identities(tracks, shots_by_id, threshold=0.88, continuity_iou=0.5, continuity_gap=0.75):
    """Give every track an identity shared with tracks in other shots that show the same thing.

    Two rules join tracks: their average crops look alike, or one ends at a shot boundary and the
    other starts at the next in nearly the same place - a boundary that a passing vehicle made look
    like a cut. Returns the set of (shot, next shot) boundaries that continuity bridged.
    """
    parent = list(range(len(tracks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    embeddings = [track_embedding(track) for track in tracks]
    bridged = set()
    for i, first in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            second = tracks[j]
            if first["shot"] == second["shot"]:
                continue
            if float(np.dot(embeddings[i], embeddings[j])) >= threshold:
                parent[find(i)] = find(j)
            early, late = (first, second) if first["shot"] < second["shot"] else (second, first)
            if late["shot"] == early["shot"] + 1:
                end, begin = early["observations"][-1], late["observations"][0]
                boundary = shots_by_id[early["shot"]]["end"]
                if (boundary - end["time"] <= continuity_gap and begin["time"] - boundary <= continuity_gap
                        and box_iou(end["box"], begin["box"]) >= continuity_iou):
                    parent[find(i)] = find(j)
                    bridged.add((early["shot"], late["shot"]))
    roots = {}
    for i, track in enumerate(tracks):
        track["identity"] = roots.setdefault(find(i), len(roots))
    return bridged


# ------------------------------------------------------------ 4: feedback

def adjusted_probability(probability, embedding, positives, negatives, scale, weight=0.5, baseline=0.0):
    """Move a probability toward crops the user confirmed and away from crops they rejected.

    The shift is added in logit space, proportional to how much closer the crop is to the nearest
    confirmed example than to the nearest rejected one. With only one kind of example, `baseline` -
    the typical similarity to those examples - stands in for the missing side.
    """
    if not len(positives) and not len(negatives):
        return float(probability)
    pull = float(np.max(np.asarray(positives) @ embedding)) if len(positives) else baseline
    push = float(np.max(np.asarray(negatives) @ embedding)) if len(negatives) else baseline
    p = min(max(float(probability), 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p)) + weight * scale * (pull - push)
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))


def feedback_exemplars(state, feedback, memory=None):
    """Embeddings of what was inside results marked right (positives) or wrong (negatives).

    `memory` adds (positives, negatives) remembered from marks on earlier searches for the same thing.
    """
    positives, negatives = [], []
    by_id = {match["match_key"]: match for match in state.get("last_matches", [])}
    for key, label in (feedback or {}).items():
        match = by_id.get(key)
        if not match or label not in ("positive", "negative"):
            continue
        bucket = positives if label == "positive" else negatives
        bucket.extend(match.get("exemplars", []))
    if memory is not None:
        positives.extend(memory[0])
        negatives.extend(memory[1])
    return np.asarray(positives, dtype=np.float32), np.asarray(negatives, dtype=np.float32)


# ------------------------------------------------------------- assembling

def spans(times, step, gap_steps=2.5):
    """Group sorted times into runs, splitting where consecutive samples are further apart than `gap_steps`."""
    runs, current = [], []
    for time in sorted(times):
        if current and time - current[-1] > gap_steps * step:
            runs.append(current)
            current = []
        current.append(time)
    if current:
        runs.append(current)
    return runs


def box_features(observations):
    """Size and placement of a result's boxes, for the learned scorer."""
    boxes = [o["box"] for o in observations]
    areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes]
    edge = [min(x0, y0, 1 - x1, 1 - y1) <= 0.01 for x0, y0, x1, y1 in boxes]
    return {"box_area": float(np.median(areas)), "edge": float(np.mean(edge))}


def assemble(state, attribute_threshold=0.5, action_threshold=2.0, feedback=None, min_run=2, memory=None):
    """Turn the saved observations and clip scores into results, applying any feedback.

    `action_threshold` is a multiple of chance: a clip's action share times the number of descriptions.
    Every result carries `features`, the measurements the learned scorer reads.
    """
    plan, shots_by_id = state["plan"], {shot["id"]: shot for shot in state["shots"]}
    scale = state["scale"]
    priority = state.get("shot_priority", {})
    positives, negatives = feedback_exemplars(state, feedback, memory)
    use_feedback = bool(len(positives) or len(negatives))

    def baseline_for(embeddings):
        vectors = [e for e in embeddings if e is not None]
        if not vectors or not use_feedback:
            return 0.0
        exemplars = positives if len(positives) else negatives
        return float(np.median((np.asarray(vectors) @ exemplars.T).max(axis=1)))

    observation_embeddings = [o["embedding"] for track in state["tracks"] for o in track["observations"]]
    obs_baseline = baseline_for(observation_embeddings)
    window_baseline = baseline_for([w["embedding"] for w in state["windows"]])

    for track in state["tracks"]:
        for o in track["observations"]:
            o["final"] = adjusted_probability(o["probability"], o["embedding"], positives, negatives, scale,
                                              baseline=obs_baseline) if use_feedback else o["probability"]
    for window in state["windows"]:
        window["final"] = adjusted_probability(window["probability"], window["embedding"], positives, negatives,
                                               scale, baseline=window_baseline) if use_feedback else window["probability"]

    # With a motion to score, the appearance description only shades confidence: a still crop mid-swat
    # rarely looks like "a cat with a raised paw", and gating on it discards clips the motion scoring
    # found. A counted relation ("two riders") is a measured fact and always decides.
    gate_on_appearance = bool(plan["target"]) and not plan["action"]
    present = lambda o: o.get("related") is not False and (o["final"] >= attribute_threshold or not gate_on_appearance)

    def object_candidate(shot, start, end, hits, step, action_windows=None):
        """One result from the detections inside [start, end]; None when none fall inside."""
        inside = [(t, o) for t, o in hits if start - step / 2 <= o["time"] <= end + step / 2]
        if not inside:
            return None
        appearance = float(max(np.mean([o["final"] for o in t["observations"]]) for t, _ in inside))
        # When appearance does not decide, it only nudges: a poorly worded pose must not make a
        # clip the motion scoring found look like a 4% guess.
        confidence = appearance if gate_on_appearance or not plan["target"] else 0.5 + 0.5 * appearance
        per_time = {}
        for _, o in inside:
            per_time[o["time"]] = per_time.get(o["time"], 0) + 1
        identities = sorted({t["identity"] for t, _ in inside})
        count = max(per_time.values())
        features = {
            "appearance": appearance, "peak_probability": max(o["final"] for _, o in inside),
            "detector_score": float(np.mean([o["score"] for _, o in inside])),
            "samples": len(per_time), "duration": end - start,
            "shot_share": (end - start) / max(1e-6, shot["end"] - shot["start"]),
            **box_features([o for _, o in inside]), "count": count, "identities": len(identities),
            "track_length": max(len(t["observations"]) for t, _ in inside),
            "has_object": 1, "has_target": int(bool(plan["target"])), "has_action": int(bool(plan["action"])),
            "shot_priority": priority.get(shot["id"], 0.0),
        }
        if action_windows:
            confidence *= strength(action_windows)
            features["action_lift"] = max(lift(w) for w in action_windows)
        return {
            "shot": shot["id"], "start": start, "end": end, "confidence": confidence,
            "identities": identities, "count": count,
            "evidence": max(inside, key=lambda hit: hit[1]["final"]),
            "exemplars": [o["embedding"] for _, o in inside][:12],
            "features": features,
        }

    def window_runs(windows):
        """Overlapping motion windows merged into contiguous runs."""
        merged = []
        for window in sorted(windows, key=lambda w: w["start"]):
            if merged and window["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], window["end"])
                merged[-1]["windows"].append(window)
            else:
                merged.append({"start": window["start"], "end": window["end"], "windows": [window]})
        return merged

    # A clip's action score is a share of probability among every description offered, so what it
    # means depends on how many there were. Compare with chance: with eight descriptions, 0.35 is
    # nearly three times what a random pick would get.
    lift = lambda w: w["final"] * w.get("choices", 1)
    # Strength is measured against the search's own threshold, so looser passes (near misses, missed
    # moments at zero) give confidences on the same scale.
    reference = max(state.get("thresholds", {}).get("action", action_threshold), 1e-6)
    strength = lambda windows: min(1.0, max(lift(w) for w in windows) / (2.0 * reference))
    good = [w for w in state["windows"] if lift(w) >= action_threshold] if plan["action"] else []
    if plan["action"] and not plan["object"] and good:
        best = max(lift(w) for w in good)
        good = [w for w in good if lift(w) >= 0.5 * best]  # with nothing to detect, keep the clear standouts

    candidates = []
    if plan["object"]:
        for shot_id, shot in shots_by_id.items():
            tracks = [t for t in state["tracks"] if t["shot"] == shot_id]
            step = 1.0 / state["shot_fps"].get(shot_id, 2.0)
            qualifying = [t for t in tracks if not gate_on_appearance
                          or np.mean([o["final"] for o in t["observations"]]) >= attribute_threshold]
            if not qualifying:
                continue
            per_time = {}
            for track in qualifying:
                for o in track["observations"]:
                    if present(o):
                        per_time.setdefault(o["time"], []).append((track, o))
            counted = sorted(time for time, hits in per_time.items() if len(hits) >= plan["min_count"])
            sampled = {o["time"] for t in tracks for o in t["observations"]}
            for run in spans(counted, step, gap_steps=1.5):
                # A real object stays in view for consecutive samples; one-sample flickers are glare, edges
                # and clutter. Shots too short to sample twice are exempt.
                if len(run) < min_run and len(sampled) >= min_run:
                    continue
                start = max(shot["start"], run[0] - step / 2)
                end = min(shot["end"], run[-1] + step / 2)
                hits = [hit for time in run for hit in per_time[time]]
                if not plan["action"]:
                    candidates.append(object_candidate(shot, start, end, hits, step))
                    continue
                # With a motion, each stretch where it happens is its own result, built only from the
                # detections inside it - in a long handheld take one track can span several events.
                overlapping = [w for w in good if w["shot"] == shot_id and w["start"] < end and w["end"] > start]
                for motion in window_runs(overlapping):
                    candidate = object_candidate(shot, max(start, motion["start"]), min(end, motion["end"]),
                                                 hits, step, motion["windows"])
                    if candidate:
                        candidates.append(candidate)
    elif plan["action"]:
        for shot_id, shot in shots_by_id.items():
            for run in window_runs([w for w in good if w["shot"] == shot_id]):
                candidates.append({
                    "shot": shot_id, "start": run["start"], "end": run["end"],
                    "confidence": float(strength(run["windows"])),
                    "identities": [], "count": 0, "evidence": None,
                    "exemplars": [w["embedding"] for w in run["windows"]][:12],
                    "features": {
                        "appearance": 0.0, "peak_probability": max(w["final"] for w in run["windows"]),
                        "samples": len(run["windows"]), "duration": run["end"] - run["start"],
                        "shot_share": (run["end"] - run["start"]) / max(1e-6, shot["end"] - shot["start"]),
                        "action_lift": max(lift(w) for w in run["windows"]),
                        "has_object": 0, "has_target": 0, "has_action": 1,
                        "shot_priority": priority.get(shot_id, 0.0),
                    },
                })

    # Join results either side of a boundary that a track was seen to continue across.
    candidates.sort(key=lambda c: c["start"])
    joined = []
    for candidate in candidates:
        previous = joined[-1] if joined else None
        if (previous and (previous["shot"], candidate["shot"]) in state["bridged"]
                and set(previous["identities"]) & set(candidate["identities"])
                and candidate["start"] - previous["end"] <= 1.0):
            previous["end"] = candidate["end"]
            previous["confidence"] = max(previous["confidence"], candidate["confidence"])
            previous["identities"] = sorted(set(previous["identities"]) | set(candidate["identities"]))
            previous["exemplars"] = (previous["exemplars"] + candidate["exemplars"])[:12]
            first, second = previous["features"], candidate["features"]
            previous["features"] = {**{key: max(first.get(key, 0), second.get(key, 0)) for key in {*first, *second}},
                                    "samples": first.get("samples", 0) + second.get("samples", 0),
                                    "duration": previous["end"] - previous["start"],
                                    "identities": len(previous["identities"])}
            continue
        joined.append(candidate)
    for candidate in joined:
        candidate["match_key"] = f"{candidate['shot']}:{candidate['start']:.2f}"
        candidate["features"]["rule_confidence"] = candidate["confidence"]
    return joined


# How far below the hand-set thresholds a result may fall and still be offered to the learned scorer.
NEAR_MISS_ATTRIBUTE = 0.6
NEAR_MISS_ACTION = 0.65


def propose(state, feedback=None, memory=None, scorer=None, checker=None):
    """The results to show: the rules' results, or once a learned scorer beats the rules, its choice.

    With a scorer, results that just missed the thresholds are considered too, so what the scorer has
    learned can recover them as well as reject the rules' false positives. `checker` asks the vision
    model about the results; without a scorer its "no" removes a result, with one it is a measurement.
    """
    thresholds = state["thresholds"]
    strict = assemble(state, thresholds["attribute"], thresholds["action"], feedback=feedback, memory=memory)
    for candidate in strict:
        candidate["features"]["rule_pass"] = 1
        candidate["rule_pass"] = True
        candidate["decided_by"] = "rules"
    if scorer is None:
        if checker is None:
            return strict
        checker(strict)
        return [c for c in strict if (c.get("vision") or {}).get("matches") is not False]
    relaxed = assemble(state, thresholds["attribute"] * NEAR_MISS_ATTRIBUTE, thresholds["action"] * NEAR_MISS_ACTION,
                       feedback=feedback, memory=memory)
    near =[c for c in relaxed if not any(mostly_inside(c, s) or mostly_inside(s, c) for s in strict)]
    for candidate in near:
        candidate["features"]["rule_pass"] = 0
        candidate["rule_pass"] = False
    if checker is not None:
        checker(strict + near)
    kept = []
    for candidate in strict + near:
        probability = scorer.probability(candidate["features"])
        candidate["learned_probability"] = probability
        candidate["decided_by"] = "learned"
        if scorer.keeps(candidate["features"], probability):
            candidate["confidence"] = probability
            kept.append(candidate)
    return sorted(kept, key=lambda c: c["start"])


# ------------------------------------------------------------------- search

def detect_search(query, resources, output_root, max_frames=400, detect_fps=2.0, box_threshold=0.3,
                  attribute_threshold=0.5, action_threshold=2.0, final_frame_fps=1.0, max_frames_per_match=8,
                  plan=None, learner=None, search_id=None, vision_check=True):
    """Answer `query` for one video by detection, tracking and clip scoring; saves state for feedback.

    `learner` (a feedback_learning.Learner) supplies marks remembered from earlier searches and, once it
    beats the hand-set thresholds, the learned scorer that decides which results to keep.
    """
    from .shots import shot_index

    manifest = resources.manifest
    video_path = manifest["video"]["path"]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    local_backend.stage("Planning detection")
    plan = normalize_plan({**plan, "edited": True, "reads_text": False}, query) if plan else plan_detection(query)
    if plan["reads_text"]:
        # Detection cannot read; the caller answers with text reading instead.
        return {"query": query, "route": "text", "plan": plan}

    local_backend.stage("Indexing shots")
    index = shot_index(video_path)
    shots = index["shots"]

    local_backend.stage("Ranking shots")
    ranking_texts = [query] + [text for text in (plan["target"], plan["object"], plan["action"]) if text]
    text_vectors = local_backend.encode(text=ranking_texts)
    ranked = rank_shots(shots, index["times"], index["vectors"], text_vectors)
    examined, unexamined, frames_used = budget_shots(ranked, detect_fps, max_frames)
    scale = local_backend.visual_logit_scale()

    tracks, shot_fps = [], {}
    if plan["object"]:
        local_backend.stage("Detecting objects")
        attribute_vectors = local_backend.encode(text=[plan["target"], *plan["contrasts"]]) if plan["target"] else None
        for shot, _, fps in local_backend.track(examined, "Detecting objects"):
            shot_fps[shot["id"]] = fps
            frames = local_backend.decode_frames(video_path, shot["start"], shot["end"], fps=fps)
            if not frames:
                continue
            related = bool(plan["with_object"] and plan["with_count"])
            phrases = [plan["object"], plan["with_object"]] if related else [plan["object"]]
            detections = local_backend.detect_objects([frame for _, frame in frames], phrases,
                                                      box_threshold=box_threshold)
            per_frame, crops, owners = [], [], []
            for (time, frame), boxes in zip(frames, detections):
                observations = []
                if related:
                    # Counting is done by the detector: image-text scores cannot tell one rider from two.
                    companions = dedupe_boxes([b["box"] for b in sorted(boxes, key=lambda b: -b["score"])
                                               if label_matches(b["label"], plan["with_object"])
                                               and not label_matches(b["label"], plan["object"])])
                    boxes = [dict(b, attached=attached_count(b["box"], companions)) for b in boxes
                             if label_matches(b["label"], plan["object"])]
                for box in boxes:
                    x0, y0, x1, y1 = box["box"]
                    if (x1 - x0) * (y1 - y0) < MIN_BOX_AREA:
                        continue  # too few pixels to recognise, let alone describe
                    tight, wide = crop(frame, box["box"]), crop(frame, box["box"], pad=CONTEXT_PAD)
                    if tight is None:
                        continue
                    observations.append({"box": box["box"], "score": box["score"], "label": box["label"],
                                         "attached": box.get("attached")})
                    crops.extend([tight, wide if wide is not None else tight])
                    owners.append(observations[-1])
                per_frame.append((time, observations))
            if crops:
                embeddings = np.concatenate([local_backend.encode(images=crops[i:i + 32])
                                             for i in range(0, len(crops), 32)])
                for position, observation in enumerate(owners):
                    tight, wide = embeddings[2 * position], embeddings[2 * position + 1]
                    # The tight crop is what identity and feedback compare; the attribute is judged on
                    # whichever view shows it better, since riders or a held object sit outside the box.
                    observation["embedding"] = tight.astype(np.float32)
                    probability = (max(contrast_probability(attribute_vectors @ tight, scale),
                                       contrast_probability(attribute_vectors @ wide, scale))
                                   if attribute_vectors is not None else 1.0)
                    observation["probability"] = probability
                    observation["related"] = (observation["attached"] >= plan["with_count"]) if related else None
            per_frame = [(time, [o for o in observations if "embedding" in o]) for time, observations in per_frame]
            shot_tracks = build_tracks(per_frame, shot["id"])
            for track in shot_tracks:
                track["id"] = len(tracks)
                tracks.append(track)

    windows = []
    if plan["action"]:
        local_backend.stage("Scoring motion")
        texts = [plan["action"], *plan["action_contrasts"], *GENERIC_ACTION_CONTRASTS]
        for shot, _, _ in local_backend.track(examined, "Scoring motion"):
            length = shot["end"] - shot["start"]
            starts = [shot["start"]] if length <= 2.0 else list(np.arange(shot["start"], shot["end"] - 1.0, 1.0))
            clips, spans_ = [], []
            for begin in starts:
                end = min(shot["end"], begin + 2.0)
                frames = local_backend.decode_frames(video_path, begin, end,
                                                     fps=max(1.0, ACTION_CLIP_FRAMES / max(end - begin, 0.25)),
                                                     max_side=320)
                if not frames:
                    continue
                picks = np.linspace(0, len(frames) - 1, ACTION_CLIP_FRAMES).round().astype(int)
                clips.append([frames[i][1] for i in picks])
                spans_.append((float(begin), float(end)))
            if not clips:
                continue
            probabilities = local_backend.action_probabilities(clips, texts)[:, 0]
            inside = lambda begin, end: (index["times"] >= begin) & (index["times"] < end)
            for (begin, end), probability in zip(spans_, probabilities):
                mask = inside(begin, end)
                vector = (index["vectors"][mask][:, 0] if mask.any()
                          else shot["views"][None, 0]).mean(axis=0)
                windows.append({"shot": shot["id"], "start": begin, "end": end, "probability": float(probability),
                                "choices": len(texts),
                                "embedding": (vector / max(float(np.linalg.norm(vector)), 1e-8)).astype(np.float32)})

    local_backend.stage("Assembling results")
    bridged = assign_identities(tracks, {shot["id"]: shot for shot in shots})
    state = {
        "query": query, "plan": plan, "scale": scale, "tracks": tracks, "windows": windows,
        "shots": [{key: shot[key] for key in ("id", "start", "end")} for shot in shots],
        "shot_fps": shot_fps, "bridged": bridged, "thresholds": {"attribute": attribute_threshold, "action": action_threshold},
        "video_duration": float(manifest["video"]["duration"]),
        "shot_priority": {shot["id"]: float(p) for shot, p in ranked},
        "vision_check": bool(vision_check),
    }
    candidates, learning = learned_candidates(state, None, learner, search_id, video_path)
    state["last_matches"] = candidates
    diagnostics = {
        "num_shots": len(shots), "num_examined_shots": len(examined), "frames_examined": frames_used,
        "unexamined_shots": unexamined, "num_tracks": len(tracks),
        "num_identities": len({t["identity"] for t in tracks}), "num_windows": len(windows),
        "bridged_boundaries": sorted(bridged), "shot_ranking": [(shot["id"], round(p, 3)) for shot, p in ranked],
    }
    state["diagnostics"] = diagnostics
    return finish(state, candidates, output_root, video_path, manifest, final_frame_fps, max_frames_per_match,
                  {**diagnostics, "learning": learning})


def learned_candidates(state, feedback, learner, search_id, video_path=None):
    """Propose results with whatever the learner knows, and say how they were decided."""
    memory = learner.memory(state["plan"], exclude_search=search_id) if learner else None
    scorer = learner.scorer() if learner else None
    checker = None
    if state.get("vision_check") and video_path:
        checker = lambda pool: vision_check(state, pool, video_path)
    candidates = propose(state, feedback=feedback, memory=memory, scorer=scorer, checker=checker)
    checks = state.get("vision_cache", {})
    return candidates, {
        "vision_check": bool(checker),
        "vision_rejected": sum(1 for answer in checks.values() if answer and not answer["matches"]),
        "vision_error": state.get("vision_error"),
        "decided_by": "learned" if scorer else "rules",
        "model_version": scorer.version if scorer else None,
        "near_misses_kept": sum(1 for c in candidates if not c.get("rule_pass", True)),
        "remembered": {"right": int(len(memory[0])), "wrong": int(len(memory[1]))} if memory is not None else None,
    }


def finish(state, candidates, output_root, video_path, manifest, final_frame_fps, max_frames_per_match, diagnostics):
    """Cut clips, draw the evidence, save the state for feedback, and shape the result like other searches."""
    from .video import materialize_final_matches

    local_backend.stage("Extracting matching clips")
    instances = [{
        "start": c["start"], "end": c["end"], "confidence": c["confidence"], "match_key": c["match_key"],
        "shot": c["shot"], "identities": c["identities"], "count": c["count"],
        "description": describe(state["plan"], c),
        "decided_by": c.get("decided_by", "rules"), "near_miss": not c.get("rule_pass", True),
        "rule_confidence": c.get("features", {}).get("rule_confidence", c["confidence"]),
        "vision": c.get("vision"),
    } for c in candidates]
    matches, result_file = materialize_final_matches(manifest, instances, state["query"], output_root=str(output_root),
                                                     frame_fps=final_frame_fps, max_frames_per_match=max_frames_per_match)
    by_key = {c["match_key"]: c for c in candidates}
    for match in matches:
        candidate = by_key.get(match.get("match_key"))
        if candidate and candidate["evidence"]:
            image = draw_evidence(video_path, candidate, state, output_root)
            if image:
                match["evidence_image_path"] = str(image)
    with open(output_root / "detect_state.pkl", "wb") as file:
        pickle.dump(state, file)
    return {
        "query": state["query"], "plan": {**state["plan"], "executor": "detect"}, "verified": False, "mode": "detect",
        "num_matches": len(matches), "matches": matches, "results_file": str(result_file),
        "diagnostics": diagnostics,
    }


def describe(plan, candidate):
    subject = plan["target"] or plan["object"] or plan["action"]
    subject = subject[0].upper() + subject[1:] if subject else "Match"
    details = []
    if candidate["count"] > 1:
        details.append(f"{candidate['count']} in view at once")
    if candidate["identities"]:
        labels = ", ".join(f"#{i}" for i in candidate["identities"])
        details.append(f"tracked as {labels}")
    return subject + (f" · {'; '.join(details)}" if details else "")


def evidence_image(video_path, candidate, state):
    """The most confident frame of a result, with every qualifying detection boxed and labelled."""
    from PIL import Image, ImageDraw

    track, observation = candidate["evidence"]
    frames = local_backend.decode_frames(video_path, observation["time"], observation["time"] + 0.05, fps=20)
    if not frames:
        return None
    image = Image.fromarray(frames[0][1])
    draw = ImageDraw.Draw(image)
    width, height = image.size
    threshold = state["thresholds"]["attribute"]
    for other in state["tracks"]:
        if other["shot"] != candidate["shot"]:
            continue
        nearest = min(other["observations"], key=lambda o: abs(o["time"] - observation["time"]))
        if abs(nearest["time"] - observation["time"]) > 0.3 or nearest.get("final", nearest["probability"]) < threshold:
            continue
        x0, y0, x1, y1 = nearest["box"]
        draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline=(255, 200, 0), width=3)
        draw.text((x0 * width + 4, y0 * height + 4),
                  f"#{other['identity']} {nearest.get('final', nearest['probability']):.0%}", fill=(255, 200, 0))
    return image


def draw_evidence(video_path, candidate, state, output_root):
    image = evidence_image(video_path, candidate, state)
    if image is None:
        return None
    folder = output_root / "evidence"
    folder.mkdir(exist_ok=True)
    path = folder / f"match_{candidate['match_key'].replace(':', '_')}.jpg"
    image.save(path, quality=88)
    return path


def clip_frames(video_path, start, end, count=3):
    """`count` frames spread across [start, end], as RGB arrays."""
    frames = []
    for position in range(count):
        time = start + (position + 0.5) * (end - start) / count
        decoded = local_backend.decode_frames(video_path, time, time + 0.05, fps=20, max_side=768)
        if decoded:
            frames.append(decoded[0][1])
    return frames


def vision_key(candidate):
    return f"{candidate['shot']}:{candidate['start']:.2f}:{candidate['end']:.2f}"


def vision_check(state, candidates, video_path, limit=MAX_VISION_CHECKS):
    """Ask the vision model whether each result shows the request; answers are cached in the state.

    The model sees the boxed evidence frame and frames spread across the clip. Its answer is stored
    on the result as `vision` and as the measurements `vision_checked` and `vision_yes`.
    """
    cache = state.setdefault("vision_cache", {})
    ranked = sorted(candidates, key=lambda c: -c["features"].get("rule_confidence", c["confidence"]))
    pending = [c for c in ranked if vision_key(c) not in cache][:limit]
    if pending and not state.get("vision_error"):
        local_backend.stage("Checking results with the vision model")
        for candidate in local_backend.track(pending, "Checking results"):
            images = []
            if candidate.get("evidence"):
                boxed = evidence_image(video_path, candidate, state)
                if boxed is not None:
                    images.append(np.asarray(boxed))
            images.extend(clip_frames(video_path, candidate["start"], candidate["end"]))
            if not images:
                continue
            boxes = " The first image marks with yellow boxes what an object detector found." if candidate.get("evidence") else ""
            prompt = (f'Someone searched a video for: "{state["query"]}".\n'
                      f"These images come from one candidate clip, {candidate['end'] - candidate['start']:.1f} seconds long, "
                      f"in time order.{boxes}\n"
                      "Does this clip show what they searched for? Judge strictly and check every part of the request: "
                      "the kind of thing, how it looks, how many there are, and what is happening. If a required part "
                      "cannot be seen, it does not match. Give matches and a one-sentence reason.")
            try:
                # image_base64 expects OpenCV's channel order.
                answer = local_backend.images_json([image[:, :, ::-1] for image in images], prompt, VISION_SCHEMA)
            except RuntimeError as exc:
                state["vision_error"] = str(exc)  # results stay as they were before the check
                break
            cache[vision_key(candidate)] = {"matches": bool(answer["matches"]), "reason": str(answer["reason"])[:300]}
    for candidate in candidates:
        answer = cache.get(vision_key(candidate))
        candidate["features"]["vision_checked"] = int(answer is not None)
        candidate["features"]["vision_yes"] = int(bool(answer and answer["matches"]))
        if answer is not None:
            candidate["vision"] = answer


def overlap_seconds(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def missed_candidate(output_root, start, end):
    """Turn a stretch the user says was missed into a result, and an example for the learner.

    The saved detections are re-assembled with every threshold at zero; the result overlapping the
    stretch most supplies its measurements and evidence. When nothing was ever detected there, the
    result has no measurements - the detector never saw it, so there is nothing to learn from.
    Returns (candidate, example or None); the candidate is saved into the search's state.
    """
    from .feedback_learning import concept_key

    path = Path(output_root) / "detect_state.pkl"
    with open(path, "rb") as file:
        state = pickle.load(file)
    span = {"start": float(start), "end": float(end)}
    middle = (span["start"] + span["end"]) / 2
    shot = next((s for s in state["shots"] if s["start"] <= middle < s["end"]), state["shots"][-1])
    loose = assemble(state, 0.0, 0.0, min_run=1)
    best = max(loose, key=lambda c: overlap_seconds(c, span), default=None)
    if best is not None and overlap_seconds(best, span) <= 0:
        best = None
    returned = [c for c in state.get("last_matches", []) if not str(c["match_key"]).startswith("missed:")]
    key = f"missed:{span['start']:.2f}"
    evidence = None
    if best is not None and best.get("evidence") and span["start"] <= best["evidence"][1]["time"] <= span["end"]:
        evidence = best["evidence"]
    candidate = {
        "shot": shot["id"], "start": span["start"], "end": span["end"], "confidence": 1.0, "match_key": key,
        "identities": best["identities"] if best else [], "count": best["count"] if best else 0,
        "evidence": evidence, "exemplars": best["exemplars"] if best else [],
        "rule_pass": any(mostly_inside({**span, "shot": shot["id"]}, c) for c in returned),
        "decided_by": "you", "detected": best is not None,
    }
    example = None
    if best is not None:
        features = {**best["features"], "duration": span["end"] - span["start"], "rule_pass": int(candidate["rule_pass"])}
        candidate["features"] = features
        exemplars = np.asarray(best["exemplars"] or [], dtype=np.float32)
        embedding = None
        if len(exemplars):
            mean = exemplars.mean(axis=0)
            embedding = (mean / max(float(np.linalg.norm(mean)), 1e-8)).astype(np.float16)
        example = {"features": features, "embedding": embedding, "concept": concept_key(state["plan"]),
                   "query": state["query"], "start": span["start"], "end": span["end"], "missed": True}
    else:
        candidate["features"] = {"duration": span["end"] - span["start"], "rule_pass": 0}
    state["last_matches"] = [c for c in state.get("last_matches", []) if c["match_key"] != key] + [candidate]
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as file:
        pickle.dump(state, file)
    temporary.replace(path)
    return candidate, example


def mostly_inside(candidate, other, share=0.5):
    """Whether at least `share` of `candidate` lies within `other`, in the same shot."""
    if candidate["shot"] != other["shot"]:
        return False
    overlap = max(0.0, min(candidate["end"], other["end"]) - max(candidate["start"], other["start"]))
    return overlap >= share * max(1e-6, candidate["end"] - candidate["start"])


def example_for(output_root, match_key):
    """What the learner stores for a mark on one result: its measurements and what was inside it."""
    try:
        with open(Path(output_root) / "detect_state.pkl", "rb") as file:
            state = pickle.load(file)
    except Exception:  # a missing or unreadable state only means this mark cannot be learned from
        return None
    from .feedback_learning import concept_key

    match = next((c for c in state.get("last_matches", []) if c["match_key"] == match_key), None)
    if match is None or "features" not in match:
        return None  # a search saved before results carried measurements
    exemplars = np.asarray(match.get("exemplars") or [], dtype=np.float32)
    embedding = None
    if len(exemplars):
        mean = exemplars.mean(axis=0)
        embedding = (mean / max(float(np.linalg.norm(mean)), 1e-8)).astype(np.float16)
    return {"features": {**match["features"], "rule_pass": int(match.get("rule_pass", True))},
            "embedding": embedding, "concept": concept_key(state["plan"]), "query": state["query"],
            "start": match["start"], "end": match["end"]}


def refine(output_root, feedback, resources, final_frame_fps=1.0, max_frames_per_match=8, learner=None, search_id=None):
    """Re-score a saved Detect search with the user's marks, without running the detector again."""
    output_root = Path(output_root)
    with open(output_root / "detect_state.pkl", "rb") as file:
        state = pickle.load(file)
    candidates, learning = learned_candidates(state, feedback, learner, search_id, resources.manifest["video"]["path"])
    # Marked results keep their verdict whatever the re-scoring says about their neighbours.
    previous = {c["match_key"]: c for c in state["last_matches"]}
    keys = {c["match_key"] for c in candidates}
    for key, label in (feedback or {}).items():
        if label == "positive" and key in previous and key not in keys:
            candidates.append(previous[key])
    # Re-scoring can move a result's start, and with it its key, so a rejected clip is recognised by overlap.
    rejected = [previous[key] for key, label in (feedback or {}).items() if label == "negative" and key in previous]
    candidates = [c for c in candidates if (feedback or {}).get(c["match_key"]) != "negative"
                  and not any(mostly_inside(c, r) for r in rejected)]
    candidates.sort(key=lambda c: c["start"])
    state["last_matches"] = candidates + [c for key, c in previous.items() if key not in {x["match_key"] for x in candidates}]
    diagnostics = {**state.get("diagnostics", {}), "learning": learning,
                   "refined_with": {label: sum(1 for v in (feedback or {}).values() if v == label)
                                    for label in ("positive", "negative")}}
    manifest = resources.manifest
    return finish(state, candidates, output_root, manifest["video"]["path"], manifest, final_frame_fps,
                  max_frames_per_match, diagnostics)
