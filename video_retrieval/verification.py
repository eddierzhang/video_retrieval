#VLM Verification of candidate frames after retrieval
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import requests
from tqdm import tqdm

from .config import FLASH_VERIFIER_MODEL, OPENROUTER_CHAT_URL, PRO_VERIFIER_MODEL, get_openrouter_api_key
from .video import materialize_vlm_clip, video_to_data_url

#Takes info from a planned query and converts to input for verification VLM
def _event_definition_text(plan):
    plan = plan or {}
    definition = plan.get("event_definition", {})
    counts = definition.get("counts_as_match", [])
    rejects = definition.get("does_not_count", [])
    predicates = plan.get("evidence_predicates", [])
    negatives = plan.get("negative_evidence", [])
    temporal_constraints = plan.get("temporal_constraints", [])

    lines = []
    if counts:
        lines.append("COUNT as a valid match when:\n- " + "\n- ".join(counts))
    if rejects:
        lines.append("DO NOT count:\n- " + "\n- ".join(rejects))

    required = [p["description"] for p in predicates if p.get("required")]
    supporting = [p["description"] for p in predicates if not p.get("required")]
    if required:
        lines.append("REQUIRED EVIDENCE to check for:\n- " + "\n- ".join(required))
    if supporting:
        lines.append("SUPPORTING/CUE EVIDENCE (helpful but not sufficient alone):\n- " + "\n- ".join(supporting))
    if temporal_constraints:
        lines.append("TEMPORAL/STATE CONSTRAINTS:\n- " + "\n- ".join(temporal_constraints))
    if negatives:
        lines.append("NEGATIVE/CONFOUNDING EVIDENCE that should reject or lower confidence:\n- " + "\n- ".join(negatives))

    return "\n\n".join(lines)

#Split candidate into multiple verification windows
def _video_windows(start, end, window=75.0, overlap=12.0):
    start = float(start)
    end = float(end)
    if end <= start:
        return []

    duration = end - start
    if duration <= window:
        return [(start, end)]

    stride = max(0.5, window - overlap)
    output = []
    t = start

    while t < end:
        e = min(t + window, end)
        output.append((t, e))
        if e >= end:
            break
        t += stride

    return output

#Parse response 
def _parse_openrouter_json_content(content):
    if isinstance(content, str):
        return json.loads(content)

    # Defensive handling for providers that return content-part arrays.
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        if text_parts:
            return json.loads("".join(text_parts))

    raise RuntimeError(f"Unexpected OpenRouter response content: {content!r}")

#Call an OpenRouter video model and returns a strict JSON-schema response.
def call_video_json(
    video_path,
    prompt,
    schema,
    model,
    schema_name="video_result",
    timeout=300,
):
    prompt = (
        prompt
        + "\n\nReturn the response as valid JSON only, matching the required schema."
    )
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video_url",
                    "video_url": {"url": video_to_data_url(video_path)},
                },
            ],
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
        "provider": {"require_parameters": True},
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {get_openrouter_api_key()}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter video call failed: HTTP {response.status_code}\n"
            f"{response.text}"
        )

    content = response.json()["choices"][0]["message"]["content"]
    return _parse_openrouter_json_content(content)

#Defines output format for verifier
MULTI_INSTANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "instances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "confidence": {"type": "number"},
                    "description": {"type": "string"},
                    "actor_description": {"type": "string"},
                    "visual_evidence": {"type": "string"},
                },
                "required": [
                    "start_seconds",
                    "end_seconds",
                    "confidence",
                    "description",
                    "actor_description",
                    "visual_evidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["present", "instances"],
    "additionalProperties": False,
}

#Calculates temporal IoU between two event
def temporal_iou(a, b):
    intersection = max(
        0.0,
        min(float(a["end"]), float(b["end"]))
        - max(float(a["start"]), float(b["start"])),
    )
    union = (
        max(float(a["end"]), float(b["end"]))
        - min(float(a["start"]), float(b["start"]))
    )
    return 0.0 if union <= 0 else intersection / union

#Compares two actors to determine whether they refer to the same person 
def _actor_similarity(a, b):
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 1.0

    generic = {
        "person", "people", "man", "woman", "male", "female",
        "adult", "individual", "someone", "the", "a", "an", "in",
        "wearing", "with", "on", "near", "standing", "visible",
    }

    def signature(text):
        tokens = re.findall(r"[a-z0-9]+", text)
        return {token for token in tokens if token not in generic}

    sa = signature(a)
    sb = signature(b)

    if sa and sb:
        return len(sa & sb) / len(sa | sb)

    # Generic descriptions such as "person" contain no identity information.
    return SequenceMatcher(None, a, b).ratio()

#Remove duplicate detections created by overlapping verifier windows.
def deduplicate_instances(instances, iou_threshold=0.65):
    ordered = sorted(
        instances,
        key=lambda x: float(x.get("confidence", 0.0)),
        reverse=True,
    )
    keep = []

    for candidate in ordered:
        duplicate = False
        for existing in keep:
            if temporal_iou(candidate, existing) < iou_threshold:
                continue

            actor_sim = _actor_similarity(
                candidate.get("actor_description", ""),
                existing.get("actor_description", ""),
            )

            # Similar or missing actor descriptions => likely same occurrence.
            if actor_sim >= 0.45:
                duplicate = True
                break

        if not duplicate:
            keep.append(candidate)

    return sorted(keep, key=lambda x: x["start"])

    # Detect ZERO, ONE, or MULTIPLE occurrences inside one candidate region.

    # Returned timestamps are absolute timestamps in the original source video.
def verify_candidate(
    manifest,
    candidate,
    query,
    plan=None,
    model=FLASH_VERIFIER_MODEL,
    verifier_window_seconds=75.0,
    verifier_overlap_seconds=12.0,
    min_confidence=0.25,
    vlm_cache_dir="vlm_clip_cache",
):

    definition_text = _event_definition_text(plan)
    all_instances = []

    windows = _video_windows(
        candidate["start"],
        candidate["end"],
        window=verifier_window_seconds,
        overlap=verifier_overlap_seconds,
    )

    for window_start, window_end in windows:
        clip_path = materialize_vlm_clip(
            manifest,
            window_start,
            window_end,
            output_dir=vlm_cache_dir,
        )
        duration = window_end - window_start

        prompt = f"""
You are verifying candidate footage for a long-video temporal retrieval system.

TARGET EVENT:
{query}

{definition_text}

This clip is {duration:.3f} seconds long.

Find EVERY DISTINCT occurrence of the target event visible in THIS clip.
There may be zero, one, or multiple occurrences.

Rules:
- Do not merge separate occurrences into one long interval.
- If two different people perform the target action at nearly the same time,
  list them as separate instances and distinguish them in actor_description.
- start_seconds/end_seconds are relative to the BEGINNING OF THIS CLIP.
- Use the smallest interval that contains the complete requested event.
- Confidence is from 0 to 1.
- If the event is absent, return present=false and instances=[].
- Do not count merely related context unless it satisfies the target event.
"""

        result = call_video_json(
            clip_path,
            prompt,
            MULTI_INSTANCE_SCHEMA,
            model=model,
            schema_name="multi_instance_verification",
        )

        for item in result.get("instances", []):
            confidence = float(item.get("confidence", 0.0))
            if confidence < min_confidence:
                continue

            rel_start = max(0.0, min(duration, float(item["start_seconds"])))
            rel_end = max(0.0, min(duration, float(item["end_seconds"])))
            if rel_end <= rel_start:
                continue

            all_instances.append({
                "start": window_start + rel_start,
                "end": window_start + rel_end,
                "confidence": confidence,
                "description": item.get("description", ""),
                "actor_description": item.get("actor_description", ""),
                "visual_evidence": item.get("visual_evidence", ""),
                "source_candidate_id": candidate.get("candidate_id"),
                "retrieval_score": float(candidate.get("score", 0.0)),
                "verification_model": model,
            })

    return deduplicate_instances(all_instances, iou_threshold=0.65)

#Run exhaustive Flash verification over candidate regions
def verify_candidates_flash(
    manifest,
    candidates,
    query,
    plan=None,
    max_candidates=None,
    **verify_kwargs,
):

    selected = candidates if max_candidates is None else candidates[:max_candidates]
    instances = []

    for candidate in tqdm(selected, desc="Flash candidate verification"):
        found = verify_candidate(
            manifest,
            candidate,
            query,
            plan=plan,
            **verify_kwargs,
        )
        instances.extend(found)

    return deduplicate_instances(instances, iou_threshold=0.65)

#Second stage verifier
PRO_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "confidence": {"type": "number"},
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "description": {"type": "string"},
        "actor_description": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "valid",
        "confidence",
        "start_seconds",
        "end_seconds",
        "description",
        "actor_description",
        "reason",
    ],
    "additionalProperties": False,
}

#verify one Flash-proposed occurrence with Gemini Pro
def verify_instance_pro(
    manifest,
    instance,
    query,
    plan=None,
    model=PRO_VERIFIER_MODEL,
    padding=5.0,
    min_confidence=0.50,
    vlm_cache_dir="vlm_clip_cache",
):

    video_duration = float(manifest["video"]["duration"])
    clip_start = max(0.0, float(instance["start"]) - padding)
    clip_end = min(video_duration, float(instance["end"]) + padding)

    clip_path = materialize_vlm_clip(
        manifest,
        clip_start,
        clip_end,
        output_dir=vlm_cache_dir,
    )
    clip_duration = clip_end - clip_start
    proposed_start = float(instance["start"]) - clip_start
    proposed_end = float(instance["end"]) - clip_start

    prompt = f"""
You are the final strict verifier for a temporal video retrieval system.

TARGET EVENT:
{query}

{_event_definition_text(plan)}

A faster model proposed one occurrence at approximately
{proposed_start:.3f} to {proposed_end:.3f} seconds in this clip.

Verify THAT proposed occurrence. Do not switch to an unrelated occurrence
elsewhere in the clip.

Return valid=true only if the complete requested event is actually visible.
If valid, give your best adjusted start/end offsets relative to this clip.
Use the tightest interval containing the complete event.
If invalid, set start_seconds=-1 and end_seconds=-1.
"""

    result = call_video_json(
        clip_path,
        prompt,
        PRO_VERIFY_SCHEMA,
        model=model,
        schema_name="strict_instance_verification",
    )

    confidence = float(result.get("confidence", 0.0))
    if not result.get("valid", False) or confidence < min_confidence:
        return None

    rel_start = float(result.get("start_seconds", -1))
    rel_end = float(result.get("end_seconds", -1))
    if rel_start < 0 or rel_end <= rel_start:
        return None

    rel_start = max(0.0, min(clip_duration, rel_start))
    rel_end = max(0.0, min(clip_duration, rel_end))
    if rel_end <= rel_start:
        return None

    verified = instance.copy()
    verified.update({
        "start": clip_start + rel_start,
        "end": clip_start + rel_end,
        "confidence": confidence,
        "description": result.get("description", instance.get("description", "")),
        "actor_description": result.get(
            "actor_description",
            instance.get("actor_description", ""),
        ),
        "pro_reason": result.get("reason", ""),
        "pro_verified": True,
        "verification_model": model,
    })
    return verified

#Verify all remaining candidates with Pro 
def verify_instances_pro(
    manifest,
    instances,
    query,
    plan=None,
    **kwargs,
):
    verified = []

    for instance in tqdm(instances, desc="Gemini Pro verification"):
        result = verify_instance_pro(
            manifest,
            instance,
            query,
            plan=plan,
            **kwargs,
        )
        if result is not None:
            verified.append(result)

    return deduplicate_instances(verified, iou_threshold=0.70)

#Defines output for video boundary refinement
BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "event_present": {"type": "boolean"},
        "confidence": {"type": "number"},
        "contains_boundary": {"type": "boolean"},
        "boundary_seconds": {"type": "number"},
        "state_before": {"type": "string"},
        "state_after": {"type": "string"},
    },
    "required": [
        "event_present",
        "confidence",
        "contains_boundary",
        "boundary_seconds",
        "state_before",
        "state_after",
    ],
    "additionalProperties": False,
}

#Creates two short overlapping clips around current boundary estimate to locate real boundary
def _boundary_probe_windows(estimate, window_size, video_duration):
    """Two overlapping windows around the current boundary estimate."""
    w = float(window_size)
    starts = [estimate - w, estimate - w / 2]
    output = []

    for start in starts:
        start = max(0.0, start)
        end = min(float(video_duration), start + w)
        if end - start >= min(0.75, w):
            output.append((start, end))

    # Deduplicate near-identical clipped windows.
    unique = []
    for item in output:
        if not any(abs(item[0] - x[0]) < 1e-6 and abs(item[1] - x[1]) < 1e-6 for x in unique):
            unique.append(item)
    return unique

#Refines one boundary by identifying which clip contains exact start transition
def refine_boundary(
    manifest,
    query,
    boundary_estimate,
    boundary_type,
    plan=None,
    model=FLASH_VERIFIER_MODEL,
    stages=(8.0, 4.0, 2.0),
    min_confidence=0.45,
    vlm_cache_dir="vlm_clip_cache",
):
    """
    Refine either the START or END boundary using progressively smaller clips.

    Each stage asks a simple local question: does this short clip contain the
    requested boundary? If yes, the model gives the boundary offset inside the
    short clip. Because the clip becomes very short, timestamp regression is
    much easier than asking for a timestamp in the full long video.
    """

    if boundary_type not in {"start", "end"}:
        raise ValueError("boundary_type must be 'start' or 'end'")

    estimate = float(boundary_estimate)
    video_duration = float(manifest["video"]["duration"])
    history = []

    boundary_word = "START" if boundary_type == "start" else "END"

    for window_size in stages:
        probes = _boundary_probe_windows(estimate, window_size, video_duration)
        stage_results = []

        for probe_start, probe_end in probes:
            clip_path = materialize_vlm_clip(
                manifest,
                probe_start,
                probe_end,
                output_dir=vlm_cache_dir,
                fps=4,
                width=768,
                crf=25,
            )
            duration = probe_end - probe_start

            prompt = f"""
TARGET EVENT:
{query}

{_event_definition_text(plan)}

You are refining the exact {boundary_word} boundary of one known occurrence.
This clip is only {duration:.3f} seconds long.

Determine whether THIS clip contains the exact {boundary_word} transition of
that occurrence.

For START: the boundary is the moment the requested event changes from not yet
happening to happening.
For END: the boundary is the moment the requested event changes from happening
to completed/no longer happening.

If the requested {boundary_word} boundary is visible, set contains_boundary=true
and boundary_seconds to its offset from the beginning of this short clip.
If it is not visible, set contains_boundary=false and boundary_seconds=-1.
Do not use timestamps from outside this clip.
"""

            result = call_video_json(
                clip_path,
                prompt,
                BOUNDARY_SCHEMA,
                model=model,
                schema_name=f"{boundary_type}_boundary_refinement",
            )

            confidence = float(result.get("confidence", 0.0))
            rel = float(result.get("boundary_seconds", -1))

            record = {
                "window_start": probe_start,
                "window_end": probe_end,
                "window_size": window_size,
                "confidence": confidence,
                "contains_boundary": bool(result.get("contains_boundary", False)),
                "boundary_seconds": rel,
                "state_before": result.get("state_before", ""),
                "state_after": result.get("state_after", ""),
            }
            stage_results.append(record)
            history.append(record)

        valid = [
            r for r in stage_results
            if r["contains_boundary"]
            and r["confidence"] >= min_confidence
            and 0 <= r["boundary_seconds"] <= (r["window_end"] - r["window_start"])
        ]

        if valid:
            # Highest-confidence local boundary estimate wins this stage.
            best = max(valid, key=lambda x: x["confidence"])
            estimate = best["window_start"] + best["boundary_seconds"]

    return estimate, history

#Refines both start and end boundary
def refine_instance(
    manifest,
    instance,
    query,
    plan=None,
    stages=(8.0, 4.0, 2.0),
    model=FLASH_VERIFIER_MODEL,
):
    """Refine both boundaries independently for one verified occurrence."""

    start, start_history = refine_boundary(
        manifest,
        query,
        instance["start"],
        "start",
        plan=plan,
        model=model,
        stages=stages,
    )

    end, end_history = refine_boundary(
        manifest,
        query,
        instance["end"],
        "end",
        plan=plan,
        model=model,
        stages=stages,
    )

    # Defensive fallback if independent estimates cross.
    if end <= start:
        start = float(instance["start"])
        end = float(instance["end"])

    refined = instance.copy()
    refined.update({
        "start": start,
        "end": end,
        "boundary_refined": True,
        "start_refinement_history": start_history,
        "end_refinement_history": end_history,
    })
    return refined

#Refine instance over all confirmed events 
def refine_instances(
    manifest,
    instances,
    query,
    plan=None,
    **kwargs,
):
    output = []
    for instance in tqdm(instances, desc="Boundary refinement"):
        output.append(
            refine_instance(
                manifest,
                instance,
                query,
                plan=plan,
                **kwargs,
            )
        )
    return output

#Detects and removes duplicates in overlapping windows
def temporal_nms(
    detections,
    iou_threshold=0.55,
    preserve_distinct_actors=True,
):
    """
    Deduplicate detections produced by overlapping candidate windows.

    If actor descriptions are clearly different, highly-overlapping events are
    preserved. This matters when two people perform the same action at nearly
    the same time.
    """

    ordered = sorted(
        detections,
        key=lambda x: float(x.get("confidence", 0.0)),
        reverse=True,
    )
    keep = []

    for detection in ordered:
        suppress = False

        for existing in keep:
            if temporal_iou(detection, existing) < iou_threshold:
                continue

            if preserve_distinct_actors:
                actor_a = detection.get("actor_description", "")
                actor_b = existing.get("actor_description", "")
                if actor_a and actor_b and _actor_similarity(actor_a, actor_b) < 0.40:
                    continue

            suppress = True
            break

        if not suppress:
            keep.append(detection)

    return sorted(keep, key=lambda x: x["start"])
