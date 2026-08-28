#Handles visual text/OCR retrieval pipeline by scanning video for text-bearing targets
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests

from .config import OPENROUTER_CHAT_URL, OCR_MODEL, get_openrouter_api_key
from .verification import call_video_json
from .video import format_timestamp_precise, materialize_final_matches, materialize_vlm_clip

#JSON format of whole video scan 
OCR_VIDEO_SCAN_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "best_frame_seconds": {"type": "number"},
                    "ocr_text": {"type": "string"},
                    "readable": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "target_description": {"type": "string"},
                    "visual_evidence": {"type": "string"},
                },
                "required": [
                    "start_seconds",
                    "end_seconds",
                    "best_frame_seconds",
                    "ocr_text",
                    "readable",
                    "confidence",
                    "target_description",
                    "visual_evidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

#Structured output for full-resolution frame refinement 
OCR_FRAME_REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "readings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_index": {"type": "integer"},
                    "target_present": {"type": "boolean"},
                    "ocr_text": {"type": "string"},
                    "readable": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "target_description": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                            "x2": {"type": "number"},
                            "y2": {"type": "number"},
                        },
                        "required": ["x1", "y1", "x2", "y2"],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "image_index",
                    "target_present",
                    "ocr_text",
                    "readable",
                    "confidence",
                    "target_description",
                    "bbox",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["readings"],
    "additionalProperties": False,
}

#Output when model is only give cropped text regions 
OCR_CROP_SCHEMA = {
    "type": "object",
    "properties": {
        "readings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_index": {"type": "integer"},
                    "ocr_text": {"type": "string"},
                    "readable": {"type": "boolean"},
                    "confidence": {"type": "number"},
                },
                "required": ["image_index", "ocr_text", "readable", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["readings"],
    "additionalProperties": False,
}

#Normalize OCR output for display without discarding useful punctuation
def normalize_ocr_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\x00", " ").strip()
    return re.sub(r"\s+", " ", text)

#Aggressive normalization used only for OCR similarity/deduplication
def _comparison_key(text: str | None) -> str:
    text = normalize_ocr_text(text).upper()
    return "".join(ch for ch in text if ch.isalnum())

#Measures how similar two OCR readings are
def _text_similarity(a: str | None, b: str | None) -> float:
    a_key = _comparison_key(a)
    b_key = _comparison_key(b)
    if not a_key or not b_key:
        return 0.0
    if min(len(a_key), len(b_key)) <= 2:
        return 1.0 if a_key == b_key else 0.0
    return SequenceMatcher(None, a_key, b_key).ratio()

#Converts OpenCV image into base64 JPEG URL 
def _image_to_data_url(image: np.ndarray, jpeg_quality: int = 92) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Could not JPEG-encode frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"

#Parses response returned by OpenRouter 
def _parse_openrouter_json_content(content):
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        if parts:
            return json.loads("".join(parts))
    raise RuntimeError(f"Unexpected OpenRouter response content: {content!r}")

#Sends images to OCR model
def call_images_json(
    images: list[np.ndarray],
    prompt: str,
    schema: dict,
    *,
    model: str = OCR_MODEL,
    schema_name: str = "image_ocr_result",
    timeout: int = 180,
):
    """Call an OpenRouter image-capable model with one or more source frames."""
    if not images:
        raise ValueError("At least one image is required")

    content = [{"type": "text", "text": prompt}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(image)},
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
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

    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers={
            "Authorization": f"Bearer {get_openrouter_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter image OCR call failed: HTTP {response.status_code}\n"
            f"{response.text}"
        )

    content = response.json()["choices"][0]["message"]["content"]
    return _parse_openrouter_json_content(content)

#Divides video into overlapping windows for scanning 
def _scan_windows(duration: float, window: float, overlap: float):
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must satisfy 0 <= overlap < window")

    stride = window - overlap
    t = 0.0
    while t < duration:
        end = min(duration, t + window)
        yield t, end
        if end >= duration:
            break
        t += stride

#Performs whole OCR scan 
def scan_ocr_candidates(
    manifest: dict,
    *,
    query: str,
    target_object: str,
    target_region: str,
    text_description: str,
    extraction_instruction: str,
    scan_window_seconds: float = 30.0,
    scan_overlap_seconds: float = 2.0,
    min_confidence: float = 0.20,
    model: str = OCR_MODEL,
    vlm_cache_dir: str | Path = "ocr_vlm_cache",
):
    """Whole-video coarse scan for appearances relevant to a visual visual-text query."""
    duration = float(manifest["video"]["duration"])
    candidates = []

    for window_start, window_end in _scan_windows(
        duration,
        scan_window_seconds,
        scan_overlap_seconds,
    ):
        clip_path = materialize_vlm_clip(
            manifest,
            window_start,
            window_end,
            output_dir=vlm_cache_dir,
            fps=6,
            width=960,
            crf=24,
            include_audio=False,
        )
        clip_duration = window_end - window_start

        prompt = f"""
You are the exhaustive VISUAL OCR detector for a long-video retrieval system.

USER QUERY:
{query}

TARGET OBJECT:
{target_object}

TARGET REGION:
{target_region}

REQUESTED TEXT TYPE:
{text_description}

EXTRACTION INSTRUCTION:
{extraction_instruction}

This clip is {clip_duration:.3f} seconds long.
Find EVERY DISTINCT APPEARANCE in this clip that contains the REQUESTED TEXT TYPE on/in the specified TARGET REGION of the specified TARGET OBJECT.

Rules:
- Only use text that is visibly present in the video frames. Do not use spoken audio.
- start_seconds/end_seconds are relative to the beginning of THIS clip.
- best_frame_seconds is the moment where the requested text-bearing target is largest, sharpest, least occluded, and most readable.
- ocr_text must contain ONLY the requested text type. Do not return nearby unrelated writing, logos, license plates, unit numbers, labels, or other text unless they are exactly what the prompt requests. Preserve spaces and punctuation when they matter.
- Never invent hidden, blurred, cropped, or ambiguous characters.
- If only part is reliably readable, return only the visible portion and set readable=false.
- If the target is visible but no requested characters are reliable, use ocr_text="" and readable=false.
- A continuously visible physical target should be one appearance, not one item per frame.
- If the same target disappears and later reappears, return another appearance.
- Different relevant targets visible at the same time are separate appearances.
- target_description should identify the physical target object, the target region, and its context well enough to find the same physical target again in source frames.
- confidence is 0..1 and reflects confidence that this is a real relevant OCR target/appearance, not certainty of every character.
- If the user's query asks for a particular visible word/string, return only appearances that plausibly satisfy that requested visible-text condition.
"""

        result = call_video_json(
            clip_path,
            prompt,
            OCR_VIDEO_SCAN_SCHEMA,
            model=model,
            schema_name="generic_visual_ocr_video_scan",
        )

        for item in result.get("items", []):
            confidence = float(item.get("confidence", 0.0))
            if confidence < min_confidence:
                continue

            rel_start = max(0.0, min(clip_duration, float(item["start_seconds"])))
            rel_end = max(0.0, min(clip_duration, float(item["end_seconds"])))
            rel_best = max(0.0, min(clip_duration, float(item["best_frame_seconds"])))
            if rel_end < rel_start:
                rel_start, rel_end = rel_end, rel_start
            if rel_end <= rel_start:
                rel_end = min(clip_duration, rel_start + 0.25)

            raw_text = normalize_ocr_text(item.get("ocr_text", ""))
            candidates.append(
                {
                    "start": window_start + rel_start,
                    "end": window_start + rel_end,
                    "best_timestamp": window_start + rel_best,
                    "coarse_ocr_text": raw_text,
                    "ocr_text": raw_text,
                    "readable": bool(item.get("readable", False)),
                    "confidence": confidence,
                    "target_description": item.get("target_description", ""),
                    "visual_evidence": item.get("visual_evidence", ""),
                }
            )

    return merge_ocr_candidates(candidates)

#MErges duplicate OCR candidates
def merge_ocr_candidates(candidates: list[dict], temporal_slack: float = 1.25):
    """Merge duplicate appearances produced by overlapping coarse scan windows."""
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda x: (x["start"], x["end"]))
    merged: list[dict] = []

    for candidate in ordered:
        match_index = None
        for i in range(len(merged) - 1, -1, -1):
            existing = merged[i]
            if candidate["start"] > existing["end"] + temporal_slack:
                break

            time_compatible = (
                candidate["start"] <= existing["end"] + temporal_slack
                and existing["start"] <= candidate["end"] + temporal_slack
            )
            if not time_compatible:
                continue

            candidate_text = candidate.get("ocr_text", "")
            existing_text = existing.get("ocr_text", "")
            desc_a = candidate.get("target_description", "")
            desc_b = existing.get("target_description", "")
            desc_similarity = (
                SequenceMatcher(None, desc_a.lower(), desc_b.lower()).ratio()
                if desc_a and desc_b
                else 0.0
            )

            if candidate_text and existing_text:
                text_similarity = _text_similarity(candidate_text, existing_text)
                # Text is the strongest identity cue. A description check helps
                # avoid merging two simultaneous targets that happen to show the
                # same short text.
                identity_compatible = text_similarity >= 0.78 and (
                    desc_similarity >= 0.35 or not (desc_a and desc_b)
                )
            else:
                identity_compatible = desc_similarity >= 0.65

            if identity_compatible:
                match_index = i
                break

        if match_index is None:
            item = dict(candidate)
            item["coarse_observations"] = [dict(candidate)]
            merged.append(item)
            continue

        existing = merged[match_index]
        existing["start"] = min(existing["start"], candidate["start"])
        existing["end"] = max(existing["end"], candidate["end"])
        existing["coarse_observations"].append(dict(candidate))
        if candidate.get("confidence", 0.0) > existing.get("confidence", 0.0):
            for key in (
                "best_timestamp",
                "coarse_ocr_text",
                "ocr_text",
                "readable",
                "confidence",
                "target_description",
                "visual_evidence",
            ):
                existing[key] = candidate.get(key)

    return sorted(merged, key=lambda x: x["start"])

#Reads one frame from original video at specific timestamp 
def _read_frame(cap: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None

#Determines which frames to inspect around coarse candidate 
def _sample_refinement_timestamps(
    candidate: dict,
    *,
    fps: float,
    padding: float,
    max_frames: int,
    video_duration: float,
):
    start = max(0.0, float(candidate["start"]) - padding)
    end = min(video_duration, float(candidate["end"]) + padding)
    best = min(end, max(start, float(candidate.get("best_timestamp", (start + end) / 2))))

    local_start = max(start, best - 1.25)
    local_end = min(end, best + 1.25)
    step = 1.0 / max(0.1, float(fps))
    local = list(np.arange(local_start, local_end + step * 0.5, step))

    anchors = [start, (start + end) / 2.0, end, best]
    timestamps = sorted(
        {
            round(min(video_duration, max(0.0, float(t))), 4)
            for t in local + anchors
        },
        key=lambda t: (abs(t - best), t),
    )
    return sorted(timestamps[: max(1, int(max_frames))])

#Extracts text from target bbox region 
def _clamp_bbox(bbox: dict):
    x1 = max(0.0, min(1000.0, float(bbox.get("x1", 0.0))))
    y1 = max(0.0, min(1000.0, float(bbox.get("y1", 0.0))))
    x2 = max(0.0, min(1000.0, float(bbox.get("x2", 0.0))))
    y2 = max(0.0, min(1000.0, float(bbox.get("y2", 0.0))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

#tTakes a frame and extracts target text region from normalized bounding box
def _crop_from_bbox(frame: np.ndarray, bbox: dict, pad_fraction: float = 0.10):
    h, w = frame.shape[:2]
    bbox = _clamp_bbox(bbox)
    x1 = bbox["x1"] / 1000.0 * w
    y1 = bbox["y1"] / 1000.0 * h
    x2 = bbox["x2"] / 1000.0 * w
    y2 = bbox["y2"] / 1000.0 * h

    if x2 <= x1 or y2 <= y1:
        return None

    pad_x = (x2 - x1) * pad_fraction
    pad_y = (y2 - y1) * pad_fraction
    ix1 = max(0, int(math.floor(x1 - pad_x)))
    iy1 = max(0, int(math.floor(y1 - pad_y)))
    ix2 = min(w, int(math.ceil(x2 + pad_x)))
    iy2 = min(h, int(math.ceil(y2 + pad_y)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return frame[iy1:iy2, ix1:ix2].copy()

#Measures sharpness of image crop 
def _sharpness(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

#Takes coarse OCR candidate and revisits them using the original resolution source frames 
def _refine_candidate_frames(
    manifest: dict,
    candidate: dict,
    *,
    query: str,
    target_object: str,
    target_region: str,
    text_description: str,
    extraction_instruction: str,
    fps: float,
    padding: float,
    max_frames: int,
    image_batch_size: int,
    model: str,
):
    video_path = str(manifest["video"]["path"])
    duration = float(manifest["video"]["duration"])
    timestamps = _sample_refinement_timestamps(
        candidate,
        fps=fps,
        padding=padding,
        max_frames=max_frames,
        video_duration=duration,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")

    frames = []
    frame_times = []
    try:
        for timestamp in timestamps:
            frame = _read_frame(cap, timestamp)
            if frame is not None:
                frames.append(frame)
                frame_times.append(timestamp)
    finally:
        cap.release()

    if not frames:
        return []

    coarse_text = candidate.get("coarse_ocr_text", "")
    target_description = candidate.get("target_description", "")
    readings = []

    batch_size = max(1, int(image_batch_size))
    for offset in range(0, len(frames), batch_size):
        batch = frames[offset : offset + batch_size]
        batch_times = frame_times[offset : offset + batch_size]
        prompt = f"""
You are refining a visual OCR candidate from a long video.

USER QUERY:
{query}

TARGET OBJECT:
{target_object}

TARGET REGION:
{target_region}

REQUESTED TEXT TYPE:
{text_description}

EXTRACTION INSTRUCTION:
{extraction_instruction}

The coarse video scan described the target as:
{target_description or '(no description)'}

The coarse OCR reading was:
{coarse_text or '(unreadable)'}

You are given {len(batch)} ORIGINAL-RESOLUTION source frames in chronological order.
For every image where the SAME relevant target is visible:
- image_index is 0-based within THIS image batch.
- target_present=true only when the requested TARGET REGION on/in the requested TARGET OBJECT is visible and relevant to the user query.
- bbox uses normalized 0..1000 coordinates around ONLY the TARGET REGION containing the REQUESTED TEXT TYPE. Do not box unrelated nearby text.
- ocr_text must contain only the requested TEXT TYPE from that target region. Explicitly exclude nearby text that serves a different role. Preserve useful spaces and punctuation.
- Never infer or invent obscured characters.
- readable=true only when the requested text is reliably readable.
- confidence is confidence in this frame-level OCR/localization.
- target_description should identify what/where the target is in this frame.
- If the target is absent, return target_present=false, ocr_text="", readable=false, confidence=0, target_description="", and bbox coordinates all 0.

Return one reading for every input image.
"""
        result = call_images_json(
            batch,
            prompt,
            OCR_FRAME_REFINE_SCHEMA,
            model=model,
            schema_name="generic_visual_ocr_frame_refinement",
        )

        for item in result.get("readings", []):
            local_index = int(item.get("image_index", -1))
            if local_index < 0 or local_index >= len(batch):
                continue
            if not bool(item.get("target_present", False)):
                continue

            frame = batch[local_index]
            crop = _crop_from_bbox(frame, item.get("bbox", {}))
            if crop is None or crop.size == 0:
                continue

            readings.append(
                {
                    "timestamp": float(batch_times[local_index]),
                    "ocr_text": normalize_ocr_text(item.get("ocr_text", "")),
                    "raw_ocr_text": item.get("ocr_text", ""),
                    "readable": bool(item.get("readable", False)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "target_description": item.get("target_description", ""),
                    "bbox": _clamp_bbox(item.get("bbox", {})),
                    "sharpness": _sharpness(crop),
                    "frame": frame,
                    "crop": crop,
                    "source": "frame_ocr",
                }
            )

    return readings

#Takes best localized crops and OCRs them again 
def _read_best_crops(
    readings: list[dict],
    *,
    query: str,
    target_object: str,
    target_region: str,
    text_description: str,
    extraction_instruction: str,
    model: str,
    max_crops: int = 5,
):
    if not readings:
        return []

    ranked = sorted(
        readings,
        key=lambda r: (
            float(r.get("confidence", 0.0)),
            math.log1p(max(0.0, float(r.get("sharpness", 0.0)))),
            r["crop"].shape[0] * r["crop"].shape[1],
        ),
        reverse=True,
    )[: max(1, int(max_crops))]

    crops = [r["crop"] for r in ranked]
    prompt = f"""
These images are crops of a semantically selected text region from a video.

USER QUERY:
{query}

TARGET OBJECT:
{target_object}

TARGET REGION:
{target_region}

REQUESTED TEXT TYPE:
{text_description}

EXTRACTION INSTRUCTION:
{extraction_instruction}

Read only the requested text type independently from each crop. Ignore any unrelated text that happens to remain inside the crop.
- image_index is 0-based.
- Preserve spaces and punctuation when meaningful.
- Do not infer hidden or ambiguous characters from context.
- If only part is readable, return only the reliable visible portion and readable=false.
- If no requested text is reliably readable, use ocr_text="" and readable=false.
- confidence is OCR confidence for the visible characters in that crop.
Return one reading for every input image.
"""
    result = call_images_json(
        crops,
        prompt,
        OCR_CROP_SCHEMA,
        model=model,
        schema_name="generic_visual_ocr_crop_reading",
    )

    output = []
    for item in result.get("readings", []):
        image_index = int(item.get("image_index", -1))
        if image_index < 0 or image_index >= len(ranked):
            continue
        source = ranked[image_index]
        output.append(
            {
                "ocr_text": normalize_ocr_text(item.get("ocr_text", "")),
                "raw_ocr_text": item.get("ocr_text", ""),
                "readable": bool(item.get("readable", False)),
                "confidence": float(item.get("confidence", 0.0)),
                "timestamp": float(source["timestamp"]),
                "source": "crop_ocr",
            }
        )
    return output

#Combines multiple readings of same text into one final answer 
def consensus_ocr_text(
    readings: Iterable[dict],
    similarity_threshold: float = 0.78,
):
    """Choose a canonical OCR reading using agreement across nearby frames."""
    usable = []
    for reading in readings:
        text = normalize_ocr_text(reading.get("ocr_text", ""))
        if not text:
            continue
        usable.append(
            {
                "text": text,
                "confidence": float(reading.get("confidence", 0.0)),
                "source": reading.get("source", "unknown"),
            }
        )

    if not usable:
        return "", 0.0

    clusters: list[list[dict]] = []
    for reading in sorted(usable, key=lambda r: r["confidence"], reverse=True):
        placed = False
        for cluster in clusters:
            if _text_similarity(reading["text"], cluster[0]["text"]) >= similarity_threshold:
                cluster.append(reading)
                placed = True
                break
        if not placed:
            clusters.append([reading])

    def cluster_score(cluster):
        return sum(max(0.05, float(x["confidence"])) for x in cluster)

    best_cluster = max(clusters, key=cluster_score)
    canonical = max(
        best_cluster,
        key=lambda x: (float(x["confidence"]), len(_comparison_key(x["text"]))),
    )["text"]
    confidences = sorted(
        [float(x["confidence"]) for x in best_cluster], reverse=True
    )[:3]
    return canonical, float(sum(confidences) / len(confidences))

#Chooses strongest frame-level observation 
def _save_best_evidence(
    candidate: dict,
    readings: list[dict],
    evidence_dir: Path,
    index: int,
):
    if not readings:
        return None, None, candidate.get("best_timestamp")

    best = max(
        readings,
        key=lambda r: (
            float(r.get("confidence", 0.0)),
            math.log1p(max(0.0, float(r.get("sharpness", 0.0)))),
        ),
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    frame_path = evidence_dir / f"appearance_{index:04d}_frame.jpg"
    crop_path = evidence_dir / f"appearance_{index:04d}_region.jpg"
    cv2.imwrite(str(frame_path), best["frame"])
    cv2.imwrite(str(crop_path), best["crop"])
    return str(frame_path), str(crop_path), float(best["timestamp"])

#Group repeated temporal appearances that resolve to the same OCR text
def _group_unique_texts(matches: list[dict], similarity_threshold: float = 0.90):
    groups: list[dict] = []

    for match in matches:
        text = normalize_ocr_text(match.get("ocr_text", ""))
        if not text:
            continue

        target = None
        for group in groups:
            if _text_similarity(text, group["text"]) >= similarity_threshold:
                target = group
                break
        if target is None:
            target = {
                "text": text,
                "confidence": float(match.get("ocr_confidence", match.get("confidence", 0.0))),
                "appearances": [],
            }
            groups.append(target)

        if float(match.get("ocr_confidence", 0.0)) > target["confidence"]:
            target["text"] = text
        target["confidence"] = max(
            target["confidence"],
            float(match.get("ocr_confidence", match.get("confidence", 0.0))),
        )
        target["appearances"].append(
            {
                "match_id": match.get("match_id"),
                "start": match.get("start"),
                "end": match.get("end"),
                "start_timestamp": match.get("start_timestamp"),
                "end_timestamp": match.get("end_timestamp"),
                "best_timestamp": match.get("best_timestamp"),
                "best_frame_path": match.get("best_frame_path"),
                "region_crop_path": match.get("region_crop_path"),
                "target_description": match.get("target_description", ""),
            }
        )

    return sorted(groups, key=lambda x: x["text"].casefold())

#main orchestration for entire visual-text pipeline 
def run_visual_text_extraction(
    manifest: dict,
    query: str,
    *,
    target_object: str,
    target_region: str,
    text_description: str,
    extraction_instruction: str,
    scan_window_seconds: float = 30.0,
    scan_overlap_seconds: float = 2.0,
    min_detection_confidence: float = 0.20,
    refine_fps: float = 6.0,
    refine_padding_seconds: float = 1.0,
    max_refine_frames: int = 14,
    image_batch_size: int = 6,
    max_crops_per_appearance: int = 5,
    model: str = OCR_MODEL,
    output_root: str | Path = "final_results",
    final_frame_fps: float = 4.0,
    max_frames_per_match: int | None = None,
    include_diagnostics: bool = True,
    return_mode: str = "all",
):
    """Extract prompt-specified visible text from arbitrary target objects/regions."""
    
    print("Started OCR")
    candidates = scan_ocr_candidates(
        manifest,
        query=query,
        target_object=target_object,
        target_region=target_region,
        text_description=text_description,
        extraction_instruction=extraction_instruction,
        scan_window_seconds=scan_window_seconds,
        scan_overlap_seconds=scan_overlap_seconds,
        min_confidence=min_detection_confidence,
        model=model,
    )

    query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    evidence_dir = Path(output_root) / query_hash / "visual_text_evidence"
    appearances = []

    for i, candidate in enumerate(candidates):
        
        print(
            f"STARTING {i+1}/{len(candidates)}",
        )
        frame_readings = _refine_candidate_frames(
            manifest,
            candidate,
            query=query,
            target_object=target_object,
            target_region=target_region,
            text_description=text_description,
            extraction_instruction=extraction_instruction,
            fps=refine_fps,
            padding=refine_padding_seconds,
            max_frames=max_refine_frames,
            image_batch_size=image_batch_size,
            model=model,
        )
        crop_readings = _read_best_crops(
            frame_readings,
            query=query,
            target_object=target_object,
            target_region=target_region,
            text_description=text_description,
            extraction_instruction=extraction_instruction,
            model=model,
            max_crops=max_crops_per_appearance,
        )

        # Crop OCR is strongest, frame OCR next, coarse-video OCR last.
        consensus_inputs = []
        consensus_inputs.extend(crop_readings)
        consensus_inputs.extend(
            {
                "ocr_text": r.get("ocr_text", ""),
                "confidence": float(r.get("confidence", 0.0)) * 0.85,
                "source": "frame_ocr",
            }
            for r in frame_readings
        )
        consensus_inputs.append(
            {
                "ocr_text": candidate.get("coarse_ocr_text", ""),
                "confidence": float(candidate.get("confidence", 0.0)) * 0.50,
                "source": "coarse_video_ocr",
            }
        )

        ocr_text, ocr_confidence = consensus_ocr_text(consensus_inputs)
        best_frame_path, crop_path, best_timestamp = _save_best_evidence(
            candidate,
            frame_readings,
            evidence_dir,
            i,
        )

        appearances.append(
            {
                "start": float(candidate["start"]),
                "end": float(candidate["end"]),
                "confidence": max(
                    float(candidate.get("confidence", 0.0)),
                    float(ocr_confidence),
                ),
                "extracted_text": ocr_text,
                "ocr_text": ocr_text,  # compatibility alias
                "readable": bool(ocr_text),
                "ocr_confidence": float(ocr_confidence),
                "target_object": target_object,
                "target_region": target_region,
                "text_description": text_description,
                "best_timestamp": float(best_timestamp),
                "best_timestamp_formatted": format_timestamp_precise(best_timestamp),
                "best_frame_path": best_frame_path,
                "region_crop_path": crop_path,
                "target_description": candidate.get("target_description", ""),
                "description": (
                    f"{text_description}: {ocr_text}"
                    if ocr_text
                    else f"Visible {target_region}; requested text unreadable"
                ),
                "visual_evidence": candidate.get("visual_evidence", ""),
                "ocr_readings": [
                    {
                        k: v
                        for k, v in reading.items()
                        if k not in {"frame", "crop"}
                    }
                    for reading in frame_readings
                ],
                "crop_ocr_readings": crop_readings,
            }
        )

    if return_mode == "best" and appearances:
        appearances = [
            max(
                appearances,
                key=lambda x: (
                    float(x.get("ocr_confidence", 0.0)),
                    float(x.get("confidence", 0.0)),
                ),
            )
        ]

    matches, result_file = materialize_final_matches(
        manifest,
        appearances,
        query,
        output_root=output_root,
        frame_fps=final_frame_fps,
        max_frames_per_match=max_frames_per_match,
    )
    text_entities = _group_unique_texts(matches)

    result = {
        "query": query,
        "executor": "visual_text_extraction",
        "target_spec": {
            "target_object": target_object,
            "target_region": target_region,
            "text_description": text_description,
            "extraction_instruction": extraction_instruction,
        },
        "num_matches": len(matches),
        "num_unique_readable_texts": len(text_entities),
        "text_entities": text_entities,
        "matches": matches,
        "results_file": str(result_file),
    }
    if include_diagnostics:
        result["diagnostics"] = {
            "num_coarse_candidates": len(candidates),
            "scan_window_seconds": scan_window_seconds,
            "scan_overlap_seconds": scan_overlap_seconds,
            "refine_fps": refine_fps,
            "max_refine_frames": max_refine_frames,
        }

    # Replace the generic materialization manifest with the richer visual-text result.
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
