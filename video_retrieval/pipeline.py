"""The retrieval pipeline: from a question to timestamped clips, for one indexed video."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any

from . import boundaries, learning, local_backend
from .config import DATA_DIR
from .local_backend import LocalModels, use_models
from .retrieval import (
    apply_temporal_ordering,
    build_temporal_evidence_map,
    candidates_from_evidence_map,
    cluster_fused_results,
    fuse_retrieval_results,
    plan_query,
    recursive_refine_candidates,
    run_retrieval_plan,
    scoring_override,
)
from .visual_text import run_visual_text_extraction
from .verification import refine_instances, temporal_nms, verify_candidates_flash, verify_instances_pro
from .video import materialize_final_matches

# Stage messages reported while a search runs, in order. Visual-text queries report
# "Planning query", "Reading visible text", then "Extracting matching clips".
SEARCH_STAGES = [
    "Planning query",
    "Searching visual, scene, and speech indexes",
    "Building evidence timeline",
    "Finding candidate moments",
    "Verifying candidates with the vision model",
    "Confirming matches with the verifier model",
    "Refining clip boundaries",
    "Extracting matching clips",
]


@dataclass
class RetrievalResources:
    """Every precomputed index and record a search over one video needs."""

    manifest: dict
    video_index: Any = None
    video_metadata: list[dict] | None = None
    metadata_index: Any = None
    metadata_records: list[dict] | None = None
    transcript_index: Any = None
    transcript_bm25: Any = None
    transcript_metadata: list[dict] | None = None
    local_models: Any = None

class VideoRetrievalPipeline:
    """One indexed video, ready to answer questions with the configured local models."""

    def __init__(self, resources: RetrievalResources):
        self.resources = resources

    def retrieve(self, query: str, reporter=None, cancel_event=None, **kwargs):
        models = self.resources.local_models or LocalModels()
        # Settings tuned by bench.tune apply only in the mode they were tuned for: padding that
        # suits a quick answer could starve the verifier of context. Explicit arguments still win.
        tuned = load_tuned_settings()
        mode = "verified" if kwargs.get("run_verification", True) else "quick"
        applies = bool(tuned) and tuned.get("mode") == mode
        if applies:
            kwargs = {**(tuned.get("pipeline") or {}), **kwargs}
        scored = scoring_override(tuned.get("scoring")) if applies else nullcontext()
        with use_models(models, reporter, cancel_event), scored:
            result = retrieve_video(query=query, resources=self.resources, **kwargs)
        result["model_backend"] = models.signature()
        if applies:
            result["tuned_settings"] = tuned.get("run")
        return result


TUNED_SETTINGS = DATA_DIR / "learning" / "retrieval_settings.json"


@lru_cache(maxsize=2)
def _read_settings(path, stamp):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tuned_settings(path=TUNED_SETTINGS):
    """Retrieval settings found by bench.tune, or None when nothing has been tuned."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return _read_settings(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError):
        return None

def locate_candidates(
    retrieval_results,
    plan,
    video_duration,
    fusion_bin_size=5,
    evidence_bin_size=2.0,
    evidence_smoothing_bins=1,
    candidate_max_gap=10,
    candidate_padding=10,
    candidate_relative_score_floor=None,
    max_candidates=None,
    recursive_candidate_search=True,
    recursive_search_depth=3,
    recursive_shrink_factor=0.50,
    recursive_child_overlap=0.50,
    recursive_child_relative_score_floor=0.60,
    recursive_max_children_per_candidate=2,
    recursive_min_window_seconds=12.0,
    recursive_context_padding=4.0,
):
    """Retrieval hits to ranked candidate windows, with no model calls."""
    fused = fuse_retrieval_results(retrieval_results, plan, bin_size=fusion_bin_size)
    evidence_map = build_temporal_evidence_map(
        retrieval_results,
        plan,
        video_duration=video_duration,
        bin_size=evidence_bin_size,
        smoothing_bins=evidence_smoothing_bins,
    )
    evidence_peak = max((float(row.get("score", 0.0)) for row in evidence_map), default=0.0)
    located = {"fused": fused, "evidence_map": evidence_map, "evidence_peak": evidence_peak,
               "initial_candidates": [], "candidates": [], "empty": evidence_peak <= 0 and not fused}
    if located["empty"]:
        return located

    return_mode = plan.get("return_mode", "all")
    if candidate_relative_score_floor is None:
        candidate_relative_score_floor = 0.05 if return_mode == "all" else 0.15

    # 4. Candidate Generation
    if evidence_peak > 0:
        initial_candidates = candidates_from_evidence_map(
            evidence_map,
            max_gap=candidate_max_gap,
            padding=candidate_padding,
            video_duration=video_duration,
            relative_score_floor=candidate_relative_score_floor,
        )
    else:
        # Defensive compatibility fallback for unusual legacy retrieval outputs.
        initial_candidates = cluster_fused_results(
            fused,
            bin_size=fusion_bin_size,
            max_gap=candidate_max_gap,
            padding=candidate_padding,
            video_duration=video_duration,
            relative_score_floor=candidate_relative_score_floor,
        )

    # 5. Recursive Evidence Search
    if recursive_candidate_search and evidence_peak > 0:
        candidates = recursive_refine_candidates(
            evidence_map,
            initial_candidates,
            plan=plan,
            max_depth=recursive_search_depth,
            shrink_factor=recursive_shrink_factor,
            child_overlap=recursive_child_overlap,
            child_relative_score_floor=recursive_child_relative_score_floor,
            max_children_per_node=recursive_max_children_per_candidate,
            min_window_seconds=recursive_min_window_seconds,
            context_padding=recursive_context_padding,
            video_duration=video_duration,
            return_mode=return_mode,
        )
    else:
        candidates = initial_candidates

    # Ordering constraints only make sense once candidate windows exist.
    candidates = apply_temporal_ordering(candidates, retrieval_results, plan)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    located.update(initial_candidates=initial_candidates, candidates=candidates)
    return located


def retrieval_instances(candidates):
    """Candidates as unverified answers, the way quick mode returns them."""
    return [
        {
            "start": float(candidate["start"]),
            "end": float(candidate["end"]),
            "confidence": float(candidate.get("score", 0.0)),
            "description": "Candidate moment ranked by retrieval evidence (not verified)",
            "source_candidate_id": candidate.get("candidate_id"),
            "retrieval_score": float(candidate.get("score", 0.0)),
        }
        for candidate in candidates
    ]


def retrieve_video(
    query,
    resources: RetrievalResources,
    retrieval_top_k=200,
    fusion_bin_size=5,
    evidence_bin_size=2.0,
    evidence_smoothing_bins=1,
    candidate_max_gap=10,
    candidate_padding=10,
    candidate_relative_score_floor=None,
    max_candidates=None,
    recursive_candidate_search=True,
    recursive_search_depth=3,
    recursive_shrink_factor=0.50,
    recursive_child_overlap=0.50,
    recursive_child_relative_score_floor=0.60,
    recursive_max_children_per_candidate=2,
    recursive_min_window_seconds=12.0,
    recursive_context_padding=4.0,
    flash_min_confidence=0.10,
    pro_min_confidence=0.25,
    run_pro_verification=True,
    run_verification=True,
    verifier_window_seconds=25.0,
    verifier_overlap_seconds=5.0,
    refine_boundaries=True,
    refinement_stages=(8.0, 4.0, 2.0),
    learned_boundaries=None,
    nms_iou_threshold=0.55,
    final_frame_fps=4.0,
    max_frames_per_match=None,
    output_root="final_results",
    ocr_scan_window_seconds=30.0,
    ocr_scan_overlap_seconds=2.0,
    ocr_min_detection_confidence=0.20,
    ocr_refine_fps=6.0,
    ocr_refine_padding_seconds=1.0,
    ocr_max_refine_frames=14,
    ocr_image_batch_size=6,
    ocr_max_crops_per_appearance=5,
    include_diagnostics=True,
    include_evidence_map=False,
):
    """Answer `query` against one video.

    Each match carries precise timestamps, a clip and frames cut from the original source.
    """
    manifest = resources.manifest
    video_index = resources.video_index
    video_metadata = resources.video_metadata
    metadata_index = resources.metadata_index
    metadata_records = resources.metadata_records
    transcript_index = resources.transcript_index
    transcript_bm25 = resources.transcript_bm25
    transcript_metadata = resources.transcript_metadata

    # 1. Query planning
    local_backend.stage("Planning query")
    plan = plan_query(query)
    return_mode = plan.get("return_mode", "all")

    if plan.get("executor") == "visual_text_extraction":
        local_backend.stage("Reading visible text")
        result = run_visual_text_extraction(
            manifest,
            query=query,
            target_object=plan["target_object"],
            target_region=plan["target_region"],
            text_description=plan["text_description"],
            extraction_instruction=plan["extraction_instruction"],
            scan_window_seconds=ocr_scan_window_seconds,
            scan_overlap_seconds=ocr_scan_overlap_seconds,
            min_detection_confidence=ocr_min_detection_confidence,
            refine_fps=ocr_refine_fps,
            refine_padding_seconds=ocr_refine_padding_seconds,
            max_refine_frames=ocr_max_refine_frames,
            image_batch_size=ocr_image_batch_size,
            max_crops_per_appearance=ocr_max_crops_per_appearance,
            output_root=output_root,
            final_frame_fps=final_frame_fps,
            max_frames_per_match=max_frames_per_match,
            include_diagnostics=include_diagnostics,
            return_mode=return_mode,
        )
        result["plan"] = plan
        return result

    # 2. High-recall multimodal retrieval
    local_backend.stage("Searching visual, scene, and speech indexes")
    retrieval_results = run_retrieval_plan(
        plan,
        video_index,
        video_metadata,
        metadata_index,
        metadata_records,
        transcript_index,
        transcript_bm25,
        transcript_metadata,
        top_k=retrieval_top_k,
    )

    # 3-5. Evidence map, candidate windows, recursive refinement and ordering. None of it calls a
    # model, which is what lets bench.tune replay this stage under many settings.
    local_backend.stage("Building evidence timeline")
    located = locate_candidates(
        retrieval_results, plan, manifest["video"]["duration"],
        fusion_bin_size=fusion_bin_size, evidence_bin_size=evidence_bin_size,
        evidence_smoothing_bins=evidence_smoothing_bins, candidate_max_gap=candidate_max_gap,
        candidate_padding=candidate_padding, candidate_relative_score_floor=candidate_relative_score_floor,
        max_candidates=max_candidates, recursive_candidate_search=recursive_candidate_search,
        recursive_search_depth=recursive_search_depth, recursive_shrink_factor=recursive_shrink_factor,
        recursive_child_overlap=recursive_child_overlap,
        recursive_child_relative_score_floor=recursive_child_relative_score_floor,
        recursive_max_children_per_candidate=recursive_max_children_per_candidate,
        recursive_min_window_seconds=recursive_min_window_seconds,
        recursive_context_padding=recursive_context_padding,
    )
    if located["empty"]:
        return {"query": query, "plan": plan, "num_matches": 0, "matches": []}
    local_backend.stage("Finding candidate moments")
    fused, evidence_map = located["fused"], located["evidence_map"]
    evidence_peak = located["evidence_peak"]
    initial_candidates, candidates = located["initial_candidates"], located["candidates"]

    
    diagnostics = {
        "num_fused_bins": len(fused),
        "num_evidence_bins": len(evidence_map),
        "evidence_peak_score": evidence_peak,
        "num_initial_candidates": len(initial_candidates),
        "num_candidates": len(candidates),
    }

    if run_verification:
        # A trained ranker reorders the candidates and keeps a conformal prediction set;
        # without one this falls back to the pointwise pre-filter, and then to doing
        # nothing at all. The candidates that survive are what the vision model sees.
        candidates, selection = learning.select_candidates(
            candidates, evidence_map, plan, manifest["video"]["duration"]
        )
        if selection:
            diagnostics["prefilter"] = selection

        # 6. First vision-model pass finds every occurrence inside each candidate
        verification_started = time.perf_counter()
        local_backend.stage("Verifying candidates with the vision model")
        instances = verify_candidates_flash(
            manifest,
            candidates,
            query,
            plan=plan,
            min_confidence=flash_min_confidence,
            verifier_window_seconds=verifier_window_seconds,
            verifier_overlap_seconds=verifier_overlap_seconds,
        )
        diagnostics["num_flash_instances"] = len(instances)

        # 7. Stricter second pass filters false positives
        if instances and run_pro_verification:
            local_backend.stage("Confirming matches with the verifier model")
            instances = verify_instances_pro(
                manifest,
                instances,
                query,
                plan=plan,
                min_confidence=pro_min_confidence,
            )
            diagnostics["num_pro_instances"] = len(instances)

        diagnostics["num_pre_refine_instances"] = len(instances)

        if return_mode == "best" and instances:
            # Refinement never changes confidence and NMS always keeps the most confident
            # occurrence, so the final best event is already known; refine only that one.
            instances = [
                max(instances, key=lambda x: float(x.get("confidence", 0.0)))
            ]

        # 8. Coarse-to-fine boundary refinement PER occurrence
        if instances and refine_boundaries:
            local_backend.stage("Refining clip boundaries")
            instances = refine_instances(
                manifest,
                instances,
                query,
                plan=plan,
                stages=refinement_stages,
            )
        diagnostics["verification_seconds"] = round(time.perf_counter() - verification_started, 3)
    else:
        # Retrieval-only search: rank evidence candidates without vision-model checks.
        instances = retrieval_instances(candidates)

    # 9. Temporal NMS / overlap deduplication
    instances = temporal_nms(
        instances,
        iou_threshold=nms_iou_threshold,
        preserve_distinct_actors=True,
    )

    # A trained boundary model tightens retrieval's padded regions from the frame embeddings.
    # By default only in quick mode: verified mode has already spent vision calls on this.
    if learned_boundaries is None:
        learned_boundaries = not run_verification
    if learned_boundaries and instances:
        instances, boundary_info = boundaries.refine_instances(manifest, query, instances)
        if boundary_info:
            diagnostics["learned_boundaries"] = boundary_info

    # For a normal singular query, only now select the best final event.
    if return_mode == "best" and instances:
        instances = [
            max(instances, key=lambda x: float(x.get("confidence", 0.0)))
        ]

    # 10. Extract FINAL matching clip + frames from original video
    local_backend.stage("Extracting matching clips")
    matches, result_file = materialize_final_matches(
        manifest,
        instances,
        query,
        output_root=output_root,
        frame_fps=final_frame_fps,
        max_frames_per_match=max_frames_per_match,
    )

    if run_verification:
        # The vision model's own verdicts are the training signal for the pre-filter.
        learning.log_candidates(
            candidates, evidence_map, plan, manifest["video"]["duration"], query,
            [match.get("source_candidate_id") for match in matches],
            search_seconds=diagnostics.get("verification_seconds"),
        )

    result = {
        "query": query,
        "plan": plan,
        "verified": bool(run_verification),
        "num_matches": len(matches),
        "matches": matches,
        "results_file": str(result_file),
    }

    if include_diagnostics:
        diagnostics["num_final_matches"] = len(matches)
        diagnostics["top_evidence_bins"] = sorted(
            evidence_map,
            key=lambda row: float(row.get("score", 0.0)),
            reverse=True,
        )[:20]
        diagnostics["initial_candidates"] = initial_candidates
        diagnostics["candidates"] = candidates
        if include_evidence_map:
            diagnostics["evidence_map"] = evidence_map
        result["diagnostics"] = diagnostics

    return result
