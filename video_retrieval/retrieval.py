#Query planning, multimodal retrieval, fusion, and initial candidate clustering.
from __future__ import annotations

import json
import math

import requests

from .config import OPENROUTER_CHAT_URL, QUERY_PLANNER_MODEL, get_openrouter_api_key
from .embeddings import search_video
from .metadata import search_metadata
from .transcript import search_transcript_bm25, search_transcript_semantic

#Turns natural language query into a structured prompt
def plan_query(query):

    prompt = f"""
You are the routing + query-planning model for a long-video retrieval system.
The user supplies exactly ONE natural-language prompt. Your job is to decide
which executor should handle it and produce all executor-specific parameters.

USER PROMPT:
{query}

Choose exactly one executor:

1. temporal_grounding
   Use this for visible actions/events/objects/people/scenes, temporal relations,
   and spoken/audio content. This executor performs multimodal retrieval,
   candidate clustering, VLM verification, and temporal boundary refinement.

2. visual_text_extraction
   Use this whenever correctly answering the user's request requires reading,
   identifying, matching, or returning TEXT/CHARACTERS VISIBLY PRINTED OR SHOWN
   IN VIDEO FRAMES. The text can appear on ANY target: license plates, police
   vehicle unit/fleet numbers, badges, signs, uniforms, labels, serial/model
   numbers, screens, documents, packages, jerseys, storefronts, road markings,
   timestamps burned into video, and other text-bearing objects/regions.

CRITICAL ROUTING RULES AND EXAMPLES:
- The user must NOT need to say "OCR". Infer visual text extraction from intent.
- "identify all license plate numbers" -> visual_text_extraction.
- "find all cop car numbers" -> visual_text_extraction. Here the desired text is
  the police unit/fleet identifier printed on the vehicle body, NOT the license
  plate unless the prompt explicitly asks for license plates.
- "read every badge number" -> visual_text_extraction.
- "find every sign that says STOP" -> visual_text_extraction because visible
  text determines the match.
- "find a red car" -> temporal_grounding because no visible text must be read.
- "find when an officer reads Miranda rights" -> temporal_grounding because the
  words are spoken audio/transcript, not visual writing.
- If visible text is only incidental and not needed to answer, use
  temporal_grounding.

For visual_text_extraction, decompose the request into FOUR semantic fields:

A. target_object
   The physical object/entity that carries the desired text.
   Examples: "vehicle", "police vehicle", "police officer", "street sign",
   "laptop", "equipment", "package".

B. target_region
   The specific region on/in that object where the desired text should appear.
   Examples: "license plate", "painted/printed unit or fleet number on the
   vehicle body", "badge/uniform identification area", "street-name panel",
   "screen", "serial-number label".

C. text_description
   The semantic TYPE of text the user wants returned.
   Examples: "license plate registration number", "police vehicle unit/fleet
   identifier", "officer badge number", "street name", "serial number".

D. extraction_instruction
   A precise instruction telling the visual-text executor what characters to
   return and what nearby text to exclude. Preserve the user's intent. When
   ambiguity is likely, explicitly distinguish the requested text from nearby
   alternatives (e.g. unit number vs license plate).

Do NOT hard-code license plates. Derive these fields from the user's prompt.

For temporal_grounding, use these retrieval channels:
1. video: text-to-video embedding similarity for visible actions/objects/scenes.
2. metadata: dense semantic descriptions of video chunks.
3. transcript_semantic: semantic search over spoken transcript.
4. transcript_bm25: exact lexical transcript search.

DECOMPOSE TEMPORAL-GROUNDING QUERIES INTO ATOMIC EVIDENCE.
Create 4-12 evidence_predicates whenever possible. Each predicate should describe
one independently searchable piece of evidence rather than simply paraphrasing the
full user query. Examples include a target action, a target object/state, a contextual
cue, or a spoken phrase. For each predicate:
- role is one of target, cue, context, speech.
- modalities lists the retrieval channels that can actually retrieve that evidence.
- required=true only when the evidence is genuinely necessary for the event.
- importance is 0..1 and reflects how discriminative that predicate is.
Also return negative_evidence for confounders that should reject a candidate and
temporal_constraints for ordering/state-transition requirements.

Examples:
- 'vehicle being pulled over at night' can decompose into vehicle stopped roadside,
  officer/person interacting near vehicle, police/emergency-light cues, nighttime,
  and a moving-to-stopped transition.
- 'person being handcuffed' can decompose into hands behind back, officer manipulating
  wrists, handcuffs/restraint cue, and detention context.
- 'reads Miranda rights' should include transcript predicates for distinctive spoken
  language rather than relying only on a single full-query embedding.

General planning rules:
- If the request says all/every/each/every time/every instance or otherwise asks
  for exhaustive results, set return_mode="all"; otherwise use "best".
- For temporal_grounding, generate multiple high-recall queries per relevant
  channel and weights that sum to 1.
- For visual_text_extraction, set retrieval weights to 0; it has its own
  whole-video coarse-to-fine scan so tiny text is not lost by embedding top-k.
  Also set evidence_predicates=[], negative_evidence=[], temporal_constraints=[],
  and set target_event to a short description of the user's requested extraction.
- Estimate duration of ONE occurrence, not the entire video.
- Define what counts and does not count as a valid result.
- Keep routing_reason short (one sentence); it is diagnostic, not chain-of-thought.
"""
    #Defines what the planner must return 
    schema = {
        "type": "object",
        "properties": {
            "query_type": {"type": "string"},
            "executor": {
                "type": "string",
                "enum": ["temporal_grounding", "visual_text_extraction"],
            },
            "routing_reason": {"type": "string"},
            "return_mode": {
                "type": "string",
                "enum": ["all", "best"],
            },
            "target_object": {"type": "string"},
            "target_region": {"type": "string"},
            "text_description": {"type": "string"},
            "extraction_instruction": {"type": "string"},
            "visual_queries": {
                "type": "array",
                "items": {"type": "string"},
            },
            "metadata_queries": {
                "type": "array",
                "items": {"type": "string"},
            },
            "transcript_queries": {
                "type": "array",
                "items": {"type": "string"},
            },
            "target_event": {"type": "string"},
            "evidence_predicates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": ["target", "cue", "context", "speech"],
                        },
                        "modalities": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "video",
                                    "metadata",
                                    "transcript_semantic",
                                    "transcript_bm25",
                                ],
                            },
                        },
                        "required": {"type": "boolean"},
                        "importance": {"type": "number"},
                    },
                    "required": [
                        "id",
                        "description",
                        "role",
                        "modalities",
                        "required",
                        "importance",
                    ],
                    "additionalProperties": False,
                },
            },
            "negative_evidence": {
                "type": "array",
                "items": {"type": "string"},
            },
            "temporal_constraints": {
                "type": "array",
                "items": {"type": "string"},
            },
            "subevents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "order": {"type": "integer"},
                    },
                    "required": ["description", "order"],
                    "additionalProperties": False,
                },
            },
            "expected_duration": {
                "type": "object",
                "properties": {
                    "min_seconds": {"type": "number"},
                    "max_seconds": {"type": "number"},
                },
                "required": ["min_seconds", "max_seconds"],
                "additionalProperties": False,
            },
            "requires_temporal_order": {"type": "boolean"},
            "event_definition": {
                "type": "object",
                "properties": {
                    "counts_as_match": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "does_not_count": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["counts_as_match", "does_not_count"],
                "additionalProperties": False,
            },
            "weights": {
                "type": "object",
                "properties": {
                    "video": {"type": "number"},
                    "metadata": {"type": "number"},
                    "transcript_semantic": {"type": "number"},
                    "transcript_bm25": {"type": "number"},
                },
                "required": [
                    "video",
                    "metadata",
                    "transcript_semantic",
                    "transcript_bm25",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "query_type",
            "executor",
            "routing_reason",
            "return_mode",
            "target_object",
            "target_region",
            "text_description",
            "extraction_instruction",
            "visual_queries",
            "metadata_queries",
            "transcript_queries",
            "target_event",
            "evidence_predicates",
            "negative_evidence",
            "temporal_constraints",
            "subevents",
            "expected_duration",
            "requires_temporal_order",
            "event_definition",
            "weights",
        ],
        "additionalProperties": False,
    }

    payload = {
        "model": QUERY_PLANNER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "retrieval_plan",
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

    #Call the planner
    response = requests.post(
        OPENROUTER_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"Query planning failed: HTTP {response.status_code}\n"
            f"{response.text}"
        )

    content = response.json()["choices"][0]["message"]["content"]
    plan = json.loads(content)

    #Type of query
    if plan.get("executor") == "visual_text_extraction":
        plan["weights"] = {
            "video": 0.0,
            "metadata": 0.0,
            "transcript_semantic": 0.0,
            "transcript_bm25": 0.0,
        }
        defaults = {
            "target_object": "text-bearing object",
            "target_region": "region containing the requested visible text",
            "text_description": "visible text requested by the user",
            "extraction_instruction": "Read only the visible text requested by the user exactly; ignore unrelated nearby text.",
        }
        for key, default in defaults.items():
            if not str(plan.get(key, "")).strip():
                plan[key] = default
        # Visual-text discovery/refinement performs temporal localization itself.
        plan["visual_queries"] = []
        plan["metadata_queries"] = []
        plan["transcript_queries"] = []
        plan["target_event"] = str(plan.get("target_event") or query)
        plan["evidence_predicates"] = []
        plan["negative_evidence"] = []
        plan["temporal_constraints"] = []
    else:
        # Text-specific fields are intentionally empty for normal temporal search.
        plan["target_object"] = ""
        plan["target_region"] = ""
        plan["text_description"] = ""
        plan["extraction_instruction"] = ""
        plan["target_event"] = str(plan.get("target_event") or query).strip()

        valid_modalities = {
            "video",
            "metadata",
            "transcript_semantic",
            "transcript_bm25",
        }
        
        #Cleans output
        cleaned_predicates = []
        seen_predicates = set()
        for i, predicate in enumerate(plan.get("evidence_predicates", [])):
            description = str(predicate.get("description", "")).strip()
            if not description:
                continue
            key = description.lower()
            if key in seen_predicates:
                continue
            seen_predicates.add(key)

            modalities = [
                m for m in predicate.get("modalities", [])
                if m in valid_modalities
            ]
            if not modalities:
                modalities = ["video", "metadata"]

            cleaned_predicates.append({
                "id": str(predicate.get("id") or f"evidence_{i}"),
                "description": description,
                "role": str(predicate.get("role") or "cue"),
                "modalities": modalities,
                "required": bool(predicate.get("required", False)),
                "importance": max(0.0, min(1.0, float(predicate.get("importance", 0.5)))),
            })

        plan["evidence_predicates"] = cleaned_predicates
        plan["negative_evidence"] = [
            str(x).strip()
            for x in plan.get("negative_evidence", [])
            if str(x).strip()
        ]
        plan["temporal_constraints"] = [
            str(x).strip()
            for x in plan.get("temporal_constraints", [])
            if str(x).strip()
        ]

        weights = {
            key: max(0.0, float(value))
            for key, value in plan["weights"].items()
        }
        total = sum(weights.values())
        
        #Normalize retrieval weights
        if total <= 0:
            weights = {
                "video": 0.5,
                "metadata": 0.5,
                "transcript_semantic": 0.0,
                "transcript_bm25": 0.0,
            }
        else:
            weights = {key: value / total for key, value in weights.items()}
        plan["weights"] = weights

    return plan
#Removes duplicate queries that are produced by the planner
def _dedupe_query_specs(specs):
    output = []
    by_text = {}

    for spec in specs:
        query = str(spec.get("query", "")).strip()
        if not query:
            continue
        key = query.lower()
        spec = spec.copy()
        spec["query"] = query

        if key not in by_text:
            by_text[key] = len(output)
            output.append(spec)
            continue

        existing = output[by_text[key]]
        if float(spec.get("importance", 1.0)) > float(existing.get("importance", 1.0)):
            output[by_text[key]] = spec

    return output

#Decides what queries need to be run for a specific channel
def _query_specs_for_channel(plan, channel):
    if channel == "video":
        base_queries = plan.get("visual_queries", [])
    elif channel == "metadata":
        base_queries = plan.get("metadata_queries", [])
    else:
        base_queries = plan.get("transcript_queries", [])

    specs = [
        {
            "query": q,
            "predicate_id": "planner_query",
            "predicate_role": "target",
            "predicate_required": False,
            "importance": 1.0,
        }
        for q in base_queries
    ]

    for predicate in plan.get("evidence_predicates", []):
        if channel not in predicate.get("modalities", []):
            continue
        specs.append({
            "query": predicate["description"],
            "predicate_id": predicate.get("id", "evidence"),
            "predicate_role": predicate.get("role", "cue"),
            "predicate_required": bool(predicate.get("required", False)),
            "importance": float(predicate.get("importance", 0.5)),
        })

    return _dedupe_query_specs(specs)

#Analyzes why prediction was created
def _annotate_ranking(ranking, spec, channel):
    annotated = []
    for item in ranking:
        row = item.copy()
        row["_retrieval_channel"] = channel
        row["_retrieval_query"] = spec["query"]
        row["_predicate_id"] = spec.get("predicate_id", "planner_query")
        row["_predicate_role"] = spec.get("predicate_role", "target")
        row["_predicate_required"] = bool(spec.get("predicate_required", False))
        row["_predicate_importance"] = float(spec.get("importance", 1.0))
        annotated.append(row)
    return annotated

#Execute both full-query searches and atomic evidence-predicate searches. Outputs retrieval results for each search channel 
def run_retrieval_plan(
    plan,
    video_index,
    video_metadata,
    metadata_index,
    metadata_records,
    transcript_index,
    transcript_bm25,
    transcript_metadata,
    top_k=100,
):
    results = {
        "video": [],
        "metadata": [],
        "transcript_semantic": [],
        "transcript_bm25": [],
    }

    weights = plan.get("weights", {})

    if video_index is not None and video_metadata and float(weights.get("video", 0.0)) > 0:
        for spec in _query_specs_for_channel(plan, "video"):
            ranking = search_video(
                spec["query"],
                video_index,
                video_metadata,
                top_k=top_k,
            )
            results["video"].append(_annotate_ranking(ranking, spec, "video"))

    if metadata_index is not None and metadata_records and float(weights.get("metadata", 0.0)) > 0:
        for spec in _query_specs_for_channel(plan, "metadata"):
            ranking = search_metadata(
                spec["query"],
                metadata_index,
                metadata_records,
                top_k=top_k,
            )
            results["metadata"].append(_annotate_ranking(ranking, spec, "metadata"))

    can_semantic = transcript_index is not None and transcript_metadata
    can_bm25 = transcript_bm25 is not None and transcript_metadata

    if can_semantic and float(weights.get("transcript_semantic", 0.0)) > 0:
        for spec in _query_specs_for_channel(plan, "transcript_semantic"):
            ranking = search_transcript_semantic(
                spec["query"],
                transcript_index,
                transcript_metadata,
                top_k=top_k,
            )
            results["transcript_semantic"].append(
                _annotate_ranking(ranking, spec, "transcript_semantic")
            )

    if can_bm25 and float(weights.get("transcript_bm25", 0.0)) > 0:
        for spec in _query_specs_for_channel(plan, "transcript_bm25"):
            ranking = search_transcript_bm25(
                spec["query"],
                transcript_bm25,
                transcript_metadata,
                top_k=top_k,
            )
            results["transcript_bm25"].append(
                _annotate_ranking(ranking, spec, "transcript_bm25")
            )

    return results

#Converts timestamp into a numbered bin
def timestamp_bin(seconds, bin_size=5):

    return int(
        seconds // bin_size
    )

#Flattens rankings 
def flatten_rankings(
    rankings
):

    output = []

    for ranking in rankings:
        output.extend(ranking)

    return output

#Assigns each finding into a different timestamp bin
def add_ranking_to_fusion(
    fused,
    ranking,
    weight,
    k=60,
    bin_size=5
):

    for rank, item in enumerate(
        ranking
    ):

        center = (
            item["start"]
            + item["end"]
        ) / 2

        bin_id = timestamp_bin(
            center,
            bin_size
        )

        score = (
            weight
            / (k + rank + 1)
        )

        if bin_id not in fused:

            fused[bin_id] = {
                "score": 0,
                "timestamps": [],
                "evidence": []
            }

        fused[bin_id]["score"] += (
            score
        )

        fused[bin_id][
            "timestamps"
        ].append(
            (
                item["start"],
                item["end"]
            )
        )

        fused[bin_id][
            "evidence"
        ].append(
            item
        )

#Calls above method for all four channels
def fuse_retrieval_results(
    retrieval_results,
    plan,
    bin_size=5
):

    fused = {}

    weights = plan["weights"]

    for ranking in retrieval_results[
        "video"
    ]:

        add_ranking_to_fusion(
            fused,
            ranking,
            weights["video"],
            bin_size=bin_size
        )

    for ranking in retrieval_results[
        "metadata"
    ]:

        add_ranking_to_fusion(
            fused,
            ranking,
            weights["metadata"],
            bin_size=bin_size
        )

    for ranking in retrieval_results[
        "transcript_semantic"
    ]:

        add_ranking_to_fusion(
            fused,
            ranking,
            weights[
                "transcript_semantic"
            ],
            bin_size=bin_size
        )

    for ranking in retrieval_results[
        "transcript_bm25"
    ]:

        add_ranking_to_fusion(
            fused,
            ranking,
            weights[
                "transcript_bm25"
            ],
            bin_size=bin_size
        )

    ranked = sorted(
        fused.items(),

        key=lambda x:
            x[1]["score"],

        reverse=True
    )

    return ranked

#Converts and normalizes all ranking scores to between 0 and 1 
def _calibrate_ranking_scores(ranking):
    """Map one retrieval ranking to stable 0..1 scores without mixing raw modalities."""
    if not ranking:
        return []

    raw = [max(0.0, float(item.get("score", 0.0))) for item in ranking]
    high = max(raw)
    low = min(raw)

    # BM25 can legitimately return an all-zero ranking. Do not manufacture signal.
    if high <= 0.0:
        return [0.0] * len(raw)

    spread = high - low
    output = []
    n = len(raw)

    # Rank decay prevents a long top-k list with numerically similar cosine
    # scores from painting the entire video as relevant.
    rank_tau = max(2.0, min(12.0, n * 0.10))

    for rank, score in enumerate(raw):
        if spread > 1e-12:
            relative_range = (score - low) / spread
        else:
            relative_range = 1.0

        relative_to_best = score / high if high > 0 else 0.0
        rank_decay = math.exp(-rank / rank_tau)

        # A score-ratio floor preserves strong secondary occurrences.
        calibrated = (
            relative_to_best * (0.05 + 0.70 * rank_decay)
            + 0.25 * relative_range
        )
        output.append(max(0.0, min(1.0, calibrated)))

    return output

#Smooth evidence scores between nieghboring time bins
def _smooth_series(values, radius=1):
    if radius <= 0 or len(values) <= 1:
        return list(values)

    output = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        weighted_sum = 0.0
        weight_sum = 0.0
        for j in range(lo, hi):
            weight = radius + 1 - abs(i - j)
            weighted_sum += values[j] * weight
            weight_sum += weight
        output.append(weighted_sum / weight_sum if weight_sum else values[i])
    return output



    # Each retrieval query first receives its own 0..1 map. Overlapping hits from the
    # same query combine with a noisy-OR, so agreement among overlapping chunks creates
    # a temporal peak instead of merely assigning a score to each chunk center. Query
    # maps are then combined within each modality and finally weighted by the planner to create a full timeline of query hits.
def build_temporal_evidence_map(
    retrieval_results,
    plan,
    video_duration,
    bin_size=2.0,
    smoothing_bins=1,
):
    video_duration = max(0.0, float(video_duration))
    bin_size = max(0.1, float(bin_size))
    n_bins = max(1, int(math.ceil(video_duration / bin_size)))
    channels = ["video", "metadata", "transcript_semantic", "transcript_bm25"]
    channel_maps = {channel: [0.0] * n_bins for channel in channels}
    channel_support = {channel: [0] * n_bins for channel in channels}

    for channel in channels:
        query_maps = []
        query_support_maps = []

        for ranking in retrieval_results.get(channel, []):
            if not ranking:
                continue

            calibrated = _calibrate_ranking_scores(ranking)
            qmap = [0.0] * n_bins
            qsupport = [0] * n_bins

            for item, base_score in zip(ranking, calibrated):
                if base_score <= 0:
                    continue

                start = max(0.0, float(item.get("start", 0.0)))
                end = min(video_duration, float(item.get("end", start)))
                if end <= start:
                    continue

                importance = max(
                    0.0,
                    min(1.0, float(item.get("_predicate_importance", 1.0))),
                )
                # Atomic cues should influence retrieval without overwhelming the
                # original full-query searches. Required/high-importance predicates
                # naturally receive more weight.
                predicate_factor = 0.55 + 0.45 * importance
                score = max(0.0, min(1.0, base_score * predicate_factor))

                first_bin = max(0, int(start // bin_size))
                last_bin = min(n_bins - 1, int(max(start, end - 1e-9) // bin_size))

                for bin_id in range(first_bin, last_bin + 1):
                    bin_start = bin_id * bin_size
                    bin_end = min(video_duration, bin_start + bin_size)
                    overlap = max(0.0, min(end, bin_end) - max(start, bin_start))
                    if overlap <= 0:
                        continue

                    overlap_fraction = overlap / max(1e-9, bin_end - bin_start)
                    contribution = score * math.sqrt(overlap_fraction)
                    # Noisy-OR rewards multiple overlapping relevant chunks while
                    # remaining bounded in [0, 1].
                    qmap[bin_id] = 1.0 - (1.0 - qmap[bin_id]) * (1.0 - contribution)
                    qsupport[bin_id] += 1

            query_maps.append(qmap)
            query_support_maps.append(qsupport)

        if not query_maps:
            continue

        for i in range(n_bins):
            scores = [qmap[i] for qmap in query_maps]
            peak = max(scores)
            mean = sum(scores) / len(scores)
            # A single strong atomic predicate can surface a candidate, while
            # agreement across predicates gives it a modest support boost.
            channel_maps[channel][i] = 0.80 * peak + 0.20 * mean
            channel_support[channel][i] = sum(1 for qmap in query_maps if qmap[i] > 0.05)

        channel_maps[channel] = _smooth_series(
            channel_maps[channel],
            radius=int(max(0, smoothing_bins)),
        )

    weights = plan.get("weights", {})
    total_scores = [0.0] * n_bins
    for i in range(n_bins):
        score = 0.0
        for channel in channels:
            score += max(0.0, float(weights.get(channel, 0.0))) * channel_maps[channel][i]
        total_scores[i] = max(0.0, min(1.0, score))

    evidence_map = []
    for i, score in enumerate(total_scores):
        start = i * bin_size
        end = min(video_duration, start + bin_size)
        evidence_map.append({
            "bin_id": i,
            "start": start,
            "end": end,
            "center": (start + end) / 2.0,
            "score": score,
            "channel_scores": {
                channel: channel_maps[channel][i]
                for channel in channels
            },
            "channel_query_support": {
                channel: channel_support[channel][i]
                for channel in channels
            },
        })

    return evidence_map

#Finds significant regions along a timeline
def candidates_from_evidence_map(
    evidence_map,
    max_gap=10.0,
    padding=10.0,
    video_duration=None,
    relative_score_floor=0.05,
    absolute_score_floor=0.02,
):
    """Turn significant evidence-map components into high-recall candidate regions."""
    if not evidence_map:
        return []

    best_score = max(float(row.get("score", 0.0)) for row in evidence_map)
    if best_score <= 0:
        return []

    threshold = min(
        best_score,
        max(
            float(absolute_score_floor),
            best_score * float(relative_score_floor),
        ),
    )
    selected = [row for row in evidence_map if float(row.get("score", 0.0)) >= threshold]
    if not selected:
        return []

    selected.sort(key=lambda row: float(row["start"]))
    regions = []

    for row in selected:
        if not regions or float(row["start"]) > regions[-1]["end"] + float(max_gap):
            regions.append({
                "start": float(row["start"]),
                "end": float(row["end"]),
                "scores": [float(row["score"])],
                "bin_ids": [int(row["bin_id"])],
            })
        else:
            region = regions[-1]
            region["end"] = max(region["end"], float(row["end"]))
            region["scores"].append(float(row["score"]))
            region["bin_ids"].append(int(row["bin_id"]))

    duration = None if video_duration is None else float(video_duration)
    for region in regions:
        region["start"] = max(0.0, region["start"] - float(padding))
        region["end"] = region["end"] + float(padding)
        if duration is not None:
            region["end"] = min(duration, region["end"])

        region["score"] = max(region["scores"])
        region["evidence_mass"] = sum(region["scores"])
        region["num_supporting_bins"] = len(region["scores"])
        region["search_stage"] = "evidence_component"

    regions.sort(
        key=lambda x: (x["score"], x["evidence_mass"], x["num_supporting_bins"]),
        reverse=True,
    )

    for i, region in enumerate(regions):
        region["candidate_id"] = i

    return regions

#Score of an arbitrary window based on the scores inside the window 
def _window_evidence_score(evidence_map, start, end):
    rows = [
        row for row in evidence_map
        if float(row["end"]) > start and float(row["start"]) < end
    ]
    if not rows:
        return 0.0, 0, []

    scores = sorted((float(row.get("score", 0.0)) for row in rows), reverse=True)
    peak = scores[0]
    top_n = max(1, int(math.ceil(len(scores) * 0.25)))
    top_mean = sum(scores[:top_n]) / top_n
    mean = sum(scores) / len(scores)
    score = 0.55 * peak + 0.35 * top_mean + 0.10 * mean
    return score, len(rows), [int(row["bin_id"]) for row in rows]

#Generates smaller overlapping windows from one larger window
def _generate_child_windows(start, end, child_width, overlap=0.50):
    width = end - start
    if child_width >= width - 1e-9:
        return [(start, end)]

    stride = max(0.25, child_width * (1.0 - overlap))
    windows = []
    t = start
    while t < end:
        child_end = min(end, t + child_width)
        child_start = max(start, child_end - child_width)
        candidate = (child_start, child_end)
        if not windows or candidate != windows[-1]:
            windows.append(candidate)
        if child_end >= end:
            break
        t += stride
    return windows

#Calculates temporal IoU
def _interval_iou(a, b):
    intersection = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return 0.0 if union <= 0 else intersection / union

#Recursively zoom candidate regions toward dense evidence peaks.
def recursive_refine_candidates(
    evidence_map,
    candidates,
    plan=None,
    max_depth=3,
    shrink_factor=0.50,
    child_overlap=0.50,
    child_relative_score_floor=0.60,
    max_children_per_node=2,
    min_window_seconds=12.0,
    context_padding=4.0,
    video_duration=None,
    return_mode="all",
):
    if not candidates:
        return []

    expected = (plan or {}).get("expected_duration", {})
    expected_max = max(0.0, float(expected.get("max_seconds", 0.0) or 0.0))
    target_width = max(float(min_window_seconds), expected_max * 1.5)
    duration = None if video_duration is None else float(video_duration)
    max_children = 1 if return_mode == "best" else max(1, int(max_children_per_node))

    leaves = []

    def recurse(node, depth, root_id, history):
        start = float(node["start"])
        end = float(node["end"])
        width = end - start
        node_score, support_bins, bin_ids = _window_evidence_score(evidence_map, start, end)

        current = node.copy()
        current.update({
            "score": max(float(node.get("score", 0.0)), node_score),
            "recursive_depth": depth,
            "recursive_root_id": root_id,
            "recursive_support_bins": support_bins,
            "bin_ids": bin_ids,
            "recursive_history": history,
        })

        if depth >= int(max_depth) or width <= target_width * 1.05:
            leaves.append(current)
            return

        child_width = max(target_width, width * float(shrink_factor))
        if child_width >= width * 0.98:
            leaves.append(current)
            return

        child_windows = _generate_child_windows(
            start,
            end,
            child_width,
            overlap=float(child_overlap),
        )
        scored = []
        for child_start, child_end in child_windows:
            score, count, child_bins = _window_evidence_score(
                evidence_map,
                child_start,
                child_end,
            )
            scored.append({
                "start": child_start,
                "end": child_end,
                "score": score,
                "recursive_support_bins": count,
                "bin_ids": child_bins,
            })

        if not scored:
            leaves.append(current)
            return

        best = max(float(child["score"]) for child in scored)
        if best <= 0:
            leaves.append(current)
            return

        threshold = best * float(child_relative_score_floor)
        selected = [child for child in scored if float(child["score"]) >= threshold]
        selected.sort(key=lambda x: float(x["score"]), reverse=True)

        # Avoid exploring several almost-identical overlapping children while still
        # preserving spatially/temporally distinct peaks.
        diverse = []
        for child in selected:
            if any(_interval_iou(child, kept) > 0.85 for kept in diverse):
                continue
            diverse.append(child)
            if len(diverse) >= max_children:
                break

        if not diverse:
            leaves.append(current)
            return

        for child in diverse:
            child_history = list(history) + [{
                "depth": depth + 1,
                "parent_start": start,
                "parent_end": end,
                "child_start": child["start"],
                "child_end": child["end"],
                "child_score": child["score"],
            }]
            recurse(child, depth + 1, root_id, child_history)

    for root in candidates:
        recurse(root, 0, root.get("candidate_id"), [])

    padded = []
    for leaf in leaves:
        row = leaf.copy()
        row["start"] = max(0.0, float(row["start"]) - float(context_padding))
        row["end"] = float(row["end"]) + float(context_padding)
        if duration is not None:
            row["end"] = min(duration, row["end"])
        row["search_stage"] = "recursive_evidence_search"
        padded.append(row)

    # Deduplicate branches from overlapping root components. Nearby leaf centers
    # represent the same evidence peak; one verifier window can still return
    # multiple true instances inside that neighborhood.
    padded.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    deduped = []
    center_merge_distance = max(float(min_window_seconds), target_width) * 0.80
    for candidate in padded:
        candidate_center = (float(candidate["start"]) + float(candidate["end"])) / 2.0
        duplicate = False
        for existing in deduped:
            existing_center = (float(existing["start"]) + float(existing["end"])) / 2.0
            same_peak = abs(candidate_center - existing_center) < center_merge_distance
            if same_peak or _interval_iou(candidate, existing) > 0.65:
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)

    deduped.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    for i, candidate in enumerate(deduped):
        candidate["candidate_id"] = i

    return deduped


def format_timestamp(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"

# Convert fused temporal hits into distinct candidate regions.

def cluster_fused_results(
    fused,
    bin_size=5,
    max_gap=10,
    padding=10,
    video_duration=None,
    relative_score_floor=0.05,
    max_bins=None,
):
    if not fused:
        return []

    best_score = float(fused[0][1]["score"])
    score_floor = best_score * float(relative_score_floor)

    selected = [
        (bin_id, info)
        for bin_id, info in fused
        if float(info["score"]) >= score_floor
    ]

    if max_bins is not None:
        selected = selected[:max_bins]

    hits = []

    for bin_id, info in selected:
        center = bin_id * bin_size
        hits.append({
            "center": center,
            "start": max(0.0, center - bin_size / 2),
            "end": center + bin_size / 2,
            "score": float(info["score"]),
            "evidence": list(info.get("evidence", [])),
            "timestamps": list(info.get("timestamps", [])),
        })

    hits.sort(key=lambda x: x["start"])
    regions = []

    for hit in hits:
        if not regions or hit["start"] > regions[-1]["end"] + max_gap:
            regions.append({
                "start": hit["start"],
                "end": hit["end"],
                "scores": [hit["score"]],
                "evidence": list(hit["evidence"]),
                "timestamps": list(hit["timestamps"]),
            })
        else:
            region = regions[-1]
            region["end"] = max(region["end"], hit["end"])
            region["scores"].append(hit["score"])
            region["evidence"].extend(hit["evidence"])
            region["timestamps"].extend(hit["timestamps"])

    for region in regions:
        region["start"] = max(0.0, region["start"] - padding)
        region["end"] = region["end"] + padding
        if video_duration is not None:
            region["end"] = min(float(video_duration), region["end"])

        region["score"] = max(region["scores"])
        region["fusion_mass"] = sum(region["scores"])
        region["num_supporting_hits"] = len(region["scores"])

    # Rank occurrences
    regions.sort(
        key=lambda x: (x["score"], x["fusion_mass"], x["num_supporting_hits"]),
        reverse=True
    )

    for i, region in enumerate(regions):
        region["candidate_id"] = i

    return regions

def merge_fused_bins(
    fused,
    bin_size=5,
    max_gap=10,
    top_n=None,
    padding=10,
    video_duration=None,
    relative_score_floor=0.05,
):
    return cluster_fused_results(
        fused=fused,
        bin_size=bin_size,
        max_gap=max_gap,
        padding=padding,
        video_duration=video_duration,
        relative_score_floor=relative_score_floor,
        max_bins=top_n,
    )
