from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests
from tqdm import tqdm

from .config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OPENROUTER_EMBEDDING_URL,
    get_openrouter_api_key,
)
from .video import materialize_embedding_clip, video_to_data_url

#Embed text with the configured multimodal embedding model through OpenRouter.
def embed_text(text, input_type=None):
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
        "dimensions": EMBEDDING_DIM,
        "encoding_format": "float",
    }
    if input_type is not None:
        payload["input_type"] = input_type

    headers = {
        "Authorization": f"Bearer {get_openrouter_api_key()}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        OPENROUTER_EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"Text embedding failed:\nHTTP {response.status_code}\n{response.text}"
        )

    return np.asarray(
        response.json()["data"][0]["embedding"],
        dtype=np.float32,
    )

#Embed one reconstructed video chunk with the same multimodal model as text.
def embed_video(video_path):
    video_path = Path(video_path)
    raw_size_mb = video_path.stat().st_size / (1024 ** 2)


    print(f"Video size: {raw_size_mb:.2f} MB")

    video_data_url = video_to_data_url(video_path)

    payload = {
        "model": EMBEDDING_MODEL,
        "input": [
            {
                "content": [
                    {
                        "type": "input_video",
                        "input_video": {
                            "data": video_data_url,
                            "format": "mp4",
                        },
                    }
                ]
            }
        ],
        "dimensions": EMBEDDING_DIM,
        "encoding_format": "float",
    }

    headers = {
        "Authorization": f"Bearer {get_openrouter_api_key()}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        OPENROUTER_EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=300,
    )

    if not response.ok:
        raise RuntimeError(
            "Video embedding failed:\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    result = response.json()
    return np.asarray(
        result["data"][0]["embedding"],
        dtype=np.float32,
    )

#Return an L2-normalized float32 vector.
def normalize_embedding(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)

    if norm == 0:
        return x

    return x / norm

#Get the duration of a chunk
def _duration(record: Mapping[str, Any]) -> float:
    if record.get("duration") is not None:
        return max(0.0, float(record["duration"]))
    return max(
        0.0,
        float(record.get("end", 0.0)) - float(record.get("start", 0.0)),
    )

#Order scales from largest to finest, e.g. coarse, medium, fine
def _scale_order_from_metadata(
    metadata_by_scale: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[str]:
    scale_stats = []

    for scale, records in metadata_by_scale.items():
        durations = [_duration(x) for x in records if _duration(x) > 0]
        median = float(np.median(durations)) if durations else 0.0
        scale_stats.append((str(scale), median))

    return [
        scale
        for scale, _ in sorted(
            scale_stats,
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
    ]

#Calculate amount of overlap between two intervals 
def _interval_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return max(
        0.0,
        min(float(a["end"]), float(b["end"]))
        - max(float(a["start"]), float(b["start"])),
    )

#Return candidate parents for a child.
def _best_parent_indices(
    child: Mapping[str, Any],
    parent_records: Sequence[Mapping[str, Any]],
    candidate_parent_indices: Sequence[int],
    padding: float = 0.0,
) -> List[int]:
    explicit = set(str(x) for x in child.get("parent_ids", []) if x is not None)
    if child.get("parent_id") is not None:
        explicit.add(str(child["parent_id"]))

    if explicit:
        matched = [
            idx
            for idx in candidate_parent_indices
            if str(parent_records[idx].get("chunk_id")) in explicit
        ]
        if matched:
            return matched

    start = float(child["start"])
    end = float(child["end"])
    center = (start + end) / 2.0

    contained = []
    overlapping = []

    for idx in candidate_parent_indices:
        parent = parent_records[idx]
        p_start = float(parent["start"]) - float(padding)
        p_end = float(parent["end"]) + float(padding)

        if p_start <= center <= p_end:
            contained.append(idx)

        overlap = max(
            0.0,
            min(end, p_end) - max(start, p_start),
        )
        if overlap > 0:
            overlapping.append((idx, overlap))

    if contained:
        return contained

    overlapping.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in overlapping]

#Infer adjacent parent/child links from timestamps.
def infer_hierarchy_links(
    metadata_by_scale: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    output = {
        scale: [dict(row) for row in records]
        for scale, records in metadata_by_scale.items()
    }

    order = _scale_order_from_metadata(output)

    for scale in order:
        for row in output[scale]:
            row.setdefault("parent_ids", [])
            row.setdefault("child_ids", [])

    for parent_scale, child_scale in zip(order[:-1], order[1:]):
        parents = output[parent_scale]
        children = output[child_scale]

        parent_lookup = {
            str(parent.get("chunk_id")): i
            for i, parent in enumerate(parents)
        }

        for child in children:
            explicit = [
                str(x)
                for x in child.get("parent_ids", [])
                if str(x) in parent_lookup
            ]

            parent_candidates = []

            if explicit:
                parent_candidates = [
                    parent_lookup[parent_id]
                    for parent_id in explicit
                ]
            else:
                overlaps = []
                for i, parent in enumerate(parents):
                    overlap = _interval_overlap(child, parent)
                    if overlap > 0:
                        overlaps.append((i, overlap))

                overlaps.sort(key=lambda x: x[1], reverse=True)

                # Prefer parents containing the child center. If no containment
                # exists, preserve every positively-overlapping parent.
                center = (
                    float(child["start"]) + float(child["end"])
                ) / 2.0
                containing = [
                    i
                    for i, _ in overlaps
                    if float(parents[i]["start"]) <= center <= float(parents[i]["end"])
                ]
                parent_candidates = containing or [i for i, _ in overlaps]

            parent_ids = []
            for parent_idx in parent_candidates:
                parent_id = str(parents[parent_idx].get("chunk_id"))
                if parent_id not in parent_ids:
                    parent_ids.append(parent_id)

                child_id = str(child.get("chunk_id"))
                parent_children = parents[parent_idx].setdefault("child_ids", [])
                if child_id not in parent_children:
                    parent_children.append(child_id)

            child["parent_ids"] = parent_ids

            if parent_candidates:
                best_parent = max(
                    parent_candidates,
                    key=lambda idx: _interval_overlap(child, parents[idx]),
                )
                child["parent_id"] = str(
                    parents[best_parent].get("chunk_id")
                )
            else:
                child["parent_id"] = None

    return output

#Embed every video chunk at one scale and persist vectors + timestamp metadata. Preserves parent/child data from manifest 
def embed_scale(
    manifest,
    scale="medium",
    cache_dir="embedding_video_cache",
    save_dir="embedding_indices",
    retry_count=3,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    chunks = manifest["chunks"][scale]

    embeddings = []
    metadata = []

    for chunk in tqdm(
        chunks,
        desc=f"Embedding {scale} chunks",
    ):
        clip_path = materialize_embedding_clip(
            manifest=manifest,
            chunk=chunk,
            output_dir=cache_dir,
        )

        success = False

        for attempt in range(retry_count):
            try:
                vector = embed_video(clip_path)
                vector = normalize_embedding(vector)
                success = True
                break

            except Exception as exc:
                print(f"\nError on {chunk['chunk_id']}: {exc}")

                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)

        if not success:
            print("Skipping", chunk["chunk_id"])
            continue

        embeddings.append(vector)

        start = float(chunk["start"])
        end = float(chunk["end"])

        metadata.append(
            {
                "chunk_id": chunk["chunk_id"],
                "scale": scale,
                "start": start,
                "end": end,
                "duration": float(chunk.get("duration", end - start)),
                "relative_path": chunk.get("relative_path"),
                "parent_id": chunk.get("parent_id"),
                "parent_ids": list(chunk.get("parent_ids", [])),
                "child_ids": list(chunk.get("child_ids", [])),
            }
        )

    if not embeddings:
        raise RuntimeError(
            f"No {scale!r} chunks were successfully embedded."
        )

    embeddings = np.vstack(embeddings).astype(np.float32)

    np.save(
        save_dir / f"{scale}_embeddings.npy",
        embeddings,
    )

    with open(
        save_dir / f"{scale}_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    return embeddings, metadata

#Build a normalized inner-product FAISS index (cosine similarity).
def build_faiss_index(
    embeddings,
    save_path=None,
):
    import faiss

    vectors = np.asarray(
        embeddings,
        dtype=np.float32,
    ).copy()

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected a 2-D embedding matrix, got {vectors.shape}."
        )

    if len(vectors) == 0:
        raise ValueError("Cannot build an index from zero embeddings.")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_path))

    return index

"""
    FAISS-compatible coarse-to-fine video index.

    It exposes ``ntotal`` and ``search(query_embeddings, k)`` so the rest of the
    retrieval stack can continue treating it like the existing visual FAISS
    index. Internally it:

      1. searches the coarsest chunks globally;
      2. restricts the next scale to children/temporal overlaps of strong parents;
      3. repeats until the finest available scale;
      4. combines direct child similarity with parent-context similarity;
      5. returns indices that point into ``output_metadata`` (the finest scale).
"""
class HierarchicalVideoIndex:

    def __init__(
        self,
        embeddings_by_scale: Mapping[str, np.ndarray],
        metadata_by_scale: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        beam_multiplier: int = 4,
        minimum_beam: int = 24,
        context_weight: float = 0.30,
        parent_padding: float = 1.0,
    ):
        if not embeddings_by_scale:
            raise ValueError("At least one embedding scale is required.")

        if set(embeddings_by_scale) != set(metadata_by_scale):
            raise ValueError(
                "embeddings_by_scale and metadata_by_scale must contain "
                "the same scale names."
            )

        self.embeddings_by_scale: Dict[str, np.ndarray] = {}
        self.metadata_by_scale: Dict[str, List[Dict[str, Any]]] = {}

        expected_dim = None

        for scale, vectors in embeddings_by_scale.items():
            vectors = np.asarray(vectors, dtype=np.float32).copy()

            if vectors.ndim != 2:
                raise ValueError(
                    f"{scale}: expected 2-D embeddings, got {vectors.shape}."
                )

            rows = [dict(x) for x in metadata_by_scale[scale]]

            if len(vectors) != len(rows):
                raise ValueError(
                    f"{scale}: {len(vectors)} vectors but {len(rows)} metadata rows."
                )

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)

            if expected_dim is None:
                expected_dim = vectors.shape[1]
            elif vectors.shape[1] != expected_dim:
                raise ValueError(
                    "All scales must use the same embedding dimension."
                )

            self.embeddings_by_scale[str(scale)] = vectors
            self.metadata_by_scale[str(scale)] = rows

        self.scale_order = _scale_order_from_metadata(
            self.metadata_by_scale
        )
        self.output_scale = self.scale_order[-1]
        self.output_metadata = self.metadata_by_scale[self.output_scale]

        self.beam_multiplier = max(1, int(beam_multiplier))
        self.minimum_beam = max(1, int(minimum_beam))
        self.context_weight = max(
            0.0,
            min(0.95, float(context_weight)),
        )
        self.parent_padding = max(0.0, float(parent_padding))

        self.ntotal = len(self.output_metadata)
        self.d = int(expected_dim or 0)

    #Returns the indexes of the top scoring candidates
    def _top_indices(
        self,
        scores: np.ndarray,
        count: int,
        candidate_indices: Optional[Sequence[int]] = None,
    ) -> List[int]:
        if candidate_indices is None:
            candidate_indices = list(range(len(scores)))
        else:
            candidate_indices = list(candidate_indices)

        if not candidate_indices:
            return []

        count = min(
            max(1, int(count)),
            len(candidate_indices),
        )

        candidate_scores = np.asarray(
            [scores[i] for i in candidate_indices],
            dtype=np.float32,
        )

        if count == len(candidate_indices):
            order = np.argsort(candidate_scores)[::-1]
        else:
            partial = np.argpartition(
                candidate_scores,
                -count,
            )[-count:]
            order = partial[
                np.argsort(candidate_scores[partial])[::-1]
            ]

        return [
            int(candidate_indices[int(i)])
            for i in order
        ]
    #Hierarchical search algorithm, searches one query embedding at every scale 
    def _search_one(
        self,
        query: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = normalize_embedding(query).reshape(-1)

        beam_size = max(
            self.minimum_beam,
            self.beam_multiplier * max(1, int(k)),
        )

        first_scale = self.scale_order[0]
        first_vectors = self.embeddings_by_scale[first_scale]
        first_scores = first_vectors @ query

        selected_indices = self._top_indices(
            first_scores,
            beam_size,
        )

        selected_score_by_id = {
            str(
                self.metadata_by_scale[first_scale][idx].get(
                    "chunk_id",
                    idx,
                )
            ): float(first_scores[idx])
            for idx in selected_indices
        }

        selected_rows = [
            (
                idx,
                float(first_scores[idx]),
                self.metadata_by_scale[first_scale][idx],
            )
            for idx in selected_indices
        ]

        if len(self.scale_order) == 1:
            final_indices = selected_indices[:k]
            return (
                np.asarray(
                    [first_scores[i] for i in final_indices],
                    dtype=np.float32,
                ),
                np.asarray(final_indices, dtype=np.int64),
            )

        for level, scale in enumerate(self.scale_order[1:], start=1):
            parent_scale = self.scale_order[level - 1]
            parent_records = self.metadata_by_scale[parent_scale]
            parent_indices = [idx for idx, _, _ in selected_rows]
            parent_score_by_index = {
                idx: score
                for idx, score, _ in selected_rows
            }

            child_records = self.metadata_by_scale[scale]
            child_vectors = self.embeddings_by_scale[scale]
            direct_scores = child_vectors @ query

            eligible = []
            context_scores: Dict[int, float] = {}

            for child_idx, child in enumerate(child_records):
                matching_parents = _best_parent_indices(
                    child,
                    parent_records,
                    parent_indices,
                    padding=self.parent_padding,
                )

                if not matching_parents:
                    continue

                parent_context = max(
                    parent_score_by_index[parent_idx]
                    for parent_idx in matching_parents
                )

                eligible.append(child_idx)
                context_scores[child_idx] = float(parent_context)

            # Defensive fallback: never allow a malformed hierarchy to make the
            # visual retrieval channel completely empty.
            if not eligible:
                eligible = list(range(len(child_records)))
                context_scores = {
                    idx: 0.0
                    for idx in eligible
                }

            combined_scores = np.full(
                len(child_records),
                -np.inf,
                dtype=np.float32,
            )

            for child_idx in eligible:
                if level == 0:
                    combined = float(direct_scores[child_idx])
                else:
                    combined = (
                        (1.0 - self.context_weight)
                        * float(direct_scores[child_idx])
                        + self.context_weight
                        * float(context_scores[child_idx])
                    )

                combined_scores[child_idx] = combined

            keep_count = (
                max(k, beam_size)
                if scale != self.output_scale
                else max(k, min(beam_size, len(eligible)))
            )

            selected_indices = self._top_indices(
                combined_scores,
                keep_count,
                candidate_indices=eligible,
            )

            selected_rows = [
                (
                    idx,
                    float(combined_scores[idx]),
                    child_records[idx],
                )
                for idx in selected_indices
            ]

        final_scores = np.asarray(
            [score for _, score, _ in selected_rows[:k]],
            dtype=np.float32,
        )
        final_indices = np.asarray(
            [idx for idx, _, _ in selected_rows[:k]],
            dtype=np.int64,
        )

        return final_scores, final_indices

    #Search function meant to mimic FAISS 
    def search(
        self,
        query_embeddings,
        k,
    ):
        queries = np.asarray(
            query_embeddings,
            dtype=np.float32,
        )

        if queries.ndim == 1:
            queries = queries[None, :]

        if queries.ndim != 2:
            raise ValueError(
                f"Expected query embeddings with shape (N, D), got {queries.shape}."
            )

        if queries.shape[1] != self.d:
            raise ValueError(
                f"Query dimension {queries.shape[1]} does not match index dimension {self.d}."
            )

        k = max(0, int(k))

        all_scores = np.full(
            (len(queries), k),
            -np.inf,
            dtype=np.float32,
        )
        all_indices = np.full(
            (len(queries), k),
            -1,
            dtype=np.int64,
        )

        if k == 0 or self.ntotal == 0:
            return all_scores, all_indices

        for query_idx, query in enumerate(queries):
            scores, indices = self._search_one(
                query,
                min(k, self.ntotal),
            )

            n = len(indices)
            all_scores[query_idx, :n] = scores
            all_indices[query_idx, :n] = indices

        return all_scores, all_indices

#Builds hierarchical index from already-existing embeddings 
def build_multiscale_video_index(
    embeddings_by_scale,
    metadata_by_scale,
    **kwargs,
):
    linked_metadata = infer_hierarchy_links(metadata_by_scale)

    index = HierarchicalVideoIndex(
        embeddings_by_scale=embeddings_by_scale,
        metadata_by_scale=linked_metadata,
        **kwargs,
    )

    return index, index.output_metadata

#Embed all scales for a new video 
def embed_all_scales(
    manifest,
    scales=None,
    cache_dir="embedding_video_cache",
    save_dir="embedding_indices",
    retry_count=3,
    build_individual_faiss=True,
    **hierarchy_kwargs,
):
    if not isinstance(manifest.get("chunks"), dict):
        raise TypeError(
            "Hierarchical retrieval requires manifest['chunks'] to be a "
            "dictionary of chunk scales."
        )

    if scales is None:
        scales = list(manifest["chunks"].keys())
    else:
        scales = list(scales)

    if not scales:
        raise ValueError("No chunk scales were supplied.")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    embeddings_by_scale = {}
    metadata_by_scale = {}

    for scale in scales:
        embeddings, metadata = embed_scale(
            manifest=manifest,
            scale=scale,
            cache_dir=cache_dir,
            save_dir=save_dir,
            retry_count=retry_count,
        )

        embeddings_by_scale[scale] = embeddings
        metadata_by_scale[scale] = metadata

        if build_individual_faiss:
            build_faiss_index(
                embeddings,
                save_path=save_dir / f"{scale}.faiss",
            )

    metadata_by_scale = infer_hierarchy_links(
        metadata_by_scale
    )

    # Persist inferred links so future loads do not need the original manifest.
    for scale, metadata in metadata_by_scale.items():
        with open(
            save_dir / f"{scale}_metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

    index = HierarchicalVideoIndex(
        embeddings_by_scale=embeddings_by_scale,
        metadata_by_scale=metadata_by_scale,
        **hierarchy_kwargs,
    )

    config = {
        "scale_order": index.scale_order,
        "output_scale": index.output_scale,
        "beam_multiplier": index.beam_multiplier,
        "minimum_beam": index.minimum_beam,
        "context_weight": index.context_weight,
        "parent_padding": index.parent_padding,
    }

    with open(
        save_dir / "multiscale_index_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    return index, index.output_metadata

#Discover what scales have been saved 
def _discover_saved_scales(save_dir: Path) -> List[str]:
    scales = []

    for metadata_path in save_dir.glob("*_metadata.json"):
        name = metadata_path.name

        if name == "multiscale_index_config.json":
            continue

        scale = name[: -len("_metadata.json")]
        embeddings_path = save_dir / f"{scale}_embeddings.npy"

        if embeddings_path.exists():
            scales.append(scale)

    return sorted(set(scales))

#Load every saved visual scale and rebuild the lightweight hierarchical index.
def load_multiscale_video_index(
    save_dir="embedding_indices",
    scales=None,
    **hierarchy_kwargs,
):
    save_dir = Path(save_dir)

    if scales is None:
        config_path = save_dir / "multiscale_index_config.json"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            scales = config.get("scale_order") or _discover_saved_scales(save_dir)

            # Saved config supplies defaults but explicit function kwargs win.
            for key in (
                "beam_multiplier",
                "minimum_beam",
                "context_weight",
                "parent_padding",
            ):
                if key not in hierarchy_kwargs and key in config:
                    hierarchy_kwargs[key] = config[key]
        else:
            scales = _discover_saved_scales(save_dir)

    scales = list(scales)

    if not scales:
        raise FileNotFoundError(
            f"No saved embedding/metadata scale pairs found in {save_dir}."
        )

    embeddings_by_scale = {}
    metadata_by_scale = {}

    for scale in scales:
        embeddings_path = save_dir / f"{scale}_embeddings.npy"
        metadata_path = save_dir / f"{scale}_metadata.json"

        if not embeddings_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing saved files for scale {scale!r}: "
                f"{embeddings_path.name}, {metadata_path.name}"
            )

        embeddings_by_scale[scale] = np.load(
            embeddings_path
        ).astype(np.float32)

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            metadata_by_scale[scale] = json.load(file)

    linked_metadata = infer_hierarchy_links(
        metadata_by_scale
    )

    index = HierarchicalVideoIndex(
        embeddings_by_scale=embeddings_by_scale,
        metadata_by_scale=linked_metadata,
        **hierarchy_kwargs,
    )

    return index, index.output_metadata

# Search either a normal FAISS index or HierarchicalVideoIndex.
def search_video(
    query,
    index,
    metadata,
    top_k=100,
):
    if top_k <= 0:
        return []

    query_embedding = normalize_embedding(
        embed_text(query)
    )

    query_embedding = query_embedding[
        None, :
    ].astype(np.float32)

    ntotal = int(getattr(index, "ntotal", len(metadata)))
    k = min(int(top_k), ntotal)

    if k <= 0:
        return []

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):
        idx = int(idx)

        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx].copy()
        item["score"] = float(score)

        # Helpful diagnostics when a hierarchical proxy is in use.
        if isinstance(index, HierarchicalVideoIndex):
            item["retrieval_mode"] = "hierarchical_multiscale"
            item["retrieval_output_scale"] = index.output_scale
            item["retrieval_scale_order"] = list(index.scale_order)

        results.append(item)

    return results

#Load new hierarchical index
def load_video_index(
    save_dir="embedding_indices",
    scale="medium",
    **hierarchy_kwargs,
):

    if scale is None or str(scale).lower() in {
        "hierarchical",
        "multiscale",
        "multi",
        "all",
    }:
        return load_multiscale_video_index(
            save_dir=save_dir,
            **hierarchy_kwargs,
        )

    import faiss

    save_dir = Path(save_dir)

    embeddings = np.load(
        save_dir / f"{scale}_embeddings.npy"
    ).astype(np.float32)

    with open(
        save_dir / f"{scale}_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    index_path = save_dir / f"{scale}.faiss"

    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        index = build_faiss_index(
            embeddings,
            save_path=index_path,
        )

    return index, metadata
