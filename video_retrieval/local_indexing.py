"""Build or load the hierarchical multimodal indexes for one video with local models."""
import hashlib
import json
from pathlib import Path

from .local_backend import LocalModels, check_runtime, stage, use_models

# Stage messages reported while an index builds, in order.
INDEX_STAGES = [
    "Splitting video into coarse, medium, and fine chunks",
    "Embedding frames at every temporal scale",
    "Transcribing speech",
    "Describing scenes with the vision model",
    "Indexing scene descriptions",
]


def index_key(source, models):
    source = Path(source).resolve()
    signature = {**models.index_signature(), "source_size": source.stat().st_size, "source_modified": source.stat().st_mtime_ns}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16], signature


def prepare_video(source, models=None, index_root=None, reporter=None, cancel_event=None):
    """Build the index for `source` (resuming saved scene descriptions) and return a pipeline.

    Indexes live under `index_root/<settings hash>/`; an existing complete index is
    loaded instead of rebuilt. Only settings that change index contents are hashed,
    so the planner and verifier models can change without re-indexing.
    """
    from .embeddings import embed_all_scales
    from .metadata import build_metadata_faiss, embed_metadata_records, generate_metadata, load_jsonl
    from .transcript import build_transcript_faiss, create_transcript_windows, embed_transcript_windows, transcribe_video
    from .video import split_video_hierarchically

    models = models or LocalModels()
    source = Path(source).resolve()
    key, signature = index_key(source, models)
    folder = Path(index_root or source.parent / "index") / key
    if (folder / "ready.json").exists():
        return load_index(folder, models, source)

    with use_models(models, reporter, cancel_event):
        digests = check_runtime(models)
        folder.mkdir(parents=True, exist_ok=True)

        stage(INDEX_STAGES[0])
        manifest = split_video_hierarchically(source, output_dir=folder / "chunks", export_clips=False)

        stage(INDEX_STAGES[1])
        embed_all_scales(manifest, save_dir=folder / "visual")

        stage(INDEX_STAGES[2])
        words, segments = transcribe_video(manifest, output_dir=folder / "transcript")
        windows = [w for w in create_transcript_windows(words, manifest["video"]["duration"]) if w["text"].strip()]
        if windows:
            vectors, transcript_metadata = embed_transcript_windows(windows, save_dir=folder / "transcript_index")
            build_transcript_faiss(vectors, save_path=folder / "transcript_index" / "transcript.faiss")

        stage(INDEX_STAGES[3])
        metadata_path = folder / "metadata.jsonl"
        failures = generate_metadata(manifest, output_path=metadata_path)
        if failures:
            raise RuntimeError(
                f"{len(failures)} scene descriptions failed (first error: {failures[0]['error']}). "
                "Retry indexing to resume from the completed scenes."
            )
        rows = load_jsonl(metadata_path)

        stage(INDEX_STAGES[4])
        vectors, records = embed_metadata_records(rows, save_dir=folder / "metadata_index")
        build_metadata_faiss(vectors, save_path=folder / "metadata_index" / "medium_metadata.faiss")

        stats = {
            "duration": manifest["video"]["duration"],
            "chunks": {scale: len(chunks) for scale, chunks in manifest["chunks"].items()},
            "scene_records": len(records),
            "speech_segments": len(segments),
            "transcript_windows": len(windows),
        }
        (folder / "ready.json").write_text(json.dumps({
            "models": signature, "model_digests": digests,
            "has_transcript": bool(windows), "stats": stats,
        }), encoding="utf-8")
    return load_index(folder, models, source)


def load_index(folder, models=None, source=None):
    """Load a complete index folder into a pipeline; `source` overrides the saved video path."""
    from .embeddings import load_multiscale_video_index
    from .metadata import load_metadata_index
    from .pipeline import RetrievalResources, VideoRetrievalPipeline
    from .transcript import load_transcript_index

    folder = Path(folder)
    saved = json.loads((folder / "ready.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "chunks" / "manifest.json").read_text(encoding="utf-8"))
    if source is not None:
        manifest["video"]["path"] = str(Path(source).resolve())
    visual, visual_metadata = load_multiscale_video_index(folder / "visual")
    metadata, metadata_records = load_metadata_index(folder / "metadata_index")
    if saved["has_transcript"]:
        transcript, bm25, transcript_metadata = load_transcript_index(folder / "transcript_index")
    else:
        transcript, bm25, transcript_metadata = None, None, []
    return VideoRetrievalPipeline(RetrievalResources(
        manifest=manifest, video_index=visual, video_metadata=visual_metadata,
        metadata_index=metadata, metadata_records=metadata_records,
        transcript_index=transcript, transcript_bm25=bm25, transcript_metadata=transcript_metadata,
        local_models=models or LocalModels(),
    ))
