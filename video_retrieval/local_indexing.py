"""Build original hierarchical/multimodal resources with local models."""
import hashlib
import json
from pathlib import Path

from .local_backend import LocalModels, check_runtime, use_local


def prepare_video(source, models=None, progress=lambda message: None):
    from .video import split_video_hierarchically
    from .embeddings import embed_all_scales, load_multiscale_video_index
    from .metadata import generate_metadata, load_jsonl, embed_metadata_records, build_metadata_faiss, load_metadata_index
    from .transcript import transcribe_video, create_audio_chunks, create_transcript_windows, embed_transcript_windows, build_transcript_faiss, load_transcript_index
    from .pipeline import RetrievalResources, VideoRetrievalPipeline
    import av

    models = models or LocalModels()
    digests = check_runtime(models)
    source = Path(source).resolve()
    signature = models.signature()
    signature['model_digests'] = digests
    signature['source_size'] = source.stat().st_size
    signature['source_modified'] = source.stat().st_mtime_ns
    # Uploaded source folders are content-addressed; settings get their own indexes.
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    folder = source.parent / 'architecture' / key
    folder.mkdir(parents=True, exist_ok=True)
    ready = folder / 'ready.json'
    visual_dir, metadata_dir, transcript_dir = (folder / name for name in ('visual', 'metadata_index', 'transcript_index'))
    with use_local(models, progress):
        progress('Building coarse, medium, and fine temporal chunks')
        manifest = split_video_hierarchically(source, output_dir=folder / 'chunks', export_clips=False)
        if ready.exists():
            saved = json.loads(ready.read_text(encoding='utf-8'))
            if saved['models'] != signature:
                raise ValueError('Index model configuration mismatch. Rebuild local indexes.')
            progress('Loading saved local multimodal indexes')
            visual, visual_metadata = load_multiscale_video_index(visual_dir)
            metadata, metadata_records = load_metadata_index(metadata_dir)
            if saved['has_transcript']:
                transcript, bm25, transcript_metadata = load_transcript_index(transcript_dir)
            else:
                transcript, bm25, transcript_metadata = None, None, []
        else:
            progress('Embedding all hierarchical video scales locally')
            visual, visual_metadata = embed_all_scales(manifest, save_dir=visual_dir,
                                                       cache_dir=folder / 'embedding_clips')
            for scale, rows in visual.metadata_by_scale.items():
                if len(rows) != len(manifest['chunks'][scale]):
                    raise RuntimeError(f'Incomplete {scale} embeddings. Retry processing.')
            progress('Generating scene metadata using the local vision model')
            metadata_path = folder / 'metadata.jsonl'
            generate_metadata(manifest, output_path=metadata_path, video_cache=folder / 'metadata_clips')
            rows = load_jsonl(metadata_path)
            if len(rows) != len(manifest['chunks']['medium']):
                raise RuntimeError('Some scene descriptions failed. Retry processing to resume.')
            vectors, metadata_records = embed_metadata_records(rows, save_dir=metadata_dir)
            if len(metadata_records) != len(rows):
                raise RuntimeError('Some metadata embeddings failed. Retry processing.')
            metadata = build_metadata_faiss(vectors, save_path=metadata_dir / 'medium_metadata.faiss')
            progress('Transcribing speech and building semantic and BM25 indexes')
            with av.open(str(source)) as container:
                has_audio = bool(container.streams.audio)
            transcript, bm25, transcript_metadata = None, None, []
            if has_audio:
                words, _ = transcribe_video(manifest, output_dir=folder / 'transcript')
                expected = create_audio_chunks(manifest, output_dir=folder / 'transcript' / 'audio_chunks')
                raw = (folder / 'transcript' / 'raw_transcriptions.jsonl').read_text(encoding='utf-8')
                completed = {json.loads(line)['chunk_id'] for line in raw.splitlines() if line.strip()}
                if {row['chunk_id'] for row in expected} - completed:
                    raise RuntimeError('Some audio chunks failed transcription. Retry processing to resume.')
                windows = [w for w in create_transcript_windows(words, manifest['video']['duration']) if w['text'].strip()]
                if windows:
                    vectors, transcript_metadata = embed_transcript_windows(windows, save_dir=transcript_dir)
                    if len(transcript_metadata) != len(windows):
                        raise RuntimeError('Some transcript embeddings failed. Retry processing.')
                    build_transcript_faiss(vectors, save_path=transcript_dir / 'transcript.faiss')
                    transcript, bm25, transcript_metadata = load_transcript_index(transcript_dir)
            ready.write_text(json.dumps({'models': signature, 'has_transcript': bool(transcript_metadata)}), encoding='utf-8')
        progress('Original retrieval pipeline ready with local models')
        return VideoRetrievalPipeline(RetrievalResources(
            manifest=manifest, video_index=visual, video_metadata=visual_metadata,
            metadata_index=metadata, metadata_records=metadata_records,
            transcript_index=transcript, transcript_bm25=bm25, transcript_metadata=transcript_metadata,
            local_models=models,
        ))
