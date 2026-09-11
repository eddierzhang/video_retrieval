"""Local indexed-video loading for the prototype (no indexing or API calls)."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent


def discover_manifests(root=ROOT):
    return sorted(root.glob('*chunks/manifest.json'))


def index_defaults(manifest_path):
    folder = Path(manifest_path).parent
    prefix = folder.name[:-len('chunks')]
    return tuple(str(folder.parent / (prefix + name)) for name in (
        'embedding_indices', 'metadata_index', 'transcript_index'
    ))


def read_manifest(path, video_override=''):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    video = manifest.get('video', {})
    if not video.get('path') or float(video.get('duration', 0)) <= 0:
        raise ValueError('The manifest must contain a video path and positive duration.')
    if 'chunks' not in manifest:
        raise ValueError('The manifest is missing hierarchical chunks.')
    if video_override.strip():
        manifest['video']['path'] = str(Path(video_override).expanduser().resolve())
    return manifest


def load_pipeline(manifest_path, video_path, visual_dir, metadata_dir, transcript_dir):
    from video_retrieval.pipeline import RetrievalResources, VideoRetrievalPipeline
    from video_retrieval.embeddings import load_multiscale_video_index
    from video_retrieval.metadata import load_metadata_index
    from video_retrieval.transcript import load_transcript_index

    manifest = read_manifest(manifest_path, video_path)
    if not Path(manifest['video']['path']).is_file():
        raise FileNotFoundError('Locate the original video using the source video field.')
    visual, visual_metadata = load_multiscale_video_index(save_dir=visual_dir)
    metadata, metadata_records = load_metadata_index(save_dir=metadata_dir)
    transcript, bm25, transcript_metadata = load_transcript_index(save_dir=transcript_dir)
    return VideoRetrievalPipeline(RetrievalResources(
        manifest=manifest, video_index=visual, video_metadata=visual_metadata,
        metadata_index=metadata, metadata_records=metadata_records,
        transcript_index=transcript, transcript_bm25=bm25,
        transcript_metadata=transcript_metadata,
    ))


def timestamp(seconds):
    seconds = max(0, int(float(seconds)))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'
