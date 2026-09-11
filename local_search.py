"""Key-free frame retrieval and optional speech transcription. No hosted inference."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent / 'local_data'
MODEL = 'openai/clip-vit-base-patch32'


@lru_cache(maxsize=1)
def clip_model():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    cache = str(ROOT / 'models')
    try:
        processor = CLIPProcessor.from_pretrained(MODEL, cache_dir=cache, local_files_only=True, use_fast=False)
        model = CLIPModel.from_pretrained(MODEL, cache_dir=cache, local_files_only=True).eval()
    except OSError:
        processor = CLIPProcessor.from_pretrained(MODEL, cache_dir=cache, use_fast=False)
        model = CLIPModel.from_pretrained(MODEL, cache_dir=cache).eval()
    return torch, processor, model


def encode(images=None, text=None):
    torch, processor, model = clip_model()
    with torch.inference_mode():
        if images is not None:
            features = model.get_image_features(**processor(images=images, return_tensors='pt'))
        else:
            features = model.get_text_features(**processor(text=[text], return_tensors='pt',
                                                           padding=True, truncation=True))
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return features.cpu().numpy()


def save_upload(upload):
    data = upload.getbuffer()
    folder = ROOT / hashlib.sha256(data).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    # Never use the supplied filename as a filesystem path.
    suffix = Path(upload.name).suffix.lower()
    if suffix not in {'.mp4', '.mov', '.mkv', '.avi', '.webm'}:
        raise ValueError('Unsupported video format.')
    path = folder / ('source' + suffix)
    if not path.exists():
        path.write_bytes(data)
    return path


def process_video(path, interval=3.0, transcribe=False, progress=lambda fraction, message: None):
    import cv2
    from PIL import Image
    path = Path(path)
    if not math.isfinite(interval) or interval < 1:
        raise ValueError('Frame interval must be at least one second.')
    index_path = path.parent / 'index.json'
    vectors_path = path.parent / 'frames.npy'
    data = None
    if index_path.exists() and vectors_path.exists():
        cached = json.loads(index_path.read_text(encoding='utf-8'))
        if cached.get('model') == MODEL and cached.get('interval') == interval:
            data = cached
    if data is None:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened() or fps <= 0 or frames <= 0:
                raise ValueError('The file is not a readable video.')
            duration = frames / fps
            if not math.isfinite(duration) or duration > 7200:
                raise ValueError('This prototype supports videos up to two hours.')
            times, vectors, batch, batch_times = [], [], [], []
            steps = list(np.arange(0, duration, interval))
            progress(0, 'Loading local visual model (first use downloads model files)…')
            for i, second in enumerate(steps):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
                ok, frame = cap.read()
                if ok:
                    batch.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    batch_times.append(float(second))
                if batch and (len(batch) == 8 or i == len(steps) - 1):
                    vectors.append(encode(images=batch))
                    times.extend(batch_times)
                    batch, batch_times = [], []
                    progress((i + 1) / len(steps) * .8, f'Indexed {len(times)} sampled frames')
            if not vectors:
                raise ValueError('No video frames could be decoded.')
            data = {'model': MODEL, 'interval': interval, 'duration': duration,
                    'source': str(path.resolve()), 'times': times, 'segments': [],
                    'transcribed': False}
            np.save(vectors_path, np.concatenate(vectors))
            index_path.write_text(json.dumps(data), encoding='utf-8')
        finally:
            cap.release()
    if transcribe and not data.get('transcribed'):
        import av
        with av.open(str(path)) as container:
            has_audio = bool(container.streams.audio)
        if has_audio:
            from faster_whisper import WhisperModel
            progress(.8, 'Transcribing speech locally (first use downloads Whisper)…')
            try:
                model = WhisperModel('base', device='cpu', compute_type='int8',
                                     download_root=str(ROOT / 'models'), local_files_only=True)
            except (OSError, ValueError):
                model = WhisperModel('base', device='cpu', compute_type='int8',
                                     download_root=str(ROOT / 'models'))
            segments, _ = model.transcribe(str(path), beam_size=1, vad_filter=True)
            rows = []
            for segment in segments:
                rows.append({'start': segment.start, 'end': segment.end, 'text': segment.text})
                progress(min(.99, .8 + .19 * segment.end / data['duration']), 'Transcribing speech…')
            data['segments'] = rows
        data['transcribed'] = True
        data['has_audio'] = has_audio
        index_path.write_text(json.dumps(data), encoding='utf-8')
    progress(1.0, 'Video ready to search')
    return data


def search(data, query, mode='Visual', top_k=5):
    if not query.strip():
        raise ValueError('Enter a search query.')
    if mode == 'Speech':
        terms = set(re.findall(r'\w+', query.lower()))
        rows = []
        for segment in data.get('segments', []):
            words = set(re.findall(r'\w+', segment['text'].lower()))
            score = len(terms & words) / max(1, len(terms))
            if score:
                rows.append({**segment, 'score': score})
        return sorted(rows, key=lambda row: row['score'], reverse=True)[:top_k]
    vectors = np.load(Path(data['source']).parent / 'frames.npy', allow_pickle=False)
    scores = vectors @ encode(text=query)[0]
    rows = []
    for i in np.argsort(-scores):
        second = data['times'][int(i)]
        start, end = max(0, second - 3), min(data['duration'], second + 5)
        if any(start < row['end'] and end > row['start'] for row in rows):
            continue
        rows.append({'start': start, 'end': end, 'score': float(scores[i])})
        if len(rows) >= top_k:
            break
    return rows
