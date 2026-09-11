"""Local model adapters for the original pipeline, scoped to one operation."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
import base64
import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import requests


@dataclass(frozen=True)
class LocalModels:
    planner: str = 'qwen2.5vl:3b'
    vision: str = 'qwen2.5vl:3b'
    verifier: str = 'qwen2.5vl:3b'
    frame_limit: int = 12

    def signature(self):
        return {'version': 1, 'embedding': 'clip-vit-base-patch32-mean-v1',
                'transcription': 'faster-whisper-base', **asdict(self)}


_models = ContextVar('local_models', default=None)
_progress = ContextVar('local_progress', default=None)
URL = 'http://127.0.0.1:11434'


def active():
    return _models.get() is not None


def model_label(original):
    if not active():
        return original
    from .config import PRO_VERIFIER_MODEL
    models = _models.get()
    return models.verifier if original == PRO_VERIFIER_MODEL else models.vision


@contextmanager
def use_local(models=None, progress=None):
    token = _models.set(models or LocalModels())
    progress_token = _progress.set(progress)
    try:
        yield
    finally:
        _progress.reset(progress_token)
        _models.reset(token)


def notify(message):
    if _progress.get():
        _progress.get()(message)


def check_runtime(models):
    try:
        response = requests.get(URL + '/api/tags', timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError('Start Ollama first: run .\\start-local.ps1 in another terminal.') from exc
    available = {row['name'] for row in response.json().get('models', [])}
    missing = {models.planner, models.vision, models.verifier} - available
    if missing:
        raise RuntimeError('Download missing Ollama models: ' + ', '.join(sorted(missing)))


def chat_json(prompt, schema, images=None, role='vision'):
    models = _models.get() or LocalModels()
    name = getattr(models, role)
    notify(f'Local {role}: {name}')
    message = {'role': 'user', 'content': prompt + '\nReturn only JSON matching this schema:\n' + json.dumps(schema)}
    if images:
        message['images'] = images
    response = requests.post(URL + '/api/chat', json={
        'model': name, 'messages': [message], 'format': schema, 'stream': False,
        'options': {'temperature': 0, 'num_ctx': 16384, 'num_predict': 4096},
        'keep_alive': '30m',
    }, timeout=1800)
    response.raise_for_status()
    result = json.loads(response.json()['message']['content'])
    jsonschema.validate(result, schema)
    return result


def sampled_frames(path, limit):
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f'Cannot decode video: {path}')
        duration = count / fps
        times = np.linspace(0, max(0, duration - 1 / fps), min(limit, max(1, int(np.ceil(duration)))))
        frames, sampled = [], []
        for second in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
                sampled.append(float(second))
        if not frames:
            raise ValueError('No frames decoded.')
        return frames, sampled
    finally:
        cap.release()


def image_base64(frame):
    height, width = frame.shape[:2]
    if max(height, width) > 768:
        frame = cv2.resize(frame, (int(width * 768 / max(height, width)), int(height * 768 / max(height, width))))
    ok, data = cv2.imencode('.jpg', frame)
    if not ok:
        raise ValueError('Unable to encode frame')
    return base64.b64encode(data).decode('ascii')


def video_json(path, prompt, schema, role='vision'):
    frames, times = sampled_frames(path, (_models.get() or LocalModels()).frame_limit)
    evidence = '\nChronological sampled frames, seconds relative to this clip: ' + json.dumps(times)
    speech = transcribe(path)
    evidence += '\nClip-relative speech transcript: ' + json.dumps(speech['segments'])
    evidence += '\nFrames are sparse samples; do not invent unseen actions between frames.'
    return chat_json(prompt + evidence, schema, [image_base64(frame) for frame in frames], role)


def image_json(images, prompt, schema):
    return chat_json(prompt, schema, [image_base64(frame) for frame in images])


def embed_text(text):
    from local_search import encode
    # Preserve coverage of long metadata/transcripts beyond CLIP's context window.
    words = str(text).split()
    pieces = [' '.join(words[i:i + 40]) for i in range(0, len(words), 40)] or ['']
    vector = np.mean([encode(text=piece)[0] for piece in pieces], axis=0)
    return (vector / max(np.linalg.norm(vector), 1e-8)).astype(np.float32)


def embed_video(path):
    from PIL import Image
    from local_search import encode
    frames, _ = sampled_frames(path, (_models.get() or LocalModels()).frame_limit)
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    vector = np.mean(encode(images=images), axis=0)
    return (vector / max(np.linalg.norm(vector), 1e-8)).astype(np.float32)


@lru_cache(maxsize=1)
def whisper():
    from faster_whisper import WhisperModel
    from local_search import ROOT
    kwargs = dict(device='cpu', compute_type='int8', download_root=str(ROOT / 'models'))
    try:
        return WhisperModel('base', local_files_only=True, **kwargs)
    except (OSError, ValueError):
        return WhisperModel('base', **kwargs)


def transcribe(path, language=None, word_timestamps=True):
    path = Path(path).resolve()
    return _transcribe(str(path), path.stat().st_mtime_ns, language, word_timestamps)


@lru_cache(maxsize=128)
def _transcribe(path, modified, language, word_timestamps):
    import av
    with av.open(path) as container:
        if not container.streams.audio:
            return {'words': [], 'segments': [], 'text': ''}
    notify('Local speech transcription')
    segments, _ = whisper().transcribe(path, language=language, word_timestamps=word_timestamps,
                                      beam_size=1, vad_filter=True)
    rows, words = [], []
    for segment in segments:
        rows.append({'start': segment.start, 'end': segment.end, 'text': segment.text})
        words.extend({'start': w.start, 'end': w.end, 'word': w.word} for w in (segment.words or []))
    return {'words': words, 'segments': rows, 'text': ' '.join(row['text'] for row in rows)}
