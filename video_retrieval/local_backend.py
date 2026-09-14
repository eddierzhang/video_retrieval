"""Local model runtime: Ollama chat, CLIP embeddings, and Whisper transcription.

Every model call in the retrieval pipeline goes through this module. Models run on
this machine; the only network requests go to the loopback Ollama server. Model
inputs are read directly from the source video by time interval, so no temporary
clips are encoded, each frame is embedded once, and speech is transcribed once.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import cv2
import jsonschema
import numpy as np
import requests
from tqdm import tqdm

from .config import (
    ACTION_CLIP_FRAMES,
    ACTION_MODEL,
    CACHE_DIR,
    DETECTOR_MODEL,
    DEFAULT_PLANNER_MODEL,
    DEFAULT_VERIFIER_MODEL,
    DEFAULT_VISION_MODEL,
    FRAME_EMBEDDING_FPS,
    MODEL_CACHE_DIR,
    OLLAMA_URL,
    TEXT_EMBEDDING_MODEL,
    VISUAL_MODEL,
    VISUAL_TILE_GRID,
    WHISPER_MODEL,
)

OLLAMA_DOWN = "The local Ollama server is not running. Start it with .\\start.ps1 (or `ollama serve`)."


@dataclass(frozen=True)
class LocalModels:
    planner: str = DEFAULT_PLANNER_MODEL
    vision: str = DEFAULT_VISION_MODEL
    verifier: str = DEFAULT_VERIFIER_MODEL
    frame_limit: int = 20

    def index_signature(self):
        """Settings that change what a saved index contains (planner/verifier do not)."""
        return {
            "version": 6,
            "embedding": f"{VISUAL_MODEL}@{FRAME_EMBEDDING_FPS:g}fps-mean-{VISUAL_TILE_GRID}x{VISUAL_TILE_GRID}tiles",
            "text_embedding": TEXT_EMBEDDING_MODEL,
            "transcription": f"faster-whisper-{WHISPER_MODEL}",
            "scene_model": self.vision,
            "frame_limit": self.frame_limit,
        }

    def signature(self):
        return {**self.index_signature(), **asdict(self)}


class Cancelled(BaseException):
    """Raised inside a model operation when its job is cancelled.

    Derives from BaseException so the pipeline's per-chunk retry handlers, which
    catch Exception, cannot swallow a cancellation and keep working.
    """


_models = ContextVar("local_models", default=None)
_reporter = ContextVar("local_reporter", default=None)
_cancel = ContextVar("local_cancel", default=None)


def current_models():
    return _models.get() or LocalModels()


@contextmanager
def use_models(models=None, reporter=None, cancel_event=None):
    """Scope model settings, progress reporting, and cancellation to one operation."""
    tokens = (_models.set(models or LocalModels()), _reporter.set(reporter), _cancel.set(cancel_event))
    try:
        yield
    finally:
        _cancel.reset(tokens[2])
        _reporter.reset(tokens[1])
        _models.reset(tokens[0])


def check_cancelled():
    event = _cancel.get()
    if event is not None and event.is_set():
        raise Cancelled("Cancelled")


def report(**event):
    check_cancelled()
    reporter = _reporter.get()
    if reporter is not None:
        reporter(event)


def stage(message):
    report(stage=message)


def track(iterable, desc, total=None):
    """tqdm-style loop that also reports progress and honours cancellation."""
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)
    report(task=desc, done=0, total=total)
    for done, item in enumerate(tqdm(iterable, desc=desc, total=total), 1):
        yield item
        report(task=desc, done=done, total=total)


# ---------------------------------------------------------------- Ollama chat

def installed_models():
    try:
        response = requests.get(OLLAMA_URL + "/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(OLLAMA_DOWN) from exc
    return {row["name"]: row for row in response.json().get("models", [])}


@lru_cache(maxsize=64)
def capabilities(name):
    """Ollama capability names such as 'vision' and 'thinking' (empty if unknown)."""
    try:
        response = requests.post(OLLAMA_URL + "/api/show", json={"model": name}, timeout=30)
        response.raise_for_status()
        return frozenset(response.json().get("capabilities") or [])
    except (requests.RequestException, ValueError):
        return frozenset()


def installed_names(installed=None):
    """Installed model names, plus the bare name for every ':latest' tag.

    Ollama reports an untagged pull as "name:latest", so a bare name in settings
    would otherwise look missing.
    """
    installed = installed_models() if installed is None else installed
    names = set(installed)
    names.update(name[: -len(":latest")] for name in installed if name.endswith(":latest"))
    return names


def check_runtime(models=None):
    models = models or current_models()
    installed = installed_models()
    roles = {"planner": models.planner, "vision": models.vision, "verifier": models.verifier}
    missing = sorted({*roles.values(), TEXT_EMBEDDING_MODEL} - installed_names(installed))
    if missing:
        raise RuntimeError("Download missing Ollama models: " + ", ".join(f"ollama pull {name}" for name in missing))
    for role in ("vision", "verifier"):
        caps = capabilities(roles[role])
        if caps and "vision" not in caps:
            raise RuntimeError(f"{roles[role]} cannot read images. Choose a vision model for the {role} role.")
    return {name: installed[name].get("digest", "") for name in set(roles.values())}


def model_name(role):
    return getattr(current_models(), role)


def chat_json(prompt, schema, images=None, role="vision"):
    name = model_name(role)
    report(model=name, role=role)
    message = {
        "role": "user",
        "content": prompt + "\nReturn only compact JSON matching this schema. Do not add whitespace padding or commentary:\n" + json.dumps(schema),
    }
    if images:
        message["images"] = images
    payload = {
        "model": name,
        "messages": [message],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 32768, "num_predict": 4096},
        "keep_alive": "30m",
    }
    if "thinking" in capabilities(name):
        # Hybrid reasoning models spend minutes thinking before structured output otherwise.
        payload["think"] = False
    for attempt in range(2):
        check_cancelled()
        try:
            response = requests.post(OLLAMA_URL + "/api/chat", json=payload, timeout=1800)
        except requests.ConnectionError as exc:
            raise RuntimeError(OLLAMA_DOWN) from exc
        if response.status_code == 404:
            raise RuntimeError(f"Ollama model {name} is not installed. Run: ollama pull {name}")
        response.raise_for_status()
        try:
            result = json.loads(response.json()["message"]["content"])
            jsonschema.validate(result, schema)
            return result
        except (ValueError, KeyError, jsonschema.ValidationError) as exc:
            if attempt:
                raise RuntimeError(f"Local {role} model {name} returned invalid or incomplete JSON. Try a stronger model or fewer frames.") from exc
            message["content"] += "\nYour last attempt was invalid. Return one complete, concise JSON object, with short arrays and no padding."


# ------------------------------------------------------------- frame inputs

def sample_frames(video_path, start, end, limit):
    """Evenly spaced BGR frames from [start, end) of a video, with absolute times."""
    start, end = max(0.0, float(start)), float(end)
    duration = max(0.0, end - start)
    # Up to 4 fps so short boundary-refinement windows still show the transition.
    count = max(1, min(int(limit), math.ceil(duration * 4)))
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {video_path}")
        frames, times = [], []
        for i in range(count):
            second = start + (i + 0.5) * duration / count
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
                times.append(second)
        if not frames:
            raise ValueError(f"No frames decoded between {start:.2f}s and {end:.2f}s.")
        return frames, times
    finally:
        cap.release()


def image_base64(frame, max_side=768):
    height, width = frame.shape[:2]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Unable to encode frame")
    return base64.b64encode(data).decode("ascii")


def interval_json(video_path, start, end, prompt, schema, role="vision", include_speech=True):
    """Ask a vision model about one interval using sampled frames and its speech."""
    frames, times = sample_frames(video_path, start, end, current_models().frame_limit)
    evidence = "\nChronological sampled frames, in seconds from the start of this clip: " + json.dumps([round(t - start, 2) for t in times])
    if include_speech:
        speech = [
            {"start": round(max(0.0, row["start"] - start), 2), "end": round(min(end, row["end"]) - start, 2), "text": row["text"].strip()}
            for row in speech_between(video_path, start, end)
        ]
        evidence += "\nSpeech transcript for this clip, on the same clock: " + json.dumps(speech)
    evidence += "\nFrames are sparse samples; do not invent actions between frames."
    return chat_json(prompt + evidence, schema, [image_base64(frame) for frame in frames], role)


def images_json(images, prompt, schema, role="vision"):
    return chat_json(prompt, schema, [image_base64(image, max_side=1024) for image in images], role)


# --------------------------------------------------------------------- CLIP

@lru_cache(maxsize=1)
def visual_model():
    """The image-text model used for frame and visual-query embeddings."""
    import torch
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = dict(cache_dir=str(MODEL_CACHE_DIR))
    try:
        processor = AutoProcessor.from_pretrained(VISUAL_MODEL, local_files_only=True, **kwargs)
        model = AutoModel.from_pretrained(VISUAL_MODEL, local_files_only=True, dtype=dtype, **kwargs)
    except OSError:
        processor = AutoProcessor.from_pretrained(VISUAL_MODEL, **kwargs)
        model = AutoModel.from_pretrained(VISUAL_MODEL, dtype=dtype, **kwargs)
    return torch, processor, model.to(device).eval(), device, dtype


def encode(images=None, text=None):
    """L2-normalized visual-model features for RGB images or one or more strings."""
    torch, processor, model, device, dtype = visual_model()
    with torch.inference_mode():
        if images is not None:
            pixels = processor(images=list(images), return_tensors="pt")["pixel_values"]
            features = model.get_image_features(pixel_values=pixels.to(device, dtype))
        else:
            texts = [text] if isinstance(text, str) else list(text)
            # SigLIP is trained with a fixed 64-token padding; anything else degrades it.
            padding = {"padding": "max_length", "max_length": 64} if "siglip" in VISUAL_MODEL else {"padding": True}
            inputs = processor(text=texts, return_tensors="pt", truncation=True, **padding)
            features = model.get_text_features(**{key: value.to(device) for key, value in inputs.items()})
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return features.cpu().numpy()


def visual_logit_scale():
    """The temperature the visual model was trained with, for softmax over competing texts.

    Its sigmoid pair probabilities are not used: on tight crops scored against short phrases they
    sit near zero for every crop, matching or not. A softmax over mutually exclusive descriptions
    separates crops cleanly instead.
    """
    torch, _, model, _, _ = visual_model()
    return float(model.logit_scale.exp().detach().float().cpu())


def embed_text(text):
    # Mean-pool 40-word pieces so long text is not truncated by the model's context.
    words = str(text).split()
    pieces = [" ".join(words[i:i + 40]) for i in range(0, len(words), 40)] or [""]
    vector = encode(text=pieces).mean(axis=0)
    return (vector / max(np.linalg.norm(vector), 1e-8)).astype(np.float32)


# nomic-embed asks for a task prefix; other models take the text unchanged.
_TASK_PREFIXES = {"document": "search_document: ", "query": "search_query: "}


def _embed_text_model(text, task):
    prefix = _TASK_PREFIXES[task] if TEXT_EMBEDDING_MODEL.startswith("nomic-embed") else ""
    report(model=TEXT_EMBEDDING_MODEL, role="text_embedding")
    try:
        response = requests.post(OLLAMA_URL + "/api/embed",
                                 json={"model": TEXT_EMBEDDING_MODEL, "input": prefix + str(text), "truncate": True},
                                 timeout=300)
    except requests.ConnectionError as exc:
        raise RuntimeError(OLLAMA_DOWN) from exc
    if response.status_code == 404:
        raise RuntimeError(f"Ollama model {TEXT_EMBEDDING_MODEL} is not installed. Run: ollama pull {TEXT_EMBEDDING_MODEL}")
    response.raise_for_status()
    vectors = response.json().get("embeddings") or []
    if not vectors:
        raise RuntimeError(f"{TEXT_EMBEDDING_MODEL} returned no embedding.")
    vector = np.asarray(vectors[0], dtype=np.float32)
    return vector / max(np.linalg.norm(vector), 1e-8)


def embed_document(text):
    """Embed a scene description or transcript window for its index."""
    return _embed_text_model(text, "document")


def embed_query(text):
    """Embed a search query for the scene and transcript indexes."""
    return _embed_text_model(text, "query")


def _file_key(path, *parts):
    path = Path(path).resolve()
    stat = path.stat()
    raw = json.dumps([str(path), stat.st_size, stat.st_mtime_ns, *parts])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def frame_views(frame, grid=VISUAL_TILE_GRID):
    """The whole frame followed by its grid x grid tiles."""
    views = [frame]
    height, width = frame.shape[:2]
    for row in range(grid):
        for col in range(grid):
            views.append(frame[row * height // grid:(row + 1) * height // grid,
                               col * width // grid:(col + 1) * width // grid])
    return views


def frame_embeddings(video_path, fps=FRAME_EMBEDDING_FPS):
    """CLIP embeddings for each frame sampled at `fps`, shaped (frames, views, dim)."""
    key = _file_key(video_path, VISUAL_MODEL, fps, VISUAL_TILE_GRID)
    return _frame_embeddings(str(Path(video_path).resolve()), key, float(fps))


@lru_cache(maxsize=4)
def _frame_embeddings(path, key, fps):
    from .video import probe_video

    cache = CACHE_DIR / "frame_embeddings" / f"{key}.npz"
    if cache.exists():
        with np.load(cache) as data:
            return data["times"], data["vectors"]
    info = probe_video(path)
    if not info["width"] or not info["height"]:
        raise ValueError(f"No video stream in {path}")
    # Decode large enough that every tile still has a full 224px short side.
    scale = 224 * VISUAL_TILE_GRID / min(info["width"], info["height"])
    width = max(2, round(info["width"] * scale / 2) * 2)
    height = max(2, round(info["height"] * scale / 2) * 2)
    frame_bytes = width * height * 3
    command = [
        "ffmpeg", "-v", "error", "-i", path, "-an", "-sn",
        "-vf", f"fps={fps},scale={width}:{height}:flags=area",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    vectors, batch = [], []
    expected = max(1, math.ceil(info["duration"] * fps))
    try:
        for data in track(iter(lambda: process.stdout.read(frame_bytes), b""), "Embedding video frames", total=expected):
            if len(data) < frame_bytes:
                break
            batch.append(np.frombuffer(data, np.uint8).reshape(height, width, 3))
            if len(batch) == 8:
                vectors.append(_encode_frame_views(batch))
                batch = []
        if batch:
            vectors.append(_encode_frame_views(batch))
        if process.wait() != 0 and not vectors:
            raise RuntimeError(f"FFmpeg could not decode {path}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if not vectors:
        raise ValueError(f"No frames decoded from {path}")
    vectors = np.concatenate(vectors).astype(np.float32)
    times = (np.arange(len(vectors)) / fps).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f"{key}.partial.npz")
    np.savez(partial, times=times, vectors=vectors)
    os.replace(partial, cache)
    return times, vectors


def _encode_frame_views(frames):
    """Embed each frame's whole-image and tile views: (frames, views, dim)."""
    views = [view for frame in frames for view in frame_views(frame)]
    features = encode(images=views)
    return features.reshape(len(frames), -1, features.shape[-1])


def decode_frames(video_path, start, end, fps, max_side=800):
    """RGB frames from [start, end) at `fps`, decoded in order so any container works: [(time, frame)].

    OpenCV cannot seek reliably inside WebM, so this reads through FFmpeg instead.
    """
    from .video import probe_video

    info = probe_video(video_path)
    if not info["width"] or not info["height"]:
        raise ValueError(f"No video stream in {video_path}")
    scale = min(1.0, max_side / max(info["width"], info["height"]))
    width = max(2, round(info["width"] * scale / 2) * 2)
    height = max(2, round(info["height"] * scale / 2) * 2)
    start = max(0.0, float(start))
    duration = max(0.0, float(end) - start)
    if duration <= 0:
        return []
    command = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video_path),
        "-an", "-sn", "-vf", f"fps={fps},scale={width}:{height}:flags=area",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    frame_bytes = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    try:
        for index, data in enumerate(iter(lambda: process.stdout.read(frame_bytes), b"")):
            if len(data) < frame_bytes:
                break
            check_cancelled()
            frames.append((start + index / fps, np.frombuffer(data, np.uint8).reshape(height, width, 3)))
        process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return frames


@lru_cache(maxsize=1)
def detector_model():
    """The open-vocabulary object detector used by Detect search."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = dict(cache_dir=str(MODEL_CACHE_DIR))
    try:
        processor = AutoProcessor.from_pretrained(DETECTOR_MODEL, local_files_only=True, **kwargs)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL, local_files_only=True, **kwargs)
    except OSError:
        processor = AutoProcessor.from_pretrained(DETECTOR_MODEL, **kwargs)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL, **kwargs)
    return torch, processor, model.to(device).eval(), device


def detect_objects(frames, phrases, box_threshold=0.3, text_threshold=0.25, batch_size=4):
    """Boxes for every instance of any phrase in each RGB frame.

    Returns one list per frame of {"box": [x0, y0, x1, y1] as fractions of the frame, "score", "label"}.
    """
    from PIL import Image

    torch, processor, model, device = detector_model()
    report(model=DETECTOR_MODEL, role="detector")
    # Grounding DINO reads a lower-case prompt of phrases, each ending in a full stop.
    prompt = " ".join(f"{phrase.strip().lower().rstrip('.')}." for phrase in phrases if phrase.strip())
    output = []
    for begin in range(0, len(frames), batch_size):
        check_cancelled()
        batch = frames[begin:begin + batch_size]
        images = [Image.fromarray(frame) for frame in batch]
        inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[image.size[::-1] for image in images])
        for image, result in zip(images, results):
            width, height = image.size
            labels = result.get("text_labels", result.get("labels"))
            boxes = []
            for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), labels):
                x0, y0, x1, y1 = box
                boxes.append({"box": [max(0.0, x0 / width), max(0.0, y0 / height), min(1.0, x1 / width), min(1.0, y1 / height)],
                              "score": float(score), "label": str(label)})
            output.append(boxes)
    return output


@lru_cache(maxsize=1)
def action_model():
    """The video-text model that scores motion over short clips rather than still frames."""
    import torch
    from transformers import XCLIPModel, XCLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = dict(cache_dir=str(MODEL_CACHE_DIR))
    try:
        processor = XCLIPProcessor.from_pretrained(ACTION_MODEL, local_files_only=True, **kwargs)
        model = XCLIPModel.from_pretrained(ACTION_MODEL, local_files_only=True, **kwargs)
    except OSError:
        processor = XCLIPProcessor.from_pretrained(ACTION_MODEL, **kwargs)
        model = XCLIPModel.from_pretrained(ACTION_MODEL, **kwargs)
    return torch, processor, model.to(device).eval(), device


def action_probabilities(clips, texts, batch_size=4):
    """For each clip of ACTION_CLIP_FRAMES RGB frames, a softmax over `texts`: (clips, texts).

    X-CLIP conditions each text on the clip it is compared with, so the whole comparison runs
    through the model rather than as a dot product of separately computed embeddings.
    """
    torch, processor, model, device = action_model()
    report(model=ACTION_MODEL, role="action")
    rows = []
    for begin in range(0, len(clips), batch_size):
        check_cancelled()
        batch = [list(clip)[:ACTION_CLIP_FRAMES] for clip in clips[begin:begin + batch_size]]
        # The tokenizer and the video processor are called separately: the combined processor
        # silently drops `videos=` in recent transformers releases.
        inputs = dict(processor.tokenizer(list(texts), return_tensors="pt", padding=True))
        inputs["pixel_values"] = processor.image_processor(images=batch, return_tensors="pt")["pixel_values"]
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits_per_video
        rows.append(logits.softmax(dim=-1).float().cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, len(texts)))


def embed_video_interval(video_path, start, end):
    """Mean embedding per view over the cached frames inside [start, end): (views, dim)."""
    times, vectors = frame_embeddings(video_path)
    inside = (times >= float(start)) & (times < float(end))
    if inside.any():
        views = vectors[inside].mean(axis=0)
    else:
        views = vectors[int(np.argmin(np.abs(times - (float(start) + float(end)) / 2)))]
    views = np.atleast_2d(views)
    return (views / np.clip(np.linalg.norm(views, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


# ------------------------------------------------------------------ Whisper

def _expose_torch_cuda_libraries():
    # ctranslate2 needs cuBLAS 12; the CUDA 12 PyTorch wheel ships it in torch/lib.
    try:
        import torch
    except ImportError:
        return
    library = Path(torch.__file__).parent / "lib"
    if library.is_dir() and str(library) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(library) + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def whisper():
    """(model, device): CUDA when its libraries load, otherwise int8 on the CPU."""
    import ctranslate2
    from faster_whisper import WhisperModel

    _expose_torch_cuda_libraries()
    options = [("cpu", "int8")]
    if ctranslate2.get_cuda_device_count() > 0:
        options.insert(0, ("cuda", "float16"))
    last_error = None
    for device, compute_type in options:
        kwargs = dict(device=device, compute_type=compute_type, download_root=str(MODEL_CACHE_DIR))
        try:
            try:
                model = WhisperModel(WHISPER_MODEL, local_files_only=True, **kwargs)
            except (OSError, ValueError):
                model = WhisperModel(WHISPER_MODEL, **kwargs)
            if device == "cuda":
                # CUDA libraries load lazily; a one-second warmup proves they work.
                segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
                list(segments)
            return model, device
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load Whisper: {last_error}")


def transcribe_media(video_path, language=None):
    """Whole-file transcript with absolute word and segment timestamps, cached on disk."""
    key = _file_key(video_path, WHISPER_MODEL, language or "auto")
    return _transcribe_media(str(Path(video_path).resolve()), key, language)


@lru_cache(maxsize=8)
def _transcribe_media(path, key, language):
    import av

    cache = CACHE_DIR / "transcripts" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    with av.open(path) as container:
        has_audio = bool(container.streams.audio)
    result = {"has_audio": has_audio, "language": language, "words": [], "segments": []}
    if has_audio:
        model, device = whisper()
        segments, info = model.transcribe(
            path, language=language, word_timestamps=True, vad_filter=True,
            beam_size=5 if device == "cuda" else 1,
        )
        total = max(1, int(math.ceil(info.duration)))
        result["language"] = info.language
        report(task="Transcribing speech", done=0, total=total)
        for segment in segments:
            result["segments"].append({"start": segment.start, "end": segment.end, "text": segment.text.strip()})
            result["words"].extend(
                {"start": word.start, "end": word.end, "word": word.word.strip()}
                for word in (segment.words or [])
            )
            report(task="Transcribing speech", done=min(total, int(segment.end)), total=total)
        report(task="Transcribing speech", done=total, total=total)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result), encoding="utf-8")
    return result


def speech_between(video_path, start, end):
    return [
        row for row in transcribe_media(video_path)["segments"]
        if row["end"] > float(start) and row["start"] < float(end)
    ]


def devices():
    """Which accelerator each local model uses (loaded lazily, so may be unknown)."""
    output = {"clip": None, "whisper": None, "gpu": None}
    try:
        import torch

        output["clip"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            output["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    if whisper.cache_info().currsize:
        output["whisper"] = whisper()[1]
    return output
