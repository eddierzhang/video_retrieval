#Video chunking and utilities for temporal search.
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm


#Get video duration in seconds using ffprobe.
def get_video_duration(video_path):
    video_path = Path(video_path)

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())

# Generate multiple overlapping [start, end] windows.
def generate_windows(
    video_duration,
    window_size,
    stride,
    min_final_fraction=0.25,
):
    windows = []
    start = 0.0

    while start < video_duration:
        end = min(start + window_size, video_duration)
        duration = end - start

        # Skip a tiny leftover window at the end
        if duration < window_size * min_final_fraction:
            break

        windows.append(
            (round(start, 3), round(end, 3))
        )

        if end >= video_duration:
            break

        start += stride

    return windows

#Calculates overalp between two clips
def temporal_overlap(a_start, a_end, b_start, b_end):
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )

#Creates a video chunk of specified duration and start time
def create_clip(
    video_path,
    output_path,
    start,
    duration,
    reencode=True,
):

    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if reencode:
        command = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "128k",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

    else:
        command = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(output_path),
        ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


    """
    Create hierarchical temporal windows:

        coarse -> medium -> fine

    Returns
    -------
    manifest : dict
        Complete hierarchy and timestamp information, saved as JSON.
    """
    
def split_video_hierarchically(
    video_path,
    output_dir="video_chunks",

    coarse_window=120,
    coarse_stride=60,

    medium_window=30,
    medium_stride=15,

    fine_window=8,
    fine_stride=4,

    export_clips=False,
    reencode=True,
    save_manifest=True,
):

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    duration = get_video_duration(video_path)

    print(f"Video: {video_path.name}")
    print(f"Duration: {duration:.2f} sec")
    print(f"Duration: {duration / 60:.2f} min")
    print()

    scales = {
        "coarse": {
            "window": coarse_window,
            "stride": coarse_stride,
        },

        "medium": {
            "window": medium_window,
            "stride": medium_stride,
        },

        "fine": {
            "window": fine_window,
            "stride": fine_stride,
        },
    }

    chunks = {}
    #Generate info about each video chunk 
    for scale_name, config in scales.items():

        windows = generate_windows(
            video_duration=duration,
            window_size=config["window"],
            stride=config["stride"],
        )

        chunks[scale_name] = []

        print(
            f"{scale_name}: {len(windows)} chunks"
        )

        for i, (start, end) in enumerate(windows):

            chunk_id = f"{scale_name}_{i:06d}"

            filename = (
                f"{chunk_id}_"
                f"{start:.3f}_"
                f"{end:.3f}.mp4"
            )

            relative_path = (
                Path(scale_name) / filename
            )

            chunk = {
                "chunk_id": chunk_id,
                "scale": scale_name,
                "index": i,

                "start": start,
                "end": end,
                "duration": end - start,

                "relative_path": str(relative_path),

                "parent_ids": [],
                "child_ids": [],
            }

            chunks[scale_name].append(chunk)

            # Optional physical chunk creation
            if export_clips:

                print(
                    f"\rCreating {scale_name}: "
                    f"{i + 1}/{len(windows)}",
                    end="",
                )

                create_clip(
                    video_path=video_path,
                    output_path=output_dir / relative_path,
                    start=start,
                    duration=end - start,
                    reencode=reencode,
                )

        if export_clips:
            print()
    #Hierarchy enables searching from coarse to fine
    hierarchy_levels = [
        ("coarse", "medium"),
        ("medium", "fine"),
    ]

    for parent_scale, child_scale in hierarchy_levels:

        parents = chunks[parent_scale]
        children = chunks[child_scale]

        for child in children:

            for parent in parents:

                overlap = temporal_overlap(
                    child["start"],
                    child["end"],
                    parent["start"],
                    parent["end"],
                )

                if overlap > 0:

                    child["parent_ids"].append(
                        parent["chunk_id"]
                    )

                    parent["child_ids"].append(
                        child["chunk_id"]
                    )

    #output saved data
    manifest = {
        "video": {
            "path": str(video_path.resolve()),
            "filename": video_path.name,
            "duration": duration,
        },

        "scales": scales,

        "chunks": chunks,
    }

    if save_manifest:

        manifest_path = (
            output_dir / "manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2
            )

        print()
        print(
            f"Manifest saved to: {manifest_path}"
        )

    return manifest

#Converts chunk into a video file only when needed for model processing.
def materialize_chunk(
    manifest,
    chunk,
    output_dir="clip_cache",
):
    """
    Create an MP4 only when a retrieved chunk
    actually needs to be sent to a model.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir /
        f"{chunk['chunk_id']}.mp4"
    )

    if output_path.exists():
        return output_path

    create_clip(
        video_path=manifest["video"]["path"],
        output_path=output_path,
        start=chunk["start"],
        duration=chunk["duration"],
        reencode=True,
    )

    return output_path


#Creates clip for Gemini Embedding Model
def materialize_embedding_clip(
    manifest,
    chunk,
    output_dir="embedding_cache",
    width=640,
    crf=24,
):
    """
    Create a lightweight video specifically for Gemini Embedding 2.

    - 1 FPS because Gemini Embedding 2 samples <=32s clips at 1 FPS
    - removes audio because Gemini Embedding 2 ignores video audio
    - scales video to <=640 px wide
    - H.264 compression
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        f"{chunk['chunk_id']}_embed.mp4"
    )

    # Reuse cached file
    if output_path.exists():
        return output_path

    source_video = manifest["video"]["path"]

    start = chunk["start"]
    duration = chunk["duration"]

    command = [
        "ffmpeg",
        "-y",

        # Seek
        "-ss", str(start),

        "-i", str(source_video),

        # Clip duration
        "-t", str(duration),

        # Gemini only needs sparse frames anyway
        "-vf",
        f"fps=1,scale='min({width},iw)':-2",

        # Remove audio
        "-an",

        # Compress
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),

        # Widely compatible output
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        str(output_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )

    size_mb = output_path.stat().st_size / (1024 ** 2)

    print(
        f"{chunk['chunk_id']}: "
        f"{size_mb:.2f} MB "
        f"(~{size_mb * 4/3:.2f} MB base64)"
    )

    return output_path

#Creates clip for metadata extraction/captioning
def materialize_metadata_clip(
    manifest,
    chunk,
    output_dir="metadata_video_cache",
    fps=4,
    width=768,
    crf=25,
):
    """
    Create a lightweight but motion-preserving clip
    for VLM metadata extraction.

    This is different from the 1-FPS embedding proxy.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir /
        f"{chunk['chunk_id']}_metadata.mp4"
    )

    # Cache
    if output_path.exists():
        return output_path

    source_video = manifest["video"]["path"]

    command = [
        "ffmpeg",
        "-y",

        "-ss", str(chunk["start"]),
        "-i", str(source_video),

        "-t", str(chunk["duration"]),

        "-vf",
        f"fps={fps},scale={width}:-2:force_original_aspect_ratio=decrease",

        # Audio retrieval will be a different module
        "-an",

        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(crf),

        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode(
                "utf-8",
                errors="ignore"
            )
        )

    return output_path



def _safe_stem(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return text[:80] or "video"

#Specific Video Path for different time segments to ensure no duplicates
def interval_cache_key(manifest, start, end, prefix="interval"):
    """Stable cache key that cannot collide across queries or source videos."""
    video_path = str(Path(manifest["video"]["path"]).resolve())
    video_hash = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:8]
    video_stem = _safe_stem(Path(video_path).stem)
    start_ms = int(round(float(start) * 1000))
    end_ms = int(round(float(end) * 1000))
    return f"{prefix}_{video_stem}_{video_hash}_{start_ms:010d}_{end_ms:010d}"

#Extract an original-quality candidate interval with query-safe caching
def extract_candidate_clip(
    manifest,
    candidate,
    output_dir="candidate_clips",
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = float(candidate["start"])
    end = float(candidate["end"])
    key = interval_cache_key(manifest, start, end, prefix="candidate")
    output_path = output_dir / f"{key}.mp4"

    if output_path.exists():
        return output_path

    source = manifest["video"]["path"]
    duration = max(0.001, end - start)

    command = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))

    return output_path

# Extract debugging frames from a candidate interval.
def extract_candidate_frames(
    manifest,
    candidate,
    output_dir="candidate_frames",
    fps=1.0,
):
    import cv2

    video_path = manifest["video"]["path"]
    start = float(candidate["start"])
    end = float(candidate["end"])

    key = interval_cache_key(manifest, start, end, prefix="candidate")
    candidate_dir = Path(output_dir) / key
    candidate_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    step = 1.0 / float(fps)
    timestamps = np.arange(start, end, step)
    frames = []

    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
        success, frame = cap.read()
        if not success:
            continue

        filename = candidate_dir / f"frame_{timestamp:.3f}.jpg"
        if not filename.exists():
            cv2.imwrite(str(filename), frame)

        frames.append({
            "timestamp": float(timestamp),
            "path": str(filename),
        })

    cap.release()
    return frames


# Materialize candidate clips/frames for manual inspection.
def materialize_candidates(
    manifest,
    candidates,
    top_k=None,
    frame_fps=1.0,
):
    selected = candidates if top_k is None else candidates[:top_k]
    output = []

    for candidate in tqdm(selected, desc="Materializing candidates"):
        candidate = candidate.copy()

        clip_path = extract_candidate_clip(manifest, candidate)
        frames = extract_candidate_frames(
            manifest,
            candidate,
            fps=frame_fps,
        )

        candidate["clip_path"] = str(clip_path)
        candidate["frames"] = frames
        output.append(candidate)

    return output

# Local video file into base64 URL for OpenRouter API 
def video_to_data_url(video_path):
    video_path = Path(video_path)
    mime_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
    }
    suffix = video_path.suffix.lower()
    if suffix not in mime_types:
        raise ValueError(f"Unsupported video format: {suffix}")
    encoded = base64.b64encode(video_path.read_bytes()).decode("utf-8")
    return f"data:{mime_types[suffix]};base64,{encoded}"

#Extracts verifier video
def materialize_vlm_clip(
    manifest,
    start,
    end,
    output_dir="vlm_clip_cache",
    fps=4,
    width=768,
    crf=26,
    include_audio=True,
    max_raw_mb=5.5,
):

    start = max(0.0, float(start))
    end = min(float(manifest["video"]["duration"]), float(end))
    if end <= start:
        raise ValueError("Invalid VLM interval")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = manifest["video"]["path"]
    key = interval_cache_key(manifest, start, end, prefix="vlm")
    output_path = output_dir / f"{key}.mp4"

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 ** 2)
        if size_mb <= max_raw_mb:
            return output_path

    # Progressively cheaper encodes if a long/complex clip is too large.
    attempts = [
        (fps, width, crf),
        (min(fps, 3), min(width, 640), max(crf, 28)),
        (min(fps, 2), min(width, 512), max(crf, 30)),
        (1, min(width, 448), max(crf, 32)),
    ]

    last_size = None

    for attempt_fps, attempt_width, attempt_crf in attempts:
        vf = (
            f"fps={attempt_fps},"
            f"scale='min({int(attempt_width)},iw)':-2"
        )

        command = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{end - start:.3f}",
            "-map", "0:v:0",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", str(attempt_crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]

        if include_audio:
            command += [
                "-map", "0:a?",
                "-c:a", "aac",
                "-b:a", "64k",
            ]
        else:
            command += ["-an"]

        command.append(str(output_path))

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))

        last_size = output_path.stat().st_size / (1024 ** 2)
        if last_size <= max_raw_mb:
            return output_path

    raise RuntimeError(
        f"Verifier clip is still {last_size:.2f} MB after compression. "
        "Reduce verifier_window_seconds or max_raw_mb."
    )

#Extract one clip from the original video 
def extract_final_clip(
    manifest,
    match,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = float(match["start"])
    end = float(match["end"])
    key = interval_cache_key(manifest, start, end, prefix="match")
    output_path = output_dir / f"{key}.mp4"

    if output_path.exists():
        return output_path

    source = manifest["video"]["path"]
    command = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))

    return output_path

#Extract matching frames from the original video 
def extract_match_frames(
    manifest,
    match,
    output_dir,
    fps=4.0,
    jpeg_quality=95,
    max_frames=None,
):

    import cv2

    video_path = manifest["video"]["path"]
    start = float(match["start"])
    end = float(match["end"])

    key = interval_cache_key(manifest, start, end, prefix="match")
    frame_dir = Path(output_dir) / key
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0

    target_fps = source_fps if fps is None else float(fps)
    target_fps = max(0.01, target_fps)

    duration = max(0.0, end - start)
    n = max(1, int(math.floor(duration * target_fps)) + 1)
    timestamps = np.linspace(start, end, n, endpoint=True)

    if max_frames is not None and len(timestamps) > max_frames:
        indices = np.linspace(0, len(timestamps) - 1, max_frames).astype(int)
        timestamps = timestamps[indices]

    frames = []

    for timestamp in timestamps:
        timestamp = float(timestamp)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()
        if not success:
            continue

        frame_index = int(round(timestamp * source_fps))
        filename = frame_dir / f"frame_{timestamp:.3f}_{frame_index:09d}.jpg"

        if not filename.exists():
            cv2.imwrite(
                str(filename),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
            )

        frames.append({
            "timestamp": timestamp,
            "frame_index": frame_index,
            "path": str(filename),
        })

    cap.release()
    return frames

#Create final clips and matching frames 
def materialize_final_matches(
    manifest,
    matches,
    query,
    output_root="final_results",
    frame_fps=4.0,
    max_frames_per_match=None,
):

    query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    query_dir = Path(output_root) / query_hash
    clips_dir = query_dir / "clips"
    frames_dir = query_dir / "frames"
    query_dir.mkdir(parents=True, exist_ok=True)

    output = []

    for i, match in enumerate(sorted(matches, key=lambda x: x["start"])):
        item = match.copy()
        item["match_id"] = i
        item["start_timestamp"] = format_timestamp_precise(item["start"])
        item["end_timestamp"] = format_timestamp_precise(item["end"])

        clip_path = extract_final_clip(manifest, item, clips_dir)
        frames = extract_match_frames(
            manifest,
            item,
            frames_dir,
            fps=frame_fps,
            max_frames=max_frames_per_match,
        )

        item["clip_path"] = str(clip_path)
        item["frames"] = frames
        output.append(item)

    result_file = query_dir / "results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {"query": query, "num_matches": len(output), "matches": output},
            f,
            indent=2,
        )

    return output, result_file

def format_timestamp_precise(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"
