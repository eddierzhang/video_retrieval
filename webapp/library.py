"""Video library on disk: uploaded sources, records, thumbnails, previews, and saved searches.

Layout under local_data/library/<video id>/:
    video.json          record shown in the UI
    source.<ext>        the upload, content-addressed by its SHA-256
    thumbnail.jpg
    preview.mp4         only for codecs browsers cannot play
    index/<key>/        hierarchical indexes built by video_retrieval.prepare_video
    searches/<id>/      search.json plus extracted clips and frames
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

from video_retrieval.config import DATA_DIR
from video_retrieval.local_backend import Cancelled
from video_retrieval.video import probe_video

LIBRARY_DIR = DATA_DIR / "library"
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}
BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov", ".mkv", ".webm"}
BROWSER_CODECS = {"h264", "vp8", "vp9", "av1"}
VIDEO_ID = re.compile(r"^[0-9a-f]{16}$")
SEARCH_ID = re.compile(r"^[0-9a-f]{12}$")


class LibraryError(Exception):
    status = 400


class NotFound(LibraryError):
    status = 404


class Conflict(LibraryError):
    status = 409


def write_json(path, data):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(data, default=_json_default), encoding="utf-8")
    # On Windows a replace fails while another thread is reading the file (the browser polls searches
    # every second), so wait out the read rather than failing the request.
    for attempt in range(20):
        try:
            os.replace(partial, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class Library:
    def __init__(self, root=LIBRARY_DIR):
        self.root = Path(root)
        (self.root / "_incoming").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- videos

    def video_dir(self, video_id):
        folder = self.root / str(video_id)
        if not VIDEO_ID.match(str(video_id)) or not (folder / "video.json").is_file():
            raise NotFound("Video not found.")
        return folder

    def get(self, video_id):
        return json.loads((self.video_dir(video_id) / "video.json").read_text(encoding="utf-8"))

    def list(self):
        records = []
        for path in self.root.glob("*/video.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda record: record.get("created_at", 0), reverse=True)

    def update(self, video_id, **fields):
        with self._lock:
            record = self.get(video_id)
            record.update(fields)
            write_json(self.video_dir(video_id) / "video.json", record)
            return record

    def recover_interrupted(self):
        """Jobs do not survive a restart; mark their videos so the UI can offer a retry."""
        for record in self.list():
            if record.get("status") in ("queued", "indexing"):
                self.update(
                    record["id"],
                    status="ready" if record.get("index") else "interrupted",
                    error=None if record.get("index") else "Indexing stopped when the app closed. Resume to continue where it left off.",
                )
            for search in self.list_searches(record["id"]):
                if search.get("status") in ("queued", "running"):
                    self.update_search(record["id"], search["id"], status="failed", error="The app closed before this search finished.")

    def incoming_file(self):
        return self.root / "_incoming" / f"{uuid.uuid4().hex}.part"

    def ingest(self, temp_path, digest, filename):
        """Move a fully received upload into the library. Returns (record, created)."""
        temp_path = Path(temp_path)
        name = Path(str(filename).replace("\\", "/")).name.strip()[:200] or "Untitled video"
        suffix = Path(name).suffix.lower()
        try:
            if suffix not in VIDEO_EXTENSIONS:
                raise LibraryError(f"Unsupported file type. Upload one of: {', '.join(sorted(VIDEO_EXTENSIONS))}.")
            video_id = digest[:16]
            with self._lock:
                folder = self.root / video_id
                if (folder / "video.json").is_file():
                    return self.get(video_id), False
                try:
                    info = probe_video(temp_path)
                except (subprocess.CalledProcessError, ValueError, OSError):
                    raise LibraryError("That file is not a readable video.")
                if not info["width"] or not math.isfinite(info["duration"]) or info["duration"] <= 0:
                    raise LibraryError("That file has no playable video stream.")
                folder.mkdir(parents=True, exist_ok=True)
                source = folder / f"source{suffix}"
                os.replace(temp_path, source)
                record = {
                    "id": video_id,
                    "name": name,
                    "source": source.name,
                    "size": source.stat().st_size,
                    "duration": info["duration"],
                    "width": info["width"],
                    "height": info["height"],
                    "video_codec": info["video_codec"],
                    "audio_codec": info["audio_codec"],
                    "playable": suffix in BROWSER_CONTAINERS and info["video_codec"] in BROWSER_CODECS,
                    "created_at": time.time(),
                    "status": "uploaded",
                    "error": None,
                    "index": None,
                }
                self._thumbnail(source, folder / "thumbnail.jpg", info["duration"])
                write_json(folder / "video.json", record)
                return record, True
        finally:
            temp_path.unlink(missing_ok=True)

    def _thumbnail(self, source, target, duration):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{min(duration * 0.1, 10.0):.2f}", "-i", str(source),
             "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def delete(self, video_id):
        folder = self.video_dir(video_id)
        with self._lock:
            for attempt in range(5):
                try:
                    shutil.rmtree(folder)
                    return
                except PermissionError:
                    # Windows keeps a file locked briefly after a media response closes.
                    if attempt == 4:
                        raise Conflict("The video is still in use. Close its player and try again.")
                    time.sleep(0.4)

    def source_path(self, video_id):
        return self.video_dir(video_id) / self.get(video_id)["source"]

    def media_path(self, video_id):
        preview = self.video_dir(video_id) / "preview.mp4"
        return preview if preview.is_file() else self.source_path(video_id)

    def thumbnail_path(self, video_id):
        path = self.video_dir(video_id) / "thumbnail.jpg"
        if not path.is_file():
            raise NotFound("No thumbnail.")
        return path

    def make_preview(self, video_id, report, cancel_event):
        """Transcode a browser-playable H.264 preview for codecs browsers cannot decode."""
        folder = self.video_dir(video_id)
        record = self.get(video_id)
        partial = folder / "preview.partial.mp4"
        command = [
            "ffmpeg", "-y", "-v", "error", "-i", str(folder / record["source"]),
            "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale='min(1280,iw)':-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(partial),
        ]
        total = max(1, int(record["duration"]))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for line in process.stdout:
                if cancel_event.is_set():
                    raise Cancelled("Cancelled")
                key, _, value = line.strip().partition("=")
                if key in ("out_time_us", "out_time_ms") and value.isdigit():
                    report({"task": "Transcoding preview", "done": min(total, int(value) // 1_000_000), "total": total})
            if process.wait() != 0:
                raise RuntimeError("Could not create a browser preview: " + process.stderr.read()[-400:])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        os.replace(partial, folder / "preview.mp4")

    def transcript(self, video_id):
        index = self.get(video_id).get("index")
        path = self.video_dir(video_id) / index["dir"] / "transcript" / "segments.json" if index else None
        if path is None or not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # -------------------------------------------------------------- searches

    def search_dir(self, video_id, search_id):
        folder = self.video_dir(video_id) / "searches" / str(search_id)
        if not SEARCH_ID.match(str(search_id)) or not (folder / "search.json").is_file():
            raise NotFound("Search not found.")
        return folder

    def create_search(self, video_id, fields):
        search_id = uuid.uuid4().hex[:12]
        folder = self.video_dir(video_id) / "searches" / search_id
        folder.mkdir(parents=True)
        record = {"id": search_id, "video_id": video_id, "created_at": time.time(), "error": None, "result": None, **fields}
        write_json(folder / "search.json", record)
        return record

    def get_search(self, video_id, search_id):
        return json.loads((self.search_dir(video_id, search_id) / "search.json").read_text(encoding="utf-8"))

    def update_search(self, video_id, search_id, **fields):
        with self._lock:
            record = self.get_search(video_id, search_id)
            record.update(fields)
            write_json(self.search_dir(video_id, search_id) / "search.json", record)
            return record

    def list_searches(self, video_id):
        rows = []
        for path in (self.video_dir(video_id) / "searches").glob("*/search.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            result = record.get("result") or {}
            rows.append({
                "id": record["id"], "query": record.get("query"), "mode": record.get("mode"),
                "status": record.get("status"), "created_at": record.get("created_at"),
                "elapsed": record.get("elapsed"), "num_matches": result.get("num_matches"),
                "error": record.get("error"),
            })
        return sorted(rows, key=lambda row: row.get("created_at") or 0, reverse=True)

    def delete_search(self, video_id, search_id):
        shutil.rmtree(self.search_dir(video_id, search_id), ignore_errors=True)

    def search_file(self, video_id, search_id, relative):
        root = self.search_dir(video_id, search_id).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.name == "search.json":
            raise NotFound("File not found.")
        return path
