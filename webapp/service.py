"""Coordinates the video library, background jobs, model settings, and loaded indexes."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import threading
import time

from video_retrieval.config import DATA_DIR
from video_retrieval.detect_search import DETECT_STAGES
from video_retrieval.local_backend import LocalModels, capabilities, check_runtime, devices, installed_models, installed_names, use_models
from video_retrieval.config import TEXT_EMBEDDING_MODEL
from video_retrieval.local_indexing import INDEX_STAGES, IndexVersionMismatch, index_key, load_index, prepare_video
from video_retrieval.pipeline import SEARCH_STAGES

from .jobs import Job, JobQueue
from .library import Conflict, Library, LibraryError, write_json

PREVIEW_STAGE = "Preparing browser preview"
VERIFICATION_STAGES = {
    "Verifying candidates with the vision model",
    "Confirming matches with the verifier model",
    "Refining clip boundaries",
}
TEXT_STAGES = ["Planning query", "Reading visible text", "Extracting matching clips"]
MAX_EVIDENCE_POINTS = 720


class Service:
    def __init__(self, library=None, jobs=None, settings_path=DATA_DIR / "settings.json"):
        self.library = library or Library()
        self.jobs = jobs or JobQueue()
        self.settings_path = Path(settings_path)
        self._pipelines = OrderedDict()
        self._lock = threading.Lock()
        self.library.recover_interrupted()

    # -------------------------------------------------------------- settings

    def settings(self):
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            return LocalModels(**{key: data[key] for key in ("planner", "vision", "verifier", "frame_limit") if key in data})
        except (OSError, ValueError, TypeError):
            return LocalModels()

    def save_settings(self, data):
        current = asdict(self.settings())
        for key in ("planner", "vision", "verifier"):
            if key in data:
                value = str(data[key]).strip()
                if not value or len(value) > 120:
                    raise LibraryError(f"Choose a valid {key} model.")
                current[key] = value
        if "frame_limit" in data:
            try:
                limit = int(data["frame_limit"])
            except (TypeError, ValueError):
                raise LibraryError("Frames per model call must be a whole number.")
            if not 4 <= limit <= 32:
                raise LibraryError("Frames per model call must be between 4 and 32.")
            current["frame_limit"] = limit
        models = LocalModels(**current)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.settings_path, asdict(models))
        return models

    def system(self):
        models = self.settings()
        status = {
            "settings": asdict(models),
            "ollama": {"running": False, "models": [], "error": None},
            "missing_models": [],
            "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
            "devices": devices(),
        }
        try:
            installed = installed_models()
        except RuntimeError as exc:
            status["ollama"]["error"] = str(exc)
            return status
        status["ollama"]["running"] = True
        status["ollama"]["models"] = [
            {
                "name": name,
                "size": row.get("size", 0),
                "parameter_size": (row.get("details") or {}).get("parameter_size"),
                "capabilities": sorted(capabilities(name)),
            }
            for name, row in sorted(installed.items())
        ]
        status["missing_models"] = sorted({models.planner, models.vision, models.verifier, TEXT_EMBEDDING_MODEL} - installed_names(installed))
        return status

    # ---------------------------------------------------------------- videos

    def video_payload(self, record):
        settings = self.settings()
        index = record.get("index") or {}
        built_with = index.get("models") or {}
        return {
            **record,
            "media_url": f"/api/videos/{record['id']}/media",
            "thumbnail_url": f"/api/videos/{record['id']}/thumbnail",
            "has_preview": (self.library.root / record["id"] / "preview.mp4").is_file(),
            "jobs": [job.to_dict() for job in self.jobs.active(record["id"])],
            "index_outdated": bool(built_with) and (
                built_with.get("scene_model") != settings.vision
                or built_with.get("frame_limit") != settings.frame_limit
                or built_with.get("version") != settings.index_signature()["version"]
            ),
        }

    def videos(self):
        return [self.video_payload(record) for record in self.library.list()]

    def video(self, video_id):
        payload = self.video_payload(self.library.get(video_id))
        payload["searches"] = self.library.list_searches(video_id)
        return payload

    def delete_video(self, video_id):
        for job in self.jobs.active(video_id):
            self.jobs.cancel(job.id)
            if job.status == "running":
                raise Conflict("Stopping this video's running job. Try deleting again in a moment.")
        with self._lock:
            for key in [key for key in self._pipelines if key[0] == video_id]:
                del self._pipelines[key]
        self.library.delete(video_id)

    # -------------------------------------------------------------- indexing

    def start_index(self, video_id):
        record = self.library.get(video_id)
        existing = self.jobs.active(video_id, kind="index")
        if existing:
            return existing[0]
        models = self.settings()
        stages = INDEX_STAGES if record["playable"] else [PREVIEW_STAGE, *INDEX_STAGES]
        self.library.update(video_id, status="ready" if record.get("index") else "queued", error=None)
        job = Job("index", video_id, record["name"], stages,
                  run=lambda job: self._run_index(job, models), on_finish=self._index_finished)
        return self.jobs.submit(job)

    def _run_index(self, job, models):
        folder = self.library.video_dir(job.video_id)
        record = self.library.get(job.video_id)
        if not record.get("index"):
            self.library.update(job.video_id, status="indexing")
        if not record["playable"] and not (folder / "preview.mp4").is_file():
            job.report({"stage": PREVIEW_STAGE})
            self.library.make_preview(job.video_id, job.report, job.cancel_event)
        source = folder / record["source"]
        pipeline = prepare_video(source, models, index_root=folder / "index",
                                 reporter=job.report, cancel_event=job.cancel_event)
        key, _ = index_key(source, models)
        ready = json.loads((folder / "index" / key / "ready.json").read_text(encoding="utf-8"))
        index = {
            "dir": f"index/{key}", "models": ready["models"], "stats": ready.get("stats", {}),
            "has_transcript": ready["has_transcript"], "built_at": time.time(),
        }
        self.library.update(job.video_id, status="ready", error=None, index=index)
        self._remember_pipeline((job.video_id, index["dir"]), pipeline)

    def _index_finished(self, job):
        if job.status == "done":
            return
        try:
            record = self.library.get(job.video_id)
        except LibraryError:
            return  # deleted while the job ran
        if record.get("index"):
            # A failed re-index leaves the previous index searchable.
            self.library.update(job.video_id, status="ready", error=job.error)
        else:
            status = "cancelled" if job.status == "cancelled" else "failed"
            self.library.update(job.video_id, status=status, error=job.error)

    # ------------------------------------------------------------- searching

    def start_search(self, video_id, query, mode="verified", options=None):
        record = self.library.get(video_id)
        if not record.get("index"):
            raise LibraryError("This video needs to finish indexing before it can be searched.")
        built_version = (record["index"].get("models") or {}).get("version")
        if built_version != self.settings().index_signature()["version"]:
            raise LibraryError("Re-index this video before searching: it was built by an earlier version of the pipeline.")
        query = " ".join(str(query or "").split())
        if not query:
            raise LibraryError("Describe what you are looking for.")
        if len(query) > 500:
            raise LibraryError("Keep the query under 500 characters.")
        if mode not in ("verified", "quick", "detect"):
            raise LibraryError("Search mode must be verified, quick or detect.")
        options = self._search_options(mode, options or {})
        search = self.library.create_search(video_id, {"query": query, "mode": mode, "options": options, "status": "queued"})
        if mode == "detect":
            stages = list(DETECT_STAGES)
        else:
            stages = [
                stage for stage in SEARCH_STAGES
                if not (stage in VERIFICATION_STAGES and (mode == "quick" or (stage == "Refining clip boundaries" and not options["refine_boundaries"])))
            ]
        job = Job(
            "search", video_id, query, stages,
            run=lambda job: self._run_search(job, search["id"]),
            on_finish=lambda job: self._search_finished(job, search["id"]),
            alternate_stages={"Reading visible text": TEXT_STAGES},
        )
        job.result = {"search_id": search["id"]}
        self.library.update_search(video_id, search["id"], job_id=job.id)
        return self.jobs.submit(job)

    @staticmethod
    def _search_options(mode, raw):
        def number(key, default, low, high, cast=float):
            try:
                value = cast(raw.get(key, default))
            except (TypeError, ValueError):
                raise LibraryError(f"Invalid value for {key}.")
            if not (math.isfinite(value) and low <= value <= high):
                raise LibraryError(f"{key} must be between {low} and {high}.")
            return value

        options = {
            "refine_boundaries": bool(raw.get("refine_boundaries", True)),
            "min_confidence": number("min_confidence", 0.25, 0.0, 1.0),
            "max_candidates": number("max_candidates", 12 if mode == "quick" else 20, 1, 50, int),
        }
        if mode == "detect":
            # How many frames the detector may examine, best-ranked shots first.
            options["max_frames"] = number("max_frames", 400, 50, 3000, int)
        return options

    def _run_search(self, job, search_id):
        search = self.library.update_search(job.video_id, search_id, status="running", started_at=time.time())
        models = self.settings()
        check_runtime(models)
        pipeline = self._pipeline(job.video_id)
        pipeline.resources.local_models = models
        options = search["options"]
        output_root = self.library.search_dir(job.video_id, search_id)
        if search["mode"] == "detect":
            from video_retrieval.detect_search import detect_search

            with use_models(models, job.report, job.cancel_event):
                result = detect_search(search["query"], pipeline.resources, output_root,
                                       max_frames=options.get("max_frames", 400), max_frames_per_match=8)
            finished = time.time()
            self.library.update_search(job.video_id, search_id, status="done", result=result,
                                       finished_at=finished, elapsed=finished - search["started_at"])
            return
        result = pipeline.retrieve(
            search["query"],
            reporter=job.report,
            cancel_event=job.cancel_event,
            run_verification=search["mode"] == "verified",
            refine_boundaries=options["refine_boundaries"],
            pro_min_confidence=options["min_confidence"],
            max_candidates=options["max_candidates"],
            output_root=str(self.library.search_dir(job.video_id, search_id)),
            include_evidence_map=True,
            final_frame_fps=1.0,
            max_frames_per_match=8,
        )
        finished = time.time()
        self.library.update_search(job.video_id, search_id, status="done", result=result,
                                   finished_at=finished, elapsed=finished - search["started_at"])

    def set_feedback(self, video_id, search_id, match_key, label):
        """Mark one Detect result right ("positive"), wrong ("negative"), or clear the mark (None)."""
        record = self.library.get_search(video_id, search_id)
        if record.get("mode") != "detect":
            raise LibraryError("Only Detect searches learn from feedback.")
        if label not in ("positive", "negative", None):
            raise LibraryError("Feedback must be positive, negative or empty.")
        keys = {match.get("match_key") for match in (record.get("result") or {}).get("matches", [])}
        known = set((record.get("feedback") or {}).keys()) | keys
        if match_key not in known:
            raise LibraryError("That result is not part of this search.")
        feedback = dict(record.get("feedback") or {})
        if label is None:
            feedback.pop(match_key, None)
        else:
            feedback[match_key] = label
        self.library.update_search(video_id, search_id, feedback=feedback)
        return self.search(video_id, search_id)

    def start_refine(self, video_id, search_id):
        """Re-score a Detect search with its feedback, without running the detector again."""
        record = self.library.get_search(video_id, search_id)
        if record.get("mode") != "detect" or record.get("status") != "done":
            raise LibraryError("Only a finished Detect search can be refined.")
        if not record.get("feedback"):
            raise LibraryError("Mark at least one result right or wrong first.")
        if not (self.library.search_dir(video_id, search_id) / "detect_state.pkl").is_file():
            raise LibraryError("This search has no saved state to refine; run it again.")
        self.library.update_search(video_id, search_id, status="queued", error=None)
        job = Job("search", video_id, record["query"], ["Assembling results", "Extracting matching clips"],
                  run=lambda job: self._run_refine(job, search_id),
                  on_finish=lambda job: self._refine_finished(job, search_id))
        job.result = {"search_id": search_id}
        self.library.update_search(video_id, search_id, job_id=job.id)
        return self.jobs.submit(job)

    def _run_refine(self, job, search_id):
        from video_retrieval.detect_search import refine

        search = self.library.update_search(job.video_id, search_id, status="running")
        started = time.time()
        pipeline = self._pipeline(job.video_id)
        with use_models(self.settings(), job.report, job.cancel_event):
            job.report({"stage": "Assembling results"})
            result = refine(self.library.search_dir(job.video_id, search_id), search.get("feedback") or {},
                            pipeline.resources, max_frames_per_match=8)
        refinements = int(search.get("refinements") or 0) + 1
        self.library.update_search(job.video_id, search_id, status="done", result=result, refinements=refinements,
                                   finished_at=time.time(), elapsed=time.time() - started)

    def _refine_finished(self, job, search_id):
        # The earlier results are still valid, so a failed or cancelled refine keeps showing them.
        if job.status == "done":
            return
        error = job.error if job.status == "failed" else "cancelled"
        try:
            self.library.update_search(job.video_id, search_id, status="done", error=error, finished_at=time.time())
        except LibraryError:
            pass

    def _search_finished(self, job, search_id):
        if job.status == "done":
            return
        try:
            self.library.update_search(job.video_id, search_id, status=job.status, error=job.error, finished_at=time.time())
        except LibraryError:
            pass

    def _pipeline(self, video_id):
        record = self.library.get(video_id)
        key = (video_id, record["index"]["dir"])
        with self._lock:
            if key in self._pipelines:
                self._pipelines.move_to_end(key)
                return self._pipelines[key]
        folder = self.library.video_dir(video_id)
        try:
            pipeline = load_index(folder / record["index"]["dir"], self.settings(), folder / record["source"])
        except IndexVersionMismatch as exc:
            raise LibraryError(str(exc))
        self._remember_pipeline(key, pipeline)
        return pipeline

    def _remember_pipeline(self, key, pipeline):
        with self._lock:
            self._pipelines[key] = pipeline
            self._pipelines.move_to_end(key)
            while len(self._pipelines) > 2:
                self._pipelines.popitem(last=False)

    def search(self, video_id, search_id):
        record = self.library.get_search(video_id, search_id)
        root = self.library.search_dir(video_id, search_id).resolve()
        base = f"/api/videos/{video_id}/searches/{search_id}/files/"
        if record.get("result"):
            record["result"] = publish(record["result"], root, base)
        job_id = record.get("job_id")
        try:
            record["job"] = self.jobs.get(job_id).to_dict() if job_id else None
        except KeyError:
            record["job"] = None
        return record


def publish(value, root, base):
    """Prepare a pipeline result for the browser: file paths become URLs, heavy diagnostics shrink."""
    if isinstance(value, list):
        return [publish(item, root, base) for item in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        if key == "results_file":
            continue
        if key == "evidence_map" and isinstance(item, list):
            output[key] = downsample_evidence(item)
        elif key in ("initial_candidates", "top_evidence_bins"):
            continue
        elif key == "candidates" and isinstance(item, list):
            output[key] = [{"start": c.get("start"), "end": c.get("end"), "score": c.get("score")} for c in item]
        elif isinstance(item, str) and (key == "path" or key.endswith("_path")):
            try:
                relative = Path(item).resolve().relative_to(root)
            except (ValueError, OSError):
                continue
            output["url" if key == "path" else key[:-4] + "url"] = base + relative.as_posix()
        else:
            output[key] = publish(item, root, base)
    return output


def downsample_evidence(rows, limit=MAX_EVIDENCE_POINTS):
    size = max(1, math.ceil(len(rows) / limit))
    output = []
    for i in range(0, len(rows), size):
        group = rows[i:i + size]
        channels = {}
        for row in group:
            for channel, score in (row.get("channel_scores") or {}).items():
                channels[channel] = max(channels.get(channel, 0.0), float(score))
        output.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "score": max(float(row.get("score", 0.0)) for row in group),
            "channel_scores": channels,
        })
    return output
