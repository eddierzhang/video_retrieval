"""Single-worker background jobs with progress reporting and cooperative cancellation.

Local models share one GPU, so jobs run one at a time in submission order.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid

from video_retrieval.local_backend import Cancelled

ACTIVE = ("queued", "running")


class Job:
    def __init__(self, kind, video_id, title, stages, run, on_finish=None, alternate_stages=None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.video_id = video_id
        self.title = title
        self.stages = list(stages)
        self.run = run
        self.on_finish = on_finish
        # A stage that switches the plan, e.g. a text-reading query skipping retrieval stages.
        self.alternate_stages = alternate_stages or {}
        self.status = "queued"
        self.stage = None
        self.tasks = {}
        self.model = None
        self.error = None
        self.result = {}
        self.created_at = time.time()
        self.started_at = None
        self.finished_at = None
        self.cancel_event = threading.Event()

    def report(self, event):
        if "stage" in event:
            self.stage = event["stage"]
            if self.stage in self.alternate_stages:
                self.stages = list(self.alternate_stages[self.stage])
            elif self.stage not in self.stages:
                self.stages.append(self.stage)
            self.tasks = {}
        if "task" in event:
            self.tasks[event["task"]] = {"done": int(event.get("done") or 0), "total": event.get("total")}
        if "model" in event:
            self.model = event["model"]

    def to_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "video_id": self.video_id,
            "title": self.title,
            "status": self.status,
            "stages": self.stages,
            "stage": self.stage,
            "tasks": [{"name": name, **progress} for name, progress in self.tasks.items()],
            "model": self.model,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_event.is_set(),
        }


class JobQueue:
    def __init__(self, keep=100, start=True):
        self._jobs = {}
        self._keep = keep
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        if start:
            threading.Thread(target=self._work, name="moments-jobs", daemon=True).start()

    def submit(self, job):
        with self._lock:
            self._jobs[job.id] = job
            finished = sorted((j for j in self._jobs.values() if j.status not in ACTIVE), key=lambda j: j.created_at)
            for old in finished[: max(0, len(self._jobs) - self._keep)]:
                del self._jobs[old.id]
        self._queue.put(job)
        return job

    def get(self, job_id):
        return self._jobs[job_id]

    def list(self):
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def active(self, video_id=None, kind=None):
        return [
            job for job in self.list()
            if job.status in ACTIVE and video_id in (None, job.video_id) and kind in (None, job.kind)
        ]

    def cancel(self, job_id):
        job = self.get(job_id)
        job.cancel_event.set()
        return job

    def run_next(self, timeout=None):
        """Run one queued job on the calling thread (the worker loop; also used by tests)."""
        job = self._queue.get(timeout=timeout)
        if job.cancel_event.is_set():
            job.status = "cancelled"
        else:
            job.status = "running"
            job.started_at = time.time()
            try:
                job.run(job)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as exc:
                traceback.print_exc()
                job.status = "failed"
                job.error = str(exc) or type(exc).__name__
        job.finished_at = time.time()
        if job.on_finish is not None:
            try:
                job.on_finish(job)
            except Exception:
                traceback.print_exc()
        return job

    def _work(self):
        while True:
            self.run_next()
