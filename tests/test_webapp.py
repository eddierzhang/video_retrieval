import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from starlette.testclient import TestClient

from video_retrieval.local_backend import LocalModels
from webapp.jobs import JobQueue
from webapp.library import Library
from webapp.server import create_app
from webapp.service import Service, downsample_evidence

HEADERS = {"X-Moments": "1"}


class WebAppTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = JobQueue(start=False)
        self.service = Service(Library(self.root / "library"), self.jobs, settings_path=self.root / "settings.json")
        self.client = TestClient(create_app(self.service, allowed_hosts=["testserver"]))

    def video_bytes(self, frames=30):
        path = self.root / f"clip{frames}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for i in range(frames):
            writer.write(np.full((48, 64, 3), i * 8 % 255, dtype=np.uint8))
        writer.release()
        return path.read_bytes()

    def upload(self, data, name="My clip.mp4"):
        return self.client.post("/api/videos", content=data, headers={**HEADERS, "X-Filename": name.replace(" ", "%20")})

    def indexed_video(self):
        video = self.upload(self.video_bytes()).json()

        def fake_prepare(source, models, index_root, reporter, cancel_event):
            reporter({"stage": "Embedding frames at every temporal scale"})
            folder = Path(index_root) / "abc"
            folder.mkdir(parents=True)
            (folder / "ready.json").write_text(json.dumps({"models": models.index_signature(), "has_transcript": False, "stats": {"scene_records": 3}}))
            return self.pipeline

        self.pipeline = Mock()
        self.pipeline.resources = Mock()
        with patch("webapp.service.prepare_video", side_effect=fake_prepare), \
                patch("webapp.service.index_key", return_value=("abc", {})), \
                patch.object(Library, "make_preview"):
            job = self.jobs.run_next(timeout=1)
        self.assertEqual(job.status, "done", job.error)
        return video

    def test_mutations_require_the_app_header(self):
        response = self.client.post("/api/videos", content=b"x", headers={"X-Filename": "a.mp4"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/videos").status_code, 200)

    def test_upload_validates_dedupes_and_queues_indexing(self):
        self.assertEqual(self.upload(b"not a video", "fake.mp4").status_code, 400)
        self.assertEqual(self.upload(b"data", "../../notes.txt").status_code, 400)
        self.assertEqual(list((self.root / "library" / "_incoming").iterdir()), [])

        data = self.video_bytes()
        first = self.upload(data)
        self.assertEqual(first.status_code, 201)
        record = first.json()
        self.assertEqual(record["name"], "My clip.mp4")
        self.assertEqual(record["status"], "queued")
        self.assertFalse(record["playable"])  # mp4v needs a browser preview
        self.assertEqual([job["kind"] for job in record["jobs"]], ["index"])

        second = self.upload(data, "renamed.mp4")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["id"], record["id"])
        self.assertEqual(len(self.jobs.active()), 1)
        self.assertEqual(self.client.get(f"/api/videos/{record['id']}/thumbnail").status_code, 200)

    def test_indexing_then_search_publishes_file_urls(self):
        video = self.indexed_video()
        payload = self.client.get(f"/api/videos/{video['id']}").json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["index"]["dir"], "index/abc")

        self.assertEqual(self.client.post(f"/api/videos/{video['id']}/searches", json={"query": " "}, headers=HEADERS).status_code, 400)
        response = self.client.post(f"/api/videos/{video['id']}/searches",
                                    json={"query": "  a red   car ", "mode": "quick"}, headers=HEADERS)
        self.assertEqual(response.status_code, 202)
        search = response.json()
        self.assertEqual((search["query"], search["status"]), ("a red car", "queued"))

        def fake_retrieve(query, reporter, cancel_event, output_root, **kwargs):
            clip = Path(output_root) / "q" / "clips" / "match.mp4"
            clip.parent.mkdir(parents=True)
            clip.write_bytes(b"clip")
            evidence = [{"start": i * 2, "end": i * 2 + 2, "score": 0.9 if i == 777 else 0.1, "channel_scores": {"video": 0.1}} for i in range(2000)]
            return {"query": query, "matches": [{"start": 1, "end": 2, "clip_path": str(clip), "frames": [{"path": str(clip), "timestamp": 1}]}],
                    "results_file": str(clip), "diagnostics": {"evidence_map": evidence, "initial_candidates": [1],
                                                               "candidates": [{"start": 0, "end": 3, "score": 0.5, "bin_ids": [1]}]}}

        self.pipeline.retrieve.side_effect = fake_retrieve
        with patch("webapp.service.check_runtime"):
            job = self.jobs.run_next(timeout=1)
        self.assertEqual(job.status, "done", job.error)
        self.assertFalse(self.pipeline.retrieve.call_args.kwargs["run_verification"])
        self.assertEqual(self.pipeline.retrieve.call_args.kwargs["max_candidates"], 10)

        result = self.client.get(f"/api/videos/{video['id']}/searches/{search['id']}").json()["result"]
        match = result["matches"][0]
        self.assertNotIn("clip_path", match)
        self.assertNotIn("results_file", result)
        self.assertEqual(self.client.get(match["clip_url"]).content, b"clip")
        self.assertTrue(match["frames"][0]["url"].endswith("/files/q/clips/match.mp4"))
        self.assertLessEqual(len(result["diagnostics"]["evidence_map"]), 720)
        self.assertEqual(max(row["score"] for row in result["diagnostics"]["evidence_map"]), 0.9)
        self.assertEqual(result["diagnostics"]["candidates"], [{"start": 0, "end": 3, "score": 0.5}])
        self.assertNotIn("initial_candidates", result["diagnostics"])

        base = f"/api/videos/{video['id']}/searches/{search['id']}/files/"
        self.assertEqual(self.client.get(base + "search.json").status_code, 404)
        self.assertEqual(self.client.get(base + "..%2F..%2Fvideo.json").status_code, 404)
        history = self.client.get(f"/api/videos/{video['id']}/searches").json()["searches"]
        self.assertEqual([row["status"] for row in history], ["done"])

    def test_failed_and_cancelled_indexing_update_the_video(self):
        video = self.upload(self.video_bytes()).json()
        with patch.object(Library, "make_preview"), patch("webapp.service.prepare_video", side_effect=RuntimeError("Ollama is not running")):
            job = self.jobs.run_next(timeout=1)
        self.assertEqual(job.status, "failed")
        record = self.client.get(f"/api/videos/{video['id']}").json()
        self.assertEqual((record["status"], record["error"]), ("failed", "Ollama is not running"))

        job = self.client.post(f"/api/videos/{video['id']}/index", headers=HEADERS).json()
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/cancel", headers=HEADERS).status_code, 200)
        self.assertEqual(self.jobs.run_next(timeout=1).status, "cancelled")
        self.assertEqual(self.client.get(f"/api/videos/{video['id']}").json()["status"], "cancelled")
        self.assertEqual(self.client.post(f"/api/videos/{video['id']}/searches", json={"query": "car"}, headers=HEADERS).status_code, 400)

    def test_settings_are_validated_and_flag_outdated_indexes(self):
        video = self.indexed_video()
        self.assertEqual(self.client.put("/api/settings", json={"frame_limit": 2}, headers=HEADERS).status_code, 400)
        self.assertFalse(self.client.get(f"/api/videos/{video['id']}").json()["index_outdated"])
        response = self.client.put("/api/settings", json={"vision": "qwen2.5vl:7b"}, headers=HEADERS)
        self.assertEqual(response.json()["vision"], "qwen2.5vl:7b")
        self.assertEqual(self.service.settings(), LocalModels(vision="qwen2.5vl:7b"))
        self.assertTrue(self.client.get(f"/api/videos/{video['id']}").json()["index_outdated"])

    def test_system_reports_a_stopped_model_server(self):
        with patch("webapp.service.installed_models", side_effect=RuntimeError("not running")), \
                patch("webapp.service.devices", return_value={"clip": "cuda"}):
            status = self.client.get("/api/system").json()
        self.assertFalse(status["ollama"]["running"])
        self.assertEqual(status["ollama"]["error"], "not running")

    def test_restart_marks_interrupted_work(self):
        video = self.upload(self.video_bytes()).json()
        self.service.library.update(video["id"], status="indexing")
        Service(self.service.library, JobQueue(start=False), settings_path=self.root / "settings.json")
        self.assertEqual(self.service.library.get(video["id"])["status"], "interrupted")

    def test_evidence_downsampling_keeps_peaks(self):
        rows = [{"start": i, "end": i + 1, "score": 1.0 if i == 1234 else 0.0, "channel_scores": {"video": float(i == 99)}} for i in range(5000)]
        small = downsample_evidence(rows, limit=500)
        self.assertLessEqual(len(small), 500)
        self.assertEqual(max(row["score"] for row in small), 1.0)
        self.assertEqual((small[0]["start"], small[-1]["end"]), (0, 5000))


if __name__ == "__main__":
    unittest.main()
