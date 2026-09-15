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
        self.assertEqual(self.pipeline.retrieve.call_args.kwargs["max_candidates"], 12)

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

    def detect_result(self, output_root, keys):
        evidence = Path(output_root) / "evidence" / "match.jpg"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_bytes(b"jpg")
        (Path(output_root) / "detect_state.pkl").write_bytes(b"state")
        return {"query": "q", "mode": "detect", "verified": False, "num_matches": len(keys), "results_file": str(evidence),
                "matches": [{"start": i, "end": i + 1, "confidence": 0.8, "match_key": key, "evidence_image_path": str(evidence)}
                            for i, key in enumerate(keys)], "diagnostics": {}}

    def test_detect_search_takes_feedback_and_refines(self):
        video = self.indexed_video()
        base = f"/api/videos/{video['id']}/searches"
        too_many = {"query": "police officers", "mode": "detect", "options": {"max_frames": 99999}}
        self.assertEqual(self.client.post(base, json=too_many, headers=HEADERS).status_code, 400)
        search = self.client.post(base, json={"query": "police officers", "mode": "detect"}, headers=HEADERS).json()
        self.assertEqual(search["options"]["max_frames"], 400)
        self.assertIn("Detecting objects", search["job"]["stages"])
        url = f"{base}/{search['id']}"

        def fake_detect(query, resources, output_root, **kwargs):
            return self.detect_result(output_root, ["0:1.00", "2:7.50"])

        with patch("webapp.service.check_runtime"), patch("video_retrieval.detect_search.detect_search", side_effect=fake_detect):
            job = self.jobs.run_next(timeout=1)
        self.assertEqual(job.status, "done", job.error)
        match = self.client.get(url).json()["result"]["matches"][0]
        self.assertEqual(self.client.get(match["evidence_image_url"]).content, b"jpg")

        self.assertEqual(self.client.post(f"{url}/refine", headers=HEADERS).status_code, 400)   # nothing marked yet
        self.assertEqual(self.client.post(f"{url}/feedback", json={"match_key": "9:9.99", "label": "positive"}, headers=HEADERS).status_code, 400)
        self.assertEqual(self.client.post(f"{url}/feedback", json={"match_key": "0:1.00", "label": "maybe"}, headers=HEADERS).status_code, 400)
        example = {"features": {"box_area": 0.1, "rule_pass": 1}, "embedding": None, "concept": "person", "query": "q"}
        with patch("webapp.service.example_for", return_value=example) as example_for:
            marked = self.client.post(f"{url}/feedback", json={"match_key": "0:1.00", "label": "positive"}, headers=HEADERS).json()
            self.client.post(f"{url}/feedback", json={"match_key": "2:7.50", "label": "negative"}, headers=HEADERS)
            self.assertEqual(self.client.get("/api/learning").json()["examples"], 2)
            cleared = self.client.post(f"{url}/feedback", json={"match_key": "2:7.50", "label": None}, headers=HEADERS).json()
        self.assertEqual(example_for.call_count, 2)   # clearing a mark needs no example
        self.assertEqual(marked["feedback"], {"0:1.00": "positive"})
        self.assertEqual(cleared["feedback"], {"0:1.00": "positive"})
        self.assertEqual((cleared["learning"]["examples"], cleared["learning"]["state"]), (1, "collecting"))
        self.assertEqual(self.service.learner.examples[f"{video['id']}/{search['id']}/0:1.00"]["label"], 1)

        response = self.client.post(f"{url}/refine", headers=HEADERS)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")

        def fake_refine(output_root, feedback, resources, **kwargs):
            self.assertEqual(feedback, {"0:1.00": "positive"})
            self.assertIs(kwargs["learner"], self.service.learner)
            return self.detect_result(output_root, ["0:1.00"])

        with patch("video_retrieval.detect_search.refine", side_effect=fake_refine):
            self.assertEqual(self.jobs.run_next(timeout=1).status, "done")
        refined = self.client.get(url).json()
        self.assertEqual((refined["status"], refined["refinements"], refined["result"]["num_matches"]), ("done", 1, 1))

        self.client.post(f"{url}/refine", headers=HEADERS)
        with patch("video_retrieval.detect_search.refine", side_effect=RuntimeError("out of memory")):
            self.assertEqual(self.jobs.run_next(timeout=1).status, "failed")
        after_failure = self.client.get(url).json()
        # A failed refine leaves the earlier results in place.
        self.assertEqual((after_failure["status"], after_failure["error"]), ("done", "out of memory"))
        self.assertEqual(after_failure["result"]["num_matches"], 1)

    def test_detect_plans_missed_moments_and_text_routing(self):
        video = self.indexed_video()
        base = f"/api/videos/{video['id']}/searches"
        bad = {"query": "officers", "mode": "detect", "options": {"plan": {"object": "", "action": ""}}}
        self.assertEqual(self.client.post(base, json=bad, headers=HEADERS).status_code, 400)
        edited = {"object": " person ", "target": "a police officer", "contrasts": "a person; a chef", "min_count": "2"}
        search = self.client.post(base, json={"query": "officers", "mode": "detect",
                                              "options": {"plan": edited, "vision_check": False}}, headers=HEADERS).json()
        self.assertEqual(search["options"]["plan"]["contrasts"], ["a person", "a chef"])
        self.assertEqual(search["options"]["plan"]["min_count"], 2)
        self.assertNotIn("Checking results with the vision model", search["job"]["stages"])
        url = f"{base}/{search['id']}"

        def fake_detect(query, resources, output_root, **kwargs):
            self.assertEqual(kwargs["plan"]["object"], "person")
            self.assertFalse(kwargs["vision_check"])
            return self.detect_result(output_root, ["0:1.00"])

        with patch("webapp.service.check_runtime"), patch("video_retrieval.detect_search.detect_search", side_effect=fake_detect):
            self.assertEqual(self.jobs.run_next(timeout=1).status, "done")

        self.assertEqual(self.client.post(f"{url}/missed", json={"start": 5, "end": 5.1}, headers=HEADERS).status_code, 400)
        candidate = {"match_key": "missed:2.00", "detected": True}
        example = {"features": {"rule_pass": 0}, "embedding": None, "concept": "person", "query": "officers"}
        with patch("webapp.service.missed_candidate", return_value=(candidate, example)) as missed:
            added = self.client.post(f"{url}/missed", json={"start": 2, "end": 2.8}, headers=HEADERS).json()
        self.assertEqual(missed.call_args.args[1:], (2.0, 2.8))
        self.assertEqual(added["feedback"], {"missed:2.00": "positive"})
        self.assertEqual(added["missed"]["missed:2.00"], {"start": 2.0, "end": 2.8, "detected": True})
        self.assertEqual(added["learning"]["examples"], 1)
        removed = self.client.post(f"{url}/feedback", json={"match_key": "missed:2.00", "label": None}, headers=HEADERS).json()
        self.assertEqual((removed["feedback"], removed["missed"], removed["learning"]["examples"]), ({}, {}, 0))

        reading = self.client.post(base, json={"query": "read the sign", "mode": "detect"}, headers=HEADERS).json()
        self.pipeline.retrieve.return_value = {"query": "read the sign", "matches": [], "plan": {"executor": "visual_text_extraction"}}
        with patch("webapp.service.check_runtime"), \
                patch("video_retrieval.detect_search.detect_search", return_value={"route": "text", "plan": {}}):
            self.assertEqual(self.jobs.run_next(timeout=1).status, "done")
        routed = self.client.get(f"{base}/{reading['id']}").json()
        self.assertEqual(routed["result"]["routed"]["to"], "verified")
        self.assertTrue(self.pipeline.retrieve.call_args.kwargs["run_verification"])
        self.assertEqual(self.client.post(f"{base}/{reading['id']}/missed", json={"start": 1, "end": 2}, headers=HEADERS).status_code, 400)

    def test_only_detect_searches_take_feedback(self):
        video = self.indexed_video()
        base = f"/api/videos/{video['id']}/searches"
        search = self.client.post(base, json={"query": "a car", "mode": "quick"}, headers=HEADERS).json()
        url = f"{base}/{search['id']}"
        self.assertEqual(self.client.post(f"{url}/feedback", json={"match_key": "0:1.00", "label": "positive"}, headers=HEADERS).status_code, 400)
        self.assertEqual(self.client.post(f"{url}/refine", headers=HEADERS).status_code, 400)
        self.assertEqual(self.client.post(f"{url}/refine").status_code, 403)
        self.assertEqual(self.client.post(base, json={"query": "a car", "mode": "fast"}, headers=HEADERS).status_code, 400)

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

    def test_json_writes_wait_out_a_reader_holding_the_file(self):
        from webapp import library

        path = self.root / "record.json"
        real_replace, calls = library.os.replace, []

        def busy_then_free(source, target):
            calls.append(target)
            if len(calls) < 3:
                raise PermissionError("Access is denied")
            real_replace(source, target)

        with patch.object(library.os, "replace", side_effect=busy_then_free), patch.object(library.time, "sleep"):
            library.write_json(path, {"a": 1})
        self.assertEqual((len(calls), json.loads(path.read_text())), (3, {"a": 1}))

    def test_evidence_downsampling_keeps_peaks(self):
        rows = [{"start": i, "end": i + 1, "score": 1.0 if i == 1234 else 0.0, "channel_scores": {"video": float(i == 99)}} for i in range(5000)]
        small = downsample_evidence(rows, limit=500)
        self.assertLessEqual(len(small), 500)
        self.assertEqual(max(row["score"] for row in small), 1.0)
        self.assertEqual((small[0]["start"], small[-1]["end"]), (0, 5000))


if __name__ == "__main__":
    unittest.main()
