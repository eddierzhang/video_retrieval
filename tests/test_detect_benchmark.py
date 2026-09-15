"""Benchmark scoring, and the checks Detect runs before showing results: vision model, missed moments, routing."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import detect as bench
from video_retrieval import detect_search as detect


class ScoreTest(unittest.TestCase):
    truth = {"intervals": [[1.0, 4.0], [10.0, 12.0]], "acceptable": [[4.0, 7.0]]}

    def test_clips_are_hits_neutral_or_false_positives(self):
        result = bench.score(self.truth, [[1.5, 3.5], [4.5, 6.5], [20.0, 22.0], [3.0, 6.0]])
        self.assertEqual(result["verdicts"], ["hit", "neutral", "false_positive", "neutral"])
        self.assertEqual((result["hits"], result["false_positives"]), (1, 1))
        self.assertEqual((result["found"], result["recall"]), (1, 0.5))
        self.assertAlmostEqual(result["coverage"], 2.5 / 5.0)   # 1.5-4.0 of the first interval, none of the second

    def test_an_interval_is_found_by_a_second_of_cover_or_half_of_it(self):
        self.assertEqual(bench.score(self.truth, [[10.0, 11.0]])["found"], 1)     # half of a 2 s interval
        self.assertEqual(bench.score(self.truth, [[1.0, 1.9]])["found"], 0)      # under a second of a 3 s one
        self.assertEqual(bench.covered([0.0, 10.0], [[1.0, 3.0], [2.0, 4.0], [8.0, 20.0]]), 5.0)

    def test_text_answers_ignore_case_and_punctuation(self):
        truth = {**self.truth, "answers": ["Atago Hills Dog Park"]}
        self.assertTrue(bench.score(truth, [], ["ATAGO HILLS DOG PARK!"])["text_correct"])
        self.assertFalse(bench.score(truth, [], ["Welcome"])["text_correct"])

    def test_summary_totals_over_queries(self):
        rows = [{"status": "done", "seconds": 3, "score": bench.score(self.truth, [[1.5, 3.5], [20, 22]])},
                {"status": "failed", "seconds": 1, "score": None}]
        summary = bench.summarise(rows)
        self.assertEqual((summary["precision"], summary["mean_recall"], summary["failed"]), (0.5, 0.5, 1))

    def test_the_ground_truth_file_is_well_formed(self):
        truth = json.loads(bench.GROUND_TRUTH.read_text(encoding="utf-8"))
        ids = [item["id"] for item in truth["queries"]]
        self.assertEqual(len(ids), len(set(ids)))
        for item in truth["queries"]:
            for start, end in item["intervals"] + item.get("acceptable", []):
                self.assertLess(start, end, item["id"])
            self.assertEqual(item["kind"] == "text", bool(item.get("answers")), item["id"])


def unit(index, size=8):
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


def state_with(tracks, plan=None):
    plan = {"object": "person", "target": "a person in uniform", "contrasts": ["a person"], "min_count": 1,
            "with_object": "", "with_count": 0, "action": "", "action_contrasts": [], **(plan or {})}
    for number, track in enumerate(tracks):
        track.update(id=number, identity=number)
    return {"query": "police officers", "plan": plan, "scale": 10.0, "tracks": tracks, "windows": [],
            "shots": [{"id": 0, "start": 0.0, "end": 10.0}], "shot_fps": {0: 2.0}, "bridged": set(),
            "thresholds": {"attribute": 0.5, "action": 2.0}}


def track(times, probability, embedding, box=(0.2, 0.2, 0.5, 0.8)):
    return {"shot": 0, "observations": [{"time": t, "box": box, "embedding": embedding, "probability": probability,
                                         "score": 0.5} for t in times]}


class VisionCheckTest(unittest.TestCase):
    def test_without_a_learned_scorer_a_no_removes_the_result(self):
        state = state_with([track([1.0, 1.5], 0.8, unit(0)), track([5.0, 5.5], 0.8, unit(1))])

        def checker(pool):
            for candidate in pool:
                candidate["vision"] = {"matches": candidate["start"] < 3, "reason": "r"}

        results = detect.propose(state, checker=checker)
        self.assertEqual([r["match_key"] for r in results], ["0:0.75"])

    def test_answers_are_cached_become_measurements_and_errors_leave_results(self):
        state = state_with([track([1.0, 1.5], 0.8, unit(0)), track([5.0, 5.5], 0.8, unit(1))])
        candidates = detect.assemble(state)
        answers = iter([{"matches": True, "reason": "a uniform"}, {"matches": False, "reason": "casual clothes"}])
        with patch.object(detect, "evidence_image", return_value=None), \
                patch.object(detect, "clip_frames", return_value=[np.zeros((4, 4, 3), np.uint8)]), \
                patch.object(detect.local_backend, "images_json", side_effect=lambda *a, **k: next(answers)) as ask, \
                patch.object(detect.local_backend, "stage"):
            detect.vision_check(state, candidates, "video.mp4")
            detect.vision_check(state, candidates, "video.mp4")         # the second pass asks nothing
        self.assertEqual(ask.call_count, 2)
        self.assertEqual([c["features"]["vision_yes"] for c in candidates], [1, 0])
        self.assertIn('"police officers"', ask.call_args_list[0].args[1])

        fresh = state_with([track([1.0, 1.5], 0.8, unit(0))])
        pool = detect.assemble(fresh)
        with patch.object(detect, "evidence_image", return_value=None), \
                patch.object(detect, "clip_frames", return_value=[np.zeros((4, 4, 3), np.uint8)]), \
                patch.object(detect.local_backend, "images_json", side_effect=RuntimeError("Ollama is not running")), \
                patch.object(detect.local_backend, "stage"):
            detect.vision_check(fresh, pool, "video.mp4")
        self.assertEqual(fresh["vision_error"], "Ollama is not running")
        self.assertEqual(pool[0]["features"]["vision_checked"], 0)
        self.assertNotIn("vision", pool[0])


class MissedMomentTest(unittest.TestCase):
    def test_a_missed_stretch_learns_from_what_was_detected_there(self):
        state = state_with([
            track([1.0, 1.5, 2.0], 0.8, unit(0)),
            track([6.0, 6.5, 7.0], 0.3, unit(1)),      # detected, but under the threshold, so not returned
        ])
        state["last_matches"] = detect.assemble(state)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "detect_state.pkl").write_bytes(pickle.dumps(state))
            candidate, example = detect.missed_candidate(folder, 5.8, 7.2)
            self.assertEqual((candidate["match_key"], candidate["detected"], candidate["rule_pass"]), ("missed:5.80", True, False))
            self.assertTrue(5.8 <= candidate["evidence"][1]["time"] <= 7.2)   # evidence from inside the stretch
            self.assertEqual((example["features"]["rule_pass"], example["missed"]), (0, True))
            self.assertAlmostEqual(example["features"]["appearance"], 0.3)
            saved = pickle.loads((Path(folder) / "detect_state.pkl").read_bytes())
            self.assertIn("missed:5.80", [c["match_key"] for c in saved["last_matches"]])

            nothing, no_example = detect.missed_candidate(folder, 8.5, 9.5)
            self.assertFalse(nothing["detected"])
            self.assertIsNone(no_example)

    def test_a_missed_stretch_in_a_motion_search(self):
        state = state_with([track([1.0, 1.5, 2.0, 6.0, 6.5, 7.0], 0.8, unit(0))],
                           plan={"target": "", "contrasts": [], "action": "petting a dog"})
        state["windows"] = [{"shot": 0, "start": 0.5, "end": 2.5, "probability": 0.5, "choices": 8, "embedding": unit(0)},
                            {"shot": 0, "start": 5.5, "end": 7.5, "probability": 0.1, "choices": 8, "embedding": unit(0)}]
        state["last_matches"] = detect.assemble(state)
        self.assertEqual(len(state["last_matches"]), 1)           # the second stretch's motion is under the threshold
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "detect_state.pkl").write_bytes(pickle.dumps(state))
            candidate, example = detect.missed_candidate(folder, 5.5, 7.5)
            self.assertTrue(candidate["detected"])
            self.assertAlmostEqual(example["features"]["action_lift"], 0.8)


class PlanRoutingTest(unittest.TestCase):
    def test_reading_requests_are_routed_and_edited_plans_are_not(self):
        self.assertTrue(detect.normalize_plan({"object": "sign", "reads_text": True}, "q")["reads_text"])
        self.assertTrue(detect.TEXT_REQUEST.search("read the number on the white taxi"))
        self.assertTrue(detect.TEXT_REQUEST.search("What does the sign say?"))
        self.assertFalse(detect.TEXT_REQUEST.search("a woman standing next to the Welcome sign"))

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(detect, "plan_detection", return_value=detect.normalize_plan({"object": "sign", "reads_text": True}, "q")), \
                patch.object(detect.local_backend, "stage"):
            resources = type("R", (), {"manifest": {"video": {"path": "v.mp4", "duration": 10.0}}})()
            self.assertEqual(detect.detect_search("read the sign", resources, folder)["route"], "text")


if __name__ == "__main__":
    unittest.main()
