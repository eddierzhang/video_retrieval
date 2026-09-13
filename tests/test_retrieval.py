"""Planning, routing, evidence fusion, candidate ordering and learned boundaries."""
import unittest
from unittest.mock import patch

import numpy as np

from video_retrieval import local_backend as local
from video_retrieval.pipeline import RetrievalResources, VideoRetrievalPipeline, retrieve_video
from video_retrieval.retrieval import plan_query


class RetrievalTest(unittest.TestCase):
    def test_negative_evidence_lowers_the_evidence_map(self):
        from video_retrieval.retrieval import build_temporal_evidence_map

        plan = {"weights": {"video": 1.0, "metadata": 0.0, "transcript_semantic": 0.0, "transcript_bm25": 0.0}}
        hit = [{"start": 10, "end": 18, "score": 0.9, "_predicate_importance": 1.0}]
        empty = {"metadata": [], "transcript_semantic": [], "transcript_bm25": []}
        without = build_temporal_evidence_map({"video": [hit], "negative": [], **empty}, plan, video_duration=40)
        against = build_temporal_evidence_map({"video": [hit], "negative": [list(hit)], **empty}, plan, video_duration=40)
        self.assertGreater(max(row["score"] for row in without), max(row["score"] for row in against))
        self.assertGreaterEqual(min(row["score"] for row in against), 0.0)

    def test_temporal_ordering_reranks_candidates(self):
        from video_retrieval.retrieval import apply_temporal_ordering

        def ranking(predicate, start, end):
            return [{"start": start, "end": end, "score": 0.9, "_predicate_id": predicate, "_predicate_importance": 1.0}]

        results = {
            "video": [ranking("a", 0, 4), ranking("b", 10, 14), ranking("b", 30, 34), ranking("a", 40, 44)],
            "metadata": [], "transcript_semantic": [], "transcript_bm25": [], "negative": [],
        }
        plan = {"ordering": [{"first": "a", "then": "b", "description": "a before b"}]}
        candidates = [{"candidate_id": 0, "start": 30, "end": 50, "score": 0.5},
                      {"candidate_id": 1, "start": 0, "end": 20, "score": 0.5}]
        ordered = apply_temporal_ordering(candidates, results, plan)
        self.assertEqual(ordered[0]["start"], 0)
        self.assertEqual(ordered[0]["ordering_satisfied"], 1)
        self.assertEqual(ordered[-1]["ordering_violated"], 1)
        # With no ordering to enforce, candidates come back untouched.
        self.assertEqual(apply_temporal_ordering(candidates, results, {}), candidates)

    def test_planner_keeps_only_ordering_between_real_predicates(self):
        raw = {
            "executor": "temporal_grounding",
            "weights": {"video": 1, "metadata": 0, "transcript_semantic": 0, "transcript_bm25": 0},
            "evidence_predicates": [
                {"id": "a", "description": "first thing", "role": "target", "modalities": ["video"], "required": True, "importance": 0.9},
                {"id": "b", "description": "second thing", "role": "cue", "modalities": ["video"], "required": False, "importance": 0.5},
            ],
            "ordering": [
                {"first": "a", "then": "b", "description": "a before b"},
                {"first": "a", "then": "ghost", "description": "names a predicate that does not exist"},
                {"first": "a", "then": "a", "description": "self reference"},
            ],
        }
        with patch.object(local, "chat_json", return_value=raw):
            plan = plan_query("something happens and then something else")
        self.assertEqual([(rule["first"], rule["then"]) for rule in plan["ordering"]], [("a", "b")])

    def test_text_route_is_checked_before_it_is_trusted(self):
        text_plan = {"executor": "visual_text_extraction", "weights": {}, "target_object": "banner"}
        moment_plan = {
            "executor": "temporal_grounding",
            "weights": {"video": 1, "metadata": 0, "transcript_semantic": 0, "transcript_bm25": 0},
            "evidence_predicates": [{"id": "a", "description": "a banner", "role": "target",
                                     "modalities": ["video"], "required": True, "importance": 0.9}],
        }
        stated = {"stated_text": "Happy Birthday", "asks_for_unknown_text": False}
        calls = []

        def planner(prompt, schema, images=None, role="vision"):
            calls.append(schema)
            if "asks_for_unknown_text" in schema["properties"]:
                return stated
            return moment_plan if len(calls) > 2 else text_plan

        with patch.object(local, "chat_json", side_effect=planner):
            plan = plan_query("a banner reading Happy Birthday")
        self.assertEqual(plan["executor"], "temporal_grounding")
        self.assertEqual(plan["rerouted"], {"from": "visual_text_extraction", "stated_text": "Happy Birthday"})
        # The re-plan is constrained, not merely asked nicely.
        self.assertEqual(calls[-1]["properties"]["executor"]["enum"], ["temporal_grounding"])

        calls.clear()
        stated = {"stated_text": "", "asks_for_unknown_text": True}
        with patch.object(local, "chat_json", side_effect=planner):
            plan = plan_query("what does the banner say")
        self.assertEqual(plan["executor"], "visual_text_extraction")
        self.assertNotIn("rerouted", plan)
        self.assertEqual(len(calls), 2)  # one plan, one check - no re-plan when the reader is right

    def test_retrieval_only_search_never_calls_vision_models(self):
        plan = {"executor": "temporal_grounding", "return_mode": "all", "expected_duration": {"min_seconds": 2, "max_seconds": 6},
                "weights": {"video": 1.0, "metadata": 0.0, "transcript_semantic": 0.0, "transcript_bm25": 0.0}}
        hits = {"video": [[{"start": 30, "end": 38, "score": 0.9}, {"start": 70, "end": 78, "score": 0.2}]],
                "metadata": [], "transcript_semantic": [], "transcript_bm25": []}
        with patch("video_retrieval.pipeline.plan_query", return_value=plan), \
                patch("video_retrieval.pipeline.run_retrieval_plan", return_value=hits), \
                patch("video_retrieval.pipeline.verify_candidates_flash", side_effect=AssertionError("vision model called")), \
                patch("video_retrieval.pipeline.materialize_final_matches", side_effect=lambda manifest, instances, query, **kw: (instances, "results.json")):
            result = retrieve_video("a car", RetrievalResources(manifest={"video": {"duration": 120}}),
                                    run_verification=False, include_evidence_map=True)
        self.assertFalse(result["verified"])
        self.assertTrue(any(m["start"] <= 30 and m["end"] >= 38 for m in result["matches"]))
        self.assertIn("evidence_map", result["diagnostics"])

    def test_tuned_settings_apply_only_in_their_own_mode(self):
        from video_retrieval import pipeline as pipeline_module
        from video_retrieval.retrieval import scoring

        tuned = {"mode": "quick", "run": "r1", "pipeline": {"candidate_padding": 1.0, "nms_iou_threshold": 0.4},
                 "scoring": {"peak_share": 0.5}}
        seen = []

        def capture(**kwargs):
            seen.append((kwargs, scoring()["peak_share"]))
            return {"matches": []}

        resources = RetrievalResources(manifest={}, local_models=local.LocalModels())
        with patch.object(pipeline_module, "load_tuned_settings", return_value=tuned), \
                patch("video_retrieval.pipeline.retrieve_video", side_effect=capture):
            quick = VideoRetrievalPipeline(resources).retrieve("q", run_verification=False, candidate_padding=7.0)
            VideoRetrievalPipeline(resources).retrieve("q", run_verification=True)
        (quick_kwargs, quick_share), (verified_kwargs, verified_share) = seen
        self.assertEqual(quick_kwargs["nms_iou_threshold"], 0.4)
        self.assertEqual(quick_kwargs["candidate_padding"], 7.0)  # an explicit argument still wins
        self.assertEqual(quick_share, 0.5)
        self.assertEqual(quick["tuned_settings"], "r1")
        self.assertNotIn("nms_iou_threshold", verified_kwargs)
        self.assertEqual(verified_share, 0.80)
        self.assertEqual(scoring()["peak_share"], 0.80)  # and nothing leaks out of the search

    def _step_signal(self):
        from video_retrieval import boundaries

        # Forty seconds; the query matches seconds 15 to 25 and nothing else.
        vectors = np.zeros((40, 1, 4), dtype=np.float32)
        vectors[:, 0, 1] = 1.0
        vectors[15:26, 0] = [1.0, 0.0, 0.0, 0.0]
        return boundaries.signals(np.arange(40, dtype=float), vectors, np.array([1.0, 0.0, 0.0, 0.0]))

    def _edge_model(self):
        from video_retrieval import boundaries

        weights = np.zeros(len(boundaries.FEATURES))
        weights[boundaries.FEATURES.index("edge")] = 1.0
        weights[boundaries.FEATURES.index("distance")] = -0.05
        return {"weights": weights.tolist(), "mean": [0.0] * len(weights), "std": [1.0] * len(weights),
                "window": 8, "context": 4}

    def test_boundaries_move_padded_proposals_onto_the_step(self):
        from video_retrieval import boundaries

        signal = self._step_signal()
        self.assertAlmostEqual(float(signal["step"]), 1.0)
        start, end = boundaries.refine_interval(signal, 10.0, 31.0, self._edge_model())
        self.assertEqual((start, end), (15.0, 26.0))  # the ten seconds of padding either side are gone
        # A boundary never moves further than its window.
        far_start, _ = boundaries.refine_interval(signal, 2.0, 31.0, self._edge_model())
        self.assertLessEqual(abs(far_start - 2.0), 8.0)

    def test_boundaries_never_cross_and_are_inert_untrained(self):
        from video_retrieval import boundaries

        # Whatever a model has learned - including nonsense - an answer is never inverted or empty.
        signal = self._step_signal()
        generator = np.random.RandomState(0)
        for _ in range(200):
            model = {"weights": generator.normal(size=len(boundaries.FEATURES)).tolist(),
                     "mean": [0.0] * len(boundaries.FEATURES), "std": [1.0] * len(boundaries.FEATURES),
                     "window": int(generator.randint(1, 12)), "context": int(generator.randint(1, 6))}
            start = float(generator.randint(0, 38))
            end = start + float(generator.randint(1, 40 - int(start)))
            new_start, new_end = boundaries.refine_interval(signal, start, end, model)
            self.assertTrue(new_end - new_start >= 1.0 or (new_start, new_end) == (start, end))
        instances = [{"start": 3.0, "end": 9.0}]
        with patch.object(boundaries, "load_boundary_model", return_value=None):
            same, info = boundaries.refine_instances({"video": {"path": "x", "duration": 40}}, "q", instances)
        self.assertIs(same, instances)
        self.assertIsNone(info)
        masked = {"mask": [1.0] * len(boundaries.FEATURES)}
        self.assertTrue(boundaries.uses_cuts(masked))
        masked["mask"][boundaries.FEATURES.index("cut")] = 0.0
        self.assertFalse(boundaries.uses_cuts(masked))


if __name__ == "__main__":
    unittest.main()
