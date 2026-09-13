"""Models trained from the pipeline's own output: pre-filter, ranker, exploration, adapter."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np



class LearningTest(unittest.TestCase):
    def test_prefilter_is_inert_until_a_model_is_trained(self):
        from video_retrieval import learning

        candidates = [{"candidate_id": i, "start": i * 10, "end": i * 10 + 8, "score": 1.0 - i * 0.1} for i in range(8)]
        with patch.object(learning, "load_prefilter", return_value=None):
            kept, info = learning.prefilter_candidates(candidates, [], {}, 100)
        self.assertIs(kept, candidates)
        self.assertIsNone(info)

    def test_prefilter_drops_weak_candidates_but_keeps_a_floor(self):
        from video_retrieval import learning

        candidates = [{"candidate_id": i, "start": i * 10, "end": i * 10 + 8, "score": 1.0 - i * 0.1} for i in range(8)]
        weights = [0.0] * len(learning.FEATURE_NAMES)
        weights[learning.FEATURE_NAMES.index("score")] = 10.0
        model = {"mean": [0.0] * len(weights), "std": [1.0] * len(weights),
                 "weights": weights, "bias": 0.0, "threshold": 0.999}
        with patch.object(learning, "load_prefilter", return_value=model):
            kept, info = learning.prefilter_candidates(candidates, [], {}, 100, keep_min=3)
        self.assertEqual(info["before"], 8)
        self.assertLess(len(kept), 8)
        self.assertGreaterEqual(len(kept), 3)
        self.assertEqual(kept[0]["candidate_id"], 0)

    def test_query_adapter_applies_only_when_trained(self):
        from video_retrieval import learning

        weights = np.zeros((4, 4), dtype=np.float32)
        weights[0, 1] = 1.0
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        with patch.object(learning, "load_adapter", return_value=weights):
            np.testing.assert_allclose(learning.apply_query_adapter(query), [0, 1, 0, 0], atol=1e-6)
        with patch.object(learning, "load_adapter", return_value=None):
            np.testing.assert_allclose(learning.apply_query_adapter(query), [1, 0, 0, 0])

    def test_candidate_features_describe_the_window(self):
        from video_retrieval import learning

        evidence = [{"start": t, "end": t + 2, "score": 0.5, "channel_scores": {
            "video": 0.8 if 10 <= t < 18 else 0.1, "metadata": 0.2,
            "transcript_semantic": 0.0, "transcript_bm25": 0.0,
            "negative": 0.3 if t >= 30 else 0.0}} for t in range(0, 40, 2)]
        candidates = [{"candidate_id": 0, "start": 10, "end": 18, "score": 0.9, "evidence_mass": 3.0},
                      {"candidate_id": 1, "start": 30, "end": 38, "score": 0.3}]
        features = learning.candidate_features(candidates[0], 0, candidates, evidence,
                                               {"expected_duration": {"max_seconds": 8}}, 40)
        self.assertAlmostEqual(features["channel_video"], 0.8)
        self.assertAlmostEqual(features["channel_negative"], 0.0)
        self.assertAlmostEqual(features["relative_score"], 1.0)
        self.assertAlmostEqual(features["duration_ratio"], 1.0)
        self.assertAlmostEqual(learning.candidate_features(candidates[1], 1, candidates, evidence, {}, 40)["channel_negative"], 0.3)
        self.assertEqual(len(learning.vectorize(features)), len(learning.FEATURE_NAMES))

    def test_ranker_forward_matches_a_hand_built_network(self):
        from video_retrieval import learning

        first = np.zeros((len(learning.FEATURE_NAMES), 2))
        first[learning.FEATURE_NAMES.index("score"), 0] = 1.0
        first[learning.FEATURE_NAMES.index("channel_video"), 1] = 1.0
        model = {"mean": [0.0] * len(learning.FEATURE_NAMES), "std": [1.0] * len(learning.FEATURE_NAMES),
                 "layers": [{"w": first.tolist(), "b": [0.0, 0.0]},
                            {"w": [[1.0], [2.0]], "b": [0.5]}],
                 "platt": {"a": 1.0, "b": 0.0}}
        features = {name: 0.0 for name in learning.FEATURE_NAMES}
        features.update({"score": 0.8, "channel_video": 0.3})
        matrix = np.stack([learning.vectorize(features)])
        # relu(0.8) * 1 + relu(0.3) * 2 + 0.5
        self.assertAlmostEqual(float(learning.ranker_scores(model, matrix)[0]), 1.9, places=6)
        # A negative feature is clipped by the relu, not passed through.
        features["score"] = -5.0
        clipped = learning.ranker_scores(model, np.stack([learning.vectorize(features)]))
        self.assertAlmostEqual(float(clipped[0]), 1.1, places=6)

    def _rejecting_ranker(self):
        from video_retrieval import learning

        weights = np.zeros((len(learning.FEATURE_NAMES), 1))
        weights[learning.FEATURE_NAMES.index("score"), 0] = -1.0
        return {"mean": [0.0] * len(learning.FEATURE_NAMES), "std": [1.0] * len(learning.FEATURE_NAMES),
                "layers": [{"w": weights.tolist(), "b": [0.0]}],
                "platt": {"a": 1.0, "b": 0.0},
                "conformal": {"threshold": 0.99, "coverage": 0.9}}

    def test_selection_reorders_but_never_empties_the_list(self):
        from video_retrieval import learning

        candidates = [{"candidate_id": i, "start": i * 10, "end": i * 10 + 8, "score": 1.0 - i * 0.1}
                      for i in range(8)]
        weights = np.zeros((len(learning.FEATURE_NAMES), 1))
        weights[learning.FEATURE_NAMES.index("score"), 0] = -1.0  # deliberately inverts the order
        model = {"mean": [0.0] * len(learning.FEATURE_NAMES), "std": [1.0] * len(learning.FEATURE_NAMES),
                 "layers": [{"w": weights.tolist(), "b": [0.0]}],
                 "platt": {"a": 1.0, "b": 0.0},
                 "conformal": {"threshold": 0.99, "coverage": 0.9}}
        with patch.object(learning, "load_ranker", return_value=model):
            # explore=0 isolates the threshold and the floor; exploration has its own test.
            kept, info = learning.select_candidates(candidates, [], {}, 100, keep_min=3, explore=0.0)
        self.assertEqual(len(kept), 3)          # the threshold rejects everything; the floor holds
        self.assertEqual(info["before"], 8)
        self.assertEqual(kept[0]["candidate_id"], 7)  # the inverted ranking really is applied
        self.assertIn("ranker_probability", kept[0])
        with patch.object(learning, "load_ranker", return_value=None), \
             patch.object(learning, "load_prefilter", return_value=None):
            untouched, absent = learning.select_candidates(candidates, [], {}, 100)
        self.assertIs(untouched, candidates)
        self.assertIsNone(absent)

    def test_selection_explores_past_its_own_beliefs(self):
        from video_retrieval import learning

        candidates = [{"candidate_id": i, "start": i * 10, "end": i * 10 + 8, "score": 1.0 - i * 0.1}
                      for i in range(8)]
        model = self._rejecting_ranker()
        with patch.object(learning, "load_ranker", return_value=model):
            # Exploring everything: the five the threshold rejected are verified anyway.
            everything, info = learning.select_candidates(
                candidates, [], {}, 100, keep_min=3, explore=1.0, rng=np.random.RandomState(0))
            # Exploring nothing is the old behaviour exactly.
            nothing, quiet = learning.select_candidates(
                candidates, [], {}, 100, keep_min=3, explore=0.0, rng=np.random.RandomState(0))
            half, sampled = learning.select_candidates(
                candidates, [], {}, 100, keep_min=3, explore=0.5, rng=np.random.RandomState(0))
        self.assertEqual((len(everything), info["explored"]), (8, 5))
        self.assertEqual((len(nothing), quiet["explored"]), (3, 0))
        self.assertTrue(all(row["explored"] for row in everything[3:]))
        self.assertFalse(any(row["explored"] for row in everything[:3]))
        # An explored row records the odds that brought it here, so training can correct for them.
        self.assertTrue(all(row["propensity"] == 0.5 for row in half if row["explored"]))
        self.assertTrue(all(row["propensity"] == 1.0 for row in half if not row["explored"]))
        self.assertEqual(sampled["explored"], sum(row["explored"] for row in half))

    def test_explored_rows_are_logged_with_their_propensity(self):
        from bench.rank import group_searches
        from video_retrieval import learning

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidates.jsonl"
            with patch.object(learning, "CANDIDATE_EXAMPLES", path), patch.object(learning, "LEARNING_DIR", Path(folder)):
                learning.log_candidates(
                    [{"candidate_id": 0, "start": 0, "end": 8, "score": 0.9},
                     {"candidate_id": 1, "start": 20, "end": 28, "score": 0.1,
                      "explored": True, "propensity": 0.1}],
                    [], {}, 40, "a person waves", survivors=[1])
            rows = learning.load_examples(path)
        self.assertEqual([row["explored"] for row in rows], [False, True])
        # The rare explored row stands for the ten like it that were never verified.
        group = group_searches(rows)[0]
        np.testing.assert_allclose(group["weights"], [1.0, 10.0])
        self.assertEqual(list(group["explored"]), [False, True])

    def test_verifier_decisions_are_logged_as_training_rows(self):
        from video_retrieval import learning

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidates.jsonl"
            with patch.object(learning, "CANDIDATE_EXAMPLES", path), patch.object(learning, "LEARNING_DIR", Path(folder)):
                learning.log_candidates(
                    [{"candidate_id": 0, "start": 0, "end": 8, "score": 0.9},
                     {"candidate_id": 1, "start": 20, "end": 28, "score": 0.4}],
                    [], {}, 40, "a person waves", survivors=[0])
            rows = learning.load_examples(path)
        self.assertEqual([row["label"] for row in rows], [1, 0])
        self.assertEqual(rows[0]["query"], "a person waves")


if __name__ == "__main__":
    unittest.main()
