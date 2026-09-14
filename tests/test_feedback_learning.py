"""Learning from right/wrong marks: the scorer, its evaluation and activation, and remembered examples."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from video_retrieval import feedback_learning as learning
from video_retrieval import detect_search as detect


def features(right, rng, rule_pass=1):
    """Results where wrong ones are small boxes on the frame edge and right ones are not, plus noise."""
    return {
        "rule_pass": rule_pass, "rule_confidence": rng.uniform(0.5, 1.0), "appearance": rng.uniform(0.5, 1.0),
        "detector_score": rng.uniform(0.3, 0.8), "samples": rng.integers(2, 8), "duration": rng.uniform(1, 5),
        "box_area": rng.uniform(0.05, 0.3) if right else rng.uniform(0.002, 0.01),
        "edge": rng.uniform(0.0, 0.2) if right else rng.uniform(0.6, 1.0),
        "has_object": 1, "has_target": 1,
    }


def unit(index, size=8):
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


class ScorerTest(unittest.TestCase):
    def test_logistic_regression_separates_a_learnable_pattern(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 200)
        X = [learning.vectorize(features(bool(label), rng)) for label in labels]
        model = learning.fit_logistic(X, labels)
        accuracy = np.mean((learning.predict(model, X) >= 0.5) == labels)
        self.assertGreater(accuracy, 0.95)
        constant = [i for i, name in enumerate(learning.FEATURES) if name == "rule_pass"][0]
        self.assertEqual(model["weights"][constant], 0.0)   # the same for every example, so it teaches nothing

    def test_missing_measurements_count_as_zero(self):
        self.assertEqual(len(learning.vectorize({})), len(learning.FEATURES))


class LearnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def mark(self, learner, count, rng, videos=3, pattern=True, concept="dog | a fluffy dog |  |  | "):
        status = None
        for i in range(count):
            right = bool(i % 2)
            label_right = right if pattern else bool(rng.integers(0, 2))
            example = {"features": features(right, rng), "embedding": unit(0 if label_right else 1).astype(np.float16),
                       "concept": concept}
            status = learner.record(f"v{i % videos}", f"s{i % 5}", f"0:{i}.00", "positive" if label_right else "negative", example)
        return status

    def test_it_collects_until_there_is_enough_to_learn_from(self):
        learner = learning.Learner(self.folder)
        status = self.mark(learner, 20, np.random.default_rng(1))
        self.assertEqual((status["state"], status["examples"], status["active"]), ("collecting", 20, False))
        self.assertIsNone(learner.scorer())

        one_video = learning.Learner(self.folder / "one")
        self.assertEqual(self.mark(one_video, 60, np.random.default_rng(1), videos=1)["state"], "collecting")

    def test_it_takes_over_only_when_it_beats_the_rules_on_unseen_videos(self):
        learner = learning.Learner(self.folder)
        status = self.mark(learner, 60, np.random.default_rng(2))
        self.assertEqual(status["state"], "active")
        self.assertGreater(status["evaluation"]["learned_accuracy"], status["evaluation"]["rules_accuracy"])
        scorer = learner.scorer()
        rng = np.random.default_rng(3)
        self.assertGreater(scorer.probability(features(True, rng)), 0.5)
        self.assertLess(scorer.probability(features(False, rng)), 0.5)

        noise = learning.Learner(self.folder / "noise")
        status = self.mark(noise, 60, np.random.default_rng(4), pattern=False)
        self.assertEqual(status["state"], "not_better")   # labels unrelated to the measurements
        self.assertIsNone(noise.scorer())

    def test_marks_persist_change_and_clear(self):
        learner = learning.Learner(self.folder)
        self.mark(learner, 60, np.random.default_rng(2))
        reloaded = learning.Learner(self.folder)
        self.assertEqual(reloaded.status()["examples"], 60)
        self.assertEqual(reloaded.status()["version"], learner.status()["version"])
        reloaded.record("v0", "s0", "0:0.00", None)
        self.assertEqual(reloaded.status()["examples"], 59)
        self.assertGreater(len(reloaded.status()["history"]), 1)

    def test_near_misses_need_more_confidence_until_enough_are_labelled(self):
        scorer = learning.Scorer({"version": 1, "mean": [0.0], "std": [1.0], "weights": [0.0], "bias": 0.0}, near_miss_labels=0)
        self.assertTrue(scorer.keeps({"rule_pass": 1}, 0.6))
        self.assertFalse(scorer.keeps({"rule_pass": 0}, 0.6))
        experienced = learning.Scorer({"version": 1, "mean": [0.0], "std": [1.0], "weights": [0.0], "bias": 0.0},
                                      near_miss_labels=learning.MIN_NEAR_MISS_LABELS)
        self.assertTrue(experienced.keeps({"rule_pass": 0}, 0.6))

    def test_memory_returns_marks_for_the_same_thing_from_other_searches(self):
        learner = learning.Learner(self.folder)
        plan = {"object": "dog", "target": "a fluffy dog"}
        concept = learning.concept_key(plan)
        example = lambda index: {"features": {}, "embedding": unit(index).astype(np.float16), "concept": concept}
        learner.record("v0", "s1", "0:1.00", "positive", example(0))
        learner.record("v0", "s1", "0:2.00", "negative", example(1))
        learner.record("v1", "s2", "0:1.00", "positive", example(2))
        learner.record("v1", "s3", "0:1.00", "positive", {**example(3), "concept": learning.concept_key({"object": "cat"})})
        positives, negatives = learner.memory({"object": " Dog", "target": "a  fluffy dog"}, exclude_search="s2")
        self.assertEqual((len(positives), len(negatives)), (1, 1))
        self.assertEqual(float(positives[0][0]), 1.0)


def detect_state(tracks):
    plan = {"object": "person", "target": "a person in uniform", "contrasts": ["a person"], "min_count": 1,
            "with_object": "", "with_count": 0, "action": "", "action_contrasts": []}
    for number, track in enumerate(tracks):
        track.update(id=number, identity=number)
    return {"query": "q", "plan": plan, "scale": 10.0, "tracks": tracks, "windows": [],
            "shots": [{"id": 0, "start": 0.0, "end": 10.0}], "shot_fps": {0: 2.0}, "bridged": set(),
            "thresholds": {"attribute": 0.5, "action": 2.0}, "shot_priority": {0: 0.12}}


def track(times, probability, box, embedding):
    return {"shot": 0, "observations": [{"time": t, "box": box, "embedding": embedding, "probability": probability,
                                         "score": 0.5} for t in times]}


class ProposeTest(unittest.TestCase):
    def state(self):
        return detect_state([
            track([1.0, 1.5, 2.0], 0.8, (0.2, 0.2, 0.5, 0.8), unit(0)),     # a real person
            track([5.0, 5.5], 0.9, (0.0, 0.4, 0.05, 0.47), unit(1)),       # a sliver on the frame edge
            track([8.0, 8.5, 9.0], 0.4, (0.3, 0.2, 0.6, 0.8), unit(0)),    # just under the rules' threshold
        ])

    def test_results_carry_the_measurements_the_scorer_reads(self):
        results = detect.propose(self.state())
        self.assertEqual([r["match_key"] for r in results], ["0:0.75", "0:4.75"])
        measured = results[1]["features"]
        self.assertEqual((measured["rule_pass"], measured["edge"], measured["samples"]), (1, 1.0, 2))
        self.assertAlmostEqual(measured["shot_priority"], 0.12)
        self.assertTrue(all(r["decided_by"] == "rules" for r in results))

    def test_a_learned_scorer_rejects_rule_false_positives_and_recovers_near_misses(self):
        class EdgeHater:
            version = 3

            def probability(self, features):
                return 0.1 if features["edge"] > 0.5 else 0.9

            def keeps(self, features, probability):
                return probability >= (0.5 if features["rule_pass"] else 0.7)

        results = detect.propose(self.state(), scorer=EdgeHater())
        self.assertEqual([(r["match_key"], r["rule_pass"]) for r in results], [("0:0.75", True), ("0:7.75", False)])
        self.assertEqual({r["decided_by"] for r in results}, {"learned"})
        self.assertAlmostEqual(results[0]["confidence"], 0.9)

    def test_remembered_marks_shift_scores_like_marks_on_this_search(self):
        memory = (np.stack([unit(0)]), np.stack([unit(1)]))
        results = detect.propose(self.state(), memory=memory)
        self.assertEqual([r["match_key"] for r in results], ["0:0.75", "0:7.75"])   # the edge sliver looks like a rejected one

    def test_a_mark_becomes_an_example_with_its_measurements(self):
        import pickle

        state = self.state()
        state["last_matches"] = detect.propose(state)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "detect_state.pkl").write_bytes(pickle.dumps(state))
            example = detect.example_for(folder, "0:4.75")
            self.assertEqual(example["features"]["edge"], 1.0)
            self.assertEqual(example["concept"], learning.concept_key(state["plan"]))
            self.assertAlmostEqual(float(np.linalg.norm(example["embedding"].astype(np.float32))), 1.0, places=3)
            self.assertIsNone(detect.example_for(folder, "9:9.99"))


if __name__ == "__main__":
    unittest.main()
