"""Detect search: shots, tracking, crop and clip scoring, feedback. Models are never loaded."""
import unittest

import numpy as np

from video_retrieval import detect_search as detect
from video_retrieval import shots


def grey(value, height=36, width=64):
    return np.full((height, width), float(value), dtype=np.float32)


class ShotTest(unittest.TestCase):
    def test_a_cut_is_a_peak_well_above_the_surrounding_motion(self):
        times = np.arange(48) / 12.0
        scores = np.full(48, 0.05, dtype=np.float32)
        scores[24] = 0.8                      # one clean cut at 2 s
        cuts = shots.find_cuts(scores, times)
        self.assertEqual(cuts, [2.0])

        busy = np.full(48, 0.35, dtype=np.float32)   # constant motion, never a cut
        busy[24] = 0.5
        self.assertEqual(shots.find_cuts(busy, times), [])

        close = scores.copy()
        close[27] = 0.9                       # a second peak a quarter second later is the same transition
        self.assertEqual(len(shots.find_cuts(close, times)), 1)

    def test_change_scores_see_both_colour_and_pixels(self):
        histograms = np.zeros((3, 128), dtype=np.float32)
        histograms[0, 0] = histograms[1, 0] = 1.0
        histograms[2, 5] = 1.0                # the third frame has entirely different colours
        greys = np.stack([grey(10), grey(10), grey(200)])
        scores = shots.change_scores(histograms, greys)
        self.assertEqual(float(scores[0]), 0.0)
        self.assertLess(float(scores[1]), 0.05)
        self.assertGreater(float(scores[2]), 0.8)

    def test_something_passing_the_camera_is_not_a_cut(self):
        before = grey(100)
        whole = grey(220)                     # the entire picture replaced: a cut
        self.assertTrue(shots.is_real_cut(before, whole))

        passing = before.copy()
        passing[:, :12] = 250.0               # a strip down one side changes, the rest stays put
        self.assertFalse(shots.is_real_cut(before, passing))
        half = before.copy()
        half[:, :32] = 250.0                  # half the picture changing is already treated as a cut
        self.assertTrue(shots.is_real_cut(before, half))

        dark_before, dark_after = grey(8), grey(8)
        dark_after[10:20, 10:30] = 30.0       # a transition through near-black keeps its cut
        self.assertTrue(shots.is_real_cut(dark_before, dark_after))


def unit(index, size=8, mix=None):
    """A unit vector along `index`, optionally leaning `mix=(other, weight)` toward another axis."""
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    if mix:
        vector[mix[0]] = mix[1]
    return vector / np.linalg.norm(vector)


def observation(box, embedding, probability=0.8, score=0.6, **extra):
    return {"box": box, "embedding": embedding, "probability": probability, "score": score, **extra}


class PlanTest(unittest.TestCase):
    def test_a_description_without_a_subject_is_given_the_category(self):
        self.assertEqual(detect.with_subject("carrying two riders", "motorcycle"), "a motorcycle carrying two riders")
        self.assertEqual(detect.with_subject("a woman in red", "person"), "a woman in red")
        self.assertEqual(detect.with_subject("person wearing a uniform", "person"), "person wearing a uniform")

    def test_normalization_clamps_counts_and_always_has_something_to_score(self):
        plan = detect.normalize_plan({"object": " dog ", "target": "black  fur", "min_count": 99, "with_count": -3}, "q")
        self.assertEqual((plan["object"], plan["target"]), ("dog", "a dog black fur"))
        self.assertEqual((plan["min_count"], plan["with_count"]), (20, 0))
        self.assertEqual(len(plan["contrasts"]), 1)   # a softmax over the target alone would always say 1
        self.assertEqual(detect.normalize_plan({}, "someone waves")["action"], "someone waves")


class RelationTest(unittest.TestCase):
    def test_riders_on_a_motorcycle_are_counted_and_a_bystander_is_not(self):
        motorcycle = (0.30, 0.40, 0.60, 0.90)
        riders = [(0.32, 0.20, 0.45, 0.70), (0.44, 0.22, 0.58, 0.72)]
        bystander = (0.80, 0.30, 0.90, 0.80)
        self.assertEqual(detect.attached_count(motorcycle, riders + [bystander]), 2)

    def test_duplicate_boxes_collapse_and_labels_match_by_word(self):
        boxes = [(0.1, 0.1, 0.5, 0.5), (0.11, 0.1, 0.5, 0.52), (0.6, 0.6, 0.9, 0.9)]
        self.assertEqual(len(detect.dedupe_boxes(boxes)), 2)
        self.assertTrue(detect.label_matches("police officer", "officer"))
        self.assertFalse(detect.label_matches("car", "person"))


class TrackingTest(unittest.TestCase):
    def test_overlapping_boxes_stay_one_track_and_a_long_gap_starts_another(self):
        a, b = unit(0), unit(1)
        frames = [
            (0.0, [observation((0.10, 0.1, 0.30, 0.5), a), observation((0.60, 0.1, 0.80, 0.5), b)]),
            (0.5, [observation((0.12, 0.1, 0.32, 0.5), a), observation((0.62, 0.1, 0.82, 0.5), b)]),
            (1.0, [observation((0.14, 0.1, 0.34, 0.5), a)]),
            (1.5, []), (2.0, []), (2.5, []),
            (3.0, [observation((0.64, 0.1, 0.84, 0.5), b)]),
        ]
        tracks = detect.build_tracks(frames, shot_id=0)
        self.assertEqual([len(t["observations"]) for t in tracks], [3, 2, 1])
        self.assertEqual([o["time"] for o in tracks[0]["observations"]], [0.0, 0.5, 1.0])

    def test_a_fast_object_links_by_appearance_but_not_across_the_frame(self):
        rider = unit(2)
        fast = [(0.0, [observation((0.10, 0.4, 0.20, 0.6), rider)]),
                (0.5, [observation((0.30, 0.4, 0.40, 0.6), rider)])]    # no overlap, same look, 0.2 away
        self.assertEqual(len(detect.build_tracks(fast, 0)), 1)
        far = [(0.0, [observation((0.00, 0.4, 0.10, 0.6), rider)]),
               (0.5, [observation((0.85, 0.4, 0.95, 0.6), rider)])]     # same look, other side of the frame
        self.assertEqual(len(detect.build_tracks(far, 0)), 2)

    def test_identities_join_lookalikes_and_bridge_false_cuts(self):
        shots_by_id = {0: {"id": 0, "start": 0.0, "end": 2.0}, 1: {"id": 1, "start": 2.0, "end": 4.0},
                       2: {"id": 2, "start": 4.0, "end": 6.0}}
        box = (0.2, 0.2, 0.5, 0.8)
        track = lambda shot, time, embedding, where=box: {"shot": shot, "observations": [
            {"time": time, "box": where, "embedding": embedding}]}
        tracks = [
            track(0, 1.8, unit(0)),
            track(1, 2.2, unit(1)),                          # looks different, but continues in place across 2 s
            track(2, 5.0, unit(0, mix=(3, 0.1))),            # far away in time, looks like the first
            track(2, 5.0, unit(4), (0.7, 0.2, 0.9, 0.8)),    # something else entirely
        ]
        bridged = detect.assign_identities(tracks, shots_by_id)
        self.assertEqual(bridged, {(0, 1)})
        identities = [t["identity"] for t in tracks]
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[0], identities[2])
        self.assertNotEqual(identities[0], identities[3])


def make_state(tracks, plan=None, windows=None, shots_=None, bridged=None):
    plan = {"object": "person", "target": "a person in uniform", "contrasts": ["a person"], "min_count": 1,
            "with_object": "", "with_count": 0, "action": "", "action_contrasts": [], **(plan or {})}
    shots_ = shots_ or [{"id": 0, "start": 0.0, "end": 10.0}]
    for number, track in enumerate(tracks):
        track.setdefault("id", number)
        track.setdefault("identity", number)
    return {"query": "q", "plan": plan, "scale": 10.0, "tracks": tracks, "windows": windows or [],
            "shots": shots_, "shot_fps": {shot["id"]: 2.0 for shot in shots_}, "bridged": bridged or set(),
            "thresholds": {"attribute": 0.5, "action": 2.0}}


def steady_track(times, probability, embedding, shot=0, box=(0.2, 0.2, 0.4, 0.8), **extra):
    return {"shot": shot, "observations": [{"time": t, "box": box, "embedding": embedding,
                                            "probability": probability, "score": 0.6, **extra} for t in times]}


class AssembleTest(unittest.TestCase):
    def test_a_persistent_confident_track_is_a_result_and_flickers_are_not(self):
        state = make_state([
            steady_track([1.0, 1.5, 2.0], 0.8, unit(0)),
            steady_track([6.0], 0.9, unit(1)),                  # one sample: glare, not an object
            steady_track([8.0, 8.5, 9.0], 0.2, unit(2)),        # in view, but does not look like the target
        ])
        results = detect.assemble(state)
        self.assertEqual(len(results), 1)
        self.assertEqual((results[0]["start"], results[0]["end"]), (0.75, 2.25))
        self.assertAlmostEqual(results[0]["confidence"], 0.8)
        self.assertEqual(results[0]["match_key"], "0:0.75")

    def test_counts_and_relations_decide(self):
        people = [steady_track([1.0, 1.5], 0.8, unit(0)), steady_track([1.0, 1.5, 2.0, 2.5], 0.8, unit(1))]
        results = detect.assemble(make_state(people, plan={"min_count": 2}))
        self.assertEqual([(r["start"], r["end"], r["count"]) for r in results], [(0.75, 1.75, 2)])

        unrelated = make_state([steady_track([1.0, 1.5], 0.9, unit(0), related=False)],
                               plan={"with_object": "person", "with_count": 2})
        self.assertEqual(detect.assemble(unrelated), [])

    def test_motion_is_judged_against_chance(self):
        windows = [
            {"shot": 0, "start": 0.0, "end": 2.0, "probability": 0.35, "choices": 8, "embedding": unit(0)},  # 2.8x chance
            {"shot": 0, "start": 5.0, "end": 7.0, "probability": 0.35, "choices": 2, "embedding": unit(1)},  # 0.7x chance
        ]
        state = make_state([], plan={"object": "", "target": "", "contrasts": [], "action": "slicing bread"}, windows=windows)
        results = detect.assemble(state)
        self.assertEqual([(r["start"], r["end"]) for r in results], [(0.0, 2.0)])

    def test_with_an_action_appearance_shades_confidence_instead_of_gating(self):
        tracks = [steady_track([1.0, 1.5, 2.0], 0.2, unit(0))]
        windows = [{"shot": 0, "start": 0.5, "end": 2.5, "probability": 0.5, "choices": 8, "embedding": unit(0)}]
        state = make_state(tracks, plan={"object": "cat", "target": "a cat with a raised paw", "action": "a cat swatting"},
                           windows=windows)
        results = detect.assemble(state)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["confidence"], 0.6)   # (0.5 + 0.5 * 0.2) * min(1, 4 / 4)

    def test_results_either_side_of_a_bridged_boundary_join(self):
        shots_ = [{"id": 0, "start": 0.0, "end": 2.0}, {"id": 1, "start": 2.0, "end": 4.0}]
        tracks = [steady_track([1.0, 1.5], 0.8, unit(0), shot=0), steady_track([2.5, 3.0], 0.8, unit(0), shot=1)]
        tracks[1]["identity"] = 0
        results = detect.assemble(make_state(tracks, shots_=shots_, bridged={(0, 1)}))
        self.assertEqual([(r["start"], r["end"]) for r in results], [(0.75, 3.25)])


class FeedbackTest(unittest.TestCase):
    def test_probability_moves_toward_confirmed_and_away_from_rejected_examples(self):
        crop = unit(0)
        positives, negatives = np.stack([unit(0, mix=(1, 0.2))]), np.stack([unit(2)])
        self.assertEqual(detect.adjusted_probability(0.4, crop, [], [], scale=10.0), 0.4)
        self.assertGreater(detect.adjusted_probability(0.4, crop, positives, negatives, scale=10.0), 0.5)
        self.assertLess(detect.adjusted_probability(0.6, unit(2), positives, negatives, scale=10.0), 0.5)

    def test_marks_rescore_lookalikes_without_touching_the_detector(self):
        uniform, jacket = unit(0), unit(1)
        state = make_state([
            steady_track([1.0, 1.5, 2.0], 0.8, uniform),
            steady_track([5.0, 5.5, 6.0], 0.8, jacket),
            steady_track([8.0, 8.5, 9.0], 0.45, unit(0, mix=(2, 0.1))),   # just under the line, looks like a uniform
        ])
        first = detect.assemble(state)
        self.assertEqual([r["match_key"] for r in first], ["0:0.75", "0:4.75"])
        state["last_matches"] = first
        feedback = {"0:0.75": "positive", "0:4.75": "negative"}
        again = detect.assemble(state, feedback=feedback)
        keys = [r["match_key"] for r in again]
        self.assertIn("0:7.75", keys)
        self.assertNotIn("0:4.75", keys)

    def test_refine_keeps_marked_verdicts_and_saves_its_state(self):
        import pickle
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch

        state = make_state([steady_track([1.0, 1.5, 2.0], 0.8, unit(0)), steady_track([5.0, 5.5, 6.0], 0.8, unit(1))])
        state["last_matches"] = detect.assemble(state)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "detect_state.pkl").write_bytes(pickle.dumps(state))
            resources = Mock(manifest={"video": {"path": "video.mp4", "duration": 10.0}})
            with patch.object(detect, "finish", return_value={"matches": []}) as finish:
                detect.refine(folder, {"0:4.75": "negative"}, resources)
            candidates = finish.call_args.args[1]
            self.assertEqual([c["match_key"] for c in candidates], ["0:0.75"])
            self.assertEqual(finish.call_args.args[-1]["refined_with"], {"positive": 0, "negative": 1})
            saved = finish.call_args.args[0]
            self.assertEqual(len(saved["last_matches"]), 2)   # the rejected result stays known for later marks

    def test_a_rejected_clip_stays_rejected_when_its_start_moves(self):
        rejected = {"shot": 0, "start": 4.75, "end": 6.25}
        self.assertTrue(detect.mostly_inside({"shot": 0, "start": 5.25, "end": 6.25}, rejected))
        self.assertFalse(detect.mostly_inside({"shot": 1, "start": 5.25, "end": 6.25}, rejected))
        self.assertFalse(detect.mostly_inside({"shot": 0, "start": 6.0, "end": 9.0}, rejected))


if __name__ == "__main__":
    unittest.main()
