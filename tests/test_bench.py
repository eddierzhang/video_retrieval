"""Benchmark construction, tuning, replay, tracking and the trainers' own logic."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from video_retrieval.pipeline import RetrievalResources, retrieve_video


class BenchTest(unittest.TestCase):
    def test_conformal_threshold_reaches_its_coverage_target(self):
        from bench.rank import calibrate_conformal, evaluate_sets

        # Twenty searches; the best confirmed candidate scores anywhere from 0.05 to 1.0.
        best = np.linspace(0.05, 1.0, 20)
        groups = []
        for value in best:
            groups.append({"labels": np.array([1.0, 0.0, 0.0, 0.0]),
                           "probabilities": np.array([value, 0.02, 0.01, 0.0])})
        probability_of = lambda group: group["probabilities"]
        conformal = calibrate_conformal(groups, probability_of, alpha=0.1)
        self.assertEqual(conformal["calibration_searches"], 20)
        # 90% coverage must keep all but the strongest-scoring couple of cuts.
        self.assertLessEqual(conformal["threshold"], float(np.quantile(best, 0.1)))
        measured = evaluate_sets(groups, probability_of, conformal["threshold"])
        self.assertGreaterEqual(measured["coverage"], 0.9)
        # A tighter alpha can only lower the bar, never raise it.
        loose = calibrate_conformal(groups, probability_of, alpha=0.5)
        self.assertGreaterEqual(loose["threshold"], conformal["threshold"])
        # Searches with nothing to find carry no information about coverage.
        empty = [{"labels": np.zeros(3), "probabilities": np.ones(3)}]
        self.assertIsNone(calibrate_conformal(empty, probability_of, alpha=0.1))

    def test_ranking_rows_group_by_search_and_split_whole(self):
        from bench.rank import group_searches, split_searches

        rows = []
        for search in range(12):
            for index in range(4):
                rows.append({"search": f"s{search}", "query": "same text for every search",
                             "at": 1000.0, "label": int(index == 0),
                             "features": {name: 0.1 for name in ["score", "channel_video"]}})
        groups = group_searches(rows)
        self.assertEqual(len(groups), 12)  # grouped by id, not by the identical query text
        self.assertEqual(len(groups[0]["labels"]), 4)
        fit, calibration, test = split_searches(groups)
        names = [{group["search"] for group in part} for part in (fit, calibration, test)]
        self.assertEqual(sum(len(part) for part in names), 12)
        self.assertFalse(names[0] & names[1] or names[1] & names[2] or names[0] & names[2])

    def test_adapter_split_keeps_every_text_of_a_span_together(self):
        from bench.adapt import split_groups

        groups = [{"video": f"v{index // 5}", "start": index * 10, "end": index * 10 + 8} for index in range(20)]
        train, holdout = split_groups(groups, holdout=0.25)
        self.assertEqual(len(train), len(groups))
        self.assertTrue((train ^ holdout).all())  # a span is on exactly one side
        self.assertGreaterEqual(holdout.sum(), 1)
        by_video = split_groups(groups, holdout=0.25, by="video")[1]
        held = {group["video"] for group, keep in zip(groups, by_video) if keep}
        self.assertTrue(all(by_video[index] == (groups[index]["video"] in held) for index in range(len(groups))))

    def test_adapter_masks_overlapping_spans_and_repeated_wording(self):
        from bench.adapt import build_mask

        groups = [
            {"video": "a", "start": 0, "end": 30},    # 0
            {"video": "a", "start": 15, "end": 45},   # 1 overlaps 0
            {"video": "a", "start": 60, "end": 90},   # 2 far away, same wording as 0
            {"video": "b", "start": 0, "end": 30},    # 3 another video
        ]
        text_groups = np.array([0, 1, 2, 3])
        vectors = np.zeros((4, 3), dtype=np.float32)
        vectors[0] = vectors[2] = [1, 0, 0]  # spans 0 and 2 described identically
        vectors[1] = [0, 1, 0]
        vectors[3] = [0, 0, 1]
        mask = build_mask(groups, text_groups, vectors, mask_iou=0.25)
        self.assertFalse(mask[np.arange(4), text_groups].any())  # never its own span
        self.assertTrue(mask[0, 1])   # overlapping in time
        self.assertTrue(mask[0, 2])   # same wording elsewhere in the video
        self.assertFalse(mask[1, 2])  # a genuine negative survives
        self.assertFalse(mask[0, 3])  # another video is never masked

    def test_adapter_evaluation_ranks_within_one_video(self):
        from bench.adapt import evaluate

        # Six spans in one video; each text matches its own span's second view exactly.
        groups = []
        for index in range(6):
            views = np.zeros((1, 2, 6), dtype=np.float32)
            views[0, 0] = 0.5
            views[0, 1, index] = 1.0
            groups.append({"video": "a", "start": index * 10, "end": index * 10 + 8, "views": views})
        texts = np.eye(6, dtype=np.float32)
        text_groups = np.arange(6)
        mask = np.zeros((6, 6), dtype=bool)
        perfect = evaluate(texts, text_groups, groups, mask, min_gallery=5)
        self.assertEqual(perfect["recall@1"], 1.0)
        self.assertEqual(perfect["texts"], 6)
        # A gallery smaller than min_gallery says nothing, so it is not scored at all.
        self.assertEqual(evaluate(texts[:2], text_groups[:2], groups[:2], mask[:2, :2], min_gallery=5)["texts"], 0)
        # Scoring against the wrong view would lose the signal the tiles carry.
        flattened = [dict(group, views=group["views"][:, :1]) for group in groups]
        self.assertLess(evaluate(texts, text_groups, flattened, mask, min_gallery=5)["recall@1"], 1.0)

    def test_tracking_records_failures_and_notices_changed_inputs(self):
        from bench import tracking

        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data.jsonl"
            data.write_text("one\n", encoding="utf-8")
            with tracking.Run("unit", {"alpha": 0.1}, seed=0, inputs=[data], root=folder) as first:
                first.summarize(score=0.5)
            data.write_text("two\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                with tracking.Run("unit", {"alpha": 0.2}, seed=0, inputs=[data], root=folder) as second:
                    raise ValueError("boom")
            with self.assertRaises(SystemExit):
                with tracking.Run("unit", {}, root=folder) as third:
                    raise SystemExit("not enough data yet")
            runs = {run["id"]: run for run in tracking.load_runs(folder)}
        self.assertEqual(runs[first.id]["status"], "done")
        self.assertEqual(runs[second.id]["status"], "failed")
        self.assertIn("boom", runs[second.id]["error"])
        self.assertEqual(runs[third.id]["status"], "stopped")  # an early exit is not a crash
        report = tracking.difference(runs[first.id], runs[second.id])
        self.assertEqual(report["args"]["alpha"], (0.1, 0.2))
        self.assertTrue(report["inputs"])  # same path, different bytes

    def test_constructed_timelines_never_repeat_source_seconds(self):
        import random

        from bench.construct import choose_segments, make_code

        def chunk(video, start, terms):
            return {"video": video, "video_id": video, "start": start, "end": start + 30,
                    "terms": terms, "actions": []}

        pool = [chunk("a", 0, ["tennis serve"]), chunk("a", 15, ["tennis rally"]),   # overlaps the first
                chunk("a", 60, ["tennis serve"]),                                     # repeats its wording
                chunk("a", 120, ["dog running park"]), chunk("b", 0, ["tennis serve"])]
        for seed in range(20):
            chosen = choose_segments(pool, 5, random.Random(seed))
            for i, left in enumerate(chosen):
                for right in chosen[i + 1:]:
                    if left["video_id"] == right["video_id"]:
                        self.assertFalse(left["start"] < right["end"] and right["start"] < left["end"])
                        self.assertNotEqual(left["terms"], right["terms"])
        code = make_code(random.Random(0))
        self.assertRegex(code, r"^[A-Z]{3}-[0-9]{4}$")
        self.assertFalse(set(code) & set("OI0158SB"))  # no glyphs a reader could honestly confuse

    def test_short_events_are_balanced_sliced_and_flag_their_look_alikes(self):
        import random

        from bench.construct import choose_segments, query_overlap, slice_chunk

        # One video with twenty chunks and three with one each: uniform draws fill up on the first.
        pool = [{"video": "tennis", "video_id": "tennis", "start": 40.0 * i, "end": 40.0 * i + 30,
                 "terms": [f"term{i}"], "actions": []} for i in range(20)]
        pool += [{"video": name, "video_id": name, "start": 0.0, "end": 30.0, "terms": [name], "actions": []}
                 for name in ("heart", "code", "court")]
        for seed in range(10):
            chosen = choose_segments(pool, 4, random.Random(seed), balanced=True)
            self.assertEqual(len({chunk["video_id"] for chunk in chosen}), 4)
        piece = slice_chunk(pool[3], random.Random(0), 4.0, 20.0)
        self.assertTrue(pool[3]["start"] <= piece["start"] < piece["end"] <= pool[3]["end"])
        self.assertTrue(4.0 <= piece["end"] - piece["start"] <= 20.0 + 1e-6)
        self.assertEqual(piece["queries"], [])  # the parent chunk's queries describe thirty seconds, not this
        self.assertGreaterEqual(query_overlap("person practicing tennis forehand on outdoor court",
                                              "person practicing tennis forehands on outdoor court"), 0.5)
        self.assertLess(query_overlap("man explains heart transplant", "person practicing tennis"), 0.5)

    def test_optimal_stopping_is_actually_optimal(self):
        from bench.replay import continuation_value

        generator = np.random.RandomState(0)
        for _ in range(300):
            count = int(generator.randint(1, 7))
            probabilities = np.sort(generator.uniform(0, 1, count))[::-1]
            costs = generator.uniform(1, 40, count)
            value = float(generator.uniform(5, 150))
            gains = probabilities * value - costs
            # Every stopping rule here is "verify the first k" (plus "stop at a hit" when only the
            # first match counts), so brute force is the best such k.
            every = max(0.0, max(np.cumsum(gains)))
            reach = np.concatenate([[1.0], np.cumprod(1.0 - probabilities)[:-1]])
            first = max(0.0, max(np.cumsum(reach * gains)))
            self.assertAlmostEqual(continuation_value(probabilities, costs, value, "all", 0), every, places=6)
            self.assertAlmostEqual(continuation_value(probabilities, costs, value, "first", 0), first, places=6)

    def test_replay_charges_what_a_policy_verifies_and_hides_outcomes(self):
        from bench.replay import replay

        group = {"labels": np.array([0.0, 1.0, 1.0, 0.0]), "costs": np.array([5.0, 7.0, 11.0, 13.0]),
                 "features": np.zeros((4, 3))}
        probability_of = lambda group: np.array([0.9, 0.2, 0.8, 0.1])  # order: 0, 2, 1, 3

        def two_then_stop(context):
            self.assertNotIn("labels", context)  # a live policy cannot see what it is deciding about
            return context["position"] < 2

        result = replay(group, two_then_stop, probability_of, value=100.0, goal="all")
        self.assertEqual((result["calls"], result["seconds"], result["found"]), (2, 16.0, 1))
        self.assertAlmostEqual(result["reward"], 100.0 - 16.0)
        first = replay(group, lambda context: True, probability_of, value=100.0, goal="first")
        self.assertEqual(first["found"], 2)
        self.assertAlmostEqual(first["reward"], 100.0 - 36.0)  # only the first hit is paid for

    def _tuning_fixture(self):
        plan = {"executor": "temporal_grounding", "return_mode": "all", "expected_duration": {"min_seconds": 2, "max_seconds": 6},
                "weights": {"video": 0.7, "metadata": 0.3, "transcript_semantic": 0.0, "transcript_bm25": 0.0}}
        hits = {"video": [[{"start": 30, "end": 38, "score": 0.9}, {"start": 70, "end": 78, "score": 0.6},
                           {"start": 100, "end": 104, "score": 0.3}]],
                "metadata": [[{"start": 34, "end": 44, "score": 0.8}, {"start": 72, "end": 80, "score": 0.7}]],
                "negative": [[{"start": 70, "end": 76, "score": 0.9}]],
                "transcript_semantic": [], "transcript_bm25": []}
        return plan, hits

    def test_tuning_replays_exactly_what_a_quick_search_answers(self):
        from bench import tune
        from video_retrieval import boundaries as learned_boundaries
        from video_retrieval.retrieval import scoring_override

        plan, hits = self._tuning_fixture()
        row = {"plan": plan, "results": hits, "duration": 120.0, "signal": None}
        unusual = tune.defaults()
        unusual["pipeline"].update(candidate_padding=2.0, evidence_bin_size=1.0, nms_iou_threshold=0.3,
                                   recursive_min_window_seconds=6.0)
        unusual["scoring"].update(negative_evidence_weight=1.4, peak_share=0.5, window_peak=0.9)
        for settings in (tune.defaults(), unusual):
            with patch("video_retrieval.pipeline.plan_query", return_value=plan), \
                    patch("video_retrieval.pipeline.run_retrieval_plan", return_value=hits), \
                    patch.object(learned_boundaries, "load_boundary_model", return_value=None), \
                    patch("video_retrieval.pipeline.materialize_final_matches",
                          side_effect=lambda manifest, instances, query, **kw: (instances, "results.json")), \
                    scoring_override(settings["scoring"]):
                live = retrieve_video("a car", RetrievalResources(manifest={"video": {"duration": 120}}),
                                      run_verification=False, **settings["pipeline"])
            expected = [(m["start"], m["end"]) for m in sorted(live["matches"], key=lambda m: -m["confidence"])]
            replayed = [(m["start"], m["end"]) for m in tune.replay(row, settings, None)]
            self.assertTrue(expected)
            self.assertEqual(replayed, expected)
        # The unusual settings really do change the answer, so the comparison above means something.
        self.assertNotEqual(tune.replay(row, tune.defaults(), None), tune.replay(row, unusual, None))

    def test_tuning_reports_by_event_length_and_fills_in_missing_settings(self):
        from bench import tune

        rows = [{"expect": {"start": 0, "end": length}} for length in (5, 12, 30)]
        scored = [(rows[0], 0.2, 0.2, 0.0), (rows[1], 0.6, 0.6, 1.0), (rows[2], 0.9, 0.9, 1.0)]
        report = tune.by_duration(scored)
        self.assertEqual(set(report), {"under 8 s", "8-15 s", "25 s and over"})
        self.assertAlmostEqual(report["under 8 s"]["top1_iou"], 0.2)
        partial = tune.settings_from({"pipeline": {"candidate_padding": 3.0}, "scoring": {}})
        self.assertEqual(partial["pipeline"]["candidate_padding"], 3.0)
        self.assertEqual(partial["pipeline"]["nms_iou_threshold"], tune.defaults()["pipeline"]["nms_iou_threshold"])
        self.assertEqual(tune.timeline_of({"video": "Constructed short timeline 1.mp4", "timeline": 1}),
                         "Constructed short timeline 1.mp4")

    def test_tuning_search_stays_in_bounds_and_folds_never_share_a_timeline(self):
        import inspect
        import random

        from bench import tune

        signature = inspect.signature(retrieve_video).parameters
        self.assertEqual(tune.defaults()["pipeline"]["candidate_padding"], signature["candidate_padding"].default)
        generator = random.Random(0)
        for _ in range(200):
            values = tune.perturb(tune.sample(tune.SCORING_SPACE, generator), tune.SCORING_SPACE, generator, 0.5)
            for name, (kind, low, high) in tune.SCORING_SPACE.items():
                self.assertTrue(low <= values[name] <= high)
        rows = [{"timeline": t, "id": f"{t}-{i}"} for t in range(7) for i in range(3)]
        folds = tune.folds_by_timeline(rows, 3, seed=0)
        self.assertEqual(sum(len(fold) for fold in folds), len(rows))
        timelines = [{row["timeline"] for row in fold} for fold in folds]
        for i, left in enumerate(timelines):
            for right in timelines[i + 1:]:
                self.assertFalse(left & right)


if __name__ == "__main__":
    unittest.main()
