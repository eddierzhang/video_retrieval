import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_retrieval import local_backend as local
from video_retrieval.embeddings import embed_scale, embed_text
from video_retrieval.metadata import generate_metadata
from video_retrieval.pipeline import RetrievalResources, VideoRetrievalPipeline, retrieve_video
from video_retrieval.retrieval import plan_query
from video_retrieval.verification import call_video_json
from video_retrieval.video import generate_windows
from video_retrieval.visual_text import call_images_json


class LocalArchitectureTest(unittest.TestCase):
    def test_short_upload_keeps_each_hierarchy_scale(self):
        self.assertEqual(generate_windows(12, 120, 60), [(0.0, 12)])
        self.assertEqual(generate_windows(1, 8, 4), [(0.0, 1)])
        with self.assertRaises(ValueError):
            generate_windows(12, 8, 0)

    def test_model_boundaries_use_the_local_runtime(self):
        manifest = {"video": {"path": "video.mp4", "duration": 30}}
        with patch("requests.post", side_effect=AssertionError("unexpected HTTP request")):
            with patch.object(local, "embed_text", return_value=np.ones(512, np.float32)) as text:
                self.assertEqual(embed_text("test").shape, (512,))
                text.assert_called_once_with("test")
            with patch.object(local, "interval_json", return_value={"ok": True}) as interval:
                call_video_json(manifest, 2, 5, "prompt", {}, role="verifier", include_speech=False)
                interval.assert_called_once_with("video.mp4", 2, 5, "prompt", {}, role="verifier", include_speech=False)
            with patch.object(local, "images_json", return_value={"text": "ABC"}):
                self.assertEqual(call_images_json([np.zeros((2, 2, 3), np.uint8)], "read", {}), {"text": "ABC"})
            with patch.object(local, "chat_json", return_value={"weights": {"video": 1}}) as chat:
                self.assertEqual(plan_query("person")["weights"]["video"], 1)
                self.assertEqual(chat.call_args.kwargs["role"], "planner")

    def test_embed_scale_reads_each_interval_from_the_source(self):
        manifest = {"video": {"path": "v.mp4"}, "chunks": {"fine": [
            {"chunk_id": "fine_0", "start": 0, "end": 8},
            {"chunk_id": "fine_1", "start": 4, "end": 12},
        ]}}
        with tempfile.TemporaryDirectory() as folder, patch.object(
            local, "embed_video_interval", side_effect=lambda path, start, end: np.full((2, 4), start, np.float32)
        ) as embed:
            vectors, rows = embed_scale(manifest, "fine", save_dir=folder)
        self.assertEqual([call.args for call in embed.call_args_list], [("v.mp4", 0.0, 8.0), ("v.mp4", 4.0, 12.0)])
        self.assertEqual(vectors.shape, (2, 2, 4))  # chunks, views, dim
        self.assertEqual([row["chunk_id"] for row in rows], ["fine_0", "fine_1"])

    def test_interval_embedding_pools_each_view_over_time(self):
        times = np.arange(10, dtype=np.float32)
        # Two views per frame: a constant whole-frame view and a tile that changes.
        vectors = np.zeros((10, 2, 3), dtype=np.float32)
        vectors[:, 0, 0] = 1.0
        vectors[2, 1, 1] = 1.0
        vectors[3, 1, 2] = 1.0
        with patch.object(local, "frame_embeddings", return_value=(times, vectors)):
            inside = local.embed_video_interval("v.mp4", 2, 4)
            outside = local.embed_video_interval("v.mp4", 20, 22)
        self.assertEqual(inside.shape, (2, 3))
        np.testing.assert_allclose(inside[0], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(inside[1], [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-6)
        # Outside every frame, the nearest frame is used as-is.
        np.testing.assert_allclose(outside, [[1, 0, 0], [0, 0, 0]], atol=1e-6)

    def test_frame_views_tile_the_image(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        views = local.frame_views(frame, grid=2)
        self.assertEqual(len(views), 5)
        self.assertEqual(views[0].shape, (48, 64, 3))
        self.assertTrue(all(view.shape == (24, 32, 3) for view in views[1:]))

    def test_index_scores_a_chunk_by_its_best_view(self):
        from video_retrieval.embeddings import HierarchicalVideoIndex

        # Chunk 0 matches weakly as a whole; chunk 1 matches only in one tile.
        vectors = np.array([[[1, 0], [1, 0]], [[0.2, 0.98], [0, 1]]], dtype=np.float32)
        rows = [{"chunk_id": "a", "start": 0, "end": 8, "duration": 8},
                {"chunk_id": "b", "start": 8, "end": 16, "duration": 8}]
        index = HierarchicalVideoIndex({"fine": vectors}, {"fine": rows})
        scores, ids = index.search(np.array([[0, 1]], dtype=np.float32), 2)
        self.assertEqual(int(ids[0][0]), 1)
        self.assertAlmostEqual(float(scores[0][0]), 1.0, places=5)

    def test_text_embeddings_use_the_dedicated_model_and_task_prefixes(self):
        response = Mock(status_code=200)
        response.json.return_value = {"embeddings": [[3.0, 4.0]]}
        with patch("requests.post", return_value=response) as post:
            document = local.embed_document("a person opens a door")
            self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/embed")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "nomic-embed-text")
            self.assertTrue(payload["input"].startswith("search_document: "))
            local.embed_query("door opening")
            self.assertTrue(post.call_args.kwargs["json"]["input"].startswith("search_query: "))
        np.testing.assert_allclose(document, [0.6, 0.8], atol=1e-6)  # normalized

    def test_index_built_by_an_older_version_is_refused(self):
        from video_retrieval.local_indexing import IndexVersionMismatch, load_index

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "ready.json").write_text(json.dumps({"models": {"version": 1}, "has_transcript": False}))
            with self.assertRaises(IndexVersionMismatch):
                load_index(path)

    def test_faiss_index_accepts_multi_view_embeddings(self):
        from video_retrieval.embeddings import build_faiss_index

        # A [:, 0] view of a 3-D array is not contiguous; FAISS refuses those.
        index = build_faiss_index(np.random.rand(3, 5, 8).astype(np.float32))
        self.assertEqual((index.ntotal, index.d), (3, 8))

    def test_untagged_model_names_match_latest_tags(self):
        # Ollama reports an untagged pull as "name:latest"; a bare name must still match.
        installed = {"nomic-embed-text:latest": {}, "qwen3.5:4b": {}}
        with patch.object(local, "installed_models", return_value=installed), \
                patch.object(local, "capabilities", return_value=frozenset({"vision"})):
            digests = local.check_runtime(local.LocalModels(planner="qwen3.5:4b", vision="qwen3.5:4b", verifier="qwen3.5:4b"))
        self.assertIn("qwen3.5:4b", digests)
        with patch.object(local, "installed_models", return_value={"qwen3.5:4b": {}}), \
                patch.object(local, "capabilities", return_value=frozenset({"vision"})):
            with self.assertRaises(RuntimeError) as error:
                local.check_runtime(local.LocalModels())
        self.assertIn("nomic-embed-text", str(error.exception))

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
            kept, info = learning.select_candidates(candidates, [], {}, 100, keep_min=3)
        self.assertEqual(len(kept), 3)          # the threshold rejects everything; the floor holds
        self.assertEqual(info["before"], 8)
        self.assertEqual(kept[0]["candidate_id"], 7)  # the inverted ranking really is applied
        self.assertIn("ranker_probability", kept[0])
        with patch.object(learning, "load_ranker", return_value=None), \
             patch.object(learning, "load_prefilter", return_value=None):
            untouched, absent = learning.select_candidates(candidates, [], {}, 100)
        self.assertIs(untouched, candidates)
        self.assertIsNone(absent)

    def _rejecting_ranker(self):
        from video_retrieval import learning

        weights = np.zeros((len(learning.FEATURE_NAMES), 1))
        weights[learning.FEATURE_NAMES.index("score"), 0] = -1.0
        return {"mean": [0.0] * len(learning.FEATURE_NAMES), "std": [1.0] * len(learning.FEATURE_NAMES),
                "layers": [{"w": weights.tolist(), "b": [0.0]}],
                "platt": {"a": 1.0, "b": 0.0},
                "conformal": {"threshold": 0.99, "coverage": 0.9}}

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

    def test_verification_windows_stay_short_enough_to_see_detail(self):
        import inspect

        from video_retrieval.verification import _video_windows, verify_candidate

        window = inspect.signature(verify_candidate).parameters["verifier_window_seconds"].default
        self.assertLessEqual(window, 30)
        # A 60s candidate is split, so each call spends its frame budget on less time.
        self.assertGreaterEqual(len(_video_windows(0, 60, window=window, overlap=5)), 3)

    def test_vision_prompt_uses_clip_relative_frames_and_speech(self):
        transcript = {"words": [], "segments": [
            {"start": 0, "end": 2, "text": "hello"},
            {"start": 5, "end": 9, "text": " stop the car "},
        ]}
        with patch.object(local, "transcribe_media", return_value=transcript), \
                patch.object(local, "sample_frames", return_value=([np.zeros((4, 4, 3), np.uint8)], [6.0])), \
                patch.object(local, "chat_json", return_value={"ok": True}) as chat:
            local.interval_json("v.mp4", 4, 8, "Describe.", {})
        prompt = chat.call_args.args[0]
        self.assertIn("[2.0]", prompt)
        self.assertIn('{"start": 1, "end": 4, "text": "stop the car"}', prompt)
        self.assertNotIn("hello", prompt)

    def test_generate_metadata_resumes_and_reports_failures(self):
        manifest = {"video": {"path": "v.mp4"}, "chunks": {"medium": [
            {"chunk_id": "m0", "scale": "medium", "start": 0, "end": 30},
            {"chunk_id": "m1", "scale": "medium", "start": 15, "end": 45},
        ]}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "metadata.jsonl"
            output.write_text(json.dumps({"chunk_id": "m0"}) + "\n")
            with patch("video_retrieval.metadata.analyze_video_clip", side_effect=RuntimeError("model down")) as analyze, \
                    patch("time.sleep"):
                failures = generate_metadata(manifest, output_path=output, retry_count=2)
            self.assertEqual(failures, [{"chunk_id": "m1", "error": "model down"}])
            self.assertEqual(analyze.call_count, 2)

            # Retry handlers catch Exception; a cancellation must still stop the loop immediately.
            with patch("video_retrieval.metadata.analyze_video_clip", side_effect=local.Cancelled()) as analyze:
                with self.assertRaises(local.Cancelled):
                    generate_metadata(manifest, output_path=output)
            self.assertEqual(analyze.call_count, 1)

    def test_cancel_event_stops_tracked_loops(self):
        import threading
        event = threading.Event()
        event.set()
        with local.use_models(cancel_event=event), self.assertRaises(local.Cancelled):
            list(local.track([1, 2], "work"))

    def test_context_is_restored_after_failure(self):
        with self.assertRaises(ValueError):
            with local.use_models(local.LocalModels(vision="custom")):
                self.assertEqual(local.model_name("vision"), "custom")
                raise ValueError("test")
        self.assertEqual(local.model_name("vision"), local.LocalModels().vision)

    def test_index_signature_ignores_query_time_models(self):
        self.assertEqual(local.LocalModels(planner="a", verifier="a").index_signature(),
                         local.LocalModels(planner="b", verifier="b").index_signature())
        self.assertNotEqual(local.LocalModels().index_signature(), local.LocalModels(vision="other").index_signature())

    def test_pipeline_scopes_models_and_progress(self):
        resources = RetrievalResources(manifest={}, local_models=local.LocalModels(planner="planner-x"))
        events = []

        def original(**kwargs):
            self.assertEqual(local.model_name("planner"), "planner-x")
            local.stage("Planning query")
            return {"matches": []}

        with patch("video_retrieval.pipeline.retrieve_video", side_effect=original):
            result = VideoRetrievalPipeline(resources).retrieve("test", reporter=events.append)
        self.assertEqual(events, [{"stage": "Planning query"}])
        self.assertEqual(result["model_backend"]["planner"], "planner-x")

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

    def test_local_chat_is_loopback_validated_and_disables_thinking(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        response = Mock(status_code=200)
        response.json.return_value = {"message": {"content": json.dumps({"ok": True})}}
        with patch.object(local, "capabilities", return_value=frozenset({"vision", "thinking"})), \
                patch("requests.post", return_value=response) as post:
            self.assertTrue(local.chat_json("test", schema)["ok"])
            self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/chat")
            self.assertNotIn("headers", post.call_args.kwargs)
            self.assertIs(post.call_args.kwargs["json"]["think"], False)
            response.json.return_value = {"message": {"content": '{"ok": "wrong"}'}}
            with self.assertRaises(RuntimeError):
                local.chat_json("test", schema)
            self.assertEqual(post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
