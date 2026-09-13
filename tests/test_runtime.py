"""The local model boundary: embeddings, indexes, model names, progress and cancellation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_retrieval import local_backend as local
from video_retrieval.embeddings import embed_scale, embed_text
from video_retrieval.metadata import generate_metadata
from video_retrieval.pipeline import RetrievalResources, VideoRetrievalPipeline
from video_retrieval.retrieval import plan_query
from video_retrieval.verification import call_video_json
from video_retrieval.video import generate_windows
from video_retrieval.visual_text import call_images_json


class ModelRuntimeTest(unittest.TestCase):
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

    def test_verification_windows_stay_short_enough_to_see_detail(self):
        import inspect

        from video_retrieval.verification import _video_windows, verify_candidate

        window = inspect.signature(verify_candidate).parameters["verifier_window_seconds"].default
        self.assertLessEqual(window, 30)
        # A 60s candidate is split, so each call spends its frame budget on less time.
        self.assertGreaterEqual(len(_video_windows(0, 60, window=window, overlap=5)), 3)


if __name__ == "__main__":
    unittest.main()
