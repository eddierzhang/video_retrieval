import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_retrieval import local_backend as local
from video_retrieval.embeddings import embed_text, embed_video
from video_retrieval.metadata import analyze_video_clip
from video_retrieval.transcript import transcribe_audio_chunk
from video_retrieval.verification import call_video_json
from video_retrieval.visual_text import call_images_json
from video_retrieval.retrieval import plan_query
from video_retrieval.pipeline import RetrievalResources, VideoRetrievalPipeline


class LocalArchitectureTest(unittest.TestCase):
    def test_all_model_boundaries_route_locally(self):
        with patch('requests.post', side_effect=AssertionError('Hosted request attempted')), local.use_local():
            with patch.object(local, 'embed_text', return_value=np.ones(512)) as text:
                self.assertEqual(embed_text('test').shape, (512,))
                text.assert_called_once_with('test')
            with patch.object(local, 'embed_video', return_value=np.ones(512)):
                self.assertEqual(embed_video('video.mp4').shape, (512,))
            with patch.object(local, 'video_json', return_value={'ok': True}) as video:
                self.assertEqual(analyze_video_clip('video.mp4'), {'ok': True})
                call_video_json('video.mp4', 'prompt', {}, 'unused')
                self.assertEqual(video.call_count, 2)
            with patch.object(local, 'image_json', return_value={'text': 'ABC'}):
                self.assertEqual(call_images_json([np.zeros((2, 2, 3), dtype=np.uint8)], 'read', {}), {'text': 'ABC'})
            with patch.object(local, 'transcribe', return_value={'words': [], 'segments': []}):
                self.assertEqual(transcribe_audio_chunk('audio.mp3')['words'], [])
            with patch.object(local, 'chat_json', return_value={'weights': {'video': 1}}):
                self.assertEqual(plan_query('person')['weights']['video'], 1)

    def test_context_restored_after_failure(self):
        self.assertFalse(local.active())
        with self.assertRaises(ValueError):
            with local.use_local():
                self.assertTrue(local.active())
                raise ValueError('test')
        self.assertFalse(local.active())

    def test_pipeline_wrapper_preserves_original_function(self):
        resources = RetrievalResources(manifest={}, local_models=local.LocalModels())
        def original(**kwargs):
            self.assertTrue(local.active())
            self.assertIs(kwargs['resources'], resources)
            return {'matches': []}
        with patch('video_retrieval.pipeline.retrieve_video', side_effect=original):
            result = VideoRetrievalPipeline(resources).retrieve('test')
        self.assertIn('model_backend', result)
        self.assertFalse(local.active())

    def test_local_chat_is_loopback_and_schema_validated(self):
        schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}
        response = Mock()
        response.json.return_value = {'message': {'content': json.dumps({'ok': True})}}
        with local.use_local(), patch('requests.post', return_value=response) as post:
            self.assertTrue(local.chat_json('test', schema)['ok'])
            self.assertEqual(post.call_args.args[0], 'http://127.0.0.1:11434/api/chat')
            self.assertNotIn('headers', post.call_args.kwargs)
            response.json.return_value = {'message': {'content': '{"ok": "wrong"}'}}
            with self.assertRaises(Exception):
                local.chat_json('test', schema)


if __name__ == '__main__':
    unittest.main()
