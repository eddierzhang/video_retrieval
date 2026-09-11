import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest
from ui_resources import ROOT, read_manifest


class SearchUITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.video = folder / 'video.mp4'
        self.video.write_bytes(b'test video')
        self.manifest = folder / 'manifest.json'
        self.manifest.write_text(json.dumps({
            'video': {'path': str(self.video), 'filename': 'video.mp4', 'duration': 60},
            'chunks': {},
        }))
        self.pipeline = Mock()
        self.pipeline.retrieve.return_value = {
            'query': 'person', 'matches': [{'start': 3, 'end': 8, 'confidence': .9,
                                          'description': 'A person appears.'}],
        }
        for patcher in (
            patch('ui_resources.discover_manifests', return_value=[self.manifest]),
            patch('ui_resources.load_pipeline', return_value=self.pipeline),
            patch('shutil.which', return_value='ffmpeg'),
            patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(str(ROOT / 'app.py')).run()
        self.app.radio[0].set_value('OpenRouter indexes').run()

    def search(self, query='person'):
        next(x for x in self.app.text_input if x.label == 'What are you looking for?').set_value(query)
        next(x for x in self.app.button if x.label == 'Search video').click().run()
        self.assertFalse(self.app.exception)

    def test_search_and_rerun_preserve_results(self):
        self.search()
        self.assertIn('1 matching moments', [x.value for x in self.app.subheader])
        self.app.run()
        self.pipeline.retrieve.assert_called_once()
        self.assertIn('1 matching moments', [x.value for x in self.app.subheader])

    def test_empty_query_and_missing_key(self):
        self.search(' ')
        self.pipeline.retrieve.assert_not_called()
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            self.search()
        self.assertTrue(self.app.error)
        self.pipeline.retrieve.assert_not_called()

    def test_failed_search_clears_old_results_and_redacts_key(self):
        self.search()
        self.pipeline.retrieve.side_effect = RuntimeError('request test-key failed')
        self.search('new query')
        self.assertNotIn('1 matching moments', [x.value for x in self.app.subheader])
        self.assertNotIn('test-key', self.app.error[0].value)

    def test_video_override_does_not_rewrite_manifest(self):
        result = read_manifest(self.manifest, str(self.video.parent / 'moved.mp4'))
        self.assertTrue(result['video']['path'].endswith('moved.mp4'))
        self.assertEqual(read_manifest(self.manifest)['video']['path'], str(self.video))


if __name__ == '__main__':
    unittest.main()
