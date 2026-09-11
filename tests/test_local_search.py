from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

import local_search


class LocalSearchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self):
        path = self.root / 'source.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
        for _ in range(100):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()
        return path

    def test_upload_uses_content_hash_not_filename(self):
        upload = BytesIO(b'video bytes')
        upload.name = '../../outside.mp4'
        with patch.object(local_search, 'ROOT', self.root):
            path = local_search.save_upload(upload)
            self.assertTrue(path.is_relative_to(self.root))
            self.assertEqual(path.name, 'source.mp4')
            self.assertEqual(local_search.save_upload(upload), path)

    def test_index_cache_and_visual_search_without_key(self):
        with patch.object(local_search, 'encode', side_effect=lambda images=None, text=None:
                          np.tile([1., 0.], (len(images) if images else 1, 1))) as encoder:
            data = local_search.process_video(self.video(), interval=3)
            self.assertEqual(len(data['times']), 4)
            calls = encoder.call_count
            local_search.process_video(data['source'], interval=3)
            self.assertEqual(encoder.call_count, calls)
            rows = local_search.search(data, 'dark scene')
            self.assertTrue(rows)
            self.assertTrue(all(0 <= row['start'] < row['end'] <= 10 for row in rows))

    def test_speech_matching_and_silent_video(self):
        data = {'segments': [{'start': 1, 'end': 2, 'text': 'The blue car'}]}
        self.assertEqual(len(local_search.search(data, 'blue', 'Speech')), 1)
        self.assertEqual(local_search.search(data, 'red', 'Speech'), [])
        with patch.object(local_search, 'encode', side_effect=lambda images=None, text=None:
                          np.tile([1., 0.], (len(images), 1))):
            data = local_search.process_video(self.video(), transcribe=True)
        self.assertFalse(data['has_audio'])
        self.assertTrue(data['transcribed'])

    def test_bad_video_and_empty_query(self):
        path = self.root / 'bad.mp4'
        path.write_bytes(b'invalid')
        with self.assertRaises(ValueError):
            local_search.process_video(path)
        with self.assertRaises(ValueError):
            local_search.search({}, ' ')

    def test_local_ui_default_and_results(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.radio[0].set_value('Local uploads (no key)').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, 'Local uploads (no key)')
        app.session_state['local_video'] = {
            'source': str(self.video()), 'duration': 10, 'times': [0, 3, 6, 9],
        }
        app.run()
        with patch.object(local_search, 'search', return_value=[{'start': 1, 'end': 4, 'score': .3}]):
            app.text_input[0].set_value('dark scene')
            next(button for button in app.button if button.label == 'Search locally').click().run()
        self.assertFalse(app.exception)
        self.assertIn('1 ranked moments', [item.value for item in app.subheader])


if __name__ == '__main__':
    unittest.main()
