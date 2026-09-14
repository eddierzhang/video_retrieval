"""Detect search: shots, tracking, crop and clip scoring, feedback. Models are never loaded."""
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
