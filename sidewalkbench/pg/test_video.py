"""Native OpenCV MP4 recording and useful errors for unavailable encoders."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from sidewalkbench.pg.utils import vis


class VideoRecordingTests(unittest.TestCase):
    def test_records_readable_mp4_and_finalizes_each_episode(self):
        state = vis.VisState()
        frame = np.full((240, 320, 3), [220, 30, 10], dtype=np.uint8)
        with TemporaryDirectory(prefix="video output ") as directory:
            for episode in range(2):
                state.pos_history = []
                for i in range(3):
                    state.pos_history.append(i)
                    vis.save_video({"demo": frame}, state, directory)
                state.episode_outcome = (episode == 0, "test failure")
                vis.finish_episode({"demo": frame}, state)
            self.assertEqual(state.video_reset_idx, 2)
            self.assertIsNone(state.video_writer)
            paths = sorted(Path(directory).glob("*.mp4"))
            self.assertEqual(len(paths), 2)
            for path in paths:
                capture = cv2.VideoCapture(str(path))
                try:
                    self.assertTrue(capture.isOpened())
                    self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 23)
                    self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 320)
                    self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 240)
                    ok, bgr = capture.read()
                    self.assertTrue(ok)
                    np.testing.assert_allclose(bgr[20, 20].astype(float), frame[20, 20, ::-1], atol=8)
                finally:
                    capture.release()

    def test_missing_encoder_reports_error_instead_of_silent_failure(self):
        unavailable = Mock()
        unavailable.isOpened.return_value = False
        state = vis.VisState()
        state.pos_history.append(0)
        frame = np.zeros((240, 320, 3), np.uint8)
        with patch.object(vis.cv2, "VideoWriter", return_value=unavailable):
            with self.assertRaisesRegex(RuntimeError, "uv sync --reinstall-package opencv-python"):
                vis.save_video({"demo": frame}, state, "unused")
        self.assertIsNone(state.video_writer)
        unavailable.release.assert_called_once()
        unavailable.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
