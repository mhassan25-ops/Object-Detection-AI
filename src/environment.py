"""
environment.py — DetectionEnvironment

PEAS description
----------------
Performance measure : counting accuracy (|predicted − ground_truth|), noise
                      rejection rate, processing throughput (fps).
Environment         : warehouse / loading-dock video stream (pre-recorded or live).
Actuators           : none — the agent is purely perceptual; it writes counts
                      to a shared status dict and pushes annotated frames.
Sensors             : camera frames delivered as BGR numpy arrays via OpenCV.

Environment classification
--------------------------
- Partially observable  (objects may be occluded)
- Stochastic            (lighting changes, motion blur, variable occlusions)
- Sequential            (counts accumulate over time; order matters)
- Dynamic               (scene changes between frames)
- Continuous            (pixel-valued sensor readings)
- Single-agent          (one detector per stream)
"""

import random
from pathlib import Path
import cv2


class DetectionEnvironment:
    """
    Wraps a video source and delivers frames one at a time to an agent.

    Attributes
    ----------
    source      : path to a video file (str or Path) or camera index (int)
    seed        : integer seed for reproducibility of any random operations
    frame_count : total frames in the source (0 if unknown)
    fps         : nominal frame-rate of the source
    width, height : frame dimensions

    Usage
    -----
    env = DetectionEnvironment("video.mp4", seed=42)
    for frame, idx, done in env:
        agent.perceive(frame)
    env.release()
    """

    def __init__(self, source, seed: int = 42):
        """
        Parameters
        ----------
        source : str | int
            Path to a video file or webcam index.
        seed   : int
            Seed applied to Python's random module for full reproducibility.
        """
        random.seed(seed)
        self.source = source
        self.seed   = seed

        self._cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not self._cap.isOpened():
            raise OSError(f"DetectionEnvironment: cannot open source '{source}'")

        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps         = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width       = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_idx  = 0

    # ── iterator interface ────────────────────────────────────────────────────

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns
        -------
        (frame, frame_index, done)
            frame       : BGR numpy array
            frame_index : 1-based frame number
            done        : True on the last frame
        """
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise StopIteration
        self._frame_idx += 1
        done = (self._frame_idx >= self.frame_count)
        return frame, self._frame_idx, done

    def release(self):
        """Release the underlying VideoCapture."""
        if self._cap.isOpened():
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    # ── overridden ────────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"DetectionEnvironment(source={self.source!r}, "
            f"frames={self.frame_count}, fps={self.fps:.1f}, "
            f"size={self.width}x{self.height})"
        )
