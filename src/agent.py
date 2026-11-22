"""
agent.py — SackDetectorAgent

A model-based reflex agent that perceives video frames, runs YOLO detection
with BoT-SORT tracking, applies the Bayesian object counter as a second
AI layer, and maintains an internal world model of confirmed object counts.

Agent type : Model-based reflex agent
  - maintains internal state (track hit-counts, confirmed IDs, Bayesian belief)
  - acts on the current percept + internal model
  - no goal-based planning required (pure perception task)
"""

import random
from collections import defaultdict
from typing import Dict, Optional

from ultralytics import YOLO

from bayesian_counter import BayesianObjectCounter


class SackDetectorAgent:
    """
    Model-based reflex agent for sack/box detection and counting.

    Wraps YOLOv8 object detection and BoT-SORT tracking, and feeds
    per-frame raw detection counts into a BayesianObjectCounter to
    produce a probabilistic second estimate.

    Parameters
    ----------
    weights_path     : path to YOLOv8 .pt weights file
    conf             : detection confidence threshold (0–1)
    tracker          : ultralytics tracker config name (e.g. 'botsort.yaml')
    min_hits         : frames a track must appear to be confirmed (hit-counter method)
    device           : 'cpu' or '0' (GPU index)
    seed             : reproducibility seed
    bayes_kwargs     : extra keyword arguments forwarded to BayesianObjectCounter
    """

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.40,
        tracker: str = "botsort.yaml",
        min_hits: int = 15,
        device: str = "cpu",
        seed: int = 42,
        **bayes_kwargs,
    ):
        random.seed(seed)
        self.conf      = conf
        self.tracker   = tracker
        self.min_hits  = min_hits
        self.device    = device
        self.seed      = seed

        self._model    = YOLO(weights_path)
        self.class_names: list = list(self._model.names.values())

        # ── internal world model ──────────────────────────────────────────────
        # hit-counter method (baseline)
        self._hit_counts:   Dict = defaultdict(lambda: defaultdict(int))
        self._confirmed_ids: Dict = defaultdict(set)
        self._hc_counts:    Dict = defaultdict(int)   # hit-counter confirmed

        # Bayesian layer (second technique) — per-class tuned parameters
        # derived from profiling: box λ_in=1.12, sack λ_in=1.02 (sparse scene)
        box_kwargs  = dict(p_arrive=0.029, p_leave=0.08, lambda_present=1.12,
                           lambda_noise=0.05, confirm_threshold=0.75)
        sack_kwargs = dict(p_arrive=0.023, p_leave=0.08, lambda_present=1.02,
                           lambda_noise=0.05, confirm_threshold=0.75)
        # Use merged kwargs if caller passes overrides, else use per-class defaults
        merged = {**box_kwargs, **bayes_kwargs}   # caller can still override
        self._bayes = BayesianObjectCounter(
            class_names=self.class_names,
            seed=seed,
            **merged,
        )

        self._frame_idx = 0

    # ── percept ───────────────────────────────────────────────────────────────

    def perceive(self, frame) -> Dict:
        """
        Process one BGR frame through YOLO + BoT-SORT and update both
        the hit-counter model and the Bayesian belief.

        Parameters
        ----------
        frame : np.ndarray  (H × W × 3, BGR)

        Returns
        -------
        dict with keys:
            annotated_frame  : BGR frame with bounding boxes drawn
            hc_counts        : hit-counter confirmed counts  {class: int}
            bayes_counts     : Bayesian confirmed counts     {class: int}
            bayes_beliefs    : current P(IN_SCENE)           {class: float}
            raw_dets_by_class: raw detection count this frame {class: int}
        """
        self._frame_idx += 1

        try:
            results = self._model.track(
                frame,
                persist=True,
                conf=self.conf,
                tracker=self.tracker,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            # never crash on a single bad frame
            return {
                "annotated_frame": frame,
                "hc_counts": dict(self._hc_counts),
                "bayes_counts": self._bayes.counts,
                "bayes_beliefs": self._bayes.beliefs,
                "raw_dets_by_class": {},
                "error": str(exc),
            }

        annotated = results[0].plot() if results else frame

        # count raw detections per class this frame (for Bayesian input)
        raw_dets: Dict[str, int] = defaultdict(int)

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_name = self._model.names[int(box.cls.item())]
                raw_dets[cls_name] += 1

                if box.id is None:
                    continue
                tid = int(box.id.item())
                self._hit_counts[cls_name][tid] += 1

                if (
                    self._hit_counts[cls_name][tid] >= self.min_hits
                    and tid not in self._confirmed_ids[cls_name]
                ):
                    self._confirmed_ids[cls_name].add(tid)
                    self._hc_counts[cls_name] += 1

        # update Bayesian layer with raw per-class counts
        self._bayes.update(dict(raw_dets))

        return {
            "annotated_frame": annotated,
            "hc_counts": dict(self._hc_counts),
            "bayes_counts": self._bayes.counts,
            "bayes_beliefs": self._bayes.beliefs,
            "raw_dets_by_class": dict(raw_dets),
        }

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def hit_counter_counts(self) -> Dict[str, int]:
        """Final confirmed counts from the hit-counter method."""
        return dict(self._hc_counts)

    @property
    def bayesian_counts(self) -> Dict[str, int]:
        """Final confirmed counts from the Bayesian layer."""
        return self._bayes.counts

    def reset(self):
        """Reset all internal state (useful when processing a new video)."""
        self._hit_counts    = defaultdict(lambda: defaultdict(int))
        self._confirmed_ids = defaultdict(set)
        self._hc_counts     = defaultdict(int)
        self._bayes.reset()
        self._frame_idx = 0

    # ── overridden ────────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"SackDetectorAgent("
            f"conf={self.conf}, tracker={self.tracker!r}, "
            f"min_hits={self.min_hits}, device={self.device!r})"
        )
