"""
bayesian_counter.py — BayesianObjectCounter

Additional AI layer on top of the raw YOLO + BoT-SORT track counts.

Motivation
----------
BoT-SORT confirms a track after `min_hits` consecutive appearances, but it
cannot distinguish between:
  (a) a new object genuinely entering the scene, and
  (b) a lost-and-re-found track for an object already counted.

This module models each class independently as a two-state Hidden Markov
Model:

  Hidden state  Z_t  ∈  {IN_SCENE, NOT_IN_SCENE}
  Observation   O_t  =  number of raw (unconfirmed) detections in frame t

Transition model  P(Z_t | Z_{t-1})
  P(IN  | IN)  = 1 − p_leave   (object stays in scene)
  P(IN  | OUT) = p_arrive       (new object enters)

Emission model  P(O_t | Z_t)
  Modelled as a Poisson distribution:
    Z=IN_SCENE   → Poisson(lambda=expected_dets_when_present)
    Z=NOT_IN_SCENE → Poisson(lambda=noise_rate)

The filter runs forward in time (Bayesian filtering / forward algorithm).
When the posterior P(IN_SCENE) crosses `confirm_threshold` for the first
time, the object is confirmed and the belief is reset so the same object
is not double-counted.

This gives a probabilistic count that is compared against the
hit-counting baseline in the experiments.
"""

import math
import random
from typing import Dict, List

from base_counter import BaseCounter


class BayesianObjectCounter(BaseCounter):
    """
    Per-class Bayesian object counter using a two-state HMM belief update.

    Parameters
    ----------
    class_names          : list of class name strings (e.g. ['box', 'sack'])
    p_arrive             : prior probability an object enters any given frame
    p_leave              : probability an in-scene object leaves in next frame
    lambda_present       : expected raw detections per frame when object is present
    lambda_noise         : expected spurious detections per frame (background)
    confirm_threshold    : posterior P(IN_SCENE) needed to confirm a new count
    seed                 : for reproducibility
    """

    def __init__(
        self,
        class_names: List[str],
        p_arrive: float = 0.02,
        p_leave: float  = 0.05,
        lambda_present: float = 2.5,
        lambda_noise: float   = 0.3,
        confirm_threshold: float = 0.80,
        seed: int = 42,
    ):
        random.seed(seed)
        self.class_names       = class_names
        self.p_arrive          = p_arrive
        self.p_leave           = p_leave
        self.lambda_present    = lambda_present
        self.lambda_noise      = lambda_noise
        self.confirm_threshold = confirm_threshold

        # belief[cls] = P(IN_SCENE) for the "next unconfirmed object"
        self._belief: Dict[str, float] = {c: p_arrive for c in class_names}
        # confirmed count per class
        self._counts: Dict[str, int]   = {c: 0 for c in class_names}
        # hysteresis state: True when currently tracking a confirmed object
        self._in_object: Dict[str, bool] = {c: False for c in class_names}

    # ── Poisson PMF (no scipy dependency) ────────────────────────────────────

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        """P(X=k) for X ~ Poisson(lam). Returns a small floor to avoid log(0)."""
        if lam <= 0:
            return 1.0 if k == 0 else 1e-12
        try:
            log_p = k * math.log(lam) - lam - math.lgamma(k + 1)
            return max(math.exp(log_p), 1e-12)
        except (ValueError, OverflowError):
            return 1e-12

    # ── belief update ─────────────────────────────────────────────────────────

    def _predict_step(self, prior: float) -> float:
        """Chapman–Kolmogorov prediction: propagate belief one step forward."""
        return prior * (1 - self.p_leave) + (1 - prior) * self.p_arrive

    def _update_step(self, predicted: float, observation: int) -> float:
        """Bayesian update given observation count."""
        p_obs_in  = self._poisson_pmf(observation, self.lambda_present)
        p_obs_out = self._poisson_pmf(observation, self.lambda_noise)
        numerator = p_obs_in * predicted
        denominator = numerator + p_obs_out * (1 - predicted)
        if denominator < 1e-15:
            return predicted
        return numerator / denominator

    # ── public interface ──────────────────────────────────────────────────────

    def update(self, detections_per_class: Dict[str, int]):
        """
        Feed one frame's raw detection counts into the filter.

        Uses confirm/exit hysteresis to avoid re-counting the same object:
          WAITING  state: belief rises above confirm_threshold → count += 1,
                          transition to IN_SCENE state.
          IN_SCENE state: belief drops below exit_threshold (0.25) →
                          transition back to WAITING, ready for next object.

        Parameters
        ----------
        detections_per_class : {class_name: n_detections_this_frame}
            Pass 0 for classes with no detections this frame.
        """
        exit_threshold = 0.25   # belief must drop this low before next confirmation

        for cls in self.class_names:
            obs = detections_per_class.get(cls, 0)

            predicted = self._predict_step(self._belief[cls])
            posterior = self._update_step(predicted, obs)
            self._belief[cls] = posterior

            if not self._in_object[cls] and posterior >= self.confirm_threshold:
                self._counts[cls]  += 1
                self._in_object[cls] = True          # now tracking this object
            elif self._in_object[cls] and posterior < exit_threshold:
                self._in_object[cls] = False         # object left; wait for next

    @property
    def counts(self) -> Dict[str, int]:
        """Current confirmed counts per class."""
        return dict(self._counts)

    @property
    def beliefs(self) -> Dict[str, float]:
        """Current P(IN_SCENE) belief per class."""
        return dict(self._belief)

    def reset(self):
        """Reset beliefs and counts (useful between videos)."""
        self._belief    = {c: self.p_arrive for c in self.class_names}
        self._counts    = {c: 0             for c in self.class_names}
        self._in_object = {c: False         for c in self.class_names}

    # ── overridden ────────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"BayesianObjectCounter("
            f"classes={self.class_names}, "
            f"p_arrive={self.p_arrive}, p_leave={self.p_leave}, "
            f"threshold={self.confirm_threshold})"
        )
