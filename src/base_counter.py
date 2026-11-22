"""
base_counter.py — BaseCounter

Abstract base class defining the interface every object counter must implement.
BayesianObjectCounter inherits from this class, satisfying the Labs 01-02
requirement for inheritance and at least one overridden method.
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseCounter(ABC):
    """
    Abstract base class for object counters.

    Any counter that tracks distinct objects across video frames must
    implement update() and expose counts and beliefs as properties.

    Encapsulates the contract between the agent and any counting strategy,
    allowing the SackDetectorAgent to swap counting methods without changing
    its own code (open/closed principle).
    """

    @abstractmethod
    def update(self, detections_per_class: Dict[str, int]) -> None:
        """
        Process one frame's raw detection counts and update internal state.

        Parameters
        ----------
        detections_per_class : dict
            Maps class name to number of raw detections in this frame.
            Pass 0 (or omit the key) for classes with no detections.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state. Called between videos."""

    @property
    @abstractmethod
    def counts(self) -> Dict[str, int]:
        """Return confirmed object counts per class."""

    def total(self) -> int:
        """Return sum of confirmed counts across all classes."""
        return sum(self.counts.values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(counts={self.counts})"
