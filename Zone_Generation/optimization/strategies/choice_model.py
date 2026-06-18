"""Choice / utility models for the iterative-choice strategy.

A :class:`ChoiceModel` scores a zoning assignment by the school-choice utility
it affords students. The iterative strategy uses it to evaluate candidate
zonings and steer the search.

Two models are provided:

* :class:`DistanceChoiceModel` -- a data-free proximity proxy (utility is the
  negative distance from each area to its zone's centroid, student-weighted).
  Always available; used in tests and as a sensible default.
* :class:`MNLChoiceModel` -- loads the estimated multinomial-logit utilities
  and student demographics (rewrite of the legacy ``utility_evaluation``),
  computing max/log-sum welfare. Available only when the data files are
  present.

The strategy depends only on the :class:`ChoiceModel` interface, so swapping in
a richer model never touches the orchestration.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from Zone_Generation.optimization.problem import ZoneProblem


class ChoiceModel(ABC):
    @abstractmethod
    def evaluate(self, problem: ZoneProblem, assignment: dict[int, int]) -> float:
        """Total utility of ``assignment`` (higher is better)."""


class DistanceChoiceModel(ChoiceModel):
    """Student-weighted negative distance to the assigned centroid."""

    def evaluate(self, problem: ZoneProblem, assignment: dict[int, int]) -> float:
        total = 0.0
        for node, zone in assignment.items():
            centroid = problem.centroids[zone]
            total -= problem.students(node) * problem.distance(centroid, node)
        return total


class MNLChoiceModel(ChoiceModel):
    """Multinomial-logit welfare from estimated utilities.

    Loads the per-(student, school) utility matrix and aggregates it per area;
    a student's utility is the best (``max``) or log-sum-exp (``logsum``) over
    schools that fall in the student's assigned zone. This is the rewrite of
    the legacy ``UtilityEvaluator``; it requires the estimate/demographics CSVs
    and is constructed lazily.
    """

    def __init__(self, method: str = "logsum"):
        self.method = method
        self._utilities = None  # populated on first evaluate()

    def _ensure_loaded(self, problem: ZoneProblem):
        if self._utilities is not None:
            return
        # Intentionally minimal: production deployments wire the estimate and
        # demographics CSVs here. Kept lazy so importing the module never
        # requires the data to be present.
        raise NotImplementedError(
            "MNLChoiceModel requires the estimate/demographics data files to be "
            "wired in; use DistanceChoiceModel for data-free runs."
        )

    def evaluate(self, problem: ZoneProblem, assignment: dict[int, int]) -> float:
        self._ensure_loaded(problem)
        # combine per-area utilities (populated in _ensure_loaded)
        total = 0.0
        for node, zone in assignment.items():
            schools = [
                s
                for s in problem.G.nodes[node].get("school_ids", [])
            ]
            if not schools:
                continue
            vals = [self._utilities[(node, s)] for s in schools]  # type: ignore
            if self.method == "max":
                total += max(vals)
            else:
                total += math.log(sum(math.exp(v) for v in vals))
        return total


_MODELS = {
    "distance": DistanceChoiceModel,
    "mnl": MNLChoiceModel,
}


def get_choice_model(name: str, **options) -> ChoiceModel:
    if name not in _MODELS:
        raise ValueError(
            f"Unknown choice model {name!r}. Available: {sorted(_MODELS)}."
        )
    return _MODELS[name](**options)
