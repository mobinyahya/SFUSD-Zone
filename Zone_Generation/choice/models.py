"""Choice models used by iterative zoning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from Zone_Generation.choice.mnl import MNLZoningUtility
from Zone_Generation.choice.objective import ChoiceCut, ChoiceEvaluation
from Zone_Generation.optimization.problem import ZoneProblem


class ChoiceModel(ABC):
    @abstractmethod
    def evaluate_with_cuts(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> ChoiceEvaluation:
        """Real utility and linearization cuts at ``assignment``."""

    def evaluate(self, problem: ZoneProblem, assignment: dict[int, int]) -> float:
        return self.preassignment_utility(problem, assignment)

    def preassignment_utility(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> float:
        return self.evaluate_with_cuts(problem, assignment).utility

    def utility_bounds(self, problem: ZoneProblem) -> tuple[float, float]:
        return (-1_000_000_000.0, 1_000_000_000.0)


class DistanceChoiceModel(ChoiceModel):
    """Student-weighted negative distance to the assigned centroid."""

    def preassignment_utility(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> float:
        return sum(
            self._zone_utility(problem, node, assignment[node])
            for node in problem.nodes
        )

    def evaluate_with_cuts(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> ChoiceEvaluation:
        utility = self.preassignment_utility(problem, assignment)
        cuts: list[ChoiceCut] = []
        for node in problem.nodes:
            for zone in problem.candidate_zones(node):
                cuts.append(
                    ChoiceCut(
                        node=node,
                        zone=zone,
                        constant=self._zone_utility(problem, node, zone),
                    )
                )
        return ChoiceEvaluation(utility=utility, cuts=tuple(cuts))

    def utility_bounds(self, problem: ZoneProblem) -> tuple[float, float]:
        values = [
            self._zone_utility(problem, node, zone)
            for node in problem.nodes
            for zone in problem.candidate_zones(node)
        ]
        if not values:
            return (-1.0, 1.0)
        return (min(values) - 1.0, max(values) + 1.0)

    @staticmethod
    def _zone_utility(problem: ZoneProblem, node: int, zone: int) -> float:
        centroid = problem.centroids[zone]
        return -problem.students(node) * problem.distance(centroid, node)


class MNLChoiceModel(ChoiceModel):
    """Thin strategy-facing wrapper around shared MNL zoning utility logic."""

    def __init__(
        self,
        method: str = "logsum",
        area_column: str | None = None,
        lower_bound: float = -1_000_000_000.0,
        upper_bound: float = 1_000_000_000.0,
        empty_utility: float = -1e10,
    ):
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.evaluator = MNLZoningUtility(
            method=method,
            area_column=area_column,
            empty_utility=empty_utility,
        )

    def evaluate_with_cuts(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> ChoiceEvaluation:
        return self.evaluator.evaluate_with_cuts(problem, assignment)

    def preassignment_utility(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> float:
        return self.evaluator.preassignment_utility(problem, assignment)

    def utility_bounds(self, problem: ZoneProblem) -> tuple[float, float]:
        return (self.lower_bound, self.upper_bound)


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


def get_configured_choice_model(options: Mapping[str, Any]) -> ChoiceModel:
    """Build the choice model described by optimization config options."""

    name = str(options.get("choice_model", "distance"))
    if name == "mnl":
        return get_choice_model(
            name,
            method=str(options.get("choice_model_method", "logsum")),
        )
    return get_choice_model(name)
