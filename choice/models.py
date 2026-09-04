"""Choice models used by iterative zoning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from choice.mnl import MNLZoningUtility
from choice.objective import ChoiceCut, ChoiceEvaluation
from loaders import DataScenario
from optimization.problem import ZoneProblem


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

    def choice_utility_hint_cuts(self, problem: ZoneProblem) -> tuple[ChoiceCut, ...]:
        return ()


class MNLChoiceModel(ChoiceModel):
    """Thin strategy-facing wrapper around shared MNL zoning utility logic."""

    def __init__(
        self,
        data: DataScenario,
        method: str = "logsum",
        area_column: str | None = None,
        lower_bound: float = -1_000_000_000.0,
        upper_bound: float = 1_000_000_000.0,
        empty_utility: float = -1e10,
    ):
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.evaluator = MNLZoningUtility(
            data,
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

    def choice_utility_hint_cuts(self, problem: ZoneProblem) -> tuple[ChoiceCut, ...]:
        return self.evaluator.choice_utility_hint_cuts(problem)


def build_mnl_choice_model(
    data: DataScenario,
    *,
    method: str = "logsum",
) -> MNLChoiceModel:
    """Build the sole supported zoning choice model."""

    return MNLChoiceModel(data=data, method=method)
