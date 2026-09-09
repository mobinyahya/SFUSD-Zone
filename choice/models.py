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


class PricedAccessChoiceModel(ChoiceModel):
    """Strategy-facing wrapper around the congestion-priced access surrogate.

    Unlike :class:`MNLChoiceModel` this needs the zoning problem up front: the
    market, the capacity prices and the per-student option lists are all derived
    from it, and none of them depend on the zoning.
    """

    def __init__(
        self,
        market,
        problem: ZoneProblem,
        prices: dict[str, float],
        *,
        cut_levels: int = 3,
    ):
        from choice.priced_access import PricedAccessUtility

        self.evaluator = PricedAccessUtility(
            market, problem, prices, cut_levels=cut_levels
        )

    @property
    def price_constant(self) -> float:
        """Capacity term of the bound; a constant shift, not part of the argmax."""
        return self.evaluator.price_constant

    def evaluate_with_cuts(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> ChoiceEvaluation:
        return self.evaluator.evaluate_with_cuts(problem, assignment)

    def preassignment_utility(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> float:
        return self.evaluator.evaluate(problem, assignment)

    def utility_bounds(self, problem: ZoneProblem) -> tuple[float, float]:
        return self.evaluator.node_utility_bounds(problem)

    def choice_utility_hint_cuts(self, problem: ZoneProblem) -> tuple[ChoiceCut, ...]:
        return self.evaluator.initial_cuts(problem)


PRICE_SOURCES = ("transport", "none")


def build_priced_access_choice_model(
    problem: ZoneProblem,
    optimization_config,
    *,
    cut_levels: int = 3,
    price_source: str = "transport",
    price_scale: float = 1.0,
    prices: dict[str, float] | None = None,
) -> PricedAccessChoiceModel:
    """Build the priced-access model, pricing capacity if no prices are given.

    ``price_scale`` multiplies the prices before use. Any non-negative price
    vector keeps the bound valid, so the scale is a free tuning knob: 0 charges
    nothing for congestion and reduces the surrogate to best-school-in-zone,
    while scales above 1 over-charge it, pushing the master to spread demand.
    """
    from choice.priced_access import transport_prices
    from optimization.data.saa import build_saa_market

    if price_source not in PRICE_SOURCES:
        raise ValueError(
            f"Unknown priced-access price source {price_source!r}; "
            f"expected one of {PRICE_SOURCES}."
        )
    if price_scale < 0.0:
        raise ValueError("priced_access_price_scale must be non-negative.")

    market = build_saa_market(problem, optimization_config)
    if prices is None:
        if price_source == "none":
            prices = {}
        else:
            prices, _ = transport_prices(market.programs, market.students)
    if price_scale != 1.0:
        prices = {key: value * price_scale for key, value in prices.items()}
    return PricedAccessChoiceModel(
        market, problem, prices, cut_levels=cut_levels
    )
