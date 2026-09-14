"""Choice models used by iterative zoning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from choice.mnl import MNLZoningUtility
from choice.objective import ChoiceCut, ChoiceEvaluation
from loaders import DataScenario
from optimization.problem import ZoneProblem


class ChoiceModel(ABC):
    # A zoning-independent upper bound on attainable welfare, when the model
    # happens to compute one, plus whatever provenance the strategy should
    # report with it. Most models compute neither; declaring them here keeps
    # the strategies free of attribute probing.
    welfare_bound: float | None = None
    bound_metadata: dict = {}

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
        citywide_schools: Iterable[object] = (),
    ):
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.evaluator = MNLZoningUtility(
            data,
            method=method,
            area_column=area_column,
            empty_utility=empty_utility,
            citywide_schools=citywide_schools,
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
    """Build the sole supported zoning choice model.

    Resolves ``include_citywide_choice_opt`` from the scenario rather than
    taking it as an argument, so every caller -- the strategy, the metrics, the
    offline replays -- gets the same answer for the same scenario.
    """

    citywide: tuple[object, ...] = ()
    if data.filter("optimization", "include_citywide_choice_opt", False):
        # Imported here: the top-level choice package must not depend on
        # optimization.data, which imports choice in turn.
        from optimization.data.loaders import citywide_school_ids

        citywide = tuple(citywide_school_ids(data))
    return MNLChoiceModel(data=data, method=method, citywide_schools=citywide)


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
        welfare_bound: float | None = None,
        bound_metadata: dict | None = None,
    ):
        from choice.priced_access import PricedAccessUtility

        self.evaluator = PricedAccessUtility(
            market, problem, prices, cut_levels=cut_levels
        )
        # A zoning-independent upper bound on attainable stable-matching
        # welfare, when the price source produced one. It certifies nothing
        # about the search, but it is valid simultaneously with the bound
        # Theorem 2 reports, so the smaller of the two is the tighter
        # certificate. ``None`` when the prices came from a source that solves
        # no zone-aware relaxation.
        self.welfare_bound = (
            None if welfare_bound is None else float(welfare_bound)
        )
        self.bound_metadata = dict(bound_metadata or {})

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


# Where the congestion prices come from. "transport" takes the capacity duals
# of the zone-blind transportation LP. The two "zoned_transport_*" sources take
# them from the zone-restricted relaxation of
# :mod:`optimization.zoned_transport` instead, so a seat is priced by its
# marginal value *after* confinement rather than before it; that solve also
# returns a welfare bound, which the strategy reports alongside its own.
PRICE_SOURCES = (
    "transport",
    "none",
    "zoned_transport_neighbors",
    "zoned_transport_flow",
)


def build_priced_access_choice_model(
    problem: ZoneProblem,
    optimization_config,
    *,
    cut_levels: int = 3,
    price_source: str = "transport",
    price_scale: float = 1.0,
    prices: dict[str, float] | None = None,
    workers: int = 1,
    centroid_neighbor_radius: int = 0,
) -> PricedAccessChoiceModel:
    """Build the priced-access model, pricing capacity if no prices are given.

    ``price_scale`` multiplies the prices before use. Any non-negative price
    vector keeps the bound valid, so the scale is a free tuning knob: 0 charges
    nothing for congestion and reduces the surrogate to best-school-in-zone,
    while scales above 1 over-charge it, pushing the master to spread demand.

    ``workers`` and ``centroid_neighbor_radius`` only reach the zone-aware
    price sources, which solve a cached LP. The worker count defaults to one,
    because concurrent LP measured slower on that model, and is excluded from
    the cache key.
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
    welfare_bound: float | None = None
    bound_metadata: dict = {}
    if prices is None:
        if price_source == "none":
            prices = {}
        elif price_source == "transport":
            prices, _ = transport_prices(market.programs, market.students)
        else:
            from optimization.welfare_bounds import ZONED_WELFARE_BOUNDS
            from optimization.zoned_transport import zoned_transport_bound

            bound = zoned_transport_bound(
                market.programs,
                market.students,
                problem,
                contiguity_model=ZONED_WELFARE_BOUNDS[price_source],
                workers=workers,
                centroid_neighbor_radius=centroid_neighbor_radius,
            )
            # Non-negative by LP duality on a "<=" row of a maximization, so
            # Proposition 5 applies to them unchanged.
            prices = dict(bound.prices)
            welfare_bound = bound.objective
            bound_metadata = dict(bound.metadata)
    if price_scale != 1.0:
        prices = {key: value * price_scale for key, value in prices.items()}
    # Scaling the prices changes the surrogate but not the relaxation that
    # produced ``welfare_bound``, which is valid for every zoning regardless.
    return PricedAccessChoiceModel(
        market,
        problem,
        prices,
        cut_levels=cut_levels,
        welfare_bound=welfare_bound,
        bound_metadata=bound_metadata,
    )
