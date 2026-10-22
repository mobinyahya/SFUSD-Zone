"""Choice models used by iterative zoning strategies."""

from __future__ import annotations

from choice.mnl import MNLChoiceModel, build_mnl_choice_model
from choice.objective import ChoiceCut, ChoiceEvaluation, ChoiceModel
from optimization.problem import ZoneProblem

__all__ = [
    "ChoiceModel",
    "MNLChoiceModel",
    "PRICE_SOURCES",
    "PricedAccessChoiceModel",
    "build_mnl_choice_model",
    "build_priced_access_choice_model",
]


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
        self.welfare_bound = None if welfare_bound is None else float(welfare_bound)
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
