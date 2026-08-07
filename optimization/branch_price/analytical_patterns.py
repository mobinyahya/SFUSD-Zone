"""Floating complete-zone patterns valued by Shi's expected-MNL program."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import TypeAlias

from optimization.analytical_bounds import ShiMechanismResult, solve_shi_menu_bound
from optimization.analytical_welfare_oracle import validate_analytical_welfare_market
from optimization.branch_price.patterns import ZonePatternValidator, zone_perimeter
from optimization.problem import (
    AnalyticalWelfareMarket,
    AnalyticalWelfareSegment,
    ZoneProblem,
)

AnalyticalPatternKey: TypeAlias = tuple[int, frozenset[int]]


@dataclass(frozen=True, slots=True)
class AnalyticalZonePattern:
    """One complete labeled zone with a separately closed Shi coefficient."""

    label: int
    nodes: frozenset[int]
    shi_welfare: float = field(compare=False)
    perimeter: int = field(compare=False)
    valuation_status: str = field(compare=False)
    school_ids: tuple[int, ...] = field(default=(), compare=False)
    segment_ids: tuple[int, ...] = field(default=(), compare=False)
    mechanism: ShiMechanismResult | None = field(
        default=None, compare=False, repr=False
    )
    valuation_seconds: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.label, bool) or not isinstance(self.label, Integral):
            raise TypeError("Analytical pattern label must be an integer.")
        nodes = frozenset(_integer("node", node) for node in self.nodes)
        if int(self.label) < 0:
            raise ValueError("Analytical pattern label must be nonnegative.")
        if not nodes:
            raise ValueError("An analytical zone pattern must contain nodes.")
        if not math.isfinite(float(self.shi_welfare)) or self.shi_welfare < 0:
            raise ValueError("shi_welfare must be finite and nonnegative.")
        if isinstance(self.perimeter, bool) or not isinstance(self.perimeter, Integral):
            raise TypeError("Analytical pattern perimeter must be an integer.")
        if self.perimeter < 0:
            raise ValueError("Analytical pattern perimeter must be nonnegative.")
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "shi_welfare", float(self.shi_welfare))
        object.__setattr__(self, "perimeter", int(self.perimeter))
        object.__setattr__(self, "school_ids", tuple(sorted(map(int, self.school_ids))))
        object.__setattr__(
            self, "segment_ids", tuple(sorted(map(int, self.segment_ids)))
        )

    @property
    def key(self) -> AnalyticalPatternKey:
        return self.label, self.nodes


class AnalyticalPatternValuator:
    """Derive and cache structurally validated local Shi programs."""

    def __init__(
        self,
        problem: ZoneProblem,
        market: AnalyticalWelfareMarket | None = None,
        *,
        centroid_neighbor_radius: int = 0,
        menu_tolerance: float = 1e-9,
        optimality_tolerance: float = 1e-6,
        max_menu_rounds: int = 1000,
        require_closed: bool = True,
    ) -> None:
        self.problem = problem
        self.market = market or problem.analytical_welfare_market
        if self.market is None:
            raise ValueError("Analytical pattern valuation requires a Shi market.")
        validate_zoned_shi_market(problem, self.market)
        self.validator = ZonePatternValidator(
            problem,
            centroid_neighbor_radius=centroid_neighbor_radius,
        )
        if not math.isfinite(menu_tolerance) or menu_tolerance <= 0:
            raise ValueError("menu_tolerance must be positive and finite.")
        if isinstance(max_menu_rounds, bool) or max_menu_rounds <= 0:
            raise ValueError("max_menu_rounds must be positive.")
        self.menu_tolerance = float(menu_tolerance)
        if not math.isfinite(optimality_tolerance) or optimality_tolerance <= 0:
            raise ValueError("optimality_tolerance must be positive and finite.")
        self.optimality_tolerance = float(optimality_tolerance)
        self.max_menu_rounds = int(max_menu_rounds)
        self.require_closed = bool(require_closed)
        self.market_fingerprint = analytical_market_fingerprint(self.market)
        self._cache: dict[tuple[str, int, frozenset[int]], AnalyticalZonePattern] = {}

    def value(
        self,
        label: int,
        nodes: frozenset[int] | set[int],
        *,
        deadline: float | None = None,
        force_revalue: bool = False,
    ) -> AnalyticalZonePattern:
        full_nodes = frozenset(nodes)
        perimeter = zone_perimeter(self.problem.G, full_nodes)
        self.validator.validate_membership(
            label=label,
            nodes=full_nodes,
            perimeter=perimeter,
        )
        cache_key = (self.market_fingerprint, int(label), full_nodes)
        cached = self._cache.get(cache_key)
        if cached is not None and not force_revalue:
            return cached

        schools = tuple(
            sorted(
                school
                for school, node in self.market.school_nodes.items()
                if node in full_nodes
            )
        )
        school_set = set(schools)
        local_segments = tuple(
            _restrict_segment(segment, school_set)
            for segment in self.market.segments
            if segment.node in full_nodes
        )
        total_mass = sum(segment.mass for segment in local_segments)
        effective_menu_tolerance = min(
            self.menu_tolerance,
            self.optimality_tolerance / max(1.0, total_mass),
        )
        mechanism = solve_shi_menu_bound(
            local_segments,
            {school: self.market.school_capacities[school] for school in schools},
            beta=self.market.beta,
            tolerance=effective_menu_tolerance,
            max_rounds=self.max_menu_rounds,
            deadline=deadline,
        )
        valuation_closed = (
            mechanism.closed
            and mechanism.max_primal_capacity_violation <= self.optimality_tolerance
            and mechanism.max_primal_probability_residual <= self.optimality_tolerance
            and mechanism.primal_dual_gap <= self.optimality_tolerance
        )
        if self.require_closed and not valuation_closed:
            raise RuntimeError(
                f"Shi valuation did not close for pattern {(label, full_nodes)}: "
                f"{mechanism.status}, residual={mechanism.max_pricing_violation}, "
                f"gap={mechanism.primal_dual_gap}."
            )
        pattern = AnalyticalZonePattern(
            label=int(label),
            nodes=full_nodes,
            shi_welfare=mechanism.primal_objective,
            perimeter=perimeter,
            valuation_status=(
                mechanism.status if valuation_closed else "NUMERICAL_NONCLOSURE"
            ),
            school_ids=schools,
            segment_ids=tuple(segment.segment_id for segment in local_segments),
            mechanism=mechanism,
            valuation_seconds=mechanism.timing_seconds,
        )
        self._cache[cache_key] = pattern
        return pattern

    def validate_pattern(self, pattern: AnalyticalZonePattern) -> None:
        """Check structural, derived-market, and closed-valuation fields."""
        self.validator.validate_membership(
            label=pattern.label,
            nodes=pattern.nodes,
            perimeter=pattern.perimeter,
        )
        expected_schools = tuple(
            sorted(
                school
                for school, node in self.market.school_nodes.items()
                if node in pattern.nodes
            )
        )
        expected_segments = tuple(
            sorted(
                segment.segment_id
                for segment in self.market.segments
                if segment.node in pattern.nodes
            )
        )
        if pattern.school_ids != expected_schools:
            raise ValueError("Analytical pattern school_ids do not match its nodes.")
        if pattern.segment_ids != expected_segments:
            raise ValueError("Analytical pattern segment_ids do not match its nodes.")
        mechanism = pattern.mechanism
        if mechanism is None or not mechanism.closed:
            raise ValueError("Analytical master requires a closed mechanism valuation.")
        if not math.isclose(
            pattern.shi_welfare,
            mechanism.primal_objective,
            rel_tol=self.optimality_tolerance,
            abs_tol=self.optimality_tolerance,
        ):
            raise ValueError("Analytical pattern welfare disagrees with its mechanism.")
        if mechanism.primal_dual_gap > self.optimality_tolerance:
            raise ValueError("Analytical pattern valuation gap exceeds tolerance.")


def validate_zoned_shi_market(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
) -> None:
    """Validate the isolated resource assumptions needed by complete zones."""
    validate_analytical_welfare_market(market)
    schools = set(market.school_capacities)
    if set(market.zone_restricted_schools) != schools:
        unrestricted = sorted(schools - set(market.zone_restricted_schools))
        raise ValueError(
            "Zoned Shi optimization requires every school to be zone restricted; "
            f"unrestricted schools: {unrestricted}."
        )
    graph_nodes = set(problem.nodes)
    missing = set(market.school_nodes.values()) - graph_nodes
    missing.update(
        segment.node for segment in market.segments if segment.node not in graph_nodes
    )
    if missing:
        raise ValueError(
            f"Analytical market uses nodes outside the graph: {sorted(missing)}."
        )
    if problem.cutoff_market is not None:
        raise ValueError(
            "A problem cannot attach cutoff and analytical recourse together."
        )


def analytical_market_fingerprint(market: AnalyticalWelfareMarket) -> str:
    """Return a deterministic digest of every Shi-objective primitive."""
    digest = hashlib.sha256()
    digest.update(float(market.beta).hex().encode())
    for school in sorted(market.school_capacities):
        digest.update(
            f"S:{school}:{market.school_nodes[school]}:{market.school_capacities[school]};".encode()
        )
    for segment in sorted(market.segments, key=lambda item: item.segment_id):
        digest.update(
            f"T:{segment.segment_id}:{segment.node}:{float(segment.mass).hex()}:"
            f"{float(segment.outside_utility).hex()};".encode()
        )
        for school in sorted(segment.eligible_schools):
            digest.update(
                f"E:{school}:{float(segment.systematic_utilities[school]).hex()};".encode()
            )
    return digest.hexdigest()[:24]


def _restrict_segment(
    segment: AnalyticalWelfareSegment,
    selected_schools: set[int],
) -> AnalyticalWelfareSegment:
    eligible = tuple(
        school for school in segment.eligible_schools if school in selected_schools
    )
    return AnalyticalWelfareSegment(
        segment_id=segment.segment_id,
        node=segment.node,
        mass=segment.mass,
        eligible_schools=eligible,
        priorities={school: segment.priorities[school] for school in eligible},
        systematic_utilities={
            school: segment.systematic_utilities[school] for school in eligible
        },
        outside_utility=segment.outside_utility,
    )


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Analytical pattern {name} must be an integer.")
    return int(value)
