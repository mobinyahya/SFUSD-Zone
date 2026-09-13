"""What a single zone is worth, under each welfare definition we decompose.

A Dantzig--Wolfe column is a labelled zone and its value. Zone values are only
additive because no program is citywide: with every program confined to a zone,
a partition splits the applicants and the seats into independent submarkets, so
district welfare is the sum of zone welfares under any of the definitions
below. That is the whole reason the decomposition exists, and it is why
:func:`restrict_market` removes citywide programs before anything else happens.

Three objectives are implemented and they differ in what "welfare" means:

``boundary``
    Minus half the zone perimeter. Not a welfare at all -- it reproduces the
    base compactness objective and exists because it makes the decomposition
    testable against an enumerated optimum without a market.

``mid``
    Finite-grid least-cutoff MID welfare, evaluated by
    :func:`optimization.mid_oracle.finite_grid_oracle`. The zone's programs are
    given the componentwise least capacity-clearing integer cutoff vector, and
    welfare is the resulting utilitarian total.

``stable_matching``
    The welfare of the applicant-proposing deferred-acceptance outcome on the
    zone submarket, under one fixed tie-breaking draw. This is the definition
    the paper's access polytope describes: maximizing utility over the stable
    admissions polytope of a market returns deferred-acceptance welfare,
    because the polytope is the convex hull of that market's stable matchings,
    DA is the applicant-optimal one, and the utilities are rank-consistent.

The consequence for ``stable_matching`` is worth being explicit about. The
column value needs no LP and no MIP: it is one run of deferred acceptance on
the zone submarket, a few hundred microseconds, and it is *exactly* realized
welfare rather than an upper bound on it. The ``saa`` strategy's recourse LP
loads the aggregated Rothblum row instead of the comb family and therefore
over-reports realized DA welfare by roughly 195 to 252 units on
:math:`\\text{Block}_2`; here that gap does not arise, because a single zone's
stable matchings are enumerated by running the mechanism rather than described
by inequalities. The inequalities are still needed -- but only inside the
pricing model, where membership is a variable and the mechanism cannot be run.
"""

from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from optimization.data.mid import MidMarket
from optimization.data.saa import (
    SAA_TIE_BREAKING_METHODS,
    SaaMarket,
    SaaSample,
    sample_school_preferences,
)
from optimization.mid_oracle import finite_grid_oracle


def restrict_market(market: MidMarket, nodes=None) -> MidMarket:
    """Remove citywide programs, optionally keeping only one zone's market.

    Students remain in the cohort even when all their preferences disappear.
    Utilities and priority tiers retain their original values and rank order.
    """
    programs = tuple(
        p
        for p in market.programs
        if not p.citywide and (nodes is None or p.school_node in nodes)
    )
    available = {p.program_id for p in programs}
    types = []
    for student in market.types:
        if nodes is not None and student.node not in nodes:
            continue
        ranks = [i for i, p in enumerate(student.programs) if p in available]
        types.append(
            replace(
                student,
                programs=tuple(student.programs[i] for i in ranks),
                priorities=tuple(student.priorities[i] for i in ranks),
                utility_sums=tuple(student.utility_sums[i] for i in ranks),
                scaled_utility_sums=tuple(
                    student.scaled_utility_sums[i] for i in ranks
                ),
            )
        )
    count = sum(t.count for t in types)
    return replace(
        market,
        programs=programs,
        types=tuple(types),
        student_count=count,
        outside_only_student_count=sum(t.count for t in types if not t.programs),
        # The source only stores the district utility-cohort count, not a flag
        # per type. This diagnostic is not used by the oracle.
        utility_student_count=(
            market.utility_student_count
            if nodes is None
            else sum(t.count for t in types if t.programs)
        ),
    )


def restrict_sampled_market(market: SaaMarket) -> SaaMarket:
    """Drop citywide programs from an individual market, keeping every student."""

    programs = tuple(program for program in market.programs if not program.citywide)
    available = {program.program_id for program in programs}
    students = []
    for student in market.students:
        ranks = [
            rank
            for rank, program_id in enumerate(student.programs)
            if program_id in available
        ]
        students.append(
            replace(
                student,
                programs=tuple(student.programs[rank] for rank in ranks),
                priorities=tuple(student.priorities[rank] for rank in ranks),
                utilities=tuple(student.utilities[rank] for rank in ranks),
                scaled_utilities=tuple(
                    student.scaled_utilities[rank] for rank in ranks
                ),
            )
        )
    return replace(market, programs=programs, students=tuple(students))


@dataclass(frozen=True)
class SampledMatchingMarket:
    """One tie-breaking draw, indexed for repeated zone-restricted DA runs.

    ``sample.school_orders`` is positional in ``market.programs``, so the two
    fields are only jointly meaningful; build these through
    :func:`prepare_sampled_matching`, which restricts the market first and
    draws second.
    """

    market: SaaMarket
    sample: SaaSample
    preferences: tuple[tuple[tuple[str, float], ...], ...]
    students_by_node: dict[int, tuple[int, ...]]
    programs_by_node: dict[int, tuple[str, ...]]
    capacity: dict[str, int]
    position: dict[str, dict[int, int]]

    @property
    def gamma_size(self) -> int:
        return sum(len(row) for row in self.preferences)


def prepare_sampled_matching(
    market: SaaMarket, sample: SaaSample
) -> SampledMatchingMarket:
    """Index a restricted market and its drawn priority orders."""

    if len(sample.school_orders) != len(market.programs):
        raise ValueError("Sampled priority orders must align with market programs.")
    if any(program.citywide for program in market.programs):
        raise ValueError("Zone welfare decomposition forbids citywide programs.")

    students_by_node: dict[int, list[int]] = {}
    programs_by_node: dict[int, list[str]] = {}
    preferences = []
    for index, student in enumerate(market.students):
        preferences.append(tuple(zip(student.programs, student.utilities)))
        students_by_node.setdefault(student.node, []).append(index)
    for program in market.programs:
        programs_by_node.setdefault(program.school_node, []).append(program.program_id)

    return SampledMatchingMarket(
        market=market,
        sample=sample,
        preferences=tuple(preferences),
        students_by_node={
            node: tuple(indices) for node, indices in students_by_node.items()
        },
        programs_by_node={
            node: tuple(ids) for node, ids in programs_by_node.items()
        },
        capacity={
            program.program_id: int(program.capacity) for program in market.programs
        },
        position={
            program.program_id: {
                student_index: rank
                for rank, student_index in enumerate(sample.school_orders[index])
            }
            for index, program in enumerate(market.programs)
        },
    )


def deferred_acceptance(
    prepared: SampledMatchingMarket, nodes
) -> tuple[dict[int, str], float]:
    """Applicant-proposing deferred acceptance on the submarket inside ``nodes``.

    Returns the matching and its utilitarian welfare. Only residents of
    ``nodes`` apply and only programs whose school sits in ``nodes`` admit, so
    this is the zone submarket of the fixed tie-breaking draw. Each program
    holds the best ``q_s`` proposals it has seen under its drawn order, which
    is why a max-heap keyed by order position is the whole data structure.
    """

    members = frozenset(nodes)
    available = {
        program_id
        for node in members
        for program_id in prepared.programs_by_node.get(node, ())
        if prepared.capacity[program_id] > 0
    }
    held: dict[str, list[tuple[int, int]]] = {
        program_id: [] for program_id in available
    }
    seat: dict[int, str] = {}
    utility: dict[int, float] = {}
    cursor: dict[int, int] = {}
    pending = [
        index
        for node in sorted(members)
        for index in prepared.students_by_node.get(node, ())
    ]
    while pending:
        student = pending.pop()
        row = prepared.preferences[student]
        rank = cursor.get(student, 0)
        while rank < len(row):
            program_id, value = row[rank]
            rank += 1
            if program_id not in available:
                continue
            place = prepared.position[program_id][student]
            heap = held[program_id]
            if len(heap) < prepared.capacity[program_id]:
                heapq.heappush(heap, (-place, student))
            else:
                worst_place, worst = heap[0]
                if -worst_place <= place:
                    continue
                heapq.heapreplace(heap, (-place, student))
                seat.pop(worst, None)
                utility.pop(worst, None)
                pending.append(worst)
            seat[student] = program_id
            utility[student] = float(value)
            break
        cursor[student] = rank
    return seat, math.fsum(utility.values())


class ZoneObjective(ABC):
    """The exact value of one labelled zone, always maximized."""

    kind: str = "objective"

    @abstractmethod
    def score(self, nodes: frozenset[int], perimeter: int) -> float: ...

    def validate(self, nodes) -> None:
        """Check the market lives on the graph the zones are cut from."""

    @abstractmethod
    def upper_bound(self) -> float:
        """An a-priori bound on a whole partition's value, for the root node.

        Only used as the root's starting bound, so it may be crude; column
        generation replaces it with the pricing-corrected bound of
        Proposition 9 on the first proved round.
        """

    def metadata(self) -> dict:
        return {"dw_objective": self.kind}


class BoundaryZoneObjective(ZoneObjective):
    """Minus half the zone perimeter, so a partition's sum is the boundary cost."""

    kind = "boundary"

    def score(self, nodes: frozenset[int], perimeter: int) -> float:
        return -perimeter / 2.0

    def upper_bound(self) -> float:
        # Every score is non-positive and a partition with no cut edges is
        # conceivable on a disconnected graph, so zero is the bound.
        return 0.0

    def metadata(self) -> dict:
        return {
            "dw_objective": self.kind,
            "objective_kind": "boundary_cost",
            "dw_zone_value": "half_weighted_perimeter",
        }


class MidZoneObjective(ZoneObjective):
    """Finite-grid least-cutoff MID welfare of the zone submarket."""

    kind = "mid"

    def __init__(self, market: MidMarket, lottery_scale: int = 20) -> None:
        if (
            isinstance(lottery_scale, bool)
            or not isinstance(lottery_scale, int)
            or lottery_scale <= 0
        ):
            raise ValueError("MID lottery scale must be a positive integer.")
        self.market = restrict_market(market)
        self.lottery_scale = lottery_scale

    def score(self, nodes: frozenset[int], perimeter: int) -> float:
        local = restrict_market(self.market, nodes)
        return float(
            finite_grid_oracle(
                local,
                {node: 0 for node in nodes},
                self.lottery_scale,
                check_minimality=False,
            ).welfare
        )

    def validate(self, nodes) -> None:
        if any(student.node not in nodes for student in self.market.types):
            raise ValueError("DW market students must belong to the graph.")
        if any(program.school_node not in nodes for program in self.market.programs):
            raise ValueError("DW market schools must belong to the graph.")

    def upper_bound(self) -> float:
        """The first-choice bound: every type assigned its own best program."""

        return math.fsum(
            max(student.utility_sums, default=0.0) for student in self.market.types
        )

    def metadata(self) -> dict:
        return {
            "dw_objective": self.kind,
            "objective_kind": "mid_program_welfare",
            "dw_zone_value": "finite_grid_least_cutoff_welfare",
            "mid_lottery_scale": self.lottery_scale,
            "dw_market_types": len(self.market.types),
            "dw_market_programs": len(self.market.programs),
        }


class StableMatchingZoneObjective(ZoneObjective):
    """Deferred-acceptance welfare of the zone submarket, one lottery draw."""

    kind = "stable_matching"

    def __init__(
        self,
        market: SaaMarket,
        *,
        tie_breaking_method: str = "MTB",
        seed: int = 42,
    ) -> None:
        method = str(tie_breaking_method).upper()
        if method not in SAA_TIE_BREAKING_METHODS:
            raise ValueError("saa_tie_breaking_method must be one of: MTB, STB.")
        # Restrict first, draw second. ``sample_school_preferences`` indexes
        # its orders by position in ``market.programs``, so removing a program
        # after the draw would attach each program another one's priorities.
        # One seed: a single draw fixes the lottery, which is what makes the
        # zone value an exact realized welfare rather than an average of them.
        restricted = restrict_sampled_market(market)
        sample = sample_school_preferences(restricted, 1, method, int(seed))[0]
        self.prepared = prepare_sampled_matching(restricted, sample)
        self.tie_breaking_method = method

    @property
    def market(self) -> SaaMarket:
        return self.prepared.market

    def score(self, nodes: frozenset[int], perimeter: int) -> float:
        return deferred_acceptance(self.prepared, nodes)[1]

    def validate(self, nodes) -> None:
        if any(student.node not in nodes for student in self.market.students):
            raise ValueError("DW market students must belong to the graph.")
        if any(program.school_node not in nodes for program in self.market.programs):
            raise ValueError("DW market schools must belong to the graph.")

    def upper_bound(self) -> float:
        """The first-choice bound: every applicant seated at their own best program."""

        return math.fsum(
            max(student.utilities, default=0.0) for student in self.market.students
        )

    def metadata(self) -> dict:
        return {
            "dw_objective": self.kind,
            "objective_kind": "sampled_stable_matching_welfare",
            "dw_zone_value": "deferred_acceptance_welfare",
            "dw_tie_breaking_method": self.tie_breaking_method,
            "dw_sample_seed": self.prepared.sample.seed,
            "dw_market_students": len(self.market.students),
            "dw_market_programs": len(self.market.programs),
            "dw_gamma_size": self.prepared.gamma_size,
        }
