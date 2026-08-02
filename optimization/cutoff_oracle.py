"""Analytical DA-STB cutoffs for fixed isolated school markets.

The zoning model represents one common lottery as ``lottery_scale`` units of
mass per student.  This module solves the same integer-grid score-limit market
without a mathematical-programming solver.  It is intentionally independent
of the CP-SAT formulation so it can validate zoning solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from optimization.problem import CutoffMarket, CutoffStudent


@dataclass(frozen=True)
class MarketCutoffResult:
    """Least capacity-feasible cutoff vector on an integer lottery grid."""

    cutoffs: dict[int, int]
    demands: dict[int, int]
    assignments: dict[int, dict[int, int]]
    lottery_scale: int
    iterations: int
    grid_minimal: bool

    @property
    def objective(self) -> int:
        return sum(self.cutoffs.values())

    @property
    def normalized_objective(self) -> float:
        return self.objective / self.lottery_scale

    @property
    def normalized_cutoffs(self) -> dict[int, float]:
        return {
            school: cutoff / self.lottery_scale
            for school, cutoff in self.cutoffs.items()
        }


@dataclass(frozen=True)
class ZonedCutoffResult:
    """Cutoff equilibria and certificate for every isolated zone."""

    zones: dict[int, MarketCutoffResult]
    school_cutoffs: dict[int, int]
    lottery_scale: int

    @property
    def objective(self) -> int:
        return sum(self.school_cutoffs.values())

    @property
    def normalized_objective(self) -> float:
        return self.objective / self.lottery_scale

    @property
    def grid_minimal(self) -> bool:
        return all(result.grid_minimal for result in self.zones.values())


@dataclass(frozen=True)
class ContinuumCutoffResult:
    """Market-clearing Azevedo-Leshno cutoffs with a continuous STB lottery."""

    cutoffs: dict[int, float]
    demands: dict[int, float]
    lottery_size: float
    iterations: int
    stable: bool

    @property
    def objective(self) -> float:
        return sum(self.cutoffs.values())


@dataclass(frozen=True)
class ZonedContinuumCutoffResult:
    """Continuous stable score-limit equilibria for all isolated zones."""

    zones: dict[int, ContinuumCutoffResult]
    school_cutoffs: dict[int, float]

    @property
    def objective(self) -> float:
        return sum(self.school_cutoffs.values())

    @property
    def stable(self) -> bool:
        return all(result.stable for result in self.zones.values())


def solve_market_cutoffs(
    students: Iterable[CutoffStudent],
    school_capacities: Mapping[int, int],
    lottery_scale: int,
) -> MarketCutoffResult:
    """Find the least integer-grid cutoff vector for one fixed market.

    A student's priority at school ``s`` is ``priority[i, s] * L + lottery``,
    with the same continuous lottery mass on ``[0, L)`` at every school.
    Restricting cutoffs to integers can leave less than one grid step of slack;
    this function therefore certifies grid minimality, not exact stability.
    """
    if isinstance(lottery_scale, bool) or not isinstance(lottery_scale, int):
        raise ValueError("lottery_scale must be a positive integer.")
    if lottery_scale <= 0:
        raise ValueError("lottery_scale must be a positive integer.")

    capacities = {int(school): int(capacity) for school, capacity in school_capacities.items()}
    if any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("School capacities must be non-negative.")

    student_list = tuple(students)
    schools = tuple(capacities)
    school_set = set(schools)
    max_priority = 0
    for student in student_list:
        if len(student.preferences) != len(set(student.preferences)):
            raise ValueError(
                f"Student {student.studentno} lists a school more than once."
            )
        unknown = set(student.preferences) - school_set
        if unknown:
            raise ValueError(
                f"Student {student.studentno} lists unknown schools: {sorted(unknown)}."
            )
        missing = set(student.preferences) - set(student.priorities)
        if missing:
            raise ValueError(
                f"Student {student.studentno} has no priorities for {sorted(missing)}."
            )
        if student.priorities:
            max_priority = max(max_priority, max(student.priorities.values()))
        if any(priority < 0 for priority in student.priorities.values()):
            raise ValueError("Student priorities must be non-negative.")

    cutoffs = {school: 0 for school in schools}
    max_cutoff = (max_priority + 1) * lottery_scale
    updates = 0
    # Every successful update raises an integer cutoff, which gives this loose
    # finite bound without relying on numerical convergence tolerances.
    max_updates = max(1, len(schools) * (max_cutoff + 1))

    while True:
        changed = False
        for school in schools:
            terms = _school_demand_terms(
                student_list, school, cutoffs, lottery_scale
            )
            required = _minimum_clearing_cutoff(
                terms,
                capacities[school] * lottery_scale,
            )
            if required > cutoffs[school]:
                cutoffs[school] = required
                updates += 1
                changed = True
                if updates > max_updates:
                    raise RuntimeError("Cutoff iteration exceeded its finite update bound.")

        assignments, demands = assignments_and_demands(
            student_list, cutoffs, lottery_scale
        )
        clears = all(
            demands[school] <= capacities[school] * lottery_scale
            for school in schools
        )
        if not changed and clears:
            break

    grid_minimal = validate_market_cutoffs(
        student_list,
        capacities,
        cutoffs,
        lottery_scale,
    )
    return MarketCutoffResult(
        cutoffs=cutoffs,
        demands=demands,
        assignments=assignments,
        lottery_scale=lottery_scale,
        iterations=updates,
        grid_minimal=grid_minimal,
    )


def solve_zoned_cutoffs(
    market: CutoffMarket,
    node_assignment: Mapping[int, int],
    *,
    num_zones: int | None = None,
) -> ZonedCutoffResult:
    """Solve every isolated market induced by a node-to-zone assignment."""
    unrestricted = set(market.school_capacities) - set(market.zone_restricted_schools)
    if unrestricted:
        raise ValueError(
            "Isolated cutoff markets require every school to be zone restricted; "
            f"unrestricted schools: {sorted(unrestricted)}."
        )

    missing_nodes = {
        student.node for student in market.students if student.node not in node_assignment
    } | {
        node for node in market.school_nodes.values() if node not in node_assignment
    }
    if missing_nodes:
        raise ValueError(f"Zoning omits market nodes: {sorted(missing_nodes)}.")

    if num_zones is None:
        num_zones = max(node_assignment.values(), default=-1) + 1
    schools_by_zone = {zone: [] for zone in range(num_zones)}
    for school in market.school_capacities:
        zone = int(node_assignment[market.school_nodes[school]])
        if zone not in schools_by_zone:
            raise ValueError(f"School {school} has invalid zone {zone}.")
        schools_by_zone[zone].append(school)

    students_by_zone = {zone: [] for zone in range(num_zones)}
    for student in market.students:
        zone = int(node_assignment[student.node])
        if zone not in students_by_zone:
            raise ValueError(f"Student {student.studentno} has invalid zone {zone}.")
        school_set = set(schools_by_zone[zone])
        preferences = tuple(
            school for school in student.preferences if school in school_set
        )
        students_by_zone[zone].append(
            CutoffStudent(
                studentno=student.studentno,
                node=student.node,
                preferences=preferences,
                priorities={
                    school: student.priorities[school] for school in preferences
                },
            )
        )

    results = {}
    all_cutoffs = {}
    for zone in range(num_zones):
        capacities = {
            school: market.school_capacities[school]
            for school in schools_by_zone[zone]
        }
        result = solve_market_cutoffs(
            students_by_zone[zone], capacities, market.lottery_scale
        )
        results[zone] = result
        all_cutoffs.update(result.cutoffs)

    return ZonedCutoffResult(
        zones=results,
        school_cutoffs=all_cutoffs,
        lottery_scale=market.lottery_scale,
    )


def solve_continuum_market_cutoffs(
    students: Iterable[CutoffStudent],
    school_capacities: Mapping[int, int],
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> ContinuumCutoffResult:
    """Solve exact continuous single-tie-breaker score limits for one market."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    students = tuple(students)
    capacities = {
        int(school): int(capacity)
        for school, capacity in school_capacities.items()
    }
    if any(capacity < 0 for capacity in capacities.values()):
        raise ValueError("School capacities must be non-negative.")
    schools = tuple(capacities)
    school_set = set(schools)
    for student in students:
        if len(student.preferences) != len(set(student.preferences)):
            raise ValueError(
                f"Student {student.studentno} lists a school more than once."
            )
        if set(student.preferences) - school_set:
            raise ValueError(f"Student {student.studentno} lists an unknown school.")

    cutoffs = {school: 0.0 for school in schools}
    demand_tolerance = tolerance * max(1, len(students))
    for iteration in range(1, max_iterations + 1):
        changed = False
        for school in schools:
            terms = []
            for student in students:
                remaining = 1.0
                for preferred in student.preferences:
                    if preferred == school:
                        terms.append((float(student.priorities[school]), remaining))
                        break
                    remaining = min(
                        remaining,
                        _continuous_threshold(
                            cutoffs[preferred], student.priorities[preferred]
                        ),
                    )
            required = _continuous_clearing_cutoff(
                terms, capacities[school], demand_tolerance
            )
            if required > cutoffs[school] + tolerance:
                cutoffs[school] = required
                changed = True

        demands = _continuum_demands(students, cutoffs)
        clears = all(
            demands[school] <= capacities[school] + demand_tolerance
            for school in schools
        )
        if not changed and clears:
            stable = all(
                cutoff <= tolerance
                or abs(demands[school] - capacities[school]) <= demand_tolerance
                for school, cutoff in cutoffs.items()
            )
            return ContinuumCutoffResult(
                cutoffs=cutoffs,
                demands=demands,
                lottery_size=1.0,
                iterations=iteration,
                stable=stable,
            )
    raise RuntimeError(
        f"Continuous cutoff iteration did not converge after {max_iterations} rounds."
    )


def solve_zoned_continuum_cutoffs(
    market: CutoffMarket,
    node_assignment: Mapping[int, int],
    *,
    num_zones: int | None = None,
) -> ZonedContinuumCutoffResult:
    """Solve exact continuous score-limit equilibria in every isolated zone."""
    zone_students, zone_capacities = _zoned_markets(
        market, node_assignment, num_zones=num_zones
    )
    results = {}
    cutoffs = {}
    for zone in zone_students:
        result = solve_continuum_market_cutoffs(
            zone_students[zone], zone_capacities[zone]
        )
        results[zone] = result
        cutoffs.update(result.cutoffs)
    return ZonedContinuumCutoffResult(results, cutoffs)


def assignments_and_demands(
    students: Iterable[CutoffStudent],
    cutoffs: Mapping[int, int],
    lottery_scale: int,
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """Reconstruct expected assignment mass from a cutoff vector."""
    demands = {int(school): 0 for school in cutoffs}
    assignments = {}
    for index, student in enumerate(students):
        remaining = lottery_scale
        student_assignment = {}
        for school in student.preferences:
            threshold = _threshold(
                cutoffs[school], student.priorities[school], lottery_scale
            )
            assigned = max(0, remaining - threshold)
            student_assignment[school] = assigned
            demands[school] += assigned
            remaining = min(remaining, threshold)
        assignments[index] = student_assignment
    return assignments, demands


def validate_market_cutoffs(
    students: Iterable[CutoffStudent],
    school_capacities: Mapping[int, int],
    cutoffs: Mapping[int, int],
    lottery_scale: int,
) -> bool:
    """Check capacity and integer-grid score-limit complementarity.

    At a positive equilibrium cutoff, lowering that school's cutoff by one
    lottery unit while holding the other score limits fixed must over-demand
    the school.  Together with capacity feasibility, this is the discrete
    market-clearing condition used by the CP-SAT model.
    """
    student_list = tuple(students)
    if set(cutoffs) != set(school_capacities):
        return False
    _, demands = assignments_and_demands(student_list, cutoffs, lottery_scale)
    for school, capacity in school_capacities.items():
        scaled_capacity = int(capacity) * lottery_scale
        if demands[school] > scaled_capacity:
            return False
        cutoff = cutoffs[school]
        if cutoff <= 0:
            continue
        lowered = dict(cutoffs)
        lowered[school] = cutoff - 1
        _, lowered_demands = assignments_and_demands(
            student_list, lowered, lottery_scale
        )
        if lowered_demands[school] <= scaled_capacity:
            return False
    return True


def _zoned_markets(
    market: CutoffMarket,
    node_assignment: Mapping[int, int],
    *,
    num_zones: int | None,
) -> tuple[dict[int, list[CutoffStudent]], dict[int, dict[int, int]]]:
    unrestricted = set(market.school_capacities) - set(market.zone_restricted_schools)
    if unrestricted:
        raise ValueError(
            "Isolated cutoff markets require every school to be zone restricted; "
            f"unrestricted schools: {sorted(unrestricted)}."
        )
    if num_zones is None:
        num_zones = max(node_assignment.values(), default=-1) + 1
    schools_by_zone = {zone: [] for zone in range(num_zones)}
    for school in market.school_capacities:
        zone = int(node_assignment[market.school_nodes[school]])
        schools_by_zone[zone].append(school)
    capacities = {
        zone: {
            school: market.school_capacities[school]
            for school in schools_by_zone[zone]
        }
        for zone in range(num_zones)
    }
    students = {zone: [] for zone in range(num_zones)}
    for student in market.students:
        zone = int(node_assignment[student.node])
        school_set = set(schools_by_zone[zone])
        preferences = tuple(
            school for school in student.preferences if school in school_set
        )
        students[zone].append(
            CutoffStudent(
                student.studentno,
                student.node,
                preferences,
                {school: student.priorities[school] for school in preferences},
            )
        )
    return students, capacities


def _continuum_demands(
    students: tuple[CutoffStudent, ...], cutoffs: Mapping[int, float]
) -> dict[int, float]:
    demands = {school: 0.0 for school in cutoffs}
    for student in students:
        remaining = 1.0
        for school in student.preferences:
            threshold = _continuous_threshold(
                cutoffs[school], student.priorities[school]
            )
            demands[school] += max(0.0, remaining - threshold)
            remaining = min(remaining, threshold)
    return demands


def _continuous_clearing_cutoff(
    terms: Iterable[tuple[float, float]],
    capacity: float,
    tolerance: float,
) -> float:
    terms = tuple(terms)
    demand = sum(upper for _, upper in terms)
    if demand <= capacity + tolerance:
        return 0.0
    events: dict[float, int] = {}
    for priority, upper in terms:
        if upper <= 0:
            continue
        events[priority] = events.get(priority, 0) - 1
        events[priority + upper] = events.get(priority + upper, 0) + 1
    cutoff = 0.0
    slope = 0
    for breakpoint in sorted(events):
        next_demand = demand + slope * (breakpoint - cutoff)
        if next_demand <= capacity + tolerance:
            if slope >= 0:
                return cutoff
            return min(breakpoint, cutoff + (demand - capacity) / -slope)
        demand = next_demand
        cutoff = breakpoint
        slope += events[breakpoint]
    raise RuntimeError("Could not find a continuous capacity-clearing cutoff.")


def _continuous_threshold(cutoff: float, priority: int) -> float:
    return min(1.0, max(0.0, cutoff - priority))


def _school_demand_terms(
    students: tuple[CutoffStudent, ...],
    school: int,
    cutoffs: Mapping[int, int],
    lottery_scale: int,
) -> list[tuple[int, int]]:
    """Return ``(priority score, prior rejection bound)`` demand terms."""
    terms = []
    for student in students:
        remaining = lottery_scale
        for preferred in student.preferences:
            if preferred == school:
                terms.append((student.priorities[school] * lottery_scale, remaining))
                break
            remaining = min(
                remaining,
                _threshold(
                    cutoffs[preferred],
                    student.priorities[preferred],
                    lottery_scale,
                ),
            )
    return terms


def _minimum_clearing_cutoff(
    terms: Iterable[tuple[int, int]], scaled_capacity: int
) -> int:
    """Invert one school's integer piecewise-linear demand curve exactly."""
    term_list = tuple(terms)
    demand = sum(upper for _, upper in term_list)
    if demand <= scaled_capacity:
        return 0

    events: dict[int, int] = {}
    for priority_score, upper in term_list:
        if upper <= 0:
            continue
        events[priority_score] = events.get(priority_score, 0) - 1
        end = priority_score + upper
        events[end] = events.get(end, 0) + 1

    cutoff = 0
    slope = 0
    for breakpoint in sorted(events):
        distance = breakpoint - cutoff
        next_demand = demand + slope * distance
        if next_demand <= scaled_capacity:
            if slope >= 0:
                return cutoff
            needed_drop = demand - scaled_capacity
            step = (needed_drop + (-slope) - 1) // (-slope)
            return min(breakpoint, cutoff + step)
        demand = next_demand
        cutoff = breakpoint
        slope += events[breakpoint]

    raise RuntimeError("Could not find a capacity-clearing cutoff.")


def _threshold(cutoff: int, priority: int, lottery_scale: int) -> int:
    return min(lottery_scale, max(0, cutoff - priority * lottery_scale))
