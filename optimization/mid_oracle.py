"""Independent least-cutoff evaluators for MID markets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from optimization.data.mid import MidMarket


@dataclass(frozen=True)
class MidOracleResult:
    cutoffs: dict[str, int | float]
    demands: dict[str, float]
    demand_masses: dict[str, int | float]
    assignment_masses: tuple[tuple[int | float, ...], ...]
    remaining_masses: tuple[tuple[int | float, ...], ...]
    outside_mass: float
    welfare: float
    fixed_point_value: int | float
    fixed_point_welfare: float
    type_fixed_point_values: tuple[int | float, ...]
    stable: bool
    minimal: bool


@dataclass(frozen=True)
class MidSeparation:
    overloaded_programs: tuple[str, ...]
    overload_prefixes: tuple[tuple[int, int], ...]
    utility_gap_prefixes: tuple[tuple[int, int], ...]


def _types_by_program(market: MidMarket) -> dict[str, tuple[tuple[int, int], ...]]:
    """Index the ``(type, rank)`` pairs that reference each program."""
    index: dict[str, list[tuple[int, int]]] = {
        program.program_id: [] for program in market.programs
    }
    for type_index, student_type in enumerate(market.types):
        for rank, program_id in enumerate(student_type.programs):
            index[program_id].append((type_index, rank))
    return {program_id: tuple(pairs) for program_id, pairs in index.items()}


def _program_demand_mass(
    market: MidMarket,
    programs: dict[str, object],
    zoning: dict[int, int],
    cutoffs: dict[str, float],
    lottery_scale: float,
    program_id: str,
    ranked: dict[str, tuple[tuple[int, int], ...]],
) -> int | float:
    """Demand mass at one program, walking only the types that rank it.

    A cutoff change at ``program_id`` can only move that program's own demand
    through the types that rank it, and only via the prefix of each such type up
    to that rank.  Bisecting on this instead of a full-market pass is what makes
    the cutoff search affordable.
    """
    zero = 0 if isinstance(lottery_scale, int) else 0.0
    total = zero
    for type_index, target_rank in ranked[program_id]:
        student_type = market.types[type_index]
        node_zone = zoning[student_type.node]
        remaining = lottery_scale
        for rank in range(target_rank + 1):
            ranked_id = student_type.programs[rank]
            priority = student_type.priorities[rank]
            program = programs[ranked_id]
            accessible = program.citywide or (node_zone == zoning[program.school_node])
            threshold = max(cutoffs[ranked_id] - priority * lottery_scale, zero)
            effective = threshold if accessible else lottery_scale
            next_remaining = min(remaining, effective)
            if rank == target_rank:
                total += student_type.count * (remaining - next_remaining)
            remaining = next_remaining
            if remaining <= zero:
                break
    return total


def _fast_demand_masses(
    market: MidMarket,
    zoning: dict[int, int],
    cutoffs: dict[str, int],
    lottery_scale: int,
) -> dict[str, int]:
    programs = market.program_by_id
    demand_masses = {program_id: 0 for program_id in programs}
    for student_type in market.types:
        remaining = lottery_scale
        type_node_zone = zoning[student_type.node]
        count = student_type.count
        for program_id, priority in zip(student_type.programs, student_type.priorities):
            program = programs[program_id]
            accessible = program.citywide or (
                type_node_zone == zoning[program.school_node]
            )
            threshold = max(cutoffs[program_id] - priority * lottery_scale, 0)
            effective = threshold if accessible else lottery_scale
            next_remaining = min(remaining, effective)
            mass = remaining - next_remaining
            if mass > 0:
                demand_masses[program_id] += count * mass
            remaining = next_remaining
            if remaining <= 0:
                break
    return demand_masses


def finite_grid_oracle(
    market: MidMarket,
    zoning: dict[int, int],
    lottery_scale: int,
    *,
    check_minimality: bool = True,
    warm_cutoffs: Mapping[str, int] | None = None,
) -> MidOracleResult:
    """Return the least integer cutoffs that clear every program capacity."""
    if isinstance(lottery_scale, bool) or not isinstance(lottery_scale, int):
        raise ValueError("MID lottery scale must be a positive integer.")
    if lottery_scale <= 0:
        raise ValueError("MID lottery scale must be a positive integer.")

    upper = _cutoff_upper_bounds(market, lottery_scale)
    programs = market.programs
    capacities = {p.program_id: p.capacity * lottery_scale for p in programs}
    program_by_id = market.program_by_id
    ranked = _types_by_program(market)

    if warm_cutoffs is not None:
        cutoffs = {
            p.program_id: min(
                int(upper[p.program_id]), max(0, int(warm_cutoffs.get(p.program_id, 0)))
            )
            for p in programs
        }
    else:
        cutoffs = {program.program_id: 0 for program in programs}

    def trial_demand(program_id: str, value: int) -> int:
        saved = cutoffs[program_id]
        cutoffs[program_id] = value
        try:
            return _program_demand_mass(
                market,
                program_by_id,
                zoning,
                cutoffs,
                lottery_scale,
                program_id,
                ranked,
            )
        finally:
            cutoffs[program_id] = saved

    demand_masses = _fast_demand_masses(market, zoning, cutoffs, lottery_scale)

    # Phase 1: Monotone Downward adjustments (only when warm_cutoffs are provided)
    if warm_cutoffs is not None:
        while True:
            decreased = False
            for program in programs:
                program_id = program.program_id
                cap = capacities[program_id]
                if cutoffs[program_id] > 0 and demand_masses[program_id] <= cap:
                    if trial_demand(program_id, cutoffs[program_id] - 1) <= cap:
                        low = 0
                        high = cutoffs[program_id] - 1
                        while low < high:
                            middle = (low + high) // 2
                            if trial_demand(program_id, middle) <= cap:
                                high = middle
                            else:
                                low = middle + 1
                        cutoffs[program_id] = low
                        demand_masses = _fast_demand_masses(
                            market, zoning, cutoffs, lottery_scale
                        )
                        decreased = True
            if not decreased:
                break

    # Phase 2: Monotone Upward adjustments
    max_updates = sum(upper.values()) + 1
    updates = 0
    while True:
        increased = False
        for program in programs:
            program_id = program.program_id
            cap = capacities[program_id]
            if demand_masses[program_id] > cap:
                low = cutoffs[program_id] + 1
                high = int(upper[program_id])
                while low < high:
                    middle = (low + high) // 2
                    if trial_demand(program_id, middle) <= cap:
                        high = middle
                    else:
                        low = middle + 1
                cutoffs[program_id] = low
                demand_masses = _fast_demand_masses(
                    market, zoning, cutoffs, lottery_scale
                )
                increased = True
                updates += 1
                if updates > max_updates:
                    raise RuntimeError(
                        "MID finite-grid cutoff iteration did not converge."
                    )
        if not increased:
            break

    result = _evaluate(market, zoning, cutoffs, lottery_scale)
    minimal = True
    if check_minimality:
        for program in programs:
            program_id = program.program_id
            if cutoffs[program_id] <= 0:
                continue
            if (
                trial_demand(program_id, cutoffs[program_id] - 1)
                <= capacities[program_id]
            ):
                minimal = False
                break
    return MidOracleResult(
        **{**result.__dict__, "minimal": minimal},
    )


def continuum_oracle(
    market: MidMarket,
    zoning: dict[int, int],
    *,
    tolerance: float = 1e-8,
) -> MidOracleResult:
    """Return continuous least cutoffs with unit lottery mass."""
    if tolerance <= 0:
        raise ValueError("MID continuum tolerance must be positive.")
    upper = _cutoff_upper_bounds(market, 1.0)
    cutoffs = {program.program_id: 0.0 for program in market.programs}
    program_by_id = market.program_by_id
    ranked = _types_by_program(market)
    max_updates = max(1, len(market.programs) * 1000)
    updates = 0

    def demand_at(program_id: str, value: float | None = None) -> float:
        saved = cutoffs[program_id]
        if value is not None:
            cutoffs[program_id] = value
        try:
            return _program_demand_mass(
                market, program_by_id, zoning, cutoffs, 1.0, program_id, ranked
            )
        finally:
            cutoffs[program_id] = saved

    while True:
        changed = False
        for program in market.programs:
            program_id = program.program_id
            if demand_at(program_id) <= program.capacity + tolerance:
                continue
            low = cutoffs[program_id]
            high = upper[program_id]
            while high - low > tolerance:
                middle = (low + high) / 2
                if demand_at(program_id, middle) <= program.capacity:
                    high = middle
                else:
                    low = middle
            cutoffs[program_id] = high
            updates += 1
            if updates > max_updates:
                raise RuntimeError("MID continuum cutoff iteration did not converge.")
            changed = True
        if not changed:
            break

    result = _evaluate(market, zoning, cutoffs, 1.0, tolerance=tolerance)
    minimal = True
    for program in market.programs:
        cutoff = cutoffs[program.program_id]
        if cutoff <= tolerance:
            continue
        delta = min(cutoff, math.sqrt(tolerance))
        if (
            demand_at(program.program_id, cutoff - delta)
            <= program.capacity + tolerance
        ):
            minimal = False
            break
    return MidOracleResult(**{**result.__dict__, "minimal": minimal})


def evaluate_cutoffs(
    market: MidMarket,
    zoning: dict[int, int],
    cutoffs: dict[str, float],
    lottery_scale: float,
) -> MidOracleResult:
    """Evaluate supplied cutoffs without changing them."""
    return _evaluate(market, zoning, cutoffs, lottery_scale)


def separate_mid_prefixes(
    market: MidMarket,
    result: MidOracleResult,
    active_prefix_lengths: dict[int, int],
    master_assignment_masses: tuple[tuple[int, ...], ...],
    lottery_scale: int,
) -> MidSeparation:
    """Return exact preference prefixes needed to refine one master result."""
    if len(result.assignment_masses) != len(market.types):
        raise ValueError("MID cutoff result does not match the market types.")
    if len(master_assignment_masses) != len(market.types):
        raise ValueError("MID master masses do not match the market types.")
    prefixes = _validated_prefix_lengths(market, active_prefix_lengths)
    for student_type, masses in zip(market.types, master_assignment_masses):
        if len(masses) != len(student_type.programs):
            raise ValueError("MID master masses do not match type preferences.")

    overloaded = tuple(
        program.program_id
        for program in market.programs
        if result.demand_masses[program.program_id] > program.capacity * lottery_scale
    )
    overload_targets: dict[int, int] = {}
    if overloaded:
        capacities = {
            program.program_id: program.capacity for program in market.programs
        }
        for program_id in overloaded:
            active_demand = 0
            candidates = []
            for type_index, student_type in enumerate(market.types):
                for rank, ranked_program in enumerate(student_type.programs):
                    if ranked_program != program_id:
                        continue
                    contribution = student_type.count * int(
                        result.assignment_masses[type_index][rank]
                    )
                    if rank < prefixes[type_index]:
                        active_demand += contribution
                    elif contribution > 0:
                        candidates.append((contribution, type_index, rank))
                    break

            selected_demand = sum(
                contribution
                for contribution, type_index, rank in candidates
                if overload_targets.get(type_index, prefixes[type_index]) > rank
            )
            capacity_mass = capacities[program_id] * lottery_scale
            for contribution, type_index, rank in sorted(
                candidates,
                key=lambda item: (-item[0], item[1], item[2]),
            ):
                if active_demand + selected_demand > capacity_mass:
                    break
                if overload_targets.get(type_index, prefixes[type_index]) > rank:
                    continue
                overload_targets[type_index] = rank + 1
                selected_demand += contribution
            if active_demand + selected_demand <= capacity_mass:
                raise RuntimeError(
                    f"MID overload separation could not cover program {program_id}."
                )

    utility_targets: dict[int, int] = {}
    if not overloaded:
        for type_index, student_type in enumerate(market.types):
            prefix_length = prefixes[type_index]
            if prefix_length == len(student_type.programs):
                continue
            master_masses = master_assignment_masses[type_index]
            master_value = sum(
                utility * mass
                for utility, mass in zip(
                    student_type.scaled_utility_sums,
                    master_masses,
                )
            )
            if master_value <= result.type_fixed_point_values[type_index]:
                continue
            actual_masses = result.assignment_masses[type_index]
            target_rank = next(
                (
                    rank
                    for rank in range(prefix_length, len(student_type.programs))
                    if master_masses[rank] != actual_masses[rank]
                ),
                None,
            )
            if target_rank is None:
                raise RuntimeError("MID utility-gap separation made no progress.")
            utility_targets[type_index] = target_rank + 1

    return MidSeparation(
        overloaded_programs=overloaded,
        overload_prefixes=tuple(sorted(overload_targets.items())),
        utility_gap_prefixes=tuple(sorted(utility_targets.items())),
    )


def _validated_prefix_lengths(
    market: MidMarket,
    active_prefix_lengths: dict[int, int],
) -> tuple[int, ...]:
    for type_index, prefix_length in active_prefix_lengths.items():
        if isinstance(type_index, bool) or not isinstance(type_index, int):
            raise ValueError("MID active-prefix type indices must be integers.")
        if type_index < 0 or type_index >= len(market.types):
            raise ValueError(f"Unknown MID type index: {type_index}.")
        if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
            raise ValueError("MID active-prefix lengths must be integers.")
        if not 0 <= prefix_length <= len(market.types[type_index].programs):
            raise ValueError(
                f"Invalid MID prefix length {prefix_length} for type {type_index}."
            )
    return tuple(
        active_prefix_lengths.get(type_index, 0)
        for type_index in range(len(market.types))
    )


def _evaluate(
    market: MidMarket,
    zoning: dict[int, int],
    cutoffs: dict[str, float],
    lottery_scale: float,
    *,
    tolerance: float = 0.0,
) -> MidOracleResult:
    programs = market.program_by_id
    expected = set(programs)
    if set(cutoffs) != expected:
        raise ValueError("MID cutoffs must contain every market program exactly once.")
    if lottery_scale <= 0:
        raise ValueError("MID lottery scale must be positive.")
    required_nodes = {student_type.node for student_type in market.types}
    required_nodes.update(
        program.school_node for program in market.programs if not program.citywide
    )
    missing_nodes = required_nodes - set(zoning)
    if missing_nodes:
        raise ValueError(f"MID zoning is missing graph nodes: {sorted(missing_nodes)}.")

    integral_grid = isinstance(lottery_scale, int) and all(
        isinstance(cutoff, int) for cutoff in cutoffs.values()
    )
    demand_masses = {program_id: 0 for program_id in programs}
    assignment_rows = []
    remaining_rows = []
    welfare_terms = []
    type_fixed_point_values = []
    outside_mass_units = 0
    for student_type in market.types:
        remaining = lottery_scale
        assignments = []
        remainders = []
        type_fixed_point_terms = []
        for rank, (program_id, priority) in enumerate(
            zip(student_type.programs, student_type.priorities)
        ):
            program = programs[program_id]
            accessible = program.citywide or (
                zoning[student_type.node] == zoning[program.school_node]
            )
            zero = 0 if integral_grid else 0.0
            threshold = max(cutoffs[program_id] - priority * lottery_scale, zero)
            effective = threshold if accessible else lottery_scale
            next_remaining = min(remaining, effective)
            mass = remaining - next_remaining
            assignments.append(mass)
            remainders.append(next_remaining)
            demand_masses[program_id] += student_type.count * mass
            welfare_terms.append(student_type.utility_sums[rank] * mass)
            type_fixed_point_terms.append(student_type.scaled_utility_sums[rank] * mass)
            remaining = next_remaining
        outside_mass_units += student_type.count * remaining
        assignment_rows.append(tuple(assignments))
        remaining_rows.append(tuple(remainders))
        type_fixed_point_values.append(
            sum(type_fixed_point_terms)
            if integral_grid
            else math.fsum(type_fixed_point_terms)
        )

    demands = {
        program_id: mass / lottery_scale for program_id, mass in demand_masses.items()
    }
    welfare = math.fsum(welfare_terms) / lottery_scale
    fixed_point_value = (
        sum(type_fixed_point_values)
        if integral_grid
        else math.fsum(type_fixed_point_values)
    )
    fixed_point_welfare = fixed_point_value / (lottery_scale * market.utility_scale)
    outside_mass = outside_mass_units / lottery_scale
    if integral_grid:
        stable = all(
            demand_masses[program.program_id] <= program.capacity * lottery_scale
            for program in market.programs
        )
        reported_cutoffs = {
            program_id: int(value) for program_id, value in cutoffs.items()
        }
    else:
        stable = all(
            demand_masses[program.program_id]
            <= (program.capacity + tolerance) * lottery_scale
            for program in market.programs
        )
        reported_cutoffs = {
            program_id: float(value) for program_id, value in cutoffs.items()
        }
    return MidOracleResult(
        cutoffs=reported_cutoffs,
        demands=demands,
        demand_masses=demand_masses,
        assignment_masses=tuple(assignment_rows),
        remaining_masses=tuple(remaining_rows),
        outside_mass=outside_mass,
        welfare=welfare,
        fixed_point_value=fixed_point_value,
        fixed_point_welfare=fixed_point_welfare,
        type_fixed_point_values=tuple(type_fixed_point_values),
        stable=stable,
        minimal=False,
    )


def cutoff_upper_bounds(
    market: MidMarket, lottery_scale: float
) -> dict[str, int | float]:
    """Smallest cutoff per program that is guaranteed to clear its capacity.

    A program only ever sees demand from the types that rank it, and the most
    it can ever see from one such type is that type's whole lottery mass -- the
    case where every better-ranked choice rejects the type outright.  So

        Dmax_s(p) = sum_g n_g * (S - clamp(p - rho_g * S, 0, S))

    upper-bounds demand at ``s`` under cutoff ``p`` for *every* zoning and every
    profile of other cutoffs.  ``Dmax_s`` is continuous and non-increasing in
    ``p``, so the least ``p`` with ``Dmax_s(p) <= q_s * S`` bounds the least
    clearing cutoff, and bounds the cutoff of any welfare-maximizing solution:
    lowering a cutoff to that point keeps ``s`` feasible, weakly lowers demand
    everywhere else, and weakly raises welfare.

    When that bound is zero the program can never be over-demanded, so its
    cutoff is identically zero and the solver can fix it.
    """
    integral = isinstance(lottery_scale, int) and not isinstance(lottery_scale, bool)
    zero = 0 if integral else 0.0
    rows: dict[str, list[tuple[int, int]]] = {
        program.program_id: [] for program in market.programs
    }
    for student_type in market.types:
        for program_id, priority in zip(student_type.programs, student_type.priorities):
            rows[program_id].append((student_type.count, priority))

    bounds: dict[str, int | float] = {}
    for program in market.programs:
        demand_rows = rows[program.program_id]
        if not demand_rows:
            bounds[program.program_id] = zero
            continue
        ceiling = (max(priority for _, priority in demand_rows) + 1) * lottery_scale
        capacity = program.capacity * lottery_scale

        def max_demand(cutoff: float, demand_rows=demand_rows) -> float:
            return sum(
                count
                * (
                    lottery_scale
                    - min(
                        max(cutoff - priority * lottery_scale, zero),
                        lottery_scale,
                    )
                )
                for count, priority in demand_rows
            )

        if max_demand(zero) <= capacity:
            bounds[program.program_id] = zero
            continue
        if integral:
            low, high = 0, int(ceiling)
            while low < high:
                middle = (low + high) // 2
                if max_demand(middle) <= capacity:
                    high = middle
                else:
                    low = middle + 1
            bounds[program.program_id] = low
        else:
            low, high = 0.0, float(ceiling)
            for _ in range(64):
                middle = (low + high) / 2
                if max_demand(middle) <= capacity:
                    high = middle
                else:
                    low = middle
            bounds[program.program_id] = high
    return bounds


def _cutoff_upper_bounds(market: MidMarket, lottery_scale: float) -> dict[str, float]:
    return cutoff_upper_bounds(market, lottery_scale)
