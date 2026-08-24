"""Independent least-cutoff evaluators for MID markets."""

from __future__ import annotations

import math
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


def finite_grid_oracle(
    market: MidMarket,
    zoning: dict[int, int],
    lottery_scale: int,
) -> MidOracleResult:
    """Return the least integer cutoffs that clear every program capacity."""
    if isinstance(lottery_scale, bool) or not isinstance(lottery_scale, int):
        raise ValueError("MID lottery scale must be a positive integer.")
    if lottery_scale <= 0:
        raise ValueError("MID lottery scale must be a positive integer.")

    upper = _cutoff_upper_bounds(market, lottery_scale)
    cutoffs = {program.program_id: 0 for program in market.programs}
    max_updates = sum(upper.values()) + 1
    updates = 0
    while True:
        changed = False
        for program in market.programs:
            program_id = program.program_id
            result = _evaluate(market, zoning, cutoffs, lottery_scale)
            if result.demand_masses[program_id] <= program.capacity * lottery_scale:
                continue
            low = int(cutoffs[program_id]) + 1
            high = upper[program_id]
            while low < high:
                middle = (low + high) // 2
                candidate = {**cutoffs, program_id: middle}
                demand_mass = _evaluate(
                    market, zoning, candidate, lottery_scale
                ).demand_masses[program_id]
                if demand_mass <= program.capacity * lottery_scale:
                    high = middle
                else:
                    low = middle + 1
            cutoffs[program_id] = low
            updates += 1
            if updates > max_updates:
                raise RuntimeError("MID finite-grid cutoff iteration did not converge.")
            changed = True
        if not changed:
            break

    result = _evaluate(market, zoning, cutoffs, lottery_scale)
    minimal = True
    for program in market.programs:
        program_id = program.program_id
        if cutoffs[program_id] <= 0:
            continue
        lower = {**cutoffs, program_id: cutoffs[program_id] - 1}
        if (
            _evaluate(market, zoning, lower, lottery_scale).demand_masses[program_id]
            <= program.capacity * lottery_scale
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
    max_updates = max(1, len(market.programs) * 1000)
    updates = 0
    while True:
        changed = False
        for program in market.programs:
            program_id = program.program_id
            result = _evaluate(market, zoning, cutoffs, 1.0, tolerance=tolerance)
            if result.demands[program_id] <= program.capacity + tolerance:
                continue
            low = cutoffs[program_id]
            high = upper[program_id]
            while high - low > tolerance:
                middle = (low + high) / 2
                candidate = {**cutoffs, program_id: middle}
                demand = _evaluate(
                    market, zoning, candidate, 1.0, tolerance=tolerance
                ).demands[program_id]
                if demand <= program.capacity:
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
        lower = {**cutoffs, program.program_id: cutoff - delta}
        demand = _evaluate(market, zoning, lower, 1.0, tolerance=tolerance).demands[
            program.program_id
        ]
        if demand <= program.capacity + tolerance:
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
        capacities = {program.program_id: program.capacity for program in market.programs}
        for program_id in overloaded:
            active_demand = 0
            candidates = []
            for type_index, student_type in enumerate(market.types):
                for rank, ranked_program in enumerate(student_type.programs):
                    if ranked_program != program_id:
                        continue
                    contribution = (
                        student_type.count
                        * int(result.assignment_masses[type_index][rank])
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


def _cutoff_upper_bounds(market: MidMarket, lottery_scale: float) -> dict[str, float]:
    priorities: dict[str, list[int]] = {
        program.program_id: [] for program in market.programs
    }
    for student_type in market.types:
        for program_id, priority in zip(student_type.programs, student_type.priorities):
            priorities[program_id].append(priority)
    return {
        program_id: (max(values) + 1) * lottery_scale if values else 0.0
        for program_id, values in priorities.items()
    }
