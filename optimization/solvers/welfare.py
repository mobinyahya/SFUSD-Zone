"""Exact finite-grid welfare maximization for isolated DA-STB markets."""

from __future__ import annotations

import math
import time

from ortools.sat.python import cp_model

from optimization.solution import ZoneSolution
from optimization.welfare_oracle import (
    outward_true_welfare_upper_bound,
    raw_welfare_upper_bound,
    solve_zoned_continuum_welfare,
    solve_zoned_welfare,
    validate_welfare_market,
)


class WelfareSolver:
    """Jointly optimize zoning and stable assignment-measure welfare."""

    def __init__(self, zoning_solver, *, utility_scale: int) -> None:
        self.zoning_solver = zoning_solver
        self.options = zoning_solver.options
        self.utility_scale = int(utility_scale)

    def solve(self, problem) -> ZoneSolution:
        market = problem.cutoff_market
        if market is None:
            raise ValueError("Welfare optimization requires a cutoff market.")
        unrestricted = set(market.school_capacities) - set(
            market.zone_restricted_schools
        )
        if unrestricted:
            raise ValueError(
                "welfare currently requires isolated markets; remove city-wide "
                f"schools before solving: {sorted(unrestricted)}."
            )
        if self.utility_scale <= 0:
            raise ValueError("welfare_utility_scale must be a positive integer.")
        validate_welfare_market(market, utility_scale=self.utility_scale)
        global_raw_upper_bound = raw_welfare_upper_bound(market, self.utility_scale)
        configured_upper_bound = self.options.get("welfare_raw_upper_bound")
        if configured_upper_bound is not None:
            if (
                isinstance(configured_upper_bound, bool)
                or not isinstance(configured_upper_bound, int)
                or configured_upper_bound < 0
            ):
                raise ValueError(
                    "welfare_raw_upper_bound must be a nonnegative integer."
                )
            global_raw_upper_bound = min(
                global_raw_upper_bound,
                configured_upper_bound,
            )

        started = time.monotonic()
        time_limit = float(self.options.get("solve_time_limit", 60.0))
        initial_assignment = (
            dict(problem.hint)
            if problem.hint is not None
            else self._initial_assignment(
                problem,
                min(30.0, max(1.0, time_limit * 0.15)),
            )
        )
        if initial_assignment is None:
            raise RuntimeError(
                "Could not construct an initial feasible welfare zoning."
            )
        initial_grid = solve_zoned_welfare(
            market,
            initial_assignment,
            num_zones=problem.Z,
            utility_scale=self.utility_scale,
        )
        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        self.zoning_solver._add_search_strategy(model, problem, x, y)
        for (zone, node), variable in x.items():
            model.AddHint(variable, int(initial_assignment[node] == zone))
        for node, variable in y.items():
            model.AddHint(variable, initial_assignment[node])

        cutoffs, raw_welfare = add_finite_grid_welfare_model(
            model,
            problem,
            x,
            initial_assignment,
            initial_grid,
            zoning_solver=self.zoning_solver,
            utility_scale=self.utility_scale,
        )
        if initial_grid.raw_scaled_welfare > global_raw_upper_bound:
            raise ValueError("welfare_raw_upper_bound is below the initial incumbent.")
        model.Add(raw_welfare <= global_raw_upper_bound)
        model.Maximize(raw_welfare)

        solver = cp_model.CpSolver()
        self.zoning_solver._configure_solver_parameters(solver)
        solver.parameters.max_time_in_seconds = max(
            0.1, time_limit - (time.monotonic() - started)
        )
        status = solver.Solve(model)
        wall_time = time.monotonic() - started
        status_name = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return ZoneSolution(
                problem=problem,
                assignment={},
                status=status_name,
                wall_time=wall_time,
                metadata={
                    **market.metadata,
                    "solver": "cp_bool",
                    "objective_kind": "stable_assignment_welfare",
                    "optimization_method": "direct_finite_grid_stable_welfare",
                },
            )

        assignment = self.zoning_solver._extract_assignment(solver, problem, x, y)
        grid = solve_zoned_welfare(
            market,
            assignment,
            num_zones=problem.Z,
            utility_scale=self.utility_scale,
        )
        continuum = solve_zoned_continuum_welfare(
            market, assignment, num_zones=problem.Z
        )
        raw_incumbent = int(round(solver.ObjectiveValue()))
        raw_upper_bound = min(
            global_raw_upper_bound,
            int(math.floor(solver.BestObjectiveBound() + 1e-6)),
        )
        certified = (
            configured_upper_bound is None
            and grid.raw_scaled_welfare >= raw_upper_bound
        )
        zone_stable = {
            str(zone): result.stable for zone, result in continuum.cutoffs.zones.items()
        }
        positive_grid_underfill = {
            school: market.school_capacities[school] * market.lottery_scale
            - zone_result.demands[school]
            for zone_result in grid.cutoffs.zones.values()
            for school, cutoff in zone_result.cutoffs.items()
            if cutoff > 0
        }
        scaled_normalizer = market.lottery_scale * self.utility_scale
        rounded_welfare = grid.raw_scaled_welfare / scaled_normalizer
        coefficient_error = math.nextafter(
            len(market.students) / (2 * self.utility_scale), math.inf
        )
        true_upper_bound = outward_true_welfare_upper_bound(
            raw_upper_bound, market, self.utility_scale
        )
        metadata = {
            **market.metadata,
            "solver": "cp_bool",
            "objective_kind": "stable_assignment_welfare",
            "optimization_method": "direct_finite_grid_stable_welfare",
            "market_coupling": "isolated_zones",
            "lottery_scale": market.lottery_scale,
            "welfare_utility_scale": self.utility_scale,
            "welfare": grid.welfare,
            "rounded_welfare": rounded_welfare,
            "raw_scaled_welfare": grid.raw_scaled_welfare,
            "raw_solver_incumbent": raw_incumbent,
            "raw_scaled_upper_bound": raw_upper_bound,
            "configured_raw_upper_bound": configured_upper_bound,
            "configured_upper_bound_scope": (
                "External diagnostic cap; not independently certified by this run."
                if configured_upper_bound is not None
                else None
            ),
            "rounded_welfare_upper_bound": raw_upper_bound / scaled_normalizer,
            "utility_rounding_error_bound": coefficient_error,
            "true_welfare_upper_bound": true_upper_bound,
            "true_welfare_gap_bound": max(
                0.0,
                true_upper_bound - grid.welfare,
            ),
            "global_optimum_certified": certified,
            "global_optimum_scope": (
                "Finite assignment and cutoff grid with utilities rounded to the "
                "configured fixed-point scale, over the encoded zoning domain."
            ),
            "school_cutoffs": grid.cutoffs.school_cutoffs,
            "normalized_school_cutoffs": {
                school: cutoff / market.lottery_scale
                for school, cutoff in grid.cutoffs.school_cutoffs.items()
            },
            "grid_minimal": grid.cutoffs.grid_minimal,
            "grid_max_positive_cutoff_underfill_mass": max(
                positive_grid_underfill.values(), default=0
            ),
            "continuum_welfare": continuum.welfare,
            "continuum_school_cutoffs": continuum.cutoffs.school_cutoffs,
            "stable": continuum.stable,
            "zone_stable": zone_stable,
            "stable_zone_count": sum(zone_stable.values()),
            "stability_definition": (
                "Independent continuous student-optimal market clearing in every zone."
            ),
            "model_stats": model.ModelStats(),
        }
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status="OPTIMAL" if certified else "FEASIBLE",
            objective=grid.welfare,
            wall_time=wall_time,
            metadata=metadata,
        )

    def _initial_assignment(self, problem, time_limit):
        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        self.zoning_solver._add_boundary_objective(model, problem, x, y)
        self.zoning_solver._add_hints(model, problem, x, y)
        self.zoning_solver._add_search_strategy(model, problem, x, y)
        solver = cp_model.CpSolver()
        self.zoning_solver._configure_solver_parameters(solver)
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        return self.zoning_solver._extract_assignment(solver, problem, x, y)

    def _add_market(self, model, problem, x, initial_assignment, initial_grid):
        return add_finite_grid_welfare_model(
            model,
            problem,
            x,
            initial_assignment,
            initial_grid,
            zoning_solver=self.zoning_solver,
            utility_scale=self.utility_scale,
        )


def add_finite_grid_welfare_model(
    model,
    problem,
    x,
    initial_assignment,
    initial_grid,
    *,
    zoning_solver,
    utility_scale,
):
    """Add the exact finite-grid assignment recurrence and capacity rows."""
    market = problem.cutoff_market
    same_zone = zoning_solver._add_vertex_school_same_zone_indicators(
        model, problem, x, market
    )
    same_zone_hints = {}
    for (node, school), variable in same_zone.items():
        hint = int(
            initial_assignment[node] == initial_assignment[market.school_nodes[school]]
        )
        model.AddHint(variable, hint)
        same_zone_hints[node, school] = hint
    return add_finite_grid_recurrence(
        model,
        market,
        same_zone,
        utility_scale=utility_scale,
        cutoff_hints=initial_grid.cutoffs.school_cutoffs,
        same_zone_hints=same_zone_hints,
    )


def add_finite_grid_recurrence(
    model,
    market,
    same_zone,
    *,
    utility_scale,
    cutoff_hints=None,
    same_zone_hints=None,
):
    """Add exact finite-grid welfare for supplied student-school membership."""
    scale = market.lottery_scale
    priorities = [
        priority
        for student in market.students
        for priority in student.priorities.values()
    ]
    max_priority = max(priorities, default=0)
    max_cutoff = (max_priority + 1) * scale
    cutoffs = {
        school: model.NewIntVar(0, max_cutoff, f"welfare_cutoff_{school}")
        for school in market.school_capacities
    }
    if cutoff_hints is not None:
        for school, variable in cutoffs.items():
            model.AddHint(variable, cutoff_hints[school])
    shared_thresholds = {}
    school_demands = {school: [] for school in market.school_capacities}
    objective_terms = []

    for student_index, student in enumerate(market.students):
        previous = scale
        previous_hint = scale
        for rank, school in enumerate(student.preferences, start=1):
            if school not in student.utilities:
                raise ValueError(
                    f"Student {student.studentno} lacks utility for school {school}."
                )
            priority = student.priorities[school]
            threshold_key = (school, priority)
            if threshold_key not in shared_thresholds:
                threshold = model.NewIntVar(
                    0, max_cutoff, f"welfare_threshold_{school}_{priority}"
                )
                model.AddMaxEquality(
                    threshold,
                    cutoffs[school] - priority * scale,
                    0,
                )
                if cutoff_hints is not None:
                    model.AddHint(
                        threshold,
                        max(0, cutoff_hints[school] - priority * scale),
                    )
                shared_thresholds[threshold_key] = threshold
            threshold = shared_thresholds[threshold_key]

            effective = model.NewIntVar(
                0,
                max_cutoff,
                f"welfare_effective_{student_index}_{rank}",
            )
            together = same_zone[(student.node, school)]
            model.Add(effective == threshold).OnlyEnforceIf(together)
            model.Add(effective == scale).OnlyEnforceIf(together.Not())
            cumulative = model.NewIntVar(
                0, scale, f"welfare_remaining_{student_index}_{rank}"
            )
            model.AddMinEquality(cumulative, [previous, effective])
            if cutoff_hints is not None and same_zone_hints is not None:
                initial_threshold = max(
                    0,
                    cutoff_hints[school] - priority * scale,
                )
                initial_effective = (
                    initial_threshold
                    if same_zone_hints[student.node, school]
                    else scale
                )
                previous_hint = min(previous_hint, initial_effective)
                model.AddHint(effective, initial_effective)
                model.AddHint(cumulative, previous_hint)
            assignment_mass = previous - cumulative
            school_demands[school].append(assignment_mass)
            coefficient = round(student.utilities[school] * utility_scale)
            if coefficient:
                objective_terms.append(coefficient * assignment_mass)
            previous = cumulative

    for school, capacity in market.school_capacities.items():
        model.Add(sum(school_demands[school]) <= scale * capacity)
    return cutoffs, sum(objective_terms)
