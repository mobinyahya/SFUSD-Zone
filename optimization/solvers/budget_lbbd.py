"""Logic-based Benders decomposition over priority-contingent budget sets."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass

from ortools.sat.python import cp_model

from optimization.solution import ZoneSolution
from optimization.solvers.cutoff_decomposition import _candidate_demands
from optimization.solvers.welfare import WelfareSolver
from optimization.welfare_oracle import (
    WelfareResult,
    outward_true_welfare_upper_bound,
    raw_welfare_upper_bound,
    solve_zoned_continuum_welfare,
    solve_zoned_welfare,
    validate_welfare_market,
)


@dataclass(frozen=True)
class _UtilityProfile:
    index: int
    node: int
    entries: tuple[tuple[int, int, int], ...]
    mass: int


@dataclass(frozen=True)
class _BudgetMaster:
    model: cp_model.CpModel
    x: dict
    y: dict
    cutoffs: dict
    qualifications: dict
    budgets: dict
    theta: dict
    profiles: tuple[_UtilityProfile, ...]
    objective: object


@dataclass(frozen=True)
class _Incumbent:
    assignment: dict[int, int]
    result: WelfareResult


def submodular_supergradient(
    coefficients: tuple[int, ...],
    reference_mask: int,
    kind: str,
) -> tuple[int, tuple[int, ...]]:
    """Return ``constant, coefficients`` for a max-utility upper cut."""
    if kind not in {"addition", "removal"}:
        raise ValueError("kind must be 'addition' or 'removal'.")
    size = len(coefficients)
    reference = [index for index in range(size) if reference_mask & (1 << index)]
    reference_value = max((coefficients[index] for index in reference), default=0)
    constant = reference_value
    slopes = [0] * size

    for index, coefficient in enumerate(coefficients):
        if index not in reference:
            slopes[index] = (
                max(0, coefficient - reference_value)
                if kind == "addition"
                else coefficient
            )
            continue
        base_indices = (
            [other for other in range(size) if other != index]
            if kind == "addition"
            else [other for other in reference if other != index]
        )
        base_value = max(
            (coefficients[other] for other in base_indices),
            default=0,
        )
        marginal = max(0, coefficient - base_value)
        constant -= marginal
        slopes[index] = marginal
    return constant, tuple(slopes)


class BudgetSetLbbdSolver(WelfareSolver):
    """Exact finite-grid LBBD with submodular utility and demand cuts."""

    optimization_method = "priority_contingent_budget_set_lbbd"
    finite_grid_formulation = "optimistic_budget_master"

    def __init__(self, zoning_solver, *, utility_scale: int) -> None:
        super().__init__(zoning_solver, utility_scale=utility_scale)
        self._utility_cut_signatures = set()
        self._capacity_cut_signatures = set()
        self._complete_schools = set()
        self._overload_counts = Counter()
        self._utility_cut_count = 0
        self._interval_capacity_cut_count = 0
        self._complete_demand_boolean_count = 0

    def solve(self, problem) -> ZoneSolution:
        market = problem.cutoff_market
        if market is None:
            raise ValueError("Budget-set LBBD requires a cutoff market.")
        unrestricted = set(market.school_capacities) - set(
            market.zone_restricted_schools
        )
        if unrestricted:
            raise ValueError(
                "budget-set LBBD requires isolated markets; remove city-wide "
                f"schools before solving: {sorted(unrestricted)}."
            )
        if self.utility_scale <= 0:
            raise ValueError("welfare_utility_scale must be a positive integer.")
        validate_welfare_market(market, utility_scale=self.utility_scale)
        self._reset_cuts()

        started = time.monotonic()
        time_limit = float(self.options.get("solve_time_limit", 60.0))
        deadline = started + time_limit
        initial_assignment = (
            dict(problem.hint)
            if problem.hint is not None
            else self._initial_assignment(
                problem,
                min(30.0, max(1.0, time_limit * 0.15)),
            )
        )
        if initial_assignment is None:
            raise RuntimeError("Could not construct an initial feasible LBBD zoning.")
        initial_result = solve_zoned_welfare(
            market,
            initial_assignment,
            num_zones=problem.Z,
            utility_scale=self.utility_scale,
        )
        incumbent = _Incumbent(initial_assignment, initial_result)

        global_upper_bound = raw_welfare_upper_bound(market, self.utility_scale)
        configured_upper_bound = self.options.get("welfare_raw_upper_bound")
        if configured_upper_bound is not None:
            if (
                isinstance(configured_upper_bound, bool)
                or not isinstance(configured_upper_bound, int)
                or configured_upper_bound < initial_result.raw_scaled_welfare
            ):
                raise ValueError(
                    "welfare_raw_upper_bound must be an integer no smaller than "
                    "the initial incumbent."
                )
            global_upper_bound = min(global_upper_bound, configured_upper_bound)

        master = self._build_master(problem, incumbent)
        master.model.Add(master.objective <= global_upper_bound)
        master.model.Add(master.objective > incumbent.result.raw_scaled_welfare)
        master.model.Maximize(master.objective)
        initial_utility_cuts = self._add_utility_cuts_for_state(
            master,
            problem,
            incumbent.assignment,
            incumbent.result.cutoffs.school_cutoffs,
        )
        pre_solve_wall_time = time.monotonic() - started

        rounds = []
        best_upper_bound = global_upper_bound
        certified = False
        termination = "time_limit"
        round_index = 0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            solver = cp_model.CpSolver()
            self.zoning_solver._configure_solver_parameters(solver)
            solver.parameters.max_time_in_seconds = max(0.01, min(180.0, remaining))
            solver.parameters.random_seed = int(self.options.get("seed", 42)) + round_index
            status = solver.Solve(master.model)
            status_name = solver.StatusName(status)
            row = {
                "round": round_index,
                "status": status_name,
                "master_objective": None,
                "master_upper_bound": None,
                "candidate_welfare": None,
                "oracle_incumbent": incumbent.result.raw_scaled_welfare,
                "overloaded_schools": 0,
                "utility_cuts_added": 0,
                "interval_capacity_cuts_added": 0,
                "complete_schools_activated": 0,
            }
            if status == cp_model.INFEASIBLE:
                certified = configured_upper_bound is None
                termination = "no_better_master_state"
                best_upper_bound = incumbent.result.raw_scaled_welfare
                rounds.append(row)
                break
            if status == cp_model.UNKNOWN:
                rounds.append(row)
                round_index += 1
                continue
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                termination = status_name.lower()
                rounds.append(row)
                break

            master_objective = int(round(solver.ObjectiveValue()))
            round_upper_bound = int(
                math.floor(solver.BestObjectiveBound() + 1e-6)
            )
            best_upper_bound = min(best_upper_bound, round_upper_bound)
            assignment = self.zoning_solver._extract_assignment(
                solver, problem, master.x, master.y
            )
            candidate_cutoffs = {
                school: int(solver.Value(variable))
                for school, variable in master.cutoffs.items()
            }
            candidate_welfare = self._state_welfare(
                master.profiles,
                market,
                assignment,
                candidate_cutoffs,
            )
            utility_cuts = self._add_utility_cuts_for_state(
                master,
                problem,
                assignment,
                candidate_cutoffs,
            )

            demands, intervals = _candidate_demands(
                problem, market, assignment, candidate_cutoffs
            )
            overloaded = [
                school
                for school, demand in demands.items()
                if demand > market.school_capacities[school] * market.lottery_scale
            ]
            interval_cuts = 0
            activations = 0
            for school in overloaded:
                if school in self._complete_schools:
                    raise RuntimeError(
                        f"Complete demand module failed to enforce capacity at {school}."
                    )
                self._overload_counts[school] += 1
                if self._overload_counts[school] == 1:
                    interval_cuts += self._add_interval_capacity_cut(
                        master,
                        market,
                        school,
                        intervals[school],
                    )
                else:
                    activations += self._activate_complete_school(
                        master, market, school
                    )

            oracle = solve_zoned_welfare(
                market,
                assignment,
                num_zones=problem.Z,
                utility_scale=self.utility_scale,
            )
            utility_cuts += self._add_utility_cuts_for_state(
                master,
                problem,
                assignment,
                oracle.cutoffs.school_cutoffs,
            )
            if oracle.raw_scaled_welfare > incumbent.result.raw_scaled_welfare:
                incumbent = _Incumbent(assignment, oracle)
                master.model.Add(master.objective > oracle.raw_scaled_welfare)

            row.update(
                {
                    "master_objective": master_objective,
                    "master_upper_bound": round_upper_bound,
                    "candidate_welfare": candidate_welfare,
                    "oracle_incumbent": incumbent.result.raw_scaled_welfare,
                    "overloaded_schools": len(overloaded),
                    "utility_cuts_added": utility_cuts,
                    "interval_capacity_cuts_added": interval_cuts,
                    "complete_schools_activated": activations,
                }
            )
            rounds.append(row)

            if (
                configured_upper_bound is None
                and round_upper_bound <= incumbent.result.raw_scaled_welfare
            ):
                certified = True
                termination = "upper_bound_reached_incumbent"
                best_upper_bound = incumbent.result.raw_scaled_welfare
                break
            if (
                configured_upper_bound is None
                and status == cp_model.OPTIMAL
                and not overloaded
                and master_objective <= incumbent.result.raw_scaled_welfare
            ):
                certified = True
                termination = "master_candidate_exact_and_feasible"
                best_upper_bound = incumbent.result.raw_scaled_welfare
                break
            if utility_cuts + interval_cuts + activations == 0:
                termination = "no_genuinely_new_cut"
                break
            round_index += 1

        return self._solution(
            problem,
            incumbent,
            started,
            certified,
            max(
                incumbent.result.raw_scaled_welfare,
                min(best_upper_bound, global_upper_bound),
            ),
            configured_upper_bound,
            termination,
            rounds,
            master,
            initial_utility_cuts,
            pre_solve_wall_time,
        )

    def _reset_cuts(self) -> None:
        self._utility_cut_signatures = set()
        self._capacity_cut_signatures = set()
        self._complete_schools = set()
        self._overload_counts = Counter()
        self._utility_cut_count = 0
        self._interval_capacity_cut_count = 0
        self._complete_demand_boolean_count = 0

    def _build_master(self, problem, incumbent) -> _BudgetMaster:
        market = problem.cutoff_market
        scale = market.lottery_scale
        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        self.zoning_solver._add_search_strategy(model, problem, x, y)
        for (zone, node), variable in x.items():
            model.AddHint(variable, int(incumbent.assignment[node] == zone))
        for node, variable in y.items():
            model.AddHint(variable, incumbent.assignment[node])

        same_zone = self.zoning_solver._add_vertex_school_same_zone_indicators(
            model, problem, x, market
        )
        same_zone_hints = {}
        for (node, school), variable in same_zone.items():
            hint = int(
                incumbent.assignment[node]
                == incumbent.assignment[market.school_nodes[school]]
            )
            model.AddHint(variable, hint)
            same_zone_hints[node, school] = hint

        cutoff_domains = {school: {0} for school in market.school_capacities}
        for student in market.students:
            for school in student.preferences:
                priority = student.priorities[school]
                cutoff_domains[school].update(
                    priority * scale + cell for cell in range(1, scale + 1)
                )
        cutoff_hints = {
            school: max(
                value
                for value in cutoff_domains[school]
                if value <= incumbent.result.cutoffs.school_cutoffs[school]
            )
            for school in market.school_capacities
        }
        cutoffs = {
            school: model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(sorted(cutoff_domains[school])),
                f"lbbd_cutoff_{school}",
            )
            for school in market.school_capacities
        }
        for school, variable in cutoffs.items():
            model.AddHint(variable, cutoff_hints[school])

        profile_counts = Counter()
        for student in market.students:
            entries = tuple(
                (
                    school,
                    student.priorities[school],
                    round(student.utilities[school] * self.utility_scale),
                )
                for school in student.preferences
            )
            if entries:
                profile_counts[student.node, entries] += 1
        profiles = tuple(
            _UtilityProfile(index, node, entries, mass)
            for index, ((node, entries), mass) in enumerate(
                sorted(profile_counts.items())
            )
        )

        qualifications = {}
        budgets = {}

        def qualification(school, priority, cell):
            key = (school, priority, cell)
            variable = qualifications.get(key)
            if variable is not None:
                return variable
            score_limit = priority * scale + cell - 1
            variable = model.NewBoolVar(
                f"lbbd_qualifies_{school}_{priority}_{cell}"
            )
            model.Add(cutoffs[school] <= score_limit).OnlyEnforceIf(variable)
            model.Add(cutoffs[school] > score_limit).OnlyEnforceIf(variable.Not())
            model.AddHint(variable, int(cutoff_hints[school] <= score_limit))
            qualifications[key] = variable
            return variable

        for profile in profiles:
            for school, priority, _coefficient in profile.entries:
                access = same_zone[profile.node, school]
                for cell in range(1, scale + 1):
                    key = (profile.node, school, priority, cell)
                    if key in budgets:
                        continue
                    qualifies = qualification(school, priority, cell)
                    budget = model.NewBoolVar(
                        f"lbbd_budget_{profile.node}_{school}_{priority}_{cell}"
                    )
                    model.AddBoolAnd([access, qualifies]).OnlyEnforceIf(budget)
                    model.AddBoolOr([budget, access.Not(), qualifies.Not()])
                    model.AddHint(
                        budget,
                        int(
                            same_zone_hints[profile.node, school]
                            and cutoff_hints[school]
                            <= priority * scale + cell - 1
                        ),
                    )
                    budgets[key] = budget

        theta = {}
        objective_terms = []
        for profile in profiles:
            upper = max(coefficient for _school, _priority, coefficient in profile.entries)
            for cell in range(1, scale + 1):
                variable = model.NewIntVar(
                    0,
                    upper,
                    f"lbbd_theta_{profile.index}_{cell}",
                )
                hint = max(
                    (
                        coefficient
                        for school, priority, coefficient in profile.entries
                        if same_zone_hints[profile.node, school]
                        and cutoff_hints[school] <= priority * scale + cell - 1
                    ),
                    default=0,
                )
                model.AddHint(variable, hint)
                theta[profile.index, cell] = variable
                objective_terms.append(profile.mass * variable)

        return _BudgetMaster(
            model=model,
            x=x,
            y=y,
            cutoffs=cutoffs,
            qualifications=qualifications,
            budgets=budgets,
            theta=theta,
            profiles=profiles,
            objective=sum(objective_terms),
        )

    def _add_utility_cuts_for_state(
        self,
        master,
        problem,
        assignment,
        cutoffs,
    ) -> int:
        market = problem.cutoff_market
        scale = market.lottery_scale
        added = 0
        for profile in master.profiles:
            coefficients = tuple(entry[2] for entry in profile.entries)
            for cell in range(1, scale + 1):
                reference_mask = 0
                for index, (school, priority, _coefficient) in enumerate(
                    profile.entries
                ):
                    if (
                        assignment[profile.node]
                        == assignment[market.school_nodes[school]]
                        and cutoffs[school] <= priority * scale + cell - 1
                    ):
                        reference_mask |= 1 << index
                variables = [
                    master.budgets[profile.node, school, priority, cell]
                    for school, priority, _coefficient in profile.entries
                ]
                for kind in ("addition", "removal"):
                    constant, slopes = submodular_supergradient(
                        coefficients, reference_mask, kind
                    )
                    signature = (
                        profile.index,
                        cell,
                        constant,
                        slopes,
                    )
                    if signature in self._utility_cut_signatures:
                        continue
                    self._utility_cut_signatures.add(signature)
                    master.model.Add(
                        master.theta[profile.index, cell]
                        <= constant
                        + sum(
                            slope * variable
                            for slope, variable in zip(slopes, variables, strict=True)
                            if slope
                        )
                    )
                    self._utility_cut_count += 1
                    added += 1
        return added

    @staticmethod
    def _state_welfare(profiles, market, assignment, cutoffs) -> int:
        scale = market.lottery_scale
        total = 0
        for profile in profiles:
            for cell in range(1, scale + 1):
                value = max(
                    (
                        coefficient
                        for school, priority, coefficient in profile.entries
                        if assignment[profile.node]
                        == assignment[market.school_nodes[school]]
                        and cutoffs[school] <= priority * scale + cell - 1
                    ),
                    default=0,
                )
                total += profile.mass * value
        return total

    def _add_interval_capacity_cut(self, master, market, school, intervals) -> int:
        profiles = Counter()
        for interval in intervals:
            student = interval.student
            blockers = tuple(
                (
                    preferred,
                    student.priorities[preferred],
                    interval.high,
                )
                for preferred in interval.higher
            )
            key = (
                student.node,
                student.priorities[school],
                interval.low,
                blockers,
            )
            profiles[key] += interval.high - interval.low + 1
        signature = (school, tuple(sorted(profiles.items())))
        if signature in self._capacity_cut_signatures:
            return 0
        self._capacity_cut_signatures.add(signature)

        terms = []
        for profile_index, (
            (node, priority, low, blockers),
            weight,
        ) in enumerate(sorted(profiles.items())):
            target = master.budgets[node, school, priority, low]
            blocker_vars = [
                master.budgets[node, preferred, preferred_priority, high]
                for preferred, preferred_priority, high in blockers
            ]
            persists = master.model.NewBoolVar(
                f"lbbd_interval_{school}_{self._interval_capacity_cut_count}_{profile_index}"
            )
            master.model.AddBoolOr(
                [persists, target.Not(), *blocker_vars]
            )
            terms.append(weight * persists)
        master.model.Add(sum(terms) <= market.lottery_scale * market.school_capacities[school])
        self._interval_capacity_cut_count += 1
        return 1

    def _activate_complete_school(self, master, market, school) -> int:
        if school in self._complete_schools:
            return 0
        self._complete_schools.add(school)
        scale = market.lottery_scale
        demand_profiles = Counter(
            (
                student.node,
                tuple(
                    (listed, student.priorities[listed])
                    for listed in student.preferences
                ),
            )
            for student in market.students
            if school in student.preferences
        )
        terms = []
        for profile_index, ((node, entries), mass) in enumerate(
            sorted(demand_profiles.items())
        ):
            target_rank = next(
                index for index, (listed, _priority) in enumerate(entries)
                if listed == school
            )
            target_priority = entries[target_rank][1]
            for cell in range(1, scale + 1):
                target = master.budgets[node, school, target_priority, cell]
                blockers = [
                    master.budgets[node, preferred, priority, cell]
                    for preferred, priority in entries[:target_rank]
                ]
                selected = master.model.NewBoolVar(
                    f"lbbd_complete_{school}_{profile_index}_{cell}"
                )
                master.model.AddBoolAnd(
                    [target, *(blocker.Not() for blocker in blockers)]
                ).OnlyEnforceIf(selected)
                master.model.AddBoolOr(
                    [selected, target.Not(), *blockers]
                )
                terms.append(mass * selected)
                self._complete_demand_boolean_count += 1
        master.model.Add(
            sum(terms) <= scale * market.school_capacities[school]
        )
        return 1

    def _solution(
        self,
        problem,
        incumbent,
        started,
        certified,
        raw_upper_bound,
        configured_upper_bound,
        termination,
        rounds,
        master,
        initial_utility_cuts,
        pre_solve_wall_time,
    ) -> ZoneSolution:
        market = problem.cutoff_market
        wall_time = time.monotonic() - started
        grid = incumbent.result
        continuum = solve_zoned_continuum_welfare(
            market,
            incumbent.assignment,
            num_zones=problem.Z,
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
        normalizer = market.lottery_scale * self.utility_scale
        true_upper_bound = outward_true_welfare_upper_bound(
            raw_upper_bound, market, self.utility_scale
        )
        metadata = {
            **market.metadata,
            "solver": "cp_bool",
            "objective_kind": "stable_assignment_welfare",
            "optimization_method": self.optimization_method,
            "finite_grid_formulation": self.finite_grid_formulation,
            "market_coupling": "isolated_zones",
            "lottery_scale": market.lottery_scale,
            "welfare_utility_scale": self.utility_scale,
            "welfare": grid.welfare,
            "rounded_welfare": grid.raw_scaled_welfare / normalizer,
            "raw_scaled_welfare": grid.raw_scaled_welfare,
            "raw_scaled_upper_bound": raw_upper_bound,
            "configured_raw_upper_bound": configured_upper_bound,
            "rounded_welfare_upper_bound": raw_upper_bound / normalizer,
            "utility_rounding_error_bound": math.nextafter(
                len(market.students) / (2 * self.utility_scale), math.inf
            ),
            "true_welfare_upper_bound": true_upper_bound,
            "true_welfare_gap_bound": max(0.0, true_upper_bound - grid.welfare),
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
            "termination": termination,
            "decomposition_rounds": rounds,
            "initial_utility_cuts": initial_utility_cuts,
            "submodular_utility_cut_count": self._utility_cut_count,
            "interval_capacity_cut_count": self._interval_capacity_cut_count,
            "complete_demand_schools": sorted(self._complete_schools),
            "complete_demand_boolean_count": self._complete_demand_boolean_count,
            "budget_profile_count": len(master.profiles),
            "qualification_boolean_count": len(master.qualifications),
            "budget_boolean_count": len(master.budgets),
            "cell_utility_variable_count": len(master.theta),
            "pre_solve_wall_time": pre_solve_wall_time,
            "model_stats": master.model.ModelStats(),
        }
        return ZoneSolution(
            problem=problem,
            assignment=incumbent.assignment,
            status="OPTIMAL" if certified else "FEASIBLE",
            objective=grid.welfare,
            wall_time=wall_time,
            metadata=metadata,
        )
