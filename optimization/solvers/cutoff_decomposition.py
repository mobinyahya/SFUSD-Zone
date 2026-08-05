"""Exact oracle-separated CP-SAT solver for cutoff zoning.

The master contains the geographic zoning variables and one cutoff per school.
School-capacity constraints are separated as revealed-preference interval cuts.
Each cut is valid for every zoning and cutoff vector, so a master lower bound
that reaches an oracle-feasible incumbent certifies global optimality.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from ortools.sat.python import cp_model

from optimization.cutoff_oracle import (
    CoupledCutoffResult,
    ZonedCutoffResult,
    solve_coupled_continuum_cutoffs,
    solve_coupled_cutoffs,
    solve_zoned_continuum_cutoffs,
    solve_zoned_cutoffs,
)
from optimization.problem import CutoffMarket, CutoffStudent, ZoneProblem
from optimization.solution import ZoneSolution


@dataclass(frozen=True)
class _Incumbent:
    assignment: dict[int, int]
    result: ZonedCutoffResult | CoupledCutoffResult


@dataclass(frozen=True)
class _DemandInterval:
    student: CutoffStudent
    low: int
    high: int
    higher: tuple[int, ...]


class CutoffDecompositionSolver:
    """Solve a cutoff problem by exact finite constraint generation."""

    def __init__(self, zoning_solver) -> None:
        self.zoning_solver = zoning_solver
        self.options = zoning_solver.options
        self._reset_cut_state()

    def _reset_cut_state(self) -> None:
        self._comparison_vars = {}
        self._zone_access_vars = {}
        self._affordability_vars = {}
        self._blocking_vars = {}
        self._demand_vars = {}
        self._capacity_cut_signatures = set()
        self._cut_count = 0
        self._cut_profile_count = 0

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        self._reset_cut_state()
        market = problem.cutoff_market
        if market is None:
            raise ValueError("Cutoff decomposition requires a cutoff market.")
        unrestricted = set(market.school_capacities) - set(
            market.zone_restricted_schools
        )
        coupled_market = bool(unrestricted)

        started = time.monotonic()
        time_limit = float(self.options.get("solve_time_limit", 60.0))
        deadline = started + time_limit
        incumbent, heuristic_rows = self._initial_incumbent(problem, deadline)
        local_deadline = min(deadline, time.monotonic() + min(45.0, time_limit * 0.4))
        local_starts = getattr(self, "_local_starts", [incumbent])
        for index, start in enumerate(local_starts):
            if time.monotonic() >= local_deadline:
                break
            starts_left = len(local_starts) - index
            remaining_local = local_deadline - time.monotonic()
            allocation = (
                remaining_local * 0.55 if index == 0 else remaining_local / starts_left
            )
            slot_deadline = min(
                local_deadline,
                time.monotonic() + allocation,
            )
            local, local_rows = self._local_improve(problem, start, slot_deadline)
            heuristic_rows.extend(local_rows)
            if local.result.objective < incumbent.result.objective:
                incumbent = local

        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        max_priority = max(
            (
                priority
                for student in market.students
                for priority in student.priorities.values()
            ),
            default=0,
        )
        max_cutoff = (max_priority + 1) * market.lottery_scale
        cutoffs = {
            school: model.NewIntVar(0, max_cutoff, f"master_cutoff_{school}")
            for school in market.school_capacities
        }
        cutoff_total = sum(cutoffs.values())
        model.Minimize(cutoff_total)
        model.Add(cutoff_total < incumbent.result.objective)

        rounds = []
        best_bound = 0
        certified = False
        termination = "time_limit"
        round_index = 0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            solver = self._new_solver(min(30.0, remaining), round_index)
            model.ClearHints()
            self._add_incumbent_hints(model, problem, x, cutoffs, incumbent)
            round_started = time.monotonic()
            status = solver.Solve(model)
            round_seconds = time.monotonic() - round_started
            status_name = solver.StatusName(status)
            row = {
                "round": round_index,
                "status": status_name,
                "wall_time": round_seconds,
                "objective": None,
                "best_bound": None,
                "overloaded_schools": 0,
                "cuts_added": 0,
            }

            if status == cp_model.INFEASIBLE:
                certified = True
                termination = "no_better_solution"
                best_bound = incumbent.result.objective
                rounds.append(row)
                break
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                termination = status_name.lower()
                rounds.append(row)
                break

            master_objective = int(round(solver.ObjectiveValue()))
            round_bound = int(math.ceil(solver.BestObjectiveBound() - 1e-9))
            best_bound = max(best_bound, round_bound)
            assignment = self._assignment_from(solver, problem, x)
            candidate_cutoffs = {
                school: int(solver.Value(var)) for school, var in cutoffs.items()
            }
            demands, intervals = _candidate_demands(
                problem, market, assignment, candidate_cutoffs
            )
            overloaded = [
                school
                for school, demand in demands.items()
                if demand > market.school_capacities[school] * market.lottery_scale
            ]

            candidate_oracle = _solve_cutoffs(problem, assignment)
            if candidate_oracle.objective < incumbent.result.objective:
                incumbent = _Incumbent(assignment, candidate_oracle)
                model.Add(cutoff_total < incumbent.result.objective)

            cuts_added = 0
            for school in overloaded:
                cuts_added += self._add_interval_capacity_cut(
                    model,
                    problem,
                    market,
                    x,
                    cutoffs,
                    school,
                    intervals[school],
                    max_cutoff,
                )

            row.update(
                {
                    "objective": master_objective,
                    "best_bound": round_bound,
                    "overloaded_schools": len(overloaded),
                    "cuts_added": cuts_added,
                    "oracle_incumbent": incumbent.result.objective,
                }
            )
            rounds.append(row)

            if not overloaded:
                # The candidate is feasible for the original recurrence.  An
                # optimal master candidate therefore closes the global gap.
                if status == cp_model.OPTIMAL:
                    certified = True
                    termination = "master_candidate_feasible"
                    best_bound = master_objective
                    break
            if best_bound >= incumbent.result.objective:
                certified = True
                termination = "lower_bound_reached_incumbent"
                break
            if cuts_added == 0:
                termination = "no_separating_cut"
                break
            round_index += 1

        wall_time = time.monotonic() - started
        raw_cutoffs = incumbent.result.school_cutoffs
        continuum = _solve_continuum_cutoffs(problem, incumbent.assignment)
        grid_demands = _grid_demands(incumbent.result)
        positive_grid_underfill = {
            school: market.school_capacities[school] * market.lottery_scale
            - grid_demands[school]
            for school, cutoff in raw_cutoffs.items()
            if cutoff > 0
        }
        if coupled_market:
            zone_stable = continuum.zone_stable
            zone_stability_checks = continuum.zone_checks
        else:
            zone_stable = {
                zone: result.stable for zone, result in continuum.zones.items()
            }
            zone_stability_checks = {
                zone: {"isolated_market_clears": stable}
                for zone, stable in zone_stable.items()
            }
        serialized_zone_stable = {
            str(zone): stable for zone, stable in zone_stable.items()
        }
        serialized_zone_checks = {
            str(zone): checks for zone, checks in zone_stability_checks.items()
        }
        metadata = {
            **market.metadata,
            "solver": "cp_bool",
            "objective_kind": "school_cutoffs",
            "optimization_method": "exact_revealed_preference_decomposition",
            "lottery_scale": market.lottery_scale,
            "school_cutoffs": raw_cutoffs,
            "normalized_school_cutoffs": {
                school: cutoff / market.lottery_scale
                for school, cutoff in raw_cutoffs.items()
            },
            "grid_minimal": incumbent.result.grid_minimal,
            "grid_max_positive_cutoff_underfill_mass": max(
                positive_grid_underfill.values(), default=0
            ),
            "stable": continuum.stable,
            "zone_stable": serialized_zone_stable,
            "zone_stability_checks": serialized_zone_checks,
            "stable_zone_count": sum(zone_stable.values()),
            "market_coupling": (
                "global_citywide_access" if coupled_market else "isolated_zones"
            ),
            "unrestricted_school_count": len(unrestricted),
            "unrestricted_schools": sorted(unrestricted),
            "stability_definition": (
                "Global market stability with shared citywide capacity; zone checks "
                "use the common citywide cutoffs."
                if coupled_market
                else "Independent continuous market clearing in every zone."
            ),
            "continuum_objective": continuum.objective,
            "continuum_school_cutoffs": continuum.school_cutoffs,
            "global_optimum_certified": certified,
            "raw_objective": incumbent.result.objective,
            "raw_best_bound": min(best_bound, incumbent.result.objective),
            "normalized_best_bound": min(best_bound, incumbent.result.objective)
            / market.lottery_scale,
            "termination": termination,
            "decomposition_rounds": rounds,
            "revealed_preference_cut_count": self._cut_count,
            "revealed_preference_profile_count": self._cut_profile_count,
            "heuristic_candidates": heuristic_rows,
        }
        return ZoneSolution(
            problem=problem,
            assignment=incumbent.assignment,
            status="OPTIMAL" if certified else "FEASIBLE",
            objective=incumbent.result.normalized_objective,
            wall_time=wall_time,
            metadata=metadata,
        )

    def _initial_incumbent(
        self, problem: ZoneProblem, deadline: float
    ) -> tuple[_Incumbent, list[dict]]:
        market = problem.cutoff_market
        student_counts = Counter(
            student.node for student in market.students if student.preferences
        )
        capacity_counts = Counter()
        for school, capacity in market.school_capacities.items():
            capacity_counts[market.school_nodes[school]] += capacity
        pressure_coeff = {
            node: student_counts[node] - capacity_counts[node] for node in problem.nodes
        }

        rows = []
        incumbents = []
        best_by_target = {}
        neutral_coefficients = {node: 0 for node in problem.nodes}
        assignment, status, elapsed = self._solve_pressure_model(
            problem,
            neutral_coefficients,
            (0,),
            min(15.0, max(0.05, deadline - time.monotonic())),
            -1,
        )
        if assignment is not None:
            result = _solve_cutoffs(problem, assignment)
            incumbents.append(_Incumbent(assignment, result))
            rows.append(
                {
                    "kind": "feasible_start",
                    "zones": [],
                    "status": status,
                    "wall_time": elapsed,
                    "raw_objective": result.objective,
                }
            )
        for target in range(problem.Z):
            if time.monotonic() >= deadline:
                break
            assignment, status, elapsed = self._solve_pressure_model(
                problem,
                pressure_coeff,
                (target,),
                min(3.0, max(0.05, deadline - time.monotonic())),
                target,
            )
            if assignment is None:
                continue
            result = _solve_cutoffs(problem, assignment)
            candidate = _Incumbent(assignment, result)
            incumbents.append(candidate)
            best_by_target[target] = candidate
            rows.append(
                {
                    "kind": "pressure",
                    "zones": [target],
                    "status": status,
                    "wall_time": elapsed,
                    "raw_objective": result.objective,
                }
            )
        if not incumbents:
            assignment, status, elapsed = self._solve_pressure_model(
                problem,
                neutral_coefficients,
                (0,),
                max(0.05, deadline - time.monotonic()),
                -2,
            )
            if assignment is not None:
                result = _solve_cutoffs(problem, assignment)
                incumbents.append(_Incumbent(assignment, result))
                rows.append(
                    {
                        "kind": "extended_feasible_start",
                        "zones": [],
                        "status": status,
                        "wall_time": elapsed,
                        "raw_objective": result.objective,
                    }
                )
        if not incumbents:
            raise RuntimeError("Could not construct an initial feasible cutoff zoning.")

        best = min(incumbents, key=lambda item: item.result.objective)
        if not best_by_target:
            self._local_starts = [best]
            return best, rows
        # Refine the most promising overloaded-zone family with deterministic
        # one-worker solves. This avoids arbitrary parallel tie choices among
        # pressure-optimal zonings with very different exact cutoff costs.
        refine_target = min(
            best_by_target,
            key=lambda zone: best_by_target[zone].result.objective,
        )
        refine_others = [zone for zone in range(problem.Z) if zone != refine_target]
        for secondary, tertiary in itertools.permutations(refine_others, 2):
            if time.monotonic() >= deadline:
                break
            assignment, status, elapsed = self._solve_pressure_model(
                problem,
                pressure_coeff,
                (refine_target, secondary, tertiary),
                min(3.0, max(0.05, deadline - time.monotonic())),
                500 + secondary * problem.Z + tertiary,
            )
            if assignment is None:
                continue
            result = _solve_cutoffs(problem, assignment)
            candidate = _Incumbent(assignment, result)
            incumbents.append(candidate)
            if result.objective < best_by_target[refine_target].result.objective:
                best_by_target[refine_target] = candidate
            if result.objective < best.result.objective:
                best = candidate
            rows.append(
                {
                    "kind": "refined_lexicographic_pressure",
                    "zones": [refine_target, secondary, tertiary],
                    "status": status,
                    "wall_time": elapsed,
                    "raw_objective": result.objective,
                }
            )
        local_starts = [best, *best_by_target.values()]
        unique = {}
        for candidate in local_starts:
            signature = tuple(candidate.assignment[node] for node in problem.nodes)
            unique.setdefault(signature, candidate)
        self._local_starts = list(unique.values())
        return best, rows

    def _local_improve(
        self,
        problem: ZoneProblem,
        incumbent: _Incumbent,
        deadline: float,
    ) -> tuple[_Incumbent, list[dict]]:
        """Use exact-oracle school swaps and support-closed boundary moves."""
        unrestricted = set(problem.cutoff_market.school_capacities) - set(
            problem.cutoff_market.zone_restricted_schools
        )
        if (
            unrestricted
            or self.zoning_solver._centroid_neighbor_radius() > 0
            or any(
                value >= 0
                for value in (
                    problem.frl_dev,
                    problem.racial_dev,
                    problem.overage,
                    problem.shortage,
                    problem.boundary_prop,
                )
            )
        ):
            return incumbent, []

        relation = problem.G.graph["closer_neighbors"]
        supports = {
            zone: {
                node: tuple(relation[node][problem.centroid_school_ids[zone]])
                for node in problem.nodes
            }
            for zone in range(problem.Z)
        }
        reverse = {
            zone: {node: [] for node in problem.nodes} for zone in range(problem.Z)
        }
        for zone in range(problem.Z):
            for node in problem.nodes:
                for supporter in supports[zone][node]:
                    reverse[zone][supporter].append(node)
        centroid_nodes = set(problem.centroids)
        total_schools = sum(problem.num_schools(node) for node in problem.nodes)
        average = total_schools / problem.Z
        school_lower = math.ceil(max(0.0, average - 1.0) - 1e-9)
        school_upper = math.floor(average + 1.0 + 1e-9)
        seen = {tuple(incumbent.assignment[node] for node in problem.nodes)}
        rows = []

        def closure(assignment, source, seed, max_size):
            moving = {seed}
            pending = [seed]
            if seed in centroid_nodes:
                return None
            while pending:
                removed = pending.pop()
                for dependent in reverse[source][removed]:
                    if assignment[dependent] != source or dependent in moving:
                        continue
                    if any(
                        assignment[supporter] == source and supporter not in moving
                        for supporter in supports[source][dependent]
                    ):
                        continue
                    if dependent in centroid_nodes:
                        return None
                    moving.add(dependent)
                    pending.append(dependent)
                    if len(moving) > max_size:
                        return None
            return moving

        def valid(assignment, changed):
            for zone, centroid in enumerate(problem.centroids):
                if assignment[centroid] != zone:
                    return False
            affected = set(changed)
            for node in changed:
                affected.update(problem.G.neighbors(node))
            for node in affected:
                zone = assignment[node]
                if zone not in problem.candidate_zones(node):
                    return False
                if node == problem.centroids[zone]:
                    continue
                if not any(
                    assignment[neighbor] == zone for neighbor in supports[zone][node]
                ):
                    return False
            counts = Counter()
            for node, zone in assignment.items():
                counts[zone] += problem.num_schools(node)
            return all(
                school_lower <= counts[zone] <= school_upper
                for zone in range(problem.Z)
            )

        def local_rank(candidate):
            costs = [
                candidate.result.zones[zone].objective for zone in range(problem.Z)
            ]
            return (
                candidate.result.objective,
                sum(costs) - max(costs, default=0),
                max(costs, default=0),
            )

        def evaluate(candidate, kind, changed, *, update=True):
            nonlocal incumbent
            signature = tuple(candidate[node] for node in problem.nodes)
            if signature in seen:
                return None
            seen.add(signature)
            result = _solve_cutoffs(problem, candidate)
            evaluated = _Incumbent(candidate, result)
            if update and local_rank(evaluated) < local_rank(incumbent):
                rows.append(
                    {
                        "kind": kind,
                        "changed_nodes": sorted(changed),
                        "raw_objective": result.objective,
                    }
                )
                incumbent = evaluated
            return evaluated

        school_nodes = [node for node in problem.nodes if problem.num_schools(node) > 0]
        base = dict(incumbent.assignment)
        for left_index, left in enumerate(school_nodes):
            if time.monotonic() >= deadline:
                break
            for right in school_nodes[left_index + 1 :]:
                if time.monotonic() >= deadline:
                    break
                left_zone = base[left]
                right_zone = base[right]
                if left_zone == right_zone:
                    continue
                left_nodes = closure(base, left_zone, left, 24)
                right_nodes = closure(base, right_zone, right, 24)
                if not left_nodes or not right_nodes or left_nodes & right_nodes:
                    continue
                candidate = dict(base)
                for node in left_nodes:
                    candidate[node] = right_zone
                for node in right_nodes:
                    candidate[node] = left_zone
                changed = left_nodes | right_nodes
                if valid(candidate, changed):
                    evaluate(candidate, "school_swap", changed)

        # Greedy support-closed transfers pick up small applicant improvements
        # after the best school composition change.
        while time.monotonic() < deadline:
            base = dict(incumbent.assignment)
            before = local_rank(incumbent)
            pass_best = incumbent
            pass_move = None
            for node in problem.nodes:
                if time.monotonic() >= deadline:
                    break
                source = base[node]
                targets = {base[neighbor] for neighbor in problem.G.neighbors(node)} - {
                    source
                }
                moving = closure(base, source, node, 16)
                if not moving:
                    continue
                for target in targets:
                    candidate = dict(base)
                    for member in moving:
                        candidate[member] = target
                    if valid(candidate, moving):
                        evaluated = evaluate(
                            candidate,
                            "boundary_transfer",
                            moving,
                            update=False,
                        )
                        if evaluated is not None and local_rank(evaluated) < local_rank(
                            pass_best
                        ):
                            pass_best = evaluated
                            pass_move = sorted(moving)
            if local_rank(pass_best) >= before:
                break
            incumbent = pass_best
            rows.append(
                {
                    "kind": "boundary_transfer",
                    "changed_nodes": pass_move,
                    "raw_objective": incumbent.result.objective,
                }
            )
        return incumbent, rows

    def _solve_pressure_model(
        self,
        problem: ZoneProblem,
        coefficients: dict[int, int],
        zones: tuple[int, ...],
        time_limit: float,
        seed_offset: int,
    ) -> tuple[dict[int, int] | None, str, float]:
        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        self.zoning_solver._add_hints(model, problem, x, y)
        self.zoning_solver._add_search_strategy(model, problem, x, y)
        expressions = [
            sum(
                coefficients[node] * x[(zone, node)]
                for node in problem.nodes
                if (zone, node) in x
            )
            for zone in zones
        ]
        if len(expressions) == 1:
            objective = expressions[0]
        else:
            weight = (
                len(problem.cutoff_market.students)
                + sum(problem.cutoff_market.school_capacities.values())
                + 1
            )
            objective = sum(
                expression * weight ** (len(expressions) - index - 1)
                for index, expression in enumerate(expressions)
            )
        model.Maximize(objective)
        solver = self._new_solver(time_limit, 0)
        solver.parameters.num_search_workers = 1
        started = time.monotonic()
        status = solver.Solve(model)
        elapsed = time.monotonic() - started
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, solver.StatusName(status), elapsed
        return (
            self._assignment_from(solver, problem, x),
            solver.StatusName(status),
            elapsed,
        )

    def _new_solver(self, time_limit: float, seed_offset: int) -> cp_model.CpSolver:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.01, time_limit)
        solver.parameters.relative_gap_limit = 0.0
        solver.parameters.num_search_workers = int(self.options.get("workers", 5))
        solver.parameters.random_seed = int(self.options.get("seed", 42)) + seed_offset
        for name in (
            "linearization_level",
            "cp_model_probing_level",
            "symmetry_level",
        ):
            value = self.options.get(name)
            if value is not None:
                setattr(solver.parameters, name, int(value))
        return solver

    @staticmethod
    def _assignment_from(solver, problem, x) -> dict[int, int]:
        return {
            node: next(
                zone
                for zone in sorted(problem.candidate_zones(node))
                if solver.Value(x[(zone, node)])
            )
            for node in problem.nodes
        }

    @staticmethod
    def _add_incumbent_hints(model, problem, x, cutoffs, incumbent) -> None:
        for (zone, node), var in x.items():
            model.AddHint(var, int(incumbent.assignment[node] == zone))
        for school, var in cutoffs.items():
            model.AddHint(var, incumbent.result.school_cutoffs[school])

    def _add_interval_capacity_cut(
        self,
        model,
        problem,
        market,
        x,
        cutoffs,
        school,
        intervals,
        max_cutoff,
    ) -> bool:
        restricted = market.zone_restricted_schools
        profiles = Counter()
        for interval in intervals:
            student = interval.student
            higher = tuple(
                sorted(
                    (
                        preferred,
                        student.priorities[preferred] * market.lottery_scale
                        + interval.high
                        - 1,
                    )
                    for preferred in interval.higher
                )
            )
            qualify_limit = (
                student.priorities[school] * market.lottery_scale + interval.low - 1
            )
            profiles[(student.node, qualify_limit, higher)] += (
                interval.high - interval.low + 1
            )

        signature = (school, tuple(sorted(profiles.items())))
        if signature in self._capacity_cut_signatures:
            return False
        self._capacity_cut_signatures.add(signature)

        terms = []
        for (student_node, qualify_limit, higher), weight in profiles.items():
            school_qualifies = self._comparison(
                model, cutoffs[school], school, qualify_limit, max_cutoff
            )
            target_access = None
            if school in restricted:
                target_access = self._zone_access(
                    model, problem, market, x, student_node, school
                )
            blockers = []
            for preferred, qualify_limit_at_high in higher:
                preferred_qualifies = self._comparison(
                    model,
                    cutoffs[preferred],
                    preferred,
                    qualify_limit_at_high,
                    max_cutoff,
                )
                if preferred in restricted:
                    preferred_access = self._zone_access(
                        model, problem, market, x, student_node, preferred
                    )
                    blockers.append(
                        self._affordability(
                            model,
                            student_node,
                            preferred,
                            qualify_limit_at_high,
                            preferred_access,
                            preferred_qualifies,
                            max_cutoff,
                        )
                    )
                else:
                    blockers.append(preferred_qualifies)
            profile = (school, student_node, qualify_limit, higher)
            demand = self._demand_vars.get(profile)
            if demand is None:
                demand = model.NewBoolVar(f"interval_demand_{len(self._demand_vars)}")
                self._demand_vars[profile] = demand
                clause = [demand, school_qualifies.Not(), *blockers]
                if target_access is not None:
                    clause.append(target_access.Not())
                model.AddBoolOr(clause)
                self._cut_profile_count += 1
            terms.append(weight * demand)

        model.Add(sum(terms) <= market.school_capacities[school] * market.lottery_scale)
        self._cut_count += 1
        return True

    def _comparison(self, model, cutoff, school, limit, max_cutoff):
        key = (school, limit)
        if key in self._comparison_vars:
            return self._comparison_vars[key]
        if limit < 0:
            var = model.NewConstant(0)
        elif limit >= max_cutoff:
            var = model.NewConstant(1)
        else:
            var = model.NewBoolVar(f"cutoff_le_{school}_{limit}")
            model.Add(cutoff <= limit).OnlyEnforceIf(var)
            model.Add(cutoff > limit).OnlyEnforceIf(var.Not())
        self._comparison_vars[key] = var
        return var

    def _zone_access(self, model, problem, market, x, block, school):
        """Return whether a block and restricted school share a zone."""
        key = (block, school)
        if key in self._zone_access_vars:
            return self._zone_access_vars[key]

        school_node = market.school_nodes[school]
        if block == school_node:
            access = model.NewConstant(1)
            self._zone_access_vars[key] = access
            return access

        access = model.NewBoolVar(f"block_zone_access_{block}_{school}")
        self._zone_access_vars[key] = access
        for zone in sorted(problem.candidate_zones(school_node)):
            school_zone = x[(zone, school_node)]
            block_zone = x.get((zone, block))
            if block_zone is None:
                model.AddImplication(school_zone, access.Not())
                continue
            # One school-zone literal is true, so these one-way implications
            # determine access without encoding the converse implication.
            model.AddBoolOr([school_zone.Not(), access.Not(), block_zone])
            model.AddBoolOr([school_zone.Not(), access, block_zone.Not()])
        return access

    def _affordability(
        self,
        model,
        block,
        school,
        limit,
        access,
        qualifies,
        max_cutoff,
    ):
        """Return the conjunction of zone access and cutoff qualification."""
        if limit < 0:
            return qualifies
        if limit >= max_cutoff:
            return access
        key = (block, school, limit)
        if key in self._affordability_vars:
            return self._affordability_vars[key]
        affordable = model.NewBoolVar(f"affordable_{block}_{school}_{limit}")
        model.AddImplication(affordable, access)
        model.AddImplication(affordable, qualifies)
        model.AddBoolOr([affordable, access.Not(), qualifies.Not()])
        self._affordability_vars[key] = affordable
        return affordable

    def _blocking(
        self,
        model,
        zone,
        school,
        limit,
        school_zone,
        school_qualifies,
    ):
        key = (zone, school, limit)
        if key in self._blocking_vars:
            return self._blocking_vars[key]
        var = model.NewBoolVar(f"blocking_{zone}_{school}_{limit}")
        model.AddImplication(var, school_zone)
        model.AddImplication(var, school_qualifies)
        model.AddBoolOr([var, school_zone.Not(), school_qualifies.Not()])
        self._blocking_vars[key] = var
        return var


def _candidate_demands(
    problem: ZoneProblem,
    market: CutoffMarket,
    assignment: dict[int, int],
    cutoffs: dict[int, int],
) -> tuple[
    dict[int, int],
    dict[int, list[_DemandInterval]],
]:
    restricted = market.zone_restricted_schools
    demands = {school: 0 for school in market.school_capacities}
    intervals = defaultdict(list)
    scale = market.lottery_scale
    for student in market.students:
        zone = assignment[student.node]
        remaining = scale
        for rank, school in enumerate(student.preferences):
            if school in restricted and assignment[market.school_nodes[school]] != zone:
                continue
            threshold = min(
                scale,
                max(0, cutoffs[school] - student.priorities[school] * scale),
            )
            if remaining > threshold:
                mass = remaining - threshold
                demands[school] += mass
                intervals[school].append(
                    _DemandInterval(
                        student,
                        threshold + 1,
                        remaining,
                        student.preferences[:rank],
                    )
                )
            remaining = min(remaining, threshold)
    return demands, intervals


def _solve_cutoffs(problem: ZoneProblem, assignment: dict[int, int]):
    market = problem.cutoff_market
    unrestricted = set(market.school_capacities) - set(market.zone_restricted_schools)
    if unrestricted:
        return solve_coupled_cutoffs(market, assignment, num_zones=problem.Z)
    return solve_zoned_cutoffs(market, assignment, num_zones=problem.Z)


def _solve_continuum_cutoffs(problem: ZoneProblem, assignment: dict[int, int]):
    market = problem.cutoff_market
    unrestricted = set(market.school_capacities) - set(market.zone_restricted_schools)
    if unrestricted:
        return solve_coupled_continuum_cutoffs(market, assignment, num_zones=problem.Z)
    return solve_zoned_continuum_cutoffs(market, assignment, num_zones=problem.Z)


def _grid_demands(result) -> dict[int, int]:
    if isinstance(result, CoupledCutoffResult):
        return result.market.demands
    return {
        school: demand
        for zone in result.zones.values()
        for school, demand in zone.demands.items()
    }
