"""Logic-based Benders decomposition for stable finite-grid welfare."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace

from ortools.sat.python import cp_model

from optimization.branch_price import ZonePattern, solve_pattern_root
from optimization.data import contiguity
from optimization.problem import CutoffMarket, CutoffStudent
from optimization.solution import ZoneSolution
from optimization.solvers.balance import balance_constraints
from optimization.solvers.cutoff_decomposition import (
    CutoffDecompositionSolver,
    _candidate_demands,
)
from optimization.solvers.welfare import add_finite_grid_welfare_model
from optimization.solvers.recom import (
    _DeadlineReached,
    _NoProposal,
    _ReComContext,
    _ReComKernel,
)
from optimization.welfare_oracle import (
    WelfareResult,
    outward_true_welfare_upper_bound,
    raw_welfare_upper_bound,
    solve_zoned_continuum_welfare,
    solve_zoned_welfare,
    validate_welfare_market,
)


@dataclass(frozen=True)
class _WelfareIncumbent:
    assignment: dict[int, int]
    result: WelfareResult


class WelfareDecompositionSolver(CutoffDecompositionSolver):
    """Separate stable demand and submodular welfare from geographic zoning."""

    def __init__(
        self, zoning_solver, *, utility_scale: int, prefix_depth: int = 10
    ) -> None:
        super().__init__(zoning_solver)
        self.utility_scale = int(utility_scale)
        self.prefix_depth = int(prefix_depth)
        self._welfare_cut_count = 0
        self._welfare_term_count = 0
        self._zoning_no_good_count = 0
        self._pattern_root_upper_bound = None

    def solve(self, problem) -> ZoneSolution:
        market = problem.cutoff_market
        if market is None:
            raise ValueError("Welfare decomposition requires a cutoff market.")
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
        self._reset_cut_state()
        self._welfare_cut_count = 0
        self._welfare_term_count = 0
        self._zoning_no_good_count = 0

        started = time.monotonic()
        time_limit = float(self.options.get("solve_time_limit", 60.0))
        deadline = started + time_limit
        incumbent, heuristic_rows, starts = self._initial_welfare_incumbent(
            problem, min(deadline, started + min(45.0, time_limit * 0.2))
        )
        incumbent, access_rows, access_starts = self._submodular_access_improve(
            problem,
            incumbent,
            min(
                deadline,
                time.monotonic() + min(60.0, time_limit * 0.1),
            ),
        )
        heuristic_rows.extend(access_rows)
        starts.extend(access_starts)
        incumbent, recom_rows, recom_starts = self._recom_welfare_improve(
            problem,
            incumbent,
            min(
                deadline,
                time.monotonic() + min(600.0, time_limit * 0.75),
            ),
        )
        heuristic_rows.extend(recom_rows)
        starts.extend(recom_starts)
        relaxation = self._assignment_relaxation(
            problem,
            incumbent,
            min(
                60.0,
                max(1.0, time_limit * 0.05),
                max(1.0, deadline - time.monotonic()),
            ),
        )
        relaxation_bound = relaxation[1]
        capacity_upper_bound = self._global_capacity_upper_bound(market)
        if relaxation[0] is not None:
            relaxation_start = _WelfareIncumbent(
                relaxation[0],
                solve_zoned_welfare(
                    market,
                    relaxation[0],
                    num_zones=problem.Z,
                    utility_scale=self.utility_scale,
                ),
            )
            starts.append(relaxation_start)
            if (
                relaxation_start.result.raw_scaled_welfare
                > incumbent.result.raw_scaled_welfare
            ):
                incumbent = relaxation_start
            heuristic_rows.append(
                {
                    "kind": "capacitated_assignment_relaxation",
                    "status": relaxation[2],
                    "raw_scaled_upper_bound": relaxation_bound,
                    "raw_scaled_welfare": relaxation_start.result.raw_scaled_welfare,
                    "welfare": relaxation_start.result.welfare,
                }
            )

        incumbent, subset_rows, subset_starts = self._subset_market_improve(
            problem,
            incumbent,
            min(
                deadline,
                time.monotonic() + min(120.0, time_limit * 0.1),
            ),
        )
        heuristic_rows.extend(subset_rows)
        starts.extend(subset_starts)

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
            school: model.NewIntVar(0, max_cutoff, f"welfare_master_cutoff_{school}")
            for school in market.school_capacities
        }
        upper_bound = raw_welfare_upper_bound(market, self.utility_scale)
        upper_bound = min(upper_bound, relaxation_bound, capacity_upper_bound)
        if self._pattern_root_upper_bound is not None:
            upper_bound = min(upper_bound, self._pattern_root_upper_bound)
        node_upper_bounds = defaultdict(int)
        for student in market.students:
            node_upper_bounds[student.node] += market.lottery_scale * max(
                (
                    round(student.utilities[school] * self.utility_scale)
                    for school in student.preferences
                ),
                default=0,
            )
        theta = {
            node: model.NewIntVar(
                0, node_upper_bounds[node], f"stable_welfare_upper_bound_{node}"
            )
            for node in node_upper_bounds
        }
        theta_total = sum(theta.values())
        model.Add(theta_total <= upper_bound)
        model.Maximize(theta_total)
        self._add_prefix_welfare_bounds(
            model,
            problem,
            market,
            theta,
            x,
            cutoffs,
            max_cutoff,
            self.prefix_depth,
        )
        for start in starts:
            self._add_interval_welfare_cut(
                model,
                problem,
                market,
                theta,
                x,
                cutoffs,
                start.assignment,
                start.result.cutoffs.school_cutoffs,
                max_cutoff,
            )

        rounds = []
        best_upper_bound = upper_bound
        certified = False
        termination = "time_limit"
        round_index = 0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            solver = self._new_solver(min(180.0, remaining), round_index)
            model.ClearHints()
            self._add_hints(model, market, x, cutoffs, theta, incumbent)
            round_started = time.monotonic()
            status = solver.Solve(model)
            round_seconds = time.monotonic() - round_started
            status_name = solver.StatusName(status)
            row = {
                "round": round_index,
                "status": status_name,
                "wall_time": round_seconds,
                "master_objective": None,
                "best_upper_bound": None,
                "oracle_incumbent": incumbent.result.raw_scaled_welfare,
                "overloaded_schools": 0,
                "capacity_cuts_added": 0,
                "welfare_cuts_added": 0,
                "zoning_no_goods_added": 0,
            }
            if status == cp_model.INFEASIBLE:
                certified = True
                termination = "no_better_solution"
                best_upper_bound = incumbent.result.raw_scaled_welfare
                rounds.append(row)
                break
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                termination = status_name.lower()
                rounds.append(row)
                break

            master_objective = int(round(solver.ObjectiveValue()))
            round_upper_bound = int(math.floor(solver.BestObjectiveBound() + 1e-6))
            best_upper_bound = min(best_upper_bound, round_upper_bound)
            assignment = self._assignment_from(solver, problem, x)
            candidate_cutoffs = {
                school: int(solver.Value(variable))
                for school, variable in cutoffs.items()
            }
            demands, intervals = _candidate_demands(
                problem, market, assignment, candidate_cutoffs
            )
            overloaded = [
                school
                for school, demand in demands.items()
                if demand > market.school_capacities[school] * market.lottery_scale
            ]
            candidate_welfare = self._add_interval_welfare_cut(
                model,
                problem,
                market,
                theta,
                x,
                cutoffs,
                assignment,
                candidate_cutoffs,
                max_cutoff,
            )
            capacity_cuts = 0
            separated_schools = sorted(
                overloaded,
                key=lambda school: (
                    demands[school]
                    - market.school_capacities[school] * market.lottery_scale
                ),
                reverse=True,
            )[:20]
            for school in separated_schools:
                capacity_cuts += self._add_interval_capacity_cut(
                    model,
                    problem,
                    market,
                    x,
                    cutoffs,
                    school,
                    intervals[school],
                    max_cutoff,
                )

            oracle = solve_zoned_welfare(
                market,
                assignment,
                num_zones=problem.Z,
                utility_scale=self.utility_scale,
            )
            if oracle.raw_scaled_welfare > incumbent.result.raw_scaled_welfare:
                incumbent = _WelfareIncumbent(assignment, oracle)
            model.Add(
                sum(x[assignment[node], node] for node in problem.nodes)
                <= problem.A - 1
            )
            self._zoning_no_good_count += 1

            row.update(
                {
                    "master_objective": master_objective,
                    "best_upper_bound": round_upper_bound,
                    "candidate_welfare": candidate_welfare,
                    "oracle_incumbent": incumbent.result.raw_scaled_welfare,
                    "overloaded_schools": len(overloaded),
                    "capacity_cuts_added": capacity_cuts,
                    "welfare_cuts_added": 1,
                    "zoning_no_goods_added": 1,
                }
            )
            rounds.append(row)
            if best_upper_bound <= incumbent.result.raw_scaled_welfare:
                certified = True
                termination = "upper_bound_reached_incumbent"
                break
            if (
                status == cp_model.OPTIMAL
                and not overloaded
                and master_objective == candidate_welfare
            ):
                certified = True
                termination = "master_candidate_exact_and_feasible"
                incumbent = _WelfareIncumbent(assignment, oracle)
                best_upper_bound = candidate_welfare
                break
            round_index += 1

        wall_time = time.monotonic() - started
        return self._solution(
            problem,
            incumbent,
            wall_time,
            certified,
            min(best_upper_bound, upper_bound),
            termination,
            rounds,
            heuristic_rows,
            model,
            relaxation[2],
            relaxation_bound,
            capacity_upper_bound,
        )

    def _initial_welfare_incumbent(self, problem, deadline):
        if problem.hint and self._valid_geographic_hint(problem, problem.hint):
            start = _WelfareIncumbent(
                dict(problem.hint),
                solve_zoned_welfare(
                    problem.cutoff_market,
                    problem.hint,
                    num_zones=problem.Z,
                    utility_scale=self.utility_scale,
                ),
            )
            return (
                start,
                [
                    {
                        "kind": "welfare_start",
                        "source": "provided_assignment",
                        "raw_scaled_welfare": start.result.raw_scaled_welfare,
                        "welfare": start.result.welfare,
                    }
                ],
                [start],
            )
        cutoff_incumbent, rows = self._initial_incumbent(problem, deadline)
        cutoff_starts = getattr(self, "_local_starts", [cutoff_incumbent])
        by_signature = {}
        for start in [cutoff_incumbent, *cutoff_starts]:
            signature = tuple(start.assignment[node] for node in problem.nodes)
            by_signature.setdefault(signature, start.assignment)
        if problem.hint and self._valid_geographic_hint(problem, problem.hint):
            signature = tuple(problem.hint[node] for node in problem.nodes)
            by_signature.setdefault(signature, dict(problem.hint))
        starts = [
            _WelfareIncumbent(
                assignment,
                solve_zoned_welfare(
                    problem.cutoff_market,
                    assignment,
                    num_zones=problem.Z,
                    utility_scale=self.utility_scale,
                ),
            )
            for assignment in by_signature.values()
        ]
        incumbent = max(starts, key=lambda item: item.result.raw_scaled_welfare)
        for start in starts:
            rows.append(
                {
                    "kind": "welfare_start",
                    "raw_scaled_welfare": start.result.raw_scaled_welfare,
                    "welfare": start.result.welfare,
                }
            )
        return incumbent, rows, starts

    def _valid_geographic_hint(self, problem, assignment):
        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        for node, zone in assignment.items():
            model.Add(x[zone, node] == 1)
        solver = self._new_solver(5.0, 20_000)
        solver.parameters.num_search_workers = 1
        return solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def _submodular_access_improve(self, problem, incumbent, deadline):
        """Optimize each student's best accessible utility as a start model."""
        market = problem.cutoff_market
        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        self.zoning_solver._add_search_strategy(model, problem, x, y)
        same_zone = self.zoning_solver._add_vertex_school_same_zone_indicators(
            model, problem, x, market
        )
        best_terms = []
        scale = market.lottery_scale
        for student_index, student in enumerate(market.students):
            coefficients = {
                school: round(student.utilities[school] * self.utility_scale)
                for school in student.preferences
            }
            upper = max(coefficients.values(), default=0)
            best = model.NewIntVar(
                0,
                upper,
                f"access_best_utility_{student_index}",
            )
            expressions = [
                coefficient * same_zone[(student.node, school)]
                for school, coefficient in coefficients.items()
            ]
            model.AddMaxEquality(best, [0, *expressions])
            incumbent_value = max(
                (
                    coefficient
                    for school, coefficient in coefficients.items()
                    if incumbent.assignment[student.node]
                    == incumbent.assignment[market.school_nodes[school]]
                ),
                default=0,
            )
            model.AddHint(best, incumbent_value)
            best_terms.append(scale * best)
        for (zone, node), variable in x.items():
            model.AddHint(variable, int(incumbent.assignment[node] == zone))
        for node, variable in y.items():
            model.AddHint(variable, incumbent.assignment[node])
        for (node, school), variable in same_zone.items():
            model.AddHint(
                variable,
                int(
                    incumbent.assignment[node]
                    == incumbent.assignment[market.school_nodes[school]]
                ),
            )
        objective = sum(best_terms)
        model.Maximize(objective)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return incumbent, [], []
        solver = self._new_solver(remaining, 22_000)
        started = time.monotonic()
        status = solver.Solve(model)
        elapsed = time.monotonic() - started
        row = {
            "kind": "submodular_access_start",
            "status": solver.StatusName(status),
            "wall_time": elapsed,
        }
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return incumbent, [row], []
        assignment = self._assignment_from(solver, problem, x)
        result = solve_zoned_welfare(
            market,
            assignment,
            num_zones=problem.Z,
            utility_scale=self.utility_scale,
        )
        row.update(
            {
                "raw_access_objective": int(round(solver.ObjectiveValue())),
                "raw_access_upper_bound": int(
                    math.floor(solver.BestObjectiveBound() + 1e-6)
                ),
                "raw_scaled_welfare": result.raw_scaled_welfare,
                "welfare": result.welfare,
            }
        )
        start = _WelfareIncumbent(assignment, result)
        if result.raw_scaled_welfare > incumbent.result.raw_scaled_welfare:
            return start, [row], [start]
        return incumbent, [row], [start]

    def _recom_welfare_improve(self, problem, incumbent, deadline):
        """Search exact welfare over CP-compatible ReCom proposals."""
        explicit_candidates = {
            node: set(problem.candidate_zones(node)) for node in problem.nodes
        }
        recom_problem = replace(
            problem,
            candidates=explicit_candidates,
            hint=dict(incumbent.assignment),
            _candidates=None,
        )
        context = _ReComContext(recom_problem)
        rng = random.Random(int(self.options.get("seed", 42)) + 25_000)
        column_seconds = min(45.0, max(0.0, deadline - time.monotonic()) * 0.25)
        recom_deadline = max(time.monotonic(), deadline - column_seconds)
        kernel = _ReComKernel(context, rng, recom_deadline)
        supports = contiguity.contiguity_supports(
            problem.G,
            problem.centroids,
            problem.centroid_school_ids,
            problem.candidate_zones,
        )
        max_boundary = math.floor(problem.boundary_prop * problem.G.number_of_edges())
        burst_length = int(self.options.get("short_bursts_length", 25))
        if burst_length <= 0:
            raise ValueError("short_bursts_length must be positive.")
        cache = {}
        transport_cache = {}

        def load_assignment(assignment):
            loaded_assignment = dict(assignment)
            loaded_state = context.build_state(context.validate_hint(loaded_assignment))
            zone_values = {
                zone: self._cached_zone_welfare(problem, loaded_assignment, zone, cache)
                for zone in range(problem.Z)
            }
            raw_value = sum(value[0] for value in zone_values.values())
            transport_values = {
                zone: self._cached_zone_transport(
                    problem, loaded_assignment, zone, transport_cache
                )
                for zone in range(problem.Z)
            }
            return (
                loaded_state,
                loaded_assignment,
                zone_values,
                raw_value,
                transport_values,
                sum(transport_values.values()),
            )

        (
            state,
            current_assignment,
            current_zone_values,
            current_raw,
            current_transport_values,
            current_transport,
        ) = load_assignment(incumbent.assignment)
        if current_raw != incumbent.result.raw_scaled_welfare:
            raise RuntimeError("Zone welfare cache does not reconstruct the incumbent.")
        best_transport = current_transport
        best_transport_assignment = dict(current_assignment)
        best = incumbent
        starts = []
        attempted = 0
        geographic_feasible = 0
        evaluated = 0
        accepted = 0
        improvements = 0
        proposal_failures = 0
        completed_bursts = 0
        selected_burst_improvements = 0
        started = time.monotonic()
        search_phase = "stable_initial"
        burst_steps = 0
        burst_base_assignment = dict(current_assignment)
        burst_base_score = current_raw
        burst_best_assignment = dict(current_assignment)
        burst_best_score = current_raw

        while time.monotonic() < recom_deadline:
            attempted += 1
            try:
                move = kernel.propose(state, selector="uniform")
            except _DeadlineReached:
                break
            except _NoProposal:
                proposal_failures += 1
                continue
            if problem.boundary_prop >= 0 and move.cut_edges > max_boundary:
                continue
            trial = state.clone()
            kernel.apply(trial, move)
            candidate = context.assignment_dict(trial.assignment)
            if not self._master_geography_feasible(
                problem, candidate, supports, explicit_candidates
            ):
                continue
            geographic_feasible += 1
            changed_zones = (move.zone_a, move.zone_b)
            candidate_values = {
                zone: self._cached_zone_welfare(problem, candidate, zone, cache)
                for zone in changed_zones
            }
            candidate_transport_values = {
                zone: self._cached_zone_transport(
                    problem, candidate, zone, transport_cache
                )
                for zone in changed_zones
            }
            evaluated += 1
            candidate_raw = current_raw + sum(
                candidate_values[zone][0] - current_zone_values[zone][0]
                for zone in changed_zones
            )
            progress = min(
                1.0,
                (time.monotonic() - started) / max(1e-9, deadline - started),
            )
            next_phase = None
            reset_assignment = None
            if progress >= 0.45 and search_phase == "stable_initial":
                next_phase = "transport"
                reset_assignment = best.assignment
            elif progress >= 0.8 and search_phase == "transport":
                next_phase = "stable_recovery"
                reset_assignment = best_transport_assignment
            if next_phase is not None:
                search_phase = next_phase
                (
                    state,
                    current_assignment,
                    current_zone_values,
                    current_raw,
                    current_transport_values,
                    current_transport,
                ) = load_assignment(reset_assignment)
                burst_steps = 0
                burst_base_assignment = dict(current_assignment)
                burst_base_score = (
                    current_transport if search_phase == "transport" else current_raw
                )
                burst_best_assignment = dict(current_assignment)
                burst_best_score = burst_base_score
                continue
            candidate_transport = current_transport + sum(
                candidate_transport_values[zone] - current_transport_values[zone]
                for zone in changed_zones
            )
            if candidate_transport > best_transport:
                best_transport = candidate_transport
                best_transport_assignment = dict(candidate)
            candidate_score = (
                candidate_transport if search_phase == "transport" else candidate_raw
            )
            if candidate_score > burst_best_score:
                burst_best_score = candidate_score
                burst_best_assignment = dict(candidate)

            # Follow an unrejected walk, then restart from the best exact score
            # visited in the fixed-length burst.
            state = trial
            current_assignment = candidate
            current_raw = candidate_raw
            current_zone_values.update(candidate_values)
            current_transport = candidate_transport
            current_transport_values.update(candidate_transport_values)
            accepted += 1
            if candidate_raw > best.result.raw_scaled_welfare:
                result = solve_zoned_welfare(
                    problem.cutoff_market,
                    candidate,
                    num_zones=problem.Z,
                    utility_scale=self.utility_scale,
                )
                if result.raw_scaled_welfare > best.result.raw_scaled_welfare:
                    best = _WelfareIncumbent(candidate, result)
                    starts.append(best)
                    improvements += 1

            burst_steps += 1
            if burst_steps < burst_length:
                continue
            if burst_best_score > burst_base_score:
                burst_base_assignment = dict(burst_best_assignment)
                burst_base_score = burst_best_score
                selected_burst_improvements += 1
            (
                state,
                current_assignment,
                current_zone_values,
                current_raw,
                current_transport_values,
                current_transport,
            ) = load_assignment(burst_base_assignment)
            burst_steps = 0
            burst_best_assignment = dict(current_assignment)
            burst_best_score = burst_base_score
            completed_bursts += 1

        combined, column_row = self._combine_zone_patterns(
            problem,
            cache,
            best,
            max(0.0, deadline - time.monotonic()),
        )
        if combined is not None:
            starts.append(combined)
            if combined.result.raw_scaled_welfare > best.result.raw_scaled_welfare:
                best = combined
                improvements += 1

        rows = [
            {
                "kind": "recom_welfare_search",
                "wall_time": time.monotonic() - started,
                "attempted_moves": attempted,
                "geographically_feasible_moves": geographic_feasible,
                "evaluated_moves": evaluated,
                "accepted_moves": accepted,
                "improvements": improvements,
                "proposal_failures": proposal_failures,
                "completed_bursts": completed_bursts,
                "selected_burst_improvements": selected_burst_improvements,
                "short_bursts_length": burst_length,
                "short_bursts_score": "stable_welfare_or_transport_upper_bound",
                "zone_cache_entries": len(cache),
                "transport_cache_entries": len(transport_cache),
                "best_transport_upper_bound": best_transport,
                "best_transport_upper_bound_welfare": best_transport
                / (problem.cutoff_market.lottery_scale * self.utility_scale),
                "raw_scaled_welfare": best.result.raw_scaled_welfare,
                "welfare": best.result.welfare,
            }
        ]
        if column_row is not None:
            rows.append(column_row)
        return best, rows, starts[-10:]

    def _cached_zone_welfare(self, problem, assignment, zone, cache):
        node_signature = frozenset(
            node for node in problem.nodes if assignment[node] == zone
        )
        if node_signature in cache:
            return cache[node_signature]
        zone_problem = self._subset_problem(problem, assignment, (zone,))
        result = solve_zoned_welfare(
            zone_problem.cutoff_market,
            assignment,
            num_zones=problem.Z,
            utility_scale=self.utility_scale,
        )
        value = (result.raw_scaled_welfare, result.welfare)
        cache[node_signature] = value
        return value

    def _cached_zone_transport(self, problem, assignment, zone, cache):
        node_signature = frozenset(
            node for node in problem.nodes if assignment[node] == zone
        )
        if node_signature in cache:
            return cache[node_signature]
        zone_problem = self._subset_problem(problem, assignment, (zone,))
        value = self._global_capacity_upper_bound(zone_problem.cutoff_market)
        cache[node_signature] = value
        return value

    def _combine_zone_patterns(self, problem, cache, incumbent, time_limit):
        if time_limit <= 0:
            return None, None

        centroids = set(problem.centroids)
        patterns = []
        for nodes, (raw_welfare, _) in cache.items():
            contained = centroids & nodes
            if len(contained) != 1:
                continue
            centroid = next(iter(contained))
            zone = problem.centroids.index(centroid)
            patterns.append(
                ZonePattern.from_graph(
                    label=zone,
                    nodes=nodes,
                    raw_welfare=raw_welfare,
                    graph=problem.G,
                )
            )
        if not patterns:
            return None, None

        started = time.monotonic()
        pricing_time = 0.8 * time_limit
        root = solve_pattern_root(
            problem,
            patterns,
            utility_scale=self.utility_scale,
            centroid_neighbor_radius=int(
                self.options.get("centroid_neighbor_radius", 0)
            ),
            pricing_time_limit=0.0,
            mip_time_limit=max(0.0, time_limit - pricing_time),
            exact_pricing_time_limit=pricing_time,
            run_access_models=False,
            workers=int(self.options.get("workers", 1)),
            random_seed=int(self.options.get("seed", 42)) + 26_000,
        )
        elapsed = time.monotonic() - started
        self._pattern_root_upper_bound = root.certificate.upper_bound
        mip = root.restricted_mip
        row = {
            "kind": "branch_price_root",
            "seed_pattern_count": len(patterns),
            "added_pattern_count": len(root.added_patterns),
            "enriched_pattern_count": len(root.patterns),
            "initial_lp_status": root.initial_lp.status,
            "initial_lp_raw_upper_bound": root.initial_lp.objective,
            "enriched_lp_status": root.enriched_lp.status,
            "enriched_lp_raw_upper_bound": root.enriched_lp.objective,
            "restricted_mip_status": mip.status,
            "restricted_mip_raw_welfare": mip.objective,
            "certificate_raw_upper_bound": root.certificate.upper_bound,
            "certificate_welfare_upper_bound": root.certificate.upper_bound
            / (problem.cutoff_market.lottery_scale * self.utility_scale),
            "zone_perimeter_cap": root.certificate.zone_perimeter_cap,
            "quantized_boundary_multiplier": (root.certificate.multipliers.boundary),
            "pricing": [
                {
                    "label": pricing.label,
                    "status": pricing.status,
                    "candidate_raw_welfare": (
                        pricing.candidate.raw_welfare
                        if pricing.candidate is not None
                        else None
                    ),
                    "candidate_lagrangian_value": (pricing.candidate_lagrangian_value),
                    "candidate_access_lagrangian_value": (
                        pricing.candidate_access_lagrangian_value
                    ),
                    "pricing_lagrangian_upper_bound": (
                        pricing.pricing_lagrangian_upper_bound
                    ),
                    "analytic_lagrangian_upper_bound": (
                        pricing.analytic_lagrangian_upper_bound
                    ),
                }
                for pricing in root.pricing
            ],
            "exact_pricing": [
                {
                    "label": pricing.label,
                    "status": pricing.status,
                    "wall_time": pricing.wall_time,
                    "build_time": pricing.build_time,
                    "solve_time": pricing.solve_time,
                    "candidate_raw_welfare": (
                        pricing.candidate.raw_welfare
                        if pricing.candidate is not None
                        else None
                    ),
                    "candidate_lagrangian_value": (pricing.candidate_lagrangian_value),
                    "pricing_lagrangian_upper_bound": (
                        pricing.pricing_lagrangian_upper_bound
                    ),
                    "analytic_lagrangian_upper_bound": (
                        pricing.analytic_lagrangian_upper_bound
                    ),
                }
                for pricing in root.exact_pricing
            ],
            "wall_time": elapsed,
        }
        if mip.assignment is None:
            return None, row
        result = solve_zoned_welfare(
            problem.cutoff_market,
            mip.assignment,
            num_zones=problem.Z,
            utility_scale=self.utility_scale,
        )
        if result.raw_scaled_welfare != mip.objective:
            raise RuntimeError(
                "Restricted pattern objective does not reconstruct exact welfare."
            )
        row.update(
            {
                "raw_scaled_welfare": result.raw_scaled_welfare,
                "welfare": result.welfare,
            }
        )
        return _WelfareIncumbent(mip.assignment, result), row

    def _master_geography_feasible(
        self,
        problem,
        assignment,
        supports,
        explicit_candidates,
    ):
        for zone, centroid in enumerate(problem.centroids):
            if assignment[centroid] != zone:
                return False
            for node in self.zoning_solver._centroid_neighborhoods(problem)[zone]:
                if assignment[node] != zone:
                    return False
        for node, zone in assignment.items():
            if zone not in explicit_candidates[node]:
                return False
            if node != problem.centroids[zone] and not any(
                assignment[neighbor] == zone
                for neighbor in supports.get((node, zone), ())
            ):
                return False

        for constraint in balance_constraints(problem):
            if problem.cutoff_market is not None and constraint.kind == "capacity":
                continue
            for zone in range(problem.Z):
                if constraint.lower_ratio is not None:
                    lower = sum(
                        round(
                            100
                            * (
                                constraint.value(node)
                                - constraint.lower_ratio * problem.students(node)
                            )
                        )
                        for node, assigned in assignment.items()
                        if assigned == zone
                    )
                    if lower < 0:
                        return False
                if constraint.upper_ratio is not None:
                    upper = sum(
                        round(
                            100
                            * (
                                constraint.value(node)
                                - constraint.upper_ratio * problem.students(node)
                            )
                        )
                        for node, assigned in assignment.items()
                        if assigned == zone
                    )
                    if upper > 0:
                        return False

        total_schools = sum(problem.num_schools(node) for node in problem.nodes)
        if total_schools:
            average = total_schools / problem.Z
            lower = round(100 * max(0.0, average - 1.0))
            upper = round(100 * (average + 1.0))
            for zone in range(problem.Z):
                count = 100 * sum(
                    problem.num_schools(node)
                    for node, assigned in assignment.items()
                    if assigned == zone
                )
                if count < lower or count > upper:
                    return False
        if problem.boundary_prop >= 0:
            boundary = sum(
                assignment[left] != assignment[right]
                for left, right in problem.G.edges()
            )
            if boundary > math.floor(
                problem.boundary_prop * problem.G.number_of_edges()
            ):
                return False
        return True

    def _subset_market_improve(self, problem, incumbent, deadline):
        """Reoptimize complete isolated markets over adjacent zone pairs."""
        rows = []
        starts = []
        pass_index = 0
        while time.monotonic() < deadline:
            pairs = self._adjacent_zone_pairs(problem, incumbent.assignment)
            candidates = []
            for zones in pairs:
                local_problem = self._subset_problem(
                    problem, incumbent.assignment, zones
                )
                local_result = solve_zoned_welfare(
                    local_problem.cutoff_market,
                    incumbent.assignment,
                    num_zones=problem.Z,
                    utility_scale=self.utility_scale,
                )
                potential = (
                    self._global_capacity_upper_bound(local_problem.cutoff_market)
                    - local_result.raw_scaled_welfare
                )
                candidates.append((potential, zones, local_problem, local_result))
            candidates.sort(reverse=True, key=lambda item: item[0])
            accepted = False
            for candidate_index, (
                potential,
                zones,
                local_problem,
                local_result,
            ) in enumerate(candidates):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                candidates_left = len(candidates) - candidate_index
                slot = min(15.0, max(3.0, remaining / candidates_left))
                assignment, row = self._solve_subset_market(
                    local_problem,
                    incumbent,
                    local_result,
                    zones,
                    min(slot, remaining),
                    pass_index * 100 + candidate_index,
                )
                row["transport_headroom"] = potential
                rows.append(row)
                if assignment is None:
                    continue
                result = solve_zoned_welfare(
                    problem.cutoff_market,
                    assignment,
                    num_zones=problem.Z,
                    utility_scale=self.utility_scale,
                )
                row["raw_scaled_welfare_after"] = result.raw_scaled_welfare
                row["welfare_after"] = result.welfare
                row["raw_gain"] = (
                    result.raw_scaled_welfare - incumbent.result.raw_scaled_welfare
                )
                if result.raw_scaled_welfare <= incumbent.result.raw_scaled_welfare:
                    continue
                incumbent = _WelfareIncumbent(assignment, result)
                starts.append(incumbent)
                accepted = True
                break
            if not accepted:
                break
            pass_index += 1
        return incumbent, rows, starts

    @staticmethod
    def _adjacent_zone_pairs(problem, assignment):
        return sorted(
            {
                tuple(sorted((assignment[left], assignment[right])))
                for left, right in problem.G.edges()
                if assignment[left] != assignment[right]
            }
        )

    @staticmethod
    def _subset_problem(problem, assignment, zones):
        selected_nodes = {node for node in problem.nodes if assignment[node] in zones}
        market = problem.cutoff_market
        schools = {
            school
            for school, node in market.school_nodes.items()
            if node in selected_nodes
        }
        students = []
        for student in market.students:
            if student.node not in selected_nodes:
                continue
            preferences = tuple(
                school for school in student.preferences if school in schools
            )
            students.append(
                CutoffStudent(
                    student.studentno,
                    student.node,
                    preferences,
                    {school: student.priorities[school] for school in preferences},
                    {school: student.utilities[school] for school in preferences},
                )
            )
        reduced_market = CutoffMarket(
            students=tuple(students),
            school_nodes={school: market.school_nodes[school] for school in schools},
            school_capacities={
                school: market.school_capacities[school] for school in schools
            },
            zone_restricted_schools=frozenset(schools),
            lottery_scale=market.lottery_scale,
            outside_option_utility=market.outside_option_utility,
            metadata=market.metadata,
        )
        candidates = {
            node: (
                problem.candidate_zones(node) & set(zones)
                if node in selected_nodes
                else {assignment[node]}
            )
            for node in problem.nodes
        }
        return replace(
            problem,
            candidates=candidates,
            hint=dict(assignment),
            cutoff_market=reduced_market,
            _candidates=None,
        )

    def _solve_subset_market(
        self,
        problem,
        incumbent,
        local_result,
        zones,
        time_limit,
        seed_offset,
    ):
        mip_started = time.monotonic()
        try:
            assignment, _, status_name = self._assignment_relaxation_mip(
                problem,
                incumbent,
                time_limit,
            )
        except ImportError:  # pragma: no cover - Gurobi is optional
            assignment = None
        else:
            mip_elapsed = time.monotonic() - mip_started
            return assignment, {
                "kind": "exact_subset_welfare",
                "engine": "gurobi_stable_recurrence",
                "zones": list(zones),
                "node_count": sum(
                    incumbent.assignment[node] in zones for node in problem.nodes
                ),
                "student_count": len(problem.cutoff_market.students),
                "school_count": len(problem.cutoff_market.school_capacities),
                "preference_entry_count": sum(
                    len(student.preferences)
                    for student in problem.cutoff_market.students
                ),
                "status": status_name,
                "wall_time": mip_elapsed,
                "raw_scaled_welfare_before": incumbent.result.raw_scaled_welfare,
                "local_raw_before": local_result.raw_scaled_welfare,
                "local_optimum_certified": False,
            }

        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        self.zoning_solver._add_search_strategy(model, problem, x, y)
        _, raw_welfare = add_finite_grid_welfare_model(
            model,
            problem,
            x,
            incumbent.assignment,
            local_result,
            zoning_solver=self.zoning_solver,
            utility_scale=self.utility_scale,
        )
        model.Maximize(raw_welfare)
        for (zone, node), variable in x.items():
            model.AddHint(variable, int(incumbent.assignment[node] == zone))
        for node, variable in y.items():
            model.AddHint(variable, incumbent.assignment[node])

        solver = self._new_solver(time_limit, 40_000 + seed_offset)
        started = time.monotonic()
        status = solver.Solve(model)
        elapsed = time.monotonic() - started
        row = {
            "kind": "exact_subset_welfare",
            "engine": "cp_sat",
            "zones": list(zones),
            "node_count": sum(
                incumbent.assignment[node] in zones for node in problem.nodes
            ),
            "student_count": len(problem.cutoff_market.students),
            "school_count": len(problem.cutoff_market.school_capacities),
            "preference_entry_count": sum(
                len(student.preferences) for student in problem.cutoff_market.students
            ),
            "status": solver.StatusName(status),
            "wall_time": elapsed,
            "raw_scaled_welfare_before": incumbent.result.raw_scaled_welfare,
            "local_raw_before": local_result.raw_scaled_welfare,
        }
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, row
        row.update(
            {
                "local_solver_objective": int(round(solver.ObjectiveValue())),
                "local_solver_upper_bound": int(
                    math.floor(solver.BestObjectiveBound() + 1e-6)
                ),
                "local_optimum_certified": status == cp_model.OPTIMAL,
            }
        )
        return self._assignment_from(solver, problem, x), row

    def _fixed_cutoff_improve(self, problem, incumbent, deadline):
        """Optimize exact assignment welfare at successive feasible cutoffs.

        The incumbent zoning is feasible at its least cutoff vector. Holding
        that vector fixed, this model optimizes geography with the complete
        assignment recurrence and all school capacities. The resulting zoning
        is therefore capacity feasible at the fixed vector; replacing it by its
        own least cutoff vector can only improve every applicant cell.
        """
        market = problem.cutoff_market
        scale = market.lottery_scale
        rows = []
        starts = []
        iteration = 0
        while time.monotonic() < deadline:
            model = cp_model.CpModel()
            x, y = self.zoning_solver._build_assignment_vars(model, problem)
            self.zoning_solver._add_core_constraints(model, problem, x, y)
            self.zoning_solver._add_search_strategy(model, problem, x, y)
            same_zone = self.zoning_solver._add_vertex_school_same_zone_indicators(
                model, problem, x, market
            )
            school_terms = {school: [] for school in market.school_capacities}
            objective_terms = []
            incumbent_cutoffs = incumbent.result.cutoffs.school_cutoffs

            for student_index, student in enumerate(market.students):
                previous = scale
                previous_hint = scale
                for rank, school in enumerate(student.preferences, start=1):
                    threshold = min(
                        scale,
                        max(
                            0,
                            incumbent_cutoffs[school]
                            - student.priorities[school] * scale,
                        ),
                    )
                    together = same_zone[(student.node, school)]
                    effective = model.NewIntVar(
                        threshold,
                        scale,
                        f"fixed_effective_{student_index}_{rank}",
                    )
                    model.Add(effective == threshold).OnlyEnforceIf(together)
                    model.Add(effective == scale).OnlyEnforceIf(together.Not())
                    cumulative = model.NewIntVar(
                        0,
                        scale,
                        f"fixed_remaining_{student_index}_{rank}",
                    )
                    model.AddMinEquality(cumulative, [previous, effective])
                    mass = previous - cumulative
                    school_terms[school].append(mass)
                    coefficient = round(student.utilities[school] * self.utility_scale)
                    if coefficient:
                        objective_terms.append(coefficient * mass)

                    in_incumbent_zone = (
                        incumbent.assignment[student.node]
                        == incumbent.assignment[market.school_nodes[school]]
                    )
                    effective_hint = threshold if in_incumbent_zone else scale
                    previous_hint = min(previous_hint, effective_hint)
                    model.AddHint(effective, effective_hint)
                    model.AddHint(cumulative, previous_hint)
                    previous = cumulative

            for school, capacity in market.school_capacities.items():
                model.Add(sum(school_terms[school]) <= scale * capacity)
            raw_welfare = sum(objective_terms)
            model.Add(raw_welfare > incumbent.result.raw_scaled_welfare)
            model.Maximize(raw_welfare)
            for (zone, node), variable in x.items():
                model.AddHint(variable, int(incumbent.assignment[node] == zone))
            for node, variable in y.items():
                model.AddHint(variable, incumbent.assignment[node])

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            solver = self._new_solver(min(60.0, remaining), 30_000 + iteration)
            started = time.monotonic()
            status = solver.Solve(model)
            elapsed = time.monotonic() - started
            row = {
                "kind": "fixed_cutoff_welfare",
                "iteration": iteration,
                "status": solver.StatusName(status),
                "wall_time": elapsed,
                "raw_scaled_welfare_before": incumbent.result.raw_scaled_welfare,
            }
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                rows.append(row)
                break

            assignment = self._assignment_from(solver, problem, x)
            result = solve_zoned_welfare(
                market,
                assignment,
                num_zones=problem.Z,
                utility_scale=self.utility_scale,
            )
            row.update(
                {
                    "fixed_cutoff_objective": int(round(solver.ObjectiveValue())),
                    "fixed_cutoff_upper_bound": int(
                        math.floor(solver.BestObjectiveBound() + 1e-6)
                    ),
                    "raw_scaled_welfare_after": result.raw_scaled_welfare,
                    "welfare_after": result.welfare,
                }
            )
            rows.append(row)
            if result.raw_scaled_welfare <= incumbent.result.raw_scaled_welfare:
                break
            incumbent = _WelfareIncumbent(assignment, result)
            starts.append(incumbent)
            iteration += 1
        return incumbent, rows, starts

    def _add_prefix_welfare_bounds(
        self,
        model,
        problem,
        market,
        theta,
        x,
        cutoffs,
        max_cutoff,
        depth,
    ):
        """Globally bound welfare with exact recurrence through top ranks."""
        scale = market.lottery_scale
        same_zone = {}
        thresholds = {}
        node_terms = defaultdict(list)
        variable_count = 0

        def together(student_node, school):
            key = (student_node, school)
            if key in same_zone:
                return same_zone[key]
            school_node = market.school_nodes[school]
            same = model.NewBoolVar(f"prefix_same_{student_node}_{school}")
            if student_node == school_node:
                model.Add(same == 1)
            else:
                for zone in problem.candidate_zones(student_node):
                    model.Add(same == x.get((zone, school_node), 0)).OnlyEnforceIf(
                        x[zone, student_node]
                    )
            same_zone[key] = same
            return same

        for student_index, student in enumerate(market.students):
            ranked = student.preferences[:depth]
            previous = scale
            welfare_terms = []
            for rank, school in enumerate(ranked, start=1):
                priority = student.priorities[school]
                threshold_key = (school, priority)
                if threshold_key not in thresholds:
                    threshold = model.NewIntVar(
                        0, max_cutoff, f"prefix_threshold_{school}_{priority}"
                    )
                    model.AddMaxEquality(
                        threshold,
                        cutoffs[school] - priority * scale,
                        0,
                    )
                    thresholds[threshold_key] = threshold
                threshold = thresholds[threshold_key]
                effective = model.NewIntVar(
                    0,
                    max_cutoff,
                    f"prefix_effective_{student_index}_{rank}",
                )
                same = together(student.node, school)
                model.Add(effective == threshold).OnlyEnforceIf(same)
                model.Add(effective == scale).OnlyEnforceIf(same.Not())
                cumulative = model.NewIntVar(
                    0, scale, f"prefix_remaining_{student_index}_{rank}"
                )
                model.AddMinEquality(cumulative, [previous, effective])
                utility = round(student.utilities[school] * self.utility_scale)
                welfare_terms.append(utility * (previous - cumulative))
                previous = cumulative
                variable_count += 2
            residual_utility = (
                round(
                    student.utilities[student.preferences[depth]] * self.utility_scale
                )
                if len(student.preferences) > depth
                else 0
            )
            if residual_utility:
                welfare_terms.append(residual_utility * previous)
            node_terms[student.node].append(sum(welfare_terms))

        for node, variable in theta.items():
            model.Add(variable <= sum(node_terms[node]))
        self._prefix_depth = depth
        self._prefix_variable_count = variable_count

    def _add_interval_welfare_cut(
        self,
        model,
        problem,
        market,
        theta,
        x,
        cutoffs,
        assignment,
        candidate_cutoffs,
        max_cutoff,
    ) -> int:
        """Bound all ranks with candidate-tight preference-interval literals."""
        scale = market.lottery_scale
        constants = defaultdict(int)
        terms = defaultdict(list)
        cut_index = self._welfare_cut_count
        interval_index = 0
        candidate_welfare = 0
        for student in market.students:
            zone = assignment[student.node]
            student_zone = x[zone, student.node]
            best_utility = max(
                (
                    round(student.utilities[school] * self.utility_scale)
                    for school in student.preferences
                ),
                default=0,
            )
            remaining = scale
            preferred_schools = []
            for school in student.preferences:
                if assignment[market.school_nodes[school]] != zone:
                    preferred_schools.append(school)
                    continue
                threshold = min(
                    scale,
                    max(
                        0,
                        candidate_cutoffs[school] - student.priorities[school] * scale,
                    ),
                )
                if remaining > threshold:
                    low = threshold + 1
                    high = remaining
                    mass = high - low + 1
                    utility = round(student.utilities[school] * self.utility_scale)
                    candidate_welfare += mass * utility
                    constants[student.node] += mass * best_utility
                    loss = best_utility - utility
                    if loss:
                        persists = self._target_interval_persists(
                            model,
                            problem,
                            market,
                            x,
                            cutoffs,
                            student,
                            zone,
                            student_zone,
                            school,
                            preferred_schools,
                            low,
                            high,
                            max_cutoff,
                            cut_index,
                            interval_index,
                        )
                        terms[student.node].append(-mass * loss * persists)
                        interval_index += 1
                remaining = min(remaining, threshold)
                preferred_schools.append(school)

            if remaining:
                constants[student.node] += remaining * best_utility
                if best_utility:
                    persists = self._outside_interval_persists(
                        model,
                        problem,
                        market,
                        x,
                        cutoffs,
                        student,
                        zone,
                        student_zone,
                        remaining,
                        max_cutoff,
                        cut_index,
                        interval_index,
                    )
                    terms[student.node].append(-remaining * best_utility * persists)
                    interval_index += 1

        for student_node, theta_variable in theta.items():
            model.Add(
                theta_variable <= constants[student_node] + sum(terms[student_node])
            )
        self._welfare_cut_count += len(theta)
        self._welfare_term_count += interval_index
        return candidate_welfare

    def _target_interval_persists(
        self,
        model,
        problem,
        market,
        x,
        cutoffs,
        student,
        zone,
        student_zone,
        school,
        preferred_schools,
        low,
        high,
        max_cutoff,
        cut_index,
        interval_index,
    ):
        target_zone = x.get((zone, market.school_nodes[school]))
        if target_zone is None:
            raise RuntimeError("Candidate target school is not a zone candidate.")
        target_limit = student.priorities[school] * market.lottery_scale + low - 1
        target_qualifies = self._comparison(
            model, cutoffs[school], school, target_limit, max_cutoff
        )
        blockers = self._interval_blockers(
            model,
            problem,
            market,
            x,
            cutoffs,
            student,
            zone,
            preferred_schools,
            high,
            max_cutoff,
        )
        persists = model.NewBoolVar(f"welfare_persist_{cut_index}_{interval_index}")
        model.Add(
            persists
            >= student_zone + target_zone + target_qualifies - 2 - sum(blockers)
        )
        return persists

    def _outside_interval_persists(
        self,
        model,
        problem,
        market,
        x,
        cutoffs,
        student,
        zone,
        student_zone,
        high,
        max_cutoff,
        cut_index,
        interval_index,
    ):
        blockers = self._interval_blockers(
            model,
            problem,
            market,
            x,
            cutoffs,
            student,
            zone,
            student.preferences,
            high,
            max_cutoff,
        )
        persists = model.NewBoolVar(
            f"welfare_outside_persist_{cut_index}_{interval_index}"
        )
        model.Add(persists >= student_zone - sum(blockers))
        return persists

    def _interval_blockers(
        self,
        model,
        problem,
        market,
        x,
        cutoffs,
        student,
        zone,
        schools,
        high,
        max_cutoff,
    ):
        blockers = []
        for school in schools:
            school_zone = x.get((zone, market.school_nodes[school]))
            if school_zone is None:
                continue
            limit = student.priorities[school] * market.lottery_scale + high - 1
            qualifies = self._comparison(
                model, cutoffs[school], school, limit, max_cutoff
            )
            blockers.append(
                self._blocking(
                    model,
                    zone,
                    school,
                    limit,
                    school_zone,
                    qualifies,
                )
            )
        return blockers

    def _global_capacity_upper_bound(self, market):
        """Drop zoning and priorities, then solve exact capacitated transport."""
        from ortools.graph.python import min_cost_flow

        flow = min_cost_flow.SimpleMinCostFlow()
        student_count = len(market.students)
        schools = tuple(market.school_capacities)
        school_nodes = {
            school: student_count + index for index, school in enumerate(schools)
        }
        source = student_count + len(schools)
        sink = source + 1
        scale = market.lottery_scale
        for student_index, student in enumerate(market.students):
            flow.add_arc_with_capacity_and_unit_cost(source, student_index, scale, 0)
            flow.add_arc_with_capacity_and_unit_cost(student_index, sink, scale, 0)
            for school in student.preferences:
                coefficient = round(student.utilities[school] * self.utility_scale)
                flow.add_arc_with_capacity_and_unit_cost(
                    student_index,
                    school_nodes[school],
                    scale,
                    -coefficient,
                )
        for school in schools:
            flow.add_arc_with_capacity_and_unit_cost(
                school_nodes[school],
                sink,
                market.school_capacities[school] * scale,
                0,
            )
        flow.set_node_supply(source, student_count * scale)
        flow.set_node_supply(sink, -student_count * scale)
        status = flow.solve()
        if status != flow.OPTIMAL:
            raise RuntimeError("Could not solve the global welfare flow bound.")
        return -int(flow.optimal_cost())

    def _assignment_relaxation(self, problem, incumbent, time_limit):
        try:
            return self._assignment_transport_mip(problem, incumbent, time_limit)
        except ImportError:  # pragma: no cover - Gurobi is optional
            return self._assignment_relaxation_cp(problem, incumbent, time_limit)

    def _assignment_transport_mip(self, problem, incumbent, time_limit):
        """Optimize the same-zone capacitated transport upper relaxation."""
        import gurobipy as gp
        from gurobipy import GRB

        from optimization.solvers.balance import balance_constraints, balance_terms
        from optimization.solvers.mip import MipSolver

        class EncodedMipBuilder(MipSolver):
            def _add_linear_constraint(self, model, x, terms, sense, rhs):
                expression = gp.quicksum(
                    round(coefficient * 100) * x[zone, node]
                    for coefficient, zone, node in terms
                    if (zone, node) in x
                )
                scaled_rhs = round(rhs * 100)
                if sense == "<=":
                    model.addConstr(expression <= scaled_rhs)
                elif sense == ">=":
                    model.addConstr(expression >= scaled_rhs)
                else:
                    model.addConstr(expression == scaled_rhs)

            def _add_balance_constraints(self, model, zone_problem, x):
                constraints = balance_constraints(zone_problem)
                for zone in range(zone_problem.Z):
                    nodes = self._candidate_nodes(zone_problem, zone)
                    for constraint in constraints:
                        if constraint.kind == "capacity":
                            continue
                        lower, upper = balance_terms(
                            zone_problem, constraint, zone, nodes
                        )
                        if lower:
                            self._add_linear_constraint(model, x, lower, ">=", 0.0)
                        if upper:
                            self._add_linear_constraint(model, x, upper, "<=", 0.0)

        market = problem.cutoff_market
        scale = market.lottery_scale
        builder = EncodedMipBuilder(
            centroid_neighbor_radius=self.options.get("centroid_neighbor_radius", 0)
        )
        model = gp.Model("welfare_transport_relaxation")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 0.0
        model.Params.Seed = int(self.options.get("seed", 42)) + 12_000
        model.Params.Threads = int(self.options.get("workers", 5))
        model.Params.NumericFocus = 2
        x = builder._build_assignment_vars(model, problem)
        builder._add_core_constraints(model, problem, x)

        same_zone = {}
        pairs = {
            (student.node, school)
            for student in market.students
            for school in student.preferences
        }
        for student_node, school in sorted(pairs):
            school_node = market.school_nodes[school]
            same = model.addVar(
                vtype=GRB.BINARY, name=f"transport_same_{student_node}_{school}"
            )
            if student_node == school_node:
                model.addConstr(same == 1)
            else:
                for zone in problem.candidate_zones(student_node):
                    school_assignment = x.get((zone, school_node))
                    if school_assignment is None:
                        model.addGenConstrIndicator(
                            x[zone, student_node], True, same == 0
                        )
                    else:
                        model.addGenConstrIndicator(
                            x[zone, student_node],
                            True,
                            same == school_assignment,
                        )
            same.Start = int(
                incumbent.assignment[student_node] == incumbent.assignment[school_node]
            )
            same_zone[student_node, school] = same

        school_terms = {school: [] for school in market.school_capacities}
        objective_terms = []
        for student_index, student in enumerate(market.students):
            student_terms = []
            initial_assignments = incumbent.result.assignments[student.studentno]
            for school in student.preferences:
                mass = model.addVar(
                    lb=0.0,
                    ub=scale,
                    vtype=GRB.CONTINUOUS,
                    name=f"transport_mass_{student_index}_{school}",
                )
                model.addConstr(mass <= scale * same_zone[student.node, school])
                mass.Start = initial_assignments.get(school, 0)
                student_terms.append(mass)
                school_terms[school].append(mass)
                coefficient = round(student.utilities[school] * self.utility_scale)
                if coefficient:
                    objective_terms.append((coefficient / self.utility_scale) * mass)
            model.addConstr(gp.quicksum(student_terms) <= scale)
        for school, capacity in market.school_capacities.items():
            model.addConstr(gp.quicksum(school_terms[school]) <= scale * capacity)
        for (zone, node), variable in x.items():
            variable.Start = int(incumbent.assignment[node] == zone)
        objective_expression = gp.quicksum(objective_terms)
        global_bound = self._global_capacity_upper_bound(market)
        objective = model.addVar(
            lb=0.0,
            ub=global_bound / self.utility_scale,
            name="transport_objective",
        )
        model.addConstr(
            objective == objective_expression,
            name="transport_objective_definition",
        )
        model.setObjective(objective, GRB.MAXIMIZE)
        progress = []
        last_progress_time = -math.inf

        def record_progress(callback_model, where):
            nonlocal last_progress_time
            if where != GRB.Callback.MIP:
                return
            runtime = callback_model.cbGet(GRB.Callback.RUNTIME)
            if runtime < last_progress_time + 5.0:
                return
            incumbent_value = (
                callback_model.cbGet(GRB.Callback.MIP_OBJBST) * self.utility_scale
            )
            bound_value = (
                callback_model.cbGet(GRB.Callback.MIP_OBJBND) * self.utility_scale
            )
            progress.append(
                {
                    "runtime": runtime,
                    "incumbent": (
                        incumbent_value if math.isfinite(incumbent_value) else None
                    ),
                    "bound": bound_value if math.isfinite(bound_value) else None,
                    "node_count": callback_model.cbGet(GRB.Callback.MIP_NODCNT),
                }
            )
            last_progress_time = runtime

        model.optimize(record_progress)

        mip_bound = global_bound
        if math.isfinite(model.ObjBound):
            raw_solver_bound = model.ObjBound * self.utility_scale
            mip_bound = min(
                global_bound,
                math.ceil(math.nextafter(raw_solver_bound, math.inf)) + 1,
            )
        self._transport_mip_details = {
            "status": int(model.Status),
            "solution_count": int(model.SolCount),
            "raw_objective": (
                model.ObjVal * self.utility_scale if model.SolCount else None
            ),
            "solver_raw_upper_bound": (
                model.ObjBound * self.utility_scale
                if math.isfinite(model.ObjBound)
                else None
            ),
            "diagnostic_raw_upper_bound": mip_bound,
            "proof_grade_raw_upper_bound": global_bound,
            "node_count": int(model.NodeCount),
            "progress": progress,
        }
        if model.SolCount <= 0:
            return None, global_bound, f"GUROBI_TRANSPORT_{model.Status}"
        assignment = {
            node: max(problem.candidate_zones(node), key=lambda zone: x[zone, node].X)
            for node in problem.nodes
        }
        status = "OPTIMAL" if model.Status == GRB.OPTIMAL else "FEASIBLE"
        return assignment, global_bound, f"GUROBI_TRANSPORT_{status}"

    def _assignment_relaxation_mip(self, problem, incumbent, time_limit):
        """Solve the zoning-aware capacitated upper bound with Gurobi."""
        import gurobipy as gp
        from gurobipy import GRB

        from optimization.solvers.balance import balance_constraints, balance_terms
        from optimization.solvers.mip import MipSolver

        class EncodedMipBuilder(MipSolver):
            def _add_linear_constraint(self, model, x, terms, sense, rhs):
                expression = gp.quicksum(
                    round(coefficient * 100) * x[zone, node]
                    for coefficient, zone, node in terms
                    if (zone, node) in x
                )
                scaled_rhs = round(rhs * 100)
                if sense == "<=":
                    model.addConstr(expression <= scaled_rhs)
                elif sense == ">=":
                    model.addConstr(expression >= scaled_rhs)
                else:
                    model.addConstr(expression == scaled_rhs)

            def _add_balance_constraints(self, model, zone_problem, x):
                constraints = balance_constraints(zone_problem)
                for zone in range(zone_problem.Z):
                    nodes = self._candidate_nodes(zone_problem, zone)
                    for constraint in constraints:
                        if constraint.kind == "capacity":
                            continue
                        lower, upper = balance_terms(
                            zone_problem, constraint, zone, nodes
                        )
                        if lower:
                            self._add_linear_constraint(model, x, lower, ">=", 0.0)
                        if upper:
                            self._add_linear_constraint(model, x, upper, "<=", 0.0)

        market = problem.cutoff_market
        scale = market.lottery_scale
        builder = EncodedMipBuilder(
            centroid_neighbor_radius=self.options.get("centroid_neighbor_radius", 0)
        )
        model = gp.Model("welfare_assignment_relaxation")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 0.0
        model.Params.Seed = int(self.options.get("seed", 42)) + 10_000
        model.Params.Threads = int(self.options.get("workers", 5))
        x = builder._build_assignment_vars(model, problem)
        builder._add_core_constraints(model, problem, x)

        same_zone = {}
        pairs = {
            (student.node, school)
            for student in market.students
            for school in student.preferences
        }
        for student_node, school in sorted(pairs):
            school_node = market.school_nodes[school]
            same = model.addVar(
                vtype=GRB.BINARY, name=f"stable_same_{student_node}_{school}"
            )
            if student_node == school_node:
                model.addConstr(same == 1)
            else:
                for zone in problem.candidate_zones(student_node):
                    school_assignment = x.get((zone, school_node))
                    if school_assignment is None:
                        model.addGenConstrIndicator(
                            x[zone, student_node], True, same == 0
                        )
                    else:
                        model.addGenConstrIndicator(
                            x[zone, student_node],
                            True,
                            same == school_assignment,
                        )
            same.Start = int(
                incumbent.assignment[student_node] == incumbent.assignment[school_node]
            )
            same_zone[student_node, school] = same

        school_terms = {school: [] for school in market.school_capacities}
        objective_terms = []
        max_priority = max(
            (
                priority
                for student in market.students
                for priority in student.priorities.values()
            ),
            default=0,
        )
        max_cutoff = (max_priority + 1) * scale
        cutoffs = {
            school: model.addVar(
                lb=0,
                ub=max_cutoff,
                vtype=GRB.INTEGER,
                name=f"stable_cutoff_{school}",
            )
            for school in market.school_capacities
        }
        for school, variable in cutoffs.items():
            variable.Start = incumbent.result.cutoffs.school_cutoffs[school]
        thresholds = {}
        for student_index, student in enumerate(market.students):
            previous = model.addVar(
                lb=scale,
                ub=scale,
                vtype=GRB.CONTINUOUS,
                name=f"stable_remaining_{student_index}_0",
            )
            previous.Start = scale
            previous_hint = scale
            for rank, school in enumerate(student.preferences, start=1):
                priority = student.priorities[school]
                threshold_key = (school, priority)
                if threshold_key not in thresholds:
                    shifted = model.addVar(
                        lb=-max_cutoff,
                        ub=max_cutoff,
                        vtype=GRB.CONTINUOUS,
                        name=f"stable_shifted_{school}_{priority}",
                    )
                    model.addConstr(shifted == cutoffs[school] - priority * scale)
                    threshold = model.addVar(
                        lb=0,
                        ub=max_cutoff,
                        vtype=GRB.CONTINUOUS,
                        name=f"stable_threshold_{school}_{priority}",
                    )
                    model.addGenConstrMax(threshold, [shifted], constant=0.0)
                    thresholds[threshold_key] = threshold
                threshold = thresholds[threshold_key]
                effective = model.addVar(
                    lb=0.0,
                    ub=max_cutoff,
                    vtype=GRB.CONTINUOUS,
                    name=f"stable_effective_{student_index}_{rank}",
                )
                together = same_zone[student.node, school]
                model.addGenConstrIndicator(together, True, effective == threshold)
                model.addGenConstrIndicator(together, False, effective == scale)
                cumulative = model.addVar(
                    lb=0.0,
                    ub=scale,
                    vtype=GRB.CONTINUOUS,
                    name=f"stable_remaining_{student_index}_{rank}",
                )
                model.addGenConstrMin(cumulative, [previous, effective])
                assignment_mass = previous - cumulative
                school_terms[school].append(assignment_mass)
                coefficient = round(student.utilities[school] * self.utility_scale)
                if coefficient:
                    objective_terms.append(
                        (coefficient / self.utility_scale) * assignment_mass
                    )
                initial_threshold = max(
                    0,
                    incumbent.result.cutoffs.school_cutoffs[school] - priority * scale,
                )
                initial_effective = (
                    initial_threshold
                    if incumbent.assignment[student.node]
                    == incumbent.assignment[market.school_nodes[school]]
                    else scale
                )
                previous_hint = min(previous_hint, initial_effective)
                effective.Start = initial_effective
                cumulative.Start = previous_hint
                previous = cumulative
        for school, capacity in market.school_capacities.items():
            model.addConstr(gp.quicksum(school_terms[school]) <= scale * capacity)
        for (zone, node), variable in x.items():
            variable.Start = int(incumbent.assignment[node] == zone)
        model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)
        model.optimize()

        diagnostic_bound = None
        if math.isfinite(model.ObjBound):
            diagnostic_bound = (
                math.ceil(
                    math.nextafter(
                        model.ObjBound * self.utility_scale,
                        math.inf,
                    )
                )
                + 1
            )
        self._stable_mip_details = {
            "status": int(model.Status),
            "solution_count": int(model.SolCount),
            "raw_objective": (
                model.ObjVal * self.utility_scale if model.SolCount else None
            ),
            "diagnostic_raw_upper_bound": diagnostic_bound,
            "node_count": int(model.NodeCount),
        }
        # Gurobi's scaled floating objective is useful for finding zonings, but
        # its bound cannot safely be rounded back into an exact integer cap.
        # Certification therefore retains the independent integer upper bound.
        bound = raw_welfare_upper_bound(market, self.utility_scale)
        if model.SolCount <= 0:
            return None, bound, f"GUROBI_{model.Status}"
        assignment = {
            node: max(problem.candidate_zones(node), key=lambda zone: x[zone, node].X)
            for node in problem.nodes
        }
        status = "OPTIMAL" if model.Status == GRB.OPTIMAL else "FEASIBLE"
        return assignment, bound, f"GUROBI_STABLE_{status}"

    def _assignment_relaxation_cp(self, problem, incumbent, time_limit):
        """Optimize a zoning-aware capacitated assignment upper bound."""
        market = problem.cutoff_market
        scale = market.lottery_scale
        model = cp_model.CpModel()
        x, y = self.zoning_solver._build_assignment_vars(model, problem)
        self.zoning_solver._add_core_constraints(model, problem, x, y)
        same_zone = self.zoning_solver._add_vertex_school_same_zone_indicators(
            model, problem, x, market
        )
        school_terms = {school: [] for school in market.school_capacities}
        objective_terms = []
        for student_index, student in enumerate(market.students):
            student_terms = []
            incumbent_assignments = incumbent.result.assignments[student.studentno]
            for school in student.preferences:
                mass = model.NewIntVar(
                    0, scale, f"relaxed_assignment_{student_index}_{school}"
                )
                model.Add(mass <= scale * same_zone[(student.node, school)])
                model.AddHint(mass, incumbent_assignments.get(school, 0))
                student_terms.append(mass)
                school_terms[school].append(mass)
                coefficient = round(student.utilities[school] * self.utility_scale)
                if coefficient:
                    objective_terms.append(coefficient * mass)
            model.Add(sum(student_terms) <= scale)
        for school, capacity in market.school_capacities.items():
            model.Add(sum(school_terms[school]) <= scale * capacity)
        for (zone, node), variable in x.items():
            model.AddHint(variable, int(incumbent.assignment[node] == zone))
        for node, variable in y.items():
            model.AddHint(variable, incumbent.assignment[node])
        objective = sum(objective_terms)
        global_bound = self._global_capacity_upper_bound(market)
        model.Add(objective <= global_bound)
        model.Maximize(objective)
        solver = self._new_solver(time_limit, 10_000)
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self._transport_cp_details = {
                "status": solver.StatusName(status),
                "raw_objective": None,
                "raw_upper_bound": global_bound,
                "wall_time": solver.WallTime(),
                "branches": solver.NumBranches(),
                "conflicts": solver.NumConflicts(),
                "response_stats": solver.ResponseStats(),
            }
            return None, global_bound, solver.StatusName(status)
        assignment = self._assignment_from(solver, problem, x)
        if status == cp_model.OPTIMAL:
            bound = int(round(solver.ObjectiveValue()))
        else:
            bound = min(
                global_bound,
                math.ceil(math.nextafter(solver.BestObjectiveBound(), math.inf)),
            )
        bound = max(bound, int(round(solver.ObjectiveValue())))
        self._transport_cp_details = {
            "status": solver.StatusName(status),
            "raw_objective": int(round(solver.ObjectiveValue())),
            "raw_upper_bound": bound,
            "wall_time": solver.WallTime(),
            "branches": solver.NumBranches(),
            "conflicts": solver.NumConflicts(),
            "response_stats": solver.ResponseStats(),
        }
        return assignment, bound, solver.StatusName(status)

    @staticmethod
    def _add_hints(model, market, x, cutoffs, theta, incumbent):
        for (zone, node), variable in x.items():
            model.AddHint(variable, int(incumbent.assignment[node] == zone))
        for school, variable in cutoffs.items():
            model.AddHint(variable, incumbent.result.cutoffs.school_cutoffs[school])
        node_welfare = defaultdict(int)
        students = {student.studentno: student for student in market.students}
        for studentno, assignments in incumbent.result.assignments.items():
            student = students[studentno]
            for school, mass in assignments.items():
                node_welfare[student.node] += mass * round(
                    student.utilities[school] * incumbent.result.utility_scale
                )
        for node, variable in theta.items():
            model.AddHint(variable, node_welfare[node])

    def _solution(
        self,
        problem,
        incumbent,
        wall_time,
        certified,
        raw_upper_bound,
        termination,
        rounds,
        heuristic_rows,
        model,
        relaxation_status,
        relaxation_bound,
        capacity_upper_bound,
    ):
        market = problem.cutoff_market
        continuum = solve_zoned_continuum_welfare(
            market, incumbent.assignment, num_zones=problem.Z
        )
        zone_stable = {
            str(zone): result.stable for zone, result in continuum.cutoffs.zones.items()
        }
        normalizer = market.lottery_scale * self.utility_scale
        raw_upper_bound = max(incumbent.result.raw_scaled_welfare, raw_upper_bound)
        coefficient_error = math.nextafter(
            len(market.students) / (2 * self.utility_scale), math.inf
        )
        rounded_upper_bound = raw_upper_bound / normalizer
        true_upper_bound = outward_true_welfare_upper_bound(
            raw_upper_bound, market, self.utility_scale
        )
        metadata = {
            **market.metadata,
            "solver": "cp_bool",
            "objective_kind": "stable_assignment_welfare",
            "optimization_method": "submodular_welfare_lbbd",
            "market_coupling": "isolated_zones",
            "lottery_scale": market.lottery_scale,
            "welfare_utility_scale": self.utility_scale,
            "welfare": incumbent.result.welfare,
            "rounded_welfare": incumbent.result.raw_scaled_welfare / normalizer,
            "raw_scaled_welfare": incumbent.result.raw_scaled_welfare,
            "raw_scaled_upper_bound": raw_upper_bound,
            "rounded_welfare_upper_bound": rounded_upper_bound,
            "utility_rounding_error_bound": coefficient_error,
            "true_welfare_upper_bound": true_upper_bound,
            "true_welfare_gap_bound": max(
                0.0,
                true_upper_bound - incumbent.result.welfare,
            ),
            "global_optimum_certified": certified,
            "global_optimum_scope": (
                "Finite assignment and cutoff grid with utilities rounded to the "
                "configured fixed-point scale, over the encoded zoning domain."
            ),
            "school_cutoffs": incumbent.result.cutoffs.school_cutoffs,
            "normalized_school_cutoffs": {
                school: cutoff / market.lottery_scale
                for school, cutoff in incumbent.result.cutoffs.school_cutoffs.items()
            },
            "grid_minimal": incumbent.result.cutoffs.grid_minimal,
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
            "welfare_cut_count": self._welfare_cut_count,
            "welfare_cut_term_count": self._welfare_term_count,
            "welfare_prefix_depth": getattr(self, "_prefix_depth", 0),
            "welfare_prefix_variable_count": getattr(self, "_prefix_variable_count", 0),
            "revealed_preference_cut_count": self._cut_count,
            "revealed_preference_profile_count": self._cut_profile_count,
            "zoning_no_good_count": self._zoning_no_good_count,
            "heuristic_candidates": heuristic_rows,
            "assignment_relaxation_status": relaxation_status,
            "assignment_relaxation_raw_upper_bound": relaxation_bound,
            "global_capacity_raw_upper_bound": capacity_upper_bound,
            "model_stats": model.ModelStats(),
        }
        return ZoneSolution(
            problem=problem,
            assignment=incumbent.assignment,
            status="OPTIMAL" if certified else "FEASIBLE",
            objective=incumbent.result.welfare,
            wall_time=wall_time,
            metadata=metadata,
        )
