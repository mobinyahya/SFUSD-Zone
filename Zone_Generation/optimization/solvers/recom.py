"""GerryChain-backed ReCom local-search solver.

The solver preserves the project-level ``ZoneProblem`` / ``ZoneSolution`` API but
delegates initial tree partitioning and ReCom proposals to GerryChain. GerryChain
handles contiguous, student-balanced tree cuts; this layer keeps the SFUSD-specific
candidate, centroid, capacity, diversity, school-count, and objective scoring.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from functools import total_ordering
from typing import Mapping, TextIO

from gerrychain import Partition
from gerrychain.proposals import recom as gerrychain_recom
from gerrychain.proposals.tree_proposals import MetagraphError
from gerrychain.tree import (
    BalanceError,
    PopulationBalanceError,
    ReselectException,
    bipartition_tree,
)
from gerrychain.updaters import Tally, cut_edges

from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.data.initial_solutions import (
    initial_solution,
    normalize_hints,
    normalize_recom_balance_metric,
    recom_balance_epsilon,
    recom_balance_pop_col,
    recom_balance_target,
    recom_gerrychain_graph,
)
from Zone_Generation.optimization.progress import (
    SolverProgressTracker,
    assignment_tuple,
)
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.base import Solver, register

_EPS = 1e-6
_GERRYCHAIN_ERRORS = (
    BalanceError,
    PopulationBalanceError,
    ReselectException,
    MetagraphError,
    IndexError,
)


@total_ordering
@dataclass(frozen=True, eq=False)
class _Score:
    penalty: float
    boundary: int
    components: Mapping[str, float] | None = None

    @property
    def feasible(self) -> bool:
        return self.penalty <= _EPS

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Score):
            return NotImplemented
        if self.feasible != other.feasible:
            return False
        if self.feasible:
            return self.boundary == other.boundary
        return abs(self.penalty - other.penalty) <= _EPS

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _Score):
            return NotImplemented
        if self.feasible != other.feasible:
            return self.feasible
        if self.feasible:
            return self.boundary < other.boundary
        return self.penalty < other.penalty - _EPS

    def worse_delta_from(self, current: "_Score") -> float:
        if self <= current:
            return 0.0
        if self.feasible and current.feasible:
            return float(self.boundary - current.boundary)
        if not self.feasible and not current.feasible:
            return float(self.penalty - current.penalty)
        if not self.feasible and current.feasible:
            return max(float(self.penalty), _EPS)
        return 0.0


@dataclass(frozen=True)
class _PenaltyContext:
    signature: tuple
    coefficients: dict[str, float]
    reference_denominators: dict[str, float]
    district_racial: dict[str, float]
    district_frl: float
    avg_schools: float
    school_lower: float
    school_upper: float


@dataclass
class _InitialState:
    assignment: dict[int, int]
    metadata: dict


@register("recom")
class ReComSolver(Solver):
    """Randomized ReCom solver using GerryChain for tree cuts."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        if problem.choice_objective is not None:
            raise NotImplementedError(
                "recom does not support iterative choice objectives; use cp_int, cp_bool, or mip."
            )

        start = time.time()
        hint_error = self._hints_error_solution(problem, start)
        if hint_error is not None:
            return hint_error
        seed = int(self.options.get("seed", 42))
        rng = random.Random(seed)
        time_limit, max_iterations = self._recom_limits()
        cut_attempts = max(1, int(self.options.get("recom_cut_attempts", 100)))
        population_epsilon = self.options.get("recom_population_epsilon")
        balance_metric = normalize_recom_balance_metric(
            self.options.get("recom_balance_metric", "students")
        )
        temperature = max(0.0, float(self.options.get("recom_temperature", 0.0)))
        log_path, progress_log = self._open_progress_log(problem)
        progress = self._new_recom_progress_tracker(problem)

        random_state = random.getstate()
        random.seed(seed)
        try:
            try:
                initial = self._initial_state(
                    problem,
                    cut_attempts,
                    population_epsilon=population_epsilon,
                    balance_metric=balance_metric,
                )
                current = dict(initial.assignment)
                current_partition = self._partition(problem, current, balance_metric)
                current_score = self._score(problem, current)
                initial_score = current_score
                best = dict(current) if _valid(current_score) else None
                best_score = current_score if best is not None else None
                best_infeasible = None if best is not None else dict(current)
                best_infeasible_score = None if best is not None else current_score
                accepted = 0
                rejected = 0
                attempted = 0
                proposal_failures = 0
                last_proposal_error = None

                self._write_progress_log(
                    progress_log,
                    start=start,
                    event="initial",
                    iteration=0,
                    score=current_score,
                    best_score=best_score,
                )
                self._record_recom_progress(
                    progress,
                    start,
                    problem,
                    current,
                    current_score,
                    iteration=0,
                )

                while self._has_recom_iteration_budget(max_iterations, attempted):
                    if time.time() - start >= time_limit:
                        break
                    attempted += 1
                    try:
                        proposal_partition = self._gerrychain_proposal(
                            problem,
                            current_partition,
                            cut_attempts,
                            population_epsilon=population_epsilon,
                            balance_metric=balance_metric,
                        )
                    except _GERRYCHAIN_ERRORS as exc:
                        rejected += 1
                        proposal_failures += 1
                        last_proposal_error = type(exc).__name__
                        continue

                    proposal = self._assignment_from_partition(proposal_partition)
                    proposal_score = self._score(problem, proposal)
                    if not _valid(proposal_score) and (
                        best_infeasible_score is None
                        or proposal_score < best_infeasible_score
                    ):
                        best_infeasible = dict(proposal)
                        best_infeasible_score = proposal_score
                    accepted_move = self._accept(
                        current_score, proposal_score, temperature, rng
                    )
                    if accepted_move:
                        current = proposal
                        current_partition = proposal_partition
                        current_score = proposal_score
                        accepted += 1
                    else:
                        rejected += 1

                    if _valid(proposal_score) and (
                        best_score is None
                        or proposal_score.boundary < best_score.boundary
                    ):
                        best = dict(proposal)
                        best_score = proposal_score
                        self._record_recom_progress(
                            progress,
                            start,
                            problem,
                            proposal,
                            proposal_score,
                            iteration=attempted,
                        )

                    self._write_progress_log(
                        progress_log,
                        start=start,
                        event="cut",
                        iteration=attempted,
                        score=proposal_score,
                        accepted=accepted_move,
                        best_score=best_score,
                    )
            finally:
                if progress_log is not None:
                    progress_log.close()
        finally:
            random.setstate(random_state)

        wall = time.time() - start
        repair_metadata = {}
        if best is None and best_infeasible is not None and best_infeasible_score:
            repaired, repaired_score, repair_metadata = self._repair_infeasible_solution(
                problem, best_infeasible, best_infeasible_score
            )
            if _valid(repaired_score):
                best = repaired
                best_score = repaired_score
                self._record_recom_progress(
                    progress,
                    start,
                    problem,
                    repaired,
                    repaired_score,
                    iteration=attempted,
                )
        if best is not None and best_score is not None:
            status = "FEASIBLE"
            assignment = best
            objective = float(best_score.boundary)
        else:
            status = "UNKNOWN"
            assignment = {}
            objective = None
        best_penalty = (
            best_score.penalty
            if best_score is not None
            else best_infeasible_score.penalty
            if best_infeasible_score is not None
            else current_score.penalty
        )

        metadata = {
            "solver": self.name,
            **self._progress_log_metadata(log_path),
            **self._solver_progress_metadata(progress),
            "hints": initial.metadata.get("hints", self._hints()),
            "iterations": max_iterations,
            "recom_balance_metric": balance_metric,
            "recom_population_epsilon": _population_epsilon(
                problem, population_epsilon, balance_metric
            ),
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "rejected_moves": rejected,
            "proposal_failures": proposal_failures,
            "initial_penalty": initial_score.penalty,
            "best_penalty": best_penalty,
            "temperature": temperature,
            **repair_metadata,
            **initial.metadata,
        }
        if last_proposal_error is not None:
            metadata["last_proposal_error"] = last_proposal_error

        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status,
            objective=objective,
            wall_time=wall,
            metadata=metadata,
            solver_progress=list(progress.entries) if progress is not None else [],
        )

    # ------------------------------------------------------------------ #
    # Limit handling
    # ------------------------------------------------------------------ #
    def _recom_limits(self) -> tuple[float, int | None]:
        recom_iterations = int(self.options.get("recom_iterations", 1000))
        if recom_iterations < 0:
            if self.options.get("solve_time_limit") is None:
                raise ValueError(
                    "solve_time_limit must be supplied when recom_iterations is negative."
                )
            return float(self.options["solve_time_limit"]), None
        return float(self.options.get("solve_time_limit", 60.0)), recom_iterations

    @staticmethod
    def _has_recom_iteration_budget(
        max_iterations: int | None,
        attempted: int,
    ) -> bool:
        return max_iterations is None or attempted < max_iterations

    # ------------------------------------------------------------------ #
    # Progress logging
    # ------------------------------------------------------------------ #
    def _open_progress_log(
        self, problem: ZoneProblem
    ) -> tuple[str | None, TextIO | None]:
        log_path = self._next_solver_log_path(problem)
        if log_path is None:
            return None, None
        return log_path, open(log_path, "w", encoding="utf-8")

    def _progress_log_metadata(self, log_path: str | None) -> dict[str, str]:
        metadata = self._solver_log_metadata(log_path)
        if log_path:
            metadata["solver_log_format"] = "jsonl"
        return metadata

    def _write_progress_log(
        self,
        log_file: TextIO | None,
        *,
        start: float,
        event: str,
        iteration: int,
        score: _Score,
        accepted: bool | None = None,
        best_score: _Score | None = None,
    ) -> None:
        if log_file is None:
            return
        timestamp = time.time()
        penalty = float(score.penalty)
        row = {
            "event": event,
            "iteration": int(iteration),
            "timestamp": timestamp,
            "elapsed_seconds": timestamp - start,
            "cut_edges": int(score.boundary),
            "feasible": _valid(score),
            "penalty": penalty if math.isfinite(penalty) else None,
            "penalty_components": _json_penalty_components(score.components),
        }
        if accepted is not None:
            row["accepted"] = bool(accepted)
        if best_score is not None:
            row["best_cut_edges"] = int(best_score.boundary)
            row["best_feasible"] = _valid(best_score)
            row["best_penalty_components"] = _json_penalty_components(
                best_score.components
            )
        json.dump(row, log_file, sort_keys=True)
        log_file.write("\n")
        log_file.flush()

    def _new_recom_progress_tracker(
        self, problem: ZoneProblem
    ) -> SolverProgressTracker | None:
        return self._new_solver_progress_tracker(problem, maximize=False)

    def _record_recom_progress(
        self,
        progress: SolverProgressTracker | None,
        start: float,
        problem: ZoneProblem,
        assignment: Mapping[int, int],
        score: _Score,
        *,
        iteration: int,
    ) -> None:
        if progress is None or not _valid(score):
            return
        progress.add(
            score.boundary,
            time.time() - start,
            assignment_tuple(problem.nodes, assignment),
            iteration=iteration,
        )

    def _repair_infeasible_solution(
        self,
        problem: ZoneProblem,
        assignment: Mapping[int, int],
        score: _Score,
    ) -> tuple[dict[int, int], _Score, dict[str, object]]:
        current = dict(assignment)
        current_score = score
        max_steps = max(
            0,
            int(self.options.get("recom_repair_iterations", max(100, problem.A * 4))),
        )
        time_limit = max(0.0, float(self.options.get("recom_repair_time_limit", 3.0)))
        repair_start = time.time()
        checked_moves = 0
        steps = 0

        for _ in range(max_steps):
            if time_limit and time.time() - repair_start >= time_limit:
                break
            best_assignment = None
            best_score = current_score
            for node in self._repair_boundary_nodes(problem, current):
                if time_limit and time.time() - repair_start >= time_limit:
                    break
                current_zone = current[node]
                adjacent_zones = {
                    int(current[nb])
                    for nb in problem.G.neighbors(node)
                    if nb in current and current[nb] != current_zone
                }
                for zone in sorted(adjacent_zones):
                    if time_limit and time.time() - repair_start >= time_limit:
                        break
                    if zone not in self._candidate_zones(problem, node):
                        continue
                    trial = dict(current)
                    trial[node] = zone
                    checked_moves += 1
                    if not contiguity.is_contiguous(
                        problem.G, trial, problem.centroids
                    ):
                        continue
                    trial_score = self._score(problem, trial)
                    if trial_score < best_score:
                        best_assignment = trial
                        best_score = trial_score
            if best_assignment is None:
                break
            current = best_assignment
            current_score = best_score
            steps += 1
            if _valid(current_score):
                break

        return current, current_score, {
            "repair_attempted": True,
            "repair_steps": steps,
            "repair_checked_moves": checked_moves,
            "repair_final_penalty": current_score.penalty,
            "repair_success": _valid(current_score),
        }

    def _repair_boundary_nodes(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> list[int]:
        nodes = {
            int(u)
            for u, v in problem.G.edges()
            if assignment.get(u) != assignment.get(v)
        } | {
            int(v)
            for u, v in problem.G.edges()
            if assignment.get(u) != assignment.get(v)
        }
        return sorted(
            nodes,
            key=lambda node: (
                -problem.num_schools(node),
                problem.distance(problem.centroids[int(assignment[node])], node)
                if int(assignment[node]) < len(problem.centroids)
                else 0.0,
                node,
            ),
        )

    # ------------------------------------------------------------------ #
    # Initial assignment
    # ------------------------------------------------------------------ #
    def _initial_state(
        self,
        problem: ZoneProblem,
        cut_attempts: int,
        *,
        population_epsilon: float | None = None,
        balance_metric: object = "students",
    ) -> _InitialState:
        return self._initial_state_with_options(
            problem,
            cut_attempts,
            population_epsilon=population_epsilon,
            balance_metric=balance_metric,
        )

    def _initial_state_with_options(
        self,
        problem: ZoneProblem,
        cut_attempts: int,
        *,
        population_epsilon: float | None = None,
        balance_metric: object = "students",
    ) -> _InitialState:
        if problem.hint:
            return _InitialState(
                assignment=self._complete_assignment(problem, problem.hint),
                metadata={"hints": "provided", "hint_source": "problem_hint"},
            )

        initial = initial_solution(
            problem,
            self._hints(),
            cut_attempts=cut_attempts,
            population_epsilon=population_epsilon,
            balance_metric=balance_metric,
        )
        if initial is None:
            raise ValueError(
                "ReCom solvers require hints to be voronoi or gerry_chain."
            )
        return _InitialState(
            assignment=initial.assignment,
            metadata=dict(initial.metadata),
        )

    def _complete_assignment(
        self, problem: ZoneProblem, seed: Mapping[int, int]
    ) -> dict[int, int]:
        assignment: dict[int, int] = {}
        for node in problem.nodes:
            zone = seed.get(node)
            candidates = self._candidate_zones(problem, node)
            if zone in candidates:
                assignment[node] = int(zone)
                continue
            if not candidates:
                raise problem.no_candidate_zones_error(node)
            assignment[node] = min(
                candidates,
                key=lambda z: problem.distance(problem.centroids[z], node),
            )
        return assignment

    def _candidate_zones(self, problem: ZoneProblem, node: int) -> set[int]:
        if node in problem.centroids:
            return set(range(problem.Z))
        if problem.candidates is not None and node in problem.candidates:
            return set(problem.candidates[node])
        if problem.fixed is not None and node in problem.fixed:
            return {int(problem.fixed[node])}
        return set(range(problem.Z))

    def _hints(self) -> str:
        return normalize_hints(self.options.get("hints", "gerry_chain"))

    def _hints_error_solution(
        self, problem: ZoneProblem, start: float
    ) -> ZoneSolution | None:
        method = self._hints()
        if method != "none":
            return None
        return ZoneSolution(
            problem=problem,
            assignment={},
            status="ERROR",
            objective=None,
            wall_time=time.time() - start,
            metadata={
                "solver": self.name,
                "hints": method,
                "error_message": (
                    "ReCom solvers require hints to be voronoi or gerry_chain."
                ),
            },
        )

    # ------------------------------------------------------------------ #
    # GerryChain proposals
    # ------------------------------------------------------------------ #
    def _gerrychain_proposal(
        self,
        problem: ZoneProblem,
        partition: Partition,
        cut_attempts: int,
        *,
        population_epsilon: float | None = None,
        balance_metric: object = "students",
    ) -> Partition:
        metric = normalize_recom_balance_metric(balance_metric)
        try:
            return gerrychain_recom(
                partition,
                pop_col=recom_balance_pop_col(metric),
                pop_target=_population_target(problem, metric),
                epsilon=_population_epsilon(problem, population_epsilon, metric),
                method=partial(
                    bipartition_tree,
                    max_attempts=cut_attempts,
                    allow_pair_reselection=False,
                ),
            )
        except RuntimeError as exc:
            if "Could not find a possible cut" in str(exc):
                raise ReselectException(str(exc)) from exc
            raise

    def _partition(
        self,
        problem: ZoneProblem,
        assignment: Mapping[int, int],
        balance_metric: object = "students",
    ) -> Partition:
        metric = normalize_recom_balance_metric(balance_metric)
        graph = recom_gerrychain_graph(problem, metric)
        return Partition(
            graph,
            assignment={int(node): int(zone) for node, zone in assignment.items()},
            updaters={
                "population": Tally(recom_balance_pop_col(metric), alias="population"),
                "cut_edges": cut_edges,
            },
        )

    def _assignment_from_partition(self, partition: Partition) -> dict[int, int]:
        return {
            int(node): int(zone) for node, zone in partition.assignment.mapping.items()
        }

    def _accept(
        self,
        current: _Score,
        proposal: _Score,
        temperature: float,
        rng: random.Random,
    ) -> bool:
        if proposal <= current:
            return True
        if temperature <= 0:
            return False
        delta = proposal.worse_delta_from(current)
        if delta <= 0:
            return True
        return rng.random() < math.exp(-delta / temperature)

    # ------------------------------------------------------------------ #
    # Constraint scoring and validation
    # ------------------------------------------------------------------ #
    def _score(self, problem: ZoneProblem, assignment: Mapping[int, int]) -> _Score:
        components = self._penalty_components(problem, assignment)
        penalty = sum(components.values())
        return _Score(
            penalty=penalty,
            boundary=contiguity.boundary_edges(problem.G, dict(assignment)),
            components=components,
        )

    def _penalty_components(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> dict[str, float]:
        components: defaultdict[str, float] = defaultdict(float)
        hard_penalty = float(problem.A + problem.Z + 1) * 1000.0

        if set(assignment) != set(problem.nodes):
            missing = set(problem.nodes) - set(assignment)
            extra = set(assignment) - set(problem.nodes)
            components["assignment"] += hard_penalty * (len(missing) + len(extra))

        for node in problem.nodes:
            zone = assignment.get(node)
            if zone not in self._candidate_zones(problem, node):
                components["candidate"] += hard_penalty

        if set(assignment) >= set(problem.nodes) and not contiguity.is_contiguous(
            problem.G, dict(assignment), problem.centroids
        ):
            components["contiguity"] += hard_penalty

        for key, value in self._constraint_penalty_components(
            problem, assignment
        ).items():
            components[key] += value
        return {key: value for key, value in components.items() if value}

    def _constraint_penalty_components(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> dict[str, float]:
        context = self._penalty_context(problem)
        components = self._balance_penalty_components(problem, assignment, context)
        schools = self._school_count_penalty(problem, assignment, context)
        if schools:
            components["schools"] += schools
        return dict(components)

    def _balance_penalty_components(
        self,
        problem: ZoneProblem,
        assignment: Mapping[int, int],
        context: _PenaltyContext,
    ) -> defaultdict[str, float]:
        components: defaultdict[str, float] = defaultdict(float)
        for z in range(problem.Z):
            nodes = [n for n in problem.nodes if assignment.get(n) == z]
            zone_components = self._zone_balance_penalty_components(
                problem, nodes, context
            )
            for key, value in zone_components.items():
                components[key] += value
        return components

    def _zone_constraint_penalty(
        self,
        problem: ZoneProblem,
        nodes: Iterable[int],
        context: _PenaltyContext | None = None,
    ) -> float:
        context = context or self._penalty_context(problem)
        zone_nodes = list(nodes)
        return sum(
            self._zone_balance_penalty_components(
                problem,
                zone_nodes,
                context,
            ).values()
        ) + self._zone_school_count_penalty(problem, zone_nodes, context)

    def _zone_balance_penalty_components(
        self,
        problem: ZoneProblem,
        nodes: list[int],
        context: _PenaltyContext,
    ) -> defaultdict[str, float]:
        components: defaultdict[str, float] = defaultdict(float)
        students = sum(problem.students(n) for n in nodes)
        if students <= _EPS:
            return components

        capacity = sum(problem.capacity(n) for n in nodes)
        lower_capacity = (1.0 - problem.shortage) * students
        upper_capacity = (1.0 + problem.overage) * students
        if capacity < lower_capacity:
            coeff = context.coefficients["shortage"]
            components["shortage"] += coeff * _target_violation_difference(
                value=capacity,
                target=students,
                lower=lower_capacity,
                upper=upper_capacity,
            )
        if capacity > upper_capacity:
            coeff = context.coefficients["overage"]
            components["overage"] += coeff * _target_violation_difference(
                value=capacity,
                target=students,
                lower=lower_capacity,
                upper=upper_capacity,
            )

        frl = sum(problem.frl(n) for n in nodes)
        frl_target = context.district_frl * students
        frl_lower = (context.district_frl - problem.frl_dev) * students
        frl_upper = (context.district_frl + problem.frl_dev) * students
        if frl < frl_lower or frl > frl_upper:
            coeff = context.coefficients["frl"]
            components["frl"] += coeff * _target_violation_difference(
                value=frl,
                target=frl_target,
                lower=frl_lower,
                upper=frl_upper,
            )

        if problem.racial_dev >= 0:
            for ethnicity in problem.ethnicities:
                key = _race_penalty_key(ethnicity)
                target_ratio = context.district_racial[ethnicity]
                value = sum(problem.ethnicity(n, ethnicity) for n in nodes)
                target = target_ratio * students
                lower = (target_ratio - problem.racial_dev) * students
                upper = (target_ratio + problem.racial_dev) * students
                if value < lower or value > upper:
                    coeff = context.coefficients[key]
                    components[key] += coeff * _target_violation_difference(
                        value=value,
                        target=target,
                        lower=lower,
                        upper=upper,
                    )
        return components

    def _balance_penalty(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> float:
        return sum(
            self._balance_penalty_components(
                problem,
                assignment,
                self._penalty_context(problem),
            ).values()
        )

    def _school_count_penalty(
        self,
        problem: ZoneProblem,
        assignment: Mapping[int, int],
        context: _PenaltyContext | None = None,
    ) -> float:
        context = context or self._penalty_context(problem)
        if context.avg_schools <= _EPS:
            return 0.0
        penalty = 0.0
        for z in range(problem.Z):
            nodes = [n for n in problem.nodes if assignment.get(n) == z]
            penalty += self._zone_school_count_penalty(problem, nodes, context)
        return penalty

    def _zone_school_count_penalty(
        self,
        problem: ZoneProblem,
        nodes: Iterable[int],
        context: _PenaltyContext,
    ) -> float:
        if context.avg_schools <= _EPS:
            return 0.0
        schools = sum(problem.num_schools(n) for n in nodes)
        if schools < context.school_lower or schools > context.school_upper:
            return context.coefficients["schools"] * abs(schools - context.avg_schools)
        return 0.0

    def _penalty_context(self, problem: ZoneProblem) -> _PenaltyContext:
        signature = _penalty_context_signature(problem)
        cached = getattr(problem, "_recom_penalty_context", None)
        if cached is not None and cached.signature == signature:
            return cached

        total_students = sum(problem.students(node) for node in problem.nodes)
        avg_students = total_students / max(1, problem.Z)
        total_schools = sum(problem.num_schools(node) for node in problem.nodes)
        avg_schools = total_schools / max(1, problem.Z)
        district_frl = problem.district_frl
        district_racial = problem.district_racial

        references: dict[str, float] = {
            "shortage": avg_students,
            "overage": avg_students,
            "frl": _proportion_reference(avg_students, district_frl),
            "schools": avg_schools,
        }
        if problem.racial_dev >= 0:
            for ethnicity in problem.ethnicities:
                references[_race_penalty_key(ethnicity)] = _proportion_reference(
                    avg_students,
                    district_racial[ethnicity],
                )

        coefficients = {
            key: _penalty_coefficient(reference)
            for key, reference in references.items()
        }
        context = _PenaltyContext(
            signature=signature,
            coefficients=coefficients,
            reference_denominators=references,
            district_racial=district_racial,
            district_frl=district_frl,
            avg_schools=avg_schools,
            school_lower=max(0.0, avg_schools - 1.0),
            school_upper=avg_schools + 1.0,
        )
        setattr(problem, "_recom_penalty_context", context)
        return context


def _valid(score: _Score) -> bool:
    return score.penalty <= _EPS


def _json_penalty_components(
    components: Mapping[str, float] | None,
) -> dict[str, float | None]:
    if not components:
        return {}
    out: dict[str, float | None] = {}
    for key, value in sorted(components.items()):
        number = float(value)
        out[str(key)] = number if math.isfinite(number) else None
    return out


def _penalty_context_signature(problem: ZoneProblem) -> tuple:
    return (
        id(problem.G),
        tuple(problem.nodes),
        tuple(problem.centroids),
        float(problem.frl_dev),
        float(problem.racial_dev),
        float(problem.overage),
        float(problem.shortage),
    )


def _proportion_reference(avg_students: float, target_ratio: float) -> float:
    ratio = min(1.0, max(0.0, float(target_ratio)))
    return float(avg_students) * max(ratio, 1.0 - ratio)


def _penalty_coefficient(reference: float) -> float:
    if not math.isfinite(reference) or reference <= _EPS:
        return 0.0
    return 1.0 / float(reference)


def _target_violation_difference(
    *,
    value: float,
    target: float,
    lower: float,
    upper: float,
) -> float:
    bound_violation = max(lower - value, value - upper, 0.0)
    return max(abs(value - target), bound_violation)


def _race_penalty_key(ethnicity: str) -> str:
    return f"race:{ethnicity}"


def _population_target(
    problem: ZoneProblem, balance_metric: object = "students"
) -> float:
    return recom_balance_target(problem, balance_metric)


def _population_epsilon(
    problem: ZoneProblem,
    population_epsilon: float | None = None,
    balance_metric: object = "students",
) -> float:
    return recom_balance_epsilon(problem, population_epsilon, balance_metric)


def _epsilon_schedule(epsilon: float) -> list[float]:
    values = [epsilon, max(epsilon, 0.10), max(epsilon, 0.25), max(epsilon, 0.50)]
    values.append(max(epsilon, 1.0))
    out = []
    for value in values:
        value = float(value)
        if value not in out:
            out.append(value)
    return out
