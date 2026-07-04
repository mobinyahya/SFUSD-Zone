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
from dataclasses import dataclass
from functools import partial
from typing import Mapping, TextIO

from gerrychain import Graph, Partition
from gerrychain.proposals import recom as gerrychain_recom
from gerrychain.proposals.tree_proposals import MetagraphError
from gerrychain.tree import (
    BalanceError,
    PopulationBalanceError,
    ReselectException,
    bipartition_tree,
    recursive_tree_part,
)
from gerrychain.updaters import Tally, cut_edges

from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.progress import (
    SolverProgressTracker,
    assignment_tuple,
)
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.balance import balance_constraints
from Zone_Generation.optimization.solvers.base import Solver, register

_EPS = 1e-6
_GERRYCHAIN_ERRORS = (
    BalanceError,
    PopulationBalanceError,
    ReselectException,
    MetagraphError,
    IndexError,
)


@dataclass(frozen=True, order=True)
class _Score:
    penalty: float
    boundary: int


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
        seed = int(self.options.get("seed", 42))
        rng = random.Random(seed)
        time_limit = float(self.options.get("solve_time_limit", 60.0))
        max_iterations = max(0, int(self.options.get("recom_iterations", 1000)))
        cut_attempts = max(1, int(self.options.get("recom_cut_attempts", 100)))
        temperature = max(0.0, float(self.options.get("recom_temperature", 0.0)))
        log_path, progress_log = self._open_progress_log(problem)
        progress = self._new_recom_progress_tracker(problem)

        random_state = random.getstate()
        random.seed(seed)
        try:
            try:
                initial = self._initial_state(problem, cut_attempts)
                current = dict(initial.assignment)
                current_partition = self._partition(problem, current)
                current_score = self._score(problem, current)
                initial_score = current_score
                best = dict(current) if _valid(current_score) else None
                best_score = current_score if best is not None else None
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

                for _ in range(max_iterations):
                    if time.time() - start >= time_limit:
                        break
                    attempted += 1
                    try:
                        proposal_partition = self._gerrychain_proposal(
                            problem, current_partition, cut_attempts
                        )
                    except _GERRYCHAIN_ERRORS as exc:
                        rejected += 1
                        proposal_failures += 1
                        last_proposal_error = type(exc).__name__
                        continue

                    proposal = self._assignment_from_partition(proposal_partition)
                    proposal_score = self._score(problem, proposal)
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
                        best_score is None or proposal_score.boundary < best_score.boundary
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
        if best is not None and best_score is not None:
            status = "FEASIBLE"
            assignment = best
            objective = float(best_score.boundary)
        else:
            status = "UNKNOWN"
            assignment = {}
            objective = None

        metadata = {
            "solver": self.name,
            **self._progress_log_metadata(log_path),
            **self._solver_progress_metadata(progress),
            "initialization_method": initial.metadata.get(
                "initialization_method", self._initialization_method(problem)
            ),
            "iterations": max_iterations,
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "rejected_moves": rejected,
            "proposal_failures": proposal_failures,
            "initial_penalty": initial_score.penalty,
            "best_penalty": best_score.penalty if best_score else current_score.penalty,
            "temperature": temperature,
            **initial.metadata,
        }
        if last_proposal_error is not None:
            metadata["last_proposal_error"] = last_proposal_error
        cache_metadata = getattr(problem, "_math_prog_initial_cache", None)
        if cache_metadata is not None:
            metadata["initial_cache"] = dict(cache_metadata)

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
        }
        if accepted is not None:
            row["accepted"] = bool(accepted)
        if best_score is not None:
            row["best_cut_edges"] = int(best_score.boundary)
            row["best_feasible"] = _valid(best_score)
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

    # ------------------------------------------------------------------ #
    # Initial assignment
    # ------------------------------------------------------------------ #
    def _initial_state(self, problem: ZoneProblem, cut_attempts: int) -> _InitialState:
        if problem.hint:
            return _InitialState(
                assignment=self._complete_assignment(problem, problem.hint),
                metadata={"initialization_method": "hint"},
            )

        method = self._initialization_method(problem)
        if method == "math_prog":
            # The strategy layer materializes math_prog hints. This fallback keeps
            # direct solver calls safe if no dataset was available to do so.
            assignment = self._fallback_initial_assignment(problem)
            return _InitialState(
                assignment=assignment,
                metadata={
                    "initialization_method": "math_prog",
                    "initialization_fallback": "nearest_centroid",
                },
            )

        return self._gerrychain_initial_state(problem, cut_attempts)

    def _gerrychain_initial_state(
        self, problem: ZoneProblem, cut_attempts: int
    ) -> _InitialState:
        target = _population_target(problem)
        epsilon = _population_epsilon(problem)
        if problem.Z < 2 or target <= 0:
            assignment = self._fallback_initial_assignment(problem)
            return _InitialState(
                assignment=assignment,
                metadata={
                    "initialization_method": "gerrychain",
                    "initialization_fallback": "nearest_centroid",
                    "gerrychain_population_target": target,
                    "gerrychain_population_epsilon": epsilon,
                },
            )

        graph = Graph.from_networkx(problem.G)
        tree_method = partial(bipartition_tree, max_attempts=cut_attempts)
        best_assignment = None
        best_score = None
        best_epsilon = epsilon
        errors: list[str] = []
        for attempt, current_epsilon in enumerate(_epsilon_schedule(epsilon), start=1):
            try:
                raw = recursive_tree_part(
                    graph,
                    parts=list(range(problem.Z)),
                    pop_target=target,
                    pop_col="ge_students",
                    epsilon=current_epsilon,
                    method=tree_method,
                )
            except _GERRYCHAIN_ERRORS as exc:
                errors.append(type(exc).__name__)
                continue
            assignment = self._normalize_gerrychain_assignment(problem, raw)
            score = self._score(problem, assignment)
            if best_score is None or score < best_score:
                best_assignment = assignment
                best_score = score
                best_epsilon = current_epsilon
            if _valid(score):
                return _InitialState(
                    assignment=assignment,
                    metadata={
                        "initialization_method": "gerrychain",
                        "gerrychain_initial_attempts": attempt,
                        "gerrychain_population_target": target,
                        "gerrychain_population_epsilon": current_epsilon,
                    },
                )

        if best_assignment is not None:
            return _InitialState(
                assignment=best_assignment,
                metadata={
                    "initialization_method": "gerrychain",
                    "gerrychain_initial_attempts": len(_epsilon_schedule(epsilon)),
                    "gerrychain_population_target": target,
                    "gerrychain_population_epsilon": best_epsilon,
                    "gerrychain_initial_penalty": best_score.penalty if best_score else None,
                },
            )

        assignment = self._fallback_initial_assignment(problem)
        metadata = {
            "initialization_method": "gerrychain",
            "initialization_fallback": "nearest_centroid",
            "gerrychain_population_target": target,
            "gerrychain_population_epsilon": epsilon,
        }
        if errors:
            metadata["gerrychain_initial_errors"] = errors[-3:]
        return _InitialState(assignment=assignment, metadata=metadata)

    def _normalize_gerrychain_assignment(
        self, problem: ZoneProblem, raw: Mapping[int, int]
    ) -> dict[int, int]:
        relabeled = self._relabel_parts_by_centroids(problem, raw)
        completed = self._complete_assignment(problem, relabeled)
        repaired = contiguity.repair(problem.G, completed, problem.centroids)
        return self._complete_assignment(problem, repaired)

    def _relabel_parts_by_centroids(
        self, problem: ZoneProblem, raw: Mapping[int, int]
    ) -> dict[int, int]:
        part_to_zone: dict[int, int] = {}
        used_zones: set[int] = set()
        for z, centroid in enumerate(problem.centroids):
            part = raw.get(centroid)
            if part is None or part in part_to_zone:
                continue
            part_to_zone[int(part)] = z
            used_zones.add(z)

        remaining_zones = [z for z in range(problem.Z) if z not in used_zones]
        remaining_parts = sorted({int(part) for part in raw.values()} - set(part_to_zone))
        for part, zone in zip(remaining_parts, remaining_zones):
            part_to_zone[part] = zone

        if not part_to_zone:
            return {}
        fallback_zone = remaining_zones[0] if remaining_zones else 0
        return {
            int(node): int(part_to_zone.get(int(part), fallback_zone))
            for node, part in raw.items()
        }

    def _fallback_initial_assignment(self, problem: ZoneProblem) -> dict[int, int]:
        assignment = self._complete_assignment(problem, {})
        repaired = contiguity.repair(problem.G, assignment, problem.centroids)
        return self._complete_assignment(problem, repaired)

    def _complete_assignment(
        self, problem: ZoneProblem, seed: Mapping[int, int]
    ) -> dict[int, int]:
        assignment: dict[int, int] = {}
        for node in problem.nodes:
            zone = seed.get(node)
            candidates = problem.candidate_zones(node)
            if zone in candidates:
                assignment[node] = int(zone)
            else:
                assignment[node] = min(
                    candidates,
                    key=lambda z: problem.distance(problem.centroids[z], node),
                )
        for z, centroid in enumerate(problem.centroids):
            assignment[centroid] = z
        return assignment

    def _initialization_method(self, problem: ZoneProblem) -> str:
        if problem.hint:
            return "hint"
        method = str(self.options.get("initialization_method", "gerrychain"))
        if method not in {"gerrychain", "math_prog"}:
            raise ValueError(
                "initialization_method must be one of: gerrychain, math_prog."
            )
        return method

    # ------------------------------------------------------------------ #
    # GerryChain proposals
    # ------------------------------------------------------------------ #
    def _gerrychain_proposal(
        self,
        problem: ZoneProblem,
        partition: Partition,
        cut_attempts: int,
    ) -> Partition:
        return gerrychain_recom(
            partition,
            pop_col="ge_students",
            pop_target=_population_target(problem),
            epsilon=_population_epsilon(problem),
            method=partial(
                bipartition_tree,
                max_attempts=cut_attempts,
                allow_pair_reselection=True,
            ),
        )

    def _partition(self, problem: ZoneProblem, assignment: Mapping[int, int]) -> Partition:
        graph = Graph.from_networkx(problem.G)
        return Partition(
            graph,
            assignment={int(node): int(zone) for node, zone in assignment.items()},
            updaters={
                "population": Tally("ge_students", alias="population"),
                "cut_edges": cut_edges,
            },
        )

    def _assignment_from_partition(self, partition: Partition) -> dict[int, int]:
        return {
            int(node): int(zone)
            for node, zone in partition.assignment.mapping.items()
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
        delta = (proposal.penalty - current.penalty) + (
            proposal.boundary - current.boundary
        )
        if delta <= 0:
            return True
        return rng.random() < math.exp(-delta / temperature)

    # ------------------------------------------------------------------ #
    # Constraint scoring and validation
    # ------------------------------------------------------------------ #
    def _score(self, problem: ZoneProblem, assignment: Mapping[int, int]) -> _Score:
        penalty = 0.0
        hard_penalty = float(problem.A + problem.Z + 1) * 1000.0

        if set(assignment) != set(problem.nodes):
            missing = set(problem.nodes) - set(assignment)
            extra = set(assignment) - set(problem.nodes)
            penalty += hard_penalty * (len(missing) + len(extra))

        for node in problem.nodes:
            zone = assignment.get(node)
            if zone not in problem.candidate_zones(node):
                penalty += hard_penalty

        for z, centroid in enumerate(problem.centroids):
            if assignment.get(centroid) != z:
                penalty += hard_penalty

        if set(assignment) >= set(problem.nodes) and not contiguity.is_contiguous(
            problem.G, dict(assignment), problem.centroids
        ):
            penalty += hard_penalty

        penalty += self._balance_penalty(problem, assignment)
        penalty += self._school_count_penalty(problem, assignment)
        return _Score(
            penalty=penalty,
            boundary=contiguity.boundary_edges(problem.G, dict(assignment)),
        )

    def _balance_penalty(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> float:
        penalty = 0.0
        for z in range(problem.Z):
            nodes = [n for n in problem.nodes if assignment.get(n) == z]
            students = sum(problem.students(n) for n in nodes)
            for constraint in balance_constraints(problem):
                value = sum(constraint.value(n) for n in nodes)
                lower = constraint.lower_ratio * students
                upper = constraint.upper_ratio * students
                if value < lower:
                    penalty += lower - value
                if value > upper:
                    penalty += value - upper
        return penalty

    def _school_count_penalty(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> float:
        total = sum(problem.num_schools(n) for n in problem.nodes)
        if total == 0:
            return 0.0
        avg = total / problem.Z
        lower = max(0.0, avg - 1.0)
        upper = avg + 1.0
        penalty = 0.0
        for z in range(problem.Z):
            schools = sum(
                problem.num_schools(n) for n in problem.nodes if assignment.get(n) == z
            )
            if schools < lower:
                penalty += lower - schools
            if schools > upper:
                penalty += schools - upper
        return penalty


def _valid(score: _Score) -> bool:
    return score.penalty <= _EPS


def _population_target(problem: ZoneProblem) -> float:
    return sum(problem.students(node) for node in problem.nodes) / max(1, problem.Z)


def _population_epsilon(problem: ZoneProblem) -> float:
    tolerances = [problem.shortage, problem.overage, 0.05]
    finite = [float(value) for value in tolerances if math.isfinite(float(value))]
    return max(0.01, min(max(finite) if finite else 1.0, 10.0))


def _epsilon_schedule(epsilon: float) -> list[float]:
    values = [epsilon, max(epsilon, 0.10), max(epsilon, 0.25), max(epsilon, 0.50)]
    values.append(max(epsilon, 1.0))
    out = []
    for value in values:
        value = float(value)
        if value not in out:
            out.append(value)
    return out
