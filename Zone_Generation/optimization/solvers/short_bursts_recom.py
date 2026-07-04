"""Short-bursts ReCom solver variant."""

from __future__ import annotations

import random
import time

from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.base import register
from Zone_Generation.optimization.solvers.recom import (
    _GERRYCHAIN_ERRORS,
    ReComSolver,
    _valid,
)


@register("short_bursts_recom")
class ShortBurstsReComSolver(ReComSolver):
    """ReCom short-bursts search using GerryChain proposals.

    Each burst follows an unconstrained ReCom random walk for a small number of
    proposals, then restarts the next burst from the lowest-penalty map seen in
    that burst. Feasible maps are tracked separately and selected by cut edges.
    """

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        if problem.choice_objective is not None:
            raise NotImplementedError(
                "short_bursts_recom does not support iterative choice objectives; "
                "use cp_int, cp_bool, or mip."
            )

        start = time.time()
        seed = int(self.options.get("seed", 42))
        time_limit = float(self.options.get("solve_time_limit", 60.0))
        max_iterations = max(0, int(self.options.get("recom_iterations", 1000)))
        cut_attempts = max(1, int(self.options.get("recom_cut_attempts", 100)))
        burst_length = max(1, int(self.options.get("short_bursts_length", 25)))
        log_path, progress_log = self._open_progress_log(problem)

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
                attempted = 0
                accepted = 0
                proposal_failures = 0
                bursts = 0
                selected_improvements = 0
                last_proposal_error = None

                self._write_progress_log(
                    progress_log,
                    start=start,
                    event="initial",
                    iteration=0,
                    score=current_score,
                    best_score=best_score,
                )

                while attempted < max_iterations:
                    if time.time() - start >= time_limit:
                        break
                    bursts += 1
                    burst_best = dict(current)
                    burst_best_partition = current_partition
                    burst_best_score = current_score
                    burst_start_score = current_score
                    walk_partition = current_partition

                    for _ in range(min(burst_length, max_iterations - attempted)):
                        if time.time() - start >= time_limit:
                            break
                        attempted += 1
                        try:
                            proposal_partition = self._gerrychain_proposal(
                                problem, walk_partition, cut_attempts
                            )
                        except _GERRYCHAIN_ERRORS as exc:
                            proposal_failures += 1
                            last_proposal_error = type(exc).__name__
                            continue

                        proposal = self._assignment_from_partition(proposal_partition)
                        proposal_score = self._score(problem, proposal)
                        walk_partition = proposal_partition
                        accepted += 1

                        if proposal_score < burst_best_score:
                            burst_best = dict(proposal)
                            burst_best_partition = proposal_partition
                            burst_best_score = proposal_score

                        if _valid(proposal_score) and (
                            best_score is None
                            or proposal_score.boundary < best_score.boundary
                        ):
                            best = dict(proposal)
                            best_score = proposal_score

                        self._write_progress_log(
                            progress_log,
                            start=start,
                            event="cut",
                            iteration=attempted,
                            score=proposal_score,
                            accepted=True,
                            best_score=best_score,
                        )

                    current = burst_best
                    current_partition = burst_best_partition
                    current_score = burst_best_score
                    if current_score < burst_start_score:
                        selected_improvements += 1
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
            "initialization_method": initial.metadata.get(
                "initialization_method", self._initialization_method(problem)
            ),
            "iterations": max_iterations,
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "rejected_moves": proposal_failures,
            "proposal_failures": proposal_failures,
            "completed_bursts": bursts,
            "selected_burst_improvements": selected_improvements,
            "short_bursts_length": burst_length,
            "short_bursts_score": "constraint_penalty_then_cut_edges",
            "initial_penalty": initial_score.penalty,
            "best_penalty": best_score.penalty if best_score else current_score.penalty,
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
        )
