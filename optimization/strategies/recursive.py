"""Recursive strategy: solve coarse, then refine at each finer level.

For each level in ``levels`` (coarse->fine, within one configured Census unit)
the previous solution is projected onto the current graph via
:class:`LevelConverter`, used as a warm-start hint, and used to narrow each
node's candidate zones to those seen near zone boundaries -- so the finer solve
only re-decides the borders. Per-level time and gap limits are applied to the
solver before each solve, with optional carry-over of unused solve time from one
level to the next.
"""

from __future__ import annotations

from optimization.data import contiguity
from optimization.data.conversion import LevelConverter
from optimization.data.dataset import Dataset
from optimization.data.initial_solutions import (
    initial_solution,
    normalize_hints,
)
from optimization.levels import LevelSpec
from optimization.problem import DuplicateCentroidError
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register


@register("recursive")
class RecursiveStrategy(Strategy):
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        time_limits = self.options.get("solve_time_limits")
        carry_over_compute = bool(self.options.get("carry_over_compute", False))
        gap_limits = self.options.get("gap_limits")
        hints = normalize_hints(self.options.get("hints", "voronoi"))
        apply_hints = hints != "none"
        looseness = float(self.options.get("looseness", 1.0))
        if looseness < 1.0:
            raise ValueError("looseness must be >= 1.0 for recursive runs.")
        radius = self.options.get("boundary_radius", 1)
        converter = LevelConverter(data=dataset.data)
        default_time_limit = solver.options.get("solve_time_limit")
        carry_over_time = 0.0

        solutions: list[ZoneSolution] = []
        prev: ZoneSolution | None = None
        prev_level: LevelSpec | None = None

        for i, level in enumerate(levels):
            configured_time_limit = self._configured_time_limit(
                time_limits, i, default_time_limit
            )
            carry_over_time_received = carry_over_time if carry_over_compute else 0.0
            effective_time_limit = self._effective_time_limit(
                configured_time_limit,
                carry_over_time_received,
                carry_over_compute,
            )
            self._apply_limits(solver, effective_time_limit, gap_limits, i)
            constraint_multiplier = self._constraint_multiplier(looseness, levels, i)

            if prev is None or not prev.assignment:
                problem = dataset.problem_for(
                    level,
                    constraint_multiplier=constraint_multiplier,
                )
            else:
                dst_G = dataset.graph_for(level)
                centroids = dataset.centroids_for(level)
                projected = converter.between(
                    prev.problem.G, prev.assignment, prev_level, dst_G, level
                )
                for z, centroid in enumerate(centroids):
                    projected[centroid] = z
                dataset.closer_neighbors_for(level)
                candidates = contiguity.boundary_candidates(
                    dst_G, projected, centroids, radius=radius
                )
                candidates = contiguity.relax_unsupported_candidates(
                    dst_G,
                    projected,
                    centroids,
                    prev.problem.centroid_school_ids,
                    candidates,
                )
                problem = dataset.problem_for(
                    level,
                    candidates=candidates,
                    hint=projected if apply_hints else None,
                    constraint_multiplier=constraint_multiplier,
                )

            _add_math_programming_initial_hint(problem, solver, hints)

            try:
                sol = solver.solve(problem)
            except DuplicateCentroidError as exc:
                if i + 1 == len(levels):
                    raise
                sol = self._skipped_duplicate_centroid_solution(problem, exc)
            unused_time = 0.0
            if carry_over_compute and i + 1 < len(levels):
                unused_time = self._unused_time(effective_time_limit, sol.wall_time)
            carry_over_time = unused_time
            if carry_over_compute:
                sol.metadata.update(
                    {
                        "configured_time_limit_seconds": configured_time_limit,
                        "carry_over_time_received_seconds": carry_over_time_received,
                        "effective_time_limit_seconds": effective_time_limit,
                        "unused_time_carried_forward_seconds": unused_time,
                    }
                )
            solutions.append(sol)
            if sol.status != "SKIPPED":
                prev, prev_level = sol, level

        return solutions

    @staticmethod
    def _skipped_duplicate_centroid_solution(
        problem, exc: DuplicateCentroidError
    ) -> ZoneSolution:
        return ZoneSolution(
            problem=problem,
            assignment={},
            status="SKIPPED",
            objective=None,
            wall_time=0.0,
            metadata={
                "skip_reason": "duplicate_centroid",
                "error_message": str(exc),
                "duplicate_centroid_node": exc.node,
                "duplicate_centroid_zones": list(exc.zones),
            },
        )

    @staticmethod
    def _configured_time_limit(time_limits, i, default_time_limit):
        if time_limits:
            idx = min(i, len(time_limits) - 1)
            return float(time_limits[idx])
        if default_time_limit is not None:
            return float(default_time_limit)
        return None

    @staticmethod
    def _effective_time_limit(configured_time_limit, carry_over_time, enabled):
        if configured_time_limit is None:
            return None
        if not enabled:
            return configured_time_limit
        return configured_time_limit + carry_over_time

    @staticmethod
    def _unused_time(effective_time_limit, wall_time):
        if effective_time_limit is None or wall_time is None:
            return 0.0
        return max(0.0, float(effective_time_limit) - float(wall_time))

    @staticmethod
    def _apply_limits(solver, time_limit, gap_limits, i):
        if time_limit is not None:
            solver.options["solve_time_limit"] = time_limit
        if gap_limits and i < len(gap_limits):
            solver.options["relative_gap_limit"] = gap_limits[i]

    @staticmethod
    def _constraint_multiplier(looseness, levels, i):
        return float(looseness) ** (len(levels) - i - 1)


def _add_math_programming_initial_hint(problem, solver: Solver, hints: str) -> None:
    if getattr(solver, "name", None) not in {"cp_int", "cp_bool", "mip"}:
        return
    if problem.hint is not None:
        return
    initial = initial_solution(problem, hints, solver_options=solver.options)
    if initial is not None:
        problem.hint = initial.assignment
