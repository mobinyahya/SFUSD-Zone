"""Direct finest-level MID welfare strategy."""

from __future__ import annotations

import time

from optimization.data.initial_solutions import initial_solution
from optimization.data.mid import build_mid_market
from optimization.levels import LevelSpec
from optimization.solvers.mid import MidCpSatSolver
from optimization.strategies.base import Strategy, register


@register("mid")
class MidStrategy(Strategy):
    def run(self, dataset, solver):
        if getattr(solver, "name", None) != "cp_bool":
            raise ValueError("mid requires solver='cp_bool'.")
        if dataset.config.program_population != "All":
            raise ValueError("mid requires program_population='All'.")

        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        target = levels[-1]
        _apply_final_limit(solver.options, self.options.get("solve_time_limits"))
        _apply_final_limit(
            solver.options,
            self.options.get("gap_limits"),
            key="relative_gap_limit",
        )

        problem = dataset.problem_for(target)
        problem.overage = -1.0
        problem.shortage = -1.0
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        hint = initial_solution(problem, self.options.get("hints", "voronoi"))
        if hint is not None:
            problem.hint = hint.assignment

        start = time.perf_counter()
        market = build_mid_market(problem, dataset.config)
        preprocessing_seconds = time.perf_counter() - start
        mid_solver = MidCpSatSolver(
            market,
            self.options.get("mid_lottery_scale", 20),
            preprocessing_seconds=preprocessing_seconds,
            **solver.options,
        )
        return [mid_solver.solve(problem)]


def _apply_final_limit(options: dict, values, *, key: str = "solve_time_limit") -> None:
    if values:
        options[key] = float(values[-1])
