"""Direct finest-level MID welfare strategy."""

from __future__ import annotations

import time

from optimization.data.initial_solutions import initial_solution
from optimization.data.mid import build_mid_market
from optimization.solvers.mid import MidCpSatSolver
from optimization.strategies.base import Strategy, register


@register("mid")
class MidStrategy(Strategy):
    reads_assignment_market = True
    capacity_balance = False

    def run(self, dataset, solver):
        if getattr(solver, "name", None) != "cp_bool":
            raise ValueError("mid requires solver='cp_bool'.")
        if dataset.config.program_population != "All":
            raise ValueError("mid requires program_population='All'.")

        solver = solver.with_options(**self.final_limits(solver))
        problem = self.problem_for(dataset, self.options["levels"][-1])
        hint = initial_solution(
            problem,
            self.options.get("hints", "voronoi"),
            solver_options=solver.options,
        )
        if hint is not None:
            problem.hint = hint.assignment

        start = time.perf_counter()
        market = build_mid_market(problem, dataset.config)
        preprocessing_seconds = time.perf_counter() - start
        mid_solver = MidCpSatSolver(
            market,
            self.options.get("mid_lottery_scale", 20),
            preprocessing_seconds=preprocessing_seconds,
            transport_bounds=self.options.get("mid_transport_bounds", True),
            complementary_slackness=self.options.get(
                "mid_complementary_slackness", False
            ),
            complementary_slackness_slack=self.options.get(
                "mid_complementary_slackness_slack", "auto"
            ),
            **solver.options,
        )
        return [mid_solver.solve(problem)]
