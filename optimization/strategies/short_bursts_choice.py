"""Choice-aware short bursts scored by the discrete MID oracle."""

from __future__ import annotations

import math
import time

from optimization.data.mid import build_mid_market, preprocess_mid_market
from optimization.levels import LevelSpec
from optimization.solvers.burst_objectives import MidBatchEvaluator
from optimization.solvers.recom import ShortBurstsSolver
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import BUDGET_ACCOUNTING_MODES, final_value


@register("short_bursts_choice")
class ShortBurstsChoiceStrategy(Strategy):
    """Use ReCom short bursts with exact discrete MID welfare as their score."""

    def run(self, dataset, solver):
        if not isinstance(solver, ShortBurstsSolver):
            raise ValueError("short_bursts_choice requires solver='short_bursts'.")
        if dataset.config.program_population != "All":
            raise ValueError("short_bursts_choice requires program_population='All'.")

        lottery_scale = self.options.get("mid_lottery_scale", 20)
        if isinstance(lottery_scale, bool) or not isinstance(lottery_scale, int):
            raise ValueError("mid_lottery_scale must be a positive integer.")
        if lottery_scale <= 0:
            raise ValueError("mid_lottery_scale must be a positive integer.")
        workers = max(1, int(solver.options.get("workers", 1)))
        total_limit = final_value(
            self.options.get("solve_time_limits"),
            solver.options.get("solve_time_limit", 60.0),
        )
        if not math.isfinite(total_limit) or total_limit < 0:
            raise ValueError("solve time limit must be finite and non-negative.")
        accounting = str(self.options.get("budget_accounting", "wall_clock"))
        if accounting not in BUDGET_ACCOUNTING_MODES:
            raise ValueError(
                "short_bursts_choice budget_accounting must be one of: "
                f"{', '.join(BUDGET_ACCOUNTING_MODES)}."
            )

        target = LevelSpec.parse(self.options["levels"][-1])
        problem = dataset.problem_for(target)
        problem.overage = -1.0
        problem.shortage = -1.0
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))

        preprocessing_start = time.perf_counter()
        market = preprocess_mid_market(
            build_mid_market(problem, dataset.config),
            problem,
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        # Under solver_time accounting the bursts get the whole budget; under
        # wall_clock they get what market preprocessing left behind.
        solver.options["solve_time_limit"] = (
            total_limit
            if accounting == "solver_time"
            else max(0.0, total_limit - preprocessing_seconds)
        )

        scorer = MidBatchEvaluator(market, lottery_scale, workers)
        try:
            solution = solver.solve_with_scorer(
                problem,
                scorer,
                objective_kind="mid_program_welfare",
            )
        finally:
            scorer.close()

        solution.wall_time = float(solution.wall_time or 0.0) + preprocessing_seconds
        solution.metadata.update(
            {
                "strategy": self.name,
                "formulation": "short_bursts_discrete_mid_oracle",
                "initial_welfare": solution.metadata.get("initial_score"),
                "final_welfare": solution.metadata.get("final_score"),
                "mid_lottery_scale": lottery_scale,
                "mid_oracle_type": "finite",
                "mid_preprocessing_seconds": preprocessing_seconds,
                "total_time_limit": total_limit,
                "budget_accounting": accounting,
                "workers": workers,
            }
        )
        return [solution]
