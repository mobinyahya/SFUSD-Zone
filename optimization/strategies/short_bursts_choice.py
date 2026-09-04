"""Choice-aware short bursts scored by the discrete MID oracle."""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from optimization.data.mid import MidMarket, build_mid_market, preprocess_mid_market
from optimization.levels import LevelSpec
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers.recom import ShortBurstsSolver
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import BUDGET_ACCOUNTING_MODES, final_value


_worker_market: MidMarket | None = None


def _initialize_worker(market: MidMarket) -> None:
    global _worker_market
    _worker_market = market


def _score_in_worker(
    args: tuple[
        tuple[tuple[int, int], ...],
        int,
        dict[str, int] | None,
    ],
) -> tuple[tuple[tuple[int, int], ...], float, dict[str, int]]:
    zoning_items, lottery_scale, warm_cutoffs = args
    if _worker_market is None:
        raise RuntimeError("MID worker was not initialized.")
    result = finite_grid_oracle(
        _worker_market,
        dict(zoning_items),
        lottery_scale,
        check_minimality=False,
        warm_cutoffs=warm_cutoffs,
    )
    return zoning_items, float(result.welfare), dict(result.cutoffs)


class _MidBatchScorer:
    def __init__(self, market: MidMarket, lottery_scale: int, workers: int) -> None:
        self.market = market
        self.lottery_scale = lottery_scale
        self.workers = workers
        self.cache: dict[tuple[tuple[int, int], ...], tuple[float, dict[str, int]]] = {}
        self.executor = self._make_executor() if workers > 1 else None

    def _make_executor(self) -> ProcessPoolExecutor:
        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context()
        return ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(self.market,),
        )

    def __call__(
        self,
        assignments: tuple[dict[int, int] | Any, ...],
        base_assignment: dict[int, int] | Any | None,
    ) -> tuple[float, ...]:
        keys = tuple(tuple(sorted(assignment.items())) for assignment in assignments)
        base_key = (
            tuple(sorted(base_assignment.items()))
            if base_assignment is not None
            else None
        )
        warm_cutoffs = self.cache.get(base_key, (0.0, None))[1]
        uncached = tuple(dict.fromkeys(key for key in keys if key not in self.cache))

        if self.executor is not None and len(uncached) > 1:
            args = tuple((key, self.lottery_scale, warm_cutoffs) for key in uncached)
            for key, welfare, cutoffs in self.executor.map(_score_in_worker, args):
                self.cache[key] = (welfare, cutoffs)
        else:
            for key in uncached:
                result = finite_grid_oracle(
                    self.market,
                    dict(key),
                    self.lottery_scale,
                    check_minimality=False,
                    warm_cutoffs=warm_cutoffs,
                )
                self.cache[key] = (float(result.welfare), dict(result.cutoffs))

        return tuple(self.cache[key][0] for key in keys)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


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

        scorer = _MidBatchScorer(market, lottery_scale, workers)
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
