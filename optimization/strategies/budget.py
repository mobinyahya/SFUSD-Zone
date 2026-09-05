"""Wall-clock budgeting shared by the iterative strategies.

An iterative strategy gets one total budget for the whole run rather than one
limit per solve. Every iteration re-reads the clock and claims a linearly
increasing share of what is left, so time an early iteration leaves unused
carries forward to the later, harder solves. Each allocation has a 30-second
floor, capped by the remaining total budget.

``budget_accounting`` chooses what the budget pays for:

``wall_clock``
    Everything after the strategy starts: problem building, market
    preprocessing, warm-start solves, master solves and oracle evaluation.
    Use this when the run itself has a wall-clock cap (a SLURM time limit).
``solver_time``
    Only the iterative master solves. Loading, preprocessing, warm starts and
    oracle/choice-model evaluation are free, so the budget measures the search
    itself and is comparable across instances whose overhead differs.
"""

from __future__ import annotations

import math
import time


BUDGET_POLICY = "linearly_increasing_with_carry_forward"
BUDGET_ACCOUNTING_MODES = ("wall_clock", "solver_time")
MIN_MASTER_SECONDS = 30.0


def final_value(values, default) -> float:
    """Last configured per-level value, falling back to the solver default."""

    return float(values[-1] if values else default)


class Budget:
    """One total time budget shared out over a strategy's iterations.

    Constructing a ``Budget`` starts its ``wall_clock`` clock, so build it at
    the point where the budget should begin paying for work.
    """

    def __init__(
        self,
        total_seconds: float,
        max_iterations: int,
        accounting: str = "wall_clock",
        *,
        label: str = "strategy",
    ) -> None:
        if accounting not in BUDGET_ACCOUNTING_MODES:
            raise ValueError(
                f"{label} budget_accounting must be one of: "
                f"{', '.join(BUDGET_ACCOUNTING_MODES)}."
            )
        if total_seconds < 0 or not math.isfinite(total_seconds):
            raise ValueError(
                f"{label} solve time limit must be finite and non-negative."
            )
        self.total_seconds = float(total_seconds)
        self.max_iterations = int(max_iterations)
        self.accounting = accounting
        self.solve_seconds = 0.0
        self._deadline = time.perf_counter() + self.total_seconds

    @property
    def remaining_seconds(self) -> float:
        """Budget left to share out, under the configured accounting."""

        if self.accounting == "solver_time":
            return self.total_seconds - self.solve_seconds
        return self._deadline - time.perf_counter()

    def exhausted(self) -> bool:
        return self.remaining_seconds <= 0

    def iteration_limit(self, iteration: int) -> float:
        """Share of the remaining budget this iteration's solve may use."""

        return master_time_limit(
            self.remaining_seconds,
            iteration,
            self.max_iterations,
        )

    def charge(self, seconds: float) -> None:
        """Record an iterative solve, the only work ``solver_time`` pays for."""

        self.solve_seconds += float(seconds)

    def metadata(self, prefix: str) -> dict:
        """Time-budget accounting reported on every stage of a run."""

        return {
            f"{prefix}_total_budget_seconds": self.total_seconds,
            f"{prefix}_budget_policy": BUDGET_POLICY,
            f"{prefix}_budget_accounting": self.accounting,
            f"{prefix}_total_master_seconds": self.solve_seconds,
        }


def make_budget(
    options,
    solver_options,
    max_iterations: int,
    *,
    label: str,
) -> tuple[Budget, float]:
    """Build a strategy's budget and read its relative gap tolerance."""

    budget = Budget(
        final_value(
            options.get("solve_time_limits"),
            solver_options.get("solve_time_limit", 60.0),
        ),
        max_iterations,
        str(options.get("budget_accounting", "wall_clock")),
        label=label,
    )
    relative_tolerance = final_value(
        options.get("gap_limits"),
        solver_options.get("relative_gap_limit", 0.0),
    )
    if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
        raise ValueError(f"{label} relative gap limit must be non-negative.")
    return budget, relative_tolerance


def master_time_limit(
    remaining_seconds: float,
    iteration: int,
    max_iterations: int,
) -> float:
    """Weighted share with a 30-second floor, capped by the remaining budget."""

    current_weight = iteration + 1
    remaining_weight = sum(range(current_weight, max_iterations + 1))
    scheduled = remaining_seconds * current_weight / remaining_weight
    return min(remaining_seconds, max(MIN_MASTER_SECONDS, scheduled))
