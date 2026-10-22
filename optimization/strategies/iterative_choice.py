"""Iterative choice strategy.

The strategy solves a zoning model with explicit utility variables, evaluates the
returned zoning against the real choice model, then adds linearized utility cuts
so the next solve has a more accurate objective approximation. The loop itself
is :func:`~optimization.strategies.cut_loop.run_cut_loop`.
"""

from __future__ import annotations

import time

from optimization.data.dataset import Dataset
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import make_budget
from optimization.strategies.choice_model import build_mnl_choice_model
from optimization.strategies.cut_loop import run_cut_loop


@register("iterative_choice")
class IterativeChoiceStrategy(Strategy):
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        max_iterations = int(self.options.get("max_iterations", 5))
        if max_iterations <= 0:
            raise ValueError("iterative_choice max_iterations must be positive.")
        use_choice_utility_hints = bool(self.options.get("choice_utility_hints", False))

        budget, relative_tolerance = make_budget(
            self.options,
            solver.options,
            max_iterations,
            label="iterative_choice",
        )
        started = time.perf_counter()

        model = build_mnl_choice_model(
            dataset.data,
            method=str(self.options.get("choice_model_method", "logsum")),
        )

        base_problem = self.problem_for(dataset, self.options["levels"][-1])
        lower_bound, upper_bound = model.utility_bounds(base_problem)
        # Without choice_utility_hints, the first iteration intentionally has
        # no cuts, matching the legacy unconstrained seed.
        cuts = []
        if use_choice_utility_hints:
            cuts.extend(model.choice_utility_hint_cuts(base_problem))
        hint_cut_count = len(cuts)
        preprocessing_seconds = time.perf_counter() - started

        def stage_metadata(iteration, time_limit, cuts_before):
            metadata = {
                "choice_iteration": iteration,
                "choice_objective_cuts": cuts_before,
                "choice_master_time_limit_seconds": time_limit,
                "choice_preprocessing_seconds": preprocessing_seconds,
            }
            if use_choice_utility_hints:
                metadata["choice_utility_hint_cuts"] = hint_cut_count
            return metadata

        def evaluation_metadata(utility, model_utility, cuts_added, cuts_total):
            return {
                "choice_cuts_added": cuts_added,
                "choice_cuts_total": cuts_total,
            }

        loop = run_cut_loop(
            self,
            dataset,
            solver,
            model,
            cuts,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            budget=budget,
            relative_tolerance=relative_tolerance,
            stage_metadata=stage_metadata,
            evaluation_metadata=evaluation_metadata,
        )
        if loop.solutions:
            loop.solutions[-1].metadata.update(
                {
                    "choice_iteration_count": loop.iterations,
                    "choice_termination_reason": loop.termination_reason,
                    **loop.budget_metadata(budget),
                }
            )
        return loop.solutions
