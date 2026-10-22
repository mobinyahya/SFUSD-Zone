"""Priced-access strategy: cutting planes on a capacity-aware welfare surrogate.

The same loop as ``iterative_choice``
(:func:`~optimization.strategies.cut_loop.run_cut_loop`) -- solve a master with
per-node utility cuts, evaluate the returned zoning, add cuts, repeat -- but the
surrogate being cut is the congestion-priced access bound of
:mod:`choice.priced_access` rather than the MNL logsum.

Two things follow from that swap. The surrogate is a genuine upper bound on
stable-matching welfare for every zoning and every tie-breaking scenario, so the
master's objective certifies rather than merely estimates; and it charges a
congestion price per program, so the master stops being rewarded for zones built
around a single oversubscribed school. The cuts are closed-form, so an iteration
costs one master solve and no recourse LP at all -- the whole per-scenario oracle
that ``saa`` spends its budget on is gone.
"""

from __future__ import annotations

import time
from dataclasses import replace

from choice.models import build_priced_access_choice_model
from optimization.data.dataset import Dataset
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import make_budget
from optimization.strategies.cut_loop import run_cut_loop


def _tightest_bound(
    model_bound: float | None, zoned_bound: float | None
) -> float | None:
    """The smaller of two simultaneously valid welfare upper bounds."""
    candidates = [value for value in (model_bound, zoned_bound) if value is not None]
    return min(candidates) if candidates else None


@register("priced_access")
class PricedAccessStrategy(Strategy):
    reads_assignment_market = True

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        max_iterations = int(self.options.get("max_iterations", 5))
        if max_iterations <= 0:
            raise ValueError("priced_access max_iterations must be positive.")
        cut_levels = int(self.options.get("priced_access_cut_levels", 3))
        price_source = str(self.options.get("priced_access_price_source", "transport"))
        price_scale = float(self.options.get("priced_access_price_scale", 1.0))

        budget, relative_tolerance = make_budget(
            self.options,
            solver.options,
            max_iterations,
            label="priced_access",
        )
        started = time.perf_counter()

        base_problem = self.problem_for(dataset, self.options["levels"][-1])
        model = build_priced_access_choice_model(
            base_problem,
            dataset.config,
            cut_levels=cut_levels,
            price_source=price_source,
            price_scale=price_scale,
            workers=int(self.options.get("zoned_transport_workers", 1)),
            centroid_neighbor_radius=int(
                solver.options.get("centroid_neighbor_radius", 0)
            ),
        )
        lower_bound, upper_bound = model.utility_bounds(base_problem)
        preprocessing_seconds = time.perf_counter() - started

        # Every node needs a cut from the very first solve: the solver gives
        # each one its own utility variable, and an uncut node floats at the
        # loosest node's bound.
        cuts: list = list(model.choice_utility_hint_cuts(base_problem))
        initial_cut_count = len(cuts)

        def stage_metadata(iteration, time_limit, cuts_before):
            return {
                "objective_kind": "priced_access_welfare_upper_bound",
                "priced_access_iteration": iteration,
                "priced_access_cut_levels": cut_levels,
                "priced_access_price_source": price_source,
                "priced_access_price_scale": price_scale,
                "priced_access_cuts_before": cuts_before,
                "priced_access_initial_cuts": initial_cut_count,
                "priced_access_price_constant": model.price_constant,
                "priced_access_zoned_transport_bound": model.welfare_bound,
                **model.bound_metadata,
                "priced_access_master_time_limit_seconds": time_limit,
                "priced_access_preprocessing_seconds": preprocessing_seconds,
            }

        def evaluation_metadata(utility, model_utility, cuts_added, cuts_total):
            # Model against actual on the same scale, both excluding the
            # constant capacity term; add price_constant to either to read it
            # as a welfare bound.
            welfare_bound = (
                None if model_utility is None else model_utility + model.price_constant
            )
            return {
                "priced_access_utility": utility,
                "priced_access_welfare_bound": welfare_bound,
                # Two independently valid bounds on the best attainable
                # sample-average welfare: the master's own outer approximation
                # of the priced bound, and the zone-aware relaxation the prices
                # came from. Report the smaller.
                "priced_access_certified_welfare_bound": _tightest_bound(
                    welfare_bound, model.welfare_bound
                ),
                "priced_access_cuts_added": cuts_added,
                "priced_access_cuts_total": cuts_total,
            }

        # As the process converges the same cut is regenerated every iteration;
        # re-adding it only slows the master down, and dropping duplicates is
        # what makes "no_separation" mean something.
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
            dedupe=True,
        )
        solutions, best = loop.solutions, loop.best

        # The harness scores the last solution returned, but the master's
        # candidate can regress between iterations, so close with a zero-time
        # replay of the best zoning found -- the same convention ``saa`` uses.
        if best is not None and solutions and solutions[-1] is not best:
            solutions.append(
                replace(
                    best,
                    wall_time=0.0,
                    metadata={
                        **best.metadata,
                        "priced_access_replay_of_iteration": best.metadata.get(
                            "priced_access_iteration"
                        ),
                        "priced_access_is_replay": True,
                    },
                )
            )

        if solutions:
            solutions[-1].metadata.update(
                {
                    "priced_access_iteration_count": loop.iterations,
                    "priced_access_termination_reason": loop.termination_reason,
                    "choice_termination_reason": loop.termination_reason,
                    **loop.budget_metadata(budget),
                }
            )
        return solutions
