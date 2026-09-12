"""Priced-access strategy: cutting planes on a capacity-aware welfare surrogate.

Same loop shape as ``iterative_choice`` -- solve a master with per-node utility
cuts, evaluate the returned zoning, add cuts, repeat -- but the surrogate being
cut is the congestion-priced access bound of :mod:`choice.priced_access` rather
than the MNL logsum.

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
from choice.objective import ChoiceObjective
from optimization.data.initial_solutions import normalize_hints
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import Budget, make_budget


def _cut_key(cut) -> tuple:
    """Identity of a cut, for suppressing exact duplicates across iterations."""
    return (
        cut.node,
        round(cut.constant, 6),
        tuple(sorted((term.node, round(term.coefficient, 6)) for term in cut.terms)),
    )


def _tightest_bound(
    model_bound: float | None, zoned_bound: float | None
) -> float | None:
    """The smaller of two simultaneously valid welfare upper bounds."""
    candidates = [
        value for value in (model_bound, zoned_bound) if value is not None
    ]
    return min(candidates) if candidates else None


def _budget_metadata(budget: Budget, evaluation_seconds: float) -> dict:
    return {
        **budget.metadata("choice"),
        "choice_total_evaluation_seconds": evaluation_seconds,
    }


@register("priced_access")
class PricedAccessStrategy(Strategy):
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        target = levels[-1]
        max_iterations = int(self.options.get("max_iterations", 5))
        tolerance = float(self.options.get("tolerance", 1e-6))
        if max_iterations <= 0:
            raise ValueError("priced_access max_iterations must be positive.")
        apply_hints = normalize_hints(self.options.get("hints", "voronoi")) != "none"
        scale = float(self.options.get("choice_utility_scale", 100.0))
        cut_levels = int(self.options.get("priced_access_cut_levels", 3))

        budget, relative_tolerance = make_budget(
            self.options,
            solver.options,
            max_iterations,
            label="priced_access",
        )
        started = time.perf_counter()
        evaluation_seconds = 0.0

        base_problem = dataset.problem_for(target)
        base_problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        model = build_priced_access_choice_model(
            base_problem,
            dataset.config,
            cut_levels=cut_levels,
            price_source=str(
                self.options.get("priced_access_price_source", "transport")
            ),
            price_scale=float(self.options.get("priced_access_price_scale", 1.0)),
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
        seen_cuts = {_cut_key(cut) for cut in cuts}
        initial_cut_count = len(cuts)
        solutions: list[ZoneSolution] = []
        best_solution: ZoneSolution | None = None
        best_utility = float("-inf")
        last_feasible: ZoneSolution | None = None
        previous_model_utility: float | None = None
        termination_reason = "iteration_limit"
        iterations_completed = 0

        for iteration in range(max_iterations):
            if budget.exhausted():
                termination_reason = "time_limit"
                break
            iteration_time_limit = budget.iteration_limit(iteration)

            choice_objective = ChoiceObjective(
                cuts=tuple(cuts),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                scale=scale,
                aggregate_cuts=False,
            )
            hint_solution = best_solution or last_feasible
            hint = (
                hint_solution.assignment
                if hint_solution is not None and apply_hints
                else None
            )
            problem = dataset.problem_for(
                target,
                hint=hint,
                choice_objective=choice_objective,
            )
            problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
            solver.options["solve_time_limit"] = iteration_time_limit
            solver.options["relative_gap_limit"] = relative_tolerance
            master_start = time.perf_counter()
            sol = solver.solve(problem)
            budget.charge(time.perf_counter() - master_start)
            iterations_completed += 1
            sol.metadata.update(
                {
                    "objective_kind": "priced_access_welfare_upper_bound",
                    "priced_access_iteration": iteration,
                    "priced_access_cut_levels": cut_levels,
                    "priced_access_price_source": str(
                        self.options.get("priced_access_price_source", "transport")
                    ),
                    "priced_access_price_scale": float(
                        self.options.get("priced_access_price_scale", 1.0)
                    ),
                    "priced_access_cuts_before": len(cuts),
                    "priced_access_initial_cuts": initial_cut_count,
                    "priced_access_price_constant": model.price_constant,
                    "priced_access_zoned_transport_bound": model.welfare_bound,
                    **model.bound_metadata,
                    "priced_access_master_time_limit_seconds": iteration_time_limit,
                    "priced_access_preprocessing_seconds": preprocessing_seconds,
                }
            )
            if not sol.feasible:
                sol.metadata.update(_budget_metadata(budget, evaluation_seconds))
                solutions.append(sol)
                termination_reason = f"master_{sol.status.lower()}"
                break

            evaluation_start = time.perf_counter()
            evaluated = model.evaluate_with_cuts(problem, sol.assignment)
            evaluation_seconds += time.perf_counter() - evaluation_start
            utility = evaluated.utility
            model_utility = sol.objective
            # As the process converges the same cut is regenerated every
            # iteration; re-adding it only slows the master down, and dropping
            # duplicates is what makes "no_separation" mean something.
            cuts_to_add = []
            for cut in evaluated.cuts:
                key = _cut_key(cut)
                if key in seen_cuts:
                    continue
                seen_cuts.add(key)
                cuts_to_add.append(cut)
            sol.metadata.update(
                {
                    # Model against actual on the same scale, both excluding the
                    # constant capacity term; add price_constant to either to
                    # read it as a welfare bound.
                    "choice_model_utility": model_utility,
                    "choice_model_utility_gap": (
                        model_utility - utility if model_utility is not None else None
                    ),
                    "choice_utility": utility,
                    "priced_access_utility": utility,
                    "priced_access_welfare_bound": (
                        None
                        if model_utility is None
                        else model_utility + model.price_constant
                    ),
                    # Two independently valid bounds on the best attainable
                    # sample-average welfare: the master's own outer
                    # approximation of the priced bound, and the zone-aware
                    # relaxation the prices came from. Report the smaller.
                    "priced_access_certified_welfare_bound": _tightest_bound(
                        None
                        if model_utility is None
                        else model_utility + model.price_constant,
                        model.welfare_bound,
                    ),
                    "priced_access_cuts_added": len(cuts_to_add),
                    "priced_access_cuts_total": len(cuts) + len(cuts_to_add),
                    **_budget_metadata(budget, evaluation_seconds),
                }
            )
            solutions.append(sol)
            last_feasible = sol
            if utility > best_utility:
                best_utility = utility
                best_solution = sol

            if iteration > 0:
                if model_utility is None or previous_model_utility is None:
                    termination_reason = "missing_objective"
                    break
                model_utility_change = abs(model_utility - previous_model_utility)
                sol.metadata["choice_model_utility_change"] = model_utility_change
                if model_utility_change <= tolerance:
                    termination_reason = "objective_change"
                    break
            previous_model_utility = model_utility

            cuts.extend(cuts_to_add)
            if not cuts_to_add:
                termination_reason = "no_separation"
                break

        # The harness scores the last solution returned, but the master's
        # candidate can regress between iterations, so close with a zero-time
        # replay of the best zoning found -- the same convention ``saa`` uses.
        if (
            best_solution is not None
            and solutions
            and solutions[-1] is not best_solution
        ):
            replay = replace(
                best_solution,
                wall_time=0.0,
                metadata={
                    **best_solution.metadata,
                    "priced_access_replay_of_iteration": best_solution.metadata.get(
                        "priced_access_iteration"
                    ),
                    "priced_access_is_replay": True,
                },
            )
            solutions.append(replay)

        if solutions:
            solutions[-1].metadata.update(
                {
                    "priced_access_iteration_count": iterations_completed,
                    "priced_access_termination_reason": termination_reason,
                    "choice_termination_reason": termination_reason,
                    **_budget_metadata(budget, evaluation_seconds),
                }
            )
        return solutions
