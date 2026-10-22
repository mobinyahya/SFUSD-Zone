"""The cutting-plane loop ``iterative_choice`` and ``priced_access`` share.

Both solve a master with one utility variable per node, evaluate the returned
zoning against the surrogate the cuts approximate, add the cuts that
evaluation generates, and repeat. Both stop on the same four conditions: the
budget runs out, the master is not feasible, the master objective stops
moving, or an iteration separates nothing. They differ only in the surrogate
model and in what each stage records, which is what :func:`run_cut_loop`
takes as arguments.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from choice.objective import ChoiceObjective
from optimization.data.initial_solutions import normalize_hints
from optimization.solution import ZoneSolution
from optimization.strategies.budget import Budget


def budget_metadata(budget: Budget, evaluation_seconds: float) -> dict:
    """Time-budget accounting shared by every stage of a cut loop."""

    return {
        **budget.metadata("choice"),
        "choice_total_evaluation_seconds": evaluation_seconds,
    }


def cut_key(cut) -> tuple:
    """Identity of a cut, for suppressing exact duplicates across iterations."""

    return (
        cut.node,
        round(cut.constant, 6),
        tuple(sorted((term.node, round(term.coefficient, 6)) for term in cut.terms)),
    )


@dataclass
class CutLoop:
    """What a cut loop produced, for the strategy to finish off."""

    solutions: list[ZoneSolution]
    best: ZoneSolution | None
    termination_reason: str
    iterations: int
    evaluation_seconds: float

    def budget_metadata(self, budget: Budget) -> dict:
        return budget_metadata(budget, self.evaluation_seconds)


def run_cut_loop(
    strategy,
    dataset,
    solver,
    model,
    cuts: list,
    *,
    lower_bound: float,
    upper_bound: float,
    budget: Budget,
    relative_tolerance: float,
    stage_metadata: Callable[[int, float, int], dict],
    evaluation_metadata: Callable[[float, float | None, int, int], dict],
    dedupe: bool = False,
) -> CutLoop:
    """Iterate master solve, evaluation and cut generation until a stop.

    ``stage_metadata(iteration, time_limit, cuts_before)`` is recorded on
    every master solution; ``evaluation_metadata(utility, model_utility,
    cuts_added, cuts_total)`` on every feasible one, after it is evaluated.
    With ``dedupe``, a cut identical to one already in the pool is dropped, so
    ``no_separation`` means no *new* cut rather than no cut at all.
    """

    target = strategy.options["levels"][-1]
    tolerance = float(strategy.options.get("tolerance", 1e-6))
    scale = float(strategy.options.get("choice_utility_scale", 100.0))
    apply_hints = normalize_hints(strategy.options.get("hints", "voronoi")) != "none"
    seen = {cut_key(cut) for cut in cuts} if dedupe else None

    solutions: list[ZoneSolution] = []
    best: ZoneSolution | None = None
    best_utility = float("-inf")
    last_feasible: ZoneSolution | None = None
    previous_model_utility: float | None = None
    termination_reason = "iteration_limit"
    iterations = 0
    evaluation_seconds = 0.0

    for iteration in range(budget.max_iterations):
        if budget.exhausted():
            termination_reason = "time_limit"
            break
        time_limit = budget.iteration_limit(iteration)

        hint_solution = best or last_feasible
        problem = strategy.problem_for(
            dataset,
            target,
            hint=(
                hint_solution.assignment
                if hint_solution is not None and apply_hints
                else None
            ),
            choice_objective=ChoiceObjective(
                cuts=tuple(cuts),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                scale=scale,
                aggregate_cuts=False,
            ),
        )
        master_start = time.perf_counter()
        sol = solver.with_options(
            solve_time_limit=time_limit,
            relative_gap_limit=relative_tolerance,
        ).solve(problem)
        budget.charge(time.perf_counter() - master_start)
        iterations += 1
        sol.metadata.update(stage_metadata(iteration, time_limit, len(cuts)))
        if not sol.feasible:
            sol.metadata.update(budget_metadata(budget, evaluation_seconds))
            solutions.append(sol)
            termination_reason = f"master_{sol.status.lower()}"
            break

        evaluation_start = time.perf_counter()
        evaluated = model.evaluate_with_cuts(problem, sol.assignment)
        evaluation_seconds += time.perf_counter() - evaluation_start
        utility = evaluated.utility
        model_utility = sol.objective
        cuts_to_add = []
        for cut in evaluated.cuts:
            if seen is not None:
                key = cut_key(cut)
                if key in seen:
                    continue
                seen.add(key)
            cuts_to_add.append(cut)
        sol.metadata.update(
            {
                "choice_model_utility": model_utility,
                "choice_model_utility_gap": (
                    model_utility - utility if model_utility is not None else None
                ),
                "choice_utility": utility,
                **evaluation_metadata(
                    utility,
                    model_utility,
                    len(cuts_to_add),
                    len(cuts) + len(cuts_to_add),
                ),
                **budget_metadata(budget, evaluation_seconds),
            }
        )
        solutions.append(sol)
        last_feasible = sol
        if utility > best_utility:
            best_utility = utility
            best = sol

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

    return CutLoop(solutions, best, termination_reason, iterations, evaluation_seconds)
