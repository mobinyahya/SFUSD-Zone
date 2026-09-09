"""Exact single-shot zoning strategy over the sampled matching model.

There is no outer loop here. ``saa`` and ``priced_access`` both cut a surrogate
and re-solve; this strategy hands Gurobi one model that already contains the
matching, so the branch-and-bound tree *is* the search and its dual bound is a
genuine bound on sample-average stable-matching welfare rather than on a
surrogate of it. The shape is therefore ``mid``'s: build the market once, build
one solver, solve once, return one stage.

Two settings are load-bearing.

``hints``
    Default ``voronoi``, but ``feasible`` is what you want on the real
    instance. A ``mid`` run with ``hints: voronoi`` once failed to find *any*
    feasible zoning in 14,400s while every ``hints: feasible`` run returned one
    -- the zoning block's FRL band, school-count band and boundary cap make
    feasibility itself hard enough that a solver spending its whole budget
    looking for a first incumbent is the normal failure mode, not an unlucky
    one. The hint is a complete assignment, so it also warm-starts every
    co-zoning indicator.

``boundary_prop``
    Off by default (``-1``) as in ``mid``. The aggregate capacity penalties are
    disabled outright: ``overage`` and ``shortage`` would otherwise enter the
    objective in units that are not welfare and make the reported bound
    incomparable to every other welfare number in the repo.
"""

from __future__ import annotations

from optimization.data.initial_solutions import initial_solution
from optimization.levels import LevelSpec
from optimization.solvers.stable_cutoff import StableCutoffMipSolver
from optimization.stable_cutoff import build_stable_cutoff_instance
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import final_value


@register("stable_cutoff")
class StableCutoffStrategy(Strategy):
    def run(self, dataset, solver):
        if getattr(solver, "name", None) != "mip":
            raise ValueError("stable_cutoff requires solver='mip'.")
        if dataset.config.program_population != "All":
            raise ValueError("stable_cutoff requires program_population='All'.")

        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        target = levels[-1]
        # One solve, so the per-level limit lists collapse to their last entry.
        solver.options["solve_time_limit"] = final_value(
            self.options.get("solve_time_limits"),
            solver.options.get("solve_time_limit", 60.0),
        )
        solver.options["relative_gap_limit"] = final_value(
            self.options.get("gap_limits"),
            solver.options.get("relative_gap_limit", 0.0),
        )

        problem = dataset.problem_for(target)
        problem.overage = -1.0
        problem.shortage = -1.0
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        hint = initial_solution(
            problem,
            self.options.get("hints", "voronoi"),
            solver_options=solver.options,
        )
        if hint is not None:
            problem.hint = hint.assignment

        instance = build_stable_cutoff_instance(
            problem,
            dataset.config,
            num_seeds=int(self.options.get("stable_cutoff_num_seeds", 3)),
            tie_breaking_method=str(
                self.options.get("stable_cutoff_tie_breaking_method", "STB")
            ),
            base_seed=int(self.options.get("seed", 42)),
        )
        cutoff_solver = StableCutoffMipSolver(
            instance.market,
            instance.samples,
            non_wastefulness=bool(
                self.options.get("stable_cutoff_non_wastefulness", True)
            ),
            aggregate_stability=bool(
                self.options.get("stable_cutoff_aggregate_stability", True)
            ),
            tie_breaking_method=instance.tie_breaking_method,
            preprocessing_seconds=instance.preprocessing_seconds,
            **solver.options,
        )
        solution = cutoff_solver.solve(problem)
        if hint is not None:
            solution.metadata.setdefault("hints", hint.metadata.get("hints"))
        return [solution]
