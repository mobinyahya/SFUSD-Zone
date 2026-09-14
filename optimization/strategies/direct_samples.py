"""Exact single-shot zoning strategy over the sampled access polytope.

There is no outer loop here. ``saa`` draws the same seeds, solves one recourse
LP per seed at the candidate zoning, and feeds the master a dual cut; this
strategy hands CP-SAT one model that already contains every seed's matching, so
the branch-and-bound tree *is* the search. Two things follow. The objective is
sample-average stable-matching welfare rather than a surrogate of it, so the
dual bound bounds the quantity of interest and no a-priori ``Welfare_max`` is
needed. And the incumbent-selection problem ``saa`` has -- its oracle
over-reports realized deferred acceptance by 195 to 252 units, and the variation
across zonings can reorder two candidates -- does not arise, because the model
is exact at every fixed zoning. See :mod:`optimization.direct_samples` for why
Boolean assignment variables remove the need for comb separation.

The shape is ``mid``'s and ``stable_cutoff``'s: build the market once, build one
solver, solve once, return one stage.

Settings
--------
``saa_num_seeds``, ``saa_tie_breaking_method``
    The same draws ``saa`` would make, so the two strategies optimize the same
    sample at the same ``seed``. Each seed costs a full matching block --
    ``|Gamma|`` Booleans and ``O(|Gamma|)`` rows -- so the seed count is the
    single knob that decides whether the model fits.

``hints``
    Default ``voronoi``, but ``feasible`` is what you want on the real
    instance: the zoning block's FRL band, school-count band and boundary cap
    make feasibility itself hard, and a solver spending its whole budget
    looking for a first incumbent is the normal failure mode. A complete hint
    also warm-starts the matching blocks, which the solver fills in by replaying
    deferred acceptance rather than leaving to search.

``boundary_prop``
    Off by default (``-1``) as in ``mid``. The aggregate capacity penalties are
    disabled outright: ``overage`` and ``shortage`` would otherwise enter the
    objective in units that are not welfare and make the reported bound
    incomparable to every other welfare number in the repo.
"""

from __future__ import annotations

import time

from optimization.data.initial_solutions import initial_solution
from optimization.data.saa import (
    build_saa_market,
    saa_market_to_mid_market,
    sample_school_preferences,
)
from optimization.levels import LevelSpec
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers.direct_samples import DirectSamplesCpSatSolver
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import final_value


@register("direct_samples")
class DirectSamplesStrategy(Strategy):
    def run(self, dataset, solver):
        if getattr(solver, "name", None) != "cp_bool":
            raise ValueError("direct_samples requires solver='cp_bool'.")
        if dataset.config.program_population != "All":
            raise ValueError("direct_samples requires program_population='All'.")

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
        lottery_scale = int(self.options.get("mid_lottery_scale", 20))
        if lottery_scale <= 0:
            raise ValueError("mid_lottery_scale must be a positive integer.")

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

        start = time.perf_counter()
        # `build_saa_market` preprocesses before returning, and the draws index
        # programs by position, so the market has to be built first.
        market = build_saa_market(problem, dataset.config)
        tie_breaking_method = str(
            self.options.get("saa_tie_breaking_method", "MTB")
        ).upper()
        samples = sample_school_preferences(
            market,
            int(self.options.get("saa_num_seeds", 5)),
            tie_breaking_method,
            int(self.options.get("seed", 42)),
        )
        preprocessing_seconds = time.perf_counter() - start

        matching_solver = DirectSamplesCpSatSolver(
            market,
            samples,
            tie_breaking_method=tie_breaking_method,
            preprocessing_seconds=preprocessing_seconds,
            **solver.options,
        )
        solution = matching_solver.solve(problem)
        if hint is not None:
            solution.metadata.setdefault("hints", hint.metadata.get("hints"))
        if not solution.feasible:
            return [solution]

        # Budget welfare, the one measure computed identically for every
        # strategy. This model's own objective is a sample average over drawn
        # lotteries, which is not what `mid`, `saa` or `priced_access` report,
        # so score the chosen zoning through the MID oracle as they do.
        reference_start = time.perf_counter()
        reference = finite_grid_oracle(
            saa_market_to_mid_market(market),
            solution.assignment,
            lottery_scale,
            check_minimality=False,
        )
        solution.metadata.update(
            {
                "mid_welfare": reference.welfare,
                "mid_discrete_welfare": reference.welfare,
                "mid_discrete_cutoffs": dict(reference.cutoffs),
                "mid_lottery_scale": lottery_scale,
                "mid_oracle_type": "finite",
                "direct_samples_reference_oracle_seconds": (
                    time.perf_counter() - reference_start
                ),
            }
        )
        return [solution]
