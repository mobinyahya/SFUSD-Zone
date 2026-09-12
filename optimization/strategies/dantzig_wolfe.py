"""ReCom-seeded exact branch-and-price for finite-grid MID welfare."""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace

from optimization.branch_price import branch_and_price
from optimization.data.initial_solutions import FeasibleHintError
from optimization.data.mid import build_mid_market
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.recom import ReComSolver
from optimization.strategies.base import Strategy, register
from optimization.zone_columns import ZonePool, boundary_limit


@register("dantzig_wolfe")
class DantzigWolfeStrategy(Strategy):
    def run(self, dataset, solver):
        if not isinstance(solver, ReComSolver) or solver.name != "recom":
            raise ValueError("dantzig_wolfe requires solver='recom'.")
        if dataset.config.include_citywide:
            raise ValueError("dantzig_wolfe requires include_citywide=false.")
        if self.options.get("budget_accounting", "wall_clock") != "wall_clock":
            raise ValueError("dantzig_wolfe requires budget_accounting='wall_clock'.")
        kind = self.options.get("dw_objective", "mid")
        if kind not in {"mid", "boundary"}:
            raise ValueError("dw_objective must be mid or boundary.")
        if kind == "mid" and dataset.config.program_population != "All":
            raise ValueError("DW MID welfare requires program_population='All'.")
        limits = self.options.get(
            "solve_time_limits", [solver.options.get("solve_time_limit", 60)]
        )
        total = float(limits[-1])
        if math.isnan(total) or total < 0:
            raise ValueError("DW time limit must be nonnegative or infinity.")
        start = time.monotonic()
        deadline = start + total
        problem = dataset.problem_for(LevelSpec.parse(self.options["levels"][-1]))
        problem.boundary_prop = float(self.options.get("boundary_prop", -1))
        market = build_mid_market(problem, dataset.config) if kind == "mid" else None
        pool = ZonePool(problem, market, self.options.get("mid_lottery_scale", 20))
        rng = random.Random(self.options.get("seed", 42))
        target = self.options.get("dw_recom_samples", 500)
        chains = self.options.get("dw_recom_chains", 4)
        samples = set()
        incumbent = ()
        incumbent_score = -math.inf
        sampling_metadata = []
        # Seeding is bounded even with unlimited exact search. Phase I can
        # construct a feasible master when there are no ReCom seeds.
        sampling_deadline = time.monotonic() + min(
            self.options.get("dw_recom_time_limit", 60.0),
            max(0.0, deadline - time.monotonic()) * 0.25,
        )
        for chain in range(chains):
            if time.monotonic() >= sampling_deadline or len(samples) >= target:
                break
            chain_limit = (sampling_deadline - time.monotonic()) / (chains - chain)
            chain_target = len(samples) + max(
                1, (target - len(samples)) // (chains - chain)
            )
            hint = dict(rng.choice(sorted(samples))) if samples else problem.hint
            chain_problem = replace(problem, hint=hint)
            chain_solver = ReComSolver(
                **{
                    **solver.options,
                    "seed": rng.randrange(2**31),
                    "solve_time_limit": chain_limit,
                    "feasible_hint_time_limit": min(
                        float(solver.options.get("feasible_hint_time_limit", 60)),
                        chain_limit,
                    ),
                }
            )

            def visit(assignment):
                nonlocal incumbent, incumbent_score
                key = tuple(sorted(assignment.items()))
                if key not in samples and all(
                    pool.feasible(
                        z, frozenset(n for n, a in assignment.items() if a == z)
                    )
                    for z in range(problem.Z)
                ):
                    columns = pool.add_partition(assignment)
                    samples.add(key)
                    respects_cap = problem.boundary_prop < 0 or sum(
                        c.perimeter for c in columns
                    ) / 2 <= boundary_limit(problem)
                    score = sum(c.score for c in columns)
                    if respects_cap and score > incumbent_score:
                        incumbent, incumbent_score = columns, score
                return (
                    len(samples) < chain_target and time.monotonic() < sampling_deadline
                )

            try:
                sampled = chain_solver.sample_feasible(chain_problem, visit)
                sampling_metadata.append(sampled.metadata)
            except FeasibleHintError as exc:
                sampling_metadata.append(
                    {"stop_reason": "no_feasible_hint", "message": str(exc)}
                )

        seeded_columns = len(pool.columns)
        search = branch_and_price(
            pool,
            deadline=deadline,
            model="grid",
            tolerance=float(self.options.get("tolerance", 1e-6)),
            incumbent=incumbent,
            workers=max(1, int(solver.options.get("workers", 1))),
        )
        incumbent = search.selected
        score = sum(c.score for c in incumbent) if incumbent else None
        assignment = {n: c.zone for c in incumbent for n in c.nodes}
        metadata = {
            "strategy": self.name,
            "solver": "recom",
            "formulation": "whole_zone_branch_and_price",
            "objective_kind": "mid_program_welfare"
            if kind == "mid"
            else "boundary_cost",
            "dw_pricing": "global_mip_gurobi_finite_grid",
            "dw_pricing_models": search.pricing_models,
            "dw_pricing_certified": search.status in {"OPTIMAL", "INFEASIBLE"},
            "dw_global_bound": search.upper_bound,
            "dw_absolute_gap": (
                max(0.0, search.upper_bound - score)
                if score is not None and search.upper_bound is not None
                else None
            ),
            "dw_bound_sense": "upper_bound_on_maximized_score",
            "dw_seed_partitions": len(samples),
            "dw_seed_columns": seeded_columns,
            "dw_columns": len(pool.columns),
            "dw_pricing_columns_added": len(pool.columns) - seeded_columns,
            "dw_branch_nodes": search.nodes,
            "dw_pricing_calls": search.pricing_calls,
            "dw_lp_iterations": search.lp_iterations,
            "dw_history": search.history,
            "dw_sampling": sampling_metadata,
            "dw_selected_columns": [
                {"zone": c.zone, "nodes": sorted(c.nodes), "score": c.score}
                for c in incumbent
            ],
            "dw_citywide_programs_removed": sum(p.citywide for p in market.programs)
            if market
            else 0,
            "dw_contiguity": "connected_unanchored_recom",
            "mid_lottery_scale": self.options.get("mid_lottery_scale", 20),
            "dw_numeric_tolerance": float(self.options.get("tolerance", 1e-6)),
            "stop_reason": search.reason,
            "budget_accounting": "wall_clock",
            "total_time_limit": total if math.isfinite(total) else None,
        }
        return [
            ZoneSolution(
                problem,
                assignment,
                search.status,
                (score if kind == "mid" else -score) if score is not None else None,
                time.monotonic() - start,
                metadata,
            )
        ]
