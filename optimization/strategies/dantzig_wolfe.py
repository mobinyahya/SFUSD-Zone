"""Anchored branch-and-price over whole-zone columns.

The master selects one admissible zone per label so that the zones cover every
vertex exactly once; the subproblem builds one zone at a time from the master's
shadow prices. Both sides read the same feasible set as ``cp_bool`` -- anchored
centroids and closer-neighbour contiguity -- so the decomposition now describes
the district model the rest of the paper solves, rather than an unanchored
relative of it.

Seeding is two-stage, and the stages do different jobs:

1. One ``cp_bool`` feasibility solve. This is the only step that reliably
   produces an *admissible* partition, so it is what makes the master feasible
   without Phase I; the same hint machinery ``saa``, ``mid`` and
   ``stable_cutoff`` use, and cached across runs.
2. ReCom chains started from that hint, rejection-filtered zone by zone. A
   ReCom step rewrites two zones and leaves the rest alone, and its cut
   candidates already respect ``candidate_zones`` -- hence centroid anchoring
   -- so the zones it produces fail admissibility essentially only on
   closer-neighbour support. Rejected zones cost a set-membership test, which
   runs *before* the welfare oracle, so a rejected sample is nearly free.

Phase I remains the fallback: an empty or uncoverable pool is recovered by
pricing feasibility with deficit artificials, so neither seeding stage is
required for correctness.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace

from optimization.branch_price import branch_and_price
from optimization.data.initial_solutions import FeasibleHintError, initial_solution
from optimization.data.mid import build_mid_market
from optimization.data.saa import build_saa_market
from optimization.dw_options import DW_OBJECTIVES
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.recom import ReComSolver
from optimization.strategies.base import Strategy, register
from optimization.zone_columns import ZonePool, boundary_limit
from optimization.zone_family import build_zone_family
from optimization.zone_redraw import ZoneRedraw
from optimization.zone_welfare import (
    BoundaryZoneObjective,
    MidZoneObjective,
    StableMatchingZoneObjective,
)


def build_objective(problem, config, options):
    """The welfare definition this run decomposes."""

    kind = str(options.get("dw_objective", "mid"))
    if kind == "boundary":
        return BoundaryZoneObjective()
    if kind == "mid":
        return MidZoneObjective(
            build_mid_market(problem, config),
            int(options.get("mid_lottery_scale", 20)),
        )
    if kind == "stable_matching":
        return StableMatchingZoneObjective(
            build_saa_market(problem, config),
            tie_breaking_method=str(options.get("saa_tie_breaking_method", "MTB")),
            seed=int(options.get("seed", 42)),
        )
    raise ValueError(f"dw_objective must be one of: {', '.join(DW_OBJECTIVES)}.")


@register("dantzig_wolfe")
class DantzigWolfeStrategy(Strategy):
    def run(self, dataset, solver):
        if getattr(solver, "name", None) != "cp_bool":
            raise ValueError("dantzig_wolfe requires solver='cp_bool'.")
        # Zone values are additive only when every program belongs to exactly
        # one zone. A citywide program is held by every zone at once, so the
        # decomposition's central identity fails; see optimization.zone_welfare.
        if dataset.config.include_citywide_choice_opt:
            raise ValueError(
                "dantzig_wolfe requires include_citywide_choice_opt=false."
            )
        if self.options.get("budget_accounting", "wall_clock") != "wall_clock":
            raise ValueError("dantzig_wolfe requires budget_accounting='wall_clock'.")
        kind = str(self.options.get("dw_objective", "mid"))
        if kind not in DW_OBJECTIVES:
            raise ValueError(
                f"dw_objective must be one of: {', '.join(DW_OBJECTIVES)}."
            )
        if kind != "boundary" and dataset.config.program_population != "All":
            raise ValueError("DW welfare objectives require program_population='All'.")
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
        radius = int(solver.options.get("centroid_neighbor_radius", 0))
        family = build_zone_family(problem, centroid_neighbor_radius=radius)
        objective = build_objective(problem, dataset.config, self.options)
        pool = ZonePool(problem, objective, family=family)

        incumbent: tuple = ()
        incumbent_score = -math.inf

        def offer(columns) -> None:
            nonlocal incumbent, incumbent_score
            if columns is None:
                return
            if problem.boundary_prop >= 0 and sum(
                c.perimeter for c in columns
            ) / 2 > boundary_limit(problem):
                return
            score = sum(c.score for c in columns)
            if score > incumbent_score:
                incumbent, incumbent_score = tuple(columns), score

        hint_metadata: dict = {}
        hint = None
        try:
            hint = initial_solution(
                problem,
                self.options.get("hints", "feasible"),
                solver_options=solver.options,
            )
        except FeasibleHintError as exc:
            hint_metadata = {
                "dw_hint_status": "no_feasible_hint",
                "dw_hint_error": str(exc),
            }
        if hint is not None:
            problem.hint = hint.assignment
            columns = pool.admit_partition(hint.assignment)
            offer(columns)
            hint_metadata = {
                **{
                    key: value
                    for key, value in hint.metadata.items()
                    if key.startswith("hint")
                },
                "dw_hint_admissible": columns is not None,
                "dw_hint_columns": len(pool.columns),
            }

        seeded_from_hint = len(pool.columns)
        sampling = self._sample(pool, solver, problem, deadline, offer)

        # Two-zone redraws, before the search rather than only inside it. A
        # seeded pool holds tilings that differ only where ReCom happened to
        # cut, and the restricted LP's solution set is essentially the tilings
        # the pool contains, so the cheapest thing that makes the master
        # non-trivial is more of them. Under a finite budget this sweep is the
        # load-bearing one: the in-search sweep only fires once a node's LP is
        # fully priced out, which a real instance may not reach.
        redraw = self._redraw(pool, solver)
        redraw_seeding: dict = {}
        if redraw is not None and incumbent:
            _, redraw_seeding = redraw.sweep(
                incumbent,
                deadline=min(
                    deadline,
                    time.monotonic()
                    + float(self.options.get("dw_redraw_time_limit", 60.0)),
                ),
                offer=offer,
            )

        seeded_columns = len(pool.columns)
        search = branch_and_price(
            pool,
            deadline=deadline,
            tolerance=float(self.options.get("tolerance", 1e-6)),
            incumbent=incumbent,
            workers=max(1, int(solver.options.get("workers", 1))),
            seed=int(self.options.get("seed", 42)),
            master_method=str(self.options.get("dw_master_method", "barrier")),
            dual_smoothing=float(self.options.get("dw_dual_smoothing", 1.0)),
            overlap_prop=float(self.options.get("dw_overlap_prop", 0.0)),
            pricing_scale=int(self.options.get("dw_pricing_scale", 1000)),
            pricing_columns_per_call=int(
                self.options.get("dw_pricing_columns_per_call", 8)
            ),
            pricing_parallel=bool(self.options.get("dw_pricing_parallel", True)),
            pricing_time_limit=float(self.options.get("dw_pricing_time_limit", 30.0)),
            redraw=redraw,
            redraw_time_limit=float(self.options.get("dw_redraw_time_limit", 60.0)),
        )
        selected = search.selected
        score = sum(c.score for c in selected) if selected else None
        assignment = {n: c.zone for c in selected for n in c.nodes}
        metadata = {
            "strategy": self.name,
            "solver": "cp_bool",
            "formulation": "anchored_whole_zone_branch_and_price",
            **objective.metadata(),
            **family.metadata(),
            **hint_metadata,
            **sampling,
            "dw_redraw": redraw is not None,
            **({} if redraw is None else redraw.metadata()),
            "dw_redraw_seeding": redraw_seeding or None,
            "dw_master_method": str(self.options.get("dw_master_method", "barrier")),
            "dw_pricing_time_limit": float(
                self.options.get("dw_pricing_time_limit", 30.0)
            ),
            "dw_dual_smoothing": float(self.options.get("dw_dual_smoothing", 1.0)),
            "dw_overlap_prop": float(self.options.get("dw_overlap_prop", 0.0)),
            "dw_pricing_models": search.pricing_models,
            "dw_pricing_certified": search.status in {"OPTIMAL", "INFEASIBLE"},
            "dw_global_bound": search.upper_bound,
            "dw_absolute_gap": (
                max(0.0, search.upper_bound - score)
                if score is not None and search.upper_bound is not None
                else None
            ),
            "dw_bound_sense": "upper_bound_on_maximized_score",
            "dw_hint_seed_columns": seeded_from_hint,
            "dw_seed_columns": seeded_columns,
            "dw_columns": len(pool.columns),
            "dw_pricing_columns_added": search.columns_added,
            "dw_branch_nodes": search.nodes,
            "dw_pricing_calls": search.pricing_calls,
            "dw_lp_iterations": search.lp_iterations,
            "dw_history": search.history,
            "dw_selected_columns": [
                {"zone": c.zone, "nodes": sorted(c.nodes), "score": c.score}
                for c in selected
            ],
            "dw_contiguity": "anchored_closer_neighbor",
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
                # Boundary runs report the cost they minimize, not its negation.
                (score if kind != "boundary" else -score)
                if score is not None
                else None,
                time.monotonic() - start,
                metadata,
            )
        ]

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #
    def _redraw(self, pool, solver) -> ZoneRedraw | None:
        """The two-zone column source, or ``None`` when it is switched off."""

        if not bool(self.options.get("dw_redraw", True)) or pool.problem.Z < 2:
            return None
        return ZoneRedraw(
            pool,
            workers=max(1, int(solver.options.get("workers", 1))),
            seed=int(self.options.get("seed", 42)),
            scale=int(self.options.get("dw_pricing_scale", 1000)),
            splits_per_call=int(self.options.get("dw_redraw_splits_per_call", 8)),
            tolerance=float(self.options.get("tolerance", 1e-6)),
        )

    def _sample(self, pool, solver, problem, deadline, offer) -> dict:
        """Rejection-filtered ReCom sampling, bounded even on an unlimited run."""

        target = int(self.options.get("dw_recom_samples", 500))
        chains = int(self.options.get("dw_recom_chains", 4))
        if target <= 0 or chains <= 0:
            return {"dw_recom_samples_visited": 0, "dw_recom_admissible": 0}

        rng = random.Random(self.options.get("seed", 42))
        sampling_deadline = time.monotonic() + min(
            float(self.options.get("dw_recom_time_limit", 60.0)),
            max(0.0, deadline - time.monotonic()) * 0.25,
        )
        visited: set = set()
        admissible: list = []
        chain_metadata = []
        for chain in range(chains):
            if time.monotonic() >= sampling_deadline or len(visited) >= target:
                break
            chain_limit = (sampling_deadline - time.monotonic()) / (chains - chain)
            chain_target = len(visited) + max(
                1, (target - len(visited)) // (chains - chain)
            )
            hint = (
                dict(rng.choice(admissible))
                if admissible
                else (dict(problem.hint) if problem.hint else None)
            )
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
                key = tuple(sorted(assignment.items()))
                if key not in visited:
                    visited.add(key)
                    # The admissibility test runs before the welfare oracle, so
                    # a rejected sample costs a set-membership walk, not a
                    # market evaluation.
                    columns = pool.admit_partition(assignment)
                    if columns is not None:
                        admissible.append(key)
                        offer(columns)
                return (
                    len(visited) < chain_target and time.monotonic() < sampling_deadline
                )

            try:
                sampled = chain_solver.sample_feasible(chain_problem, visit)
                chain_metadata.append(sampled.metadata)
            except FeasibleHintError as exc:
                chain_metadata.append(
                    {"stop_reason": "no_feasible_hint", "message": str(exc)}
                )
        return {
            "dw_recom_samples_visited": len(visited),
            "dw_recom_admissible": len(admissible),
            "dw_recom_chains_run": len(chain_metadata),
            "dw_sampling": chain_metadata,
        }
