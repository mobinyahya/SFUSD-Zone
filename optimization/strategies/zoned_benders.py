"""Direct geography-master Benders optimization for analytical Shi welfare."""

from __future__ import annotations

import math
import time

from optimization.analytical_benders import solve_zoned_shi_benders
from optimization.analytical_welfare_oracle import (
    EULER_GAMMA,
    evaluate_zoned_analytical_welfare,
)
from optimization.branch_price.analytical_patterns import (
    AnalyticalPatternValuator,
    validate_zoned_shi_market,
)
from optimization.column_generation_seeds import collect_column_generation_seeds
from optimization.data.cutoffs import build_analytical_welfare_market
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register
from optimization.strategies.zoned_column_generation import (
    NUMERICAL_SCOPE,
    _mechanism_artifact,
)


@register("zoned_benders")
class ZonedBendersStrategy(Strategy):
    """Jointly optimize complete zones and isolated analytical Shi mechanisms."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        started = time.monotonic()
        wall_limit = float(self.options["zoned_benders_wall_time_limit"])
        deadline = started + wall_limit
        optimization_deadline = (
            started + 0.9 * wall_limit
            if self.options.get("zoned_benders_evaluate_stable_diagnostics", True)
            else deadline
        )
        self._validate(dataset, solver)
        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        problem = dataset.problem_for(levels[-1])
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        market = build_analytical_welfare_market(
            dataset,
            problem,
            assignment_config=self.options["cutoff_assignment_config"],
            ctip_path=self.options["cutoff_ctip_path"],
            lottery_scale=int(self.options["cutoff_lottery_scale"]),
            beta=float(self.options["cutoff_gumbel_scale"]),
            remove_city_wide=True,
            outside_systematic_utility=0.0,
        )
        problem.analytical_welfare_market = market
        validate_zoned_shi_market(problem, market)
        radius = int(self.options.get("centroid_neighbor_radius", 0))
        valuator = AnalyticalPatternValuator(
            problem,
            market,
            centroid_neighbor_radius=radius,
            menu_tolerance=float(self.options["zoned_benders_menu_tolerance"]),
            optimality_tolerance=float(
                self.options["zoned_benders_optimality_tolerance"]
            ),
            max_menu_rounds=int(self.options["zoned_benders_menu_max_rounds"]),
        )
        seeds = collect_column_generation_seeds(
            problem,
            solver,
            valuator,
            seed_paths=tuple(self.options.get("zoned_benders_seed_paths") or ()),
            recom_seed_runs=int(self.options["zoned_recom_seed_runs"]),
            local_move_rounds=int(self.options["zoned_benders_local_move_rounds"]),
            centroid_neighbor_radius=radius,
            random_seed=int(self.options.get("seed", 0)),
            workers=int(self.options.get("workers", 1)),
            deadline=min(deadline, optimization_deadline),
        )
        result = solve_zoned_shi_benders(
            problem,
            seeds.patterns,
            seeds.best_assignment,
            valuator=valuator,
            wall_time_limit=max(0.0, deadline - time.monotonic()),
            max_rounds=int(self.options["zoned_benders_max_rounds"]),
            master_time_limit=float(
                self.options["zoned_benders_master_time_limit"]
            ),
            feasibility_tolerance=float(
                self.options["zoned_benders_master_feasibility_tolerance"]
            ),
            optimality_tolerance=float(
                self.options["zoned_benders_optimality_tolerance"]
            ),
            centroid_neighbor_radius=radius,
            workers=int(self.options.get("workers", 1)),
            random_seed=int(self.options.get("seed", 0)),
            deadline=min(deadline, optimization_deadline),
        )
        selected = result.selected_patterns
        objective = sum(pattern.shi_welfare for pattern in selected)
        raw_constant = sum(
            segment.mass * (segment.outside_utility + market.beta * EULER_GAMMA)
            for segment in market.segments
        )
        diagnostics = self._stable_diagnostics(
            market,
            result.assignment,
            deadline=deadline,
        )
        metadata = {
            "solver": getattr(solver, "name", "unknown"),
            "seed_solver": getattr(solver, "name", "unknown"),
            "optimization_engine": "gurobi_logic_based_benders",
            "objective_kind": "analytical_shi_expected_mnl_welfare",
            "objective_normalization": "outside_and_euler_gamma_constant_removed",
            "optimization_method": "direct_geography_analytical_shi_benders",
            "numerical_scope": NUMERICAL_SCOPE,
            "market_coupling": "isolated_zones",
            "shi_normalized_welfare": objective,
            "raw_welfare_constant": raw_constant,
            "raw_expected_welfare": objective + raw_constant,
            "benders_status": result.status,
            "benders_closed": result.closed,
            "benders_termination_reason": result.termination_reason,
            "benders_rounds": result.rounds,
            "benders_master_solves": result.master_solves,
            "benders_subproblem_calls": result.subproblem_calls,
            "benders_price_cuts_added": result.price_cuts_added,
            "benders_point_cuts_added": result.point_cuts_added,
            "benders_master_status": result.master_status,
            "benders_incumbent_objective": result.incumbent_objective,
            "benders_upper_bound": result.upper_bound,
            "incumbent_upper_bound_gap": result.additive_gap,
            "max_subproblem_primal_dual_gap": result.max_recourse_gap,
            "seed_fallback_used": result.seed_fallback_used,
            "global_optimum_certified": False,
            "global_optimum_scope": "solver_reported_floating_numerical_bound",
            "numerical_optimum_within_tolerance": result.closed,
            "q20_welfare": diagnostics.get("q20_welfare"),
            "continuum_stable_welfare": diagnostics.get(
                "continuum_stable_welfare"
            ),
            "stable_diagnostics_status": diagnostics.get("status"),
            "seed_assignment_count": len(seeds.assignments),
            "seed_pattern_count": len(seeds.patterns),
            "seed_provenance": [item.to_dict() for item in seeds.provenance],
            "seed_rejected_count": seeds.rejected_count,
            "market_fingerprint": valuator.market_fingerprint,
            "centroid_school_ids": list(problem.centroid_school_ids),
        }
        artifacts = {}
        if self.options.get("zoned_benders_save_mechanism", True):
            artifact = _mechanism_artifact(problem.level.name, selected)
            artifacts["shi_mechanism"] = artifact
            metadata["mechanism_artifact"] = artifact.filename
            metadata["mechanism_summary"] = artifact.summary
        return [
            ZoneSolution(
                problem=problem,
                assignment=dict(result.assignment),
                status="OPTIMAL" if result.closed else "FEASIBLE",
                objective=objective,
                wall_time=time.monotonic() - started,
                metadata=metadata,
                artifacts=artifacts,
            )
        ]

    def _validate(self, dataset: Dataset, solver: Solver) -> None:
        config = dataset.config
        if list(config.years) != [23]:
            raise ValueError("zoned_benders currently requires years: [23].")
        if config.population_type != "All":
            raise ValueError("zoned_benders requires population_type: 'All'.")
        if not bool(self.options.get("remove_city_wide")):
            raise ValueError("zoned_benders requires remove_city_wide: true.")
        beta = float(self.options["cutoff_gumbel_scale"])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("zoned_benders requires positive finite beta.")
        accepted = {
            "cp_int",
            "cp_bool",
            "mip",
            "recom",
            "relaxed_recom",
            "short_bursts",
        }
        if getattr(solver, "name", None) not in accepted:
            raise ValueError("Unsupported zoned-Benders seed solver.")

    def _stable_diagnostics(self, market, assignment, *, deadline) -> dict[str, object]:
        if not self.options.get("zoned_benders_evaluate_stable_diagnostics", True):
            return {"status": "DISABLED"}
        if time.monotonic() >= deadline:
            return {"status": "SKIPPED_GLOBAL_DEADLINE"}
        try:
            q20 = evaluate_zoned_analytical_welfare(
                market,
                assignment,
                num_zones=max(assignment.values()) + 1,
                cutoff_grid=market.lottery_scale,
                deadline=deadline,
            )
            continuum = evaluate_zoned_analytical_welfare(
                market,
                assignment,
                num_zones=max(assignment.values()) + 1,
                cutoff_grid=None,
                deadline=deadline,
            )
        except (RuntimeError, ValueError, TimeoutError) as exc:
            return {"status": f"ERROR: {exc}"}
        return {
            "status": "COMPUTED",
            "q20_welfare": q20.normalized_welfare,
            "continuum_stable_welfare": continuum.normalized_welfare,
        }
