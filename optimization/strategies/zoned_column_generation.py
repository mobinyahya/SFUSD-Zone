"""Joint complete-zone and analytical Shi mechanism optimization."""

from __future__ import annotations

import math
import time
from optimization.analytical_welfare_oracle import (
    EULER_GAMMA,
    evaluate_zoned_analytical_welfare,
)
from optimization.branch_price.analytical_patterns import (
    AnalyticalPatternValuator,
    validate_zoned_shi_market,
)
from optimization.branch_price.analytical_root import solve_analytical_pattern_root
from optimization.column_generation_seeds import collect_column_generation_seeds
from optimization.data.cutoffs import build_analytical_welfare_market
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import JsonArtifact, ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register

NUMERICAL_SCOPE = "FLOATING_ANALYTICAL_NOT_PROOF_GRADE"


@register("zoned_column_generation")
class ZonedColumnGenerationStrategy(Strategy):
    """Optimize complete zones with independent priority-designed Shi markets."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        started = time.monotonic()
        wall_limit = float(self.options["zoned_cg_wall_time_limit"])
        deadline = started + wall_limit
        optimization_deadline = started + 0.9 * wall_limit
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
        menu_tolerance = float(self.options["zoned_cg_menu_tolerance"])
        valuator = AnalyticalPatternValuator(
            problem,
            market,
            centroid_neighbor_radius=radius,
            menu_tolerance=menu_tolerance,
            optimality_tolerance=float(self.options["zoned_cg_optimality_tolerance"]),
        )
        seeds = collect_column_generation_seeds(
            problem,
            solver,
            valuator,
            seed_paths=tuple(self.options.get("zoned_cg_seed_paths") or ()),
            recom_seed_runs=int(self.options["zoned_recom_seed_runs"]),
            local_move_rounds=int(self.options["zoned_cg_local_move_rounds"]),
            centroid_neighbor_radius=radius,
            random_seed=int(self.options.get("seed", 0)),
            workers=int(self.options.get("workers", 1)),
            deadline=min(deadline, optimization_deadline),
        )
        root = solve_analytical_pattern_root(
            problem,
            seeds.patterns,
            seeds.best_assignment,
            valuator=valuator,
            wall_time_limit=max(0.0, deadline - time.monotonic()),
            max_rounds=int(self.options["zoned_cg_max_rounds"]),
            pricing_time_limit=float(self.options["zoned_cg_pricing_time_limit"]),
            pricing_node_limit=int(self.options["zoned_cg_pricing_node_limit"]),
            columns_per_label=int(self.options["zoned_cg_columns_per_label"]),
            reduced_cost_tolerance=float(
                self.options["zoned_cg_reduced_cost_tolerance"]
            ),
            menu_tolerance=menu_tolerance,
            master_feasibility_tolerance=float(
                self.options["zoned_cg_master_feasibility_tolerance"]
            ),
            optimality_tolerance=float(self.options["zoned_cg_optimality_tolerance"]),
            mip_time_limit=float(self.options["zoned_cg_mip_time_limit"]),
            centroid_neighbor_radius=radius,
            workers=int(self.options.get("workers", 1)),
            random_seed=int(self.options.get("seed", 0)),
            deadline=min(deadline, optimization_deadline),
        )
        selected = tuple(
            valuator.value(pattern.label, pattern.nodes)
            for pattern in root.selected_patterns
        )
        objective = sum(pattern.shi_welfare for pattern in selected)
        raw_constant = sum(
            segment.mass * (segment.outside_utility + market.beta * EULER_GAMMA)
            for segment in market.segments
        )
        optimality_tolerance = float(self.options["zoned_cg_optimality_tolerance"])
        numerical_optimum = root.incumbent_upper_bound_gap <= optimality_tolerance
        status = "OPTIMAL" if numerical_optimum else "FEASIBLE"
        diagnostics = self._stable_diagnostics(
            market,
            root.assignment,
            deadline=deadline,
        )
        final_pricing = root.pricing_results[-problem.Z :]
        metadata = {
            "solver": getattr(solver, "name", "unknown"),
            "objective_kind": "analytical_shi_expected_mnl_welfare",
            "objective_normalization": "outside_and_euler_gamma_constant_removed",
            "optimization_method": "complete_zone_nested_column_generation",
            "numerical_scope": NUMERICAL_SCOPE,
            "market_coupling": "isolated_zones",
            "shi_normalized_welfare": objective,
            "raw_welfare_constant": raw_constant,
            "raw_expected_welfare": objective + raw_constant,
            "root_lp_status": root.root_lp_status,
            "root_lp_closed": root.root_lp_closed,
            "root_lp_objective": root.root_lp_objective,
            "root_lp_upper_bound": root.root_lp_upper_bound,
            "root_lp_integral": root.root_lp_integral,
            "root_lp_additive_gap": root.root_lp_additive_gap,
            "root_lp_rounds": root.rounds,
            "max_pricing_upper_bound_reduced_cost": (
                root.max_pricing_upper_bound_reduced_cost
            ),
            "max_menu_residual_bound": max(
                (result.menu_residual_bound for result in root.pricing_results),
                default=0.0,
            ),
            "menu_tolerance_contribution_per_label": menu_tolerance
            * sum(segment.mass for segment in market.segments),
            "menu_tolerance_contribution_outer_total": menu_tolerance
            * sum(segment.mass for segment in market.segments)
            * problem.Z,
            "outer_tolerance_contribution": float(
                self.options["zoned_cg_reduced_cost_tolerance"]
            )
            * problem.Z,
            "final_pricing_diagnostics": [
                {
                    "label": result.label,
                    "status": result.status,
                    "pricing_upper_bound": result.pricing_upper_bound,
                    "reduced_cost_upper_bound": result.reduced_cost_upper_bound,
                    "menu_residual_bound": result.menu_residual_bound,
                    "closure_reason": result.closure_reason,
                    "fallbacks": [
                        {
                            "kind": fallback.kind,
                            "status": fallback.status,
                            "upper_bound": fallback.upper_bound,
                            "timing_seconds": fallback.timing_seconds,
                        }
                        for fallback in result.fallbacks
                    ],
                }
                for result in final_pricing
            ],
            "pricing_calls": root.pricing_calls,
            "pricing_status_counts": root.pricing_status_counts or {},
            "column_count": len(root.patterns),
            "seed_pattern_count": len(seeds.patterns),
            "restricted_mip_status": root.restricted_mip_status,
            "restricted_mip_objective": objective,
            "restricted_mip_seed_fallback_used": root.seed_fallback_used,
            "incumbent_upper_bound_gap": root.incumbent_upper_bound_gap,
            "global_optimum_certified": False,
            "global_optimum_scope": "solver_reported_floating_numerical_bound",
            "numerical_optimum_within_tolerance": numerical_optimum,
            "q20_welfare": diagnostics.get("q20_welfare"),
            "continuum_stable_welfare": diagnostics.get("continuum_stable_welfare"),
            "stable_diagnostics_status": diagnostics.get("status"),
            "seed_provenance": [item.to_dict() for item in seeds.provenance],
            "seed_rejected_count": seeds.rejected_count,
            "market_fingerprint": valuator.market_fingerprint,
            "centroid_school_ids": list(problem.centroid_school_ids),
        }
        artifacts = {}
        if self.options.get("zoned_cg_save_mechanism", True):
            artifact = _mechanism_artifact(problem.level.name, selected)
            artifacts["shi_mechanism"] = artifact
            metadata["mechanism_artifact"] = artifact.filename
            metadata["mechanism_summary"] = artifact.summary
        return [
            ZoneSolution(
                problem=problem,
                assignment=dict(root.assignment),
                status=status,
                objective=objective,
                wall_time=time.monotonic() - started,
                metadata=metadata,
                artifacts=artifacts,
            )
        ]

    def _validate(self, dataset: Dataset, solver: Solver) -> None:
        config = dataset.config
        if list(config.years) != [23]:
            raise ValueError("zoned_column_generation currently requires years: [23].")
        if config.population_type != "All":
            raise ValueError("zoned_column_generation requires population_type: 'All'.")
        if not bool(self.options.get("remove_city_wide")):
            raise ValueError("zoned_column_generation requires remove_city_wide: true.")
        beta = float(self.options["cutoff_gumbel_scale"])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("zoned_column_generation requires positive finite beta.")
        accepted = {
            "cp_int",
            "cp_bool",
            "mip",
            "recom",
            "relaxed_recom",
            "short_bursts",
        }
        if getattr(solver, "name", None) not in accepted:
            raise ValueError("Unsupported zoned-column-generation seed solver.")

    def _stable_diagnostics(
        self,
        market,
        assignment,
        *,
        deadline: float,
    ) -> dict[str, object]:
        if not self.options.get("zoned_cg_evaluate_stable_diagnostics", True):
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


def _mechanism_artifact(level: str, selected_patterns) -> JsonArtifact:
    zones = []
    nonzero_menus = 0
    for pattern in sorted(selected_patterns, key=lambda item: item.label):
        mechanism = pattern.mechanism
        if mechanism is None:
            raise RuntimeError("Selected analytical pattern has no mechanism witness.")
        segments = []
        for segment_id, entries in sorted(mechanism.menu_probabilities.items()):
            menus = [
                {"schools": list(menu), "probability": probability}
                for menu, probability in entries
                if probability > 1e-12
            ]
            nonzero_menus += len(menus)
            segments.append({"segment_id": segment_id, "menus": menus})
        zones.append(
            {
                "label": pattern.label,
                "nodes": sorted(pattern.nodes),
                "shi_welfare": pattern.shi_welfare,
                "segments": segments,
                "quotas": [
                    {"school_id": school, "quota": quota}
                    for school, quota in sorted(mechanism.quotas.items())
                ],
                "school_prices": [
                    {"school_id": school, "price": price}
                    for school, price in sorted(mechanism.school_prices.items())
                ],
                "type_potentials": [
                    {"segment_id": segment_id, "potential": potential}
                    for segment_id, potential in sorted(
                        mechanism.type_potentials.items()
                    )
                ],
                "max_menu_residual": mechanism.max_pricing_violation,
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "analytical_shi_continuum_mechanism",
        "numerical_scope": NUMERICAL_SCOPE,
        "continuum_large_market_witness": True,
        "finite_student_exact": False,
        "hard_eligibility": True,
        "fractional_quotas": True,
        "minimum_admissibility_threshold": 1,
        "priority_rule": (
            "For a segment-t agent, draw menu S from its saved distribution and "
            "d uniformly on (0,1); eligible school j receives score 1[j in S] + d."
        ),
        "implementation_scope": (
            "Continuum/large-market DA witness with hard eligibility and threshold "
            "one; not an exact finite-student assignment."
        ),
        "zones": zones,
    }
    return JsonArtifact(
        filename=f"artifacts/shi_mechanism_{level}.json",
        summary={
            "schema_version": 1,
            "kind": "analytical_shi_continuum_mechanism",
            "zone_count": len(zones),
            "nonzero_menu_count": nonzero_menus,
            "finite_student_exact": False,
        },
        payload=payload,
    )
