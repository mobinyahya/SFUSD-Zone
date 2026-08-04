"""Evaluate a zoning under the analytical expected-MNL welfare objective."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from optimization.analytical_welfare_oracle import (
    AnalyticalMarketResult,
    ZonedAnalyticalWelfareResult,
    evaluate_zoned_analytical_welfare,
)
from optimization.config import OptimizationConfig
from optimization.data.cutoffs import build_analytical_welfare_market
from optimization.verify_welfare_scenario import (
    _valid_boundary,
    _valid_encoded_geography,
    _valid_frl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_output" / "analytical_welfare_evaluation"


def target_config() -> OptimizationConfig:
    """Return the canonical 6-zone-9 analytical-welfare scenario."""
    return OptimizationConfig(
        centroids_type="6-zone-9",
        levels=["Block_2"],
        solver="cp_bool",
        strategy="welfare",
        frl_dev=0.15,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=0.2,
        solve_time_limits=[540.0],
        gap_limits=[0.0],
        hints="voronoi",
        seed=42,
        workers=5,
        years=[23],
        population_type="All",
        capacity_scenario="A",
        new_schools=True,
        include_k8=False,
        cutoff_lottery_scale=20,
        cutoff_gumbel_scale=1.0,
        remove_city_wide=True,
    )


def build_target_scenario():
    """Build the canonical geographic problem and systematic-utility market."""
    config = target_config()
    dataset = config.make_dataset()
    problem = dataset.problem_for("Block_2")
    problem.boundary_prop = config.boundary_prop
    market = build_analytical_welfare_market(
        dataset,
        problem,
        assignment_config=config.cutoff_assignment_config,
        ctip_path=config.cutoff_ctip_path,
        lottery_scale=config.cutoff_lottery_scale,
        beta=config.cutoff_gumbel_scale,
        remove_city_wide=True,
        outside_systematic_utility=0.0,
    )
    problem.analytical_welfare_market = market
    return config, problem, market


def load_node_assignment(path: Path) -> dict[int, int]:
    with path.expanduser().resolve().open(encoding="utf-8") as input_file:
        return {int(node): int(zone) for node, zone in json.load(input_file).items()}


def geography_checks(problem, assignment: dict[int, int]) -> dict[str, bool]:
    complete = set(assignment) == set(problem.nodes) and set(
        assignment.values()
    ) <= set(range(problem.Z))
    return {
        "complete_assignment": complete,
        "frl_deviation_0_15": complete and _valid_frl(problem, assignment),
        "boundary_proportion_0_2": complete
        and _valid_boundary(problem, assignment),
        "encoded_geography": complete
        and _valid_encoded_geography(problem, assignment),
    }


def result_summary(result: ZonedAnalyticalWelfareResult) -> dict:
    return {
        "objective_kind": result.objective_kind,
        "normalized_welfare": result.normalized_welfare,
        "raw_welfare_constant": result.raw_welfare_constant,
        "raw_expected_welfare": (
            result.normalized_welfare + result.raw_welfare_constant
        ),
        "cutoff_grid": result.cutoff_grid,
        "school_cutoffs": result.school_cutoffs,
        "school_cutoff_indices": result.school_cutoff_indices,
        "school_demands": result.school_demands,
        "stable": result.stable,
        "least_cutoff_numerically_verified": (
            result.least_cutoff_numerically_verified
        ),
        "timing_seconds": result.timing_seconds,
        "zones": {
            str(zone): market_result_summary(zone_result)
            for zone, zone_result in result.zones.items()
        },
    }


def market_result_summary(result: AnalyticalMarketResult) -> dict:
    return {
        "normalized_welfare": result.normalized_welfare,
        "cutoffs": result.cutoffs,
        "cutoff_indices": result.cutoff_indices,
        "demands": result.demands,
        "capacities": result.capacities,
        "iterations": result.iterations,
        "capacity_feasible": result.capacity_feasible,
        "capacity_feasibility_tolerance": result.capacity_feasibility_tolerance,
        "complementarity_valid": result.complementarity_valid,
        "complementarity_tolerance": result.complementarity_tolerance,
        "grid_minimal": result.grid_minimal,
        "grid_underfill": result.grid_underfill,
        "grid_lowered_demand": result.grid_lowered_demand,
        "least_cutoff_numerically_verified": (
            result.least_cutoff_numerically_verified
        ),
        "least_cutoff_residual": result.least_cutoff_residual,
        "least_cutoff_tolerance": result.least_cutoff_tolerance,
        "max_capacity_violation": result.max_capacity_violation,
        "max_mass_balance_residual": result.max_mass_balance_residual,
        "timing_seconds": result.timing_seconds,
    }


def _assignment_payload(result: ZonedAnalyticalWelfareResult) -> dict:
    return {
        "objective_kind": result.objective_kind,
        "assignment_measures": result.assignment_measures,
        "outside_measures": result.outside_measures,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--skip-continuum", action="store_true")
    parser.add_argument("--write-assignment-measures", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    build_started = time.monotonic()
    config, problem, market = build_target_scenario()
    build_seconds = time.monotonic() - build_started
    assignment = load_node_assignment(args.assignment)
    checks = geography_checks(problem, assignment)
    if not all(checks.values()):
        raise ValueError(f"Assignment fails target geography checks: {checks}.")

    q20 = evaluate_zoned_analytical_welfare(
        market,
        assignment,
        num_zones=problem.Z,
        cutoff_grid=config.cutoff_lottery_scale,
        tolerance=args.tolerance,
    )
    continuum = None
    if not args.skip_continuum:
        continuum = evaluate_zoned_analytical_welfare(
            market,
            assignment,
            num_zones=problem.Z,
            cutoff_grid=None,
            tolerance=args.tolerance,
        )
    payload = {
        "passed": (
            all(checks.values())
            and q20.stable
            and q20.least_cutoff_numerically_verified
            and (
                continuum is None
                or (
                    continuum.stable
                    and continuum.least_cutoff_numerically_verified
                )
            )
        ),
        "assignment_path": str(args.assignment.expanduser().resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "market_build_seconds": build_seconds,
        "geography_checks": checks,
        "scenario": {
            "centroids_type": config.centroids_type,
            "level": "Block_2",
            "frl_dev": config.frl_dev,
            "boundary_prop": config.boundary_prop,
            "lottery_scale": config.cutoff_lottery_scale,
            "beta": market.beta,
            "segment_count": len(market.segments),
            "eligible_incidence_count": sum(
                len(segment.eligible_schools) for segment in market.segments
            ),
            "school_count": len(market.school_capacities),
            "market_metadata": market.metadata,
        },
        "q20": result_summary(q20),
        "continuum": result_summary(continuum) if continuum is not None else None,
    }
    verification_path = args.output / "analytical_welfare_verification.json"
    with verification_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
    if args.write_assignment_measures:
        with (args.output / "q20_expected_assignments.json").open(
            "w", encoding="utf-8"
        ) as output_file:
            json.dump(_assignment_payload(q20), output_file, sort_keys=True)
        if continuum is not None:
            with (args.output / "continuum_expected_assignments.json").open(
                "w", encoding="utf-8"
            ) as output_file:
                json.dump(_assignment_payload(continuum), output_file, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
