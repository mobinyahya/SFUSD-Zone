"""Run and independently verify the requested stable-welfare benchmark."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from optimization.config import OptimizationConfig
from optimization.data import loaders
from optimization.data.contiguity import boundary_edges
from optimization.welfare_oracle import (
    solve_zoned_continuum_welfare,
    solve_zoned_welfare,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_output" / "welfare_6_zone_9"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-seconds", type=float, default=540.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--utility-scale", type=int, default=1_000_000)
    parser.add_argument("--initial-assignment", type=Path)
    args = parser.parse_args(argv)
    if args.solve_seconds <= 0 or args.solve_seconds >= 600:
        parser.error("--solve-seconds must be positive and below 600")
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    config = OptimizationConfig(
        centroids_type="6-zone-9",
        levels=["Block_2"],
        solver="cp_bool",
        strategy="welfare",
        frl_dev=0.15,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=0.2,
        solve_time_limits=[args.solve_seconds],
        gap_limits=[0.0],
        hints="voronoi",
        seed=42,
        workers=5,
        linearization_level=1,
        cp_sat_search_strategy="distance_to_centroid",
        years=[23],
        population_type="All",
        capacity_scenario="A",
        new_schools=True,
        include_k8=False,
        cutoff_assignment_config="assignment/configs/kumar.config.yaml",
        cutoff_ctip_path=(
            "/share/data/school_choice/Data/2025_cleaned_data/"
            "Cleaned_new/ETB_2024.npy"
        ),
        cutoff_lottery_scale=20,
        cutoff_gumbel_scale=1.0,
        cutoff_preference_seed=2023,
        remove_city_wide=True,
        welfare_utility_scale=args.utility_scale,
        welfare_initial_assignment_path=(
            str(args.initial_assignment) if args.initial_assignment else ""
        ),
    )
    dataset = config.make_dataset()
    expected_student_ids = set(
        map(int, loaders.load_students(dataset.ingest)["studentno"])
    )
    solution = config.make_strategy().run(
        dataset, config.make_solver(output_dir=str(args.output))
    )[0]
    solution.save(str(args.output))
    elapsed = time.monotonic() - started

    checks, reconstruction = validate_solution(
        solution, expected_student_ids, runtime_seconds=elapsed
    )
    payload = {
        "passed": all(checks.values()),
        "status": solution.status,
        "objective": solution.objective,
        "elapsed_seconds": elapsed,
        "solver_wall_time": solution.wall_time,
        "num_nodes": solution.problem.A,
        "num_edges": solution.problem.G.number_of_edges(),
        "num_zones": solution.problem.Z,
        "checks": checks,
        "reconstruction": reconstruction,
        "config": {
            "centroids_type": config.centroids_type,
            "level": config.levels[-1],
            "frl_dev": config.frl_dev,
            "boundary_prop": config.boundary_prop,
            "lottery_scale": config.cutoff_lottery_scale,
            "utility_scale": config.welfare_utility_scale,
            "prefix_depth": config.welfare_prefix_depth,
            "remove_city_wide": config.remove_city_wide,
        },
        "artifact": str(args.output / "solution_Block_2.json"),
    }
    with (args.output / "verification.json").open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


def validate_solution(solution, expected_student_ids, *, runtime_seconds):
    problem = solution.problem
    market = problem.cutoff_market
    assignment = solution.assignment
    complete = solution.feasible and len(assignment) == problem.A
    grid = solve_zoned_welfare(
        market,
        assignment,
        num_zones=problem.Z,
        utility_scale=solution.metadata["welfare_utility_scale"],
    )
    continuum = solve_zoned_continuum_welfare(
        market, assignment, num_zones=problem.Z
    )
    student_ids = {student.studentno for student in market.students}
    zone_stable = {
        str(zone): result.stable
        for zone, result in continuum.cutoffs.zones.items()
    }
    checks = {
        "runtime_below_10_minutes": runtime_seconds < 600.0,
        "feasible_complete_assignment": complete,
        "six_zones": problem.Z == 6,
        "contiguous": complete and solution.is_contiguous(),
        "frl_deviation_0_15": complete and _valid_frl(problem, assignment),
        "boundary_proportion_0_2": complete and _valid_boundary(problem, assignment),
        "lottery_scale_20": market.lottery_scale == 20,
        "isolated_markets_only": (
            set(market.school_capacities) == set(market.zone_restricted_schools)
            and solution.metadata.get("market_coupling") == "isolated_zones"
        ),
        "all_students_modeled": student_ids == expected_student_ids,
        "no_sampled_tie_breakers": (
            "no applicant tie-break scores are sampled"
            in solution.metadata.get("utility_definition", "")
        ),
        "grid_minimal": grid.cutoffs.grid_minimal,
        "grid_cutoffs_reconstructed": (
            grid.cutoffs.school_cutoffs == solution.metadata.get("school_cutoffs")
        ),
        "grid_welfare_reconstructed": (
            grid.raw_scaled_welfare
            == solution.metadata.get("raw_scaled_welfare")
            and math.isclose(
                grid.welfare,
                float(solution.metadata.get("welfare", math.inf)),
                abs_tol=1e-9,
            )
        ),
        "continuous_stable": continuum.stable,
        "all_zone_markets_stable": (
            len(zone_stable) == problem.Z
            and all(zone_stable.values())
            and zone_stable == solution.metadata.get("zone_stable")
        ),
        "continuous_cutoffs_reconstructed": _float_maps_close(
            continuum.cutoffs.school_cutoffs,
            solution.metadata.get("continuum_school_cutoffs", {}),
        ),
        "finite_grid_global_optimum_certified": (
            solution.status == "OPTIMAL"
            and solution.metadata.get("global_optimum_certified") is True
            and solution.metadata.get("raw_scaled_upper_bound")
            == solution.metadata.get("raw_scaled_welfare")
        ),
    }
    reconstruction = {
        "welfare": grid.welfare,
        "rounded_welfare": (
            grid.raw_scaled_welfare
            / (market.lottery_scale * grid.utility_scale)
        ),
        "raw_scaled_welfare": grid.raw_scaled_welfare,
        "raw_scaled_upper_bound": solution.metadata.get("raw_scaled_upper_bound"),
        "true_welfare_upper_bound": solution.metadata.get(
            "true_welfare_upper_bound"
        ),
        "true_welfare_gap_bound": solution.metadata.get("true_welfare_gap_bound"),
        "continuum_welfare": continuum.welfare,
        "zone_stable": zone_stable,
        "student_count": len(student_ids),
        "school_count": len(market.school_capacities),
    }
    return checks, reconstruction


def _valid_frl(problem, assignment):
    lower = problem.district_frl - problem.frl_dev
    upper = problem.district_frl + problem.frl_dev
    for zone in range(problem.Z):
        nodes = [node for node, assigned in assignment.items() if assigned == zone]
        students = sum(problem.students(node) for node in nodes)
        if students <= 0:
            return False
        ratio = sum(problem.frl(node) for node in nodes) / students
        if ratio < lower - 1e-4 or ratio > upper + 1e-4:
            return False
    return True


def _valid_boundary(problem, assignment):
    return boundary_edges(problem.G, assignment) <= math.floor(
        problem.boundary_prop * problem.G.number_of_edges()
    )


def _float_maps_close(left, right, tolerance=1e-8):
    return set(left) == set(right) and all(
        math.isclose(float(value), float(right[key]), abs_tol=tolerance)
        for key, value in left.items()
    )


if __name__ == "__main__":
    main()
