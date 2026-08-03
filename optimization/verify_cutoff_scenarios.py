"""Run and validate the requested real-data cutoff zoning scenarios.

Examples:
    uv run python -m optimization.verify_cutoff_scenarios --case block_1
    uv run python -m optimization.verify_cutoff_scenarios --case zones_13
    uv run python -m optimization.verify_cutoff_scenarios --case citywide
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from optimization.config import OptimizationConfig
from optimization.cutoff_oracle import (
    solve_coupled_continuum_cutoffs,
    solve_coupled_cutoffs,
    solve_zoned_continuum_cutoffs,
    solve_zoned_cutoffs,
)
from optimization.data import loaders
from optimization.data.contiguity import boundary_edges


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_output" / "cutoff_requested_verification"


@dataclass(frozen=True)
class VerificationCase:
    level: str
    centroids_type: str
    frl_dev: float
    boundary_prop: float
    remove_city_wide: bool = True
    runtime_limit: float = 900.0


CASES = {
    "block_1": VerificationCase("Block_1", "6-zone-9", -1.0, -1.0),
    "block_0": VerificationCase("Block_0", "6-zone-9", -1.0, -1.0),
    "constraints_6": VerificationCase("Block_2", "6-zone-9", 0.15, 0.25),
    "zones_10": VerificationCase("Block_2", "10-zone-3", -1.0, -1.0),
    "zones_13": VerificationCase("Block_2", "13-zone-6", -1.0, -1.0),
    "zones_18": VerificationCase("Block_2", "18-zone-1", -1.0, -1.0),
    "citywide": VerificationCase(
        "Block_2",
        "6-zone-9",
        0.15,
        0.25,
        remove_city_wide=False,
        runtime_limit=600.0,
    ),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES),
        help="Scenario to run; repeat the option to run several. Defaults to all.",
    )
    parser.add_argument(
        "--solve-seconds",
        type=float,
        default=240.0,
        help="CP-SAT limit per scenario; runtime acceptance limits remain fixed.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.solve_seconds <= 0:
        parser.error("--solve-seconds must be positive")

    selected = args.case or list(CASES)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in selected:
        row = run_case(
            name,
            CASES[name],
            solve_seconds=min(args.solve_seconds, CASES[name].runtime_limit - 30.0),
            output_root=args.output,
        )
        rows.append(row)
        print(
            f"{name}: {'PASS' if row['passed'] else 'FAIL'} "
            f"status={row['status']} objective={row['objective']} "
            f"elapsed={row['elapsed_seconds']:.2f}s"
        )

    payload = {
        "all_passed": all(row["passed"] for row in rows),
        "cases": rows,
    }
    summary_path = args.output / "summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
    print(f"Saved {summary_path}")
    if not payload["all_passed"]:
        raise SystemExit(1)


def run_case(
    name: str,
    case: VerificationCase,
    *,
    solve_seconds: float,
    output_root: Path,
) -> dict:
    started = time.monotonic()
    config = OptimizationConfig(
        centroids_type=case.centroids_type,
        levels=[case.level],
        solver="cp_bool",
        strategy="cutoffs",
        frl_dev=case.frl_dev,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=case.boundary_prop,
        solve_time_limits=[solve_seconds],
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
            "/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/ETB_2024.npy"
        ),
        cutoff_lottery_scale=20,
        cutoff_gumbel_scale=1.0,
        cutoff_preference_seed=2023,
        remove_city_wide=case.remove_city_wide,
    )
    dataset = config.make_dataset()
    expected_student_ids = set(
        map(int, loaders.load_students(dataset.ingest)["studentno"])
    )
    solver = config.make_solver(output_dir=str(output_root / name))
    solution = config.make_strategy().run(dataset, solver)[0]
    case_output = output_root / name
    solution.save(str(case_output))

    checks = validate_solution(solution, case, expected_student_ids)
    elapsed = time.monotonic() - started
    checks["runtime_within_limit"] = elapsed < case.runtime_limit
    row = {
        "name": name,
        "case": asdict(case),
        "solve_time_limit": solve_seconds,
        "elapsed_seconds": elapsed,
        "measured_runtime_scope": "data build, solve, validation, and solution write",
        "solver_wall_time": solution.wall_time,
        "status": solution.status,
        "objective": solution.objective,
        "num_nodes": solution.problem.A,
        "num_edges": solution.problem.G.number_of_edges(),
        "num_zones": solution.problem.Z,
        "market_coupling": solution.metadata.get("market_coupling"),
        "global_optimum_certified": solution.metadata.get("global_optimum_certified"),
        "checks": checks,
        "passed": all(checks.values()),
        "artifact": str(case_output / f"solution_{case.level}.json"),
    }
    with (case_output / "verification.json").open("w", encoding="utf-8") as output_file:
        json.dump(row, output_file, indent=2, sort_keys=True)
    return row


def validate_solution(
    solution,
    case: VerificationCase,
    expected_student_ids: set[int],
) -> dict[str, bool]:
    problem = solution.problem
    assignment = solution.assignment
    feasible = solution.feasible and len(assignment) == problem.A
    market = problem.cutoff_market
    unrestricted = set(market.school_capacities) - set(market.zone_restricted_schools)
    if unrestricted:
        grid = solve_coupled_cutoffs(market, assignment, num_zones=problem.Z)
        continuum = solve_coupled_continuum_cutoffs(
            market, assignment, num_zones=problem.Z
        )
        zone_stable = continuum.zone_stable
        zone_stability_checks = continuum.zone_checks
    else:
        grid = solve_zoned_cutoffs(market, assignment, num_zones=problem.Z)
        continuum = solve_zoned_continuum_cutoffs(
            market, assignment, num_zones=problem.Z
        )
        zone_stable = {zone: result.stable for zone, result in continuum.zones.items()}
        zone_stability_checks = {
            zone: {"isolated_market_clears": stable}
            for zone, stable in zone_stable.items()
        }
    serialized_zone_stable = {str(zone): stable for zone, stable in zone_stable.items()}
    serialized_zone_checks = {
        str(zone): checks for zone, checks in zone_stability_checks.items()
    }

    cutoff_students = list(market.students)
    cutoff_student_ids = [student.studentno for student in cutoff_students]
    missing_preference_students = set(
        solution.metadata.get("missing_preference_studentnos", [])
    )
    outside_option_students = set(
        solution.metadata.get("outside_option_only_studentnos", [])
    )
    checks = {
        "feasible_complete_assignment": feasible,
        "expected_zone_count": problem.Z == len(problem.centroid_school_ids),
        "contiguous": feasible and solution.is_contiguous(),
        "grid_minimal": grid.grid_minimal,
        "grid_cutoffs_reconstructed": (
            grid.school_cutoffs == solution.metadata.get("school_cutoffs")
            and math.isclose(
                grid.normalized_objective,
                float(solution.objective),
                abs_tol=1e-12,
            )
        ),
        "continuous_stable": continuum.stable,
        "continuous_cutoffs_reconstructed": (
            _float_maps_close(
                continuum.school_cutoffs,
                solution.metadata.get("continuum_school_cutoffs", {}),
            )
            and math.isclose(
                continuum.objective,
                float(solution.metadata.get("continuum_objective", math.inf)),
                abs_tol=1e-8,
            )
        ),
        "all_zone_stability_checks": (
            len(zone_stable) == problem.Z
            and all(zone_stable.values())
            and len(zone_stability_checks) == problem.Z
            and all(
                all(zone_checks.values())
                for zone_checks in zone_stability_checks.values()
            )
        ),
        "zone_metadata_matches_reconstruction": (
            serialized_zone_stable == solution.metadata.get("zone_stable")
            and serialized_zone_checks == solution.metadata.get("zone_stability_checks")
        ),
        "frl_constraint": feasible and _valid_frl(problem, assignment),
        "boundary_constraint": feasible and _valid_boundary(problem, assignment),
        "all_optimization_students_modeled": (
            len(cutoff_student_ids) == len(set(cutoff_student_ids))
            and set(cutoff_student_ids) == expected_student_ids
            and len(cutoff_student_ids) == solution.metadata.get("cutoff_student_count")
        ),
        "missing_preferences_use_outside_option": (
            missing_preference_students <= outside_option_students
            and len(outside_option_students)
            == solution.metadata.get("outside_option_only_student_count")
            and all(
                not student.preferences
                for student in cutoff_students
                if student.studentno in outside_option_students
            )
        ),
    }
    if not case.remove_city_wide:
        checks.update(
            {
                "global_citywide_market": (
                    bool(unrestricted)
                    and solution.metadata.get("market_coupling")
                    == "global_citywide_access"
                ),
                "citywide_schools_included": (
                    len(unrestricted)
                    == solution.metadata.get("unrestricted_school_count")
                    and solution.metadata.get("excluded_citywide_school_count") == 0
                ),
            }
        )
    return checks


def _float_maps_close(left: dict, right: dict, tolerance: float = 1e-8) -> bool:
    return set(left) == set(right) and all(
        math.isclose(float(value), float(right[key]), abs_tol=tolerance)
        for key, value in left.items()
    )


def _valid_frl(problem, assignment: dict[int, int]) -> bool:
    if problem.frl_dev < 0:
        return True
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


def _valid_boundary(problem, assignment: dict[int, int]) -> bool:
    if problem.boundary_prop < 0:
        return True
    return boundary_edges(problem.G, assignment) <= math.floor(
        problem.boundary_prop * problem.G.number_of_edges()
    )


if __name__ == "__main__":
    main()
