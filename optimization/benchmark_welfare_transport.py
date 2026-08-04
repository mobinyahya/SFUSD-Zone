"""Run the zoning-aware capacitated transport bound in isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import gurobipy as gp
import ortools

from optimization.config import OptimizationConfig
from optimization.data.cutoffs import build_cutoff_market
from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY
from optimization.levels import LevelSpec
from optimization.solution import graph_fingerprint
from optimization.solvers.welfare_decomposition import (
    WelfareDecompositionSolver,
    _WelfareIncumbent,
)
from optimization.welfare_oracle import solve_zoned_welfare


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSIGNMENT = (
    PROJECT_ROOT
    / "benchmark_output"
    / "welfare_recom_seed46_840"
    / "zone_dict_Block_2.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "benchmark_output" / "welfare_transport_bound" / "result.json"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--engine", choices=("gurobi", "cp-sat"), default="gurobi")
    parser.add_argument("--initial-assignment", type=Path, default=DEFAULT_ASSIGNMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.solve_seconds <= 0 or args.solve_seconds >= 895:
        parser.error("--solve-seconds must be positive and below 895")

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
        seed=args.seed,
        workers=args.workers,
        linearization_level=2 if args.engine == "cp-sat" else None,
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
        remove_city_wide=True,
        welfare_utility_scale=1_000_000,
    )
    dataset = config.make_dataset()
    problem = dataset.problem_for(LevelSpec.parse("Block_2"))
    problem.boundary_prop = config.boundary_prop
    problem.cutoff_market = build_cutoff_market(
        dataset,
        problem,
        assignment_config=config.cutoff_assignment_config,
        ctip_path=config.cutoff_ctip_path,
        lottery_scale=config.cutoff_lottery_scale,
        gumbel_scale=config.cutoff_gumbel_scale,
        preference_seed=config.cutoff_preference_seed,
        remove_city_wide=True,
        outside_option_utility=0.0,
    )
    with args.initial_assignment.open(encoding="utf-8") as input_file:
        assignment = {
            int(node): int(zone) for node, zone in json.load(input_file).items()
        }
    if set(assignment) != set(problem.nodes):
        raise ValueError("Initial assignment must cover every problem node.")
    incumbent_result = solve_zoned_welfare(
        problem.cutoff_market,
        assignment,
        num_zones=problem.Z,
        utility_scale=config.welfare_utility_scale,
    )
    incumbent = _WelfareIncumbent(assignment, incumbent_result)
    solver = WelfareDecompositionSolver(
        config.make_solver(),
        utility_scale=config.welfare_utility_scale,
    )
    if args.engine == "gurobi":
        candidate_assignment, proof_grade_bound, status = (
            solver._assignment_transport_mip(
                problem,
                incumbent,
                args.solve_seconds,
            )
        )
        solver_details = solver._transport_mip_details
        raw_bound = solver_details["diagnostic_raw_upper_bound"]
        proof_grade = False
    else:
        candidate_assignment, raw_bound, status = solver._assignment_relaxation_cp(
            problem,
            incumbent,
            args.solve_seconds,
        )
        proof_grade_bound = raw_bound
        solver_details = solver._transport_cp_details
        proof_grade = True
    candidate = (
        solve_zoned_welfare(
            problem.cutoff_market,
            candidate_assignment,
            num_zones=problem.Z,
            utility_scale=config.welfare_utility_scale,
        )
        if candidate_assignment is not None
        else None
    )
    normalizer = config.cutoff_lottery_scale * config.welfare_utility_scale
    candidate_path = args.output.with_name("candidate_zone_dict.json")
    candidate_hash = None
    if candidate_assignment is not None:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        with candidate_path.open("w", encoding="utf-8") as candidate_file:
            json.dump(candidate_assignment, candidate_file, indent=2, sort_keys=True)
        candidate_hash = _sha256(candidate_path)
    elapsed = time.monotonic() - started
    payload = {
        "status": status,
        "elapsed_seconds": elapsed,
        "solve_seconds": args.solve_seconds,
        "engine": args.engine,
        "graph_fingerprint": graph_fingerprint(problem.G),
        "num_nodes": problem.A,
        "num_edges": problem.G.number_of_edges(),
        "num_zones": problem.Z,
        "incumbent_raw_welfare": incumbent_result.raw_scaled_welfare,
        "incumbent_welfare": incumbent_result.raw_scaled_welfare / normalizer,
        "candidate_raw_welfare": (
            candidate.raw_scaled_welfare if candidate is not None else None
        ),
        "candidate_welfare": (
            candidate.raw_scaled_welfare / normalizer if candidate is not None else None
        ),
        "candidate_assignment": (
            str(candidate_path) if candidate_assignment is not None else None
        ),
        "candidate_assignment_sha256": candidate_hash,
        "raw_upper_bound": raw_bound,
        "welfare_upper_bound": raw_bound / normalizer,
        "proof_grade": proof_grade,
        "proof_grade_raw_upper_bound": proof_grade_bound,
        "proof_grade_welfare_upper_bound": proof_grade_bound / normalizer,
        "proof_grade_raw_gap": proof_grade_bound - incumbent_result.raw_scaled_welfare,
        "proof_grade_welfare_gap": (
            proof_grade_bound - incumbent_result.raw_scaled_welfare
        )
        / normalizer,
        "raw_gap": raw_bound - incumbent_result.raw_scaled_welfare,
        "welfare_gap": (raw_bound - incumbent_result.raw_scaled_welfare) / normalizer,
        "bound_is_at_least_incumbent": (
            raw_bound >= incumbent_result.raw_scaled_welfare
        ),
        "finite_bound": math.isfinite(raw_bound),
        "solver_details": solver_details,
        "provenance": {
            "initial_assignment": str(args.initial_assignment.resolve()),
            "initial_assignment_sha256": _sha256(args.initial_assignment),
            "market_sha256": _market_sha256(problem),
            "instance_sha256": _instance_sha256(problem),
            "source_sha256": _source_sha256(),
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "ortools": ortools.__version__,
            "gurobi": gp.gurobi.version(),
            "preference_seed": config.cutoff_preference_seed,
            "gumbel_scale": config.cutoff_gumbel_scale,
            "lottery_scale": config.cutoff_lottery_scale,
            "utility_scale": config.welfare_utility_scale,
            "seed": args.seed,
            "workers": args.workers,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _market_sha256(problem) -> str:
    market = problem.cutoff_market
    payload = {
        "graph_fingerprint": graph_fingerprint(problem.G),
        "centroids": problem.centroids,
        "centroid_school_ids": problem.centroid_school_ids,
        "lottery_scale": market.lottery_scale,
        "school_nodes": market.school_nodes,
        "school_capacities": market.school_capacities,
        "students": [
            {
                "studentno": student.studentno,
                "node": student.node,
                "preferences": student.preferences,
                "priorities": student.priorities,
                "utilities": student.utilities,
            }
            for student in market.students
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _instance_sha256(problem) -> str:
    payload = {
        "graph_fingerprint": graph_fingerprint(problem.G),
        "centroids": problem.centroids,
        "centroid_school_ids": problem.centroid_school_ids,
        "frl_dev": problem.frl_dev,
        "racial_dev": problem.racial_dev,
        "overage": problem.overage,
        "shortage": problem.shortage,
        "boundary_prop": problem.boundary_prop,
        "nodes": [
            (node, dict(sorted(problem.G.nodes[node].items())))
            for node in sorted(problem.nodes)
        ],
        "candidates": {
            node: sorted(problem.candidate_zones(node)) for node in problem.nodes
        },
        "closer_neighbors": problem.G.graph.get(CLOSER_NEIGHBORS_GRAPH_KEY),
    }
    encoded = json.dumps(
        payload,
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256() -> str:
    paths = [
        PROJECT_ROOT / "optimization" / "solvers" / "welfare.py",
        PROJECT_ROOT / "optimization" / "solvers" / "welfare_decomposition.py",
        PROJECT_ROOT / "optimization" / "welfare_oracle.py",
        *sorted((PROJECT_ROOT / "optimization" / "branch_price").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
