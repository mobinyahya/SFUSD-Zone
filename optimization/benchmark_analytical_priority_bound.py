"""Run the canonical FLOATING diagnostic common-cutoff/common-STB bound."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from optimization.analytical_priority_bound import (
    FLOATING_DIAGNOSTIC_SCOPE,
    solve_common_cutoff_stb_bound,
)
from optimization.evaluate_analytical_welfare import build_target_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = (
    PROJECT_ROOT / "benchmark_output" / "analytical_cardinality_shi" / "result.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_output" / "analytical_priority_bound"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--inclusion-shortlist", type=int, default=4)
    parser.add_argument(
        "--relax-integrality",
        action="store_true",
        help="Solve the direct continuous LP instead of the strengthened MIP.",
    )
    parser.add_argument("--enforce-geography", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be positive and finite")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.inclusion_shortlist < 0:
        parser.error("--inclusion-shortlist must be non-negative")

    started = time.monotonic()
    price_bytes = PRICE_PATH.read_bytes()
    price_payload = json.loads(price_bytes)
    if not isinstance(price_payload, dict):
        raise ValueError("The cardinality-Shi price artifact must contain an object.")
    raw_prices = price_payload.get("school_prices")
    if not isinstance(raw_prices, dict):
        raise ValueError("The cardinality-Shi price artifact has no school_prices map.")
    school_prices = {int(school): float(price) for school, price in raw_prices.items()}
    cardinality = price_payload.get("cardinality")
    if isinstance(cardinality, bool) or not isinstance(cardinality, int):
        raise ValueError("The cardinality-Shi artifact has no integer cardinality.")

    scenario_started = time.monotonic()
    config, problem, market = build_target_scenario()
    scenario_seconds = time.monotonic() - scenario_started
    result = solve_common_cutoff_stb_bound(
        problem,
        market,
        school_prices,
        cardinality=cardinality,
        inclusion_shortlist=args.inclusion_shortlist,
        relax_integrality=args.relax_integrality,
        enforce_geography=args.enforce_geography,
        time_limit=args.time_limit,
        workers=args.workers,
    )
    output_path = _result_path(args.output)
    payload = {
        "method": "COMMON_CUTOFF_COMMON_STB_CONDITIONAL_LOSS_FLOATING_DIAGNOSTIC",
        "numerical_scope": FLOATING_DIAGNOSTIC_SCOPE,
        "proof_grade": False,
        "status": result.status,
        "mode": (
            "DIRECT_CONTINUOUS_LP_FLOATING_DIAGNOSTIC"
            if args.relax_integrality
            else "INTEGRALITY_STRENGTHENED_MIP_FLOATING_DIAGNOSTIC"
        ),
        "bounds": {
            "direct_continuous_lp_bound": result.continuous_lp_bound,
            "integrality_strengthened_mip_obj_bound": (
                result.integrality_strengthened_mip_obj_bound
            ),
            "integrality_strengthened_mip_incumbent_not_a_bound": (
                result.integrality_strengthened_mip_incumbent
            ),
            "cardinality_shi_bound_recomputed_at_prices": (
                result.cardinality_shi_bound_at_prices
            ),
            "capacity_price_constant": result.capacity_price_constant,
            "diagnostic_bound_reduction_from_cardinality_shi": (
                result.bound_reduction_from_cardinality_shi
            ),
        },
        "dimensions": asdict(result.dimensions),
        "timings_seconds": {
            "canonical_scenario_build": scenario_seconds,
            "validation": result.validation_seconds,
            "conditional_pricing": result.conditional_pricing_seconds,
            "model_build": result.model_build_seconds,
            "solve": result.solve_seconds,
            "bound_end_to_end": result.total_seconds,
            "cli_end_to_end": time.monotonic() - started,
        },
        "configuration": {
            "centroids_type": config.centroids_type,
            "level": "Block_2",
            "zones": problem.Z,
            "cutoff_grid": market.lottery_scale,
            "beta": market.beta,
            "cardinality": cardinality,
            "inclusion_shortlist": args.inclusion_shortlist,
            "relax_integrality": args.relax_integrality,
            "enforce_geography": args.enforce_geography,
            "time_limit": args.time_limit,
            "workers": args.workers,
            "mip_focus": None if args.relax_integrality else 3,
        },
        "provenance": {
            "canonical_scenario_builder": (
                "optimization.evaluate_analytical_welfare.build_target_scenario"
            ),
            "bound_solver": (
                "optimization.analytical_priority_bound.solve_common_cutoff_stb_bound"
            ),
            "price_artifact": str(PRICE_PATH.resolve()),
            "price_artifact_sha256": hashlib.sha256(price_bytes).hexdigest(),
            "price_artifact_status": price_payload.get("status"),
            "price_artifact_upper_bound": price_payload.get("upper_bound"),
            "price_artifact_residual_repaired_upper_bound": price_payload.get(
                "residual_repaired_upper_bound"
            ),
            "school_prices": {
                str(school): price for school, price in result.school_prices.items()
            },
        },
        "raw_solver_status": result.raw_solver_status,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "numerical_scope": FLOATING_DIAGNOSTIC_SCOPE,
                "proof_grade": False,
                "status": result.status,
                "mode": payload["mode"],
                "bounds": payload["bounds"],
                "output": str(output_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _result_path(output: Path) -> Path:
    output = output.expanduser()
    return output if output.suffix.lower() == ".json" else output / "result.json"


if __name__ == "__main__":
    main()
