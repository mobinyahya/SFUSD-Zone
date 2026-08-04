"""Re-score saved Block_2 zonings under the analytical welfare objective."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from optimization.analytical_welfare_oracle import evaluate_zoned_analytical_welfare
from optimization.evaluate_analytical_welfare import (
    build_target_scenario,
    geography_checks,
    load_node_assignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "benchmark_output"
DEFAULT_OUTPUT = DEFAULT_INPUT / "analytical_welfare_baselines"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--continuum-top", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)
    if args.continuum_top < 0:
        parser.error("--continuum-top must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    config, problem, market = build_target_scenario()
    paths = sorted(args.input.rglob("zone_dict_Block_2.json"))
    unique = {}
    errors = []
    for path in paths:
        try:
            assignment = load_node_assignment(path)
            signature = tuple(sorted(assignment.items()))
            unique.setdefault(signature, {"assignment": assignment, "paths": []})[
                "paths"
            ].append(str(path.resolve()))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})

    records = []
    for item in unique.values():
        assignment = item["assignment"]
        checks = geography_checks(problem, assignment)
        if not all(checks.values()):
            records.append(
                {"paths": item["paths"], "geography_checks": checks, "feasible": False}
            )
            continue
        evaluation_started = time.monotonic()
        q20 = evaluate_zoned_analytical_welfare(
            market,
            assignment,
            num_zones=problem.Z,
            cutoff_grid=config.cutoff_lottery_scale,
            tolerance=args.tolerance,
        )
        records.append(
            {
                "paths": item["paths"],
                "geography_checks": checks,
                "feasible": True,
                "q20_welfare": q20.normalized_welfare,
                "q20_seconds": time.monotonic() - evaluation_started,
                "q20_stable": q20.stable,
                "q20_grid_minimal": all(
                    zone.grid_minimal is True for zone in q20.zones.values()
                ),
                "assignment": assignment,
            }
        )

    feasible = sorted(
        (record for record in records if record["feasible"]),
        key=lambda record: record["q20_welfare"],
        reverse=True,
    )
    for record in feasible[: args.continuum_top]:
        evaluation_started = time.monotonic()
        continuum = evaluate_zoned_analytical_welfare(
            market,
            record["assignment"],
            num_zones=problem.Z,
            cutoff_grid=None,
            tolerance=args.tolerance,
        )
        record["continuum_welfare"] = continuum.normalized_welfare
        record["continuum_seconds"] = time.monotonic() - evaluation_started
        record["continuum_stable"] = continuum.stable

    payload = {
        "objective_kind": "analytical_gumbel_stable_welfare_cutoff_grid_20",
        "elapsed_seconds": time.monotonic() - started,
        "source_path_count": len(paths),
        "unique_assignment_count": len(unique),
        "target_feasible_count": len(feasible),
        "continuum_top": args.continuum_top,
        "errors": errors,
        "records": records,
        "ranking": [
            {
                key: value
                for key, value in record.items()
                if key not in {"assignment", "geography_checks"}
            }
            for record in feasible
        ],
    }
    with (args.output / "results.json").open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
    if feasible:
        with (args.output / "best_zone_dict_Block_2.json").open(
            "w", encoding="utf-8"
        ) as output_file:
            json.dump(feasible[0]["assignment"], output_file, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "elapsed_seconds": payload["elapsed_seconds"],
                "source_path_count": len(paths),
                "unique_assignment_count": len(unique),
                "target_feasible_count": len(feasible),
                "best": payload["ranking"][:1],
                "output": str((args.output / "results.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
