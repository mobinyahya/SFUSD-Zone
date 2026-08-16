"""Optimization entry point.

    uv run python -m optimization.run [config.yaml] [--output DIR]

Loads a :class:`OptimizationConfig`, builds the dataset / solver / strategy, runs
the strategy, and saves each produced :class:`ZoneSolution`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from optimization.config import OptimizationConfig
from metrics import MetricsCalculator

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.example.yaml")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SFUSD zoning optimization.")
    parser.add_argument(
        "config", nargs="?", default=DEFAULT_CONFIG, help="Path to a config YAML."
    )
    parser.add_argument(
        "-o", "--output", default="./optimization_output", help="Output directory."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save PNG solution visualization file(s).",
    )
    parser.add_argument(
        "--viz-stages",
        choices=["final", "all"],
        default="final",
        help="Visualize only the final solution or every recursive/iterative stage.",
    )
    args = parser.parse_args(argv)

    config = OptimizationConfig.from_yaml(args.config)
    os.makedirs(args.output, exist_ok=True)
    dataset = config.make_dataset()
    solver = config.make_solver(output_dir=args.output)
    strategy = config.make_strategy()

    print(
        f"Running strategy={config.strategy} solver={config.solver} "
        f"levels={config.levels} centroids={config.centroids_type}"
    )
    solutions = strategy.run(dataset, solver)

    for sol in solutions:
        sol.save(args.output)
        contig = sol.is_contiguous() if sol.feasible else "n/a"
        print(
            f"  {sol.level}: status={sol.status} objective={sol.objective} "
            f"contiguous={contig} ({sol.wall_time:.1f}s)"
        )

    metrics = MetricsCalculator(solutions, config=config).compute()
    result_path = os.path.join(args.output, "result.json")
    with open(result_path, "w") as f:
        json.dump(_result_payload(metrics, config, solutions), f, indent=2)
    print(f"  metrics: saved {result_path}")

    if args.visualize:
        from optimization.visualization import visualize_solutions

        viz_results = visualize_solutions(
            solutions,
            output_dir=args.output,
            stages=args.viz_stages,
            config=config,
        )
        for result in viz_results:
            if result.skipped:
                print(f"  visualization {result.stage}: skipped ({result.skipped})")
                continue
            saved = ", ".join(str(path) for path in result.figure_paths)
            artifact_info = (
                f" geometry={result.geometry_artifact}"
                if result.geometry_artifact
                else ""
            )
            if saved:
                print(f"  visualization {result.stage}: saved {saved}{artifact_info}")
            else:
                print(f"  visualization {result.stage}: no figure saved{artifact_info}")
    print(f"Saved {len(solutions)} solution(s) to {args.output}")


def _result_payload(metrics, config: OptimizationConfig, solutions) -> dict:
    payload = metrics.to_full_dict()
    run = payload.get("run", {})
    payload.update(
        {
            "status": run.get("final_status"),
            "error_message": None,
            "total_wall_time": run.get("total_wall_time", 0.0),
            "levels": [sol.level.name for sol in solutions],
            "config": _config_snapshot(config),
        }
    )
    return payload


def _config_snapshot(config: OptimizationConfig) -> dict:
    snapshot = asdict(config)
    snapshot["unit"] = config.unit
    return snapshot


if __name__ == "__main__":
    main()
