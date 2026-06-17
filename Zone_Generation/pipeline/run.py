"""Pipeline entry point.

    uv run python -m Zone_Generation.pipeline.run [config.yaml] [--output DIR]

Loads a :class:`PipelineConfig`, builds the dataset / solver / strategy, runs
the strategy, and saves each produced :class:`ZoneSolution`.
"""

from __future__ import annotations

import argparse
import os

from Zone_Generation.pipeline.config import PipelineConfig

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.example.yaml")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SFUSD zoning pipeline.")
    parser.add_argument(
        "config", nargs="?", default=DEFAULT_CONFIG, help="Path to a config YAML."
    )
    parser.add_argument(
        "-o", "--output", default="./pipeline_output", help="Output directory."
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

    config = PipelineConfig.from_yaml(args.config)
    dataset = config.make_dataset()
    solver = config.make_solver()
    strategy = config.make_strategy()

    print(
        f"Running strategy={config.strategy} solver={config.solver} "
        f"levels={config.levels} centroids={config.centroids_type}"
    )
    solutions = strategy.run(dataset, solver)

    os.makedirs(args.output, exist_ok=True)
    for sol in solutions:
        sol.save(args.output)
        contig = sol.is_contiguous() if sol.feasible else "n/a"
        print(
            f"  {sol.level}: status={sol.status} objective={sol.objective} "
            f"contiguous={contig} ({sol.wall_time:.1f}s)"
        )

    if args.visualize:
        from Zone_Generation.pipeline.visualization import visualize_solutions

        viz_results = visualize_solutions(
            solutions,
            output_dir=args.output,
            is_local=config.is_local,
            stages=args.viz_stages,
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


if __name__ == "__main__":
    main()
