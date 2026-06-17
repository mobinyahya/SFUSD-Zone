"""Single CLI entrypoint for pipeline benchmark sweeps."""

from __future__ import annotations

import argparse
import os

from Zone_Generation.Running_Analysis.benchmark.config import SimulationSweep
from Zone_Generation.Running_Analysis.benchmark.parallel import run_tasks
from Zone_Generation.Running_Analysis.benchmark.regenerate import regenerate_metrics
from Zone_Generation.Running_Analysis.benchmark.results import aggregate_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SFUSD zoning benchmark sweeps.")
    parser.add_argument("config", help="Path to simulation sweep YAML.")
    parser.add_argument(
        "--mode",
        choices=["run", "metrics"],
        help="Override the mode declared in the YAML file.",
    )
    args = parser.parse_args(argv)

    sweep = SimulationSweep.from_yaml(args.config)
    mode = args.mode or sweep.mode
    output_dir = os.path.expanduser(sweep.execution.output_dir)

    if mode == "run":
        tasks = sweep.generate_tasks()
        print(f"Generated {len(tasks)} benchmark task(s).")
        batch = run_tasks(tasks, execution=sweep.execution, metrics=sweep.metrics)
        print(
            f"Completed {batch.successful}/{batch.total}; "
            f"failed={batch.failed}, skipped={batch.skipped}, "
            f"wall={batch.total_wall_time / 60:.1f} min"
        )
        _aggregate(output_dir, sweep)
    elif mode == "metrics":
        regen = regenerate_metrics(
            output_dir,
            strict=sweep.metrics.strict,
            fail_fast=sweep.execution.fail_fast,
        )
        print(
            f"Regenerated {regen.regenerated}/{regen.total}; "
            f"failed={regen.failed}, skipped={regen.skipped}"
        )
        _aggregate(output_dir, sweep)


def _aggregate(output_dir: str, sweep: SimulationSweep) -> None:
    run_df, stage_df = aggregate_results(
        output_dir,
        summary_csv=sweep.metrics.summary_csv,
        stages_csv=sweep.metrics.stages_csv,
    )
    print(
        f"Aggregated {len(run_df)} run row(s) and {len(stage_df)} stage row(s) "
        f"under {output_dir}."
    )


if __name__ == "__main__":
    main()
