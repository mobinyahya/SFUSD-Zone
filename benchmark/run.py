"""Single CLI entrypoint for optimization benchmark sweeps."""

from __future__ import annotations

import argparse
import os

from benchmark.config import SimulationSweep
from benchmark.parallel import run_tasks
from benchmark.regenerate import regenerate_metrics
from benchmark.results import aggregate_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SFUSD zoning benchmark sweeps.")
    parser.add_argument("config", help="Path to simulation sweep YAML.")
    parser.add_argument(
        "--mode",
        choices=["run", "metrics", "matching"],
        help="Override the mode declared in the YAML file.",
    )
    args = parser.parse_args(argv)

    sweep = SimulationSweep.from_yaml(args.config)
    mode = args.mode or sweep.mode
    output_dir = os.path.expanduser(sweep.execution.output_dir)

    if mode == "run":
        tasks = sweep.generate_tasks()
        print(f"Generated {len(tasks)} benchmark task(s).")
        batch = run_tasks(
            tasks,
            execution=sweep.execution,
            metrics=sweep.metrics,
            matching=sweep.matching,
            visualization=sweep.visualization,
        )
        print(
            f"Completed {batch.completed}/{batch.total}; "
            f"{batch.status_count_summary(separator=', ')}, "
            f"wall={batch.total_wall_time / 60:.1f} min"
        )
        _aggregate(output_dir, sweep)
    elif mode == "matching":
        from benchmark.assignment import run_assignments_for_existing_runs

        result = run_assignments_for_existing_runs(
            output_dir,
            sweep.matching,
            fail_fast=sweep.execution.fail_fast,
        )
        print(
            f"Assigned {result.successful}/{result.total}; "
            f"failed={result.failed}, skipped={result.skipped}"
        )
        _aggregate(output_dir, sweep)
    elif mode == "metrics":
        regen = regenerate_metrics(
            output_dir,
            strict=sweep.metrics.strict,
            compute_stage_metrics=sweep.metrics.compute_stage_metrics,
            visualization=sweep.visualization,
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
