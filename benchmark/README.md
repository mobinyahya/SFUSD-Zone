# Benchmark Package

Optimization-native benchmark orchestration for large SFUSD zoning simulation sweeps.

This package runs the `optimization` stack directly. It does not use the legacy benchmark path.

## Capabilities

- Runs large cartesian simulation sweeps from one YAML file.
- Supports varying any `OptimizationConfig` field across a sweep, including `solver`, `strategy`, `levels`, time limits, seeds, and balance constraints.
- Executes tasks in parallel with capacity-aware scheduling.
- Recycles worker processes with `max_tasks_per_worker` to reduce long-run memory growth.
- Skips existing completed results only when the manifest schema and config hash match.
- Saves every optimization stage, not just the final solution.
- Recomputes metrics from saved stage artifacts without rerunning optimization.
- Aggregates run-level and stage-level outputs into CSV files.
- Keeps benchmark output focused on optimization artifacts and aggregate metrics.

## Entry Point

```bash
uv run python -m benchmark.run path/to/sweep.yaml
uv run python -m benchmark.run path/to/sweep.yaml --mode metrics
```

The YAML `mode` can be `run` or `metrics`. The CLI `--mode` flag overrides the YAML value.

Aggregation is automatic after both modes.

## Sweep YAML

See `sweep.example.yaml` for a full example.

Top-level sections:

| Section | Purpose |
|---|---|
| `name` | Human-readable sweep name. |
| `mode` | Default mode: `run` or `metrics`. |
| `optimization_defaults` | Base `OptimizationConfig` values shared by all tasks. |
| `sweep` | Cartesian product values for `OptimizationConfig` fields. |
| `tasks` | Explicit per-task overrides, crossed with `sweep` values. |
| `execution` | Parallelism, capacity, skipping, and output options. |
| `metrics` | Metric strictness, stage metric opt-in, and aggregation output settings. |

Example with solver and strategy variation:

```yaml
optimization_defaults:
  centroids_type: '5-zone-AF'
  frl_dev: 0.2
  racial_dev: 0.2
  overage: 0.8
  shortage: 0.2
  max_distance: 5
  seed: 42

sweep:
  solver: ['cp_int', 'cp_bool']
  seed: [42, 14]

tasks:
  - strategy: 'single'
    levels: ['BlockGroup_0']
    solve_time_limits: [60]
    gap_limits: [0]
    workers: 8

  - strategy: 'recursive'
    levels: ['BlockGroup_1', 'BlockGroup_0']
    solve_time_limits: [45, 60]
    gap_limits: [0, 0]
    workers: 8

execution:
  output_dir: './benchmark_output'
  capacity: 32
  max_workers: 5
  max_tasks_per_worker: 25
  skip_existing: true
  rerun_failed: true

metrics:
  strict: true
  compute_stage_metrics: false
  summary_csv: 'summary.csv'
  stages_csv: 'stages.csv'
```

## Execution Model

Each expanded task is a concrete `OptimizationConfig` plus benchmark metadata.

Capacity scheduling uses `capacity_slots` per task. By default this equals the task's `workers` value, so CP-SAT thread counts are reflected in the scheduler. You can override this globally with `execution.task_capacity`.

For the `overlapping` strategy, that worker budget is used for concurrent
one-worker school solves. Each school solve receives
`school_solve_time_limit`; the final all-school solve receives the full worker
budget.

Important execution fields:

| Field | Description |
|---|---|
| `output_dir` | Root directory for all run outputs. |
| `capacity` | Total capacity slots available on the machine. |
| `max_workers` | Maximum concurrent Python worker processes. |
| `max_tasks_per_worker` | Number of tasks before recycling a worker process. |
| `skip_existing` | Skip valid completed outputs with matching config hash. |
| `rerun_failed` | Rerun failed tasks instead of treating them as complete. |
| `sequential` | Run tasks in-process for debugging. |
| `fail_fast` | Stop on the first task error. |
| `output_template` | Optional format string for run output paths. |

## Output Contract

Each task writes one run directory.

```text
benchmark_manifest.json
result.json
zone_dict_<level>.json
zone_dict_area_<level>.json
solution_<level>.json
stages/
  stage_00_<level>/
    zone_dict_<level>.json
    zone_dict_area_<level>.json
    solution_<level>.json
  stage_01_<level>/
    ...
```

`benchmark_manifest.json` stores task identity, config hash, schema version, stage paths, status, timings, and the selected final stage.

`result.json` stores metrics, zone data, run metadata, levels, status, and a config snapshot.

Root-level `zone_dict_*`, `zone_dict_area_*`, and `solution_*` files are aliases for the metrics-selected final solution.

`zoned_cg_seed_paths` and `zoned_benders_seed_paths` are sequence-valued. A
flat list is one task's complete seed-path list; use a list of lists to sweep
over alternative path sets. When a zoned analytical strategy saves its
mechanism, each stage and the final
root alias contain `artifacts/shi_mechanism_<level>.json`. Solution, manifest,
and result payloads contain only its relative filename and compact summary.
Metrics-only reconstruction preserves that reference and does not duplicate or
rewrite the sparse mechanism payload.

## Modes

`run` expands the YAML into tasks, executes optimization, writes artifacts, computes final-solution metrics, and writes aggregate CSVs. Recursive/iterative stage objective and timing metadata are always preserved; expensive per-stage cut-edge/compactness metrics run only when `metrics.compute_stage_metrics: true`.

`metrics` discovers existing manifests under `execution.output_dir`, reconstructs saved `ZoneSolution` stages, recomputes metrics, rewrites `result.json`, and writes aggregate CSVs.

## Public API

Primary objects and functions:

| Symbol | File | Purpose |
|---|---|---|
| `SimulationSweep` | `config.py` | Parse YAML and generate benchmark tasks. |
| `BenchmarkTask` | `config.py` | Concrete optimization task with config hash and output path. |
| `run_optimization_task` | `runner.py` | Run one optimization task and save artifacts. |
| `load_solutions` | `runner.py` | Reconstruct `ZoneSolution` stages from saved artifacts. |
| `run_tasks` | `parallel.py` | Capacity-aware task execution. |
| `regenerate_metrics` | `regenerate.py` | Metrics-only recomputation. |

## Notes

- The benchmark package expects new optimization level names such as `BlockGroup_0` and `BlockGroup_1`.
- `mip` requires Gurobi and a valid license in the execution environment.
- Metrics-only mode requires the graph cache and source data needed by `OptimizationConfig.make_dataset()`.
- Existing results are considered reusable only when `benchmark_manifest.json`, `result.json`, schema version, and config hash all match.
