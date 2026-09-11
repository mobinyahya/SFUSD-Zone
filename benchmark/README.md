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

### Slurm

Generate a task snapshot and submission script without contacting Slurm:

```bash
uv run python -m benchmark.slurm generate --config path/to/sweep.yaml
```

Submit the same two-phase job graph directly:

```bash
uv run python -m benchmark.slurm submit --config path/to/sweep.yaml
```

Each sweep task is one optimization job using the config's `workers` count. A
one-core metrics job runs after it and safely updates `summary.csv` and
`stages.csv`. Recursive and iterative strategy stages stay within one
optimization job. Slurm mode rejects enabled `matching` and assignment-based
`choice_metrics`; local capacity and worker-pool settings are not used.

Plans, scripts, and logs are written beneath
`<execution.output_dir>/.slurm/`. All jobs use Slurm account and partition
`soal`.

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

### Objective Trajectories

`save_solver_logs: true` adds one log per solve call under `solver_logs/`, named `solver_<index>_<level>_<solver>.<ext>`. The index increments per solve, so a recursive or iterative strategy produces one log per stage in execution order. Each stage's `solver_log_path` and `solver_log_format` are recorded in `stages.csv`.

| Solver | Format | Contents |
|---|---|---|
| `mip` | Gurobi log | Node table: incumbent, best bound, and gap on the display interval plus every new incumbent. |
| `cp_bool`, `cp_int` | CP-SAT log | `#N` and `#Bound` events with timestamps, plus the response summary (`gap_integral` included). |
| `recom`, `relaxed_recom`, `short_bursts`, `adaptive_short_bursts` | JSONL | One record per strictly improving state: boundary cost, every individual constraint penalty, and both Lagrangians. |

Heuristic records are improvement-only, ordered by `(not feasible, boundary cost if feasible else unweighted Lagrangian)`. Any feasible state supersedes any infeasible one, so penalties are nonzero only while no feasible partition has been found yet. Penalty units follow the solver: `adaptive_short_bursts` normalizes residuals to a percentage of zone students, the rest report absolute student counts, and each log states which in its `penalty_scale` header field.

`save_solver_progress: true` is a separate, heavier mechanism: `mip` and `cp_bool` only, and it writes two full `zone_dict` files per incumbent alongside `solver_progress/<id>/progress.jsonl`. Prefer the logs when only objective values are needed.

Flatten a whole sweep into one long-format table:

```bash
python -m benchmark.solver_logs <output_dir> --out solver_progress.csv
```

`incumbent` and `bound` are the columns comparable across backends (`bound` is empty for the heuristics, which have none); `elapsed_seconds` is measured from the start of each solve.

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
| `collect` | `solver_logs.py` | Parse Gurobi/CP-SAT/heuristic solver logs into one trajectory frame. |

## Notes

- The benchmark package expects new optimization level names such as `BlockGroup_0` and `BlockGroup_1`.
- `mip` requires Gurobi and a valid license in the execution environment.
- Metrics-only mode requires the graph cache and source data needed by `OptimizationConfig.make_dataset()`.
- Existing results are considered reusable only when `benchmark_manifest.json`, `result.json`, schema version, and config hash all match.
