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

### Cut-edge convergence plots

```bash
uv run python -m benchmark.plot_edges benchmark/configs/benchmark_edges.yaml

# Local output and a comparison of individual runs on the same instance:
uv run python -m benchmark.plot_edges benchmark/configs/benchmark_edges.yaml \
  --centroids 6-zone-3 --seeds 1 --levels Block_2 --time-limits 600 \
  --output-dir analysis/plots/output/edges_6_zone_3
```

The positional argument can also be a benchmark output directory or its
`summary.csv`. A YAML supplies only `execution.output_dir`: saved manifests
identify the historical run settings, even if the YAML has since changed.
No graph caches or district source tables are loaded. `--solvers` filters methods.

The default output directory is `<benchmark root>/plots/edges`. Outputs are
`cut_edges_Block_2_tl_600s_median.png`, etc. (one figure per budget **and level**),
`cut_edges_events.csv` (raw parsed events with run identity), and
`cut_edges_runs.csv` (run inventory, including missing logs, failures, and a
`plotted` flag). `cut_edges_aggregates.csv` contains medians, quartiles, and
the contributor count `n` at each event time.

Panels separate zone counts. The default `--aggregate median` draws one median
line per method with a light interquartile band (middle 50%) across seeds and
centroid variants **within the same zone count, graph level, and budget**.
It does not pool different zone counts or levels. Use `--aggregate individual`
for the individual run traces; those filenames end in `_individual.png`.
Never-feasible runs are excluded entirely, including their bounds and legend
entries. The inventory and raw-event CSV retain them for auditing.

`--skip-initial 1` is the default: omit each run's first distinct feasible
objective value.
Repeated node-table values, bound updates, and summary lines do not count.
`--skip-initial 0` shows every feasible improvement. Runs with no remaining
improvements are omitted entirely, including their bounds. Trimming applies
before aggregation and is recorded in the figure subtitle and run CSV.

The band shows the middle 50% of the **available** retained feasible values,
including before every run is ready. No solid line appears during this early
interval. With only one available run the band has zero width.
The solid median uses a **fixed set of successful runs** for each method within
a panel. Its curve starts at the latest first **retained** feasible timestamp
among those runs. Earlier solutions are not backfilled, and later arrivals
cannot cause upward jumps: every contributing incumbent is non-increasing,
so the median and both quartiles are also non-increasing from that point onward.
The early band can shift as additional runs become available. The tradeoff is that
methods with a late feasible run start later; use `--aggregate individual` to
inspect the early progress. This remains a success-conditional summary and
does not represent failure rates. The band is spread across runs, not a
confidence interval.
Final logged values are held through the common horizon (at least the time
budget) in aggregate mode so completed runs do not disappear from the cohort.
No improvements are invented after the last log event. Lower bounds are omitted
from all figures and the aggregate CSV; parsed bounds remain in the raw-event
CSV for auditing.

Use the filters above to inspect a single instance. Separate output directories
keep filtered plots from replacing the overview.

- Solid steps show **feasible logged objectives** (medians by default).
  No line appears before a feasible incumbent is found. Infeasible heuristic
  penalties and CP-SAT's `UNKNOWN` summary objective are never plotted as solutions.
- This is a minimization problem: the feasible objective itself is an upper
  bound on the optimum. Solver lower bounds are not plotted.
- Time is the elapsed clock reported by each solver, not end-to-end benchmark
  wall time. Pre-solve graph/model/hint work may be excluded. Individual steps stop at
  the last logged timestamp, including a termination summary when present;
  they are not extended to an unobserved time limit. Native logs have rounded
  timestamps and sampled bounds. Gurobi root-relaxation durations are omitted
  because they are phase durations, not cumulative solve times.
- Only single-solve, unweighted boundary objectives are included. Recursive,
  weighted, and other objective variants are counted in the skip report so
  metres or unrelated objectives cannot silently appear on a `Cut edges` axis.

CP/MIP boundary indicators are constrained to be positive on cut edges, but can
also be positive on uncut edges in a suboptimal incumbent. Thus the native log
objective is in cut-edge units and can overstate the actual cut count of that
incumbent's zoning. The logs alone cannot recover the exact zoning cut count;
that requires assignment snapshots or logging an explicitly recomputed count.

### Comparing recursive stages

Unweighted cut counts on different levels count different adjacency graphs;
stitching their native objectives would not form a comparable trajectory.
For existing logs, a useful extension would plot **only the target-level stage**
with its start shifted by preceding stages' recorded durations, and show the
coarse stages as a shaded preparation interval. This preserves cut-edge units
but cannot recover earlier fine-level incumbents. Exact end-to-end timing would
require explicit stage-start timestamps, because per-solve logs omit setup work.

For new experiments, the existing `optimization_defaults.weight_edges: true`
setting enables integer-metre shared-boundary costs for all these methods.
Aggregation sums crossing parent-edge weights, so a coarse zoning and its
unchanged projection onto a finer graph have exactly the same weighted cost.
Use a **new `execution.output_dir`** for that experiment. Plot those results on
a `Weighted boundary length (m)` axis with stage transitions; the cut-edge
plotter intentionally excludes them. Keep `looseness: 1.0` when comparing
feasibility across stages, and verify projected solutions against the target
problem before calling them feasible target-level incumbents.

For exact polygon geometry, the sum of zone perimeters equals the fixed district
perimeter plus twice the internal shared-boundary length. The implemented cost
is a perimeter proxy: base lengths are rounded to integer metres, point-touch
edges have a minimum weight of one, and synthetic manual bridges receive
positive weights. Its projection invariance holds for those implemented
weights, rather than for an exact geometrical perimeter.

Even with comparable weighted objectives, **recursive bounds remain local to
each stage**. Coarse aggregation restricts possible partitions, and finer
recursive solves restrict candidate zones near the previous boundary. Their
lower bounds do not certify the unrestricted fine-level optimum and should
not be merged into a global bound curve. Show them separately and reset at
stage transitions, or omit them from a cross-method comparison.

An alternative that retains `Cut edges` as the common metric is to enable
`save_solver_progress: true` for recursive CP/MIP runs, project each saved
incumbent to the chosen target graph, and recount cut edges there. That costs
more storage and requires a separate snapshot-based plotter; old scalar logs
cannot reconstruct it, and coarse bounds cannot be converted by recounting.

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
