# Metrics Package

Optimization-native metrics for zoning solutions produced by `optimization`.

Metrics operate on `ZoneSolution` objects, not legacy `(zone_dict, graph)` pairs. A metric run can include a single solution, recursive stages, or iterative-choice attempts.

## Capabilities

- Computes final solution quality metrics from a `ZoneSolution`.
- Computes run-level objective and timing metadata across every stage returned by a optimization strategy.
- Supports single-shot, recursive, and iterative-choice strategies.
- Selects the final solution consistently with optimization semantics.
- Produces flat metric dictionaries for CSV aggregation.
- Produces full JSON payloads with run metadata and per-zone data.
- Exposes modular metric files so new metric groups can be added without changing the benchmark runner.

## Entry Point

Use `MetricsCalculator` from `metrics`.

```python
from metrics import MetricsCalculator

result = MetricsCalculator(solutions, config=config).compute()
flat_metrics = result.metrics
full_payload = result.to_full_dict()
```

`solutions` can be one `ZoneSolution` or a sequence of `ZoneSolution` objects.

## Result Shape

`MetricsResult` has three sections:

| Section | Purpose |
|---|---|
| `metrics` | Flat scalar values suitable for CSV columns. |
| `zone_data` | Per-zone nested data for demographics, programs, distance, and quality. |
| `run` | Stage metadata, final-stage selection, statuses, objectives, and timings. |

Use `metrics` for aggregation tables and `to_full_dict()` for `result.json`
payloads. Column names are defined once, in `metrics/columns.py`.

## Final Solution Selection

`MetricsContext` selects the final solution as follows:

- For iterative-choice runs with `choice_utility`, select the stage with the best utility.
- Otherwise select the last solution with an assignment.
- If no solution has an assignment, select the last stage.

This means benchmark `result.json` can summarize a best iterative-choice solution while still preserving every stage under `run.stages`.

## Metric Modules

Default modules are registered in `calculator.py` through `DEFAULT_MODULES`.

| Module | Capabilities |
|---|---|
| `run_metrics.py` | Final status, objectives, wall time, and per-stage metadata. Per-stage cut edges and compactness scores are opt-in through `compute_stage_metrics`. |
| `diversity.py` | FRL, race and ethnicity balance, AALPI balance, and seat disparity. |
| `programs.py` | Program counts by zone, language programs, special education, and GE access. |
| `distance.py` | In-zone GE school distances and nearby out-of-zone GE school access. |
| `quality.py` | School quality balance using math and English score attributes. |
| `structure.py` | Zone count, Block_0-normalized cut edges, fractional cut edges, Reock/Polsby-Popper compactness, contiguity, and solution code. |

## Per-Zone Data

Metric modules can add per-zone values through `MetricOutput.zone_data`.

Examples include:

- Population: `students` and `capacity` are the problem's own population
  (`problem.student_attribute`/`capacity_attribute`, all-program counts under
  `program_population: All`, named in `run.zone_population`). Every share and
  `seat_disparity` is taken over them. `ge_students`, `ge_capacity`,
  `all_prog_students` and `all_prog_capacity` report each population under its
  own name.
- Demographics: `frl_pct`, `aalpi_pct`, `ethnicity_pcts` (the five graph
  groups), and `supplementary_ethnicity_pcts`, read from the run's student area
  table: Filipino, Two or More, American Indian, Not Specified, Decline to
  State, Missing (blank) and `Ethnicity_Unreported`, the sum of the last three.
  It is `None` when the config names no data.
- Schools: `school_ids`, `attendance_school_ids` (category `Attendance`) and
  `ge_school_ids` physically in the zone. `schools_in_attendance_area` is *not*
  a school count: it counts the zone's schools whose `attendance_area` ID is a
  school in the same zone, so a school sharing another's area (Mission Bay,
  area 497) reads 0 apart from that school.
- Distance: great-circle miles between node points, a school at its node's
  point. `avg_any_ge_school_distance`/`avg_farthest_ge_school_distance` are
  unweighted node means of the mean/max over in-zone GE schools. The nearest
  fields pair each node with its nearest in-zone GE school: for the GE students
  (`ge_` prefix) and the problem population (no prefix), the weighted average,
  the maximum over populated nodes (positive weight), the populated-node count
  and the students in a zone with no GE school.
- Shape: `polsby_popper_score`, `reock_score`, `area_sq_m` and `perimeter_m` on
  the dissolved Block_0 geometry (EPSG:32610); `None` for a zone without
  positive-area geometry.
- `programs`, `avg_math_score`, `avg_eng_score`.

The benchmark package stores this data in `result.json` and keeps scalar summaries in `summary.csv`.

## Adding A Metric Module

Metric modules are functions with this shape:

```python
from metrics.base import MetricOutput, MetricsContext

def compute(context: MetricsContext) -> MetricOutput:
    return MetricOutput(
        metrics={"example_metric": 1.0},
        zone_data={0: {"example_zone_value": 2.0}},
        run={"example_run_value": "ok"},
    )
```

To enable a module by default, add it to `DEFAULT_MODULES` in `calculator.py`.

For one-off calculations, pass a custom module list:

```python
result = MetricsCalculator(solutions, modules=[custom_compute]).compute()
```

## Data Expectations

Metrics expect `ZoneSolution.problem.G` to follow the graph attribute schema used by the new optimization.

Required graph data includes:

- Node demographic and capacity attributes such as `ge_students`, `ge_capacity`, `FRL`, and ethnicity counts.
- Node school attributes such as `school_ids` and `num_schools`.
- Graph-level `distance_dict` for distance metrics.
- Graph-level `school_data` for program and quality metrics.
- Graph-level district proportions `F` and `R` for diversity metrics.

## Error Handling

`MetricsCalculator(..., strict=True)` raises if any metric module fails.

`MetricsCalculator(..., strict=False)` records module errors under `result.run["metric_errors"]` and continues with the remaining modules.

Benchmark runs default to strict metrics so metric regressions fail visibly during sweeps.
