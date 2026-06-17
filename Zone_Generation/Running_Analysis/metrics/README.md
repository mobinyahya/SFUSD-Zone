# Metrics Package

Pipeline-native metrics for zoning solutions produced by `Zone_Generation.pipeline`.

Metrics operate on `ZoneSolution` objects, not legacy `(zone_dict, graph)` pairs. A metric run can include a single solution, recursive stages, or iterative-choice attempts.

## Capabilities

- Computes final solution quality metrics from a `ZoneSolution`.
- Computes run-level metrics across every stage returned by a pipeline strategy.
- Supports single-shot, recursive, and iterative-choice strategies.
- Selects the final solution consistently with pipeline semantics.
- Produces flat metric dictionaries for CSV aggregation.
- Produces full JSON payloads with run metadata and per-zone data.
- Exposes modular metric files so new metric groups can be added without changing the benchmark runner.

## Entry Point

Use `MetricsCalculator` from `Zone_Generation.Running_Analysis.metrics`.

```python
from Zone_Generation.Running_Analysis.metrics import MetricsCalculator

result = MetricsCalculator(solutions, config=config).compute()
flat_metrics = result.to_flat_dict()
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

Use `to_flat_dict()` for aggregation tables.

Use `to_full_dict()` for `result.json` payloads.

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
| `run_metrics.py` | Final status, objectives, wall time, boundary cost, and per-stage metadata. |
| `diversity.py` | FRL, race and ethnicity balance, AALPI balance, and seat disparity. |
| `programs.py` | Program counts by zone, language programs, special education, and GE access. |
| `distance.py` | In-zone GE school distances and nearby out-of-zone GE school access. |
| `quality.py` | School quality balance using math and English score attributes. |
| `structure.py` | Zone count, boundary cost, compactness, contiguity, and solution code. |

## Per-Zone Data

Metric modules can add per-zone values through `MetricOutput.zone_data`.

Examples include:

- `ge_students`
- `frl_pct`
- `ethnicity_pcts`
- `seat_disparity`
- `programs`
- `avg_math_score`
- `avg_eng_score`
- `avg_any_ge_school_distance`
- `avg_farthest_ge_school_distance`

The benchmark package stores this data in `result.json` and keeps scalar summaries in `summary.csv`.

## Adding A Metric Module

Metric modules are functions with this shape:

```python
from Zone_Generation.Running_Analysis.metrics.base import MetricOutput, MetricsContext

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

Metrics expect `ZoneSolution.problem.G` to follow the graph attribute schema used by the new pipeline.

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
