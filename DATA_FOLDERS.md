# Data Folders

This document inventories the application data locations used by SFUSD-Zone.
It covers checked-in configuration data, external source data, generated caches,
optimization and benchmark results, matching artifacts, and retained legacy
paths. Python environments, package build products, editor state, `.git`, and
ordinary Python/test/linter caches are not application data and are outside this
inventory.

No raw student, school, or census dataset is checked into this repository.
End-to-end runs require the external files described below.

## Path Roots

The code uses the following roots. In the rest of this document, symbolic names
make the directory layouts easier to read.

| Symbol | Current path | Defined by | Purpose |
|---|---|---|---|
| `REPO` | Repository root | Process checkout | Code, checked-in configuration, and default local outputs |
| `SFUSD_SHARED` | `/share/data/school_choice` | `SFUSD_DATA_ROOT` in `Config/Constants.py` | Shared source data, graph/artifact caches, and HPC results |

Application data is limited to these two roots. Source datasets and shared
caches live under `SFUSD_SHARED`; checked-in configuration and default local
outputs live under `REPO`. The loaders do not provide a home-directory or local
mode fallback.

The important path overrides are:

| Setting or argument | Default | Effect |
|---|---|---|
| Optimization `graphs_dir` | `SFUSD_SHARED/Zones/Optimization/Graphs` | Moves the graph and closer-neighbor caches |
| Optimization CLI `--output` | `./optimization_output` | Moves a single run's result files |
| Benchmark `execution.output_dir` | `./benchmark_output` | Sets a sweep's result root |
| Benchmark `execution.output_template` | Generated hierarchy | Replaces the per-task directory hierarchy |
| Metrics context `programs_path` | `SFUSD_SHARED/Data/Cleaned/programs_withMissionBay_2324.csv` | API-only override for fallback program data |
| Metrics context `block0_graph_path` or `graphs_dir` | Legacy flat-graph search | API-only override for the Block-level graph used by spatial metrics |
| Metrics context `shape_metric_artifact_dir` | `SFUSD_SHARED/Data/Computed/shape_metric_artifacts` | API-only override for compactness geometry artifacts |
| Visualization `artifact_dir` or `--artifact-dir` | `SFUSD_SHARED/Data/Computed/visualization_artifacts` | Function argument or benchmark-visualization CLI option for map geometry artifacts |
| Matching source `paths.*` | Defaults described below | Overrides matching input roots and files; benchmark output paths are enforced by the runner |
| Assignment `paths.student-save` | `./assignment_output/precomputed` | Stores reusable standalone matching inputs under the repository when run from `REPO` |
| Assignment `paths.assignment-folder` | `./assignment_output/assignments` | Stores standalone raw assignments under the repository when run from `REPO` |

The metrics-context entries above are not `OptimizationConfig` fields. Adding
them to an optimization or sweep YAML causes an unknown-key error.

Relative optimization and benchmark output paths are resolved from the current
working directory. Relative matching template paths are also resolved from the
current working directory, not from the sweep YAML's directory. Benchmark task
output, metrics API input/artifact paths, and most matching paths expand `~`.
Custom `graphs_dir`, standalone `optimization.run --output`, a visualization
function's `artifact_dir`, and benchmark `metrics.summary_csv` and
`metrics.stages_csv` are used without `expanduser`; avoid `~` in those values
and use an absolute path instead.

## External Source Data

### `SFUSD_SHARED/Data/Cleaned/`

This is the main CSV input directory. It is read by optimization ingestion,
choice utility evaluation, metrics, and matching.

| File pattern or name | Consumer | Description |
|---|---|---|
| `enrolled_<YY><YY+1>.csv` | `optimization/data/loaders.py` | Kindergarten enrollment rows used when `drop_optout: true` |
| `student_<YY><YY+1>.csv` | `optimization/data/loaders.py` | Kindergarten student rows used when `drop_optout: false` |
| `schools_rehauled_1819.csv` | Optimization loaders | Legacy school table used when `new_schools: false` |
| `schools_table_for_zone_development_updated.csv` | Optimization loaders | School IDs, census locations, coordinates, categories, and school metrics; used when `new_schools: true` |
| `stanford_capacities_12.23.21.csv` | Optimization loaders | Program capacities by scenario and school |
| `programs_withMissionBay_2324.csv` | Program metrics | Fallback school/program mapping when graph data does not contain programs |
| `r1_filter_student_without_specialprogs_2324.csv` | MNL choice model and matching | Student demographics, locations, and ranked choices |
| `programs_without_specialprogs_2324.csv` | Matching | Programs available to the assignment simulation |
| `schools_rehauled_withMissionBay_2324.csv` | Matching | Schools used by the assignment simulation |

The school table is also the canonical source of school point coordinates used
for map markers and closer-neighbor calculations. Census polygon geometry still
comes from the shapefile.

The directory also receives generated student caches:

```text
Cleaned_Students_<years>_pop<population_type>_drop<0-or-1>.csv
```

These caches combine and filter the configured source years. The key includes
`years`, `population_type`, and `drop_optout`, but not source-file timestamps.
Delete the matching cache file to force re-ingestion after source CSVs change.
Because the cache is written beside source data, an uncached run needs write
permission to this directory.

### `SFUSD_SHARED/shapefiles/`

The active census geometry is:

```text
geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp
```

GeoPandas reads this file to build Block and BlockGroup geometry, area
centroids, visualizations, and shape metrics. Keep the normal ESRI Shapefile
sidecars, such as `.dbf`, `.shx`, and `.prj`, in the same directory.

### `SFUSD_SHARED/Zones/Optimization/`

This directory contains geographic crosswalks and precomputed graph inputs.

| File | Access | Description |
|---|---|---|
| `block_blockgroup_tract.csv` | Read | Valid census areas and the Block-to-BlockGroup crosswalk |
| `adjacency_matrix_b.csv` | Read | Block adjacency rows |
| `adjacency_matrix_bg.csv` | Read | BlockGroup adjacency rows |
| `distances_b2b_schools.csv` | Read/write | Block distance matrix; historically rectangular around school blocks |
| `distances_bg2bg.csv` | Read/write | BlockGroup distance matrix |

If a distance file is absent, ingestion computes a complete matrix from census
centroids and writes it atomically into this directory. The adjacency and
crosswalk files are not generated by current code.

### `SFUSD_SHARED/simulation-files/choice-model/`

`estimates_2324_exp8_0514.csv` contains student-by-program utility estimates.
It is read directly by `choice/mnl.py` and is the default matching
`estimate-path`.

### `SFUSD_SHARED/simulation-files/zones/`

This directory contains zone definitions used by matching and analysis. The
selected plans used by `analysis/visualize_selected_zones.py` and
`analysis/evaluate_zone_subconfig_matches.py` are:

```text
Zones_13-FRL_Dev_0.25-Objective_2500_BG.csv
Zones_6-FRL_Dev_0.10-Objective_1430_BG.csv
Zones_10-FRL_Dev_0.15-Objective_2250_BG.csv
```

### Matching Path Resolution

The default generated matching configuration has `paths.sfusd` set to
`SFUSD_SHARED`. Its relative defaults resolve beneath that root:

```text
Data/Cleaned/r1_filter_student_without_specialprogs_2324.csv
Data/Cleaned/programs_without_specialprogs_2324.csv
Data/Cleaned/schools_rehauled_withMissionBay_2324.csv
simulation-files/choice-model/estimates_2324_exp8_0514.csv
```

Matching templates may override `paths.sfusd`, `student-data`, `program-data`,
`school-data`, `estimate-path`, and `citywide-or-lp-zones`. Direct paths may be
absolute; matching-library handling of relative paths depends on the individual
path key. For benchmark matching, the runner overwrites `zone-files`,
`assignment-folder`, `student-save`, and `utility-model.save-path` after merging
the template, so templates cannot relocate benchmark-generated zones, raw
assignments, precomputed data, or utility arrays.

## Shared Generated Caches

### `SFUSD_SHARED/Zones/Optimization/Graphs/`

This is the active graph cache and the default `graphs_dir`. A `Dataset` lazily
loads or creates graphs in a parameter-specific namespace:

```text
Graphs/
  Block_<12-character-hash>/
    Block_0.pickle
    Block_1.pickle
    ...
  BlockGroup_<12-character-hash>/
    BlockGroup_0.pickle
    BlockGroup_1.pickle
    ...
  closer_neighbors_<level>.pickle
  closer_neighbors_<level>.pickle.lock       # persistent coordination lock file
```

The graph namespace includes the cache schema, census unit, student years,
population type, opt-out policy, capacity scenario, school/K-8 policy,
partition policy, and, for Block graphs, the manual-edge fingerprint. It is safe
to remove a namespace when the source data or graph-building policy changes;
the next request rebuilds it. Coarser levels are generated from their immediate
finer parent.

The `closer_neighbors_<level>.pickle` files are shared across graph namespaces.
Each store can hold multiple graph-fingerprint variants. The corresponding
`.lock` file is created for coordinated access and normally remains afterward.

### `SFUSD_SHARED/Data/Computed/visualization_artifacts/`

Visualization lazily writes reusable dissolved geometry:

```text
geometry_<level>_<graph-fingerprint>.pkl
geometry_<level>_<graph-fingerprint>.json
```

The pickle contains node geometry; the JSON records level, unit, node count,
and fingerprint. Rendered PNGs belong in the run output directory, not here.

### `SFUSD_SHARED/Data/Computed/shape_metric_artifacts/`

Spatial metrics lazily write projected geometry used for Reock and
Polsby-Popper calculations:

```text
area_perimeter_<level>_<graph-fingerprint>.pkl
area_perimeter_<level>_<graph-fingerprint>.json
```

These files can be deleted and regenerated from the census shapefile and graph.

### `SFUSD_SHARED/Data/Computed/Graphs/`

This is a legacy flat graph location, not the active namespaced graph cache.
Spatial metrics still look for `Block_0.pickle` here when a solution is not
already at `Block_0` and no graph is injected or explicitly configured. This
includes aggregated levels such as `Block_1`, not only BlockGroup solutions.

The metrics-context `graphs_dir` fallback checks only
`<graphs_dir>/Block_0.pickle` and `<graphs_dir-parent>/Block_0.pickle`; it does
not discover the active `<graphs_dir>/Block_<hash>/Block_0.pickle` layout. Pass
an exact `block0_graph_path` or inject `block0_graph` when using a namespaced
cache through the metrics API.

## Repository Data Directories

### `REPO/Config/`

This directory mixes Python constants with checked-in YAML data.

| Entry | Status | Purpose |
|---|---|---|
| `centroids.yaml` | Active | Maps `centroids_type` names to school IDs |
| `manual_block_edges.yaml` | Active | Reviewed closer-neighbor edge overrides for `Block_0` |
| `manual_block_edge_additions.yaml` | Active | Explicit missing Block adjacency overrides |
| `automatic_centroids.yaml` | Retained, no current reader | Historical centroid definitions |
| `school_closure_centroids.yaml` | Retained, no current reader | Historical closure centroid definitions |
| `config.yaml` | Legacy | Pre-rewrite optimization configuration |
| `recursive_config.yaml` | Legacy | Pre-rewrite recursive configuration |
| `config_zone_grid_search_default.yaml` | Legacy | Pre-rewrite grid-search configuration |
| `Constants.py`, `metrics_config.py` | Active code/config metadata | Constants, path helpers, and metric descriptions |

Use `optimization/config.example.yaml` as the current single-run configuration
shape. The active parser rejects unknown legacy keys.

### `REPO/benchmark/configs/`

This contains checked-in benchmark sweep YAML files. `SimulationSweep` reads
them to generate concrete optimization tasks and determine each sweep's output
root. Current configurations write under `SFUSD_SHARED/local_runs/` using these
subdirectories:

```text
59_recur_5/
feasible_mcmc_5/
full_recursive_sweep/
iterative_choice_test_mnl/
iterative_choice_updated_8/
sfusd_zone_test_3/
solver_comparison_5/
test_5/
test_cp_params_2/
test_objectives_4/
test_single_solver_2/
```

`benchmark/sweep.example.yaml` and `benchmark/sweep.test.yaml` at the package
root are additional example/test sweep data; they use `./optimization_output`.
The enabled matching section in `benchmark/sweep.test.yaml` currently refers to
`benchmark/matching/medium_zones_no_reserves_no_sib.yaml`, which is absent.
Change it to an existing template or restore that file before running matching
from this sweep.

### `REPO/benchmark/matching/`

The YAML files in this mixed code/config directory are matching policy
templates. The current templates are `sd.yaml`,
`zones+hard_reserves_06frl.yaml`, `zones+no_reserves.yaml`, and
`zones+soft_reserves_06frl.yaml`. The hard-reserves template is the default.
At runtime the benchmark runner merges a template with matching defaults and
replaces zone, assignment, precomputed-data, and utility-matrix output paths.

### `REPO/analysis/misc/`

This directory stores the manual Block-edge review workflow's inputs and
generated review tables:

| Entry | Role |
|---|---|
| `manual_case_selections.yaml` | Human selections from numbered review plots |
| `manual_block_edge_additions.yaml` | Human-authored explicit edge additions |
| `manual_case_manifest.json` | Generated mapping from review cases to census IDs |
| `manual_case_summary.csv` | Generated review summary |

The `compile` command writes `Config/manual_block_edges.yaml`; the separate
`compile-additions` command writes `Config/manual_block_edge_additions.yaml`.
Generated case images are written to `REPO/analysis/plots/manual_cases/`.

### `REPO/optimization/data/`

Despite its name, this directory contains Python ingestion and graph-building
code only. It is not a raw-data directory.

## Run Output Directories

### `REPO/optimization_output/`

This is the default output for `python -m optimization.run` when invoked from
the repository root. It is ignored by Git. A run may contain:

```text
optimization_output/
  result.json
  solution_<level>.json
  zone_dict_<level>.json
  zone_dict_area_<level>.json
  visualization_<stage>.png
  stages/
    stage_<index>_<level>/
      solution_<level>.json
      zone_dict_<level>.json
      zone_dict_area_<level>.json
  solver_logs/
    solver_<index>_<level>_<solver>.log
  solver_progress/
    <solver-id>/
      progress.jsonl
      zone_dict_<level>_<index>.json
      zone_dict_area_<level>_<index>.json
```

`result.json` contains metrics, run metadata, level names, status, and a config
snapshot. The standalone runner saves each recursive or iterative solution
directly into the output root. Recursive stages with different levels leave one
file trio per level; repeated iterative-choice stages at the same level
overwrite the preceding trio.

The metrics-selected iterative solution can differ from the literal last
iteration. In a standalone run, `result.json` reports the selected metrics, but
the same-level root assignment files retain whichever iteration was saved last.
Benchmark output avoids this ambiguity by saving every stage separately and
explicitly saving its metrics-selected final solution at the task root.

The output path is arbitrary when `--output` is supplied, so
`optimization_output/` describes a default layout rather than a required
location.

### Benchmark Result Roots

The default sweep root is `./benchmark_output/`, which is
`REPO/benchmark_output/` only when the command runs from the repository root.
Most checked-in HPC sweeps instead use a named child of
`SFUSD_SHARED/local_runs/`, listed above.

By default, each task is nested as follows:

```text
<benchmark-root>/
  <centroids-type>/
    seed<seed>/
      frl<frl>_racial<racial>/
        overage<overage>_shortage<shortage>/
          <strategy>_<solver>_<levels>_tl_<limits>_<hash>/
```

`execution.output_template` can replace that hierarchy. Every task directory
uses the optimization output contract plus:

```text
benchmark_manifest.json
stages/<stage-name>/...
```

The benchmark root receives `summary.csv` and `stages.csv` after aggregation.
Those filenames are configurable; absolute paths place the CSVs outside the
benchmark root. These two settings do not expand `~`; use an actual absolute
path rather than a tilde path to write outside the root.

The optional `metrics.solution_code.build_solution_code_index()` helper can
also write `solution_codes.json` directly under any result root. Normal
optimization and benchmark CLI runs do not create this index.

### `REPO/assignment_output/`

The committed assignment path configs use this ignored directory for standalone
generated data when commands run from the repository root:

```text
assignment_output/
  precomputed/
  assignments/
```

Benchmark-integrated matching overrides these paths and writes into its task or
stage result directory instead.

### Matching Output Within a Run

When enabled, matching creates a `matching/` subtree under the benchmark task
or stage directory:

```text
matching/
  zones.csv
  config.generated.yaml
  assignments_raw/**/*.csv
  student_school_assignments.csv
  school_populations.csv
  program_populations.csv
  summary.json
  choice_metrics_by_assignment.csv
  choice_metrics_summary.json
  precomputed/
    utility_matrix.npy
    student_program_distances*.csv
```

With multiple matching policies, policy-specific outputs are stored under
`matching/<policy>/`, while `matching/precomputed/<policy>/` stores the
corresponding arrays and distance tables. Stage matching repeats the same
structure under `stages/<stage-name>/matching/` and normally shares the final
run's precomputed root during one invocation.

A final matching run deletes and recreates its whole `matching/` directory,
including `precomputed/`, before writing. These artifacts are shared within an
invocation but are not preserved across final matching regeneration. The
standalone `sfusd-match simulate` command can write raw assignment CSVs to any
directory supplied through `--assignments-dir`, `--assignment-folder`, or
`paths.assignment-folder`.

### `REPO/analysis/plots/`

Analysis scripts write generated CSV summaries and PNG figures here by default.
The directory is ignored by Git. The scripts read benchmark `summary.csv`,
`result.json`, and `solver_progress/**/progress.jsonl` from configured or
command-line result roots. `analysis/plots/manual_cases/` is reserved for the
manual Block-edge review images.

## Legacy Paths

These repository paths remain for historical compatibility but are not the
normal source for a current optimization run.

| Path | Current use |
|---|---|
| `REPO/output/` | Old ignored output convention; no current writer |

Some analysis scripts also retain historical hard-coded result roots, such as
`SFUSD_SHARED/local_runs/solver_comparison`, while current sweep YAML files use
versioned names such as `solver_comparison_5`. Prefer the sweep's
`execution.output_dir` or an explicit analysis CLI argument.

Existing generated analysis CSVs additionally record historical
`SFUSD_SHARED/local_runs/solver_comparison_3/` and
`SFUSD_SHARED/local_runs/feasible_mcmc_3/` inputs. The checked-in
`feasible_mcmc_penalties_objective_over_time_*` CSV/PNG artifacts have no
current producer script and should be treated as historical outputs rather than
as reproducible current analysis products.

## Cache Invalidation

Most cache identities describe configuration or graph structure, not the
contents or modification times of their source files. Source-data updates
therefore require manual invalidation:

| Changed source | Remove before rerunning |
|---|---|
| Enrollment or student CSVs | Matching `Cleaned_Students_*.csv` files, then affected graph namespaces |
| School, capacity, adjacency, crosswalk, or distance CSVs | Affected graph namespaces |
| Area geometry or census identifiers | Affected distance CSVs, graph namespaces, closer-neighbor stores, visualization artifacts, and shape-metric artifacts |
| School coordinates | Affected `closer_neighbors_<level>.pickle` stores |

Distance matrices are reused based on file existence alone. Graph namespaces
hash ingestion settings and partition policy but not source-file contents.
Closer-neighbor stores fingerprint graph data but not shapefile or school-point
source contents. Visualization and shape artifacts fingerprint graph membership,
not the underlying polygon geometry. Remove all dependent caches after a source
update rather than relying on automatic detection.

## Ownership and Regeneration Summary

| Directory class | Back up? | Safe to delete? | Write access needed? |
|---|---|---|---|
| External source CSVs and shapefiles | Yes | No | Usually no |
| `SFUSD_SHARED/Zones/Optimization` adjacency/crosswalk files | Yes | No | Only distance files are regenerated |
| Cleaned student cache files | No | Yes | Yes when a keyed cache is absent |
| Active graph cache | No | Yes | Yes when a graph or relation is absent |
| Visualization and shape artifacts | No | Yes | Yes when an artifact is absent |
| Optimization and benchmark outputs | As required | Yes, if results are disposable | Yes |
| Checked-in `Config`, sweep, matching, and manual-review YAML | Yes, through Git | No | Only when intentionally editing configuration/review data |

The source files that establish this contract are `Config/Constants.py`,
`optimization/data/loaders.py`, `optimization/data/dataset.py`,
`optimization/config.py`, `optimization/visualization.py`, `metrics/spatial.py`,
`choice/mnl.py`, `benchmark/config.py`, `benchmark/runner.py`,
`benchmark/matching/runner.py`, and `benchmark/results.py`.
