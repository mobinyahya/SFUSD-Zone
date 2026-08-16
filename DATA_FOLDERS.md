# Data, Scenarios, and Artifacts

This document describes the active data contract for SFUSD-Zone. Source files
are selected through the shared `loaders` package, derived data is stored in
versioned content-addressed caches, and run outputs remain separate from both.

No raw student, school, or census dataset is checked into this repository.
End-to-end runs require access to the configured external sources.

## Configuration Source of Truth

The checked-in loader configuration has three layers:

| Layer | Location | Purpose |
|---|---|---|
| Base | `loaders/configs/base.yaml` | Schema 2 source catalog, root defaults, and central `school_years` registry |
| Scenario | `loaders/configs/scenarios/*.yaml` | Invariant source roles and complete default selectors for one coherent dataset |
| Run | Optimization, benchmark, or assignment YAML | Selects a scenario and overrides selectors or exceptional sources |

Every executable configuration uses this strict shape:

```yaml
data:
  scenario: legacy
  overrides:
    roots: {}
    sources: {}
    filters: {}
```

Unknown keys, roots, catalog IDs, and filters are rejected, as are malformed
source roles. Years are canonical four-character strings such as `"2324"`.
A scenario is either a bundled name or an explicit YAML path. Bundled scenarios
are standalone definitions; they do not inherit from each other.

Catalog IDs such as `optimization.students.enrolled.2223` and
`zones.selected.6` resolve through `loaders/configs/base.yaml`. A scenario or
run may instead use a direct source object:

```yaml
assignment.estimate:
  path: simulation-files/choice-model/estimates.csv
  root: data
  classification: restricted
  geography_vintage: "2010"  # location-bearing tables only
```

Rootless direct paths in checked-in YAML are anchored to the YAML file that
declares them. `anchor_data_config()` preserves this behavior when a parent
configuration is loaded from another working directory. There is no
hostname-based or home-directory input fallback.

The merged selectors choose annual student, program, and school sources from
`school_years`. An unsupported year, population, grade, capacity profile, or
Mission Bay variant fails with the registered alternatives; it never falls
back to another year or file. Explicit entries in `data.overrides.sources` are
for exceptional experimental inputs and take precedence over registry-derived
roles.

## Path Roots

The loader resolves these declared and built-in roots:

| Root | Default | Notes |
|---|---|---|
| `data` | `/share/data/school_choice` | External source datasets |
| `cache` | `/share/data/school_choice/Data/caches` | Shared derived artifacts |
| `student_assignment` | Required, no default | External checkout containing the large generated-zone collection |
| `package` | `loaders/configs/` | Built-in special root; cannot be overridden |
| `repository` | Repository root | Built-in special root; cannot be overridden |

`SFUSD_DATA_ROOT` and `SFUSD_CACHE_ROOT` override the two shared defaults.
Run YAML can override any declared non-special root with
`data.overrides.roots`. Run overrides take precedence over environment
variables. Relative root overrides are anchored to the run YAML.

For example, the generated-zone assignment configs provide the required root
without embedding a developer-specific checkout path in the scenario:

```yaml
data:
  scenario: assignment-generated-zones-2324
  overrides:
    roots:
      student_assignment: <STUDENT_ASSIGNMENT_PATH>
```

The scenario then resolves its direct zone sources below
`<STUDENT_ASSIGNMENT_PATH>/data/zones/`.

To relocate all shared caches for a run, override the cache root:

```yaml
data:
  scenario: legacy
  overrides:
    roots:
      cache: /absolute/path/to/caches
```

There is no active `graphs_dir` optimization setting. Visualization's optional
`artifact_dir` argument acts as an alternate cache root for visualization
geometry only.

## Bundled Scenarios

| Scenario | Intended use |
|---|---|
| `legacy` | Default optimization inputs and the legacy assignment source selection |
| `historical-2324` | 2023-24 assignment runs excluding Mission Bay |
| `mission-bay-2324` | 2023-24 optimization/assignment integration including Mission Bay |
| `assignment-generated-zones-2324` | Large 2023-24 assignment policy sweeps over generated zone CSVs |

Scenarios own invariant sources and complete selector defaults. Run filter
overrides select canonical years/grades, applicant or enrolled students,
preference rounds, special-program behavior, capacity variants, and school
inclusion policy. Optimization additionally selects program population and K-8
and citywide inclusion. Assignment executes exactly one registered year and one
grade per market; optimization accepts multiple years and grades where the
underlying data path supports them.

Both groups default `capacity_scenario` to `programs`. Assignment uses the
capacity column from its registry-selected program table, while optimization
uses the current 2023-24 program table. Selecting an explicit scenario overlays
matching school/program/grade capacities from the shared scenario source;
unmatched programs retain their selected-table capacities.

Both filter groups select `geography_vintage`, currently `"2010"` or `"2020"`.
Location-bearing catalog sources declare their own Census vintage. If it matches
the selected vintage, existing Block, BlockGroup, and Tract columns are retained.
Otherwise students and schools are mapped from WGS84 latitude/longitude to the
selected Block geometry, with parent IDs obtained from the selected crosswalk.
Programs inherit the normalized geography of their school. A point outside all
district polygons has blank Block, BlockGroup, and Tract values. The
`outside_district_students` selector is `ignore` by default and filters students
with blank selected-vintage Blocks; `include` retains them for assignment and
other non-graph workflows. Optimization graph construction fails if retained
students lack the graph's selected geography.

`include_mission_bay` centrally controls Mission Bay handling. When enabled,
the shared table loader derives the `909 -> 999` school alias across student,
program, and school data. Runs do not supply an alias map.

The six large generated-zone run configs share
`assignment-generated-zones-2324`. The scenario exposes 256 zone aliases and
one citywide-zone alias. `assignment/configs/all_zones_selected.yaml` has a
different zone collection and remains explicitly configured.

## External Source Layout

The catalog in `loaders/configs/base.yaml` is the authoritative per-file list.
The default `data` root currently contains these source families:

| Directory | Contents |
|---|---|
| `Data/Cleaned/` | Student, enrollment, program, school, and prepared choice-model CSVs |
| `Data/capacity_management/` | Current capacity files |
| `Data/Tie-breakers/` | CTIP/tie-breaker arrays |
| `shapefiles/` | Census Block geometry and required Shapefile companions |
| `Census/2020/` | Official 2020 TIGER/Line Block, Block Group, and Tract layers, source ZIPs/metadata, crosswalk, and adjacency CSVs |
| `Zones/Optimization/` | 2010 Block crosswalk and Block/BlockGroup/Tract adjacency source CSVs |
| `simulation-files/choice-model/` | Utility estimate CSV/NumPy inputs |
| `simulation-files/zones/` | Shared named assignment zone plans |

The census Shapefile catalog entry includes `.dbf`, `.shx`, and `.prj`
companions. All companion checksums participate in source identity.

Prepared KG round-one and no-special files remain catalogued for historical
reproduction and exceptional experiments. Standard runtime student selection
uses the annual registry source plus selectors, not those prepared files.

The files under `Zones/Optimization/` are source geography, not generated graph
caches. Area distances are now derived from projected census centroids and
stored in `area_distances/v3`; old flat distance CSVs are not active inputs.

The 2020 files are the Census Bureau's county-level 2020 P.L. 94-171 TIGER/Line
releases for San Francisco County (`06075`). The catalog retains each original
ZIP and included ISO metadata as source companions. The selected source bundle
is resolved centrally from `geographies` in `loaders/configs/base.yaml`.

| Layer | Official archive | SHA-256 |
|---|---|---|
| Block | `https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/06_CALIFORNIA/06075/tl_2020_06075_tabblock20.zip` | `0f858e6c7748070b3f1a564ec39a55c2ef6913afb41dc6adb935621c771516b9` |
| Block Group | `https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/06_CALIFORNIA/06075/tl_2020_06075_bg20.zip` | `cb71d9f9c6fb48d3318f8c91e20a4cf3d07f3c2f177a7e73834f0883ebaa8ecd` |
| Tract | `https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/06_CALIFORNIA/06075/tl_2020_06075_tract20.zip` | `58a7c130fdcd1f0efc9e69577a9b1870365c56972ccd33fea69fad581e23cb71` |

The 2020 crosswalk derives BlockGroup and Tract parents from the official Block
`GEOID20`. Adjacency rows were generated with the Shapely `touches` predicate;
the files contain every official area exactly once and symmetric endpoints.

## Checked-In Data

The repository contains only configuration and small public support data:

| Location | Purpose |
|---|---|
| `Config/centroids.yaml` | Named centroid school sets |
| `loaders/configs/base.yaml` | Roots and source catalog |
| `loaders/configs/scenarios/` | Bundled source/filter selections |
| `loaders/configs/manual_block_edges.yaml` | Reviewed closer-neighbor Block edges |
| `loaders/configs/manual_block_edge_additions.yaml` | Explicit missing Block adjacencies |
| `loaders/configs/manual_block_edges_2020.yaml` | Empty placeholder until 2020 Block edges receive manual review |
| `optimization/config.example.yaml` | Current single-run example |
| `benchmark/configs/` and `benchmark/sweep*.yaml` | Benchmark sweep definitions |
| `benchmark/matching/*.yaml` | Matching policy templates |
| `assignment/configs/` | Assignment run and policy configuration |

`Config/` no longer owns manual Block-edge YAML. The canonical runtime files
are under `loaders/configs/` so they are packaged with the loader scenarios.

### Manual Block-Edge Review

`analysis/misc/` contains the human review inputs and compilation tooling. It
is not a runtime data root. The workflow is documented in
`analysis/misc/README.md`.

```bash
uv run python analysis/misc/manual_block_edge_cases.py generate
uv run python analysis/misc/manual_block_edge_cases.py compile
uv run python analysis/misc/manual_block_edge_cases.py compile-additions
```

`compile` writes `loaders/configs/manual_block_edges.yaml`, and
`compile-additions` writes
`loaders/configs/manual_block_edge_additions.yaml`. Their content checksums are
part of Block graph cache identity.

## Content-Addressed Caches

Most derived artifacts use `CacheStore` and this layout below the resolved
`cache` root:

```text
<cache-root>/
  <artifact>/
    v<schema-version>/
      .<sha256-key>.lock
      <sha256-key>/
        manifest.json
        <payloads...>
```

The full SHA-256 key covers:

- artifact name and caller-owned schema version;
- normalized derivation parameters;
- selected source roles, resolved paths, classifications, presence state, and
  current SHA-256 content checksums;
- scenario ID and loader schema version; and
- output classification.

Each manifest records the same identity plus every payload's format, size, and
checksum. Reads validate both manifest identity and payload checksum. Invalid,
missing, or corrupt entries are treated as cache misses. Writes use file locks,
temporary files, `fsync`, and atomic replacement.

The schema version in the directory name is the derived artifact's version. It
is independent of the data-catalog schema version in source manifests.

### Active Namespaces

| Namespace | Payload | Producer/consumer |
|---|---|---|
| `students/v6/<key>/` | `students.csv` | Filtered multi-year optimization students |
| `area_distances/v3/<key>/` | `distances.csv` | Source-aware area centroid distances |
| `graphs/v11/<key>/` | `Block_*.pickle`, `BlockGroup_*.pickle`, or `Tract_0.pickle` | Optimization graph hierarchy |
| `closer_neighbors/v3/` | `closer_neighbors_<level>.pickle` | Geometry distances and strictly closer adjacent nodes |
| `student_program_distances/v4/<key>/` | `distances.pkl` | Assignment student-to-program distances |
| `visualization_geometry/v4/<key>/` | `geometry.pkl` | Dissolved graph-node geometry for maps and spatial metrics |

With the defaults, a graph payload therefore lives at a path like:

```text
/share/data/school_choice/Data/caches/graphs/v11/<sha256>/Block_0.pickle
```

`closer_neighbors/v3` predates the generic namespace layout. It keeps one
locked file per level and stores validated graph/source fingerprint variants
inside that file. It is still source-aware and schema-versioned.

The `students` and `student_program_distances` namespaces are classified
`restricted-derived`. Their directories are created with group-only access
(`0770`) and files/locks with `0660`. Cache references embedded in matching
configuration are path-free; they identify artifact, schema, key, parameters,
roles, classification, and payload.

### Invalidation

Source content changes automatically produce new content-addressed keys. So do
relevant filter, algorithm, graph-policy, and schema-version changes. Manual
deletion is not required for correctness.

To force regeneration, remove only the affected `<key>/` directory (or the
affected closer-neighbor level file). Do not edit a manifest or payload in
place; checksum validation will reject it. Old keys are orphaned rather than
overwritten and may be removed by an explicit storage-retention policy.

## Run Outputs

Run outputs are not caches and are not placed below the cache root.

### Optimization

`optimization.run` defaults to `./optimization_output`; `--output` selects
another directory. A run can contain:

```text
result.json
solution_<level>.json
zone_dict_<level>.json
zone_dict_area_<level>.json
visualization_<stage>.png
stages/<stage-name>/...
solver_logs/...
solver_progress/...
```

The output config snapshot retains the strict `data` block. Spatial metrics
reconstruct `Block_0` through that scenario and the `graphs/v11` cache; they do
not search legacy flat graph paths. Compactness metrics reuse
`visualization_geometry/v4` and project the geometry in memory.

### Benchmark

A sweep writes below `execution.output_dir`, which defaults to
`./benchmark_output`. The default per-task hierarchy is:

```text
<centroids-type>/seed<seed>/frl<frl>_racial<racial>/
  overage<overage>_shortage<shortage>/<strategy-and-config-hash>/
```

`execution.output_template` can replace this hierarchy. Each task stores
`benchmark_manifest.json`, `result.json`, root-level final-solution aliases,
and every recursive/iterative result under `stages/`. Aggregation writes
`summary.csv` and `stages.csv` at the sweep root by default.

The benchmark config hash includes optimization semantics, scenario semantics,
and current source manifests. It deliberately excludes only the cache-root
location, so relocating equivalent caches does not change task identity.

### Assignment and Matching

Standalone assignment output locations come from explicit `paths` settings in
the run config. The saved `config.json`/`config.yaml` is a replayable snapshot
of the strict external configuration. Runtime-only resolved input keys and
`data-provenance` are not written back into that snapshot.

Benchmark matching writes this subtree in a task or stage directory:

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
  precomputed/...
```

Multiple matching templates use `matching/<policy>/` and
`matching/precomputed/<policy>/`. The generated config records the path-free
`student_program_distances/v4` reference used by assignment metrics.

Within a matching invocation, `StudentAssignmentSession` can retain separate
markets for distinct immutable assignment source identities and reconfigure a
matching market when only zones or policy settings change. Source identity
includes assignment filters and checksummed immutable table/estimate sources;
zone files are intentionally excluded so stage and policy runs can reuse those
loaded tables safely.

## Retired Locations

Current code does not read or write these pre-centralization locations:

| Retired location | Replacement |
|---|---|
| `Data/Cleaned/Cleaned_Students_*.csv` | `Data/caches/students/v6/<key>/students.csv` |
| `Zones/Optimization/distances_b2b_schools.csv` and `distances_bg2bg.csv` | `Data/caches/area_distances/v3/<key>/distances.csv` |
| `Zones/Optimization/Graphs/` | `Data/caches/graphs/v11/<key>/` |
| `Data/Computed/visualization_artifacts/` | `Data/caches/visualization_geometry/v4/<key>/` |
| `Data/Computed/shape_metric_artifacts/` | Shared visualization geometry plus in-memory projection |
| `Data/Computed/Graphs/` | Scenario-backed `graphs/v11` reconstruction |
| `Config/manual_block_edges.yaml` | `loaders/configs/manual_block_edges.yaml` |
| `Config/manual_block_edge_additions.yaml` | `loaders/configs/manual_block_edge_additions.yaml` |

Historical result directories may still contain snapshots that mention old
paths. Keep those artifacts for provenance, but do not use them as templates
for new runs.

## Implementation References

The code defining this contract is:

- `loaders/config.py`: strict scenario resolution and source manifests;
- `loaders/cache.py`: cache identity, validation, locking, and permissions;
- `loaders/configs/base.yaml`: schema 2 roots, source catalog, and year registry;
- `optimization/data/loaders.py`: student and area-distance artifacts;
- `optimization/data/dataset.py`: graph namespaces;
- `optimization/data/closer_neighbors.py`: closer-neighbor variant store;
- `optimization/visualization.py`: shared geometry artifacts;
- `assignment/student_assignment/data_interfaces/students.py`: assignment
  distance artifacts; and
- `benchmark/matching/runner.py`: matching snapshots and market reuse.
