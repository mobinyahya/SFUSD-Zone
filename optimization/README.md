# Zoning Optimization Optimization

A standalone, three-layer rewrite of the zone-generation optimizer. Every layer
is swappable in isolation, levels are first-class, and graph generation, level
conversion, and contiguity all live inside the package.

```
OptimizationConfig ──▶ Dataset ──build──▶ ZoneProblem ──┐
                  (Data layer)                       │
                                                     ▼
              Strategy ◀── composes ──▶ Solver ──▶ ZoneSolution ──▶ JSON
            (orchestration)            (algorithm)
```

This top-level package is the optimization implementation used by the benchmark
runner.

## The three layers

| Layer | Contract | Built-ins | Add a new one |
|-------|----------|-----------|---------------|
| **Data** | `Dataset` → `ZoneProblem` | Predefined Block / BlockGroup hierarchies and `Tract_0` | extend `data/loaders.py` / `graph_builder.py` |
| **Solver** | `Solver.solve(problem) → ZoneSolution` | `cp_int`, `cp_bool`, `cp_single_zone`, `mip`, `recom`, `relaxed_recom`, `short_bursts`, `adaptive_short_bursts` | subclass `Solver`, `@register("name")` |
| **Strategy** | `Strategy.run(dataset, solver) → [ZoneSolution]` | `single`, `recursive`, `iterative_choice`, `mid`, `mid_decomp`, `saa`, `short_bursts_choice` | subclass `Strategy`, `@register("name")` |

The two layers communicate only through `ZoneProblem` (a solver-agnostic
instance) and `ZoneSolution` (its result), so solvers and strategies vary
independently.

## Key design points

- **Solver-owned implementations.** `cp_bool`, `cp_int`, `mip`, and the ReCom
  family each build their own solver-specific representation from `ZoneProblem`.
  The CP-SAT solvers share common helpers, while the Gurobi MIP implementation
  stays separate.
- **ReCom label semantics.** ReCom uses centroids to determine the zone count and
  to generate an optional Voronoi hint, but does not enforce centroid anchors or
  `max_distance` during the walk. Explicit `candidates` and `fixed` restrictions
  are still hard constraints.
- **Strict contiguity** (`data/contiguity.py`) uses the shortest-path-tree
  formulation: a non-centroid node may join a zone only if a strictly-closer
  neighbor does too. Same module validates contiguity and repairs assignments.
- **Levels are data, not file edits.** A `LevelSpec` is `(unit, depth)`
  (`"BlockGroup_0"`, `"Block_2"`). `Dataset` generates and caches whatever
  graph a level needs; `LevelConverter` maps assignments between any two
  levels (across depth or unit).
- **Nested graph hierarchy.** Level 0 contains the source census units. Every
  coarser level is built from its immediate finer parent with KaHIP strong mode.
  School nodes remain singleton vertices; only non-school nodes are partitioned,
  balancing the population selected by `program_population` with progressively
  relaxed imbalance. Requested sizes are upper targets because KaHIP may return
  fewer nonempty partitions.
- **Shared graph cache.** Parameter-specific graph namespaces are stored below
  `/soalnas/share/data/school_choice/Data/caches/graphs/v11` by default. The cache key
  includes scenario filters, exact source contents, and the partition policy.
- **Cached feasible hints.** `hints: feasible` runs an objective-free CP-SAT
  solve for a warm start, using the shared `workers` setting (default: 8).
  Results are stored below
  `/soalnas/share/data/school_choice/Data/caches/feasible_hint/v2` and keyed by a
  fingerprint of the feasibility model alone (candidate zones, balance inputs,
  closer-neighbor supports, edges, fixed nodes, plus `centroid_neighbor_radius`,
  which fixes centroid neighborhoods). Search settings are deliberately not in
  the key: `seed`, `workers`, `feasible_hint_time_limit`, and the CP-SAT tuning
  parameters change only how hard the search works, not which assignments are
  feasible, so every run of one model shares one hint. Cached assignments are
  re-validated against the problem before use.

## Running

```bash
uv run python -m optimization.run optimization/config.example.yaml -o ./out
```

Save a PNG visualization of the final solution:

```bash
uv run python -m optimization.run optimization/config.example.yaml -o ./out --visualize
```

For recursive zoning or iterative choice, save every produced stage:

```bash
uv run python -m optimization.run optimization/config.example.yaml -o ./out --visualize --viz-stages all
```

Rendered maps are written to the optimization output directory as
`visualization_<stage>.png`. Cached geometry artifacts are written under
`/soalnas/share/data/school_choice/Data/caches/visualization_geometry/v4/<sha256>/` as
`geometry.pkl` with a validated `manifest.json`. This does not use benchmark,
choice, or heatmap code.

Switch granularity, solver, or strategy purely in the config:

```yaml
levels: ['Block_2', 'Block_1', 'Block_0']
solver: 'cp_bool'
strategy: 'recursive'
data:
  scenario: legacy
  overrides:
    filters:
      optimization:
        years: ["2122", "2223", "2324"]
        grades: [KG]
        student_population: enrolled  # applicant | enrolled
        rounds: [1]                    # or all
        special_programs: include     # include | exclude_only_special | exclude_any_special
        program_population: GE
        capacity_scenario: programs    # programs | A | B | C | D
        include_k8: false
        include_citywide: false
        include_mission_bay: true
        geography_vintage: "2020"
        outside_district_students: ignore  # ignore | include
```

`loaders/configs/base.yaml` schema 2 is the central source catalog and
`school_years` registry. A scenario supplies invariant geography/capacity roles
and complete selector defaults; run filter overrides select annual registry
sources. Years must be canonical strings and grades canonical labels. Multiple
years and grades are accepted where the optimization ingestion supports them.
Missing registry combinations fail and never use a neighboring year or legacy
fallback.

`capacity_scenario: programs` is the default and aggregates `capacity` from the
current 2023-24 program table into GE and all-program school capacity. Explicit
scenarios such as `A` through `D` overlay matching school/program/grade rows
from the central scenario table before aggregation.

Spatial conversion assigns Census geography only when a student point intersects
the selected district geometry. `outside_district_students: ignore` filters rows
with blank Blocks and is the default. `include` keeps them available to non-graph
consumers, but optimization graph construction fails if an included student has
no geography for the graph unit.

Explicit `data.overrides.sources` take precedence over registry-derived roles,
but are intended only for exceptional experimental inputs. The shared student
normalizer sorts selected rounds and retains one row per unique student whose
filtered choices are nonempty in any selected round. Its authoritative choices
come from that student's earliest remaining selected round.

## Tests

`tests/` holds data-free unit tests for the level/contiguity/conversion logic:

```bash
uv run python -m pytest optimization/tests
```

End-to-end runs require the SFUSD source data (census shapefiles, student/
school/distance/adjacency files) on the shared non-local data paths.

## Status / follow-ups

- `MNLChoiceModel` evaluates welfare and builds choice cuts using student choice data.
