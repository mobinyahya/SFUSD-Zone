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
| **Data** | `Dataset` → `ZoneProblem` | Predefined Block / BlockGroup hierarchies | extend `data/loaders.py` / `graph_builder.py` |
| **Solver** | `Solver.solve(problem) → ZoneSolution` | `cp_int`, `cp_bool`, `mip`, `recom`, `relaxed_recom`, `short_bursts` | subclass `Solver`, `@register("name")` |
| **Strategy** | `Strategy.run(dataset, solver) → [ZoneSolution]` | `single`, `recursive`, `iterative_choice`, `overlapping`, `cutoffs`, `welfare` | subclass `Strategy`, `@register("name")` |

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
  balancing the population selected by `population_type` with progressively
  relaxed imbalance. Requested sizes are upper targets because KaHIP may return
  fewer nonempty partitions.
- **Shared graph cache.** Parameter-specific graph namespaces are stored below
  `/share/data/school_choice/Zones/Optimization/Graphs` by default. The cache
  key includes ingestion parameters and the graph-partition policy.
- **Overlapping school zones.** The `overlapping` strategy ignores
  `centroids_type`, solves one partial zone per eligible school with one worker,
  and runs those solves concurrently within the configured worker budget. It
  fixes only nodes belonging to exactly one partial zone and outside every
  partial-zone boundary band, then solves the complete all-school problem with
  the configured solver and full worker budget. Schools must resolve to unique
  graph nodes.
- **Stable welfare.** The `welfare` strategy excludes citywide schools, solves
  isolated finite-grid DA-STB markets, and maximizes expected fixed-point
  cardinal utility. A top-rank recurrence and all-rank preference-interval cuts
  provide global upper bounds; timed runs remain `FEASIBLE` unless bounds close.

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
`/share/data/school_choice/Data/Computed/visualization_artifacts/` as
`geometry_<level>_<fingerprint>.pkl` and `.json`. This does not use benchmark,
choice, or heatmap code.

Switch granularity, solver, or strategy purely in the config:

```yaml
levels: ['Block_2', 'Block_1', 'Block_0']
solver: 'cp_bool'
strategy: 'recursive'
```

## Tests

`tests/` holds data-free unit tests for the level/contiguity/conversion logic:

```bash
uv run python -m pytest optimization/tests
```

End-to-end runs require the SFUSD source data (census shapefiles, student/
school/distance/adjacency files) on the shared non-local data paths.

## Status / follow-ups

- `MNLChoiceModel` needs the estimate/demographics CSVs wired in;
  `DistanceChoiceModel` is the data-free default.
