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
| **Data** | `Dataset` → `ZoneProblem` | Block / BlockGroup at any depth | extend `data/loaders.py` / `graph_builder.py` |
| **Solver** | `Solver.solve(problem) → ZoneSolution` | `cp_int`, `cp_bool`, `mip`, `local_search` (stub), `recom`, `relaxed_recom`, `short_bursts_recom` | subclass `Solver`, `@register("name")` |
| **Strategy** | `Strategy.run(dataset, solver) → [ZoneSolution]` | `single`, `recursive`, `iterative_choice` | subclass `Strategy`, `@register("name")` |

The two layers communicate only through `ZoneProblem` (a solver-agnostic
instance) and `ZoneSolution` (its result), so solvers and strategies vary
independently.

## Key design points

- **Solver-owned implementations.** `cp_bool`, `cp_int`, and `mip` each build
  their own solver-native models from `ZoneProblem`. The CP-SAT solvers share
  common helpers, while the Gurobi MIP implementation stays separate.
- **ReCom heuristics.** `recom`, `relaxed_recom`, and `short_bursts_recom`
  start from an explicit hint when provided. When
  no hint exists, strategies can build a cached `Block_0` initial assignment by
  solving `BlockGroup_1` with loose `cp_bool` constraints per `centroids_type`,
  then convert that seed to the active level.
- **Strict contiguity** (`data/contiguity.py`) uses the shortest-path-tree
  formulation: a non-centroid node may join a zone only if a strictly-closer
  neighbor does too. Same module validates contiguity and repairs assignments.
- **Levels are data, not file edits.** A `LevelSpec` is `(unit, depth)`
  (`"BlockGroup_0"`, `"Block_2"`). `Dataset` generates and caches whatever
  graph a level needs; `LevelConverter` maps assignments between any two
  levels (across depth or unit).

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
levels: ['Block_2', 'Block_0']   # arbitrary unit/depth sequence
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

- `local_search` is an interface stub (seed + contiguity repair); real search
  logic plugs in behind the same interface.
- `MNLChoiceModel` needs the estimate/demographics CSVs wired in;
  `DistanceChoiceModel` is the data-free default.
