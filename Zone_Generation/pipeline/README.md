# Zoning Optimization Pipeline

A standalone, three-layer rewrite of the zone-generation optimizer. Every layer
is swappable in isolation, levels are first-class, and graph generation, level
conversion, and contiguity all live inside the package.

```
PipelineConfig ──▶ Dataset ──build──▶ ZoneProblem ──┐
                  (Data layer)                       │
                                                     ▼
              Strategy ◀── composes ──▶ Solver ──▶ ZoneSolution ──▶ JSON
            (orchestration)            (algorithm)
```

This package does **not** touch the legacy `Zone_Generation/Optimization`
package, the benchmark runner, or the website. It is a parallel replacement to
migrate onto gradually.

## The three layers

| Layer | Contract | Built-ins | Add a new one |
|-------|----------|-----------|---------------|
| **Data** | `Dataset` → `ZoneProblem` | Block / BlockGroup at any depth | extend `data/loaders.py` / `graph_builder.py` |
| **Solver** | `Solver.solve(problem) → ZoneSolution` | `cp_int`, `cp_bool`, `mip`, `local_search` (stub) | subclass `Solver`, `@register("name")` |
| **Strategy** | `Strategy.run(dataset, solver) → [ZoneSolution]` | `single`, `recursive`, `iterative_choice` | subclass `Strategy`, `@register("name")` |

The two layers communicate only through `ZoneProblem` (a solver-agnostic
instance) and `ZoneSolution` (its result), so solvers and strategies vary
independently.

## Key design points

- **Constraints written once.** `solvers/constraints.py` defines assignment,
  centroid-fixing, strict contiguity, capacity, diversity, and school-count in
  terms of a tiny `ModelBackend` interface. CP-SAT and Gurobi each implement
  that interface — no duplicated constraint math.
- **Strict contiguity** (`data/contiguity.py`) uses the shortest-path-tree
  formulation: a non-centroid node may join a zone only if a strictly-closer
  neighbor does too. Same module validates contiguity and repairs assignments.
- **Levels are data, not file edits.** A `LevelSpec` is `(unit, depth)`
  (`"BlockGroup_0"`, `"Block_2"`). `Dataset` generates and caches whatever
  graph a level needs; `LevelConverter` maps assignments between any two
  levels (across depth or unit).

## Running

```bash
uv run python -m Zone_Generation.pipeline.run Zone_Generation/pipeline/config.example.yaml -o ./out
```

Switch granularity, solver, or strategy purely in the config:

```yaml
levels: ['Block_2', 'Block_0']   # arbitrary unit/depth sequence
solver: 'cp_bool'
strategy: 'recursive'
```

## Tests

`tests/` holds data-free unit tests for the level/contiguity/conversion logic:

```bash
uv run python -m pytest Zone_Generation/pipeline/tests
```

End-to-end runs require the SFUSD source data (census shapefiles, student/
school/distance/adjacency files) on the configured `is_local` path.

## Status / follow-ups

- `local_search` is an interface stub (seed + contiguity repair); real search
  logic plugs in behind the same interface.
- `MNLChoiceModel` needs the estimate/demographics CSVs wired in;
  `DistanceChoiceModel` is the data-free default.
- Consumers (benchmark, website) still use the legacy package; migrating them
  and deleting `Zone_Generation/Optimization` is a separate step.
