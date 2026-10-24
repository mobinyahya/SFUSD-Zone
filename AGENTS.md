# CLAUDE.md

SFUSD-Zone optimizes school district zoning for San Francisco Unified School District using OR-Tools CP-SAT (primary) and Gurobi MIP. It creates geographically contiguous zones balancing demographic diversity, geographic access, school capacity, program access, and school quality.

## Commands

All commands use `uv`. Python 3.13+ required.

```bash
uv sync                          # Install dependencies
uv run python -m optimization.run optimization/config.example.yaml  # Single optimization run

# Benchmarking (from project root)
uv run python -m benchmark.run path/to/sweep.yaml
uv run python -m benchmark.run path/to/sweep.yaml --mode metrics
```

## Directory Structure

- `Config/` - centroids.yaml, Constants.py
- `loaders/` - shared source catalog, scenarios, normalized table readers, and content-addressed caches
- `optimization/` - config.py, problem.py, solution.py, solvers/, strategies/, data/
- `assignment/` - student-assignment simulation, policy configs, and assignment analysis
- `benchmark/` - config.py, runner.py, results.py, parallel.py
- `metrics/` - calculator.py, diversity.py, distance.py, programs.py, quality.py, choice.py, spatial.py

## Data Registry and Selectors

`loaders/configs/base.yaml` schema 2 is the single source catalog and central
`school_years` registry. Scenarios under `loaders/configs/scenarios/` provide
invariant sources and complete selector defaults; `data.overrides.filters`
selects a run, while explicit source overrides are reserved for exceptional
experimental inputs and take precedence over registry-derived sources.

Optimization selectors are canonical `years`, `grades`, `student_population`,
`rounds`, `special_programs`, `program_population`, `capacity_scenario`, and the
`include_k8`, `include_citywide_zoning`, `include_citywide_choice_opt`, and
`include_mission_bay` flags. The two citywide selectors are independent and
documented in `loaders/README.md`. Optimization alone also selects an optional
`enrollment_share` (default `null`): the share of each attendance area's
residents expected to enrol, which scales the students the capacity
(`overage`/`shortage`) rows count and nothing else. `include_citywide_zoning` governs the school
table, so it decides whether a citywide school occupies a graph node and joins
the capacity and school-count balance of the zone containing it.
`include_citywide_choice_opt` governs the welfare markets built by
`optimization/data/mid.py` and `optimization/data/saa.py`, and the MNL zoning
utility in `choice/mnl.py`; those markets draw their programs from the
*assignment* table, which carries no such selector, so they apply it
themselves. The default pairing is `zoning: false, choice_opt: true`: no zone
owns a citywide school, but every student may still choose one, which the
markets model with `school_node=None` and every oracle reads as reachable from
every zone. Both groups
also select an optional `frl_estimate`. Assignment
uses canonical `year`, a scalar `grade`, `student_population`, `rounds`,
`special_programs`, `capacity_profile`, `capacity_scenario`, and
`include_mission_bay`: one year and one grade per market. Unsupported registry
combinations fail rather than falling back.

**Enrolled is not assigned.** School **899** ("Central Enrollment",
`loaders.tables.NOT_ENROLLED_SCHOOL_ID`) means enrolled nowhere. It is never
a school. For 2024-25 on, `enrolled_<year>.csv` holds only seated students
whose fall enrolment record (demographics `SCHOOL_CODE`/`GRADE`) is at
kindergarten at a real school. A record at 899, a missing record, or a record
at another grade all mean not enrolled. Its
`enrolled_idschool`/`enrolled_programcode` come from that record,
while `final_school`, `r1_*`, and the student table's `enrolled_idschool`
stay the Main Round seat. The assignment group's enrolled population also
drops blank-or-899 rows at load, which is what removes 2023-24's 794
non-enrollees. Optimization keeps them. See `loaders/README.md`.

`capacity_scenario` defaults to `programs` for both groups. Assignment uses the
capacity values in its year/profile-selected program table; optimization uses
the current 2023-24 program table. Explicit scenarios such as `A` through `D`
overlay matching school/program/grade capacities from the shared scenario table.
The district's 2026-27 KG workbook scenarios (`ideal_baseline_2627`,
`ideal_bessie_ge_2627`, `ideal_*_w2w_2627`; converted by
`analysis/data_prep/convert_capacity_workbook.py`) also set each listed school's
`category`: a school with any attendance-area seats is `Attendance`, so it owns
a graph node under `program_population: All`; the rest are `Citywide`.
They also label each program's attendance-area seats, which loaded programs
carry as `attendance_area_capacity` and the `aa_capacity_match` objective scores.
The workbook's new Mandarin Immersion TK-8 (placeholder school `400`) is in no
data file; both groups' `include_mandarin_immersion` (default `false`) adds it
from base `planned_schools`. Its site and its ME/MN (non-native/native) seat
split are still null, so setting it true fails at load, naming what is missing.

**Mission Bay ES is school 1731** (`loaders.tables.MISSION_BAY_SCHOOL_ID`). The
2023-24 tables and the 2023-24 choice estimate carry the old placeholders 999
and 909; with `include_mission_bay: true` every loader reads them as 1731, and
with `false` all three are dropped.

`frl_estimate` defaults to `null`, which retains the student table's FRL fields.
Named estimates come from the central `student_frl_estimates` registry. The
`updated_2526` estimate derives exact block rates from FRL/student counts and
requires `geography_vintage: "2020"`; missing and zero-student blocks fall back
to the student table.

Both groups select `geography_vintage` (`2010` or `2020`). Same-vintage Census
columns are retained; other location-bearing sources are spatially mapped from
lat/lon. Points outside the district geometry retain blank Census geography.
`outside_district_students` defaults to `ignore`, which filters these students;
`include` retains them for non-graph workflows, but optimization graph
construction rejects included students without its selected geography. Programs
inherit geography from their school.

## Graph Object Structure

The optimization uses NetworkX undirected graphs. Nodes are geographic areas and edges are adjacency. `optimization/data/graph_builder.py` creates these predefined hierarchies:
- **Block_0** - Direct Census Block units
- **Block_0.5** - Up to 3,000 nodes, built from Block_0 (KaHIP returns ~1,850 nodes,
  no aggregate above ~17 students against Block_1's ~46). Block_1 is still built
  from Block_0, so the two are not nested. Parents other than `depth - 1` are
  declared in `LEVEL_PARENT_DEPTHS` (`optimization/levels.py`) and read through
  `LevelSpec.parent()`.
- **Block_1** - Up to 1,200 nodes, built from Block_0
- **Block_2** - Up to 579 nodes, built from Block_1
- **Block_3** - Up to 250 nodes, built from Block_2
- **Block_4** - Up to 125 nodes, built from Block_3
- **BlockGroup_0** - Direct Census BlockGroup units
- **BlockGroup_1** - Up to 250 nodes, built from BlockGroup_0
- **BlockGroup_2** - Up to 125 nodes, built from BlockGroup_1
- **Tract_0** - Direct Census Tract units

### Node Attributes

```python
{
    'area_id': int,              # Census GEOID for the selected unit
    'ge_students': float,        # General education students (count)
    'ge_capacity': float,        # GE school seats
    'all_prog_students': float,  # All program students
    'all_prog_capacity': float,  # All program capacity
    'ge_enrolling_students': float,        # ge_students scaled by enrollment_share (capacity rows only)
    'all_prog_enrolling_students': float,  # all_prog_students scaled likewise
    'num_schools': int,
    'school_ids': list,
    'FRL': float,                # Free/Reduced Lunch students (count, not proportion)
    'lat': float,
    'lon': float,
    # Ethnicity counts (see AREA_ETHNICITIES in Constants.py):
    'Ethnicity_Black_or_African_American': float,
    'Ethnicity_Hispanic/Latinx': float,
    'Ethnicity_White': float,
    'Ethnicity_Asian': float,
    'Ethnicity_Pacific_Islander': float,
    'block_ids': list,           # Only on aggregated graphs: original base-unit GEOIDs
}
```

### Graph-Level Attributes (G.graph)

```python
{
    'distance_dict': dict,  # {node_idx: {node_idx: distance}} - indexed by node index, NOT area_id
    'school_data': dict,    # {school_id: school_info_dict}
    'F': float,             # District-wide FRL proportion (0-1)
    'R': dict,              # District-wide ethnicity proportions {ethnicity: proportion}
    'partition': dict,      # Aggregated graphs only: {parent_node: aggregated_node}
}
```

### Edges

Unweighted undirected edges from shapefile geometry adjacency (touches). Used to enforce zone contiguity.

### Hierarchical Aggregation

`optimization/data/graph_builder.py` builds multi-level graphs:
1. `build_base_graph()` creates depth 0 directly from the selected census units.
2. Each coarser level uses its parent (the immediate finer level, unless `LEVEL_PARENT_DEPTHS` says otherwise) rather than repartitioning level 0.
3. School nodes are removed and retained as singleton vertices. KaHIP strong mode partitions the remaining nodes using the population selected by `program_population`; imbalance starts at 0.8 and doubles until validation passes.
4. `aggregate()` sums node attributes, flattens base area IDs, recomputes distances, and derives every child edge from crossing parent edges. This also reconnects school nodes to every aggregate containing one of their former neighbors.
5. Requested node counts include school singletons and are upper targets because KaHIP can return fewer nonempty partitions.

Graphs are cached by exact source contents, data filters, and partition-policy
parameters under `/soalnas/share/data/school_choice/Data/caches/graphs/v<N>/<sha256>/`,
where `<N>` is `graph_builder.GRAPH_CACHE_SCHEMA_VERSION` (16 at time of writing).

## Benchmarking

### CLI Commands

Benchmarking is configured from one simulation sweep YAML file. The same entrypoint can run the full sweep or recalculate metrics from saved stage results. Aggregation always runs after either mode.

- `mode: run` - Generate tasks from YAML and run the full optimization sweep.
- `mode: metrics` - Reconstruct saved `ZoneSolution` stages, rewrite `result.json` with updated metrics, and aggregate outputs.

See `benchmark/sweep.example.yaml` for the YAML shape.

### Output Structure

Each run produces a folder at `{centroids_type}/seed{seed}/frl{frl}_racial{racial}/...`:
```
benchmark_manifest.json  # Task id, config hash, stage paths, status, timing
result.json              # Status, metrics, zone_data, config, run metadata
zone_dict_<level>.json   # Final root-level assignment alias
zone_dict_area_<level>.json
solution_<level>.json    # Status, metadata, and the generating config ("config")
zoning_config.yaml       # That config, runnable by optimization.run as is
stages/<stage>/<files>   # Every recursive/iterative level result, each with both
```

Aggregation produces `summary.csv` with one row per run and `stages.csv` with one row per saved stage.

### Key Classes

- `SimulationSweep` (benchmark/config.py) - YAML-backed sweep definition
- `BenchmarkTask` (benchmark/config.py) - Concrete optimization task
- `run_sweep` (benchmark/parallel.py) - Capacity-aware process executor with worker recycling
- `MetricsCalculator` (metrics/calculator.py) - Optimization-native metrics over `ZoneSolution` stages

## Config Reference (`optimization/config.example.yaml`)

| Parameter | Example | Description |
|---|---|---|
| `centroids_type` | `'5-zone-AF'` | Zone count and centroid configuration; `all` makes every school the zoning problem holds a centroid (the attendance-area schools, plus citywide ones under `include_citywide_zoning`), in ascending id order |
| `frl_dev` | `0.3` | Max FRL deviation from district average |
| `racial_dev` | `0.3` | Max racial/ethnic deviation |
| `solver` | `'cp_int'` | Solver: cp_int, cp_bool, or mip |
| `levels` | `['BlockGroup_1','BlockGroup_0']` | Hierarchical solve order |
| `solve_time_limits` | `[30, 30]` | Seconds per recursive level; iterative strategies use the last entry as one total run budget |
| `budget_accounting` | `wall_clock` | What that budget pays for: `wall_clock` (everything) or `solver_time` (iterative solves only) |
| `overage` / `shortage` | `0.8` / `0.2` | Capacity tolerance (proportion) |
| `capacity_scenario` | `programs` | Program-table capacities, or an explicit scenario overlay |
| `hints` | `voronoi` | Warm-start method: `feasible`, `voronoi`, or `none` |
| `seed` | `42` | Solver seed |

Cache paths carry the artifact's schema version, which lives in code and moves
without this file: `graph_builder.GRAPH_CACHE_SCHEMA_VERSION` (16),
`initial_solutions.FEASIBLE_HINT_CACHE_SCHEMA_VERSION` (3). Read the constant
rather than the number quoted here.

Graph cache path: `.../caches/graphs/v16/<sha256>/`.

Feasible-hint cache path: `.../caches/feasible_hint/v3/<sha256>/`, keyed by the
*feasible set* and nothing else: `feasibility_fingerprint(problem)` plus
`_hint_model_identity` (the hint solver, `CP_SAT_SCALE`,
`COEFFICIENT_ROUNDING`, and `centroid_neighbor_radius`). Search settings are
excluded by construction — `workers`, `seed`, `feasible_hint_time_limit` and
the CP-SAT tuning knobs (`linearization_level`, `cp_model_probing_level`,
`symmetry_level`, `cp_sat_search_strategy`) all change how hard a run looks,
never which assignments satisfy the model. So a hint solved once with many
workers and a long limit is served to every later run whatever its own budget,
which is what makes warming hints up front worth doing. Every hint is
revalidated with `check_zoning` on write and on read, so a stale or
solver-specific one is rejected rather than reused.

`boundary_prop` is part of that fingerprint, and `Dataset.problem_for` does
*not* set it — it returns `-1.0` (constraint off). Every strategy builds its
problems through `Strategy.problem_for`, which applies its own
`options["boundary_prop"]` and, for the welfare strategies
(`capacity_balance = False`), switches off `overage`/`shortage`. Anything that
builds a problem outside a strategy and expects to share the strategy's hints
has to go through the same hook (`analysis/warm_feasible_hints.py` does), or it
keys a different feasible set.

Zoned-transport bound cache path: `.../caches/zoned_transport_bound/v2/<sha256>/`
(`zoned_transport.ZONED_TRANSPORT_CACHE_SCHEMA_VERSION`), keyed by the zoning feasible set, the market, the contiguity description (`neighbors` or `flow`), and `centroid_neighbor_radius`. `zoned_transport_workers` is deliberately excluded: threads change how long the LP takes, not its value.
