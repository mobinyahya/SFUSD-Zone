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
| **Strategy** | `Strategy.run(dataset, solver) → [ZoneSolution]` | `single`, `recursive`, `iterative_choice`, `mid`, `mid_decomp`, `saa`, `direct_samples`, `short_bursts_choice`, `priced_access`, `stable_cutoff` | subclass `Strategy`, `@register("name")` |

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
        include_citywide_zoning: false
        include_citywide_choice_opt: true
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

## Anchored whole-zone Dantzig-Wolfe decomposition

Run `uv run python -m optimization.run optimization/dantzig_wolfe.example.yaml`.
`strategy: dantzig_wolfe` runs **branch-and-price** over whole-zone columns.
Each column is one label's admissible zone and its exact welfare; the master
covers every vertex once and takes one zone per label. `solver: cp_bool` is
required: the master LP is Gurobi and the subproblem is CP-SAT, and `cp_bool` is
named because it produces the feasible hint and because its feasible set is the
one the decomposition reproduces label by label.

### One feasible set, shared by three places

`zone_family.py` is the single definition of "which sets are legal zones for
label z", and the seeding filter, the column pool and the pricing model all read
it. It is the base model's label-z rows: the centroid and its
`centroid_neighbor_radius` ball forced in, every other label's ball and
centroid excluded, `candidate_zones` (hence `max_distance`) respected,
closer-neighbour contiguity, and the FRL/racial/aggregate-capacity and
school-count rows *in the integer-rounded form `cp_bool` writes*. Because both
sides evaluate the same integers there is no feasibility tolerance anywhere.

Closer-neighbour contiguity implies connectedness and is strictly stronger than
it, so the pricing model needs no rooted flow: connectedness costs one clause
per candidate vertex instead of `2|E|` arc variables with big-`M` capacities.
Anchoring is what keeps a label's pricing problem near its own centroid; the
previous unanchored build returned zones of 340 to 392 of 579 vertices.

### Three welfare definitions

`dw_objective` selects what a zone is worth. All three are additive across a
partition, which is why `include_citywide_choice_opt: false` is required: a
citywide program's seats are contested by every zone, so zone welfare would
stop being additive. The two welfare objectives also need `program_population: All` and
matching geography vintages.

- `mid` -- finite-grid least-cutoff MID welfare, scored by
  `finite_grid_oracle`. `mid_lottery_scale` sets the lottery lattice; original
  floating-point utilities are retained.
- `stable_matching` -- the welfare of the applicant-proposing
  deferred-acceptance outcome on the zone submarket under **one** tie-breaking
  draw, taken from `saa_tie_breaking_method` and `seed`. Maximizing utility over
  a market's stable-admissions polytope returns DA welfare, so the column value
  is one run of the mechanism: no LP, no MIP, and *exactly* realized welfare
  rather than the +195 to +252 over-report the `saa` strategy's aggregated
  recourse row carries.
- `boundary` -- minus half the zone perimeter, reproducing the base compactness
  objective. Its reported objective is the boundary cost, i.e. the negative of
  the internal maximized score.

### Pricing

One CP-SAT model per label and phase, built once and re-aimed between
column-generation rounds: branch fixings are assumptions (released with
`ClearAssumptions`) and the duals are a fresh objective, so nothing is rebuilt
when only the prices move. `dw_pricing_models` reports the count.
`dw_pricing_parallel` prices the labels in threads against one deadline, each
with `workers / Z` search workers; sequential pricing splits the round evenly
instead. A solution callback harvests up to `dw_pricing_columns_per_call`
improving zones per call rather than only the last one.
`dw_pricing_time_limit` caps what one column-generation round spends pricing
and doubles only when a round neither adds a column nor proves a bound --
uncapped, the first round on a real instance eats the whole budget (measured:
240s bought two master LPs).

CP-SAT needs integer coefficients, so the objective is scaled by
`dw_pricing_scale` and rounded *directionally*: utilities up, duals down. The
scaled objective therefore dominates the true reduced welfare, and CP-SAT's
proven bound stays a valid upper bound on it. The price is a reported absolute
tolerance, `dw_bound_slack`, which is what the residual rounding is worth;
raising `dw_pricing_scale` shrinks it and slows each solve. Nothing else is
trusted: a returned zone is re-validated by the family, scored by the exact
oracle, and its reduced cost recomputed by the master.

### Two-zone redraws

Pricing asks a question about one label and the answer is a *bound*, which is
what the guarantee needs. But `Z` labels priced independently are under no
pressure to be mutually complementary, so the pool can grow by hundreds of
individually improving zones while still holding only the one tiling it was
seeded with -- and then every entering column is a degenerate pivot, positive
reduced cost and zero step length. That, not the dual point, is what kept the
master LP pinned to its seed value.

`zone_redraw.py` is the second column source. It fixes an admissible partition,
picks two labels, and re-partitions their joint territory `U = A_a u A_b`
optimally, leaving the other `Z - 2` zones alone -- ReCom's move, with the split
chosen by CP-SAT under the run's real welfare objective and admissibility as a
constraint rather than a rejection test. Every feasible solution of that model
is a complete admissible tiling of `V`, so it is the only source here that grows
the number of ways the master can cover the district. It needs no duals: the two
new zones cover `U` exactly once, so the cover prices on `U` are the same
constant for every solution and maximizing pair welfare also maximizes the
pair's summed reduced cost. The boundary cap is enforced exactly, as an integer
row on the pair's own cut indicators.

It is strictly **additive** to pricing and never replaces it. Its optimum ranges
over a subset of one label's family, so it is a lower bound rather than an upper
one and contributes no `R_z`; `branch_price.py` never writes a node bound from a
redraw, and a node still closes only when every label proved optimality with no
improving zone. `dw_redraw: false` switches it off.

It fires before the search from the seeded incumbent, and inside the search
whenever a round's restricted LP fails to improve or pricing stops adding
columns. `dw_redraw_time_limit` bounds one sweep, `dw_redraw_splits_per_call`
how many splits a call harvests. Solved pairs are memoized on
`(pair, territory, fixings)`, which expires exactly when another pair disturbs
one of the two zones, so sweeps after the incumbent settles cost ~0s.

Measured on Block_2 (501 units, `Z = 6`, 4,154 applicants,
`max_distance: 3.1`, six workers, 600s), seeded with one `cp_bool` feasibility
hint and ReCom off so the comparison isolates the redraw: `mid` goes
10,307.2692 -> 11,822.4615 (**+14.70%**) and `stable_matching` goes
10,344.5486 -> 12,105.1983 (**+17.02%**), with the pool going from one tiling
to 25+ in both. With the redraw off the improvement is **+0.000%** -- exact,
not rounded: 21 rounds and 542-555 columns of strictly positive reduced cost,
and the restricted LP never leaves the hint value. Extending `mid` to 1800s
reaches 12,050.5995 (+16.91%), so ten minutes is budget-limited rather than
neighbourhood-limited, and that partition re-validates independently (cover,
disjointness, per-zone family admissibility, and a fresh oracle re-score
matching to 1e-6).

The certified bound is *identical* with the redraw and without it, and equals
the first-choice constant in every run at every budget. That is the point and
the limitation together: this construction converges on the primal side and
does nothing whatever for the certificate. See
[the formulation](DANTZIG_WOLFE.md) for the traces and the arithmetic.

### The master and its duals

`dw_master_method: barrier` (the default) solves the master LP by barrier
without crossover, so the duals are an *interior* point of the dual polyhedron
rather than a degenerate set-partitioning vertex. `dw_dual_smoothing` adds
Wentges smoothing on top. Both are free in rigour -- the pricing-corrected bound
holds at any dual point with a non-negative boundary multiplier -- and both cost
bit-for-bit reproducibility; `dual` restores the old vertex duals. When a
smoothed point mis-prices, the round re-prices at the raw LP duals before
concluding anything, and moves the smoothing centre there.

Neither is the binding problem. The measured obstruction is that the master
LP's feasible set is essentially the single tiling the pool holds, so *every*
entering column has a zero step length and which dual you read off that point
cannot matter. `dw_overlap_prop` is the repair that addresses the primal: the
Phase-II cover rows become `sum(lambda) + d_v - e_v = 1` with one row rationing
`sum_v w_v (d_v + e_v) <= K`, which makes the LP full-dimensional in `lambda`
so a colliding priced column can enter with a positive step. It is a budget
rather than a penalty because the row's dual *is* the penalty `M`, chosen by
the LP each round instead of guessed in welfare-per-node units. A relaxation at
every `K`, so the bound stays valid and a loose `K` costs bound quality only;
`0.0` is the exact master and the default until `K` has been swept. See
[the formulation](DANTZIG_WOLFE.md) for the tuning table and the diagnostics
each round reports.

### Seeding and budget

`hints: feasible` runs one `cp_bool` feasibility solve, which is the only
seeding step that reliably yields an *admissible* partition, and it is cached
across runs. ReCom then samples from that hint with `dw_recom_samples` /
`dw_recom_chains` / `dw_recom_time_limit` (also capped at 25% of the remaining
budget), and every sampled zone is rejection-filtered against the family. The
filter runs *before* the welfare oracle, so a rejected sample is nearly free,
and it is per zone rather than per partition -- a recombination step rewrites
two zones and leaves the rest of an admissible partition alone. Neither stage is
required: Phase I recovers feasibility from an empty pool by pricing with
deficit artificials.

The last `solve_time_limits` entry is the total wall-clock budget after strategy
entry; `.inf` runs the complete search. `max_iterations` does not truncate
branch-and-price, and only `budget_accounting: wall_clock` is supported.

`OPTIMAL` means the whole tree closed to `tolerance + dw_bound_slack`, not
merely that a collected-column master was solved. On a time limit the result is
`FEASIBLE` or `UNKNOWN` with a valid global upper bound and the incumbent gap.
Bounds use ordinary floating-point LP arithmetic and CP-SAT's integer bounds,
not formal exact-arithmetic certificates. See
[the formulation and finite-convergence proof](DANTZIG_WOLFE.md).

### Known limitation

Pricing is fast -- anchoring plus the CP-SAT formulation took a label's solve
from ~600s for a 0.62% gap to 0.1-8s proved optimal -- and two-zone redraws
now move the incumbent 15-17% off the feasibility hint on Block_2, but the
*bound* side is untouched. The certified bound is the first-choice constant in
every run: for it to stay at 16,830.95 while the restricted LP sits at
12,050.60, the per-label pricing residuals must sum to ~4,780, against total
headroom to the transportation bound of 3,906. Each label separately reports
more available improvement than provably exists, and no six of the zones they
return can tile the district -- which is the per-label question doing exactly
what it is asked, not a defect in the pricing model.

So the reported ~25% gap is mostly certificate slack. `DANTZIG_WOLFE.md` names
the untried repairs on the decomposition's own duals (elastic master,
completability-targeted pricing, a wider redraw neighbourhood); the cheaper
route is to replace the constant the bound falls back on with one of the
zoning-aware bounds developed for `saa`/`priced_access`, which requires nothing
of the decomposition.

## Tests

`tests/` holds data-free unit tests for the level/contiguity/conversion logic:

```bash
uv run python -m pytest optimization/tests
```

End-to-end runs require the SFUSD source data (census shapefiles, student/
school/distance/adjacency files) on the shared non-local data paths.

## Status / follow-ups

- `MNLChoiceModel` evaluates welfare and builds choice cuts using student choice data.
