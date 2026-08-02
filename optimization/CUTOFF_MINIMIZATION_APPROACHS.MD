# Cutoff Minimization Approaches

## 1. Scope and terminology

This document records every cutoff-zoning approach investigated for
`optimization/config.example.cutoffs.yaml`, including unsuccessful methods.
The spelling of this filename follows the requested artifact name.

The benchmark instance has:

| Quantity | Value |
|---|---:|
| Geographic level | `Block_2` |
| Graph vertices / edges | 432 / 939 |
| Zones | 6 |
| Graph schools | 59 |
| Zone-restricted cutoff-market schools | 58 |
| Applicants represented in the cutoff market | 3,803 |
| Applicant-school preference entries | 220,574 |
| Market capacity | 3,426 |
| Lottery scale | 20 |
| Priority tiers | 8 |

All 58 modeled schools are zone-restricted because `remove_city_wide: true`.
The six markets are consequently independent after a zoning is fixed.

Two cutoff concepts must be distinguished:

1. The optimized finite-grid objective uses integer cutoff units at scale
   `L=20`. It is an expected-demand score-limit approximation. Its least
   capacity-feasible vector is called **grid-minimal**.
2. The continuous Azevedo-Leshno calculation uses a common continuous STB
   lottery. The theoretical equilibrium requires exact market clearing at a
   positive cutoff; the implementation verifies this numerically to tolerance.
   This is the **continuous stable score-limit equilibrium** target.

The best known zoning has grid objective `2269 / 20 = 113.45`. The saved valid
global lower bound is `333 / 20 = 16.65`; therefore `113.45` is an incumbent,
not a proved finite-instance optimum.

## 2. Common market formulation

Let `I` be applicants, `S` schools, `Z` zones, and `L` the lottery scale.
Applicant `i` has a strict preference order after the fixed MNL plus Gumbel
utility realization, and integer school priority tier `p_is`. Larger priority
is better. For cutoff `c_s`, rejected lottery mass is

```text
t_is(c_s) = min(L, max(0, c_s - L p_is)).
```

After filtering the preference order to accessible schools
`s_i1, ..., s_iK`, define

```text
r_i0 = L
r_ik = min(r_i,k-1, t_i,s_ik(c_s_ik))
a_i,s_ik = r_i,k-1 - r_ik.
```

`a_is / L` is the assignment measure at school `s`. Capacity is

```text
sum_i a_is <= L q_s.
```

The zoning objective is `min sum_s c_s / L` subject to the geographic master
constraints and these market-capacity conditions.

## 3. Baseline: monolithic CP-SAT

### 3.1 Formulation

The first implementation embedded the complete cutoff recurrence in the
zoning CP-SAT model. It introduced:

- Boolean `x_zv` for vertex-zone assignment.
- Boolean same-zone indicators for applicant-node/school pairs.
- One cutoff variable per school.
- Shared threshold variables for each school-priority pair.
- One effective threshold, cumulative rejected mass, and assignment-mass
  variable for every applicant-school preference entry.
- One capacity constraint per school.

The implementation remains in
`optimization/solvers/cpsat.py::_add_cutoff_objective`. The normal `cutoffs`
strategy no longer invokes it; it can still be run directly by constructing a
`ZoneProblem`, attaching a `CutoffMarket`, and calling the CP solver:

```python
from optimization.config import OptimizationConfig
from optimization.data.cutoffs import build_cutoff_market

config = OptimizationConfig.from_yaml(
    "optimization/config.example.cutoffs.yaml"
)
dataset = config.make_dataset()
problem = dataset.problem_for("Block_2")
problem.cutoff_market = build_cutoff_market(
    dataset,
    problem,
    assignment_config=config.cutoff_assignment_config,
    ctip_path=config.cutoff_ctip_path,
    lottery_scale=config.cutoff_lottery_scale,
    gumbel_scale=config.cutoff_gumbel_scale,
    preference_seed=config.cutoff_preference_seed,
    remove_city_wide=config.remove_city_wide,
)
solution = config.make_solver().solve(problem)
```

### 3.2 Measured result

Artifact: `optimization_output/solution_Block_2.json`.

| Statistic | Value |
|---|---:|
| Status | `FEASIBLE` |
| Runtime | 901.59 seconds |
| Reported grid objective | 417.90 |
| Raw objective | 8,358 |
| Raw best bound | 87 |
| Initial variables | 686,718 |
| Student-school preference entries | 220,574 |

Re-evaluating this zoning with the analytical oracle gave objective `257.50`.
The difference is not a disagreement between models: CP-SAT had found a
feasible but nonminimal cutoff vector when its time limit expired.

### 3.3 Strengths

- Directly represents the intended finite-grid market.
- Produces a valid CP-SAT objective bound.
- Handles zoning and market clearing in one model.
- Useful as an independent small-instance cross-check of the analytical
  oracle.

### 3.4 Weaknesses

- Market variables dominate the 2,562 zoning-assignment variables.
- Min/max equalities have weak propagation before zoning access is known.
- The lower bound remained economically uninformative after 15 minutes.
- A time-limited solution can contain cutoffs far above the least vector for
  its zoning.
- Scaling to finer graphs, more zones, or a larger lottery grid is poor.

## 4. Analytical fixed-zoning cutoff oracle

### 4.1 Method

`optimization/cutoff_oracle.py` adapts the piecewise-linear inversion in the
external `DA/analytical.py` implementation.

For fixed cutoffs at other schools, each applicant contributes a term

```text
max(0, u_i - max(c_s - L p_is, 0))
```

to school `s`, where `u_i` is the lottery mass rejected by every more-preferred
accessible school. The demand slope changes only at `L p_is` and
`L p_is + u_i`. Scanning these breakpoints gives the smallest integer cutoff
that satisfies capacity. Cyclic school updates start from zero.

The school best-response map is isotone: raising another school's cutoff can
only send weakly more demand to later schools. Starting from zero therefore
converges to the coordinatewise least feasible grid cutoff vector. The same
algorithm without integer rounding gives the continuous market-clearing
cutoff vector.

### 4.2 Reproduction

For any saved zoning:

```python
result = solve_zoned_cutoffs(market, node_assignment, num_zones=6)
continuous = solve_zoned_continuum_cutoffs(
    market, node_assignment, num_zones=6
)
```

Tests in `optimization/tests/test_cutoffs.py` compare the oracle with brute
force and with the direct CP formulation on tiny markets.

### 4.3 Performance

Across the real six-zone experiments, a complete oracle evaluation generally
took `0.12-0.25` seconds. A fixed-zoning direct CP-SAT verification of the
`122.50` zoning took 31.57 seconds and returned exactly the oracle cutoffs.

### 4.4 Strengths

- Exact and fast for a fixed zoning under the grid model.
- Independent of CP-SAT's min/max encoding.
- Supplies exact recourse values for heuristics and decomposition.
- Reconstructs assignment measures and school demand.
- The continuous version numerically verifies stable market clearing without
  applicant identity tie scores.

### 4.5 Weaknesses

- Does not optimize geography.
- The current zoned wrapper requires all schools to be zone-restricted.
- Integer-grid minimality allows rounding slack at positive cutoffs.
- Continuous expected-demand equilibrium is not the expectation of finite
  realized DA cutoffs.

## 5. Zoning-only pressure surrogates

### 5.1 Motivation

Core geographic CP-SAT models solve in well under a second for many linear
objectives. The experiments therefore generated zonings using cheap load and
capacity surrogates, then scored each zoning with the exact oracle.

For zone `z`, let `N_z` be modeled applicants, `Q_z` market capacity, and
`P_z = N_z - Q_z` market pressure. Concentrating positive pressure empirically
reduced the number of simultaneously congested markets.

### 5.2 Families tested

Artifacts:

- `benchmark_output/cutoff_oracle_surrogates.json`
- `benchmark_output/cutoff_oracle_surrogates_runs.csv`
- `benchmark_output/cutoff_oracle_surrogates_phase1.json` through `phase5.json`

| Family | Runs | Master objective | Best oracle objective |
|---|---:|---|---:|
| Market students | 6 | maximize `N_z` | 131.85 |
| Minimum boundary | 1 | minimize cut edges | 333.45 |
| Market pressure | 6 | maximize `P_z` | 122.60 |
| Student/capacity lexicographic | 6 | maximize `3427 N_z - Q_z` | 132.00 |
| Weighted pressure | 42 | maximize `N_z - w Q_z` | 124.60 |
| Suppress other positive pressure | 6 | pressure plus secondary penalty | 122.60 |
| Two-zone pressure lexicographic | 30 | maximize `B P_a + P_b` | 122.55 |
| Pair pressure sum | 15 | maximize `P_a + P_b` | 164.85 |
| Three-zone pressure lexicographic | 20 | maximize `B^2 P_a+B P_b+P_c` | 122.50 |

Here `B=7,230` is larger than the possible pressure range and therefore makes
the weighted objectives lexicographic.

In total, 132 zonings were evaluated. 128 surrogate solves were optimal and 4
pair-pressure solves were time-limited feasible solutions. Summed CP time was
268.09 seconds; summed oracle time was 19.76 seconds.

The best run was `pressure_lex_three_z5_z0_z2`, with raw objective 2,450.

### 5.3 Reconstruction procedure

The historical generation script was temporary and is not retained. The
experiments can be reconstructed using `CpBoolSolver` internals:

1. Build assignment variables with `_build_assignment_vars`.
2. add `_add_core_constraints`;
3. compute applicant and market-capacity counts by graph node;
4. add one of the linear objectives above;
5. solve with the config's seed and worker count;
6. extract `x_zv` and call `solve_zoned_cutoffs`.

The current decomposition solver implements deterministic one-zone and
three-zone pressure variants in `_initial_incumbent`.

### 5.4 Strengths

- Very small CP models and rapid feasible solutions.
- Generated the structural starting point for the best incumbent.
- Easy to combine with any master-only zoning constraint.
- Useful warm starts even when exact separation remains difficult.

### 5.5 Weaknesses

- Surrogate values are neither cutoff objectives nor valid cutoff bounds.
- Multiple pressure-optimal zonings can have very different cutoff values.
- Parallel CP tie choices caused substantial run-to-run variation; the current
  code uses deterministic one-worker pressure solves.
- Minimum boundary was actively poor for this objective.
- The concentration pattern is data-dependent and has no general optimality
  theorem.

## 6. Exact-oracle local search

### 6.1 Neighborhoods

Starting from the best pressure zoning, the search considered:

- A boundary node plus every same-zone node that depends uniquely on it for a
  closer-neighbor support path.
- Support-closed multi-node transfers.
- Simultaneous transfers in opposite directions.
- Paired moves involving three zones.
- Swaps of school nodes together with their required support closures.

Every candidate preserved centroid anchors, candidate zones, closer-neighbor
support, and the 9/10 graph-school count. Only affected zone markets were
recomputed and zone results were cached.

### 6.2 Results

Artifact: `benchmark_output/cutoff_oracle_local_search_best.json`.

| Statistic | Value |
|---|---:|
| Starting raw objective | 2,450 |
| Final raw objective | 2,269 |
| Normalized objective | 113.45 |
| Search runtime | 285.01 seconds |
| Total runtime | 287.97 seconds |
| Exact zoning evaluations | 4,032 |
| Exact zone-market solves | 5,462 |
| Generated move candidates | 28,968 |
| Feasible move candidates | 13,379 |

The accepted historical path was one school swap followed by three applicant
node transfers: `2450 -> 2273 -> 2272 -> 2271 -> 2269`.

A separate six-start run is saved in
`benchmark_output/cutoff_search_multistart_best.json`. It ran for 258.58
seconds and reached 2,273 from the target-zone-5 start. Other target-zone starts
finished between 2,492 and 3,140.

### 6.3 Reproducibility

The original broad local-search script was temporary. A narrower deterministic
implementation is retained in
`CutoffDecompositionSolver._local_improve`. It performs support-closed school
swaps and boundary transfers, with each candidate scored by the exact oracle.
It is disabled when demographic, capacity, boundary, or nonzero centroid-radius
constraints are active because its lightweight feasibility checker does not
reimplement those constraints.

### 6.4 Strengths

- Produced the best known incumbent.
- Uses the true cutoff objective rather than a proxy.
- Every accepted zoning has independently certified recourse.
- Caching makes thousands of exact evaluations practical.

### 6.5 Weaknesses

- No global lower bound.
- No proof that the move graph reaches every feasible zoning.
- Local minima and tie-breaking affect the result.
- Broad neighborhoods become expensive on finer graphs.
- Historical standalone runs are artifact-reproducible but not directly
  script-reproducible.

## 7. Zero-cutoff constraint generation

### 7.1 Question and cut

The concentration results suggested testing whether all cutoff pressure could
be placed in one zone. For a school `s` required to have cutoff zero, an
applicant demands `s` exactly when the applicant and school share zone `z` and
no school preferred to `s` is in `z`.

Applicants with the same residence node and same set of higher-ranked school
nodes were grouped. A demand Boolean `d_gsz` was lower-bounded by

```text
d_gsz + sum_(higher node h) x_zh >= x_z,residence(g) + x_z,node(s) - 1.
```

The exact zero-cutoff capacity cut is

```text
sum_g group_size(g) d_gsz <= q_s.
```

### 7.2 Certificate

Artifacts:

- `benchmark_output/cutoff_one_positive_infeasibility_certificate.json`
- `benchmark_output/cutoff_one_positive_certificate_replay.json`

Each of the six zones was separately allowed to be the only positive-cutoff
zone. All six models were infeasible. The combined search added 202 exact cuts,
evaluated 14 oracle candidates, and took 42.20 seconds. Independent replay of
all six certificates took 32.94 seconds.

This proves, under the benchmark's core zoning constraints, that no zoning can
have five zero-cutoff markets and only one positive-cutoff market.

### 7.3 Strengths

- Rigorous structural infeasibility certificate.
- Much smaller than representing all cutoff recurrences.
- Demonstrates how applicant groups can generate strong lazy constraints.

### 7.4 Weaknesses

- Only distinguishes zero from positive cutoff.
- Does not lower-bound the magnitude of positive cutoffs.
- The generation script was temporary and is not retained.
- The certificate does not include FRL, boundary, or zone-capacity constraints;
  those were disabled.

## 8. Aggregate, Hall, and first-choice relaxations considered

These methods were analyzed but not retained as production implementations.
They should not be confused with benchmarked methods above.

### 8.1 Aggregate capacity/priority bound

If `u_i` is applicant `i`'s unassigned lottery mass in zone `z`, total capacity
implies

```text
sum_(i in z) u_i >= L max(0, N_z - Q_z).
```

For every school in that zone, `u_i <= t_is(c_s)`. Aggregating applicants by
priority tier gives an order-statistic lower bound on each school cutoff. A
prototype-sized formulation was estimated at roughly 14,748 variables, versus
686,718 in the monolith.

Weakness: aggregate capacity ignores preference congestion at individual
schools. Experimental reasoning found large gaps between this relaxation and
the exact oracle, so it was not adopted as the primary solver.

### 8.2 First-choice lower bounds

Applicants for whom `s` is the top accessible school generate unavoidable
demand at `s`. This can yield strong fixed-zoning cutoff bounds. Globally, the
top accessible school depends on the complete school partition, so an explicit
model reintroduces most of the 220,574 preference interactions. Dynamic
first-choice constraints are better used as separated cuts.

### 8.3 Hall subset inequalities

At a fixed cutoff vector, each applicant lottery cell has a set of qualifying
schools. For every school subset `T`, the cells whose nonempty qualifying set
is contained in `T` cannot exceed total capacity in `T`. These are valid Hall
relaxations and can be separated by a maximum-closure/min-cut problem.

Weakness: Hall feasibility permits arbitrary reassignment among qualifying
schools and does not enforce that applicants choose their most-preferred
qualifying school. Hall cuts are therefore valid lower-bound cuts but not a
complete replacement for revealed-preference cuts.

## 9. Exact revealed-preference decomposition

### 9.1 Master and subproblem

The production implementation is
`optimization/solvers/cutoff_decomposition.py`.

The master contains only:

- The geographic zoning variables and all core zoning constraints.
- One integer cutoff per school.
- Revealed-preference interval cuts accumulated from prior candidates.

For a candidate zoning and cutoff vector, the subproblem reconstructs exact
assignment intervals. If applicant `i` sends lottery interval `[a,b]` to
school `s` in zone `z`, then this interval must continue to demand `s` whenever:

```text
i and s are assigned to z,
c_s <= L p_is + a - 1,
every school r preferred to s is outside z or c_r >= L p_ir + b.
```

A Boolean profile variable is forced under these sufficient conditions, and
the weighted profile variables must fit school capacity. The resulting cut is
globally valid: it may undercount demand after zoning/cutoff changes, but it
cannot overcount guaranteed demand.

Every infeasible finite candidate has an overloaded school and receives a
violated cut. Because the zoning and cutoff domains are finite, exhaustive
constraint generation terminates in principle.

### 9.2 Incumbent generation

Before exact separation, the current implementation:

1. Solves one deterministic pressure model for each possible dominant zone.
2. Refines the best target with deterministic three-zone lexicographic
   pressure objectives.
3. Applies exact-oracle school swaps and support-closed boundary transfers.

This reproduces the raw 2,269 incumbent reliably.

### 9.3 Optimality conditions

The solver reports `OPTIMAL` only if:

1. The valid-cut master under the strict incumbent objective is infeasible.
2. A master-optimal candidate has no exact school overload.
3. The integral master lower bound reaches the oracle-feasible incumbent.

Otherwise it reports `FEASIBLE` and preserves the lower bound.

### 9.4 Final benchmark

Command:

```bash
uv run python -m optimization.run \
  optimization/config.example.cutoffs.yaml \
  --output benchmark_output/cutoff_exact_decomposition_final
```

Artifact: `benchmark_output/cutoff_exact_decomposition_final/`.

| Statistic | Value |
|---|---:|
| Artifact production span | 544 seconds |
| Status | `FEASIBLE` |
| Grid objective | 113.45 |
| Raw objective | 2,269 |
| Continuous stable objective | 113.13455460193593 |
| Raw valid lower bound | 333 |
| Normalized lower bound | 16.65 |
| Revealed-preference cuts | 824 |
| Grouped cut profiles | 62,909 |
| Continuous stable markets | 6 of 6 |
| Graph school counts | `[10, 9, 10, 10, 10, 10]` |

An earlier nondeterministic-start run in
`benchmark_output/cutoff_exact_decomposition/` returned grid objective 129.65,
illustrating why deterministic incumbent construction matters.

### 9.5 Strengths

- Exact finite algorithm in principle.
- Keeps CP-SAT focused on zoning, where it is strongest.
- Uses the analytical oracle for exact feasible incumbents.
- Every cut is globally valid and every saved lower bound is rigorous for the
  encoded grid model.
- Honest anytime behavior: time limits never become false optimality claims.
- Supports master-only constraints without changing the recourse proof.

### 9.6 Weaknesses

- CP-SAT restarts after every separation round and loses its search tree.
- The final lower bound, 16.65, is still 85.3 percent below the incumbent.
- A large number of school-access patterns remain indistinguishable until
  separated.
- Universally accessible schools currently violate the isolated-market
  assumption and are rejected.
- Finer grids increase the cutoff domain and can slow both oracle and master.
- Exactness in principle did not yield a finite-instance certificate within
  ten minutes.

## 10. Continuous stability verification

The final zoning was independently re-solved with continuous cutoffs. All 17
positive-cutoff schools cleared capacity within tolerance; the maximum residual
was approximately `6.4e-9`. Reordering and renumbering applicants did not change
the result, confirming that no applicant identity tie score enters the oracle.

This establishes stable continuous score limits for the reported zoning. It
does not establish that the zoning maximizes welfare or globally minimizes the
continuous cutoff objective.

## 11. Comparative conclusions

| Method | Feasible incumbent | Global bound | Exact fixed-zoning recourse | Best objective |
|---|---|---|---|---:|
| Monolithic CP-SAT | Yes | Yes, very weak | Embedded | 417.90 reported; 257.50 after oracle |
| Analytical oracle | Requires zoning | No | Yes | Evaluation only |
| Pressure surrogates | Yes | No | Oracle post-check | 122.50 |
| Exact-oracle local search | Yes | No | Yes | **113.45** |
| Zero-cutoff cuts | Structural certificate | Partial | At cutoff zero | Infeasibility result |
| Revealed-preference decomposition | Yes | Yes | Yes | **113.45**, bound 16.65 |

The substantive gains came from separating geography from market clearing, not
from tuning CP-SAT parameters. Pressure objectives supplied useful structure,
the analytical oracle made exact local evaluation cheap, and revealed-
preference cuts restored a globally valid lower bound. The unresolved issue is
bound strength, not incumbent feasibility or stable-market validation.

## 12. Data and reproducibility caveats

The run depends on local files referenced by the config, including:

- `assignment/configs/kumar.config.yaml`
- the configured ETB/CTIP NumPy file
- cached graph and assignment data under the project data root

`benchmark_output/`, `optimization_output/`, and `papers/` are ignored by Git
in this workspace. Preserve the listed artifacts separately if they are needed
for archival reproduction. Historical surrogate and broad local-search scripts
were temporary; their complete inputs, outputs, formulas, and reconstruction
steps are recorded here, but the scripts themselves are not available.
