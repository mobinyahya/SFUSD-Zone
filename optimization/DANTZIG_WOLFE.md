# Anchored whole-zone branch-and-price

## Scope

Let `G = (V, E)` be the level's adjacency graph and `z = 0, ..., Z-1` the zone
labels, each anchored at a centroid `c_z` carrying school id `sigma_z`. Citywide
schools are excluded from the geographic school/capacity data and citywide
programs are removed from every preference list before market evaluation.
Students remain in their geographic vertices, including those left with no
available preferences.

## The zone families

For each label, `F_z` is the family of admissible individual zones, and it is
the base zoning model's label-`z` rows -- nothing added, nothing relaxed. A set
`A subset V` lies in `F_z` iff

1. `c_z in A`, and `B_r(c_z) subset A` where `r = centroid_neighbor_radius`;
2. `A` is disjoint from `B_r(c_{z'})` for every other label `z'` (this also
   excludes every other centroid);
3. `z in candidate_zones(v)` for every `v in A`, which carries `max_distance`,
   explicit `candidates` and `fixed`;
4. every `v in A \ {c_z}` has a *closer-neighbour support* in `A`: a graph
   neighbour `u in A` whose geometry is strictly closer to `sigma_z` than `v`'s
   is, drawn from `contiguity.contiguity_supports`;
5. the FRL, racial and aggregate-capacity balance rows and the school-count
   band hold, **in the integer-rounded form `cp_bool` writes them** -- each
   coefficient `round(CP_SAT_SCALE * a_v)` against an integer right-hand side.

Condition 4 implies `G[A]` is connected, and is strictly stronger: iterating
the support gives every member a distance-decreasing path to `c_z`, so a zone
that wraps around and re-approaches its anchor from the far side is connected
and inadmissible. Condition 4 is also what removes the rooted single-commodity
flow the previous build priced with: connectedness is `|A|` clauses, not `2|E|`
arc variables with big-`M` capacities.

Condition 5 is why there is no feasibility tolerance anywhere in this code. The
previous build tested float rows with a 1e-6 slack in the pool and re-stated
them with a different rounding in the pricer, so the two disagreed about the
boundary of `F_z` in both directions. `ZoneFamily.feasible` and the pricing
model now evaluate the same integers.

Candidate sets are pruned to a fixed point: a non-anchor vertex with an empty
support list can never be a member, which can empty another vertex's support
list. `cp_bool` reaches the same fixings by propagation, so the pruning changes
no feasible set; it makes the family's own description closed under the
implication.

**Exact decomposition.** Every row of the base model involves a single label,
except the assignment row (one label per vertex) and the boundary cap. So a
tuple `(A_0, ..., A_{Z-1})` is an admissible zoning iff each `A_z in F_z` and
the sets partition `V` -- the first becomes the master's convexity rows, the
second its cover rows, and the cap its boundary row. `test_dantzig_wolfe.py`
checks this against `cp_bool`'s own solution enumeration on a small graph.

## What a zone is worth

Three definitions, selected by `dw_objective`. All three are additive over a
partition, which is the whole reason the decomposition exists: with no citywide
program, a partition splits the applicants and the seats into independent
submarkets.

**`mid`.** `W_L(A)` is finite-grid least-cutoff MID welfare at integer lottery
scale `L = mid_lottery_scale`, computed by `finite_grid_oracle` over the
residents of `A` and the programs whose schools lie in `A`. Proposition 7 is
what licenses computing it by maximizing over capacity-clearing cutoffs: the
finite-grid oracle raises an overloaded program's cutoff to its smallest
clearing value, every iterate stays componentwise below every clearing vector,
integrality gives finite termination, and with positive rank-ordered utilities
the least clearing vector maximizes welfare.

**`stable_matching`.** Fix one tie-breaking draw `psi` -- `saa_tie_breaking_method`
(`MTB` or `STB`) and `seed`, one sample, drawn *after* citywide programs are
removed so the positional program order still lines up. `W^psi(A)` is the
largest welfare of a stable matching of the zone submarket. Because
maximizing utility over a market's stable-admissions polytope returns the
applicant-proposing deferred-acceptance outcome -- the polytope is the convex
hull of the stable matchings, DA is applicant-optimal, and the utilities are
rank-consistent -- the column value is **one run of the mechanism**. No LP, no
MIP, a few hundred microseconds, and *exactly* realized welfare: the +195 to
+252 over-report that `SaaOracle`'s aggregated Rothblum row carries on
`Block_2` does not arise here, because a single zone's stable matchings are
enumerated by running the mechanism rather than described by inequalities.

**`boundary`.** `-b(A)`, where `b(A)` is half the weighted cut of `A`, so a
partition's scores sum to the district boundary cost. Not a welfare; it exists
because it makes the decomposition testable against an enumerated optimum
without a market.

## The master

With `b(A)` as above and an optional district budget `B`:

```text
maximize  sum(z,A) W(A) lambda[z,A]
subject to
  sum(z,A: v in A) lambda[z,A] = 1        for every v in V
  sum(A in F_z)    lambda[z,A] = 1        for every label z
  sum(z,A) b(A) lambda[z,A] <= B          optional boundary cap
  lambda[z,A] in {0,1}
```

Each cut edge appears in two zone perimeters, so the boundary row is the
district boundary cost, and `B = floor(boundary_prop * sum_E w)` is the same
number `cp_bool`'s cut-edge cap uses at unit weights.

Relax `lambda >= 0`. Let `alpha_v` and `gamma_z` be the free duals of the cover
and convexity rows and `mu >= 0` the boundary dual. Pricing label `z` maximizes
the reduced welfare

```text
max over A in F_z:  W(A) - sum(v in A) alpha_v - gamma_z - mu b(A).
```

### The duals are the interesting part

The master is a set-partitioning LP and is therefore massively degenerate. The
previous build solved it with dual simplex, pinned for reproducibility, and
measured the consequence directly: on `BlockGroup_0` with a 404-column pool,
three to six columns of genuinely positive reduced cost entered every round and
the LP value sat at 11,142.80 for eight consecutive rounds. A positive reduced
cost that does not move the objective is a *degenerate pivot* -- the entering
column's step length is zero. Only 132 of 579 cover rows carried a nonzero
dual, so pricing had almost no guidance, and six unanchored labels priced
independently returned six zones of 340 to 392 vertices out of 579, no six of
which can tile anything.

Three things address that, and none of them costs rigour:

- **Interior duals.** `dw_master_method: barrier` with `Crossover = 0` returns
  the barrier iterate rather than walking it to a vertex, spreading the prices
  over the rows. `dual` restores the old vertex duals; the cost of the default
  is bit-for-bit reproducibility, not validity.
- **Wentges smoothing.** `dw_dual_smoothing = a` prices against
  `a * (this LP's duals) + (1-a) * (previous centre)`. When the smoothed point
  *mis-prices* -- finds no zone improving at the LP's own duals -- the round
  re-prices at the raw duals before concluding anything and moves the centre
  there. Without that fallback a smoothed point reports convergence the LP does
  not have, and its dual objective gives a needlessly weak bound.
- **Anchored zones.** A label can no longer return two thirds of the district,
  so a priced zone is at least the right size to be part of a tiling.
- **A bounded round.** `dw_pricing_time_limit` caps what one round spends
  pricing and doubles only when a round neither adds a column nor proves a
  bound. Uncapped, the first round on a real instance ate the whole budget:
  240s bought two master LPs.

### The pool, not the prices: why this needed a second column source

None of that was sufficient, and the reason is not the dual degeneracy those
remedies were chosen for. Measured on `Block_2` (501 units, six anchors, 4,154
applicants) seeded with one `cp_bool` feasibility hint, both welfare objectives
ran 21 rounds and admitted 542 to 555 columns of strictly positive reduced cost
inside 600s while the restricted LP stayed at the hint value to four decimal
places on every round, leaving a pool that still admits **exactly one tiling**.
The same on a district-sized synthetic instance (576 vertices, 4,032
applicants, `|Gamma| = 12,096`): all four combinations of `{dual, barrier}` x
`{alpha = 1, alpha = 0.5}` admitted 400 to 424 such columns in ten rounds and
left the LP at the seed value 13,161.9302, and deleting the `Z` seed columns
from the resulting 413-column pool makes the integer master `INFEASIBLE`.

The obstruction is a dimension count, not a choice of basis. The restricted LP
carries `|V| + Z` equality rows -- 582 here -- against however many columns
have been generated, so while the pool is small relative to `|V|` the system is
over-determined and its solution set is essentially the tilings it happens to
contain. Pricing each label independently maximizes reduced welfare with no
reference to the other labels, so `Z` priced zones are under no pressure to be
mutually complementary, and the pool can grow by hundreds of individually
improving zones without acquiring a second tiling. Every entering column is
then a degenerate pivot: strictly positive reduced cost, zero step length.

This is what `zone_redraw.py` addresses, and it is the reason the module
exists. Pricing cannot fix it -- the per-label question is the *right* question
for the bound and the wrong one for the pool.

Two other repairs were identified. The first is now implemented, in the section
below. The second is not: **completability-targeted pricing**, a cardinality or
student-mass window per label read off the incumbent, on the column-generating
pass only.

## The budgeted-elastic master

`dw_overlap_prop` addresses the primal deficiency directly. Each Phase-II cover
row becomes

```text
sum(z,A: v in A) lambda[z,A]  +  d_v  -  e_v  =  1,      d_v, e_v >= 0
sum(v) w_v (d_v + e_v)  <=  K                            one extra row
```

where `w_v = max(1, students(v))`. `d_v` is deficit -- `v` left uncovered --
and `e_v` is surplus -- `v` claimed by two zones at once. Both are required,
and the surplus is the load-bearing half. Raising a new label-`z` column forces
the incumbent's label-`z` column down by the same amount, which under-covers
the nodes only the incumbent held and over-covers the ones only the newcomer
holds; Phase I's deficit-only artificials keep structural coverage `<= 1`, so
they cannot absorb the collision and leave the step length at zero even during
Phase I. The convexity rows stay hard -- one zone per label, or labels
evaporate.

With both in place the LP is full-dimensional in `lambda`: a priced zone that
collides with the incumbent's other zones can enter with a positive step
length, paying budget for the overlap and banking `W(A)`, and it does so
exactly when the welfare gain is worth the ration. Three things follow that the
exact master cannot give: the LP value moves, so "did this column help?"
becomes answerable; the cover duals are set by where overlap is contested
rather than left at zero by degeneracy, which is what shrinks the ~800
per-label pricing residual; and the overlap pattern at the optimum names the
contested vertices, which is completability information that per-label pricing
structurally cannot see.

### Why a budget and not a penalty

The classical elastic master penalizes mismatch at a large `M`. `M` is not a
free parameter: at the optimum `d_v` has reduced cost `-M - alpha_v` and `e_v`
has `-M + alpha_v`, so `d = e = 0` is optimal exactly when
`M >= max_v |alpha_v|`. Above that exact-penalty threshold the elastic
variables price themselves out and the degenerate single point is back, with
nothing gained; below it the LP drifts toward overlapping high-welfare zones
that will never tile. `M` therefore has to be discovered per instance, per
objective and per graph level, in welfare-per-node units -- on `Block_2` a
scale of roughly 24, from 12,050 welfare over 501 units.

Budgeting inverts that. The budget row's dual `mu_K >= 0` *is* `M`, chosen by
the LP and re-chosen every round, and the elastic pair's own dual-feasibility
rows make the relationship explicit: `|alpha_v| <= w_v mu_K`. What is left to
choose is `K`, in student-equivalents and expressed as a proportion of the
district total, which is the same kind of knob as `boundary_prop`.

Two details are load-bearing. The weights are student mass rather than uniform
because block mass spans an order of magnitude, and a uniform weight charges
the same for double-claiming a 2-student block and a 40-student one, so the LP
would spend the whole budget where the welfare payoff is and the duals would be
least informative exactly there. The floor at one unit is there because a
student-free node would otherwise be double-claimable for nothing, which is the
one way a budget denominated in students can be spent on geometry: two labels
each running their own support chain through the same free node.

`d` and `e` carry no objective coefficient and no upper bound. The price of a
unit of mismatch is the row's dual, and a variable bound whose reduced cost the
dual objective did not account for would report a bound *below* the relaxation
it was computed from -- the one error that is wrong rather than merely weak.
The budget row bounds them anyway: `w_v >= 1`, so `e_v <= K`.

### What it costs in rigour: nothing

Every tiling has `d = e = 0` and the same objective, so the elastic LP contains
the exact master's feasible set and agrees with it there. Its optimum is
therefore `>=` the exact master LP's at every `K`, and `K = 0` is the exact
master. Proposition 9 survives unchanged, with one arithmetic obligation:
`mu_K * K` must be added to the dual objective, since dropping it understates
the dual point's value and so reports a bound below the relaxation's own
optimum. `DualPoint` carries `mu_K` for that reason, which also makes Wentges
smoothing safe -- `|alpha_v| <= w_v mu_K` cuts out a convex set, so a blend of
two dual-feasible points is dual-feasible provided `mu_K` is blended alongside
`alpha`, and the rows do not mention `K`, so a point smoothed across a change
in `K` stays feasible.

A loose `K` therefore costs bound *quality* and never correctness. Note also
that the integer master is never elastic: it is the only thing here that
produces an incumbent, and an incumbent has to be a tiling. Phase I is never
elastic either -- its completion test is zero deficit, so a surplus variable
would let it pass by double-claiming a node instead of by covering the graph.

### The one dead end, and its exit

Elasticity creates a state the exact master cannot reach: integral marginals
with mismatch still spent. There is nothing fractional to branch on and the LP
is not a partition. The search halves `K` and re-solves; `K = 0` is the exact
master, so this terminates in finitely many halvings, and because `K` never
rises again the bound the node finally certifies is the tighter one.

### Tuning K

`K` is a one-dimensional sweep on weakly monotone quantities, and the
instrumentation is per round in `dw_history`:

| symptom | reading | move |
|---|---|---|
| `elastic_mass = 0`, or the LP still at the exact-master value | the overlap bought nothing: either `K` is too small to matter, or the pool already holds the LP optimum | raise `K`; if the LP is already optimal, elasticity is not this instance's problem |
| `overlap_dual = 0` with `elastic_mass > 0` and the LP saturated | `K` exceeds what overlap can buy, so the ration is a free spend the LP is indifferent to. The search recognizes this and drops `K` to zero rather than bisecting | lower `K` |
| `overlap_dual > 0` and `elastic_mass` at `K` | the budget binds and is pricing the mismatch: the working range | -- |

`duals_pinned` is the weakest of the four and does *not* read as "`K` too
small". A binding budget bounds `alpha` by `w_v mu_K` and many nodes sit at
that bound, so the count is high across the whole working range -- 10 of 12 in
the ladder below. It says the budget row is setting the prices, which is what a
binding budget does; the discriminating signals are `overlap_dual` and whether
the LP moved.

A ladder on a 12-node, 2-label instance with the `mid` objective, enumerated
down to one tiling plus 187 individually-good zones that cannot complete it
(integer optimum over the *full* enumeration: 6.00):

| `K` | LP | `elastic_mass` | `elastic_nodes` | `mu_K` | `duals_pinned` |
|---|---|---|---|---|---|
| 0.00 | 2.4000 | 0.000 | 0 | 0.0000 | 0 |
| 0.24 | 2.5920 | 0.240 | 1 | 0.8000 | 10 |
| 0.60 | 2.8800 | 0.600 | 1 | 0.8000 | 10 |
| 1.20 | 3.3600 | 1.200 | 2 | 0.8000 | 10 |
| 3.00 | 4.0000 | 3.000 | 3 | 0.0000 | 0 |
| 6.00 | 4.0000 | 4.000 | 4 | 0.0000 | 0 |
| 12.00 | 4.0000 | 4.000 | 4 | 0.0000 | 0 |

The exact master is stuck at its one tiling's 2.40. The budget binds through
`K <= 1.2`, spending every unit it is given at a stable price of 0.80 per unit
and climbing the LP monotonically; by `K = 3` the LP has saturated at 4.00 and
`mu_K` has gone to zero, which is the upper edge. The window is `K` in
`(0, 3)`, and the shape -- a binding budget at a stable price, then saturation
-- is what to look for on a real instance.

On a pool that already contains the LP optimum every positive `K` reads as the
saturated row: mass spent, `mu_K = 0`, LP unmoved. That is a correct answer --
there is nothing for the elasticity to buy -- and not a badly chosen `K`.

Calibrate on a *fixed* pool before spending any pricing budget: the saved
400-555-column pools re-solve at a geometric ladder of `K` in seconds, with no
pricing and no CP-SAT, which brackets the window for free. The acceptance
metric is not the LP value but the certified bound, `elastic_LP(K) + sum_z
max(0, b_z)` against the a-priori constant -- the two terms move in opposite
directions as `K` falls, so there is an interior optimum that has to be
measured rather than reasoned. One master plus one pricing round per `K` is
under a minute now that a label prices in 0.1-8s. The primal pass/fail stays
the one-tiling diagnostic: delete the `Z` seed columns and re-solve the integer
master.

### Measured on Block_2, 2026-09-14

`analysis/probe_overlap_budget.py`, mid objective, 501 nodes, 4,154 students,
constant 16,830.9547. The exact master reproduced the pathology precisely: LP
pinned at **10,229.7554 for all nine rounds**, 34 columns, `one_tiling = True`.

| `prop` | `K` | LP | mass | `mu_K` | pinned | residual | certified |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 10,229.76 | 0 | 0 | 0 | 133,566 | 143,795 |
| 0.002 | 8.42 | 10,239.13 | 8.42 | 1.11356 | 499 | 49,181 | 59,420 |
| 0.01 | 42.10 | 10,276.64 | 42.10 | 1.11356 | 499 | 49,195 | 59,472 |
| 0.05 | 210.50 | 10,464.16 | 210.50 | 1.11356 | 499 | 46,611 | 57,075 |
| 0.25 | 1052.50 | 11,401.78 | 1052.50 | 1.11356 | 499 | 43,580 | **54,982** |
| 0.5 | 2105.00 | 12,181.12 | 2105.00 | 0.60091 | 499 | 46,106 | 58,287 |

The budget is fully spent at every `K` and the LP gain is *exactly* `mu_K * K`
over two orders of magnitude -- there is no interior window of the kind the toy
instance showed, and `duals_pinned` is 499 of 501 throughout, so the cover
duals become `±w_v mu_K`: student mass with a sign. That is still a large
improvement on `K = 0`, where only **6 of 501** duals are nonzero, and the
residual does fall 63% at the smallest budget. **But no `K` beats the
constant.**

Two findings matter more than the table.

**Raising `K` makes the residual target harder.** `certified = LP + sum_z b_z`
and the elastic LP is a relaxation, so `K` inflates the first term while
shrinking the second:

| | LP | needed `sum b_z` | per label |
|---|---|---|---|
| `K = 0` | 10,230 | < 4,967 | 830 |
| `prop = 0.25` | 11,402 | < 3,795 | 632 |

So the budget buys dual quality and pays for it in bound inflation. What has no
such cost is simply *more columns at `K = 0`* -- a richer pool
de-degenerates the duals while the LP stays a valid tight bound, which points
at the redraw rather than at this knob.

**The residual at the default pricing budget is mostly solver slack.** See the
REVISIT note in `branch_price.py`: 20,783-24,161 per label at ~20s against
~5,300 at 600s, with a 0.45-0.58% gap to the label's own best zone. Any `K`
sweep run at the default budget is therefore measuring slack more than
optimism, which is the main caveat on the table above.

What is verified independently of all this is the mechanism:
`test_dantzig_wolfe.py` reproduces the zero step length on a colliding column
and removes it with a budget, checks the relaxation ordering and the
strong-duality identity including `mu_K * K`, and checks
`|alpha_v| <= w_v mu_K` at both dual points.

## The two-zone redraw

`zone_redraw.py` is the second column source. Fix an admissible partition
`(A_1, ..., A_Z)`, choose two labels `a, b`, and re-partition their joint
territory `U = A_a u A_b` optimally, leaving the other `Z - 2` zones untouched.
This is ReCom's move with the guesswork removed: the same neighbourhood, but
the split is chosen by CP-SAT under the run's real welfare objective instead of
by a spanning-tree cut, and admissibility is a constraint rather than a
rejection test.

Four properties make it the right move here.

1. **Every feasible solution is a tiling.** `A'_a` and `A'_b` partition `U` by
   construction and the untouched zones partition `V \ U`, so *any* solution of
   the pair model -- not just the optimum, not just an improving one -- is a
   complete admissible partition of `V`, and its two columns provably complete
   the `Z - 2` already in the pool. After `k` successful redraws the pool holds
   at least `k + 1` tilings. That is exactly the dimension the master was
   missing.
2. **No duals are needed.** The two new zones cover `U` exactly once, so
   `sum_{v in U} pi_v` is the same number for every feasible solution and
   cannot influence which one is best. The pair's summed reduced cost and its
   summed welfare differ by a constant: maximizing welfare maximizes both. The
   redraw therefore runs with no master, no LP and no dual point -- it is
   simultaneously an exact primal local search on the map and the
   reduced-cost-maximizing pair split. The boundary cap is handled *exactly*
   rather than priced: the other zones' perimeters are fixed and so is the part
   of the pair's perimeter facing them, so the district cap becomes an integer
   row on the pair's own cut indicators.
3. **A pair's territory is invariant under its own redraw.** `A'_a u A'_b = U`,
   so once a pair is solved to optimality it stays optimal until some *other*
   pair moves a block into or out of one of its two zones. Memoizing on
   `(pair, U, branch fixings)` gives exact "don't look" bookkeeping for free --
   an entry expires precisely when it should, with no invalidation logic. A
   sweep that reaches a state where every adjacent pair is solved has produced
   a tiling no two-zone move can improve. Measured: after the incumbent
   stabilizes, a sweep costs 0.0s.
4. **Non-adjacent pairs are provably no-ops**, so they are never enumerated. If
   no edge joins `A_a` and `A_b` then every `U`-neighbour of a block of `A_a`
   lies in `A_a`; a block joining `A'_b` needs a strictly closer support in
   `A'_b`, and iterating that requirement gives a support chain that must
   terminate at `c_b`, which is not in `A_a`. No block can cross in either
   direction.

### It is additive, and that is not optional

The redraw is **not** a pricing problem and produces no `R_z`. Its search space
is `{A in F_a : A subset U}`, a *subset* of `F_a`, so its optimum is a lower
bound on the label's pricing optimum rather than an upper bound; and it couples
two labels, so it does not answer the per-label question at all. Proposition 9
and Theorem 3 consume the single-zone pricer's bounds and nothing else, and
`branch_price.py` never writes `node_bound` from a redraw. What survives is
what has to: adding columns can only enlarge the master's feasible set and
tighten its LP, every column is re-validated by `ZoneFamily` through
`ZonePool.admit` before entering, and a node still closes only when every label
proved `OPTIMAL` with no improving zone at the LP's own duals. **If the redraw
ever replaced pricing, both results would be lost.**

### The model

One Boolean per block per admissible label over `U`, tied by an exactly-one
row -- which presolve substitutes into a *single* Boolean per block wherever
both labels are candidates, so the solver sees the same
one-variable-per-block zoning model the single-zone pricer does. Contiguity,
balance and anchoring are `zone_family.py`'s rows applied once per label, and
welfare is one block per label, built by the same functions the pricer uses
(`add_mid_welfare`, `add_matching_welfare`, `add_cut_indicators` in
`zone_pricing.py`, shared rather than reimplemented):

- the **`stable_matching`** block keeps its all-Boolean core exactly. The block
  is per label either way, so the seat, clearing and prefix variables stay
  Boolean and the drawn-order chain still encodes the cutoff with no cutoff
  variable;
- the **`mid`** block keeps its small-domain integers (cutoffs, thresholds, the
  mass recurrence), again per label;
- the **access conjunction** is the one place two labels cost more than one: it
  becomes `x_{i,z} AND x_{l(s),z}` for each of the two labels rather than a
  single conjunction -- the halfway point between the single-zone form and the
  `Z`-joint form a whole-district master would need. Against that, `U` is two
  zones' worth of blocks rather than a label's whole candidate set, which on
  the measured instance is 1.5 to 2.5 times smaller.

The incumbent split is supplied as a CP-SAT hint, so an interrupted solve still
returns at least the incumbent and the harvest is never empty. Because there
are no duals to floor, the only rounding is the welfare block's own ceiling
inflation, so a redraw's reported `allowance` is strictly smaller than a
pricing call's.

### When it fires

- **Before the search**, in the strategy, from the seeded incumbent. Under a
  finite budget this is the load-bearing sweep.
- **Inside the search**, whenever a round's restricted LP fails to improve --
  the degenerate-pivot signature -- or pricing stops adding columns. Waiting
  for pricing to be *exhausted* is too late: on a district-sized instance under
  a finite budget it never is, and a redraw hook placed there measurably never
  fired. A round that only redraws re-solves the master and cannot close the
  node. On `Block_2` this trigger is the whole difference between +0.000% and
  +15%.

Each pending pair gets a share of the sweep's remaining budget and pairs are
queued by attempt count, so one expensive pair can neither consume a sweep nor
monopolize later ones.

### Measured on Block_2

The district instance, `max_distance: 3.1`, `centroids_type: 6-zone-9`,
`program_population: All`, citywide programs removed: 501 geographic units,
1,340 edges, `Z = 6`, 4,154 applicants -> 2,309 compressed MID types over 105
programs, and 4,154 individual applicants over 100 programs. Anchoring leaves
candidate sets of `[147, 147, 253, 309, 248, 170]`, and 10 of the 15 label
pairs are graph-adjacent at the seed, so a third of the neighbourhood is
discarded before a model is built.

The seed is one `cp_bool` **feasibility** solve -- it optimizes nothing -- and
its zones are correspondingly lopsided: `[62, 65, 67, 81, 104, 122]`. ReCom
sampling is off so the comparison isolates the redraw. Six workers, 600s.

| objective | hint | `dw_redraw: false` | default | transportation | first-choice |
|---|---|---|---|---|---|
| `mid` | 10,307.2692 | 10,307.2692 | **11,822.4615** | 15,956.4997 | 16,830.9547 |
| `stable_matching` | 10,344.5486 | 10,344.5486 | **12,105.1983** | 15,180.9833 | 15,983.6543 |

**+14.70%** on `mid` and **+17.02%** on `stable_matching` over the seed,
against **+0.000%** without -- and that zero is exact, not rounded. With the
redraw off, both objectives ran 21 rounds and generated 542-555 columns of
strictly positive reduced cost inside the budget, the restricted LP sat at the
hint value to four decimals on every round, and the resulting pool still held
exactly one tiling. The one-tiling diagnosis is not an artefact of the
synthetic instance it was found on.

With the redraw on, the same pool holds 25+ tilings and the LP climbs through
distinct values rather than stepping once:
`10307.27 -> 11579.37 -> 11624.64 -> 11820.40 -> 11822.46` (`mid`) and
`10344.55 -> 11651.25 -> 12028.39 -> 12105.20` (`stable_matching`). Redraw
activity: 63 pairs / 14 improvements / 325 columns / +1,515.19 (`mid`), 75 / 25
/ 271 / +1,760.65 (`stable_matching`).

**At 600s it is budget-limited, not neighbourhood-limited.** Extending `mid` to
1800s reaches 12,050.5995 (**+16.91%**) over 107 pair solves and 28
improvements, with 1,756 columns. That partition was re-validated independently
of the search -- covers V exactly once, every zone satisfies its own family,
and a fresh oracle evaluation per zone reproduces the reported value to 1e-6 --
and its zone sizes are `[65, 65, 77, 90, 94, 110]` against the hint's 62-122.
The gain is the rebalancing of an unbalanced feasibility solve, which is what
program capacity rewards.

For reference, a synthetic grid (324 units, `Z = 6`, capacity at 80% of
applicants needing seats, nearest-centroid seed) gives +3.40% and +0.58%
against the same +0.000%. The district gains are far larger because a
feasibility hint leaves much more on the table than a Voronoi seed does. All
figures are single runs and vary, since CP-SAT is nondeterministic above one
search worker.

### Is it converging?

The two sides of the gap behave completely differently.

**Primal: yes, and on the district it is still converging when the budget
expires.** 600s gave +14.70%, 1800s gave +16.91%. On the small synthetic a
sweep exhausts the neighbourhood in seconds and every later sweep costs 0.0s;
on Block_2 it does not get that far in ten minutes.

**Bound: not at all.** In every run measured -- both instances, both
objectives, redraw on and off, 90s through 1800s -- the certified bound is
*exactly* the first-choice constant. The mechanism is visible in the
arithmetic. By LP duality the first two terms of the pricing-corrected bound
are the restricted master's own objective, so for the bound to stay at
16,830.95 while the LP sits at 12,050.60, the pricing residuals must sum to at
least 4,780 -- close to 800 per label. The entire headroom to the
transportation bound is 15,956.50 - 12,050.60 = 3,906. Six labels are each
separately reporting more available improvement than provably exists in total,
and no six of the zones they return can occupy one tiling.

That is not a defect in the pricing problem, which answers the question the
bound asks it and answers it to proved optimality. It is the price of asking a
per-label question, and it means the reported ~25% gap is mostly slack in the
certificate: against the transportation bound the same incumbents are within
24.5% and 20.3%, and against the zoned transportation bound (which on this
instance at this `max_distance` cuts a further 1,134-1,241 units off the
transportation bound) closer still.

So what remains open is exactly the dual side, and it admits two different
attacks: repair the decomposition's own duals (elastic master,
completability-targeted pricing, a wider redraw neighbourhood), or replace the
constant the bound falls back on with one of the zoning-aware bounds already
developed elsewhere in the paper -- which requires nothing of the
decomposition at all.

## Pricing, in CP-SAT

`zone_pricing.py` solves each pricing problem globally as a CP-SAT model over
binary memberships `x_v`, `v` in label `z`'s candidate set. Geometry is
conditions 1-5 above. Perimeter indicators are created only when they are
needed -- the boundary objective, or a live boundary cap.

CP-SAT rather than Gurobi for two model-specific reasons. The `min` recurrence
that *defines* finite-grid MID welfare is native here (`AddMinEquality`), where
a MIP needs a big-`M` disjunction per `(type, rank)` pair: about 24,000
auxiliary integers on `Block_2`, and the weakest part of that relaxation. And
the zoning block is what CP-SAT is best at in this repo.

Two simplifications come from there being exactly one zone:

- **Access is one Boolean.** Co-zoning of a student vertex `u` and a school
  vertex `w` is `x_u AND x_w`: one variable and three rows, not the `Z`
  conjunction variables plus a `sum(joints) <= 1` row a whole-district master
  needs. In the matching block it can be eliminated outright -- its only
  appearances are an upper bound on `y` (replace by the two separate bounds)
  and the right-hand side of the two stability rows (replace by
  `x_u + x_w - 1`), both exact at integral memberships -- and is kept only
  because the reification propagates better. The MID block needs it: its
  effective-rejection row *selects* between the threshold and `L` rather than
  bounding one of them, so the substitution is over-restrictive there.
- **Capacity tightens.** A program outside the zone seats nobody, so the
  aggregate row is `sum_i y[i,s] <= q_s x_{l(s)}` rather than `<= q_s`, and
  `<= L q_s x_{l(s)}` for the finite-grid masses.

### The `mid` block

For each in-zone program an integer cutoff `p_s in [0, pbar_s]`, with `pbar_s`
from `cutoff_upper_bounds` on the candidate-restricted market -- valid for every
sub-selection, since removing students can only lower the least clearing
cutoff. Then, with `a_tk` the access Boolean of type `t`'s vertex and rank
`k`'s school vertex,

```text
t_{s,rho} = min(L, max(p_s - rho L, 0))            native min/max
e_tk      = t   if a_tk else L                     conditional equality
R_t0      = L,   R_tk = min(R_t,k-1, e_tk)         native min
R_tk     >= R_t,k-1 + e_tk - L                     valid inequality
d_tk      = R_t,k-1 - R_tk
sum_{t,k at s} n_t d_tk <= L q_s x_{l(s)}
```

The valid inequality is what stops the relaxation of the `min` from
manufacturing mass at high-utility ranks; it is valid because the thresholds are
clamped to the lottery interval. Every rank of every list is represented; an
alternative whose school is outside the candidate set has `e = L` identically
and is dropped, which is exact.

### The `stable_matching` block

The access polytope's rows, made integral. `y[i,s]` seats student `i` at `s`,
`z[i,s]` says `i` clears `s`'s realized cutoff, `pi[i,s]` is the prefix
"seated at something weakly preferred to `s`", and `Gamma(s)` is the drawn
priority order restricted to retained students:

```text
(F1)  pi[i, last] <= 1                       Boolean domain of pi
(F2)  sum_i y[i,s] <= q_s x_{l(s)}
(F3)  y[i,s] <= a_{i,s}
(F4)  y[i,s] <= z[i,s]
(F5)  z[i,s] <= z[i',s]  for consecutive (i', i) in Gamma(s)
(F6)  pi[i,s] >= a_{i,s} + z[i,s] - 1
(S1)  sum_{i'} y[i',s] >= q_s (1 - z[i,s])
(S2)  q_s pi[i,s] + sum_{i' >_s i} y[i',s] >= q_s a_{i,s}
```

(F5) makes `{i : z[i,s] = 1}` a prefix of the drawn order, which *is* the
cutoff, so no cutoff variable is needed and nothing has to live on a grid.
Declaring `pi` Boolean states (F1) as a domain rather than a row. (S1) is
unnecessary for exactness but is worth roughly 100x in time through presolve;
(S2) is the aggregate stability row and tightens the relaxation. (S2)'s inner
sum is carried by one running prefix variable per pair, so the family stays
`O(|Gamma(s)|)` rows. Students outside the candidate set are dropped from the
chain and from every sum: their `y` is zero, their (F6) and (S2) rows are
vacuous at `a = 0`, and an absent `z` in the chain is the same as a free one.

`test_branch_price.py` checks both blocks against their oracles at *every*
admissible zone of a test instance, with membership pinned by branch
assumptions: the CP-SAT welfare equals `finite_grid_oracle` and equals the
deferred-acceptance run respectively.

### Integer arithmetic, and what it costs

CP-SAT needs integer coefficients, so the objective is scaled by `K` --
`dw_pricing_scale * L` for `mid`, `dw_pricing_scale` for `stable_matching`,
`2 * dw_pricing_scale` for `boundary` -- and rounded *directionally*: utilities
with a ceiling, duals with a floor. Both push the objective up, so the scaled
objective dominates the true reduced welfare pointwise and CP-SAT's proven
objective bound divided by `K` is a valid upper bound on it.

The cost is that the proven bound can sit slightly *above* zero when no
improving zone exists. That residual is bounded: the ceiling inflates by under
one scaled unit per welfare term (and a type's masses sum to at most `L`, an
applicant's seats to at most 1), and the floor under-charges by under one per
membership or cut variable. `PricingResult.allowance` reports
`units / K` per label; the search sums it into `dw_bound_slack` and uses
`tolerance + dw_bound_slack` as its incumbent comparison. That is the search's
absolute optimality tolerance, stated rather than hidden, and raising
`dw_pricing_scale` shrinks it. The allowance is never subtracted from the
bound, which would break validity.

Nothing else is trusted. A returned zone is re-validated by
`ZoneFamily.feasible`, scored by the exact oracle, and its reduced cost
recomputed by the master, so a rounding artefact can waste a round but can
never admit a column that does not improve.

### Model reuse and harvesting

One model per label and phase, built once and re-aimed: branch fixings are
CP-SAT *assumptions*, released with `ClearAssumptions`, and the duals are a
fresh objective on a welfare variable that binds the whole block, so re-aiming
touches two terms rather than thousands. `dw_pricing_models` should be at most
`2 Z` however many rounds and branch nodes the search took. `problem.fixed` and
`problem.candidates` are structural rows, not assumptions, so releasing a
fixing cannot loosen them.

A solution callback harvests up to `dw_pricing_columns_per_call` improving
zones per call rather than only the last one. When steps are degenerate, more
columns per round is the cheapest thing that helps.

`dw_pricing_parallel` prices the labels in threads against one shared deadline,
each with `workers / Z` search workers. That matters for the bound: it needs a
finite bound from *every* label, and under a sequential split the first label
used to consume the budget and the rest returned unpriced. Sequential pricing
splits the round evenly instead.

## Feasibility pricing and bounds

Seeding is two-stage. A `cp_bool` feasibility solve is the only step that
reliably yields an admissible partition, and it is cached across runs. ReCom
then samples from that hint, and every sampled zone is rejection-filtered
against `F_z`. The filter is per zone, not per partition -- a recombination step
rewrites two zones and leaves the rest of an admissible partition alone -- and
it runs before the welfare oracle, so a rejected sample is nearly free. ReCom's
own cut candidates already respect `candidate_zones` and hence centroid
anchoring, so what its zones typically fail is closer-neighbour support.

Neither stage is required. If the pool cannot cover the graph, Phase I adds
non-negative **deficit-only** artificials to the cover and convexity rows and
maximizes minus their sum. An empty pool is Phase-I feasible; objective zero is
equivalent to a feasible original master. Phase-I pricing uses zero column cost
and the same geometry and branch restrictions. Artificials never appear in a
returned partition.

If each pricing problem has a global reduced-cost upper bound `R_z` at the dual
point `(alpha, gamma, mu)` that was priced, then

```text
UB = sum_v alpha_v + sum_z gamma_z + mu B + sum_z max(0, R_z)
```

bounds the full master: raise each convexity dual by `max(0, R_z)` and the dual
becomes feasible for every omitted column, so weak duality applies. It holds at
*any* dual point with `mu >= 0`, which is exactly why interior duals and
smoothing are free -- the bound is simply computed where the pricing happened.
The same correction is valid in Phase I, because a deficit artificial has
coefficient +1 in its own row, so raising a convexity dual preserves its dual
inequality; and a convex combination of two Phase-I dual points still satisfies
those inequalities, which is why the smoothing centre is reset when the phase
flips. A strictly negative certified Phase-I upper bound proves the branch
infeasible. Time-limited or failed pricing is never read as proof that no
improving column exists.

## Integer optimality and finite convergence

Exact root pricing alone certifies only the LP relaxation, so
`branch_price.py` runs column generation at **every** node of a
branch-and-bound tree. For the current LP define
`q_vz = sum(A containing v) lambda[z,A]`. If `q_vz` is fractional, branch into
`q_vz = 0` and `q_vz = 1`, the latter also fixing every other label for `v` to
zero. Stored columns are filtered by those decisions and the identical
membership fixings are imposed in every subsequent pricing model. Phase I is
re-run when needed; a missing compatible column is not proof of infeasibility.

In exact arithmetic, assuming globally solved bounded LPs and pricing models:

1. There are at most `Z (2^|V| - 1)` distinct labelled memberships. Every
   improving round introduces a new column, so each node's column-generation
   loop terminates. No improving column means its complete LP has been solved,
   or Phase I certifies it infeasible.
2. Branches are disjoint and exhaustive. Each fixes a previously unfixed binary
   geographic-assignment marginal, so tree depth is at most `|V| Z`.
3. If all `q_vz` are integral, every positive column of a given label has
   exactly the membership `q` specifies. Because columns are deduplicated by
   label and membership, its weight is one: the LP yields an integer partition.
4. Infeasible branches, and branches whose certified bound cannot beat the
   incumbent, are discarded. Finite tree exhaustion therefore returns a
   globally optimal admissible partition over the prescribed families, or
   proves none exists.

The two-zone redraw does not enter this argument and does not disturb it.
Step 1 needs only that a round which adds columns adds *new* ones from a finite
set, which the pool's deduplication and `ZoneFamily` re-validation guarantee
whatever the source; steps 2-4 read `node_bound`, which is written by pricing
alone. Termination of a node's loop is likewise unaffected: a redraw sweep adds
finitely many columns, and its `(pair, territory, fixings)` memo means repeated
sweeps at an unchanged incumbent solve nothing, so the redraw cannot generate an
infinite sequence of rounds. Correspondingly it cannot *close* a node either --
a round that only redraws re-solves the master and returns to pricing.

This is finite convergence without relying on ReCom's mixing, on sampling every
partition, or on a heuristic pricing neighbourhood. It is not a polynomial
guarantee.

The one caveat is the scaling above. Because the pricing objective is rounded,
"no improving column" is detected as *pricing proved optimality of the rounded
objective and no zone it returned improves the master exactly*, and the node
bound retains the residual. So the guarantee delivered is optimality to
`tolerance + dw_bound_slack` rather than to `tolerance`, with both reported.
Branching remains exhaustive regardless, so the residual costs tightness, not
correctness.

## Practical stopping and verification

Use `solve_time_limits: [.inf]` for unlimited search; hint and ReCom seeding
still have their own finite budgets. `max_iterations` does not cap this
algorithm. With a finite budget, the incumbent and a valid remaining upper bound
are returned without claiming optimality. `dw_absolute_gap`, `dw_bound_slack`,
`dw_branch_nodes`, pricing-call counts, per-round history and the termination
reason are all in solution metadata.

`OPTIMAL` means the tree closed to the stated tolerance. Numerical stalls return
an unresolved status with a distinguishing `stop_reason`:
`pricing_incomplete` (a label did not prove a bound),
`column_generation_stall` (the LP claimed optimality with an exactly improving
column already in the pool), `phase_one_numerical_stall`,
`integrality_numerical_stall`, `master_*`, `time_limit`. The proof above assumes
exact arithmetic; the implementation uses Gurobi floating-point LP duals and
CP-SAT integer bounds.

Tests compare each welfare block with its oracle at every admissible zone,
compare global pricing with enumeration over every admissible zone under random
duals, reach an enumerated integer optimum from an empty pool for all three
objectives, exercise a strict LP/integer gap requiring branching, check that
Phase I proves an infeasible instance, confirm a reused model honours each
call's assumptions, confirm a never-optimal pricer still contributes its columns
without closing a node, and check that smoothing and both master methods reach
the same certified optimum.

For the redraw specifically (`tests/test_zone_redraw.py`): its optimum is
compared with enumeration of every admissible re-partition of a pair's
territory, for all three objectives, both when that territory is the whole
district and when it is a strict subset; every harvested split is checked to be
family-admissible on both halves and to complete the untouched zones into a
tiling; the dual cancellation of property 2 is checked directly against random
dual points; a seeded pool with exactly one tiling is checked to gain more;
the memo is checked to make a second sweep solve nothing and the swept
partition to be two-zone optimal; branch fixings and the boundary cap are
checked in both the satisfiable and the unsatisfiable direction; and
`branch_and_price` is checked to certify the same optimum and the same valid
bound with the redraw on as with it off.

References: [SCIP pricing callbacks and infeasible-node pricing](https://www.scipopt.org/doc-7.0.1/html/PRICER.php),
[SCIP branch-and-price example](https://www.scipopt.org/doc-6.0.0/html/BINPACKING_MAIN.php),
[Gurobi attributes for LP duals (`Pi`) and the barrier `Crossover` parameter](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes.html),
[CP-SAT assumptions and solution callbacks](https://developers.google.com/optimization/cp/cp_solver).
