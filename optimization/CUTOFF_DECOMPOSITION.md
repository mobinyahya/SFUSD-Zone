# Cutoff zoning decomposition

## Target model

For the configured lottery scale `L`, student `i` is represented by `L` units
of common single-tie-breaker lottery mass. At school `s`, priority tier `p_is`
and cutoff `c_s` give rejection mass

```text
t_is(c_s) = min(L, max(0, c_s - L p_is)).
```

Scanning the schools in the student's strict utility order, let `r_i0 = L`,
`r_ik = min(r_i,k-1, t_i,s_k)`, and assign mass `r_i,k-1 - r_ik` to `s_k`.
Every school has an upper capacity constraint. The finite-grid objective is the
sum of integer cutoffs divided by `L`.

This is an expected-demand score-limit approximation, not an expectation of
finite realized DA outcomes. Integer rounding can leave less than one cutoff
grid step of slack at a positive-cutoff school. Output therefore calls this
vector `grid_minimal`, not stable. A separate continuous Azevedo-Leshno oracle
checks exact market clearing and stability of every reported isolated market.

## Fixed-market oracle

For fixed other cutoffs, a school's demand is continuous, nonincreasing, and
piecewise linear in its own cutoff. `solve_market_cutoffs` scans the priority
breakpoints and selects the smallest integer cutoff satisfying capacity.

Let `F_s(c_-s)` be this smallest response. Increasing another school's cutoff
can only reject more lottery mass before students reach `s`, so `F_s` is
isotone. Cyclic updates start at zero. If `v` is any capacity-feasible cutoff
vector, induction gives every iterate `c^k <= v`. Integer cutoffs are bounded,
so the updates terminate at a feasible vector below every other feasible
vector. It is the coordinatewise least vector and minimizes cutoff sum.

`solve_continuum_market_cutoffs` applies the same breakpoint inversion without
integer rounding. It verifies

```text
demand_s <= capacity_s
cutoff_s > 0  =>  demand_s = capacity_s
```

to numerical tolerance. These are the continuous stable score-limit clearing
conditions.

## Exact zoning master

The master in `optimization/solvers/cutoff_decomposition.py` contains only:

- Node-to-zone variables and the existing centroid, closer-neighbor,
  school-count, demographic, and optional boundary constraints.
- One integer cutoff per school.
- Revealed-preference interval cuts separated from candidate solutions.

Suppose candidate lottery interval `[a,b]` of student `i` chooses school `s` in
zone `z`. It must continue choosing `s` whenever:

```text
i and s remain in z,
c_s <= L p_is + a - 1,
for every r preferred to s: r is outside z or c_r >= L p_ir + b.
```

Under these conditions `s` admits the interval's lowest lottery value and all
preferred in-zone schools reject its highest value. Monotonicity covers the
whole interval. A Boolean lower bound counts that mass against `s`'s capacity.
The cut may undercount demand away from the candidate, but can never exclude a
capacity-feasible zoning/cutoff vector. Identical conditions are grouped.

The algorithm is finite because zoning and cutoff domains are finite and every
infeasible master candidate receives a violated cut. It reports `OPTIMAL` only
when one of these certificates is obtained:

1. The valid-cut master with objective below the incumbent is infeasible.
2. A master-optimal candidate passes every exact capacity check.
3. The integral master lower bound reaches the oracle-feasible incumbent.

Otherwise it reports `FEASIBLE` with the exact grid objective and the valid
global lower bound. It never turns a time-limited incumbent into an optimality
claim.

## Large-market interpretation

The continuous score-limit result applies under the usual assumptions: a fixed
finite school set, independent continuous lotteries across students, the same
lottery at every school for one student, convergent applicant-type and
per-capita capacity distributions, and a unique limiting clearing cutoff.
For endogenous zoning, uniform convergence follows when the geographic graph
and its finite feasible-zoning family are fixed and each induced limiting
market satisfies those regularity conditions. The grid approximation also
requires `L` to increase for its error to vanish; this project intentionally
uses `L=20` and reports both grid and continuous values.

## Why utility is not the objective

For a fixed market, the least cutoff vector is student-optimal, so every
preference-consistent utility representation selects the same stable outcome.
Across zonings, access sets change and there is no common lattice order. A
partition can lower cutoff sum by concentrating competition while another can
give more students high-utility schools. Therefore maximizing student utility
is not an equivalent replacement for cutoff-sum minimization when zoning is a
decision.

## Running

```bash
uv run python -m optimization.run \
  optimization/config.example.cutoffs.yaml \
  --output benchmark_output/cutoff_exact_decomposition
```

The example uses a 540-second optimization limit so loading, continuous
verification, serialization, and metrics remain inside ten minutes.

## Verified benchmark

The final run is in `benchmark_output/cutoff_exact_decomposition_final`.

```text
artifact production span:       544 seconds
grid objective (L=20):          113.45 (raw 2269)
continuous stable objective:    113.13455460193593
graph school counts:            [10, 9, 10, 10, 10, 10]
connected/support-valid zones:  6 / 6
continuous stable markets:      6 / 6
certified grid lower bound:      16.65 (raw 333)
status:                          FEASIBLE
```

The benchmark does not close the finite-instance optimality gap and does not
claim to. The exact decomposition supplies a valid lower bound and reports
`OPTIMAL` only under the certificate conditions above. The verified result is
the best incumbent found under the ten-minute budget, while exact global
optimization remains an anytime computation.
