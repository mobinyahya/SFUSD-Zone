# Citywide-School Cutoff Extension

## Result and guarantee scope

Citywide schools cannot be added to independent geographic markets by giving
each zone the school's full capacity. That duplicates seats. The implemented
extension instead solves one globally coupled matching market:

- attendance schools are accessible only within their geographic zone;
- citywide schools are accessible from every zone;
- every citywide school's capacity and cutoff occur exactly once; and
- zone stability means no blocking accessible pair for residents of that zone
  under the common global citywide cutoffs.

The revealed-preference decomposition remains an exact finite algorithm in
principle. It certifies a global finite-grid optimum only when its master bound
reaches its feasible incumbent. The real run below is a stable, grid-minimal
`FEASIBLE` incumbent, not a certified globally optimal zoning.

## Market model

Let `R` be zone-restricted schools, `U` citywide schools, and `S = R union U`.
For zoning `x`, applicant `i` at vertex `v(i)` can access school `s` when

```text
A_is(x) = 1                                      if s is in U,
A_is(x) = 1{zone_x(v(i)) = zone_x(h(s))}         if s is in R.
```

Applicant `i` has strict preferences and school priority tier `p_is`. One
continuous single tie-breaker `l_i` is shared across schools. The applicant
qualifies at score cutoff `c_s` when

```text
p_is + l_i > c_s.
```

At a cutoff vector, each applicant chooses their most-preferred accessible
qualifying school. Let `D_s^x(c)` denote resulting continuum demand and `q_s`
capacity. Stable market clearing is

```text
D_s^x(c) <= q_s,
c_s > 0  implies  D_s^x(c) = q_s.
```

The zoning objective counts every school once:

```text
C(x) = sum_(s in S) w_s c_s*(x),
```

with fixed, commensurable weights `w_s`; the implementation uses `w_s = 1`.
Citywide cutoffs are not repeated once per zone.

## Why isolated markets fail

Take two zones with unit applicant mass in each, one local fallback per zone,
and one citywide school of capacity one. Everyone ranks the citywide school
first and has one uniform priority score. Giving each isolated zone capacity
one produces cutoff zero and combined demand two. In the correct global market,

```text
D_u(c_u) = 2(1 - c_u),
```

so `c_u = 1/2` and demand is one. Independent geographic markets are valid only
after changing the mechanism, for example by creating distinct zonal reserve
contracts whose capacities sum to the citywide capacity. They are not
equivalent to one districtwide priority ranking.

## Exact finite-grid decomposition

For a fixed zoning, `solve_coupled_cutoffs` filters each preference list to all
citywide schools plus restricted schools in the applicant's zone, then calls
the analytical oracle once with one capacity per school.

The master has geographic assignment variables and integer grid cutoffs
`kappa_s`. Suppose candidate evaluation assigns lottery interval `[a,b]` of
applicant `i`, currently in zone `z`, to school `s`. Define

```text
Q_is^a = 1{kappa_s <= L p_is + a - 1}.
```

For every school `r` preferred to `s`, define the high-end blocker

```text
B_ir^b = 1{kappa_r <= L p_ir + b - 1}                  if r is citywide,
B_ir^b = A_(v(i),r) AND 1{kappa_r <= L p_ir + b - 1}  if r is restricted,
```

where `A_(v(i),r)` says that the applicant block and school share a zone. The
generated demand indicator is linked by the clause

```text
d_J OR NOT Q_is^a OR OR_(r preferred to s) B_ir^b
```

for a citywide target. For a restricted target, add `NOT A_(v(i),s)` to the
clause. Access literals are shared by every interval involving the same block
and school. For every candidate school zone they are encoded only as
`school_in_z -> (A_(v(i),s) == block_in_z)`, not as a biconditional. The
exactly-one school assignment makes those implications sufficient to determine
access.

Capacity is imposed once per school over intervals generated in every zone:

```text
sum_(J targeting s) |J| d_J <= L q_s.
```

### Cut validity and finite termination

If a generated indicator is forced to one, every lottery cell in its interval:

1. remains in the generating access market;
2. qualifies at the target at the interval's low endpoint; and
3. is rejected by every preferred accessible school at the high endpoint.

It therefore demands the target throughout the interval. The cut may
undercount after geography or cutoffs change, but cannot overcount.

At the candidate that generated the cut, all target intervals satisfy these
conditions exactly. If candidate demand exceeds capacity, the new capacity cut
is violated and removes that candidate. The zoning domain and integer cutoff
domain are finite, so exact master solves plus repeated separation terminate at
a global optimum of the encoded finite-grid model. Time-limited master solves
retain valid incumbents and bounds but need not produce that certificate.

## Large-market expected-optimality theorem

Let `Omega` be the finite set of zonings allowed by the encoded geographic
constraints. Assume:

1. the schools, graph, and `Omega` remain fixed as market size `n` grows;
2. empirical applicant types converge to a distribution `mu` (iid sampling is
   sufficient), and capacities satisfy `q_s^n / n -> q_s > 0`;
3. preferences are strict, priorities are responsive, and applicant
   tie-breakers are atomless and independent across applicants;
4. every zoning has a unique limiting globally coupled cutoff equilibrium;
5. cutoffs are bounded and objective weights are fixed and nonnegative; and
6. there are no couples, non-substitutable reserves, or other choice functions
   outside the ordinary school-priority model.

Let

```text
x* in argmin_(x in Omega) C(x),
C(x) = sum_s w_s c_s*(x).
```

Then the matching induced by `c*(x)` is stable in the full mixed-access market
for every `x`. Restricted schools have no blocking pair with eligible residents
of their zone, and citywide schools have none with applicants districtwide.

Let `P^n(x)` be the canonical finite stable cutoff vector and

```text
Cbar_n(x) = E[sum_s w_s P_s^n(x)].
```

Under the assumptions above, finite-market cutoff convergence for each zoning
is uniform because `Omega` is finite. Bounded convergence gives uniform
convergence in expectation, and therefore

```text
0 <= Cbar_n(x*) - min_(x in Omega) Cbar_n(x)
   <= 2 max_(x in Omega) |Cbar_n(x) - C(x)| -> 0.
```

Thus the continuum optimizer is asymptotically globally optimal in expectation.
If `x*` has a strict objective gap, it is also the expected-cutoff optimizer for
all sufficiently large `n`.

Uniqueness is an assumption, not a consequence of a common continuous STB
lottery. It can instead follow from regular demand and generic capacities. A
numerical `stable: true` result verifies market clearing, not uniqueness for
every possible zoning.

## Implementation and verification

Relevant code:

- `optimization/cutoff_oracle.py`: coupled grid and continuum oracles plus
  per-zone access, clearing, reconciliation, and no-blocking checks.
- `optimization/solvers/cutoff_decomposition.py`: global candidate demand and
  generalized interval cuts.
- `optimization/data/cutoffs.py`: citywide access classification and explicit
  outside-option records.
- `optimization/verify_cutoff_scenarios.py`: reproducible real-data matrix.
- `optimization/tests/test_cutoffs.py`: shared-capacity, tamper-detection, and
  exhaustive tiny-zoning tests.

The real citywide run used `Block_2`, six zones, `frl_dev=0.15`, and
`boundary_prop=0.25`:

| Statistic | Result |
|---|---:|
| End-to-end measured time | 171.93 seconds |
| Status | `FEASIBLE` |
| Grid objective | 215.90 |
| Continuous objective | 182.7170785455 |
| Applicants represented | 3,872 of 3,872 |
| Applicants assumed outside-option-only | 69 |
| Restricted / citywide schools | 58 / 14 |
| Stable zone checks | 6 of 6 |
| Grid minimal | Yes |
| Contiguous zones | 6 of 6 |
| Global zoning optimum certified | No |

The 69 applicants without source preference records are explicitly represented
with empty school lists. Stability for them is conditional on the moderate,
visible assumption that they accept only the outside option; it does not recover
their unknown preferences.

Reproduce the run with:

```bash
uv run python -m optimization.verify_cutoff_scenarios \
  --case citywide \
  --solve-seconds 120 \
  --output benchmark_output/cutoff_requested_verification/run_citywide
```

Two separate independent audits passed:

- The mathematical audit validated universal/restricted targets and blockers,
  shared capacity, finite separation, and the large-market theorem under its
  assumptions.
- The runtime/stability audit independently reconstructed grid and continuous
  demands, all six zone checks, FRL, boundary count, contiguity, applicant
  coverage, and the sub-10-minute runtime criterion.
