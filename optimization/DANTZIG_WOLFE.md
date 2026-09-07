# Whole-zone MID branch-and-price

## Scope

The objective is the repository's **finite-grid MID welfare**, with integer
lottery mass `L = mid_lottery_scale`. Citywide schools are excluded from the
geographic school/capacity data, and citywide programs are removed from all
preference lists before market evaluation. Students remain in their geographic
nodes, including those left with no available preferences.

Let `N` be the finite geographic node set and `z = 0,...,Z-1` the zone labels.
Let `F_z` contain every nonempty connected subset satisfying the configured
demographic, aggregate capacity, school-count, explicit candidate, and fixed
assignment constraints for label z. These are the existing ReCom semantics:
centroids do not impose hard anchors or implicit distance limits.
The implementation carries the same 1e-6 feasibility slack in zone validation,
the pricing inequalities, and the master boundary cap (including its dual-bound
calculation). Thus pricing does not silently exclude a zone admitted by seeding.

For `S in F_z`, let `W(S)` be its least-cutoff MID welfare, computed using only
residents of S and programs whose schools lie in S. Without citywide access,
different zones have disjoint students and program capacities. Therefore
district welfare is exactly the sum of zone welfares. The partition master is

```text
maximize  sum(z,S) W(S) lambda[z,S]
subject to
  sum(z,S: i in S) lambda[z,S] = 1       for every i in N
  sum(S in F_z) lambda[z,S] = 1          for every label z
  sum(z,S) b(S) lambda[z,S] <= B         optional boundary cap
  lambda[z,S] in {0,1}
```

Here `b(S)` is half the weighted zone perimeter. Each cut edge appears in two
zone perimeters, so the sum equals the district boundary cost.

## Exact zone pricing

Relax lambda to nonnegative real values. Let `pi_i` and `sigma_z` be the free
duals of cover and label rows, and `mu >= 0` the optional boundary-row dual.
The pricing problem for each label is

```text
maximize over S in F_z:
  W(S) - sum(i in S) pi_i - sigma_z - mu b(S).
```

`zone_pricing.py` solves this globally as a MIP. A binary `x_i` selects node i.
A freely chosen root supplies one flow unit to each selected node; flow can
traverse only selected endpoints, enforcing connectedness without strengthening
it to distance-monotone connectedness. All balance and branch constraints are
part of this same pricing model.

MID uses an integer cutoff `p_j` for each non-citywide program j. For type t,
rank k, priority tier `rho_tk`, and program school node `s(t,k)`, define

```text
a_tk = x[node(t)] AND x[s(t,k)]
h_tk = min(L, max(0, p_j - rho_tk L))
e_tk = h_tk if a_tk = 1, otherwise L
r_t0 = L
r_tk = min(r_t,k-1, e_tk)
d_tk = r_t,k-1 - r_tk
```

The model imposes `sum(t,k at j) count_t d_tk <= capacity_j L` and maximizes
`sum(t,k) utility_sum_tk d_tk / L` minus the membership-dependent dual terms.
Every min/max and access expression has an exact binary linearization; no
prefix tail is relaxed, no candidate zone is omitted, and utility coefficients
are not replaced by rounded integer utilities. Co-located student/school pairs
still require membership: omitting their node cannot earn welfare or use seats.
Cutoffs are bounded by `(maximum observed priority tier + 1) L`, which suffices
to reject all lottery mass.

Why does maximizing over capacity-clearing cutoffs give **least-cutoff** welfare?
Starting at zero, the finite-grid oracle monotonically raises an overloaded
program's cutoff to its smallest clearing value. Every iterate is componentwise
below every capacity-clearing cutoff vector: lowering other programs' cutoffs
can only reduce the remaining mass demanding this program. Integer bounded
cutoffs imply finite termination at the least clearing vector. Lower cutoffs
weakly improve each type's allocation in rank order. With the market's positive,
rank-ordered utilities, least cutoffs therefore maximize welfare among all
clearing vectors. For fixed S all dual terms are constants, so the joint pricing
model attains exactly `W(S)` at a global optimum. Ties between cutoff vectors
do not affect that optimal value.

## Feasibility pricing and bounds

ReCom seeds accelerate the search but are not required. If the restricted pool
cannot cover the graph, Phase I adds nonnegative **deficit-only** artificials to
the cover and label equalities and maximizes minus their sum. An empty pool is
feasible in Phase I; objective zero is equivalent to a feasible original master.
Phase-I pricing uses zero original column cost and the same geometric constraints
and branch restrictions. Every newly admitted zone is scored with the exact MID
oracle before Phase II uses it. Artificial variables never appear in a returned
partition.

If each pricing problem has a global reduced-cost upper bound `R_z`, then

```text
UB = sum(i) pi_i + sum(z) sigma_z + mu B + sum(z) max(0, R_z)
```

is a valid full-master upper bound: increase each label dual by `max(0,R_z)`.
This covers every omitted column. The same correction is valid in Phase I
because deficit artificials have coefficient +1 in their row; increasing a label
dual preserves their dual inequalities. A strictly negative certified Phase-I
upper bound proves the branch infeasible. Time-limited or failed pricing is
never interpreted as proof that no improving column exists.

## Integer optimality and finite convergence

Exact root pricing alone certifies only the LP relaxation. `branch_price.py`
therefore runs column generation at **every node** of a branch-and-bound tree.
For the current LP define `q_iz = sum(S containing i) lambda[z,S]`. If q_iz is
fractional, branch into `q_iz = 0` and `q_iz = 1`. The latter also fixes all other
labels for node i to zero. Filter stored columns by these decisions and impose
the identical membership fixings in every subsequent pricing model. Re-run
Phase I if needed; a missing compatible column is not proof of infeasibility.

In exact arithmetic, assuming globally solved bounded LPs and pricing MIPs:

1. There are at most `Z * (2^|N| - 1)` distinct labelled memberships. Every
   positive-reduced-cost iteration introduces a new column, so each node's
   column-generation loop terminates. No improving columns means its complete
   LP has been solved, or Phase I certifies it infeasible.
2. Branches are disjoint and exhaustive. Each fixes a previously unfixed binary
   geographic-assignment marginal, so tree depth is at most `|N| Z`.
3. If all q_iz are integral, every positive column of a given label must have
   exactly the membership specified by q. Because columns are deduplicated by
   label and membership, its lambda is one: the LP yields an integer partition.
4. Infeasible branches and branches whose certified upper bound cannot beat the
   incumbent can be discarded. Finite tree exhaustion therefore returns a
   globally MID-optimal feasible partition, or proves no feasible partition exists.

This establishes finite convergence without relying on ReCom's mixing,
sampling every partition, or a heuristic pricing neighborhood. It is not a
polynomial-time guarantee; exact pricing and the search tree can be expensive.

## Practical stopping and verification

Use `solve_time_limits: [.inf]` for unlimited search; ReCom seeding still has a
finite `dw_recom_time_limit`. `max_iterations` does not cap this algorithm.
With a finite time budget, return the incumbent and valid remaining upper bound
without claiming optimality. `dw_absolute_gap`, `dw_branch_nodes`, pricing-call
counts, per-node history, and termination reason are saved in solution metadata.

The implementation uses SCIP/GLOP floating-point arithmetic and an explicit
`tolerance` (default 1e-6), rather than rational or interval proof certificates.
Its `OPTIMAL` status means completion to these numerical tolerances. Numerical
stalls return an unresolved status. The mathematical proof above assumes exact
arithmetic and zero gap; it applies to the specified finite-grid model, not to
continuous-lottery MID or to a finer geographic graph.

Tests compare global pricing with every feasible zone on small graphs, compare
the complete search with a fully enumerated integer master, exercise a strict
LP/integer gap requiring branching, recover an empty pool through Phase I, and
check infeasibility and interrupted-pricing behavior.

References: [SCIP pricing callbacks and infeasible-node pricing](https://www.scipopt.org/doc-7.0.1/html/PRICER.php),
[SCIP branch-and-price example](https://www.scipopt.org/doc-6.0.0/html/BINPACKING_MAIN.php),
[OR-Tools LP dual and MIP bound APIs](https://or-tools.github.io/docs/pdoc/ortools/linear_solver/pywraplp.html).
