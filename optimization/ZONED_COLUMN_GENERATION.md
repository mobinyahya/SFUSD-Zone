# Zoned Shi Column Generation: Implementation Assignment

## 1. Assignment status

This document is an implementation assignment. It specifies the mathematical
target, proof obligations, software architecture, tests, and performance gate
for a new optimization strategy. It does not describe the existing
`analytical_column_generation.py` experiment as complete or exact.

The implementation must add a new strategy named
`zoned_column_generation`. It must jointly optimize:

1. a partition of the target graph into complete labeled zones; and
2. the optimal priority-based allocation mechanism within each selected zone,
   using the analytical expected-MNL model of Shi.

The first production target is the `Block_2`, six-zone, year-23 instance with
all citywide schools removed. The target wall-clock budget is 45 minutes.

## 2. Decisions fixed for this assignment

The following decisions are requirements, not open design questions.

| Topic | Required decision |
|---|---|
| Objective | Analytical Shi expected-MNL welfare |
| Geographic column | One complete labeled zone |
| Market coupling | Every modeled school is zone restricted |
| Contiguity | Existing centroid-monotone support constraints |
| Root guarantee | Full outer LP pricing closure, to configured tolerance |
| Integer result | Restricted integer master after root closure or timeout |
| Solvers | Gurobi for floating LP/MIP; OR-Tools CP-SAT and existing solvers for seeds and bounding geographic relaxations |
| First graph | `Block_2` |
| Runtime target | 2,700 seconds total |
| Numerical scope | Exact combinatorial identities; ordinary floating analytical MNL arithmetic |
| Compatibility | Preserve existing finite-grid and fixed-state experimental APIs |

The implementation must not claim a proof-grade real-number certificate from
binary64 `exp`, `log`, LP duals, or MIP bounds. It must provide a complete
mathematical correctness proof in exact real arithmetic and clearly label the
computed result as floating numerical evidence for that model. Integer graph
membership, perimeter, node coverage, and reconstructed assignments must be
checked exactly.

## 3. Objective semantics

This strategy does not optimize the existing finite-grid stable-welfare
objective and does not minimize the sum of DA cutoffs. It optimizes over the
larger class of priority systems characterized by Shi.

For a fixed zone, the social planner may give observationally identical agents
a random budget set of schools. Agents choose their utility-maximizing school
from that budget set. Expected school use may not exceed physical capacity.
Shi's characterization then constructs priorities and quotas that implement
the resulting budget-set probabilities in the continuum/large-market model.
Because this repository has hard segment-specific eligibility, the
implementation witness must retain that hard eligibility and use either the
corresponding generalized regularity condition over effective eligible-set
expansions or the minimum admissibility threshold described in Section H.1.1
of Shi. It is not an exact finite-student implementation without a separate
quota-rounding and simulation layer.

The optimized objective removes the zoning-independent outside-option and
Euler-gamma constant:

\[
\sum_t m_t(v_{t0}+\beta\gamma).
\]

`ZoneSolution.objective` must contain normalized Shi welfare. The omitted
constant and the corresponding raw expected welfare must be reported in
metadata. Existing Q20 and continuum stable-welfare values may be reported as
diagnostics, but they are not this strategy's objective.

The `priorities` currently stored on `AnalyticalWelfareSegment` are not inputs
to the Shi objective. They are relevant only when the selected zoning is later
evaluated under the existing DA-STB stable-welfare oracle. The new mechanism
designs priorities instead of treating them as fixed.

## 4. Scope and non-goals

### 4.1 Required scope

The implementation must:

- support `years: [23]` and `population_type: All`;
- require `remove_city_wide: true`;
- require a finite, strictly positive `cutoff_gumbel_scale`, interpreted as
  the analytical MNL scale `beta`;
- use the finest configured level as the target and initially benchmark
  `Block_2`;
- enforce current candidate-zone, centroid, centroid-neighborhood,
  centroid-monotone support, demographic, school-count, and boundary rules;
- skip the graph-level aggregate seats-to-students constraint because school
  capacities are modeled inside each Shi market;
- produce at least one valid complete zoning whenever a valid seed exists;
- return a sparse mechanism witness for every selected zone;
- close the complete-zone root LP when all label pricing upper bounds are at
  most the reduced-cost tolerance;
- preserve a valid incumbent and report nonclosure after any time limit.

### 4.2 Explicit non-goals

The first implementation must not:

- support citywide schools or capacities shared across zones;
- use ordinary graph connectivity in place of the current monotone-support
  domain;
- claim that a closed fractional root LP proves integer optimality;
- implement full outer branch-and-price for the integer zoning problem;
- treat deterministic full-zone access as equivalent to Shi's optimal local
  mechanism;
- use finite-grid access or stable-welfare pricing as an analytical MNL bound;
- use the existing fixed-initial-bundle analytical experiment as a complete
  pricing oracle;
- silently ignore the configured solver, seed, workers, or time limits.

## 5. Mathematical definitions

### 5.1 Geography

Let `G = (V,E)` be the undirected zoning graph and let
`K = {0,...,Z-1}` be the zone labels. Label `k` has centroid node `c_k`.

For a node set `Q` define its edge perimeter as

\[
\delta(Q)=|\{uv\in E:u\in Q,\ v\notin Q\}|.
\]

Let `Omega_k` be the family of locally legal node sets for label `k`. A set
`Q` belongs to `Omega_k` exactly when it satisfies all of the following:

1. `c_k` is in `Q` and every other centroid is outside `Q`.
2. Every selected node admits label `k` under `problem.candidate_zones`.
3. Every node in the configured centroid-neighborhood radius is selected.
4. For each selected noncentroid node, at least one configured support node is
   selected. The strict closer relation makes repeated support traversal end
   at `c_k`.
5. Every enabled FRL and racial balance inequality holds using the repository's
   CP-SAT semantics: round each node coefficient
   `100 * (value_v - ratio * students_v)` first, then sum the resulting
   integers. Do not substitute Gurobi's unrounded floating balance rows.
6. The existing per-zone graph-school count bounds hold.
7. If the district permits at most
   `B = floor(boundary_prop * |E|)` cut edges, then
   `delta(Q) <= B`. This is a necessary local condition for any complete
   partition satisfying the district cap.

The aggregate graph capacity balance constraint must not be part of
`Omega_k` for this strategy. Physical school capacities enter the local Shi
program below.

### 5.2 Segments and schools induced by a zone

Let `T` be the analytical segments and `J` the modeled schools. Segment `t`
has node `o(t)`, mass `m_t > 0`, eligible school set `E_t`, systematic school
utilities `v_tj`, and outside systematic utility `v_t0`. School `j` has graph
node `l(j)` and integer physical capacity `c_j >= 0`.

For a zone `Q`, define

\[
T(Q)=\{t:o(t)\in Q\},\qquad
J(Q)=\{j:l(j)\in Q\}.
\]

Every school is zone restricted, so segments in `T(Q)` can consume capacity
only from schools in `J(Q)`.

### 5.3 MNL menu values

For `j in E_t`, define the attraction

\[
a_{tj}=\exp((v_{tj}-v_{t0})/\beta)>0.
\]

The outside attraction is normalized to one. For a menu
`S subseteq E_t`, define

\[
D_t(S)=1+\sum_{j\in S}a_{tj},
\]

\[
P_t(j,S)=
\begin{cases}
a_{tj}/D_t(S), & j\in S,\\
0, & j\notin S,
\end{cases}
\]

and normalized expected utility

\[
U_t(S)=\beta\log D_t(S).
\]

The empty menu is always available and has `U_t(empty)=0` and zero school
choice probabilities.

### 5.4 Optimal Shi value of one fixed zone

For fixed `Q`, let `y_tS` be the probability that segment `t in T(Q)` receives
menu `S subseteq E_t intersect J(Q)`. The exact local Shi primal is

\[
F(Q)=\max_y
\sum_{t\in T(Q)}m_t\sum_S U_t(S)y_{tS}
\tag{SP(Q)}
\]

subject to

\[
\sum_S y_{tS}=1
\qquad \forall t\in T(Q),
\]

\[
\sum_{t\in T(Q)}m_t\sum_S P_t(j,S)y_{tS}\le c_j
\qquad \forall j\in J(Q),
\]

\[
y_{tS}\ge0.
\]

The associated dual is

\[
F(Q)=\min_{u,p}
\sum_{t\in T(Q)}m_tu_t+\sum_{j\in J(Q)}c_jp_j
\tag{SD(Q)}
\]

subject to

\[
u_t+\sum_{j\in J(Q)}p_jP_t(j,S)\ge U_t(S)
\qquad \forall t,\ S\subseteq E_t\cap J(Q),
\]

\[
p_j\ge0,\qquad u_t\text{ free}.
\]

Both programs have exponentially many menu variables or constraints. They
are solved by Shi's MNL assortment separation:

\[
g_t(p;A)=\max_{S\subseteq E_t\cap A}
\left\{U_t(S)-\sum_jp_jP_t(j,S)\right\}.
\]

For unrestricted menus this optimum lies among prefixes of eligible schools
ordered by nondecreasing `p_j`. The existing prefix implementation in
`analytical_bounds.py` is the starting point, but it must be upgraded to
return matching primal and dual witnesses after closure.

## 6. Complete-zone master

### 6.1 Pattern contract

An analytical zone pattern `P` contains:

- `label`: zone label `k`;
- `nodes`: the complete node set `Q in Omega_k`;
- `perimeter`: exact integer `delta(Q)`;
- `shi_welfare`: closed local primal value `F(Q)`;
- `school_ids`: derived from `J(Q)`;
- `segment_ids`: derived from `T(Q)`;
- `mechanism`: optional sparse local primal witness;
- `valuation_status`, residuals, and timing diagnostics.

The stable pattern key is `(label, frozenset(nodes))`. School and segment sets
must be derived and checked, not accepted as independent identity fields.
Two occurrences of the same key with materially different closed welfare are
a correctness error.

Do not change `ZonePattern.raw_welfare` to floating point. Add a separate
`AnalyticalZonePattern` contract so the finite-grid branch-price code keeps its
integer guarantees.

### 6.2 Full master

Let `lambda_P` select an analytical pattern. The full master LP is

\[
\max_\lambda\sum_PF(P)\lambda_P
\tag{MP}
\]

subject to label convexity

\[
\sum_{P:\operatorname{label}(P)=k}\lambda_P=1
\qquad \forall k\in K,
\tag{1}
\]

node exact cover

\[
\sum_{P:v\in P}\lambda_P=1
\qquad \forall v\in V\setminus\{c_k:k\in K\},
\tag{2}
\]

and the district boundary cap

\[
\sum_P\delta(P)\lambda_P\le2B.
\tag{3}
\]

Centroid coverage rows are omitted because each duplicates its label's
convexity row. If `boundary_prop < 0`, omit row (3), or equivalently use the
nonbinding cap `2|E|`.

The integer restricted master replaces `lambda_P >= 0` by
`lambda_P in {0,1}`. It is solved only over generated columns.

### 6.3 Outer master dual and reduced cost

Let `sigma_k` be free duals for (1), `pi_v` free duals for (2), and `mu >= 0`
the boundary multiplier in the repository's maximization sign convention.
The reduced cost of pattern `P in Omega_k` is

\[
\bar c(P)=F(P)-\sigma_k-\sum_{v\in P}\pi_v-\mu\delta(P),
\tag{4}
\]

where omitted centroid coverage prices are zero.

Pricing for label `k` is therefore

\[
\Theta_k(\pi,\mu)=
\max_{Q\in\Omega_k}
\left\{F(Q)-\sum_{v\in Q}\pi_v-\mu\delta(Q)\right\}.
\tag{PRICE(k)}
\]

The maximum reduced cost for label `k` is `Theta_k - sigma_k`.

## 7. Exact all-bundle pricing formulation

The key implementation requirement is to solve `PRICE(k)` without enumerating
all zones and without fixing the school bundle.

### 7.1 Variables

For one label `k`, introduce:

- binary `x_v`, equal to one when node `v` belongs to the candidate zone;
- continuous `b_uv in [0,1]`, equal to one when an integral geographic
  solution places exactly one endpoint of edge `uv` in the candidate zone;
- continuous `w_tS >= 0`, equal to the mass of segment `t` offered menu `S`.

The menu variables are generated dynamically. They are not binary: random
budget-set probabilities are part of Shi's mechanism.

### 7.2 Joint pricing model

The exact joint formulation is

\[
\max_{x,b,w}
\sum_t\sum_{S\subseteq E_t}U_t(S)w_{tS}
-\sum_v\pi_vx_v-\mu\sum_{uv\in E}b_{uv}
\tag{JP(k)}
\]

subject to all linear constraints defining `x in Omega_k`, exact edge XOR
constraints, and

\[
\sum_{S\subseteq E_t}w_{tS}=m_tx_{o(t)}
\qquad \forall t,
\tag{5}
\]

\[
\sum_t\sum_{S\subseteq E_t}P_t(j,S)w_{tS}
\le c_jx_{l(j)}
\qquad \forall j,
\tag{6}
\]

\[
w_{tS}\ge0.
\]

For edge `uv`, use the complete XOR hull

\[
b_{uv}\ge x_u-x_v,\quad b_{uv}\ge x_v-x_u,
\]

\[
b_{uv}\le x_u+x_v,\quad
b_{uv}\le2-x_u-x_v.
\]

No explicit `w_tS <= x_l(j)` constraint is needed. Equation (6) is the
school-activation link. If `x_l(j)=0`, its right-hand side is zero. Every MNL
menu containing eligible school `j` has `P_t(j,S)>0`, so nonnegativity forces
all such menu mass to zero. This observation is what preserves Shi's original
polynomial menu separation.

### 7.3 Menu-column separation inside pricing

At a branch-and-price node, relax all unfixed `x_v` to `[0,1]` and solve a
restricted version of `JP(k)`. Let `u_t` be the type-row dual and `p_j >= 0`
the school-capacity price. The reduced cost of a missing menu variable is

\[
U_t(S)-u_t-\sum_jp_jP_t(j,S).
\tag{7}
\]

Thus separation for every type is exactly Shi's MNL subproblem

\[
\max_{S\subseteq E_t}
\left\{U_t(S)-\sum_jp_jP_t(j,S)\right\},
\tag{8}
\]

which is solved by prefix enumeration. For each type, let

\[
d_t=\max\{0,g_t(p;J)-u_t\}
\tag{8a}
\]

be its maximum positive omitted reduced cost. Add every unseen menu with
reduced cost above `zoned_cg_menu_tolerance`, then re-solve the LP. The empty
menu must exist for every segment from initialization onward.

The restricted LP objective is not an upper bound until menu separation has
closed at zero reduced cost. With positive tolerance, let `D_R` be a feasible
restricted-LP dual objective. Since equation (5) implies that total menu mass
for type `t` is at most `m_t`,

\[
UB_{node}=D_R+\sum_tm_td_t
\le D_R+\text{menu_tolerance}\sum_tm_t
\tag{8b}
\]

is a valid exact-arithmetic upper bound on the full node LP. Use this repaired
bound for branch pruning and global pricing bounds. For a fixed integral zone,
the equivalent repair raises local type potential `u_t` by `d_t`. Keep the
primal objective as the feasible column value and the repaired dual as its
upper bound. Never use an uncorrected restricted LP objective to prune or
certify pricing.

### 7.4 Branching and global pricing bound

Implement a small explicit best-bound branch-and-price driver around Gurobi
LP solves. Do not attempt to add columns from a Gurobi MIP callback.

For each open node:

1. apply inherited `x_v = 0/1` fixes;
2. solve its LP relaxation with menu generation and compute either its closed
   bound or the repaired bound (8b);
3. prune when that valid bound is no better than the pricing incumbent;
4. if all `x_v` are integral, extract a legal zone and local Shi witness; if
   any `d_t > 0`, retain its repaired bound in a terminal-bound ledger;
5. otherwise branch on a fractional `x_v`, preferring school nodes and values
   nearest `0.5`;
6. push both feasible children into the best-bound queue.

Maintain the following queue invariant under every timeout: each unexplored
integer subtree is represented by a valid bound. Do not remove a parent bound
until both children have inherited that bound or obtained stronger valid
bounds, or until the parent is fathomed. Include the currently processed
subtree in the global bound. A child initially inherits its closed or repaired
parent bound. If menu generation or a fallback model times out before producing
a stronger bound, retain the inherited value.

The final pricing upper bound is the maximum of the incumbent, open-node
bounds, inherited/fallback bounds, and every residual-positive integral node
in the terminal-bound ledger. An integral geography is not fully fathomed by
its primal value while its menu residual remains positive. A terminal node
whose bound is within tolerance of the incumbent may stop receiving work, but
its bound must remain in the final ledger, or be replaced by another explicit
valid bound, such as `incumbent + tolerance`. Never discard it merely because
the remaining gap is small.

Preserve the best validated complete seed assignment outside the restricted
integer master and pass it as a MIP start. If the timed restricted MIP returns
no incumbent, return the seed. If the root LP itself is integral, retain that
reconstructed assignment directly.

The menu variables remain continuous at an integral geographic leaf. This is
intentional and exactly represents randomized budget sets.

The pricing result must expose:

- best integral pattern and exact revalued reduced cost;
- best global upper bound across the incumbent, open/inherited/fallback nodes,
  and the residual-positive terminal ledger;
- whether the branch-and-price search closed;
- menu-generation and geographic branch counts;
- statuses and times for every fallback bound;
- an auditable reason when closure is false.

### 7.5 Analytical fallback bounds

Every time-limited pricing call must still return a mathematically admissible
exact-real relaxation and a solver-reported floating upper bound for that
relaxation. Use at least the following two bounds.

The unconstrained-access bound is

\[
A_v=\sum_{t:o(t)=v}m_t\beta
\log\left(1+\sum_{j\in E_t\cap J_k^+}a_{tj}\right),
\]

where `J_k^+` contains every school whose node can legally join label `k`.
Then solve

\[
\max_{x\in\Omega_k}
\sum_v(A_v-\pi_v)x_v-\mu\delta(x).
\tag{9}
\]

This drops capacities and gives every selected resident all potentially
available schools, so it upper-bounds `PRICE(k)`.

A stronger capacity-price bound may use any nonnegative school-price vector
`p`:

\[
G_t^+(p)=\max_{S\subseteq E_t\cap J_k^+}
\left\{U_t(S)-\sum_jp_jP_t(j,S)\right\},
\]

\[
F(Q)\le
\sum_{t\in T(Q)}m_tG_t^+(p)
+\sum_{j\in J(Q)}c_jp_j.
\tag{10}
\]

The right-hand side is additive by graph node and can be optimized with the
same geographic MIP or CP-SAT model. Use prices from a relaxed Shi market,
the current candidate zone, and prior rounds. Take the minimum of all valid
upper bounds, never the maximum.

An incumbent from local search, a solution pool, restricted menu enumeration,
or fixed-cutoff geography is only a lower bound for maximization pricing. It
must not be reported as the global pricing upper bound.

Also provide an immediate finite model-free fallback that does not require a
second solver to finish. It may drop capacity, support, balance, and perimeter
penalties and add the positive part of every optional node contribution to the
forced-node contribution. It will be loose, but it guarantees that a popped
branch node never loses representation in the timeout bound.

## 8. Root column-generation algorithm

The outer algorithm is:

```text
build and validate target problem and analytical market
collect complete seed assignments from every configured source
extract, validate, deduplicate, and exactly value every zone pattern
require at least one seed partition that makes the restricted master feasible

repeat until deadline or zoned_cg_max_rounds:
    solve restricted analytical master LP
    read convexity, coverage, and perimeter duals
    price every label, in parallel within the worker budget
    exactly revalue every candidate pattern before insertion
    add all unseen patterns with positive reduced cost
    if every label upper-bound reduced cost <= tolerance:
        mark root LP closed and stop
    if no columns were added but any label remains unresolved:
        continue with stronger pricing or stop unclosed on budget

solve restricted integer master over every generated pattern
reconstruct and exactly validate the selected partition
re-solve each selected local Shi program and save its mechanism witness
optionally evaluate Q20 and continuum stable welfare as diagnostics
return one ZoneSolution
```

Do not price only the label with the largest raw bound. Prioritize labels by
`pricing_upper_bound - sigma_k` and resolve every label whose upper reduced
cost remains positive.

At any round, label bounds `U_k >= Theta_k` that are valid for the exact-real
relaxation yield a repaired full-master dual bound

\[
UB=\sum_v\pi_v+2B\mu
+\sum_k\max(\sigma_k,U_k).
\tag{11}
\]

Report the solver-reported numerical version of this value even on timeout.
`nextafter` is a useful final outward adjustment, but by itself it does not
make floating coefficient construction or a commercial solver proof-grade.

## 9. Mathematical correctness proof

This section is a required proof obligation. The implementation subagent must
verify each lemma against code and preserve the assumptions explicitly.

### Lemma 1: Monotone supports imply connectedness

For every selected noncentroid node `v` in a legal pattern, the local support
constraint selects at least one support node strictly closer to `c_k`.
Repeatedly choose such a support. Strict decrease prevents cycles and, on the
finite graph, the sequence must terminate. Every selected noncentroid with no
support is forbidden, while the only allowed terminal is `c_k`. Reversing the
sequence gives a selected path from `c_k` to `v`. Therefore the selected
subgraph is connected.

### Lemma 2: Perimeter identity

Let `(Q_k)` be a node partition. Every uncut edge has either both endpoints in
one `Q_k` or neither endpoint in each other zone, so it contributes zero to
every `delta(Q_k)`. Every cut edge has one endpoint in one zone and the other
endpoint in another zone, so it contributes once to each of exactly two zone
perimeters. Hence

\[
\sum_k\delta(Q_k)=2|\{uv\in E:\operatorname{zone}(u)\ne
\operatorname{zone}(v)\}|.
\]

This proves the factor two in outer master row (3).

### Lemma 3: `JP(k)` equals the fixed-zone Shi program at integral geography

Fix an integral feasible `x` and let `Q={v:x_v=1}`.

If `t notin T(Q)`, equation (5) has right-hand side zero. Nonnegativity implies
`w_tS=0` for every menu. If school `j notin J(Q)`, equation (6) has right-hand
side zero. Every eligible MNL menu containing `j` has strictly positive
`P_t(j,S)`, so nonnegativity implies `w_tS=0` for every such menu with positive
mass. Therefore positive menu mass exists only for residents of `Q` and menus
contained in `J(Q)`.

For `t in T(Q)`, define `y_tS=w_tS/m_t`. Equation (5) becomes
`sum_S y_tS=1`. Equation (6) becomes exactly the capacity row of `SP(Q)`, and
the objective contribution becomes

\[
\sum_{t\in T(Q)}m_t\sum_SU_t(S)y_{tS}.
\]

Thus every `JP(k)` solution induces an `SP(Q)` solution of equal welfare.
Conversely, any `SP(Q)` solution extends to `JP(k)` by setting
`w_tS=m_ty_tS` for residents and zero otherwise. Therefore the optimal
`JP(k)` value at fixed integral `x` is

\[
F(Q)-\sum_{v\in Q}\pi_v-\mu\delta(Q).
\]

### Lemma 4: Shi prefix separation closes the pricing-node LP

Fix a branch node and its current LP relaxation of geography. In the full LP,
the only omitted variables are `w_tS`. Their objective coefficient is
`U_t(S)`, their coefficient in type row (5) is one, and their coefficient in
capacity row (6) is `P_t(j,S)`. Therefore equation (7) is their exact reduced
cost.

Shi's MNL assortment theorem returns a menu attaining the maximum in (8).
If this maximum minus `u_t` is nonpositive for every type, no omitted variable
has positive reduced cost. The restricted LP dual is then feasible for the
full menu LP, while the restricted primal is feasible for the full LP.
Strong LP duality proves equality of the restricted and full node-LP values.

If separation stops at positive tolerance, the restricted objective alone is
not an upper bound. For any feasible full solution, omitted menu variables of
type `t` have total mass at most `m_t` and reduced cost at most `d_t`.
Weak duality plus this residual contribution gives the repaired bound (8b).
This proves exact closure only when every `d_t=0`; otherwise it proves the
stated additive residual bound.

### Lemma 5: Inner branch-and-price solves `PRICE(k)`

The binary variables of `JP(k)` are exactly the geographic `x` variables (and
edge variables determined by them); menu variables are continuous by model
definition. Branching on a fractional `x_v` partitions the remaining integer
feasible solutions into the disjoint exhaustive cases `x_v=0` and `x_v=1`.
By Lemma 4, every closed or residual-repaired branch-node LP is an upper bound
on every integer completion below that node. By Lemma 3, every leaf with
integral `x` supplies the exact reduced objective of its legal zone when its
local menu LP closes, or a primal value and repaired dual bound otherwise.
Standard best-bound branch-and-bound therefore returns the optimum of
`PRICE(k)` when all nodes are fathomed with zero residual. Under positive
tolerance it returns the corresponding reported additive gap.

Under interruption, the queue invariant in Section 7.4 ensures that every
unexplored integer completion remains below either an open-node bound, an
inherited parent bound, a fallback bound, or a residual-positive integral
terminal bound. The maximum of the incumbent and all such bounds therefore
remains an upper bound on `Theta_k` in exact real arithmetic.

### Lemma 6: The complete-zone integer master is equivalent to joint zoning

Any feasible integer master selects exactly one locally legal pattern for each
label by row (1), and rows (2) make the selected node sets disjoint and
exhaustive. Centroid ownership follows from pattern validity. Lemma 2 and row
(3) enforce the district cut-edge limit. Therefore selected patterns form one
legal district zoning.

Conversely, every legal labeled zoning gives one pattern `Q_k in Omega_k` for
each label. Selecting those patterns satisfies rows (1)-(3). Because schools
are zone restricted, the local priority mechanisms consume no resources from
another zone, so their feasible sets and objectives are independent. The best
joint mechanism for the fixed zoning consequently has value
`sum_k F(Q_k)`. The complete-zone integer master therefore optimizes exactly
over legal zonings and independent within-zone Shi mechanisms.

### Lemma 7: Outer reduced-cost closure solves the full master LP

The dual constraint for pattern `P in Omega_k` is

\[
\sigma_k+\sum_{v\in P}\pi_v+\mu\delta(P)\ge F(P).
\]

By equation (4), this is equivalent to `bar c(P) <= 0`. Exact label pricing
maximizes the left-out reduced-cost expression over every pattern in
`Omega_k`. If `Theta_k-sigma_k <= 0` for every label, the restricted-master
dual is feasible for every column in the full master. The restricted primal is
already feasible for the full master. Strong duality proves that the
restricted and full master LP objectives are equal.

With tolerance `epsilon`, closure means no missing column has reduced cost
greater than `epsilon`. Since every feasible master solution has total column
weight `Z` by the convexity rows, the full-master objective can exceed the
restricted objective by at most `Z*epsilon`, ignoring floating arithmetic
error. Equation (11) supplies the sharper repaired-dual bound actually
reported by the implementation.

### Lemma 8: Conditions that prove integer optimality

The full master LP is a relaxation of the complete-zone integer master. If an
exactly closed full-master LP has an integral optimum, it is itself a feasible
integer zoning. Its value is both an upper bound on every integer zoning and
the value of a feasible integer zoning, so it is globally integer optimal.
This condition is sufficient, not necessary. More generally, any feasible
integer incumbent with value `L` is proved optimal whenever a valid
full-master upper bound `U` satisfies `U=L`.

If the closed LP is fractional and its upper bound is strictly above the
incumbent, an integer solution over generated patterns remains only an
incumbent. With positive reduced-cost or solver tolerances, report the additive
gap `U-L`; do not claim exact-real integer optimality merely because the
restricted root solution appears integral.

### Lemma 9: Mechanism witness is implementable by DA

For a selected fixed zone, let `y_tS` be its optimal local primal and define

\[
q_j=\sum_tm_t\sum_SP_t(j,S)y_{tS}\le c_j.
\]

For a segment-`t` agent in the continuum market, independently draw menu `S`
with probability `y_tS` and draw `d` uniformly on `(0,1)`. Retain the hard
eligibility set `E_t` and give eligible school `j` priority score

\[
\rho_{tj}=1[j\in S]+d.
\]

Use quota `q_j` together with a minimum admissibility threshold of one, as in
Shi's Section H.1.1. Equivalently, prove the generalized regularity result only
over hard-eligible budget-set expansions. The agent can then afford exactly
the eligible schools in `S` and chooses the favorite according to the MNL
utility realization. Expected demand is exactly `q_j`. Positive MNL
attractions make outside-option probability strictly decrease whenever an
effective eligible school is added. Thus the sparse local primal plus quotas
is a constructive continuum/large-market DA priority-system witness.

This witness has fractional continuum quotas. It must not be labeled an exact
implementation in the repository's finite-student assignment runner. A finite
deployment requires a separate quota-rounding rule and simulation validation.

### Theorem: Correctness of the completed strategy

Assume finite utilities, positive `beta`, positive attractions for eligible
schools, hard segment eligibility or the stated minimum admissibility
threshold, isolated zone-restricted school capacities, exact menu separation,
and exact real arithmetic. Lemmas 1-6 show that each complete-zone column has
the correct locally optimized Shi coefficient and that the integer master is
the intended joint zoning-and-mechanism problem. Lemma 7 proves full outer LP
closure. Lemma 8 gives sufficient and bound-based conditions for integer
optimality. Lemma 9 constructs the within-zone continuum DA priority
mechanism. Therefore the specified algorithm returns a valid joint zoning and
continuum priority mechanism and a globally optimal full-master LP value on
exact closure. It proves an integer zoning globally optimal whenever its value
meets a valid full-master upper bound, including the sufficient case of an
integral exactly closed root optimum. Positive tolerances instead give the
explicit additive gaps derived above.

## 10. Numerical requirements

The proof above is in exact real arithmetic. The implementation uses floating
arithmetic and must follow these rules:

1. Reject an analytical log-attraction range greater than 700, consistently
   with `analytical_welfare_oracle.py`.
2. Never use an unclosed inner restricted LP as a pricing upper bound without
   the residual repair (8b).
3. Recompute each candidate's local primal and dual after extraction.
4. Require primal feasibility, dual feasibility, and primal-dual agreement to
   configured tolerances before accepting a column coefficient as closed.
5. When a violating menu is already present, report numerical nonclosure
   instead of silently terminating.
6. Treat Gurobi `ObjBound` plus outward `nextafter` as a solver-reported
   numerical bound, not an exact-real certificate. If CP-SAT is used for an
   analytical bound, upper-round every maximization coefficient under directed
   integer scaling, divide outward, and outward-round subsequent arithmetic;
   nearest-integer scaling is invalid for an upper bound.
7. Keep incumbent objectives separate from upper bounds in every result type.
8. Validate integer node coverage, centroids, candidates, supports, balances,
   school count, perimeter, and the factor-two perimeter identity in Python.
9. Set metadata `numerical_scope` to
   `FLOATING_ANALYTICAL_NOT_PROOF_GRADE`.
10. Do not reuse the integer `LagrangianCertificate` for floating Shi values.
11. Report local menu residual, outer reduced-cost residual, incumbent-bound
    gap, and the exact tolerance contribution separately.

If proof-grade analytical bounds are later required, add a separate mode using
directed interval evaluation of `exp` and `log`, rational/outward coefficient
bounds, and a verified LP/MIP certificate. That work is not part of this
assignment.

## 11. Software architecture

### 11.1 New files

Create the following modules.

`optimization/strategies/zoned_column_generation.py`

- Register `@register("zoned_column_generation")`.
- Validate strategy-level assumptions.
- Build `ZoneProblem` and `AnalyticalWelfareMarket`.
- Set `problem.boundary_prop` explicitly.
- Collect seeds, invoke the root driver, construct one `ZoneSolution`, and
  save optional mechanism diagnostics through JSON-safe metadata or artifacts.

`optimization/branch_price/analytical_patterns.py`

- Define `AnalyticalPatternKey` and `AnalyticalZonePattern`.
- Derive residents and schools from node membership.
- Wrap objective-independent structural validation.
- Cache closed Shi valuations by market fingerprint and node-set key.

`optimization/branch_price/analytical_master.py`

- Implement the floating restricted LP with the same row and sign conventions
  as `master.py`.
- Implement a Gurobi restricted integer master with a time limit.
- Reconstruct and validate assignments.
- Keep finite-grid `RestrictedPatternMaster` unchanged.

`optimization/branch_price/analytical_pricing.py`

- Implement `JP(k)`, menu separation, explicit geographic branch-and-price,
  access and capacity-price fallback bounds, candidate revaluation, and result
  diagnostics.
- Share menu columns across branch nodes and cache prefix solutions by the
  school-price vector only when the cache key is numerically safe.

`optimization/branch_price/analytical_root.py`

- Implement repeated outer LP solving and all-label pricing to closure.
- Enforce a single global deadline.
- Add multiple positive columns per label and round.
- Assemble the repaired floating upper bound in equation (11).
- Solve the restricted integer master and report status semantics.

`optimization/column_generation_seeds.py`

- Load portable saved assignments.
- Normalize labels by centroid membership.
- Expand coarse assignments with `LevelConverter`.
- generate solver/ReCom seeds without trusting ReCom labels or boundary limits;
- generate validated support-closed boundary and school-swap local moves;
- extract, validate, value, and deduplicate complete-zone patterns.

`optimization/config.example.zoned_column_generation.yaml`

- Provide the canonical year-23, six-zone, `Block_2`, isolated-market run.

`optimization/tests/test_analytical_pricing.py`

- Add exhaustive menu and zone-pricing tests.

`optimization/tests/test_zoned_column_generation.py`

- Add outer closure, strategy, serialization, and status tests.

### 11.2 Modified files

Modify these files without breaking their current APIs.

`optimization/analytical_bounds.py`

- Preserve `solve_shi_menu_bound` for existing callers.
- Add a result that includes closed primal value, primal menu probabilities,
  quotas, dual value, prices, potentials, residuals, and numerical status.
- Solve the restricted primal after dual separation closes.
- Detect repeated violations and align attraction validation with the stable
  analytical oracle.

`optimization/branch_price/patterns.py`

- Extract objective-independent structural validation.
- Skip graph aggregate capacity when either a cutoff market or analytical Shi
  market supplies school-level capacity recourse.
- Preserve integer pattern behavior and public finite-grid functions.

`optimization/branch_price/geography.py`

- Apply the same objective-independent graph-capacity rule.
- Keep the finite-grid behavior unchanged.

`optimization/problem.py` and `optimization/solvers/balance.py`

- Add one shared predicate or `ZoneProblem` property such as
  `has_school_capacity_recourse` that is true for cutoff markets and analytical
  Shi markets.
- Make exclusion of graph aggregate capacity explicit instead of testing only
  `problem.cutoff_market` in scattered modules.

`optimization/solvers/cpsat.py`, `optimization/solvers/mip.py`, and
`optimization/solvers/recom.py`

- Apply the shared capacity-recourse predicate when these solvers generate
  analytical seeds. Without this change they can reject zones that are legal
  in `Omega_k` before the new strategy sees them.
- Preserve ordinary zoning behavior when no school-level recourse market is
  attached.

`optimization/config.py`

- Add defaulted `zoned_cg_*` fields and strategy-specific validation.
- Forward every field through `make_strategy()`.
- Keep old YAML and benchmark snapshots loadable.

`optimization/strategies/__init__.py`

- Import the new strategy so registration executes.

`optimization/branch_price/__init__.py`

- Export only stable public analytical contracts.

`optimization/config.example.yaml`

- List the new strategy and refer to its dedicated example.

`optimization/README.md`

- Document Shi objective semantics, isolated-market scope, root closure,
  floating numerical scope, and integer-optimality limitations.

`optimization/solution.py` and `benchmark/runner.py`

- Add an optional, backward-compatible JSON artifact contract so a strategy
  can attach a sparse mechanism witness to a `ZoneSolution` and have both the
  direct CLI and benchmark stage/final save paths persist it.
- Store only artifact filenames and summaries in `solution_<level>.json` and
  `result.json`; do not duplicate the full sparse mechanism in metadata.
- Preserve artifacts or explicit references during benchmark reconstruction
  and metrics-only runs.

`benchmark/config.py`

- If seed paths are list-valued, add the field to
  `SEQUENCE_OPTIMIZATION_FIELDS` so paths are not interpreted as sweep values.

### 11.3 Existing experiments

Keep `run_fixed_state_column_generation` in
`analytical_column_generation.py`. It may become a compatibility wrapper or a
clearly named seed/bound generator, but its existing imports and tests must
continue to work. Its fixed bundle and cutoff values must never be presented
as full all-bundle pricing closure.

## 12. Required public data contracts

The implementation may refine names, but it must provide typed equivalents of
the following contracts.

```python
@dataclass(frozen=True, slots=True)
class ShiMechanismResult:
    primal_objective: float
    dual_objective: float
    repaired_upper_bound: float
    menu_probabilities: dict[int, tuple[tuple[tuple[int, ...], float], ...]]
    quotas: dict[int, float]
    school_prices: dict[int, float]
    type_potentials: dict[int, float]
    max_pricing_violation: float
    closed: bool
    status: str

@dataclass(frozen=True, slots=True)
class AnalyticalZonePattern:
    label: int
    nodes: frozenset[int]
    shi_welfare: float
    perimeter: int
    valuation_status: str

@dataclass(frozen=True, slots=True)
class AnalyticalPricingResult:
    label: int
    candidate: AnalyticalZonePattern | None
    candidate_reduced_cost: float | None
    reduced_cost_upper_bound: float
    menu_residual_bound: float
    closed: bool
    status: str
    branch_nodes: int
    menu_columns: int
    timing_seconds: float

@dataclass(frozen=True, slots=True)
class ZonedColumnGenerationResult:
    patterns: tuple[AnalyticalZonePattern, ...]
    root_lp_objective: float
    root_lp_upper_bound: float
    root_lp_closed: bool
    root_lp_integral: bool
    root_lp_additive_gap: float
    restricted_mip_objective: float
    incumbent_upper_bound_gap: float
    assignment: dict[int, int]
    rounds: int
    pricing_calls: int
    timing_seconds: float
```

Never overload one field to hold both an incumbent and a bound.

## 13. Configuration contract

Add backward-compatible defaults for at least these options:

```yaml
strategy: zoned_column_generation

zoned_cg_wall_time_limit: 2700
zoned_cg_max_rounds: 100
zoned_cg_pricing_time_limit: 300
zoned_cg_pricing_node_limit: 10000
zoned_cg_columns_per_label: 10
zoned_cg_reduced_cost_tolerance: 1.0e-7
zoned_cg_menu_tolerance: 1.0e-9
zoned_cg_master_feasibility_tolerance: 1.0e-8
zoned_cg_optimality_tolerance: 1.0e-6
zoned_cg_mip_time_limit: 300
zoned_cg_seed_paths: []
zoned_cg_recom_seed_runs: 4
zoned_cg_local_move_rounds: 100
zoned_cg_save_mechanism: true
zoned_cg_evaluate_stable_diagnostics: true
```

The total global deadline takes precedence over per-call limits. Validate
positive finite times and tolerances, nonnegative counts, positive pool sizes,
year 23, all-program population, positive `beta`, and isolated schools.

The configured top-level solver is the primary seed generator. Document which
solvers are accepted. The canonical example should use `cp_bool`; ReCom and
saved benchmark outputs supply additional seeds. Gurobi is an internal lazy
dependency of the analytical master and pricing implementation.

## 14. Seed requirements

The initial master must contain at least one complete feasible partition. Use
all configured sources:

1. the current or supplied incumbent;
2. portable benchmark/saved area assignments with graph fingerprints;
3. coarse-to-fine assignments through `LevelConverter`;
4. CP-SAT feasible solutions;
5. ReCom and short-burst solutions after centroid relabeling;
6. support-closed boundary moves;
7. school-node swaps followed by geographic repair.

Every seed must be normalized so zone `k` contains centroid `c_k`, then
validated under the exact pattern domain. ReCom's own feasibility and labels
are insufficient because it intentionally does not enforce all centroid,
candidate, support, or boundary semantics.

Deduplicate patterns globally by `(label,nodes)`. Revalue every unique zone
once and cache the result. Do not require every seed assignment to be valid;
reject invalid seeds with recorded reasons, but require at least one valid
complete partition before solving the restricted master.

## 15. Strategy output

Return one final `ZoneSolution`. Do not return one solution per column-
generation round.

`ZoneSolution.objective` is the selected restricted-MIP zoning's recomputed
normalized Shi welfare. Set status as follows:

- `OPTIMAL` only when the incumbent meets the reported full-master numerical
  upper bound within `zoned_cg_optimality_tolerance`; an integral exactly
  closed root matching the incumbent is one sufficient case;
- `FEASIBLE` when a valid incumbent exists but does not meet the reported
  numerical upper bound within that tolerance, regardless of whether the root
  is integral, fractional, closed, or unclosed;
- no feasible `ZoneSolution` when no complete valid partition exists; raise a
  descriptive error through the normal strategy lifecycle.

Because the analytical computation is not proof-grade, `OPTIMAL` means
solver-reported numerical optimality within the saved additive gap. Keep
`global_optimum_certified: false` for exact-real certification and add a
separate numerical flag.

Required JSON-safe metadata includes:

```text
solver
objective_kind = analytical_shi_expected_mnl_welfare
objective_normalization = outside_and_euler_gamma_constant_removed
optimization_method = complete_zone_nested_column_generation
numerical_scope = FLOATING_ANALYTICAL_NOT_PROOF_GRADE
market_coupling = isolated_zones
shi_normalized_welfare
raw_welfare_constant
raw_expected_welfare
root_lp_status
root_lp_closed
root_lp_objective
root_lp_upper_bound
root_lp_integral
root_lp_additive_gap
root_lp_rounds
max_pricing_upper_bound_reduced_cost
pricing_calls
pricing_status_counts
column_count
seed_pattern_count
restricted_mip_status
restricted_mip_objective
incumbent_upper_bound_gap
global_optimum_certified
global_optimum_scope
numerical_optimum_within_tolerance
q20_welfare
continuum_stable_welfare
seed_provenance
```

If `zoned_cg_save_mechanism` is true, save a separate sparse JSON artifact
containing selected-zone menu probabilities, quotas, and the priority-rule
description from Lemma 9. Label it a continuum/large-market witness with hard
eligibility or a minimum admissibility threshold, not a finite-student
assignment. Do not place the full mechanism in `ZoneSolution` metadata if
doing so would materially inflate `result.json`.

## 16. Unit and proof-verification tests

### 16.1 Mathematical proof review

Before implementation acceptance, an independent subagent must review Section
9 and produce findings against the following checklist:

1. Does school-capacity activation alone exclude every menu containing an
   unselected eligible school?
2. Are zero-capacity schools and outside-only segments handled without
   invalidating Lemma 3?
3. Does menu reduced cost exactly match Shi's prefix subproblem, with no
   omitted geographic linking dual?
4. Does the geographic branch exhaust all and only `Omega_k`?
5. Is every claimed upper bound valid under interruption?
6. Is the outer perimeter coefficient exactly doubled once and only once?
7. Does outer LP closure imply only the claims stated in Lemmas 7 and 8?
8. Does the priority construction implement the sparse primal under the
   hard-eligibility/minimum-threshold assumptions actually provided by the MNL
   market?
9. Are all isolated-market assumptions explicit?
10. Are floating numerical claims separated from exact-real mathematical
    claims?
11. Does positive menu tolerance use the residual repair (8b) rather than the
    restricted primal objective as a bound?
12. Does the branch queue preserve one valid bound for every unexplored
    subtree at every interruption point?

Any material proof finding must be resolved in this document and in tests.

### 16.2 Required unit tests

Add data-free tests for all of the following.

1. Compare `solve_shi_menu_bound` with an explicitly enumerated primal on tiny
   markets, including nonunit masses and nonzero outside utilities.
2. Compare a closed `JP(k)` leaf with a separately solved local `SP(Q)` for the
   same fixed zone.
3. Enumerate every legal zone on a tiny graph and verify that analytical
   pricing returns the maximum reduced-cost pattern.
4. Construct an improving pattern that moves a noncentroid school from its
   initial zone, proving all-bundle rather than fixed-bundle pricing.
5. Enumerate every complete partition on a tiny graph and compare the extensive
   master with the closed outer column-generation LP.
6. Include an instance requiring more than one outer pricing round.
7. Compare all fallback pricing upper bounds with exhaustive pricing optima.
8. Force an inner timeout and verify that closure remains false while the
   reported upper bound remains above the exhaustive optimum.
9. Verify enabled and disabled FRL, racial, boundary, and graph-capacity rules.
10. Reject wrong-centroid, disconnected, unsupported, candidate-invalid,
    demographic-invalid, school-count-invalid, and over-perimeter patterns.
11. Test school co-location, zero capacity, outside-only segments, repeated
    segment nodes, and segments with different positive masses.
12. Verify beta and outside-option normalization and the separate raw constant.
13. Verify centroid relabeling for ReCom seeds and coarse-to-fine expansion.
14. Test raw-node, area-assignment, and benchmark-directory seed loading,
    including graph-fingerprint mismatch and missing nodes.
15. Verify `OPTIMAL` versus `FEASIBLE` status rules for integral, fractional,
    and unclosed roots.
16. Round-trip `ZoneSolution.save`, benchmark reconstruction, metadata, and the
    sparse mechanism artifact.
17. Run the existing finite-grid branch-price, analytical-bound, analytical
    welfare-oracle, strategy, conversion, ReCom, and aggregation tests to prove
    backward compatibility.
18. Stop menu generation at positive tolerance and verify (8b) against the
    exhaustive full-menu node LP.
19. Interrupt pricing while a node is popped, before child solves and before a
    fallback MIP finishes; verify parent-bound inheritance and global coverage.
20. Time out the restricted integer master before it finds a solution and
    verify that the validated seed is returned unchanged.
21. Attach an analytical recourse market and verify CP-SAT, Gurobi MIP, ReCom,
    geography pricing, and pattern validation all skip only graph aggregate
    capacity while preserving every other enabled balance row.
22. Test the continuum priority witness with hard eligibility, an outside-only
    segment, and an effective eligible-set expansion; verify the artifact never
    claims finite-student exactness.
23. Test a fractional root whose incumbent meets the full-master bound and an
    apparently integral tolerance-closed root with a positive residual gap.
24. Verify CP sum-of-rounded balance semantics exactly match analytical
    Gurobi pricing and exhaustive pattern validation near a rounding boundary.
25. Compare repaired outer bound (11) with an exhaustive full-master LP under
    both exact and time-limited label bounds.
26. Reach an integral geographic node with positive menu residual and verify
    its repaired bound remains in the terminal ledger until separation closes
    or an explicit replacement bound is installed. After tolerance fathoming,
    verify the final reported pricing bound still exceeds the exhaustive
    optimum.

Use existing fixtures from:

- `optimization/tests/synthetic.py`;
- `optimization/tests/test_analytical_bounds.py`;
- `optimization/tests/test_analytical_welfare_oracle.py`;
- `optimization/tests/test_welfare_branch_price.py`;
- `optimization/tests/test_conversion.py`;
- `optimization/tests/test_recom.py`;
- `benchmark/test_aggregation.py`.

## 17. Performance plan

The canonical 45-minute `Block_2` run must use a single global deadline and
record time by phase. Recommended allocation is:

| Phase | Soft allocation |
|---|---:|
| Market build and seed validation | 10% |
| Outer LP and fast additive pricing bounds | 15% |
| Exact nested label pricing | 55% |
| Restricted integer master | 10% |
| Final revaluation, diagnostics, and save | 10% |

Before committing to the full production driver, implement a one-label
prototype and record: root menu-generation rounds, generated menu count,
geographic LP gap, fallback-bound gap, branch count, peak memory, and time.
The benchmark artifact must record CPU model, core count, memory, Gurobi
version, OR-Tools version, and worker settings. The 45-minute closure target is
an empirical acceptance gate on that documented hardware, not a consequence of
the correctness proof; the underlying pricing problem remains NP-hard.

Required performance techniques:

- aggregate segment computations by node when utilities and eligibility allow
  exact aggregation;
- precompute attractions and menu prefix data once;
- share discovered type-menu columns across labels and branch nodes;
- cache local Shi valuations by `(market fingerprint,label,nodes)`;
- price labels in parallel within the global worker budget;
- add multiple columns per label and use Gurobi solution pools only for
  incumbents, never bounds;
- warm-start master bases and pricing branch queues where APIs permit;
- use capacity-price geographic bounds before exact nested pricing;
- prioritize unresolved labels by upper reduced cost;
- reuse coarse and neighboring patterns as branch incumbents;
- record peak menu-column count and branch queue size.

The implementation performance milestone is:

1. `root_lp_closed == true` on the canonical `Block_2` instance within 2,700
   seconds on the documented workstation/HPC hardware;
2. a valid restricted integer zoning is always saved if a seed was valid;
3. timeout runs report a repaired root upper bound and never misreport
   closure;
4. the final assignment and all selected local Shi values recompute within
   configured tolerances.

If the first implementation cannot meet milestone 1, it is functionally
correct but does not satisfy the performance definition of done. Preserve its
valid incumbent and bounds, profile the unresolved labels, and tighten pricing
bounds rather than weakening closure semantics.

## 18. Implementation sequence

Implement in this order.

### Phase A: local Shi solver

1. Refactor attraction preparation into a shared analytical MNL helper.
2. Extend `analytical_bounds.py` to return closed primal and dual witnesses.
3. Add extensive tiny-primal equivalence tests.
4. Add sparse mechanism and quota reconstruction.

### Phase B: analytical patterns and master

1. Extract objective-independent pattern validation.
2. Add `AnalyticalZonePattern`.
3. Add floating LP and restricted integer masters.
4. Port dual-sign, perimeter, reconstruction, and integrality tests from the
   finite-grid master.

### Phase C: exact zoned pricing

1. Implement fixed-geography `JP(k)` and prove equality with local Shi.
2. Add dynamic menu separation.
3. Add geographic LP relaxation and `x` branching.
4. Add access and capacity-price upper bounds.
5. Validate against exhaustive tiny graphs and all school bundles.

### Phase D: outer closure

1. Implement the repeated all-label master/pricing loop.
2. Add repaired floating upper bounds.
3. Add multiple-column insertion and global deadline behavior.
4. Add integral/fractional/unclosed status tests.

### Phase E: seeds and strategy

1. Add portable seed loading and level conversion.
2. Add validated solver, ReCom, boundary-move, and school-swap seeds.
3. Register the strategy and configuration.
4. Produce `ZoneSolution` and sparse mechanism artifacts.

### Phase F: benchmark and documentation

1. Add a deterministic tiny smoke benchmark.
2. Add the canonical 45-minute `Block_2` benchmark.
3. Add serialization and aggregation tests.
4. Update README and example configs.

Do not proceed to a later phase while an exhaustive equivalence test from the
current phase is failing.

## 19. Definition of done

The assignment is complete only when all of the following are true.

- `strategy: zoned_column_generation` runs through both the optimization CLI
  and benchmark runner.
- The strategy builds and validates an isolated analytical MNL market.
- At least one valid complete partition seeds the restricted master.
- Every analytical pattern is structurally valid and valued by a closed local
  Shi primal.
- Label pricing permits every legal school bundle and resident set.
- Inner menu separation is exactly Shi's MNL prefix problem.
- Outer root closure uses global pricing upper bounds for every label.
- A time-limited run keeps a valid incumbent and reports closure false when
  any label remains unresolved.
- The restricted integer assignment is complete, centroid-valid,
  candidate-valid, support-connected, balanced, school-count-valid, and
  perimeter-valid.
- `ZoneSolution.objective` is recomputed normalized Shi welfare.
- Q20 and continuum stable welfare are clearly labeled diagnostics.
- A sparse budget-set, quota, and priority-rule witness is saved for every
  selected zone.
- `OPTIMAL` is used only when the incumbent meets the reported numerical upper
  bound within the configured tolerance, with exact-real certification kept
  false unless a future proof-grade mode supplies it.
- Exhaustive tiny tests match the full menu and full zone universes.
- Existing finite-grid and analytical experimental tests continue to pass.
- The independent proof review has no unresolved material findings, including
  residual repairs, hard eligibility, timeout queue invariants, and numerical
  status semantics.
- The canonical `Block_2` root LP closes within the 45-minute performance
  target.
