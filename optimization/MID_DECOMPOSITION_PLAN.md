# MID Continuum Utility Decomposition Plan

## Decision

Optimize expected assignment utility directly in the large-market continuum
model. Do not minimize cutoffs, either across zonings or as the fixed-zoning
subproblem. Utility maximization provides the same cutoff-selection pressure as
the former monolithic MID formulation.

The current CP-SAT model with lottery scale `L` optimizes a finite-grid
approximation. Increasing `L` can improve that approximation, but no finite `L`
certifies the continuum optimum. The exact structural continuum model has
continuous cutoff and mass variables plus binary zoning and piecewise-linear
selectors, so the generated master must be a MIP.

The decomposition below is exact for the represented continuum market. It
terminates at a globally optimal zoning when generated masters are solved to
global optimality and no hard time or iteration limit interrupts the proof.

## Continuum Market

Let:

- `x in Omega` be a feasible zoning.
- `S` be the programs, with capacity `q[s]`.
- `G` be the compressed applicant types.
- `n[g]` be the number of applicants represented by type `g`.
- `s[g,r]` and `rho[g,r]` be its program and priority tier at rank `r`.
- `H[g,r]` be the sum of represented applicants' cardinal utilities at rank
  `r`.
- `a[g,r](x)` indicate whether program `s[g,r]` is accessible under `x`.

Every member of a compressed type has the same ordered programs, priorities,
and access-relevant node, so all members have the same assignment mass at each
rank. The market builder orders alternatives by utility and uses a zero-utility
outside option. The required objective-order condition is therefore:

```text
H[g,0] >= H[g,1] >= ... > 0.
```

This condition, not positivity alone, makes lower cutoffs welfare-improving.
It must be validated after market construction and preprocessing.

Normalize each applicant's lottery mass to one. For program cutoff `p[s]`,
define the clipped rejected lottery mass for priority tier `rho`:

```text
t[s,rho] = min(1, max(p[s] - rho, 0)).
```

For a ranked alternative, its effective rejected mass is:

```text
e[g,r] = t[s[g,r],rho[g,r]]  if a[g,r](x) = 1
e[g,r] = 1                   if a[g,r](x) = 0.
```

Assignment follows the exact remaining-mass recurrence:

```text
R[g,-1] = 1
R[g,r]  = min(R[g,r-1], e[g,r])
d[g,r]  = R[g,r-1] - R[g,r].
```

Program demand and continuum welfare are:

```text
D[s](x,p) = sum(n[g] * d[g,r] for (g,r) with s[g,r] = s)
W(x,p)    = sum(H[g,r] * d[g,r] for g,r).
```

The target problem is:

```text
W* = max W(x,p)
     subject to x in Omega
                D[s](x,p) <= q[s] for every program s
                the exact cutoff recurrence above.
```

## Why Utility Maximization Is Correct

For fixed zoning `x`, let:

```text
C(x) = {p : D[s](x,p) <= q[s] for every s}.
```

Demand at a program is nonincreasing in its own cutoff and nondecreasing in
every other program's cutoff. Define the least own cutoff that clears program
`s` against the other cutoffs:

```text
B[s](p[-s]) = min {z : D[s](x, (z, p[-s])) <= q[s]}.
```

The reject-all cutoff bounds make this set nonempty. Since `B` is isotone, its
least fixed point `p*(x)` is the coordinatewise least capacity-feasible cutoff
vector. Equivalently, start at zero and repeatedly raise an overloaded
coordinate to its least clearing value. If `p^k <= p` for any feasible `p`,
then:

```text
B[s](p^k[-s]) <= B[s](p[-s]) <= p[s],
```

so the iteration remains below every feasible vector and converges to `p*(x)`.
On the finite grid this terminates finitely; in the continuum the monotone limit
gives the least fixed point under the continuous piecewise-linear demand map.

Lowering any program cutoff can only move lottery mass from a lower-ranked
program or the outside option to that program. Under the objective-order
condition:

```text
p*(x) <= p  implies  W(x,p*(x)) >= W(x,p).
```

Therefore:

```text
max(x in Omega, p in C(x)) W(x,p)
    = max(x in Omega) W(x,p*(x)).
```

This is the objective-pressure argument used by the former MID approach. The
joint optimizer need not return the literal least cutoff vector when several
cutoffs produce the same welfare, but it returns least-cutoff welfare and an
optimal zoning. A post-solve monotone oracle may report canonical least cutoffs;
cutoff minimization is not needed in the optimization path.

Objective pressure selects welfare-optimal cutoffs, but it does not replace the
exact `min` recurrence. Relaxing that equality to one-sided inequalities would
let the model claim assignment mass for applicants who do not clear a cutoff.

## Exact Continuum MIP

Use the existing Gurobi zoning formulation for `x in Omega` and its exact
same-zone access indicators. Replace integer lottery-grid quantities with the
following continuous variables.

### Cutoffs And Thresholds

For every program:

```text
0 <= p[s] <= max_observed_priority[s] + 1.
```

For every observed `(s, rho)`, impose exactly:

```text
t[s,rho] = min(1, max(p[s] - rho, 0)).
```

This is a three-segment piecewise-linear equality. Encode it with Gurobi general
`min` and `max` constraints or equivalent segment binaries. It must not be
replaced by only the convex side of the equality.

For citywide programs, set `e[g,r] = t[s,rho]`. For restricted programs, use
the access indicator:

```text
a[g,r] = 1  implies e[g,r] = t[s,rho]
a[g,r] = 0  implies e[g,r] = 1.
```

### Exact Remaining Mass

For each represented recurrence, keep `R[g,r]` in `[0,1]`. The equality
`R = min(P,E)` can be represented with selector `b[g,r]` and unit big-M:

```text
R <= P
R <= E
R >= P - b
R >= E - (1 - b)
b in {0,1}.
```

Here `P = R[g,r-1]` and `E = e[g,r]`. These four rows impose the exact minimum,
including ties.

### Capacity And Objective

Add globally coupled program rows:

```text
sum(n[g] * (R[g,r-1] - R[g,r]) for (g,r): s[g,r] = s) <= q[s].
```

Maximize literal continuum utility using unrounded `H[g,r]`:

```text
maximize sum(H[g,r] * (R[g,r-1] - R[g,r]) for g,r).
```

Using fixed-point utility coefficients would optimize a different, rounded
continuum objective. CP-SAT scaling and `mid_lottery_scale` do not belong in the
continuum objective or its optimality bounds.

## Generated Master

Let `A` be the activated compressed types. The master includes the exact
continuum recurrence and demand only for `g in A`. For each inactive type use
the valid optimistic bound:

```text
U_bar[g] = H[g,0]
```

and use zero for a type with no alternatives. The generated master is:

```text
maximize
    sum(H[g,r] * d[g,r] for g in A and all r)
    + sum(U_bar[g] for g not in A)

subject to
    x in Omega
    shared continuous cutoffs, thresholds, and access constraints
    exact recurrence for g in A
    sum(active demand at s) <= q[s] for every s.
```

This is a valid maximization relaxation:

- Removing inactive demand relaxes every capacity row.
- `sum(r, H[g,r] * d[g,r]) <= H[g,0]` because assigned mass is at most one and
  utility is nonincreasing down the preference list.
- Every complete feasible solution projects to a feasible generated-master
  solution with weakly larger master objective.

Consequently:

```text
OPT(full continuum MID) <= OPT(master(A)).
```

Activating every type reproduces the full exact continuum MIP.

## Decomposition Algorithm

Maintain:

```text
LB = utility of the best complete continuum-feasible solution
UB = smallest certified generated-master upper bound
A  = activated types.
```

### Initialization

1. Build the finest-level `ZoneProblem` and retain all intended zoning
   constraints except aggregate overage and shortage.
2. Build and preprocess the MID market once.
3. Validate rank-utility monotonicity for every compressed type.
4. Build a feasible zoning hint when configured.
5. For the hinted zoning, solve the fixed-zoning continuum welfare subproblem by
   maximizing utility with all types and exact recurrences. Use it as the first
   incumbent and `LB`.
6. Start with no activated types or a deterministic seed set.

The fixed-zoning subproblem uses utility objective pressure. It does not
minimize cutoff magnitudes. The current monotone continuum oracle may provide a
warm start, but its tolerance-based result is not the source of a global proof.

### Master Iteration

1. Build the exact continuum master for the current `A`.
2. Warm-start zoning, cutoffs, access, and active recurrence variables from the
   best incumbent.
3. Solve the master to global optimality in exact mode.
4. Update `UB` only from the globally optimal master value or the solver's
   certified upper bound.
5. Evaluate the candidate zoning and cutoffs with the complete independent
   continuum recurrence.
6. Solve the complete fixed-zoning continuum welfare subproblem for the
   candidate zoning and update the incumbent and `LB` if it improves.

For a maximization problem, a feasible master candidate objective is not an
upper bound. It can be below both the master optimum and the full MID optimum.
Only the proven master optimum or a valid solver objective bound may update
`UB`.

### Exact Separation

At master candidate `(x_bar, p_bar)`:

1. Compute complete demand and actual utility contributions for every type.
2. If a program is overloaded, activate every inactive type with positive
   demand at that program.
3. If complete demand is feasible, activate every inactive type satisfying:

```text
U_bar[g] > actual_utility[g](x_bar, p_bar).
```

The active capacity row already satisfies capacity. Restoring all omitted
demand at an overloaded program therefore makes the current candidate
infeasible. Utility-gap activation does not remove the candidate from the
feasible region; it removes optimistic objective value at that point. Both are
valid refinements.

### Termination And Certificate

Stop with a global-optimality certificate only when one of these holds:

```text
UB <= LB                                      in common objective units
no separator exists at a globally optimal master candidate
all types are active and the full master is globally solved.
```

For the no-separator case, complete demand is feasible and every inactive type
attains its upper bound. Hence the candidate has equal master and full utility:

```text
W_full(candidate) = W_master(candidate)
                  = OPT(master)
                  >= OPT(full)
                  >= W_full(candidate),
```

so all inequalities are equalities and the zoning is globally optimal.

Whenever separation is required, at least one inactive type is activated.
There are at most `|G|` activation rounds and at most `|G| + 1` master solves,
including the solve after the final activation. Thus the method finitely reduces
to the full exact continuum MIP in the worst case.

A positive absolute or relative gap proves only tolerance-optimality. A time or
iteration limit proves nothing beyond the reported `LB` and `UB`; return the
best incumbent with status `FEASIBLE`, not `OPTIMAL`. If `max_iterations` is
used only for generation, reaching it should activate all remaining types and
start the final full solve rather than claim convergence.

## Large-Market Interpretation

The continuum model is the result target. Replicate every type count and every
program capacity by market size `K`, draw independent continuous uniform
lotteries, and normalize demand and welfare by `K`. Under the standard no-atom
lottery and cutoff-regularity assumptions, realized DA demand and welfare
converge to the deterministic recurrence above as `K` grows.

The finite-grid CP-SAT model is not this limit. For example, three identical
applicants competing for one seat require continuum cutoff `p = 2/3` and fill
the seat. With `L = 20`, the least integer cutoff is `14`, assignment is `0.9`,
and capacity is left unused. A grid model can therefore rank zonings
differently from the continuum model.

Keep finite-grid results only as optional warm starts or diagnostics. Do not use
their objective or bound to certify the continuum solve.

## Numerical Meaning Of Optimality

The mathematical guarantee assumes exact rational coefficients and exact MIP
solves. Gurobi represents the loaded utility coefficients in floating point and
uses feasibility and integrality tolerances. Production metadata must therefore
distinguish:

- Mathematical global optimality of the represented formulation.
- Solver-reported optimality within configured numerical tolerances.
- Tolerance-optimal or time-limited feasible results.

Use the same raw continuum utility coefficients for master objectives,
fixed-zoning incumbents, bounds, and gap calculations. Never compare a
finite-grid fixed-point value with a continuum MIP bound.

## Implementation Changes

### `optimization/solvers/mid.py`

1. Replace the CP-SAT MID market model with a Gurobi continuum MID builder, or
   add a dedicated `MidMipSolver` and remove the finite-grid solver from the
   `mid` strategy path.
2. Reuse `add_gurobi_zoning_geography()` for the canonical zoning constraints.
3. Accept an activated type-index set.
4. Create shared continuous cutoffs and clipped priority thresholds.
5. Create access indicators from the complete preprocessed market.
6. Add exact continuous recurrences and capacity terms only for active types.
7. Add inactive optimistic utility as an objective constant.
8. Expose candidate cutoffs, candidate objective, `ObjBound`, status, and model
   sizes without converting to fixed-point units.
9. Restrict hints to variables present in the generated master.

### `optimization/mid_oracle.py`

1. Keep independent recurrence evaluation for separation.
2. Expose per-type continuum demand and utility contributions.
3. Keep the existing monotone continuum routine as a warm start and diagnostic,
   not an exact certificate.
4. Add a fixed-zoning exact continuum welfare solve that maximizes utility with
   all types and recurrences. Do not minimize cutoff sums.
5. Preserve the finite-grid oracle only for explicit approximation diagnostics.

### `optimization/strategies/mid.py`

1. Require the MIP solver path rather than `cp_bool` for continuum MID.
2. Own the activated set, incumbent, global bounds, and iteration metadata.
3. Generate and globally solve continuum masters.
4. Run complete separation and the fixed-zoning welfare subproblem after each
   candidate.
5. Return the best complete continuum-feasible zoning.
6. Report `OPTIMAL` only with one of the certificates above.
7. On a hard limit, report `FEASIBLE` with the residual certified gap.

### `optimization/data/mid.py`

1. Validate that raw utility sums are nonincreasing down every compressed type's
   preference list after preprocessing.
2. Retain unrounded utility sums as the continuum objective coefficients.
3. Continue requiring identical node, ordered programs, and priorities for type
   compression.

## Result Metadata

Record at least:

- Formulation name identifying continuum generated MID utility.
- Master and fixed-zoning solver statuses.
- Iteration count and activated/total type counts.
- Overload and utility-gap activations.
- Candidate objective and certified master upper bound by iteration.
- Continuum `LB`, `UB`, absolute gap, and relative gap.
- Best incumbent iteration and termination reason.
- Candidate cutoffs and post-solve diagnostic least cutoffs.
- Total master, separation, and fixed-zoning solve time.
- Exact, tolerance-optimal, or unproven status.
- Separate optional finite-grid diagnostic values.

Set `ZoneSolution.objective` to the complete continuum utility used for
incumbent comparison. Do not overwrite it with finite-grid welfare.

## Tests

Add focused synthetic coverage for:

1. Exhaustive tiny-zoning equivalence among the decomposition, full continuum
   MIP, and enumerated fixed-zoning continuum optima.
2. A market where cutoff-sum minimization ranks zonings differently from
   continuum utility maximization.
3. The three-applicant, one-seat example with cutoff `2/3`, continuum welfare
   `1`, and a different finite-grid value.
4. Fixed-zoning utility maximization attaining least-cutoff welfare without a
   cutoff objective.
5. Rejection of a type whose utility coefficients increase down its preference
   list.
6. Overload separation cutting off the current candidate.
7. Utility-gap separation lowering optimistic objective value.
8. Every certified master bound remaining above the exhaustive continuum
   optimum.
9. Every complete incumbent remaining below or equal to that optimum.
10. No-separation at an optimal master certifying the exhaustive optimum.
11. Full activation reproducing the direct continuum MIP.
12. Restricted and citywide programs sharing global capacity rows.
13. Same-node access, impossible overlap, and variable same-zone access.
14. Time-limit exhaustion returning `FEASIBLE` with a residual gap.
15. Continuum and finite-grid objectives never being mixed in bounds.

## Verification

Run:

```bash
uv run python -m pytest optimization/tests/test_mid.py optimization/tests/test_strategies.py
uv run python -m pytest optimization/tests
```

For real-data validation, compare generated and full continuum MIPs on the same
configuration and record activated type counts, iterations, model sizes, wall
time, final continuum utility, certified gap, and whether full activation was
required.
