# MID Implementation Plan

## Objective

Add an optimization strategy named `mid` that directly maximizes expected
program-assignment welfare under a large-market DA continuum approximation.
The implementation uses a finite integer mass grid but does not sample lottery
numbers or create lottery-cell variables.

## Agreed Semantics

- Choice alternatives are programs. Program capacities are enforced separately.
- Zoning access remains school-based: every program at a non-citywide school is
  accessible only when the student's geographic node and the school's node are
  assigned to the same zone.
- Programs at citywide schools remain globally accessible.
- The welfare market uses the assignment-selected cohort. For the current
  `summer-26-zoning` scenario, this is the 2023-24 KG applicant market selected
  by the assignment filters.
- The zoning graph may continue to use a different or multi-year student
  population for geographic and demographic constraints.
- Status-quo base policy priorities supply the integer priority tiers. Include
  the CTIP, sibling, current attendance-area/zone, language-program, and other
  base status-quo components.
- Do not add a sampled lottery, round priority, listed/non-designation boost,
  reserves, or guardrails to the priority tiers.
- The finite mass grid analytically represents the continuum limit. There is no
  explicit lottery-cell variable `ell` in the CP-SAT model.
- `mid_lottery_scale` is configurable and defaults to `20`.
- Aggregate zoning overage and shortage constraints are disabled for MID.
  Program-level market capacity constraints remain active.
- All other constraints represented by `x in Omega` remain active, including
  candidate zones, centroid anchors, monotone contiguity, demographic balance,
  school-count balance, maximum distance, and an optional boundary limit.
- MID performs one direct solve at the finest configured level, `levels[-1]`.
- MID requires the `cp_bool` solver and `program_population: All` so the zoning
  constraints continue to use school-level/all-program graph values.

## Utility Handling

Add `mid_utility_handling` with two supported values:

- `omit_nonpositive`, the default: omit intrinsically eligible alternatives
  whose systematic choice-model utility is not strictly positive. Retained
  cardinal utility is the original systematic utility.
- `exponentiate`: include every intrinsically eligible alternative with finite
  systematic utility and transform it per student as
  `exp(V_is - max_s V_is)`. This preserves the student's ordering, keeps the
  largest utility at one, and makes every included utility strictly positive.

Negative infinity continues to denote intrinsic ineligibility and is never
included. Preferences are ordered by descending systematic utility, with
program identity as the deterministic tie-break for exact utility ties.

Use the repository's current CP-SAT scale, `100`, rather than adding another
utility-scale option. Fixed-point utility is:

```text
scaled_utility = max(1, round(100 * utility))
```

The lower bound of one preserves strict positivity after rounding and therefore
preserves the least-cutoff welfare argument.

Programs selected by the assignment table but absent from the choice-model
matrix are not market alternatives. Assignment-selected students absent from
the choice-model matrix are represented as outside-option-only students and are
counted in result metadata.

## Market Data And Type Compression

Add `optimization/data/mid.py` containing the program-market and compressed-type
data structures plus the market builder.

Load all inputs through the current `DataScenario`:

- Students from `assignment.students` using assignment filters.
- Programs and capacities from `assignment.programs` using the selected year,
  grade, capacity profile, and capacity scenario.
- Schools and citywide classification from `assignment.schools`.
- Systematic utilities from `choice.estimate`.
- Status-quo base priorities from the existing assignment priority machinery.

Read the choice matrix directly by student and program identity so the MID
market can use the intersection of selected programs and programs estimated by
the choice model. Do not draw utility shocks.

Map each student to the optimization graph node containing the student's Census
geography. Map every non-citywide program's school to its graph node. Citywide
programs do not require a school node for access because they are unrestricted.
Reject ambiguous identities, duplicate programs, non-integral priorities, and
non-integral or negative capacities.

Compress students using the key:

```text
(graph node, ordered program list, aligned priority-tier list)
```

For each type, retain:

- The number of represented students.
- The ordered programs and aligned priority tiers.
- The sum of original transformed utilities at every rank.
- The sum of fixed-point utility coefficients at every rank.

Utilities need not be part of the type key. Students with the same location,
preferences, and priority tiers have identical assignment masses, so their
utility coefficients can be summed rank by rank. This preserves the exact
individual objective while reducing recurrence variables.

## Independent Cutoff Oracle

Add `optimization/mid_oracle.py` with finite-grid and continuous least-cutoff
solvers. Both should operate directly on compressed types.

For a fixed zoning, first apply access:

- Retain every citywide program in every type's preference list.
- Retain a restricted program only when the type's node and program school's
  node have the same assigned zone.

The finite-grid oracle iteratively raises each program cutoff to the smallest
integer value that clears its capacity given the other cutoffs. It returns:

- Least cutoffs.
- Program demands and assignment masses.
- Outside-option mass.
- Finite-grid welfare using unrounded transformed utility.
- Fixed-point welfare using scaled utility.
- A grid-minimality certificate obtained by lowering each positive cutoff by
  one unit and checking that capacity is violated.

The continuum oracle uses the analogous piecewise-linear demand inversion with
mass normalized to one per student. It returns continuous cutoffs, demands,
welfare, outside-option mass, and market-clearing stability checks.

Citywide and restricted programs remain in one coupled market so citywide
capacity is imposed exactly once across all zones.

## CP-SAT Formulation

Add `optimization/solvers/mid.py` as a MID-specific wrapper around the existing
`cp_bool` zoning solver. Reuse its assignment variables, core zoning
constraints, search strategy, hints, logging, progress tracking, solver
parameter configuration, and assignment extraction.

### Shared Access Indicators

Create one same-zone Boolean for each required `(student node, school node)`
pair, not for each student-program pair. All programs at the school reuse it.

If the nodes are identical, fix the indicator to one. Otherwise, reify it from
the sparse zoning assignment variables. Candidate-disallowed zone assignments
must be treated as zero.

Citywide programs do not need access indicators.

### Cutoffs And Thresholds

For each ranked program `s`, create cutoff `P_s`. A sufficient program-specific
domain is:

```text
0 <= P_s <= (max_priority_s + 1) * L
```

Share one rejection threshold for each observed `(program, priority tier)`:

```text
T_rho_s = max(P_s - rho * L, 0)
```

The threshold domain must be large enough to exceed `L`. Do not clip the
threshold to `L`; a cutoff may need to reject all mass from several lower
priority tiers before partially admitting a higher tier.

Share effective thresholds by `(student node, program, priority tier)`:

```text
effective = T_rho_s        when accessible
effective = L              when inaccessible
```

For citywide programs, the effective threshold is the shared rejection
threshold directly.

### Remaining And Assignment Mass

For each compressed type, set:

```text
remaining_0 = L
remaining_r = min(remaining_(r-1), effective_r)
assignment_mass_r = remaining_(r-1) - remaining_r
```

Every remaining-mass variable has domain `[0, L]`. The minimum with the previous
remaining mass performs the required implicit clipping even when the rejection
threshold exceeds `L`.

Do not create explicit `ell` variables. Do not create a separate clipping
constraint. Assignment mass can remain a linear expression rather than an
additional variable.

### Program Capacities

For each program, add:

```text
sum(type_count * assignment_mass) <= L * program_capacity
```

These rows replace only aggregate zoning overage/shortage constraints. They do
not replace demographic, geography, or school-count constraints.

### Welfare Objective

For every type and rank, multiply assignment mass by the sum of member
fixed-point utilities at that rank:

```text
maximize sum(type_scaled_utility_sum_r * assignment_mass_r)
```

The normalized fixed-point objective is the raw CP-SAT objective divided by
`L * 100`.

Check the objective bound before solving so all integer arithmetic stays inside
CP-SAT limits and exact objective reporting remains reliable.

## Strategy Integration

Add `optimization/strategies/mid.py` and register it as `mid`.

The strategy should:

1. Parse the configured levels and select `levels[-1]`.
2. Apply the final configured solve-time and gap limit to the solver.
3. Build the finest-level `ZoneProblem`.
4. Set `problem.overage = -1` and `problem.shortage = -1`.
5. Apply the configured `boundary_prop` explicitly.
6. Build an existing Voronoi hint when hints are enabled.
7. Build and compress the MID market for this graph level.
8. Use the least-cutoff oracle to seed cutoff and recurrence variables for the
   initial zoning hint when one is available.
9. Run the joint CP-SAT model.
10. Independently reevaluate the returned zoning with both cutoff oracles.
11. Return one `ZoneSolution`, making it the literal final benchmark result.

Add `mid` to the strict strategy allowlist in `optimization/config.py`, import
its module in `optimization/strategies/__init__.py`, and forward the MID options
from `OptimizationConfig.make_strategy()`.

Expose the current CP-SAT coefficient scale as a shared constant instead of
duplicating the value in the MID implementation.

## Result Reporting

Set `ZoneSolution.objective` to independently recomputed finite-grid welfare
using unrounded transformed utilities:

```text
finite_grid_welfare =
    sum(original_transformed_utility * assignment_mass) / L
```

Store at least the following metadata:

- `objective_kind` and formulation name.
- Lottery mass scale and fixed-point utility scale.
- Utility handling mode.
- Assignment student count and outside-only student count.
- Program count, restricted program count, and citywide program count.
- Compressed type count and compression ratio.
- Preference incidence count.
- Access-indicator, threshold, effective-threshold, and remaining-variable counts.
- Raw solver objective and normalized fixed-point objective.
- Finite-grid oracle welfare, cutoffs, demands, and outside mass.
- Grid-minimality status.
- Continuum welfare, cutoffs, demands, outside mass, and stability status.
- Solver cutoff agreement with the independent finite-grid oracle.
- Model statistics and preprocessing/oracle timing.
- Explicit metadata that aggregate overage and shortage were disabled.

Use stable program IDs in metadata even if contiguous integer program numbers
are used internally for smaller CP-SAT expressions and variable names.

## Tests

Add focused synthetic tests in `optimization/tests/test_mid.py` covering:

- Default omission of nonpositive systematic utilities.
- Per-student exponentiated normalization.
- Fixed-point coefficients never rounding below one.
- No Gumbel or priority-lottery draw.
- Status-quo base-priority extraction and validation.
- Program-level capacities and school-level access.
- Type compression preserving expanded-student demand and welfare.
- Shared threshold counts by `(program, priority tier)`.
- Shared access by `(student node, school node)`.
- A cutoff whose rejection threshold exceeds `L`, demonstrating that the
  remaining-mass minimum handles clipping without a clip variable.
- Partial assignment mass at multiple programs.
- Restricted access and globally coupled citywide capacity.
- Finite-grid least-cutoff and grid-minimality certificates.
- Continuous cutoff and stability evaluation.
- Direct CP-SAT welfare matching exhaustive feasible zonings on a tiny graph.
- Aggregate overage/shortage disabled while other `Omega` constraints remain.
- Correct normalized solver objective and independently reported welfare.

Update `optimization/tests/test_strategies.py` for registration, configuration
validation, finest-level selection, final time/gap limits, boundary propagation,
and solver compatibility.

Add one `real_data` smoke test that builds the MID market from
`summer-26-zoning`, verifies student/program alignment, and confirms that type
compression and utility metadata are internally consistent.

## Verification Commands

```bash
uv run python -m pytest optimization/tests/test_mid.py optimization/tests/test_strategies.py
uv run python -m pytest optimization/tests
```

Do not modify existing benchmark sweeps as part of the implementation. A sweep
can opt into the strategy by setting `strategy: mid`, retaining `solver:
cp_bool`, and optionally configuring `mid_lottery_scale` and
`mid_utility_handling`.
