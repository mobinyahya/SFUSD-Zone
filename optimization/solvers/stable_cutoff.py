"""Joint zoning and per-student cutoff model of sampled deferred acceptance.

`MidCpSatSolver` extends the Boolean zoning model with the finite-grid lottery
recurrence; this solver extends the *same* zoning model -- the Gurobi one, so
the linear relaxation is readable and the co-zoning linearization is exact
rather than a CP-SAT multiplication -- with the individual matching instead. For
each sampled strict priority order it writes the student-optimal stable matching
as a mixed-integer program in per-student assignment variables ``y[i, s]`` and
per-student cutoff-clearing indicators ``z[i, s]``, and maximizes the average of
those matchings' welfare over the samples. Every block shares one copy of ``x``,
so the samples argue about one zoning.

The formulation, for one sample, with ``A_is`` equal to 1 for a citywide program
and the co-zoning indicator of ``i``'s node with ``s``'s school node otherwise,
and ``Gamma(s)`` that sample's strict priority order for ``s``:

    (F1)  sum_s y[i, s] <= 1                                  one seat per student
    (F2)  sum_i y[i, s] <= q_s                                capacity
    (F3)  y[i, s] <= A_is                                      zoned access
    (F4)  y[i, s] <= z[i, s]                                   only admits are seated
    (F5)  z[i, s] <= z[i', s]  for consecutive (i', i) in Gamma(s)   cutoff chain
    (F6)  sum_{s' >=_i s} y[i, s'] >= A_is + z[i, s] - 1       no blocking pair

Why no cutoff variable
----------------------
(F5) makes ``z[., s]`` monotone down ``s``'s realised priority order, so the set
``{i : z[i, s] = 1}`` is a prefix of ``Gamma(s)`` -- and a prefix of the
priority order *is* the cutoff. An explicit cutoff variable would only re-encode
which prefix was chosen, and would then need either a grid to live on or big-M
rows to link it back to per-student admission. The chain needs neither: it is
``|Gamma(s)| - 1`` rows of the form ``z_i - z_{i'} <= 0``, all coefficients
unit, nothing to scale, and its relaxation is tight on the prefix polytope.
See :mod:`optimization.stable_cutoff` for the data-side consequence, namely
that ``z`` is meaningful only against the order it was built from.

Why the zoning does not leak here
---------------------------------
(F3) is the only row the zoning appears in, and it appears as an upper bound on
a variable whose objective coefficient is linear in it. A fractional
``A_is = 0.3`` buys 0.3 of a seat and 0.3 of the utility. That is the structural
difference from the finite-grid recurrence, whose ``min`` is concave in the
lottery mass and therefore pays *more* than pro rata for fractional access.
Measured on the real 6-zone-3 / Block_2 instance at eight enumerated feasible
zonings, this model's LP sits 0.73-0.91% above realised DA welfare, against the
aggregated recurrence's +0.8% and the SAA recourse LP's own +225 mean at the
same zonings -- i.e. it is a strictly better relaxation of zoned DA than the
Rothblum rows the ``saa`` strategy recourses through. (The joint root bound is a
separate matter and is *not* improved; see
``analysis/probe_cutoff_formulation`` GATE C.)

The two strengthenings
----------------------
(S1)  sum_{i'} y[i', s] >= q_s (1 - z[i, s])                   non-wastefulness
      If ``i`` failed to clear ``s``'s cutoff then ``s`` filled up;
      contrapositively an under-subscribed program rejects nobody. This is the
      family (F1)-(F6) leaves out, and yet the probe found the *integral*
      optimum identical with and without it at all eight zonings, reproducing DA
      to 2.4e-11 either way: the omitted matchings all have a blocking pair at
      an empty seat, and a blocking pair at an empty seat can always be
      exploited to a weakly better matching, so it never binds at the maximum.
      It is on by default anyway because it is worth roughly 100x in time --
      0.2-0.3s against 8-55s for a fixed-zoning MIP solve -- entirely through
      presolve, which uses it to fix ``z`` on the programs that cannot fill.
      Only turn it off to re-measure that.

(S2)  q_s sum_{s' >=_i s} y[i, s'] + sum_{i' >_s i} y[i', s] >= q_s A_is
      The aggregate stability row `SaaOracle` itself uses: either ``i`` is
      seated somewhere it likes at least as much as ``s``, or ``s`` is filled by
      students it *strictly* prefers to ``i``. It implies (F6) without
      mentioning ``z``, so as a relaxation it can only help (+78.8 mean LP slack
      against +98.6 without it), at the cost of one running-prefix variable per
      pair. On by default; the inner sum is carried by those prefix variables so
      the family stays ``O(|Gamma|)`` rows rather than ``O(|Gamma|^2)``.

Welfare is accumulated in the market's own unscaled utilities, so the objective
this solver reports is directly comparable to ``saa``'s welfare and to a
realised DA welfare, with no rounding drift from the integer utility scaling the
CP-SAT models need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gurobipy as gp
from gurobipy import GRB

from optimization.data.saa import SaaMarket, SaaSample
from optimization.problem import ZoneProblem
from optimization.saa_oracle import access_state
from optimization.solvers.mip import MipSolver, _AssignmentVars
from optimization.stable_cutoff import (
    restricted_access_pair_count,
    validate_sample_orders,
)


# Keys are (sample index, student index, program id).
_PairKey = tuple[int, int, str]


@dataclass
class _StableCutoffVariables:
    y: dict[_PairKey, gp.Var]
    z: dict[_PairKey, gp.Var]
    prefix: dict[_PairKey, gp.Var]
    seated: dict[tuple[int, str], gp.Var]
    utility: dict[_PairKey, float]
    access_indicator_count: int
    access_pair_count: int
    blocked_pair_count: int
    chain_row_count: int
    non_wastefulness_row_count: int
    aggregate_stability_row_count: int
    aggregate_stability_prefix_count: int


class StableCutoffMipSolver(MipSolver):
    """Extend the Gurobi zoning model with one exact matching block per sample."""

    def __init__(
        self,
        market: SaaMarket,
        samples: tuple[SaaSample, ...],
        *,
        non_wastefulness: bool = True,
        aggregate_stability: bool = True,
        tie_breaking_method: str = "STB",
        preprocessing_seconds: float = 0.0,
        **options,
    ) -> None:
        super().__init__(**options)
        samples = tuple(samples)
        # The market is *not* re-preprocessed here: preprocessing renumbers
        # programs and the samples index programs by position, so the two have
        # to be prepared together. See optimization/stable_cutoff.py.
        validate_sample_orders(market, samples)
        if not isinstance(non_wastefulness, bool):
            raise ValueError("stable_cutoff non_wastefulness must be a Boolean.")
        if not isinstance(aggregate_stability, bool):
            raise ValueError("stable_cutoff aggregate_stability must be a Boolean.")
        self.market = market
        self.samples = samples
        self.non_wastefulness = non_wastefulness
        self.aggregate_stability = aggregate_stability
        self.tie_breaking_method = str(tie_breaking_method).upper()
        self.preprocessing_seconds = float(preprocessing_seconds)
        self.relax_matching = False
        self._variables: _StableCutoffVariables | None = None

    # ------------------------------------------------------------------ #
    # Objective
    # ------------------------------------------------------------------ #
    def _add_model_objective(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> bool:
        self._variables = self._add_matching_model(m, problem, x)
        return True

    def _add_matching_model(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> _StableCutoffVariables:
        market = self.market
        programs = market.program_by_id
        capacity = {
            program.program_id: float(program.capacity) for program in market.programs
        }
        # The sample average. Each pair's utility coefficient carries its 1/S
        # share, so the objective is the mean matching welfare and reads on the
        # same scale as a single realised DA welfare.
        weight = 1.0 / len(self.samples)
        vtype = GRB.CONTINUOUS if self.relax_matching else GRB.BINARY

        # Access first: the co-zoning indicators are a property of the zoning
        # alone, so all samples share one copy of them.
        access_vars: dict[tuple[int, int], Any] = {}
        access_joints: dict[tuple[int, int, int], Any] = {}
        access_of: dict[tuple[int, str], Any] = {}
        for student_index, student in enumerate(market.students):
            for program_id in student.programs:
                pair, fixed = access_state(problem, student.node, programs[program_id])
                if pair is None:
                    access_of[(student_index, program_id)] = float(fixed)
                    continue
                access_of[(student_index, program_id)] = self._get_or_create_access_var(
                    m, problem, x, access_vars, access_joints, pair[0], pair[1]
                )

        y: dict[_PairKey, gp.Var] = {}
        z: dict[_PairKey, gp.Var] = {}
        prefix: dict[_PairKey, gp.Var] = {}
        seated: dict[tuple[int, str], gp.Var] = {}
        utility: dict[_PairKey, float] = {}
        objective = gp.LinExpr()
        blocked = 0
        chain_rows = 0
        non_wasteful_rows = 0
        stability_rows = 0
        stability_prefixes = 0

        for sample_index in range(len(self.samples)):
            for student_index, student in enumerate(market.students):
                previous = None
                for rank, program_id in enumerate(student.programs):
                    key = (sample_index, student_index, program_id)
                    access = access_of[(student_index, program_id)]
                    seat = m.addVar(
                        lb=0.0,
                        ub=1.0,
                        vtype=vtype,
                        name=f"sc_y_{sample_index}_{student_index}_{program_id}",
                    )
                    clears = m.addVar(
                        lb=0.0,
                        ub=1.0,
                        vtype=vtype,
                        name=f"sc_z_{sample_index}_{student_index}_{program_id}",
                    )
                    y[key] = seat
                    z[key] = clears

                    if isinstance(access, (int, float)):            # (F3)
                        if float(access) <= 0.0:
                            seat.UB = 0.0
                            blocked += 1
                    else:
                        m.addConstr(seat <= access)                 # (F3)
                    m.addConstr(seat <= clears)                     # (F4)

                    # The running "seated at something weakly preferred to s"
                    # sum, shared by (F1), (F6) and (S2). An equality
                    # definition, so it changes no relaxation value.
                    cumulative = m.addVar(
                        lb=0.0,
                        name=f"sc_p_{sample_index}_{student_index}_{rank}",
                    )
                    m.addConstr(
                        cumulative == (seat if previous is None else previous + seat)
                    )
                    prefix[key] = cumulative

                    # (F6): access plus clearing the cutoff means i must be
                    # seated at s or better, or (i, s) is a blocking pair.
                    if isinstance(access, (int, float)):
                        m.addConstr(cumulative - clears >= float(access) - 1.0)
                    else:
                        m.addConstr(cumulative - clears - access >= -1.0)

                    utility[key] = float(student.utilities[rank])
                    objective += weight * utility[key] * seat
                    previous = cumulative
                if previous is not None:
                    m.addConstr(previous <= 1.0)                    # (F1)

        for sample_index, sample in enumerate(self.samples):
            for program_index, program in enumerate(market.programs):
                program_id = program.program_id
                order = sample.school_orders[program_index]
                quota = capacity[program_id]
                # The seat total gets its own variable so that (F2) and every
                # (S1) row is a two-nonzero row instead of |Gamma(s)| of them.
                total = m.addVar(
                    lb=0.0,
                    ub=quota,
                    name=f"sc_seats_{sample_index}_{program_id}",
                )
                m.addConstr(                                        # (F2)
                    total
                    == gp.quicksum(
                        y[(sample_index, student_index, program_id)]
                        for student_index in order
                    )
                )
                seated[(sample_index, program_id)] = total

                running: Any = 0.0
                previous_clears = None
                for position, student_index in enumerate(order):
                    key = (sample_index, student_index, program_id)
                    clears = z[key]
                    if previous_clears is not None:
                        m.addConstr(clears <= previous_clears)      # (F5)
                        chain_rows += 1
                    previous_clears = clears

                    if self.non_wastefulness:
                        # sum_i' y[i', s] >= q_s (1 - z[i, s])
                        m.addConstr(total + quota * clears >= quota)
                        non_wasteful_rows += 1

                    if self.aggregate_stability:
                        access = access_of[(student_index, program_id)]
                        m.addConstr(quota * prefix[key] + running >= quota * access)
                        stability_rows += 1
                        # Running prefix over the students s *strictly* prefers
                        # to i, extended one position at a time so the family
                        # stays O(|Gamma(s)|) rows and columns. The last
                        # position has nobody left to serve, hence the guard.
                        if position + 1 < len(order):
                            tail = m.addVar(
                                lb=0.0,
                                ub=quota,
                                name=(
                                    f"sc_tail_{sample_index}_{program_id}_"
                                    f"{position + 1}"
                                ),
                            )
                            m.addConstr(tail == running + y[key])
                            stability_prefixes += 1
                            running = tail

        m.setObjective(objective, GRB.MAXIMIZE)
        return _StableCutoffVariables(
            y=y,
            z=z,
            prefix=prefix,
            seated=seated,
            utility=utility,
            access_indicator_count=sum(
                1 for var in access_vars.values() if not isinstance(var, (int, float))
            ),
            access_pair_count=restricted_access_pair_count(market),
            blocked_pair_count=blocked,
            chain_row_count=chain_rows,
            non_wastefulness_row_count=non_wasteful_rows,
            aggregate_stability_row_count=stability_rows,
            aggregate_stability_prefix_count=stability_prefixes,
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def sample_welfares(self) -> list[float]:
        """Per-sample matching welfare of the incumbent, in market utilities."""

        variables = self._variables
        if variables is None:
            return []
        welfares = [0.0] * len(self.samples)
        for key, var in variables.y.items():
            welfares[key[0]] += variables.utility[key] * var.X
        return welfares

    def _additional_solution_metadata(
        self, m: gp.Model, status: str
    ) -> dict[str, object]:
        variables = self._variables
        if variables is None:
            return {}
        metadata: dict[str, object] = {
            "formulation": "stable_cutoff_sampled_matching",
            "objective_kind": "stable_cutoff_sample_average_welfare",
            "stable_cutoff_num_seeds": len(self.samples),
            "stable_cutoff_sample_seeds": [sample.seed for sample in self.samples],
            "stable_cutoff_tie_breaking_method": self.tie_breaking_method,
            "stable_cutoff_gamma_size": self.market.preference_count,
            "stable_cutoff_y_vars": len(variables.y),
            "stable_cutoff_z_vars": len(variables.z),
            "stable_cutoff_prefix_vars": len(variables.prefix),
            "stable_cutoff_access_pair_count": variables.access_pair_count,
            "stable_cutoff_access_indicator_count": variables.access_indicator_count,
            "stable_cutoff_blocked_pair_count": variables.blocked_pair_count,
            "stable_cutoff_chain_row_count": variables.chain_row_count,
            "stable_cutoff_non_wastefulness": self.non_wastefulness,
            "stable_cutoff_non_wastefulness_row_count": (
                variables.non_wastefulness_row_count
            ),
            "stable_cutoff_aggregate_stability": self.aggregate_stability,
            "stable_cutoff_aggregate_stability_row_count": (
                variables.aggregate_stability_row_count
            ),
            "stable_cutoff_aggregate_stability_prefix_count": (
                variables.aggregate_stability_prefix_count
            ),
            "stable_cutoff_student_count": len(self.market.students),
            "stable_cutoff_program_count": len(self.market.programs),
            "stable_cutoff_utility_handling": self.market.utility_handling,
            "stable_cutoff_relaxed_matching": self.relax_matching,
            "stable_cutoff_model_variable_count": int(m.NumVars),
            "stable_cutoff_model_constraint_count": int(m.NumConstrs),
            "stable_cutoff_model_nonzero_count": int(m.NumNZs),
            "stable_cutoff_preprocessing_seconds": self.preprocessing_seconds,
            "aggregate_capacity_overage_disabled": True,
            "aggregate_capacity_shortage_disabled": True,
        }
        # The dual bound is reported whether or not an incumbent exists. It is
        # the whole point of this formulation -- a bound on sample-average
        # stable-matching welfare rather than on a surrogate -- and timing out
        # before the first incumbent is the normal shape on the real instance
        # without a feasible hint, exactly the case where knowing what was
        # proved matters most. Nesting it under ``SolCount`` throws that away.
        #
        # A pure LP has no ObjBound; ``IsMIP`` is 0 exactly when every matching
        # variable was relaxed and no zoning binary survived presolve, which
        # only happens in the relaxation probes.
        if m.IsMIP:
            bound = float(m.ObjBound)
            # Infinite until the root relaxation finishes.
            if math.isfinite(bound):
                metadata["stable_cutoff_best_objective_bound"] = bound
        if m.SolCount > 0:
            metadata["stable_cutoff_welfare"] = float(m.ObjVal)
            metadata["stable_cutoff_sample_welfares"] = self.sample_welfares()
            if m.IsMIP:
                metadata["stable_cutoff_relative_gap"] = float(m.MIPGap)
            else:
                metadata["stable_cutoff_best_objective_bound"] = float(m.ObjVal)
                metadata["stable_cutoff_relative_gap"] = 0.0
        return metadata
