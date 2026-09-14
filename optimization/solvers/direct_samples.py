"""Joint Boolean zoning and per-seed stable-matching model.

`MidCpSatSolver` extends the Boolean zoning model with the finite-grid lottery
recurrence and `StableCutoffMipSolver` extends the Gurobi one with a cutoff
chain; this solver extends the *same* Boolean zoning model with one copy of the
stable-matching access polytope per sampled tie-breaking seed, and maximizes
the average of their welfares. Every copy shares one set of co-zoning
indicators, so the seeds argue about one zoning, and the indicators are built
by `_get_or_create_access_var` -- the same linearization the choice, MID and
SAA models use.

The formulation, for one seed ``psi``, with ``A_is`` equal to 1 for a citywide
program and the co-zoning indicator of ``i``'s node with ``s``'s school node
otherwise, and ``Gamma(s)`` that seed's drawn strict priority order for ``s``:

    (A1)  sum_s D[i, s] <= 1                                one seat per student
    (A2)  sum_{i' in prefix of Gamma(s)} D[i', s] <= q_s    capacity
    (A3)  D[i, s] <= A_is                                   zoned access
    (A4)  q_s sum_{s' >=_i s} D[i, s'] + sum_{i' >_s i} D[i', s] >= q_s A_is

Why (A4) alone is exact
-----------------------
(A4) is the aggregated stability row the SAA recourse LP loads, and as a
*continuous* relaxation it is famously incomplete -- it has fractional extreme
points that no stable matching reaches, which is what the comb family of
Proposition 2 exists to cut off and what makes a recourse LP over it
over-report realized deferred acceptance by 195 to 252 units on Block_2. With
``D`` Boolean none of that survives: read (A4) at an integral point and it is
Lemma 1 verbatim, so its integral solutions are exactly the stable matchings of
the market the zoning leaves, the comb rows are implied, and there is nothing
to separate. That is the entire argument for this strategy, and
:meth:`solve` checks it by replaying deferred acceptance on the zoning the
search returns.

Two consequences worth stating. There is no non-wastefulness family to add:
(A4) already covers the empty-seat case, because a program that cannot fill has
a shaft below ``q_s`` and therefore has to admit everyone with access who wants
it. And there is no epigraph variable and no a-priori ``Welfare_max``: the
objective *is* the matching welfare, so the dual bound branch and bound reports
is a bound on sample-average stable-matching welfare rather than on a surrogate
of it.

The shaft chain
---------------
``sum_{i' >_s i} D[i', s]`` is a prefix of ``Gamma(s)``, so it is carried by one
running variable per position -- ``|Gamma|`` equality rows of two nonzeros each
per seed -- rather than re-summed into every row, which would cost
``O(|Gamma| n_s)`` nonzeros. Bounding each running variable above by ``q_s`` is
(A2): the prefixes increase, so capping every prefix is capping the total, and
the last one is the total. The weak-preference sum on the left of (A4) is
written inline instead, because ranked lists average 3.13 entries and a
variable would cost more than the two nonzeros it saves.

Over the first ``q_s`` positions of an order the shaft cannot reach ``q_s``, so
(A4) there is equivalent to the clause ``A_is -> sum_{s' >=_i s} D[i, s'] >=
1`` and the running variable is redundant -- a program with ``n_s <= q_s``
needs no chain at all. Writing that out was measured and reverted: it removes
about a third of the shaft variables at build time (97,086 columns down to
93,273 on a Block_2-scale instance), but CP-SAT presolve derives it unaided and
lands within four columns either way -- 82,892 against 82,888 -- and the two
solve a pinned matching in the same 0.2s. Presolve is the right place for it.

What the access block costs
---------------------------
The SAA master instantiates a co-zoning indicator only for the vertex pairs
carrying a non-zero marginal dual value -- about 1,700 on Block_2. This model
needs one for *every* pair in ``Gamma``, because every seat variable is bounded
by its own, and each indicator brings one conjunction variable per zone the two
vertices share. That block, not the seats, is the bulk of the model, and it is
shared across seeds: going from one seed to five leaves it untouched.

Welfare is accumulated in `CP_SAT_SCALE`-scaled integer utilities, as CP-SAT
requires. Scaling is monotone, so it moves no argmax and the reported objective
is still the realized matching welfare -- but it is the replay in
``direct_samples_welfare``, not the rounded solver objective, that
:meth:`solve` puts on `ZoneSolution.objective`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model

from optimization.data.saa import SaaMarket, SaaSample
from optimization.direct_samples import (
    matching_welfare,
    replay,
    scaled_matching_welfare,
)
from optimization.problem import ZoneProblem
from optimization.saa_oracle import access_state, required_access_pairs
from optimization.solution import ZoneSolution
from optimization.solvers.cpsat import (
    CP_SAT_SCALE,
    CpBoolSolver,
    _AssignmentVars,
    _ZoneVars,
)
from optimization.stable_cutoff import validate_sample_orders


# Keys are (sample index, student index, preference rank).
_SeatKey = tuple[int, int, int]


@dataclass
class _DirectSamplesVariables:
    #: The model these variables index. CP-SAT variables are proto indices, so
    #: hinting one against a different model would silently hint whatever sits
    #: at that index -- possible here because `find_feasible_solution` builds a
    #: model that never calls `_add_model_objective`.
    model: cp_model.CpModel
    seats: dict[_SeatKey, cp_model.IntVar]
    #: The strict shaft at each seat's position: a running variable, or the
    #: constant 0 at the first position of an order.
    shafts: dict[_SeatKey, Any]
    access: dict[tuple[int, int], Any]
    objective_bound: int
    access_pair_count: int
    blocked_pair_count: int
    stability_row_count: int


class DirectSamplesCpSatSolver(CpBoolSolver):
    """Extend `cp_bool` with one Boolean stable matching per sampled seed."""

    def __init__(
        self,
        market: SaaMarket,
        samples: tuple[SaaSample, ...],
        *,
        tie_breaking_method: str = "MTB",
        preprocessing_seconds: float = 0.0,
        **options,
    ) -> None:
        super().__init__(**options)
        samples = tuple(samples)
        # The market is *not* re-preprocessed here: preprocessing renumbers
        # programs and the samples index programs by position, so the two have
        # to be prepared together. See optimization/stable_cutoff.py.
        validate_sample_orders(market, samples)
        self.market = market
        self.samples = samples
        self.tie_breaking_method = str(tie_breaking_method).upper()
        self.preprocessing_seconds = float(preprocessing_seconds)
        self._variables: _DirectSamplesVariables | None = None
        self.sample_matchings: tuple[tuple[int, ...], ...] | None = None

    # ------------------------------------------------------------------ #
    # Objective
    # ------------------------------------------------------------------ #
    def _add_model_objective(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> tuple[bool, float]:
        self._variables = self._add_matching_model(model, problem, x)
        return True, float(len(self.samples) * CP_SAT_SCALE)

    def _add_matching_model(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
    ) -> _DirectSamplesVariables:
        market = self.market
        programs = market.program_by_id
        rank_of = {
            (student_index, program_id): rank
            for student_index, student in enumerate(market.students)
            for rank, program_id in enumerate(student.programs)
        }

        # Access first: a co-zoning indicator is a property of the zoning
        # alone, so every seed shares one copy of it.
        access_vars: dict[tuple[int, int], Any] = {}
        access_joints: dict[tuple[int, int, int], Any] = {}
        access_of: dict[tuple[int, int], Any] = {}
        for student_index, student in enumerate(market.students):
            for rank, program_id in enumerate(student.programs):
                pair, fixed = access_state(problem, student.node, programs[program_id])
                access_of[(student_index, rank)] = (
                    int(fixed)
                    if pair is None
                    else self._get_or_create_access_var(
                        model, problem, x, access_vars, access_joints, pair[0], pair[1]
                    )
                )

        # A pair no zoning can reach carries no seat variable in any seed, and
        # its (A4) row is vacuous because the right-hand side is zero.
        # `preprocess_saa_market` normally removes these before the draw, so
        # this counts what survived it.
        blocked = sum(
            1 for access in access_of.values() if isinstance(access, int) and not access
        )

        seats: dict[_SeatKey, cp_model.IntVar] = {}
        objective_terms: list[cp_model.IntVar] = []
        objective_weights: list[int] = []
        for sample_index in range(len(self.samples)):
            for student_index, student in enumerate(market.students):
                row = []
                for rank, program_id in enumerate(student.programs):
                    access = access_of[(student_index, rank)]
                    if isinstance(access, int) and access == 0:
                        continue
                    seat = model.NewBoolVar(
                        f"ds_d_{sample_index}_{student_index}_{rank}"
                    )
                    if not isinstance(access, int):
                        model.Add(seat <= access)                          # (A3)
                    seats[(sample_index, student_index, rank)] = seat
                    row.append(seat)
                    objective_terms.append(seat)
                    objective_weights.append(int(student.scaled_utilities[rank]))
                if len(row) > 1:
                    model.AddAtMostOne(row)                                # (A1)

        shafts: dict[_SeatKey, Any] = {}
        stability_rows = 0
        for sample_index, sample in enumerate(self.samples):
            for program_index, program in enumerate(market.programs):
                program_id = program.program_id
                quota = int(program.capacity)
                # The strict shaft: how many applicants this program prefers to
                # the one at the current position are seated here. Starts empty
                # and is extended one position at a time.
                shaft: Any = 0
                order = sample.school_orders[program_index]
                for place, student_index in enumerate(order):
                    rank = rank_of[(student_index, program_id)]
                    key = (sample_index, student_index, rank)
                    seat = seats.get(key)
                    if seat is None:
                        continue
                    shafts[key] = shaft
                    access = access_of[(student_index, rank)]
                    # (A4). The weak-preference sum is inline; ranks above
                    # `rank` that lost their seat variable are unreachable and
                    # contribute nothing.
                    weak = [
                        seats[(sample_index, student_index, better)]
                        for better in range(rank + 1)
                        if (sample_index, student_index, better) in seats
                    ]
                    stability = cp_model.LinearExpr.WeightedSum(
                        weak, [quota] * len(weak)
                    )
                    if isinstance(access, int):
                        model.Add(stability + shaft >= quota)
                    else:
                        model.Add(stability + shaft - quota * access >= 0)
                    stability_rows += 1

                    # (A2) rides on the domain: prefixes of a sum of Booleans
                    # increase, so capping every prefix at q_s caps the total.
                    extended = model.NewIntVar(
                        0,
                        min(place + 1, quota),
                        f"ds_shaft_{sample_index}_{program_index}_{place + 1}",
                    )
                    model.Add(extended == shaft + seat)
                    shaft = extended

        model.Maximize(
            cp_model.LinearExpr.WeightedSum(objective_terms, objective_weights)
        )

        return _DirectSamplesVariables(
            model=model,
            seats=seats,
            shafts=shafts,
            access=access_vars,
            objective_bound=len(self.samples)
            * sum(
                max(student.scaled_utilities, default=0)
                for student in market.students
            ),
            access_pair_count=len(required_access_pairs(market)),
            blocked_pair_count=blocked,
            stability_row_count=stability_rows,
        )

    # ------------------------------------------------------------------ #
    # Hints
    # ------------------------------------------------------------------ #
    def _add_hints(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        super()._add_hints(model, problem, x, y)
        variables = self._variables
        if variables is None or variables.model is not model:
            return
        if not problem.hint or set(problem.hint) != set(problem.nodes):
            return

        # The hint zoning already fixes the matching: deferred acceptance on it
        # is feasible for every row above and optimal for the objective, so
        # hinting `x` alone would leave CP-SAT to rediscover by search
        # something a propose-and-reject loop hands over in milliseconds.
        matchings = replay(self.market, self.samples, problem, problem.hint)
        for sample_index, matching in enumerate(matchings):
            for student_index, matched_rank in enumerate(matching):
                student = self.market.students[student_index]
                for rank in range(len(student.programs)):
                    seat = variables.seats.get((sample_index, student_index, rank))
                    if seat is not None:
                        model.AddHint(seat, int(rank == matched_rank))

        # Walk the drawn orders exactly as the chain was built, so each running
        # variable is hinted the prefix the hinted seats actually give it.
        rank_of = {
            (student_index, program_id): rank
            for student_index, student in enumerate(self.market.students)
            for rank, program_id in enumerate(student.programs)
        }
        for sample_index, sample in enumerate(self.samples):
            matching = matchings[sample_index]
            for program_index, program in enumerate(self.market.programs):
                shaft = 0
                for student_index in sample.school_orders[program_index]:
                    rank = rank_of[(student_index, program.program_id)]
                    key = (sample_index, student_index, rank)
                    if key not in variables.seats:
                        continue
                    running = variables.shafts[key]
                    if not isinstance(running, int):
                        model.AddHint(running, shaft)
                    if matching[student_index] == rank:
                        shaft += 1

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def sample_welfares(self, solver: cp_model.CpSolver) -> list[float]:
        """Per-seed matching welfare of the incumbent, in market utilities."""

        variables = self._variables
        if variables is None:
            return []
        welfares = [0.0] * len(self.samples)
        for (sample_index, student_index, rank), seat in variables.seats.items():
            if solver.Value(seat):
                welfares[sample_index] += self.market.students[
                    student_index
                ].utilities[rank]
        return welfares

    def _additional_solution_metadata(
        self, solver: cp_model.CpSolver, model: cp_model.CpModel, status: int
    ) -> dict[str, object]:
        variables = self._variables
        if variables is None:
            return {}
        scale = len(self.samples) * CP_SAT_SCALE
        metadata: dict[str, object] = {
            "formulation": "direct_samples_stable_matching",
            "objective_kind": "direct_samples_sample_average_welfare",
            "direct_samples_num_seeds": len(self.samples),
            "direct_samples_sample_seeds": [sample.seed for sample in self.samples],
            "direct_samples_tie_breaking_method": self.tie_breaking_method,
            "direct_samples_gamma_size": self.market.preference_count,
            "direct_samples_seat_vars": len(variables.seats),
            "direct_samples_shaft_vars": sum(
                1 for shaft in variables.shafts.values() if not isinstance(shaft, int)
            ),
            "direct_samples_stability_row_count": variables.stability_row_count,
            "direct_samples_access_pair_count": variables.access_pair_count,
            "direct_samples_access_indicator_count": sum(
                1 for var in variables.access.values() if not isinstance(var, int)
            ),
            "direct_samples_blocked_pair_count": variables.blocked_pair_count,
            "direct_samples_student_count": len(self.market.students),
            "direct_samples_utility_student_count": (
                self.market.utility_student_count
            ),
            # Students whose whole list was dropped -- by `omit_nonpositive`,
            # or because no zoning reaches any of it. They cost no variables:
            # the outside option is the absence of a seat, not a column.
            "direct_samples_outside_only_student_count": sum(
                not student.programs for student in self.market.students
            ),
            "direct_samples_program_count": len(self.market.programs),
            "direct_samples_utility_handling": self.market.utility_handling,
            "direct_samples_utility_scale": CP_SAT_SCALE,
            "direct_samples_objective_upper_bound": variables.objective_bound,
            "direct_samples_model_variable_count": len(model.Proto().variables),
            "direct_samples_model_constraint_count": len(model.Proto().constraints),
            "direct_samples_model_hint_count": len(model.Proto().solution_hint.vars),
            "direct_samples_preprocessing_seconds": self.preprocessing_seconds,
            "aggregate_capacity_overage_disabled": True,
            "aggregate_capacity_shortage_disabled": True,
        }
        # The dual bound is reported whether or not an incumbent exists: it is
        # a bound on sample-average stable-matching welfare itself, and timing
        # out before the first zoning is the normal shape on the real instance,
        # exactly the case where knowing what was proved matters most.
        if status not in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
            raw_bound = float(solver.BestObjectiveBound())
            # Infinite until the first bound is proved.
            if math.isfinite(raw_bound):
                metadata["direct_samples_raw_best_objective_bound"] = raw_bound
                metadata["direct_samples_best_objective_bound"] = raw_bound / scale
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raw_objective = int(round(solver.ObjectiveValue()))
            metadata["direct_samples_raw_solver_objective"] = raw_objective
            metadata["direct_samples_solver_welfare"] = raw_objective / scale
            metadata["direct_samples_sample_solver_welfares"] = self.sample_welfares(
                solver
            )
        return metadata

    # ------------------------------------------------------------------ #
    # Replay
    # ------------------------------------------------------------------ #
    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        solution = super().solve(problem)
        if not solution.feasible:
            return solution

        replay_start = time.perf_counter()
        matchings = replay(self.market, self.samples, problem, solution.assignment)
        replay_seconds = time.perf_counter() - replay_start
        self.sample_matchings = matchings
        welfares = [
            matching_welfare(self.market, matching) for matching in matchings
        ]
        scaled = [
            scaled_matching_welfare(self.market, matching) for matching in matchings
        ]
        welfare = sum(welfares) / len(welfares)
        raw_objective = int(solution.metadata["direct_samples_raw_solver_objective"])

        # The model's integral points are the stable matchings of the market
        # this zoning leaves, and deferred acceptance is the best of them, so
        # an objective *above* the replay is not a better solution -- it is a
        # matching the rows should have excluded.
        if raw_objective > sum(scaled):
            raise RuntimeError(
                "direct_samples objective exceeds deferred-acceptance welfare "
                f"({raw_objective} > {sum(scaled)}): the matching block admits "
                "an unstable matching."
            )
        solution.objective = welfare
        solution.metadata.update(
            {
                "objective_kind": "direct_samples_sample_average_welfare",
                "direct_samples_welfare": welfare,
                "direct_samples_sample_welfares": welfares,
                "direct_samples_raw_replay_objective": sum(scaled),
                # Equal whenever the search closed the gap; below it by the
                # amount the incumbent's matchings are short of optimal
                # otherwise. Not an error either way -- a timed-out run holds a
                # stable matching that is merely not the applicant-optimal one.
                "direct_samples_replay_agreement": raw_objective == sum(scaled),
                "direct_samples_replay_seconds": replay_seconds,
            }
        )
        return solution
