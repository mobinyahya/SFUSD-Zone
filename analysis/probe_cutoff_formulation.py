#!/usr/bin/env python3
"""Is a per-student cutoff formulation an exact, tighter model of zoned DA?

Every joint model measured so far links welfare to the zoning through the
aggregated finite-grid recurrence, and `probe_access_convexification` showed why
that architecture cannot be repaired from the outside: the root relaxation sits
at 17,377 against a best-known zoning of about 14,950, and the leaking point is
a perfectly consistent co-zoning matrix, so no valid inequality written in
``a``-space can cut it off.  The recurrence trades one variable per (type, rank)
for a *continuum* of lottery mass, and a fractional co-zoning weight buys a
pro-rata slice of a school that no integral zoning could hand out.

This probe tests the other end of the design space.  Fix a single lottery draw
(STB, one sample), drop the lottery continuum entirely, and write the
student-optimal stable matching as a mixed-integer program over per-student
assignment variables ``y[i, s]`` and per-student cutoff-clearing indicators
``z[i, s]``.  ``z`` is what replaces the recurrence: a program's cutoff is no
longer a number on a grid but a *position* in that program's realised priority
order, encoded by monotonicity down the order.  The whole formulation is

    (F1)  sum_{s in Gamma(i)} y[i, s] <= 1                     one seat per student
    (F2)  sum_{i in Gamma(s)} y[i, s] <= q_s                   capacity
    (F3)  y[i, s] <= A_is                                      zoned access
    (F4)  y[i, s] <= z[i, s]                                   only admits are seated
    (F5)  z[i, s] <= z[i', s]   for i' the immediate predecessor of i in s's
                                priority order                 cutoff monotonicity
    (F6)  sum_{s' >=_i s} y[i, s'] >= A_is + z[i, s] - 1       no blocking pair

with ``A_is`` the co-zoning product ``sum_z x[z, v_i] x[z, u_s]``, McCormick-
linearised exactly as `probe_mid_root_bound.RootModel._access` does it, and 1 or
0 wherever the pair is settled by construction (citywide programs, a school on
the student's own node, a pair with no common candidate zone).

The point of interest is that (F3) is the *only* place the zoning enters, and it
enters as an upper bound on a variable that is 0/1 in every integral solution.
There is no pro-rata slice to buy: a fractional ``A_is = 0.3`` lets student ``i``
take 0.3 of a seat at ``s``, and 0.3 of a seat is worth 0.3 of the utility --
linear, not concave -- so the leak the recurrence suffers from is structurally
absent.  Whether that makes the *bound* better is a different question from
whether it makes the model exact, hence three separate gates.

GATE A -- exactness.  A model that is not exact is worthless no matter how tight
it is, and (F1)-(F6) is not obviously exact: it omits non-wastefulness.  Nothing
in it forces ``z[i, s] = 1`` when ``s`` ends up under-subscribed, so an integral
point may describe a matching with a blocking pair at an empty seat.  Whether
the *maximum* is still the DA welfare is an empirical question, so: fix ``x`` to
an enumerated feasible zoning, declare ``y`` and ``z`` binary, solve to
optimality, and compare against the realised student-optimal stable matching on
the same zoning and the same sample.  Run it with and without (S1) so the cost
of the omission is measured rather than assumed.

The reference this gate was meant to use was `SaaOracle`, and the first run of
this probe showed it cannot be: `SaaOracleResult.welfare` is the optimum of the
*continuous* stable-admissions LP, and that LP is a relaxation of the many-to-one
stable matching polytope, not a description of it.  Measured here it over-reports
the realised DA welfare by 195 to 252 per zoning, which is larger than every
quantity the three gates are trying to resolve.  So the probe runs deferred
acceptance itself (:class:`DeferredAcceptance`) and reports the oracle's own
over-report alongside, as a by-product worth knowing: it is the per-scenario
slack every SAA Benders cut inherits.

GATE B -- LP slack at fixed ``x``.  The analogue of the recurrence's documented
+0.8%.  Relax ``y`` and ``z`` and re-solve at the same eight zonings.  Anything
negative here is a validity bug, not slack, and is reported as such.

GATE C -- joint root bound, which is the number that decides whether any of this
is worth building.  Free ``x``, relax everything, and read the bound off against
the five reference points collected on this same instance: best known MID
welfare ~14,950; the transportation bound 18,699.31; the first-choice bound
19,311.29; the aggregated monolith's root-equivalent LP bound 17,377.24; and the
`mid` strategy's dual bound after 1800s of branch and bound, 16,572.83.

Two optional strengthenings are switchable arms rather than defaults, because
each one costs |Gamma| rows:

(S1)  sum_{i'} y[i', s] >= q_s (1 - z[i, s])
      Non-wastefulness.  If ``i`` failed to clear ``s``'s cutoff then ``s``
      filled up; contrapositively an under-subscribed program rejects nobody.
      This is the family (F1)-(F6) leaves out.

(S2)  q_s sum_{s' >=_i s} y[i, s'] + sum_{i' >_s i} y[i', s] >= q_s A_is
      The stability row `SaaOracle` itself uses: either ``i`` is seated
      somewhere it likes at least as much as ``s``, or ``s`` is filled by
      students it strictly prefers to ``i``.  It implies (F6) without mentioning
      ``z``, so as an LP it can only help; the inner sum is carried by running
      prefix variables so the family stays O(|Gamma|) rows and columns.

What it measured, on 514 nodes / 6 zones / 123 programs / 3,953 students /
|Gamma| = 12,370, against 8 enumerated feasible zonings spread by Hamming
distance (min 249, mean 298, max 354 of 514 vertices):

GATE A  exact.  All 8 zonings, both (F1)-(F6) and +(S1): the MIP optimum
        reproduces the DA welfare to 2.4e-11.  Omitting non-wastefulness costs
        nothing in value -- but it costs a great deal in time.  (S1) neither
        raises nor lowers a single LP value anywhere in the probe, and yet it
        takes the fixed-``x`` MIP from 8-55s to 0.1-0.3s, a 100x speedup that
        comes entirely from presolve.
GATE B  +98.6 mean (+81 to +119, 0.91% of DA) for the base and +(S1) arms,
        +78.8 mean (+65 to +97, 0.73%) with (S2).  Never negative.  Both are
        *tighter* than the SaaOracle LP's own +225 mean at the same fixed
        zoning, so the ``z``-monotone model is a better relaxation of zoned DA
        than the Rothblum rows the SAA strategy currently recourses through.
GATE C  17,314.35 (base, +S1) and 17,307.92 (+S2, +S1+S2), against the
        aggregated monolith's 17,377.24.  That is the verdict: an exact model
        whose welfare block is tight to 0.7-0.9% at fixed ``x`` still leaves a
        joint root bound within 70 of the model it was meant to replace.  The
        leak is not in the welfare model.  It is in the fractional co-zoning
        that (F3) hands the relaxation, and swapping the welfare block does
        nothing about it.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import statistics
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from analysis.probe_access_convexification import feasible_zonings
from analysis.probe_mid_root_bound import RootModel
from benchmark.slurm import create_plan
from optimization.data.saa import build_saa_market, sample_school_preferences
from optimization.levels import LevelSpec
from optimization.saa_oracle import SaaOracle, access_state
from optimization.solvers.cpsat import CP_SAT_SCALE
from optimization.welfare_bounds import first_choice_upper_bound, transport_upper_bound


# All measured on this instance (priced_access_seeds.yaml, 6-zone-3), so the
# GATE C column is directly comparable to them.  Note that the first entry is a
# MID (lottery-expectation) welfare for the best zoning found, whereas GATE A
# and GATE B report realised single-draw welfare for enumerated zonings: the
# bounds are comparable, the incumbents are not.
REFERENCES = (
    ("best known MID welfare", 14_950.0),
    ("mid dual bound after 1800s B&B", 16_572.83),
    ("aggregated monolith root LP", 17_377.24),
    ("transportation bound", 18_699.31),
    ("first-choice bound", 19_311.29),
)

ARMS = (
    ("base (F1-F6)", ()),
    ("+S1", ("S1",)),
    ("+S2", ("S2",)),
    ("+S1+S2", ("S1", "S2")),
)

# Arms whose integral exactness is genuinely in question.  (S2) is the
# Rothblum-style stability row, whose *integral* points are exactly the stable
# matchings even though its LP relaxation is not the stable matching polytope,
# so an arm carrying it is exact by construction and only needs a spot check;
# the interesting cases are the two that rely on (F6) and monotone `z` alone.
GATE_A_ARMS = frozenset({"base (F1-F6)", "+S1"})


def build_individual_instance(config_path: str, centroids: str):
    """`probe_mid_root_bound.build_instance`, but for the individual market.

    Same body -- and therefore the same problem, the same nodes and the same
    disabled non-welfare penalties -- except that it returns the finite
    per-student market `SaaOracle` consumes instead of the compressed MID market
    the finite-grid oracle consumes.
    """
    plan = create_plan(config_path)
    config = plan.tasks[0].optimization_config()
    config.centroids_type = centroids
    target = LevelSpec.parse(config.levels[-1])
    dataset = config.make_dataset()
    problem = dataset.problem_for(target)
    # The probe measures welfare alone; the shape penalties would otherwise
    # enter the objective and make the bound incomparable.
    problem.overage = -1.0
    problem.shortage = -1.0
    problem.boundary_prop = -1.0
    market = build_saa_market(problem, dataset.config)
    return problem, market


def spread_zonings(zonings: list[dict], count: int) -> list[dict]:
    """A farthest-point subset, because enumeration returns near-duplicates.

    CP-SAT's solution pool walks a neighbourhood: the first eight zonings it
    emits differ from one another in one or two vertices, which would make GATE
    A and GATE B eight repetitions of the same test.  Greedy farthest-point
    traversal in Hamming distance picks a subset that actually varies.
    """
    if len(zonings) <= count:
        return zonings
    nodes = sorted(zonings[0])
    vectors = [tuple(assignment[node] for node in nodes) for assignment in zonings]

    def distance(left: int, right: int) -> int:
        return sum(a != b for a, b in zip(vectors[left], vectors[right]))

    chosen = [0]
    while len(chosen) < count:
        best, best_gap = None, -1
        for index in range(len(zonings)):
            if index in chosen:
                continue
            gap = min(distance(index, picked) for picked in chosen)
            if gap > best_gap:
                best, best_gap = index, gap
        chosen.append(best)
    return [zonings[index] for index in chosen]


def hamming_summary(zonings: list[dict]) -> tuple[int, float, int]:
    if len(zonings) < 2:
        return 0, 0.0, 0
    nodes = sorted(zonings[0])
    vectors = [tuple(assignment[node] for node in nodes) for assignment in zonings]
    distances = [
        sum(a != b for a, b in zip(left, right))
        for left, right in itertools.combinations(vectors, 2)
    ]
    return min(distances), statistics.mean(distances), max(distances)


class DeferredAcceptance:
    """Exact student-proposing DA for one lottery draw: the probe's ground truth.

    `SaaOracle` was the intended reference and cannot be one.  Its ``welfare``
    field is the optimum of the continuous stable-admissions LP -- the
    Rothblum-style rows ``sum_{i' >_s i} y[i', s] + q_s sum_{s' >=_i s} y[i, s']
    >= q_s A_is`` -- which describes the stable matching polytope in the
    one-to-one case but is only a relaxation of it in the many-to-one case.  It
    over-reports here by 200 to 245 welfare per zoning, several times the LP
    slack the probe is trying to measure.

    Student-proposing DA returns the student-optimal stable matching, which
    every student weakly prefers to every other stable matching and which
    therefore maximises total student utility over them.  That is exactly what
    the MIP of GATE A is supposed to reproduce.
    """

    def __init__(self, problem, market, sample):
        self.problem = problem
        self.market = market
        self.capacity = {
            program.program_id: int(program.capacity) for program in market.programs
        }
        self.position = {
            (student_index, program.program_id): position
            for index, program in enumerate(market.programs)
            for position, student_index in enumerate(sample.school_orders[index])
        }

    def _lists(self, zoning: dict[int, int]):
        """Each student's preference list, filtered to the accessible programs."""
        programs = self.market.program_by_id
        lists = []
        for student in self.market.students:
            sequence = []
            for rank, program_id in enumerate(student.programs):
                pair, fixed = access_state(
                    self.problem, student.node, programs[program_id]
                )
                if pair is None:
                    reachable = bool(fixed)
                else:
                    reachable = zoning[pair[0]] == zoning[pair[1]]
                if reachable:
                    sequence.append((program_id, student.utilities[rank]))
            lists.append(sequence)
        return lists

    def solve(self, zoning: dict[int, int]) -> tuple[float, int]:
        lists = self._lists(zoning)
        nxt = [0] * len(lists)
        # Each program keeps a heap on the *negated* priority position, so
        # heap[0] is the least-preferred student it currently holds -- the one
        # a better applicant bumps.
        held: dict[str, list] = {program_id: [] for program_id in self.capacity}
        seated: dict[int, float] = {}
        free = [index for index, sequence in enumerate(lists) if sequence]
        while free:
            index = free.pop()
            while nxt[index] < len(lists[index]):
                program_id, utility = lists[index][nxt[index]]
                nxt[index] += 1
                capacity = self.capacity[program_id]
                if capacity <= 0:
                    continue
                position = self.position[(index, program_id)]
                heap = held[program_id]
                if len(heap) < capacity:
                    heapq.heappush(heap, (-position, index, utility))
                    seated[index] = utility
                    break
                worst_negated, worst_index, _ = heap[0]
                if -worst_negated > position:
                    heapq.heapreplace(heap, (-position, index, utility))
                    seated[index] = utility
                    del seated[worst_index]
                    free.append(worst_index)
                    break
        return sum(seated.values()), len(seated)


class CutoffModel:
    """The (F1)-(F6) cutoff formulation jointly with the zoning, in Gurobi.

    ``x`` starts free and continuous, so the model as built is the GATE C
    relaxation; :meth:`fix_zoning` turns it into the GATE B model and
    :meth:`set_integral` on top of that into the GATE A model.  Building once
    and re-stamping bounds keeps the three gates on literally the same rows.
    """

    def __init__(self, problem, market, sample, extras=()):
        self.problem = problem
        self.market = market
        self.sample = sample
        self.extras = frozenset(extras)
        unknown = self.extras - {"S1", "S2"}
        if unknown:
            raise ValueError(f"Unknown strengthenings: {sorted(unknown)}")
        self.model = gp.Model()
        self.model.Params.OutputFlag = 0
        self.model.ModelSense = GRB.MAXIMIZE
        self.counts: dict[str, int] = {}
        # The zoning block, verbatim: one-hot per vertex, centroids pinned,
        # contiguity in its linear form, the FRL band and the school-count band.
        RootModel._zoning(self)
        self.model.update()
        self._x_bounds = {key: (var.LB, var.UB) for key, var in self.x.items()}
        self._access()
        self._matching()
        self.model.update()

    # -- access ------------------------------------------------------------- #

    def _access(self) -> None:
        """Co-zoning variables for the (student node, school node) pairs in use.

        Only the pairs some student's preference list actually needs, and only
        the ones `saa_oracle.access_state` reports as undecided -- which is
        exactly the set of access coefficients the oracle itself varies, so the
        two models see the same access structure by construction.
        """
        problem, m = self.problem, self.model
        needed = set()
        for student in self.market.students:
            for program_id in student.programs:
                pair, _ = access_state(
                    problem, student.node, self.market.program_by_id[program_id]
                )
                if pair is not None:
                    needed.add((min(pair), max(pair)))

        self.a: dict[tuple[int, int], object] = {}
        self.both: dict[tuple[int, int, int], object] = {}
        for u, v in sorted(needed):
            common = sorted(problem.candidate_zones(u) & problem.candidate_zones(v))
            if not common:
                self.a[(u, v)] = 0.0
                continue
            pair_var = m.addVar(lb=0.0, ub=1.0, name=f"a_{u}_{v}")
            joints = []
            for zone in common:
                xu, xv = self.x[(zone, u)], self.x[(zone, v)]
                both = m.addVar(lb=0.0, ub=1.0, name=f"b_{u}_{v}_{zone}")
                m.addConstr(both <= xu)
                m.addConstr(both <= xv)
                m.addConstr(both >= xu + xv - 1.0)
                self.both[(u, v, zone)] = both
                joints.append(both)
            m.addConstr(pair_var == gp.quicksum(joints))
            self.a[(u, v)] = pair_var
        self.counts["access_pairs"] = sum(
            1 for value in self.a.values() if not isinstance(value, float)
        )
        self.counts["access_joints"] = len(self.both)

    def _access_expr(self, student_node: int, program):
        """``A_is``: 1.0, 0.0, or the co-zoning variable for the pair."""
        pair, fixed = access_state(self.problem, student_node, program)
        if pair is None:
            return float(fixed)
        return self.a.get((min(pair), max(pair)), 0.0)

    # -- the matching block -------------------------------------------------- #

    def _matching(self) -> None:
        market, m = self.market, self.model
        capacity = {
            program.program_id: float(program.capacity) for program in market.programs
        }
        orders = {
            program.program_id: self.sample.school_orders[index]
            for index, program in enumerate(market.programs)
        }

        self.y: dict[tuple[int, str], object] = {}
        self.z: dict[tuple[int, str], object] = {}
        # prefix[(i, s)] == sum of y over the programs i weakly prefers to s.
        self.prefix: dict[tuple[int, str], object] = {}
        self.access_of: dict[tuple[int, str], object] = {}
        objective = gp.LinExpr()
        blocked = 0

        for index, student in enumerate(market.students):
            previous = None
            for rank, program_id in enumerate(student.programs):
                program = market.program_by_id[program_id]
                access = self._access_expr(student.node, program)
                self.access_of[(index, program_id)] = access
                y = m.addVar(lb=0.0, ub=1.0, name=f"y_{index}_{program_id}")
                z = m.addVar(lb=0.0, ub=1.0, name=f"z_{index}_{program_id}")
                self.y[(index, program_id)] = y
                self.z[(index, program_id)] = z

                if isinstance(access, float):
                    if access <= 0.0:  # (F3)
                        y.UB = 0.0
                        blocked += 1
                else:
                    m.addConstr(y <= access)  # (F3)
                m.addConstr(y <= z)  # (F4)

                # The running "seated at something weakly preferred" sum, shared
                # by (F1), (F6) and (S2); an equality definition, so it changes
                # no relaxation value.
                cumulative = m.addVar(lb=0.0, name=f"P_{index}_{rank}")
                m.addConstr(cumulative == (y if previous is None else previous + y))
                self.prefix[(index, program_id)] = cumulative

                # (F6): access and clearing the cutoff together mean i must be
                # seated at s or better, or (i, s) blocks.
                if isinstance(access, float):
                    m.addConstr(cumulative - z >= access - 1.0)
                else:
                    m.addConstr(cumulative - z - access >= -1.0)

                objective += float(student.scaled_utilities[rank]) * y
                previous = cumulative
            if previous is not None:
                m.addConstr(previous <= 1.0)  # (F1)

        self.counts["gamma"] = len(self.y)
        self.counts["access_blocked_pairs"] = blocked
        self.counts["y_vars"] = len(self.y)
        self.counts["z_vars"] = len(self.z)
        self.counts["prefix_vars"] = len(self.prefix)

        # Per-program rows.  The seat total gets its own variable so that (F2)
        # and every (S1) row are two-nonzero rows instead of |Gamma(s)| of them.
        self.seated: dict[str, object] = {}
        self.tail: dict[tuple[str, int], object] = {}
        s1_rows = s2_rows = f5_rows = 0
        for program in market.programs:
            program_id = program.program_id
            order = orders[program_id]
            interested = {i for (i, other) in self.y if other == program_id}
            if set(order) != interested or len(order) != len(interested):
                raise RuntimeError(
                    f"Priority order for {program_id!r} is not Gamma(s)."
                )
            seats = m.addVar(lb=0.0, ub=capacity[program_id], name=f"T_{program_id}")
            m.addConstr(  # (F2)
                seats == gp.quicksum(self.y[(i, program_id)] for i in order)
            )
            self.seated[program_id] = seats

            running = 0.0
            previous_z = None
            for position, i in enumerate(order):
                y = self.y[(i, program_id)]
                z = self.z[(i, program_id)]
                if previous_z is not None:
                    m.addConstr(z <= previous_z)  # (F5)
                    f5_rows += 1
                previous_z = z
                if "S1" in self.extras:
                    # sum_i' y[i', s] >= q_s (1 - z[i, s])
                    m.addConstr(
                        seats + capacity[program_id] * z >= capacity[program_id]
                    )
                    s1_rows += 1
                if "S2" in self.extras:
                    access = self.access_of[(i, program_id)]
                    rhs = capacity[program_id] * access
                    m.addConstr(
                        capacity[program_id] * self.prefix[(i, program_id)] + running
                        >= rhs
                    )
                    s2_rows += 1
                    nxt = m.addVar(
                        lb=0.0,
                        ub=capacity[program_id],
                        name=f"B_{program_id}_{position}",
                    )
                    m.addConstr(nxt == running + y)
                    self.tail[(program_id, position + 1)] = nxt
                    running = nxt

        self.counts["seated_vars"] = len(self.seated)
        self.counts["s2_prefix_vars"] = len(self.tail)
        self.counts["f5_rows"] = f5_rows
        self.counts["s1_rows"] = s1_rows
        self.counts["s2_rows"] = s2_rows

        self.denominator = float(CP_SAT_SCALE)
        m.setObjective(objective, GRB.MAXIMIZE)

    # -- gate switching ------------------------------------------------------ #

    def fix_zoning(self, assignment: dict[int, int]) -> None:
        for (zone, node), var in self.x.items():
            value = 1.0 if assignment[node] == zone else 0.0
            _, high = self._x_bounds[(zone, node)]
            if value == 1.0 and high < 1.0:
                raise RuntimeError(
                    f"Zoning puts node {node} in zone {zone}, which the linear "
                    "contiguity block forbids."
                )
            var.LB, var.UB = value, value
        for node, zone in assignment.items():
            if (zone, node) not in self.x:
                raise RuntimeError(f"Zone {zone} is not a candidate for node {node}.")

    def free_zoning(self) -> None:
        for key, var in self.x.items():
            var.LB, var.UB = self._x_bounds[key]

    def set_integral(self) -> None:
        for var in self.y.values():
            var.VType = GRB.BINARY
        for var in self.z.values():
            var.VType = GRB.BINARY

    # -- solving ------------------------------------------------------------- #

    def optimize(self, *, seconds: float | None = None, method: int | None = None):
        self.model.Params.Method = -1 if method is None else method
        self.model.Params.TimeLimit = GRB.INFINITY if seconds is None else seconds
        start = time.perf_counter()
        self.model.optimize()
        elapsed = time.perf_counter() - start
        status = self.model.Status
        if status == GRB.OPTIMAL:
            pass
        elif status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            pass
        else:
            raise RuntimeError(f"solve ended with Gurobi status {status}")
        bound = None
        gap = None
        if self.model.IsMIP:
            bound = float(self.model.ObjBound) / self.denominator
            gap = float(self.model.MIPGap)
        return {
            "value": float(self.model.ObjVal) / self.denominator,
            "bound": bound,
            "gap": gap,
            "seconds": elapsed,
            "status": int(status),
        }

    def raw_value(self) -> float:
        """The incumbent's welfare in the oracle's own (unscaled) utilities.

        The objective uses ``scaled_utilities``, which are
        ``max(1, round(100 u))``, so dividing by 100 is off by the per-pair
        rounding.  `SaaOracle` reports unscaled utility, and this recomputation
        separates rounding drift from formulation slack.
        """
        total = 0.0
        for index, student in enumerate(self.market.students):
            for rank, program_id in enumerate(student.programs):
                total += student.utilities[rank] * self.y[(index, program_id)].X
        return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="benchmark/configs/priced_access_seeds.yaml"
    )
    parser.add_argument("--centroids", default="6-zone-3")
    parser.add_argument("--seed", type=int, default=1, help="STB sample base seed")
    parser.add_argument("--zonings", type=int, default=8)
    parser.add_argument("--enumerate-pool", type=int, default=400)
    parser.add_argument("--enumerate-seconds", type=float, default=180.0)
    parser.add_argument("--gate-a-zonings", type=int, default=8)
    parser.add_argument("--gate-a-spot-check", type=int, default=2)
    parser.add_argument("--mip-seconds", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-gate-a", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    problem, market = build_individual_instance(args.config, args.centroids)
    sample = sample_school_preferences(market, 1, "STB", args.seed)[0]
    gamma = sum(len(student.programs) for student in market.students)
    print(
        f"nodes={len(problem.nodes)} Z={problem.Z} programs={len(market.programs)} "
        f"citywide={sum(1 for p in market.programs if p.citywide)} "
        f"students={len(market.students)} |Gamma|={gamma} "
        f"mean_list={gamma / len(market.students):.3f} lottery_seed={sample.seed}",
        flush=True,
    )
    print(
        f"first_choice_bound={first_choice_upper_bound(market.students):,.2f} "
        f"transport_bound={transport_upper_bound(market.programs, market.students):,.2f}",
        flush=True,
    )

    start = time.perf_counter()
    pool = feasible_zonings(
        problem, args.enumerate_pool, args.enumerate_seconds, args.workers
    )
    zonings = spread_zonings(pool, args.zonings)
    low, mean, high = hamming_summary(zonings)
    print(
        f"enumerated {len(pool)} feasible zonings in "
        f"{time.perf_counter() - start:.1f}s; kept {len(zonings)} with pairwise "
        f"Hamming distance min={low} mean={mean:.1f} max={high}",
        flush=True,
    )

    # Ground truth, plus the SaaOracle LP value it was supposed to be.
    matcher = DeferredAcceptance(problem, market, sample)
    oracle = SaaOracle(market, sample, 0, problem, workers=args.workers)
    da_welfare, oracle_welfare, seated_counts = [], [], []
    for index, assignment in enumerate(zonings):
        welfare, seated = matcher.solve(assignment)
        da_welfare.append(welfare)
        seated_counts.append(seated)
        oracle_welfare.append(oracle.solve(assignment).welfare)
        print(
            f"  zoning {index}: DA welfare {welfare:12,.4f} seated={seated:5d}"
            f"   SaaOracle LP {oracle_welfare[-1]:12,.4f}"
            f"   over-report {oracle_welfare[-1] - welfare:9,.4f}",
            flush=True,
        )
    over = [lp - da for lp, da in zip(oracle_welfare, da_welfare)]
    print(
        f"  SaaOracle LP over-report: mean={statistics.mean(over):,.2f} "
        f"min={min(over):,.2f} max={max(over):,.2f}  "
        "(the oracle's welfare field is the continuous stable-admissions LP, "
        "not the realised matching)",
        flush=True,
    )

    records: list[dict] = []
    for label, extras in ARMS:
        start = time.perf_counter()
        arm = CutoffModel(problem, market, sample, extras)
        build_seconds = time.perf_counter() - start
        rows, columns = arm.model.NumConstrs, arm.model.NumVars
        print(
            f"\n=== {label} === rows={rows:,} cols={columns:,} "
            f"nonzeros={arm.model.NumNZs:,} build={build_seconds:.1f}s",
            flush=True,
        )
        print(
            "    " + "  ".join(f"{k}={v:,}" for k, v in sorted(arm.counts.items())),
            flush=True,
        )

        # GATE C first: the model as built already has x free and continuous.
        root = arm.optimize(method=2)
        print(
            f"  GATE C  joint root LP = {root['value']:12,.2f}"
            f"   lp={root['seconds']:7.1f}s",
            flush=True,
        )

        # GATE B: same rows, x pinned to each enumerated zoning.
        gate_b = []
        for index, assignment in enumerate(zonings):
            arm.fix_zoning(assignment)
            result = arm.optimize()
            slack = result["value"] - da_welfare[index]
            gate_b.append(
                {
                    "zoning": index,
                    "lp": result["value"],
                    "da": da_welfare[index],
                    "slack": slack,
                    "oracle_lp": oracle_welfare[index],
                    "seconds": result["seconds"],
                }
            )
            flag = "  *** NEGATIVE (validity violation)" if slack < -1e-4 else ""
            print(
                f"  GATE B  zoning {index}: LP={result['value']:12,.2f} "
                f"DA={da_welfare[index]:12,.2f} slack={slack:9,.2f}"
                f"  oracleLP={oracle_welfare[index]:12,.2f}"
                f"  ({result['seconds']:5.1f}s){flag}",
                flush=True,
            )
        slacks = [row["slack"] for row in gate_b]
        print(
            f"  GATE B  slack mean={statistics.mean(slacks):,.2f} "
            f"min={min(slacks):,.2f} max={max(slacks):,.2f}",
            flush=True,
        )

        gate_a = []
        # The two arms whose exactness is in question get every zoning; the
        # (S2) arms are exact by construction and get a bounded spot check.
        gate_a_count = (
            args.gate_a_zonings if label in GATE_A_ARMS else args.gate_a_spot_check
        )
        if not args.skip_gate_a and gate_a_count > 0:
            arm.set_integral()
            for index, assignment in enumerate(zonings[:gate_a_count]):
                arm.fix_zoning(assignment)
                result = arm.optimize(seconds=args.mip_seconds)
                raw = arm.raw_value()
                gate_a.append(
                    {
                        "zoning": index,
                        "mip": result["value"],
                        "mip_raw_utility": raw,
                        "mip_bound": result["bound"],
                        "mip_gap": result["gap"],
                        "da": da_welfare[index],
                        "difference": result["value"] - da_welfare[index],
                        "raw_difference": raw - da_welfare[index],
                        "seconds": result["seconds"],
                        "status": result["status"],
                    }
                )
                flag = ""
                if result["status"] == GRB.TIME_LIMIT:
                    flag += f"  [TIME LIMIT, gap={result['gap']:.2%}]"
                # ``mip`` is in rounded scaled utility so it can disagree with
                # DA in the third decimal for free; ``raw`` cannot, and which
                # side it lands on says which way the formulation is wrong.
                if raw - da_welfare[index] < -1e-3:
                    flag += "  *** MIP BELOW DA: formulation excludes the true matching"
                elif raw - da_welfare[index] > 1e-3:
                    flag += (
                        "  *** MIP ABOVE DA: formulation admits an unstable matching"
                    )
                print(
                    f"  GATE A  zoning {index}: MIP={result['value']:12,.4f} "
                    f"raw={raw:12,.4f} DA={da_welfare[index]:12,.4f} "
                    f"diff={result['value'] - da_welfare[index]:9,.4f} "
                    f"raw_diff={raw - da_welfare[index]:9,.4f}"
                    f"  ({result['seconds']:6.1f}s){flag}",
                    flush=True,
                )
            arm.free_zoning()

        records.append(
            {
                "arm": label,
                "extras": list(extras),
                "rows": rows,
                "columns": columns,
                "nonzeros": arm.model.NumNZs,
                "build_seconds": build_seconds,
                "root_bound": root["value"],
                "root_seconds": root["seconds"],
                "counts": dict(arm.counts),
                "gate_b": gate_b,
                "gate_a": gate_a,
            }
        )
        arm.model.dispose()

    print("\n" + "=" * 100)
    print(
        f"{'arm':16s} {'root LP':>12s} {'rows':>10s} {'cols':>10s} "
        f"{'lp s':>8s} {'B slack mean':>13s} {'B slack min':>12s} "
        f"{'A max |diff|':>13s} {'A n':>4s}"
    )
    for record in records:
        slacks = [row["slack"] for row in record["gate_b"]]
        diffs = [abs(row["raw_difference"]) for row in record["gate_a"]]
        print(
            f"{record['arm']:16s} {record['root_bound']:12,.2f} "
            f"{record['rows']:10,} {record['columns']:10,} "
            f"{record['root_seconds']:8.1f} {statistics.mean(slacks):13,.2f} "
            f"{min(slacks):12,.2f} "
            f"{(max(diffs) if diffs else float('nan')):13,.6f} "
            f"{len(diffs):4d}"
        )
    print("-" * 100)
    print(
        f"  DA welfare of the {len(zonings)} tested zonings: "
        f"min {min(da_welfare):,.2f}  mean {statistics.mean(da_welfare):,.2f}  "
        f"max {max(da_welfare):,.2f}"
    )
    for name, value in REFERENCES:
        print(f"  reference: {name:34s} {value:12,.2f}")
    print("=" * 100)

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "instance": {
                        "nodes": len(problem.nodes),
                        "zones": problem.Z,
                        "programs": len(market.programs),
                        "students": len(market.students),
                        "gamma": gamma,
                        "lottery_seed": sample.seed,
                    },
                    "da_welfare": da_welfare,
                    "da_seated": seated_counts,
                    "saa_oracle_lp": oracle_welfare,
                    "references": dict(REFERENCES),
                    "arms": records,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
