#!/usr/bin/env python3
"""Where does the joint MID model's root bound come from, and what closes it?

The welfare model is already nearly tight once the zoning is fixed:
`MidCpSatSolver.tighten_min_recurrence` takes its relaxation from +38.6% to
+0.8% with `x` fixed. Jointly it is not, and this script measures why by
rebuilding the same model in Gurobi -- so the relaxation is inspectable rather
than buried inside CP-SAT -- and reading the root bound off as families are
added. Contiguity is included in its linear form and nothing here is a
heuristic, so every arm is a valid upper bound on MID welfare and the arms are
directly comparable.

Two reference points frame the numbers. Unrestricted DA welfare, every node in
one zone, is 18,219.41; the best zoning anyone has found scores about 14,950.
So confinement to zones is worth roughly 3,270, and a bound that cannot see
confinement cannot get below 18,219.

The measured diagnosis, which is *not* the obvious one. The aggregated model
links welfare to the zoning through the co-zoning product
``a[v, u] = sum_z x[z, v] x[z, u]``, and the standard story is that spreading
`x` evenly over the `Z` zones lets every conjunction reach `1/Z`, every
`a[v, u]` reach 1, and the bound collapse to the unrestricted transportation
value. That is not what happens here. At the root optimum the mean co-zoning
weight is 2.34 of an available 10.67 and only 14.5% of the access pairs the
welfare model uses are near 1, at mean ``a = 0.68`` -- the relaxation does not
want broad access. It wants *fractional* access, because a fractional ``a``
buys a pro-rata slice of a school no integral zoning could hand out, and the
capacity rows only limit the total. Every student gets a sliver of a popular
school.

That distinction decides what can possibly help. The leaking point is a
perfectly consistent co-zoning matrix, so no valid inequality written purely in
``a``-space can cut it off, and the families below are here to demonstrate
that rather than to fix it:

knapsack
    ``sum_u m(u) a[v, u] <= M`` for every node ``v``, with ``m(u)`` the number
    of schools at ``u`` and ``M`` the per-zone school ceiling the core model
    already imposes. Valid, and it does bind -- the row reaches ``M`` exactly --
    but preference lists average 3.94 entries, so a student needs about four
    schools and the ten-school ceiling has nothing to say about welfare.

disaggregated
    ``sum_u m(u) both[v, u, z] <= M x[z, v]`` for every node and zone. Summing
    over ``z`` recovers the aggregated form, so this dominates it; the core
    model only yields the aggregated form multiplied by ``Z``, which is
    vacuous.

triangle
    ``a[u,v] + a[v,w] - a[u,w] <= 1`` on school-school-student triples.

The `by-zone` arms are the alternative the diagnosis points at: drop ``a``
entirely and replicate the recurrence per zone, so the coupling to ``x`` is
linear and a program's seats are consumed only by the zone that holds it. See
:class:`ZoneDisaggregatedModel`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from benchmark.slurm import create_plan
from optimization.data import contiguity
from optimization.data.mid import build_mid_market, preprocess_mid_market
from optimization.levels import LevelSpec
from optimization.mid_oracle import cutoff_upper_bounds
from optimization.solvers.balance import balance_constraints, balance_terms
from optimization.welfare_bounds import first_choice_upper_bound, transport_upper_bound


def problem_market_types(market):
    """The market's student types; a seam for the probe's own bookkeeping."""
    return market.types


FAMILIES = (
    "knapsack",
    "disaggregated",
    "triangle",
    "lower",
    "lower_dis",
    "triangle_dis",
)


def build_instance(config_path: str, centroids: str):
    plan = create_plan(config_path)
    config = plan.tasks[0].optimization_config()
    config.centroids_type = centroids
    target = LevelSpec.parse(config.levels[-1])
    dataset = config.make_dataset()
    problem = dataset.problem_for(target)
    problem.overage = -1.0
    problem.shortage = -1.0
    problem.boundary_prop = -1.0
    market = preprocess_mid_market(build_mid_market(problem, dataset.config), problem)
    return problem, market


class RootModel:
    """The joint zoning + finite-grid MID welfare relaxation, in Gurobi."""

    def __init__(self, problem, market, lottery_scale: int, families: set[str]):
        self.problem = problem
        self.market = market
        self.scale = lottery_scale
        self.families = families
        self.model = gp.Model()
        self.model.Params.OutputFlag = 0
        self.model.ModelSense = GRB.MAXIMIZE
        self.counts: dict[str, int] = {}
        self.triangle_limit = 400_000
        self.ranked_school_nodes: dict[int, set[int]] = {}
        for student_type in problem_market_types(market):
            for program_id in student_type.programs:
                program = market.program_by_id[program_id]
                if program.citywide or program.school_node is None:
                    continue
                self.ranked_school_nodes.setdefault(student_type.node, set()).add(
                    program.school_node
                )
        self._zoning()
        self._access()
        self._welfare()
        self._extras()

    # -- zoning ------------------------------------------------------------ #

    def _zoning(self) -> None:
        problem, m = self.problem, self.model
        self.x = {
            (z, v): m.addVar(lb=0.0, ub=1.0, name=f"x_{z}_{v}")
            for v in problem.nodes
            for z in problem.candidate_zones(v)
        }
        for v in problem.nodes:
            m.addConstr(
                gp.quicksum(self.x[(z, v)] for z in problem.candidate_zones(v)) == 1.0
            )
        # Centroids are pinned to their own zone (radius 0).
        for zone, centroid in enumerate(problem.centroids):
            for other in problem.candidate_zones(centroid):
                self.x[(other, centroid)].LB = 1.0 if other == zone else 0.0
                self.x[(other, centroid)].UB = 1.0 if other == zone else 0.0

        # Contiguity, in the linear form the Boolean model uses:
        # x[z, v] <= sum of x[z, .] over v's supported neighbours.
        supports = contiguity.contiguity_supports(
            problem.G, problem.centroids, problem.centroid_school_ids,
            problem.candidate_zones,
        )
        closer = contiguity.closer_supports(
            problem.G, problem.centroids, problem.centroid_school_ids,
            problem.candidate_zones,
        )
        for (v, z), support in supports.items():
            if (z, v) not in self.x:
                continue
            if not closer[(v, z)] or not support:
                self.x[(z, v)].UB = 0.0
                continue
            m.addConstr(
                self.x[(z, v)]
                <= gp.quicksum(self.x[(z, n)] for n in support if (z, n) in self.x)
            )

        for z in range(problem.Z):
            nodes = [v for v in problem.nodes if z in problem.candidate_zones(v)]
            for constraint in balance_constraints(problem):
                lower, upper = balance_terms(problem, constraint, z, nodes)
                if lower:
                    m.addConstr(
                        gp.quicksum(c * self.x[(zz, v)] for c, zz, v in lower) >= 0.0
                    )
                if upper:
                    m.addConstr(
                        gp.quicksum(c * self.x[(zz, v)] for c, zz, v in upper) <= 0.0
                    )

        self.schools = {v: problem.num_schools(v) for v in problem.nodes}
        total_schools = sum(self.schools.values())
        self.school_ceiling = total_schools / problem.Z + 1.0
        # A zone's school count is an integer in [avg - 1, avg + 1], so the
        # binding floor is the ceiling of the lower end.
        self.school_floor = math.ceil(max(0.0, total_schools / problem.Z - 1.0))
        if total_schools:
            for z in range(problem.Z):
                nodes = [v for v in problem.nodes if z in problem.candidate_zones(v)]
                terms = gp.quicksum(
                    self.schools[v] * self.x[(z, v)] for v in nodes
                )
                m.addConstr(terms >= max(0.0, total_schools / problem.Z - 1.0))
                m.addConstr(terms <= self.school_ceiling)

    # -- access ------------------------------------------------------------ #

    def _pair_needed(self):
        """Node pairs that need a co-zoning variable.

        The welfare model needs every (student node, school node) pair a type
        actually ranks.  The school-count families additionally need every
        (node, school node) pair: a band on a node's whole school row says
        nothing unless the whole row is present, and the lower side would be
        outright infeasible if summed over a subset.
        """
        problem, market = self.problem, self.market
        needed = {
            (student_type.node, market.program_by_id[program_id].school_node)
            for student_type in market.types
            for program_id in student_type.programs
            if not market.program_by_id[program_id].citywide
        }
        if self.families & {
            "knapsack",
            "disaggregated",
            "lower",
            "lower_dis",
            "triangle_dis",
        }:
            school_nodes = {v for v, count in self.schools.items() if count}
            needed |= {
                (v, u) for v in problem.nodes for u in school_nodes
            }
        return {
            (min(a, b), max(a, b)) for a, b in needed if a is not None and b != a
        }

    def _access(self) -> None:
        problem, m = self.problem, self.model
        self.a: dict[tuple[int, int], object] = {}
        self.both: dict[tuple[int, int, int], object] = {}
        for u, v in sorted(self._pair_needed()):
            common = sorted(
                problem.candidate_zones(u) & problem.candidate_zones(v)
            )
            if not common:
                self.a[(u, v)] = 0.0
                continue
            pair = m.addVar(lb=0.0, ub=1.0, name=f"a_{u}_{v}")
            joints = []
            for z in common:
                xu, xv = self.x[(z, u)], self.x[(z, v)]
                both = m.addVar(lb=0.0, ub=1.0, name=f"b_{u}_{v}_{z}")
                m.addConstr(both <= xu)
                m.addConstr(both <= xv)
                m.addConstr(both >= xu + xv - 1.0)
                self.both[(u, v, z)] = both
                joints.append(both)
            m.addConstr(pair == gp.quicksum(joints))
            self.a[(u, v)] = pair
        self.counts["access_pairs"] = sum(
            1 for value in self.a.values() if not isinstance(value, float)
        )
        self.counts["access_joints"] = len(self.both)

    def _access_of(self, u: int, v: int):
        if u == v:
            return 1.0
        return self.a.get((min(u, v), max(u, v)), 0.0)

    # -- welfare ----------------------------------------------------------- #

    def _welfare(self) -> None:
        market, m, L = self.market, self.model, self.scale
        bounds = cutoff_upper_bounds(market, L)
        self.cutoffs = {
            program.program_id: m.addVar(
                lb=0.0, ub=min(float(bounds[program.program_id]), float(L) * 40),
                name=f"P_{program.program_id}",
            )
            for program in market.programs
        }
        self.counts["cutoffs_fixed_zero"] = sum(
            1 for program in market.programs if bounds[program.program_id] <= 0
        )

        thresholds: dict[tuple[str, int], object] = {}

        def threshold(program_id: str, priority: int):
            key = (program_id, priority)
            if key not in thresholds:
                var = m.addVar(lb=0.0, ub=float(L), name=f"t_{program_id}_{priority}")
                # t = min(L, max(P - priority*L, 0)); the `>=` side is exact
                # under maximisation because welfare falls as t rises.
                m.addConstr(var >= self.cutoffs[program_id] - priority * L)
                thresholds[key] = var
            return thresholds[key]

        capacity_terms: dict[str, list] = {
            program.program_id: [] for program in market.programs
        }
        objective = gp.LinExpr()
        n_mass = 0
        for student_type in market.types:
            previous = None
            for rank, (program_id, priority) in enumerate(
                zip(student_type.programs, student_type.priorities)
            ):
                program = market.program_by_id[program_id]
                thresh = threshold(program_id, priority)
                if program.citywide:
                    effective = thresh
                else:
                    access = self._access_of(student_type.node, program.school_node)
                    if isinstance(access, float) and access == 0.0:
                        effective = float(L)
                    elif isinstance(access, float) and access == 1.0:
                        effective = thresh
                    else:
                        effective = m.addVar(lb=0.0, ub=float(L), name=f"e_{rank}")
                        # e = t if co-zoned else L.  Both lower bounds together
                        # are the McCormick-tight envelope, and only the lower
                        # side can bind: welfare falls as e rises.
                        m.addConstr(effective >= thresh)
                        m.addConstr(effective >= L * (1.0 - access))
                remaining = m.addVar(lb=0.0, ub=float(L), name=f"R_{rank}")
                previous_expr = float(L) if previous is None else previous
                m.addConstr(remaining <= previous_expr)
                m.addConstr(remaining <= effective)
                # min is concave, so the two `<=` sides alone let the relaxation
                # manufacture mass at high-utility ranks.  For a, b in [0, L],
                # min(a, b) >= a + b - L, i.e. you cannot take more mass at a
                # rank than your acceptance chance there.
                m.addConstr(remaining >= previous_expr + effective - float(L))
                mass = previous_expr - remaining
                capacity_terms[program_id].append(student_type.count * mass)
                objective += student_type.scaled_utility_sums[rank] * mass
                previous = remaining
                n_mass += 1
        self.counts["mass_vars"] = n_mass

        for program in market.programs:
            terms = capacity_terms[program.program_id]
            if terms:
                m.addConstr(gp.quicksum(terms) <= float(L) * program.capacity)

        self.denominator = float(L) * float(market.utility_scale)
        m.setObjective(objective, GRB.MAXIMIZE)

    # -- the families under test ------------------------------------------- #

    def _extras(self) -> None:
        problem, m = self.problem, self.model
        school_nodes = sorted(v for v, count in self.schools.items() if count)
        added = {name: 0 for name in FAMILIES}

        if "knapsack" in self.families:
            for v in problem.nodes:
                terms = gp.quicksum(
                    self.schools[u] * self._access_of(v, u)
                    for u in school_nodes
                    if not isinstance(self._access_of(v, u), float)
                )
                fixed = sum(
                    self.schools[u] * self._access_of(v, u)
                    for u in school_nodes
                    if isinstance(self._access_of(v, u), float)
                )
                m.addConstr(terms <= self.school_ceiling - fixed)
                added["knapsack"] += 1

        if "disaggregated" in self.families:
            for v in problem.nodes:
                for z in problem.candidate_zones(v):
                    terms = gp.quicksum(
                        self.schools[u] * self.both[(min(v, u), max(v, u), z)]
                        for u in school_nodes
                        if u != v and (min(v, u), max(v, u), z) in self.both
                    )
                    own = self.schools[v]
                    m.addConstr(
                        terms <= (self.school_ceiling - own) * self.x[(z, v)]
                    )
                    added["disaggregated"] += 1

        if "lower" in self.families or "lower_dis" in self.families:
            # The school-count band has a *lower* side too: every zone holds at
            # least ceil(58/6 - 1) = 9 schools.  That is the side that bites
            # here.  The root optimum grants each vertex only 2.34 schools'
            # worth of co-zoning -- it dodges confinement by giving every
            # student a private fractional set of just the three or four
            # schools they rank, so no coherent zone ever forms and capacity is
            # spread across all 58 schools instead of concentrating into six
            # groups.  Forcing the row up to 9 makes the LP commit.
            floor = max(0.0, self.school_floor)
            for v in problem.nodes:
                own = self.schools[v]
                if "lower" in self.families:
                    terms = gp.quicksum(
                        self.schools[u] * self._access_of(v, u)
                        for u in school_nodes
                        if u != v and not isinstance(self._access_of(v, u), float)
                    )
                    fixed = sum(
                        self.schools[u] * self._access_of(v, u)
                        for u in school_nodes
                        if u != v and isinstance(self._access_of(v, u), float)
                    )
                    m.addConstr(terms >= floor - own - fixed)
                    added["lower"] += 1
                if "lower_dis" in self.families:
                    for z in problem.candidate_zones(v):
                        terms = gp.quicksum(
                            self.schools[u] * self.both[(min(v, u), max(v, u), z)]
                            for u in school_nodes
                            if u != v and (min(v, u), max(v, u), z) in self.both
                        )
                        m.addConstr(terms >= (floor - own) * self.x[(z, v)])
                        added["lower_dis"] += 1

        if "triangle_dis" in self.families:
            # Transitivity is NOT implied by the per-zone conjunction
            # linearization: x = 1/6 everywhere with both[v,u,z] = both[v,u',z]
            # = 1/6 and both[u,u',z] = 0 gives a[v,u] = a[v,u'] = 1 while
            # a[u,u'] = 0.  The disaggregated form dominates the aggregated one,
            # since summing it over z recovers it.
            # All (node, school pair, zone) triples would be ~4M rows, so
            # restrict to the school pairs a node's own students actually rank:
            # those are the ones the relaxation wants to violate.
            limit = self.triangle_limit
            for v in problem.nodes:
                row = sorted(self.ranked_school_nodes.get(v, ()))
                if len(row) < 2:
                    continue
                for u, w in itertools.combinations(row, 2):
                    if added["triangle_dis"] >= limit:
                        break
                    for z in problem.candidate_zones(v):
                        key_vu = (min(v, u), max(v, u), z)
                        key_vw = (min(v, w), max(v, w), z)
                        key_uw = (min(u, w), max(u, w), z)
                        if key_vu not in self.both or key_vw not in self.both:
                            continue
                        side = self.both.get(key_uw, 0.0)
                        m.addConstr(
                            self.both[key_vu] + self.both[key_vw] - side
                            <= self.x[(z, v)]
                        )
                        added["triangle_dis"] += 1

        if "triangle" in self.families:
            # School-school-student triples: the pair graph is otherwise close
            # to bipartite, and a bipartite graph has no triangles to bind on.
            for v in problem.nodes:
                row = [
                    u for u in school_nodes
                    if u != v and not isinstance(self._access_of(v, u), float)
                ]
                for u, w in itertools.combinations(row, 2):
                    side = self._access_of(u, w)
                    if isinstance(side, float):
                        continue
                    a_vu, a_vw = self._access_of(v, u), self._access_of(v, w)
                    m.addConstr(a_vu + a_vw - side <= 1.0)
                    added["triangle"] += 1
        self.counts.update(added)

    # -- solve -------------------------------------------------------------- #

    def relax(self) -> float:
        self.model.Params.Method = 2
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"root LP status {self.model.Status}")
        return float(self.model.ObjVal) / self.denominator



class ZoneDisaggregatedModel:
    """The same welfare model, replicated per zone instead of per co-zoning pair.

    The aggregated model links welfare to the zoning through the co-zoning
    product ``a[v, u] = sum_z x[z, v] x[z, u]``, and that link is where the
    relaxation leaks: a fractional ``a`` hands a student a pro-rata slice of a
    school no integral zoning could give them, and the measured optimum sits at
    mean ``a = 0.68`` rather than at the all-access corner.  No valid inequality
    written purely in ``a`` can close that, because the leaking point is
    perfectly consistent as a co-zoning matrix.

    So drop ``a`` entirely.  Carry one copy of each type's cutoff recursion per
    zone, and let

        mass of type t in zone z  <=  L x[z, v_t]                 (membership)
        seats of program s in zone z  <=  L q_s x[z, u_s]         (supply)
        effective_{t,r,z} >= L (1 - x[z, u_s])                    (access)

    do the coupling.  Every term is linear in ``x`` -- there is no product left
    to convexify -- and the supply family is what the aggregated model cannot
    say at all: a program's seats are consumed only by the zone that contains
    it, so the LP can no longer sell the same popular school to all six zones.

    Exact at integrality: for the one zone containing ``v_t`` the recursion is
    the finite-grid recursion verbatim, and every other zone is forced to zero
    mass by the membership constraint.
    """

    def __init__(self, problem, market, lottery_scale: int, *, disaggregate_supply=True):
        self.problem = problem
        self.market = market
        self.scale = lottery_scale
        self.disaggregate_supply = disaggregate_supply
        self.model = gp.Model()
        self.model.Params.OutputFlag = 0
        self.model.ModelSense = GRB.MAXIMIZE
        self.counts: dict[str, int] = {}
        # Reuse the zoning block unchanged; only the welfare block differs.
        RootModel._zoning(self)
        self._welfare()

    def _welfare(self) -> None:
        market, m, L = self.market, self.model, self.scale
        problem = self.problem
        bounds = cutoff_upper_bounds(market, L)
        self.cutoffs = {
            program.program_id: m.addVar(
                lb=0.0, ub=min(float(bounds[program.program_id]), float(L) * 40),
                name=f"P_{program.program_id}",
            )
            for program in market.programs
        }
        thresholds: dict[tuple[str, int], object] = {}

        def threshold(program_id: str, priority: int):
            key = (program_id, priority)
            if key not in thresholds:
                var = m.addVar(lb=0.0, ub=float(L))
                m.addConstr(var >= self.cutoffs[program_id] - priority * L)
                thresholds[key] = var
            return thresholds[key]

        effective: dict[tuple[int, str, int, int], object] = {}

        def effective_of(node: int, program, priority: int, zone: int):
            """``threshold`` if both the student and ``program`` sit in ``zone``.

            Gating on the *student's* membership as well as the school's is what
            makes the replication valid.  Leave a rank ungated -- as a citywide
            program or a school on the student's own node invites -- and the
            copies in the five zones the student does not occupy still run the
            recurrence, so the only way to satisfy a cap on their mass is to
            drive that program's cutoff to the top of the interval, which
            destroys its welfare for everyone.  Measured: ungated citywide
            programs put the fixed-``x`` value at 7,069 against a true 10,761.
            """
            thresh = threshold(program.program_id, priority)
            key = (node, program.program_id, priority, zone)
            if key in effective:
                return effective[key]
            if (zone, node) not in self.x:
                effective[key] = float(L)      # student cannot be in this zone
                return effective[key]
            gates = [self.x[(zone, node)]]
            if not program.citywide and program.school_node != node:
                school_node = program.school_node
                if (zone, school_node) not in self.x:
                    effective[key] = float(L)  # school cannot be in this zone
                    return effective[key]
                gates.append(self.x[(zone, school_node)])
            var = m.addVar(lb=0.0, ub=float(L))
            m.addConstr(var >= thresh)
            for gate in gates:
                m.addConstr(var >= L * (1.0 - gate))
            effective[key] = var
            return var

        # program -> zone -> capacity terms
        supply: dict[str, dict[int, list]] = {
            program.program_id: {} for program in market.programs
        }
        objective = gp.LinExpr()
        n_mass = 0
        for student_type in market.types:
            node = student_type.node
            for zone in sorted(problem.candidate_zones(node)):
                previous = None
                for rank, (program_id, priority) in enumerate(
                    zip(student_type.programs, student_type.priorities)
                ):
                    program = market.program_by_id[program_id]
                    value = effective_of(node, program, priority, zone)
                    remaining = m.addVar(lb=0.0, ub=float(L))
                    previous_expr = float(L) if previous is None else previous
                    m.addConstr(remaining <= previous_expr)
                    m.addConstr(remaining <= value)
                    m.addConstr(remaining >= previous_expr + value - float(L))
                    mass = previous_expr - remaining
                    supply[program_id].setdefault(zone, []).append(
                        student_type.count * mass
                    )
                    objective += student_type.scaled_utility_sums[rank] * mass
                    previous = remaining
                    n_mass += 1
                if previous is not None:
                    # Mass conservation, and the family that does the real
                    # tightening.  A type's mass in zone ``z`` is
                    # ``L - R_last``, so this caps it at ``L x[z, node]`` and,
                    # summed over zones, at ``L`` in total.  Gating already
                    # forces every ``R`` to ``L`` when ``x[z, node] = 0``, so
                    # unlike the ungated version this puts no pressure at all
                    # on the cutoffs -- it is satisfied there for free.
                    m.addConstr(previous >= L * (1.0 - self.x[(zone, node)]))
        self.counts["mass_vars"] = n_mass
        self.counts["effective_vars"] = sum(
            1 for value in effective.values() if not isinstance(value, float)
        )

        for program in market.programs:
            per_zone = supply[program.program_id]
            if not per_zone:
                continue
            total = gp.quicksum(
                term for terms in per_zone.values() for term in terms
            )
            m.addConstr(total <= float(L) * program.capacity)
            if not self.disaggregate_supply or program.citywide:
                continue
            school_node = program.school_node
            for zone, terms in per_zone.items():
                if (zone, school_node) not in self.x:
                    m.addConstr(gp.quicksum(terms) <= 0.0)
                    continue
                # Seats are consumed only by the zone holding the program.
                m.addConstr(
                    gp.quicksum(terms)
                    <= float(L) * program.capacity * self.x[(zone, school_node)]
                )

        self.denominator = float(L) * float(market.utility_scale)
        m.setObjective(objective, GRB.MAXIMIZE)

    relax = RootModel.relax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark/configs/priced_access_seeds.yaml")
    parser.add_argument("--centroids", default="6-zone-3")
    parser.add_argument("--lottery-scale", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    problem, market = build_instance(args.config, args.centroids)
    students = [
        student
        for student in getattr(market, "students", ())
    ]
    print(
        f"nodes={len(problem.nodes)} Z={problem.Z} programs={len(market.programs)} "
        f"types={len(market.types)} students={market.student_count}",
        flush=True,
    )

    arms: list[tuple[str, object]] = [
        ("core", ("aggregated", ())),
        ("knapsack", ("aggregated", ("knapsack",))),
        ("disaggregated", ("aggregated", ("disaggregated",))),
        ("knapsack+disaggregated", ("aggregated", ("knapsack", "disaggregated"))),
        (
            "knapsack+disaggregated+triangle",
            ("aggregated", ("knapsack", "disaggregated", "triangle")),
        ),
        ("lower", ("aggregated", ("lower",))),
        ("core+triangle", ("aggregated", ("triangle",))),
        ("lower_dis", ("aggregated", ("lower_dis",))),
        ("lower+lower_dis", ("aggregated", ("lower", "lower_dis"))),
        (
            "lower_dis+knapsack+disaggregated",
            ("aggregated", ("lower", "lower_dis", "knapsack", "disaggregated")),
        ),
        ("triangle_dis", ("aggregated", ("triangle_dis",))),
        (
            "everything",
            (
                "aggregated",
                (
                    "lower",
                    "lower_dis",
                    "knapsack",
                    "disaggregated",
                    "triangle",
                    "triangle_dis",
                ),
            ),
        ),
        ("by-zone (shared supply)", ("by_zone", False)),
        ("by-zone", ("by_zone", True)),
    ]
    records = []
    for label, (kind, spec) in arms:
        start = time.perf_counter()
        if kind == "aggregated":
            root = RootModel(problem, market, args.lottery_scale, set(spec))
        else:
            root = ZoneDisaggregatedModel(
                problem, market, args.lottery_scale, disaggregate_supply=spec
            )
        build_seconds = time.perf_counter() - start
        start = time.perf_counter()
        bound = root.relax()
        solve_seconds = time.perf_counter() - start
        record = {
            "arm": label,
            "root_bound": bound,
            "build_seconds": build_seconds,
            "solve_seconds": solve_seconds,
            "rows": root.model.NumConstrs,
            "columns": root.model.NumVars,
            **root.counts,
        }
        records.append(record)
        print(
            f"{label:38s} root={bound:12,.2f}  rows={record['rows']:>9,}"
            f"  cols={record['columns']:>9,}  build={build_seconds:6.1f}s"
            f"  lp={solve_seconds:7.1f}s",
            flush=True,
        )

    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
