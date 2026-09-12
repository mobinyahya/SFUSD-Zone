"""Global MIP pricing for one connected zone and its MID market.

The binary disjunctions encode the cutoff/minimum recurrence exactly. Lottery
masses are integral; utility coefficients are never rounded. Gurobi supplies a
global reduced-cost bound, including on interrupted solves.

:class:`ZonePricer` keeps one model alive per zone label. Between calls only
the branch fixings and the master duals change, and both are attribute updates
on a live model -- branch fixings are variable bounds, duals are objective
coefficients. Rebuilding meant discarding a 30,779-variable, 71,602-row model
per label per column-generation round per branch node, along with everything
Gurobi had learned about it.

REVISIT -- the recourse encoding is the expensive part of this model, and it
may be the wrong one. ``_welfare_model`` spends roughly 24,000 of the model's
27,140 integer variables on big-M ``_minimum`` disjunctions, one chain per
(type, preference-rank) pair, because finite-grid MID welfare is defined by
market-clearing *cutoffs* -- an equilibrium, not a maximization -- so the
cutoffs have to be variables and Proposition 7 is what turns maximizing over
them into least-cutoff welfare. Those chains are also what makes the
relaxation weak.

Pricing the SAA stable-admissions LP instead would leave the ~579 membership
binaries as the only integers: the recourse becomes continuous, and the
co-zoning access indicators are already pinned exactly by their three rows
given binary membership. Per-zone additivity still holds, provided every zone
is evaluated on the *same* scenario set. The costs are that a column's value
becomes the stable-admissions LP optimum rather than realized welfare (an
upper bound, so the master bound stays valid while the primal must still be
scored with ``finite_grid_oracle``), and that the certificate is then about a
fixed sample rather than a deterministic objective. See the note in
``optimization/branch_price.py`` -- this would buy column-generation rounds,
not fix the degeneracy that currently stalls them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from optimization.zone_columns import ZONE_FEASIBILITY_TOLERANCE

# Tight enough that the pricing bound can be compared against enumeration, and
# consistent with the 1e-6 slack the pool's own feasibility test allows.
PRICING_FEASIBILITY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PricingResult:
    status: str
    bound: float
    nodes: frozenset[int] = frozenset()
    score: float | None = None
    reduced_cost: float | None = None


def compatible(column, decisions):
    return all(
        (node in column.nodes) == value
        for (node, zone), value in decisions.items()
        if zone == column.zone
    )


def _minimum(m, a, b, upper, big_m):
    """``result == min(a, b)``, for integral ``a`` and ``b`` in ``[0, upper]``."""
    result = m.addVar(lb=0.0, ub=upper, vtype=GRB.INTEGER)
    select = m.addVar(vtype=GRB.BINARY)
    m.addConstr(result <= a)
    m.addConstr(result <= b)
    m.addConstr(result >= a - big_m * select)
    m.addConstr(result >= b - big_m * (1 - select))
    return result


def _welfare_model(m, market, membership, lottery_scale):
    """Maximizing over capacity-clearing cutoffs yields least-cutoff welfare.

    Omitted students and schools have zero access even when co-located. All
    preference prefixes are represented, without optimistic tail relaxations.
    """
    scale = lottery_scale
    programs = market.program_by_id
    priorities = {p: set() for p in programs}
    for student in market.types:
        for p, priority in zip(student.programs, student.priorities):
            priorities[p].add(priority)
    thresholds = {}
    for p in programs:
        upper = (max(priorities[p], default=0) + 1) * scale
        cutoff = m.addVar(lb=0.0, ub=upper, vtype=GRB.INTEGER)
        for priority in priorities[p]:
            positive = m.addVar(vtype=GRB.BINARY)
            raw = m.addVar(lb=0.0, ub=upper, vtype=GRB.INTEGER)
            delta = cutoff - priority * scale
            m.addConstr(raw >= delta)
            m.addConstr(raw <= delta + upper * (1 - positive))
            m.addConstr(raw <= upper * positive)
            thresholds[p, priority] = _minimum(m, raw, scale, scale, upper)
    access = {}
    capacities = {p: [] for p in programs}
    welfare = gp.LinExpr()
    for student in market.types:
        previous = gp.LinExpr(float(scale))
        for p, priority, utility in zip(
            student.programs, student.priorities, student.utility_sums
        ):
            program = programs[p]
            if program.citywide:
                raise ValueError("Exact DW pricing forbids citywide programs.")
            pair = (student.node, program.school_node)
            if pair not in access:
                both = m.addVar(vtype=GRB.BINARY)
                a, b = membership[pair[0]], membership[pair[1]]
                m.addConstr(both <= a)
                m.addConstr(both <= b)
                m.addConstr(both >= a + b - 1)
                access[pair] = both
            both = access[pair]
            threshold = thresholds[p, priority]
            effective = m.addVar(lb=0.0, ub=scale, vtype=GRB.INTEGER)
            m.addConstr(effective >= threshold)
            m.addConstr(effective <= threshold + scale * (1 - both))
            m.addConstr(effective >= scale * (1 - both))
            remaining = _minimum(m, previous, effective, scale, scale)
            mass = previous - remaining
            capacities[p].append(student.count * mass)
            welfare.add(mass, utility / scale)
            previous = gp.LinExpr(remaining)
    for p, terms in capacities.items():
        m.addConstr(gp.quicksum(terms) <= programs[p].capacity * scale)
    return welfare


def _add_geometry(m, pool, zone, x):
    """A connected, balanced, school-bounded zone for one label.

    ``problem.fixed`` and ``problem.candidates`` are written as rows rather
    than variable bounds on purpose: bounds are what a branch fixing uses, and
    :class:`ZonePricer` resets those between calls.
    """
    p = pool.problem
    for n, fixed in (p.fixed or {}).items():
        m.addConstr(x[n] == int(fixed == zone))
    for n, allowed in (p.candidates or {}).items():
        if zone not in allowed:
            m.addConstr(x[n] == 0)
    # A freely selected root supplies one unit to each selected vertex.
    # Flow is possible only on edges whose endpoints are selected.
    roots = {n: m.addVar(vtype=GRB.BINARY) for n in p.nodes}
    supplies = {n: m.addVar(lb=0.0, ub=p.A) for n in p.nodes}
    flows = {
        (a, b): m.addVar(lb=0.0, ub=p.A)
        for u, v in p.G.edges
        for a, b in ((u, v), (v, u))
    }
    m.addConstr(gp.quicksum(roots.values()) == 1)
    for n in p.nodes:
        m.addConstr(roots[n] <= x[n])
        m.addConstr(supplies[n] <= p.A * roots[n])
        m.addConstr(
            supplies[n]
            + gp.quicksum(flows[v, n] - flows[n, v] for v in p.G.neighbors(n))
            == x[n]
        )
    for (u, v), flow in flows.items():
        m.addConstr(flow <= p.A * x[u])
        m.addConstr(flow <= p.A * x[v])
    students = gp.quicksum(p.students(n) * x[n] for n in p.nodes)
    for row in pool.constraints:
        value = gp.quicksum(row.value(n) * x[n] for n in p.nodes)
        if row.lower_ratio is not None:
            m.addConstr(
                value >= row.lower_ratio * students - ZONE_FEASIBILITY_TOLERANCE
            )
        if row.upper_ratio is not None:
            m.addConstr(
                value <= row.upper_ratio * students + ZONE_FEASIBILITY_TOLERANCE
            )
    if pool.school_total:
        schools = gp.quicksum(p.num_schools(n) * x[n] for n in p.nodes)
        average = pool.school_total / p.Z
        m.addConstr(schools >= max(0, average - 1) - ZONE_FEASIBILITY_TOLERANCE)
        m.addConstr(schools <= average + 1 + ZONE_FEASIBILITY_TOLERANCE)


def _add_perimeter(m, problem, x):
    """Half the cut weight of the selected set.

    The cut indicators are continuous: the four rows pin each one to
    ``|x[u] - x[v]|`` whenever ``x`` is integral, so declaring them binary
    would only add redundant branching candidates.
    """
    edges = gp.LinExpr()
    for u, v in problem.G.edges:
        cut = m.addVar(lb=0.0, ub=1.0)
        m.addConstr(cut >= x[u] - x[v])
        m.addConstr(cut >= x[v] - x[u])
        m.addConstr(cut <= x[u] + x[v])
        m.addConstr(cut <= 2 - x[u] - x[v])
        edges.addTerms(problem.boundary_weight(u, v) / 2, cut)
    return edges


@dataclass
class _ZoneModel:
    """One label's live model, plus the handles needed to re-aim it."""

    model: gp.Model
    x: dict[int, gp.Var]
    score: gp.LinExpr
    perimeter: gp.LinExpr
    decided: set[int] = field(default_factory=set)


class ZonePricer:
    """Persistent global pricing models, one per zone label.

    Callable with :func:`price_zone`'s signature, so it drops straight into
    ``branch_and_price(pool, pricer=...)``. Close it when the search is done,
    or use it as a context manager.
    """

    def __init__(self, pool, *, model="grid", workers=0, seed=42):
        if model != "grid":
            raise ValueError(
                "Exact DW pricing currently certifies only finite-grid MID."
            )
        self.pool = pool
        self.workers = int(workers)
        self.seed = int(seed)
        self._env = gp.Env(params={"OutputFlag": 0})
        self._built: dict[int, _ZoneModel] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        for built in self._built.values():
            built.model.dispose()
        self._built.clear()
        self._env.close()

    @property
    def models_built(self) -> int:
        """How many models exist; one per label is the whole point."""
        return len(self._built)

    def _new_model(self, zone, fixed_nodes=None) -> _ZoneModel:
        pool = self.pool
        p = pool.problem
        m = gp.Model(f"zone_pricing_{zone}", env=self._env)
        m.Params.MIPGap = 0.0
        m.Params.MIPGapAbs = 0.0
        m.Params.FeasibilityTol = PRICING_FEASIBILITY_TOLERANCE
        m.Params.IntFeasTol = PRICING_FEASIBILITY_TOLERANCE
        m.Params.OptimalityTol = PRICING_FEASIBILITY_TOLERANCE
        m.Params.Seed = self.seed
        if self.workers:
            m.Params.Threads = self.workers
        x = {n: m.addVar(vtype=GRB.BINARY, name=f"member_{n}") for n in p.nodes}
        if fixed_nodes is not None:
            for n in p.nodes:
                m.addConstr(x[n] == int(n in fixed_nodes))
        else:
            _add_geometry(m, pool, zone, x)
        perimeter = _add_perimeter(m, p, x)
        score = (
            -perimeter
            if pool.market is None
            else _welfare_model(m, pool.market, x, pool.lottery_scale)
        )
        m.update()
        return _ZoneModel(m, x, score, perimeter)

    def _aim(self, built, zone, master, decisions, phase_one):
        """Point an existing model at this node's fixings and duals."""
        # Release the previous call's branch fixings before applying this
        # call's. Only bounds are touched, so the structural rows -- including
        # problem.fixed and problem.candidates -- survive untouched.
        for n in built.decided:
            built.x[n].LB, built.x[n].UB = 0.0, 1.0
        built.decided.clear()
        for (node, label), value in decisions.items():
            if label != zone:
                continue
            built.x[node].LB = built.x[node].UB = float(value)
            built.decided.add(node)

        objective = gp.LinExpr() if phase_one else gp.LinExpr(built.score)
        constant = 0.0
        if master is not None:
            for node, variable in built.x.items():
                objective.addTerms(-float(master.node_duals[node]), variable)
            objective.add(built.perimeter, -float(master.boundary_dual))
            constant = -float(master.zone_duals[zone])
        built.model.setObjective(objective + constant, GRB.MAXIMIZE)

    def __call__(
        self,
        pool,
        zone,
        master,
        decisions,
        *,
        deadline,
        model="grid",
        phase_one=False,
        fixed_nodes=None,
    ):
        """Globally optimize reduced welfare over ALL feasible zones for one label.

        `fixed_nodes` bypasses geometric constraints for scoring an already
        validated zone. Phase I prices geometric feasibility with zero original
        column cost.
        """
        if model != "grid":
            raise ValueError(
                "Exact DW pricing currently certifies only finite-grid MID."
            )
        if pool is not self.pool:
            raise ValueError("A ZonePricer is bound to the pool it was built for.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return PricingResult("TIME_LIMIT", math.inf)

        if fixed_nodes is not None:
            # A one-off scoring model with no geometry: not the shape the
            # search loop reuses, so it is not worth keeping.
            built = self._new_model(zone, fixed_nodes=fixed_nodes)
        else:
            built = self._built.get(zone)
            if built is None:
                built = self._built[zone] = self._new_model(zone)
        try:
            self._aim(built, zone, master, decisions, phase_one)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return PricingResult("TIME_LIMIT", math.inf)
            if math.isfinite(remaining):
                built.model.Params.TimeLimit = max(1e-3, remaining)
            else:
                built.model.Params.TimeLimit = GRB.INFINITY
            built.model.optimize()
            return self._read(built)
        finally:
            if fixed_nodes is not None:
                built.model.dispose()

    def _read(self, built) -> PricingResult:
        m = built.model
        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            # Every variable is bounded, so unboundedness is not a possibility
            # the presolver could be reporting here.
            return PricingResult("INFEASIBLE", -math.inf)
        if m.SolCount == 0:
            return PricingResult(
                "TIME_LIMIT" if m.Status == GRB.TIME_LIMIT else "ERROR", math.inf
            )
        nodes = frozenset(n for n, v in built.x.items() if v.X > 0.5)
        return PricingResult(
            "OPTIMAL" if m.Status == GRB.OPTIMAL else "FEASIBLE",
            m.ObjBound,
            nodes,
            built.score.getValue(),
            m.ObjVal,
        )


def price_zone(
    pool,
    zone,
    master,
    decisions,
    *,
    deadline,
    model="grid",
    phase_one=False,
    fixed_nodes=None,
):
    """One-shot pricing. Inside a loop, hold a :class:`ZonePricer` instead."""
    with ZonePricer(pool, model=model) as pricer:
        return pricer(
            pool,
            zone,
            master,
            decisions,
            deadline=deadline,
            model=model,
            phase_one=phase_one,
            fixed_nodes=fixed_nodes,
        )
