"""CP-SAT pricing of one anchored zone against the master's shadow prices.

Pricing asks a question about a *single* zone: over all sets admissible for
label ``z``, which one maximizes its own welfare minus the prices the master
has put on the vertices it would cover? It is not a search for a good zoning,
and the geometry it carries reflects that. The zone is anchored at label
``z``'s centroid, its members are restricted to that label's candidate set,
and connectedness is the closer-neighbour relation of
:mod:`optimization.zone_family` -- the same rows ``cp_bool`` writes. The
previous implementation instead chose its own root and enforced connectedness
with a rooted single-commodity flow, which let one label's pricing problem
range over every connected subset of the graph; on the real instance it
returned zones of 340 to 392 of 579 vertices, no six of which could tile
anything.

Three things follow from anchoring and from there being exactly one zone:

* No flow. Support-monotonicity implies connectedness, so the ``2|E|``
  arc-flow variables, the root indicators, the supply variables and their
  big-``M`` capacity rows all disappear, replaced by one clause per candidate.
* No per-label access machinery. Co-zoning of a student vertex and a school
  vertex is the conjunction of *two* membership variables, so it is one
  Boolean with three rows -- not ``Z`` conjunction variables and a
  ``sum(joints) <= 1`` row, which is what a whole-district master needs. In the
  matching block the variable is not even necessary: its only appearances are
  an upper bound on ``y`` (replace by the two separate bounds) and the
  right-hand side of the two stability rows (replace by ``x_u + x_w - 1``),
  and both substitutions are exact at integral memberships. It is kept because
  the reification propagates in both directions and the substituted form does
  not. The finite-grid block genuinely needs it: its effective-rejection row
  *selects* between the threshold and ``L`` rather than bounding one of them,
  and the substitution is over-restrictive there.
* Program capacity tightens. A program outside the zone seats nobody, so the
  aggregate row becomes ``sum_i y[i,s] <= q_s x_{l(s)}`` rather than
  ``<= q_s``, and likewise ``<= L q_s x_{l(s)}`` for the finite-grid masses.

CP-SAT rather than Gurobi, for two reasons specific to this model. The ``min``
recurrence that defines finite-grid MID welfare is a native constraint here
(``AddMinEquality``), where a MIP needs a big-``M`` disjunction per
``(type, rank)`` pair -- roughly 24,000 auxiliary integers on
:math:`\\text{Block}_2`, and the weakest part of that relaxation. And the
zoning block is what CP-SAT is best at in this repo.

Integer arithmetic and the bound
--------------------------------
CP-SAT needs integer coefficients, so the objective is scaled by ``K`` and
rounded. The rounding is *directional*, which is what keeps the bound of
Proposition 9 valid: utilities are scaled with a ceiling, so the model's
welfare is never below the true welfare, and the duals are scaled with a
floor, so the model's penalty is never above the true penalty. The scaled
objective therefore dominates the true reduced welfare pointwise, and CP-SAT's
proven objective bound divided by ``K`` is a valid upper bound on it. In the
other direction nothing is assumed: a returned zone is re-validated by
:meth:`ZoneFamily.feasible`, scored exactly by the welfare oracle, and its
reduced cost is recomputed by the master, so a rounding artefact can waste a
round but can never admit a column that does not improve.

One model per label and phase is built once and re-aimed between
column-generation rounds: branch fixings are CP-SAT *assumptions*, released
with ``ClearAssumptions``, and the duals are a fresh objective, so nothing is
rebuilt when only the prices move. A solution callback harvests every
improving zone the search passes through, not only the last one, because a
set-partitioning master stalls on degenerate pivots and more columns per round
is the cheapest thing that helps.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from optimization.mid_oracle import cutoff_upper_bounds
from optimization.zone_welfare import restrict_market


#: CP-SAT's integer limit, halved so a sum of two objective-sized terms fits.
_INTEGER_LIMIT = (2**63 - 1) // 2

_STATUS = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
}


@dataclass(frozen=True)
class PricingResult:
    """What one label's pricing solve proved, and what it found.

    ``bound`` is a global upper bound on that label's reduced welfare and is
    what Proposition 9 consumes; it is finite whenever CP-SAT proved anything,
    including on an interrupted solve. ``candidates`` holds every *improving*
    zone the search passed through, best first, and ``nodes`` is the best.

    ``allowance`` is how much ``bound`` can over-state the true reduced-welfare
    optimum purely because the objective is integer-scaled and directionally
    rounded. Ceiling a utility inflates the objective by less than one scaled
    unit per welfare term, and flooring a dual deflates its penalty by less
    than one per membership or cut variable, so the total inflation is below
    ``units / K`` with ``units`` a count of terms. It is reported rather than
    subtracted: subtracting it would break the bound's validity, while adding
    it to the incumbent comparison turns the search's guarantee into a stated
    absolute tolerance.
    """

    status: str
    bound: float
    nodes: frozenset[int] = frozenset()
    reduced_cost: float | None = None
    candidates: tuple[frozenset[int], ...] = ()
    allowance: float = 0.0
    wall_time: float = 0.0


def compatible(column, decisions):
    return all(
        (node in column.nodes) == value
        for (node, zone), value in decisions.items()
        if zone == column.zone
    )


class _Harvest(cp_model.CpSolverSolutionCallback):
    """Record the membership of every solution the search passes through."""

    def __init__(self, variables, limit):
        super().__init__()
        self._variables = tuple(variables)
        self._limit = int(limit)
        self.found: list[tuple[float, frozenset[int]]] = []

    def on_solution_callback(self) -> None:
        self.found.append(
            (
                self.ObjectiveValue(),
                frozenset(
                    node for node, var in self._variables if self.Value(var) == 1
                ),
            )
        )
        if len(self.found) > self._limit:
            # Keep the best ``limit`` seen so far. The master re-checks every
            # column, so this caps work rather than correctness.
            self.found.sort(key=lambda item: -item[0])
            del self.found[self._limit :]

    def improving(self, scale: float, offset: float) -> tuple[frozenset[int], ...]:
        """Memberships whose reduced cost is positive, best first, deduplicated."""

        seen: set[frozenset[int]] = set()
        ordered = []
        for value, nodes in sorted(self.found, key=lambda item: -item[0]):
            if value / scale - offset <= 0 or nodes in seen:
                continue
            seen.add(nodes)
            ordered.append(nodes)
        return tuple(ordered)


@dataclass
class _ZoneModel:
    """One label's live model plus the handles needed to re-aim it."""

    model: cp_model.CpModel
    x: dict[int, cp_model.IntVar]
    welfare: object = None
    cut_terms: tuple[tuple[tuple[int, int], int, object], ...] = ()
    allowance_units: int = 0
    infeasible: bool = False
    stats: dict = field(default_factory=dict)


class ZonePricer:
    """Persistent CP-SAT pricing models, one per zone label and phase."""

    def __init__(
        self,
        pool,
        *,
        workers=0,
        seed=42,
        scale=1000,
        columns_per_call=8,
        parallel=True,
        non_wastefulness=True,
        aggregate_stability=True,
    ):
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError("dw_pricing_scale must be a positive integer.")
        if (
            isinstance(columns_per_call, bool)
            or not isinstance(columns_per_call, int)
            or columns_per_call <= 0
        ):
            raise ValueError("dw_pricing_columns_per_call must be a positive integer.")
        self.pool = pool
        self.workers = max(0, int(workers))
        self.seed = int(seed)
        self.scale = int(scale)
        self.columns_per_call = int(columns_per_call)
        self.parallel = bool(parallel)
        self.non_wastefulness = bool(non_wastefulness)
        self.aggregate_stability = bool(aggregate_stability)
        self._built: dict[tuple[int, bool], _ZoneModel] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._built.clear()

    @property
    def models_built(self) -> int:
        return len(self._built)

    @property
    def objective_scale(self) -> int:
        """``K``: the integer units the pricing objective is measured in."""

        objective = self.pool.objective
        if objective.kind == "mid":
            return self.scale * objective.lottery_scale
        if objective.kind == "boundary":
            return 2 * self.scale
        return self.scale

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #
    def price_all(self, pool, master, decisions, *, deadline, phase_one=False):
        """Price every label against one deadline, in parallel when asked.

        Parallel pricing gives each label the whole remaining budget rather
        than a share of it, and that matters because Proposition 9 needs a
        finite bound from *every* label: under a sequential split the first
        label used to consume the budget and the rest returned unpriced.
        """

        labels = tuple(range(pool.problem.Z))
        if not self.parallel or len(labels) <= 1:
            results = {}
            for priced, zone in enumerate(labels):
                share = (deadline - time.monotonic()) / (len(labels) - priced)
                results[zone] = self(
                    pool,
                    zone,
                    master,
                    decisions,
                    deadline=min(deadline, time.monotonic() + share),
                    phase_one=phase_one,
                )
            return results
        # CP-SAT releases the interpreter lock inside Solve, so threads give
        # real parallelism. Each label takes its own share of the search
        # workers so the labels together do not oversubscribe the machine.
        with ThreadPoolExecutor(max_workers=len(labels)) as pods:
            futures = {
                zone: pods.submit(
                    self,
                    pool,
                    zone,
                    master,
                    decisions,
                    deadline=deadline,
                    phase_one=phase_one,
                )
                for zone in labels
            }
            return {zone: future.result() for zone, future in futures.items()}

    def __call__(self, pool, zone, master, decisions, *, deadline, phase_one=False):
        """Globally maximize reduced welfare over every zone admissible for ``zone``."""

        if pool is not self.pool:
            raise ValueError("A ZonePricer is bound to the pool it was built for.")
        start = time.monotonic()
        if deadline - start <= 0:
            return PricingResult("TIME_LIMIT", math.inf)

        built = self._built.get((zone, phase_one))
        if built is None:
            built = self._built[(zone, phase_one)] = self._new_model(zone, phase_one)
        if built.infeasible or self._branch_excludes(built, zone, decisions):
            return PricingResult("INFEASIBLE", -math.inf)

        offset = 0.0 if master is None else float(master.zone_duals[zone])
        self._aim(built, master, decisions, zone)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return PricingResult("TIME_LIMIT", math.inf)
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = self.seed
        solver.parameters.num_search_workers = self._search_workers(pool)
        if math.isfinite(remaining):
            solver.parameters.max_time_in_seconds = max(1e-3, remaining)
        harvest = _Harvest(sorted(built.x.items()), self.columns_per_call)
        status = solver.Solve(built.model, harvest)
        return self._read(
            solver,
            status,
            harvest,
            offset,
            built.allowance_units / self.objective_scale,
            time.monotonic() - start,
        )

    def _search_workers(self, pool) -> int:
        if not self.workers:
            return 1
        if self.parallel and pool.problem.Z > 1:
            return max(1, self.workers // pool.problem.Z)
        return self.workers

    @staticmethod
    def _branch_excludes(built, zone, decisions) -> bool:
        """Does the branch pin a non-candidate vertex *into* this label?"""

        return any(
            value and node not in built.x
            for (node, label), value in decisions.items()
            if label == zone
        )

    # ------------------------------------------------------------------ #
    # Re-aiming
    # ------------------------------------------------------------------ #
    def _aim(self, built, master, decisions, zone):
        """Point an existing model at this call's fixings and duals."""

        built.model.ClearAssumptions()
        literals = [
            built.x[node] if value else built.x[node].Not()
            for (node, label), value in sorted(decisions.items())
            if label == zone and node in built.x
        ]
        if literals:
            built.model.AddAssumptions(literals)

        variables = []
        coefficients = []
        if built.welfare is not None:
            variables.append(built.welfare)
            coefficients.append(1)
        if master is not None:
            scale = self.objective_scale
            for node, var in sorted(built.x.items()):
                # Floor, so the model never charges more than the true dual.
                coefficient = math.floor(scale * float(master.node_duals[node]))
                if coefficient:
                    variables.append(var)
                    coefficients.append(-coefficient)
            price = float(master.boundary_dual)
            if price and not built.cut_terms:
                # The master only has a boundary row when the cap is on, and
                # the cut variables exist exactly then, so a priced boundary
                # with nothing to price it against is a wiring error rather
                # than a term to drop.
                raise ValueError(
                    "A boundary dual was supplied but the pricing model has no "
                    "cut variables; set problem.boundary_prop >= 0."
                )
            if price:
                for _, weight, var in built.cut_terms:
                    coefficient = math.floor(scale * price * weight / 2)
                    if coefficient:
                        variables.append(var)
                        coefficients.append(-coefficient)
        if variables:
            built.model.Maximize(
                cp_model.LinearExpr.WeightedSum(variables, coefficients)
            )
        else:
            built.model.Maximize(0)

    def _read(self, solver, status, harvest, offset, allowance, wall):
        name = _STATUS.get(status, "UNKNOWN")
        if name == "INFEASIBLE":
            return PricingResult("INFEASIBLE", -math.inf, wall_time=wall)
        if name == "MODEL_INVALID":
            return PricingResult("ERROR", math.inf, wall_time=wall)
        scale = self.objective_scale
        bound = solver.BestObjectiveBound() / scale - offset
        if not math.isfinite(bound):
            bound = math.inf
        candidates = harvest.improving(scale, offset)
        if not harvest.found:
            return PricingResult(
                "TIME_LIMIT" if name == "UNKNOWN" else name,
                bound,
                allowance=allowance,
                wall_time=wall,
            )
        return PricingResult(
            name if name in {"OPTIMAL", "FEASIBLE"} else "FEASIBLE",
            bound,
            candidates[0] if candidates else frozenset(),
            solver.ObjectiveValue() / scale - offset,
            candidates,
            allowance,
            wall,
        )

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #
    def _new_model(self, zone, phase_one) -> _ZoneModel:
        pool = self.pool
        family = pool.family
        problem = pool.problem
        model = cp_model.CpModel()
        members = sorted(family.candidates[zone])
        x = {node: model.NewBoolVar(f"x_{node}") for node in members}
        built = _ZoneModel(model, x)
        if not family.forced[zone] <= family.candidates[zone]:
            built.infeasible = True
            return built

        for node in sorted(family.forced[zone]):
            model.Add(x[node] == 1)
        for node, support in family.supports[zone].items():
            model.AddBoolOr([x[node].Not(), *[x[other] for other in support]])
        for row in family.rows[zone]:
            terms = [
                (node, coefficient)
                for node, coefficient in sorted(row.coefficients.items())
                if coefficient and node in x
            ]
            expression = cp_model.LinearExpr.WeightedSum(
                [x[node] for node, _ in terms], [value for _, value in terms]
            )
            if row.sense == ">=":
                model.Add(expression >= row.rhs)
            else:
                model.Add(expression <= row.rhs)

        objective = pool.objective
        if objective.kind == "boundary" or problem.boundary_prop >= 0:
            built.cut_terms = add_cut_indicators(model, problem, x)

        stats = {}
        welfare_units = 0
        if not phase_one:
            built.welfare, stats = add_welfare_block(
                model,
                x,
                objective,
                self.scale,
                cut_terms=built.cut_terms,
                non_wastefulness=self.non_wastefulness,
                aggregate_stability=self.aggregate_stability,
            )
            welfare_units = welfare_allowance_units(objective, stats)
        # Flooring a dual under-charges by under one scaled unit per membership
        # and per cut variable it multiplies. A welfare block adds its own
        # ceiling inflation on top; see :func:`welfare_allowance_units`.
        built.allowance_units = welfare_units + len(members) + len(built.cut_terms)
        built.stats = {
            "candidate_nodes": len(members),
            "forced_nodes": len(family.forced[zone]),
            "cut_indicators": len(built.cut_terms),
            "allowance_units": built.allowance_units,
            **stats,
        }
        return built

    def metadata(self) -> dict:
        return {
            "dw_pricing": "cpsat_anchored_closer_neighbor",
            "dw_pricing_models": self.models_built,
            "dw_pricing_scale": self.scale,
            "dw_pricing_objective_scale": self.objective_scale,
            "dw_pricing_parallel": self.parallel,
            "dw_pricing_columns_per_call": self.columns_per_call,
            "dw_pricing_non_wastefulness": self.non_wastefulness,
            "dw_pricing_aggregate_stability": self.aggregate_stability,
            "dw_pricing_model_stats": {
                f"zone_{zone}": built.stats
                for (zone, phase_one), built in sorted(self._built.items())
                if not phase_one
            },
        }




# ---------------------------------------------------------------------- #
# Shared model builders
# ---------------------------------------------------------------------- #
# Everything below is written against a *membership dictionary*: ``x`` maps a
# node to the literal "this node belongs to the zone being built", and its
# keys are the nodes the zone may draw from. That is the only interface a
# welfare block needs, which is what lets one label's pricing model
# (:class:`ZonePricer`) and a two-label redraw
# (:mod:`optimization.zone_redraw`) share the rows exactly rather than by
# reimplementation. ``tag`` prefixes the generated variable names so a model
# carrying two blocks stays readable; it changes nothing else.


def solver_status(status) -> str:
    """CP-SAT's status enum as the string this package reports."""

    return _STATUS.get(status, "UNKNOWN")


def ceil_scale(value: float, scale: int) -> int:
    """Scale a utility upward, so the model's welfare never understates."""

    if not math.isfinite(float(value)):
        raise ValueError(f"DW pricing utility is not finite: {value!r}.")
    return math.ceil(float(value) * scale)


def bind_welfare(model, terms, bound, *, name="zone_welfare"):
    """One integer variable holding a welfare block, so re-aiming is cheap.

    The pricing objective is re-set once per column-generation round. Binding
    the thousands of welfare terms to a single variable makes that a two-term
    expression instead of a rebuild of the whole sum, and it is also what lets
    a two-label model add two blocks together in two terms.
    """

    if bound > _INTEGER_LIMIT:
        raise ValueError("DW pricing objective exceeds CP-SAT integer limits.")
    welfare = model.NewIntVar(0, int(bound), name)
    if terms:
        model.Add(welfare == sum(terms))
    else:
        model.Add(welfare == 0)
    return welfare


def boundary_welfare(model, cut_terms, scale, *, name="zone_welfare"):
    """Minus half the perimeter, in units of ``1 / (2 * scale)``."""

    total = sum(weight for _, weight, _ in cut_terms) * scale
    welfare = model.NewIntVar(-int(total), 0, name)
    model.Add(
        welfare
        == -cp_model.LinearExpr.WeightedSum(
            [var for _, _, var in cut_terms],
            [scale * weight for _, weight, _ in cut_terms],
        )
    )
    return welfare


def add_cut_indicators(model, problem, x, *, tag=""):
    """``|x_u - x_v|`` for every edge with at least one endpoint in ``x``.

    A node absent from ``x`` can never belong to the zone, so an edge with one
    endpoint outside ``x`` is cut exactly when its inside endpoint joins and
    needs no variable of its own. The returned terms are therefore the exact
    perimeter of whatever set ``x`` selects.
    """

    terms = []
    for u, v in problem.G.edges:
        weight = problem.boundary_weight(u, v)
        if not weight:
            continue
        inside_u, inside_v = x.get(u), x.get(v)
        if inside_u is None and inside_v is None:
            continue
        if inside_u is None or inside_v is None:
            # The non-candidate endpoint can never join, so the edge is cut
            # exactly when the candidate endpoint joins. No variable.
            only = inside_v if inside_u is None else inside_u
            terms.append(((u, v), weight, only))
            continue
        cut = model.NewBoolVar(f"{tag}cut_{u}_{v}")
        model.Add(inside_u != inside_v).OnlyEnforceIf(cut)
        model.Add(inside_u == inside_v).OnlyEnforceIf(cut.Not())
        terms.append(((u, v), weight, cut))
    return tuple(terms)


def access_indicator(model, x, student_node, school_node, cache, *, tag=""):
    """``x_u AND x_w``: the only place the zoning enters a welfare block."""

    if student_node == school_node:
        return x[student_node]
    key = (min(student_node, school_node), max(student_node, school_node))
    if key in cache:
        return cache[key]
    first, second = x[key[0]], x[key[1]]
    both = model.NewBoolVar(f"{tag}access_{key[0]}_{key[1]}")
    model.AddImplication(both, first)
    model.AddImplication(both, second)
    model.AddBoolOr([first.Not(), second.Not(), both])
    cache[key] = both
    return both


def add_mid_welfare(model, x, objective, scale, *, tag=""):
    """Finite-grid MID welfare of the selected zone, exactly.

    The structure is ``MidCpSatSolver``'s, restricted to one label: an integer
    cutoff per in-zone program, native ``min``/``max`` for the threshold and
    the remaining-mass recurrence, and the valid inequality
    ``R_r >= R_{r-1} + e_r - L``, which is what keeps the relaxation of that
    recurrence from manufacturing mass at high-utility ranks. Proposition 7 is
    what makes maximizing over capacity-clearing cutoffs the same as
    evaluating at the least clearing one.

    ``scale`` is the utility scale; the lottery grid ``L`` comes from the
    objective. This block is *not* all-Boolean -- cutoffs, thresholds and the
    mass recurrence are small-domain integers -- and the access conjunction is
    genuinely needed here, because the effective-rejection row *selects*
    between the threshold and ``L`` rather than bounding one of them.
    """

    lottery = objective.lottery_scale
    market = restrict_market(objective.market, frozenset(x))
    programs = market.program_by_id
    upper = cutoff_upper_bounds(market, lottery)

    tiers: dict[str, set[int]] = {program_id: set() for program_id in programs}
    for student in market.types:
        for program_id, priority in zip(student.programs, student.priorities):
            tiers[program_id].add(priority)

    cutoffs = {}
    thresholds = {}
    for program_id, priorities in tiers.items():
        limit = int(upper[program_id])
        cutoff = model.NewIntVar(0, limit, f"{tag}cutoff_{program_id}")
        cutoffs[program_id] = cutoff
        for priority in sorted(priorities):
            offset = priority * lottery
            reach = max(0, limit - offset)
            threshold = model.NewIntVar(
                0, min(lottery, reach), f"{tag}threshold_{program_id}_{priority}"
            )
            if reach <= lottery:
                model.AddMaxEquality(threshold, [cutoff - offset, 0])
            else:
                raw = model.NewIntVar(0, reach, f"{tag}raw_{program_id}_{priority}")
                model.AddMaxEquality(raw, [cutoff - offset, 0])
                model.AddMinEquality(threshold, [raw, lottery])
            thresholds[program_id, priority] = threshold

    access: dict[tuple[int, int], object] = {}
    capacity_terms: dict[str, list] = {program_id: [] for program_id in programs}
    welfare_terms = []
    bound = 0
    for index, student in enumerate(market.types):
        previous = lottery
        for rank, (program_id, priority, utility) in enumerate(
            zip(student.programs, student.priorities, student.utility_sums)
        ):
            program = programs[program_id]
            indicator = access_indicator(
                model, x, student.node, program.school_node, access, tag=tag
            )
            threshold = thresholds[program_id, priority]
            effective = model.NewIntVar(0, lottery, f"{tag}effective_{index}_{rank}")
            model.Add(effective == threshold).OnlyEnforceIf(indicator)
            model.Add(effective == lottery).OnlyEnforceIf(indicator.Not())
            remaining = model.NewIntVar(0, lottery, f"{tag}remaining_{index}_{rank}")
            model.AddMinEquality(remaining, [previous, effective])
            model.Add(remaining >= previous + effective - lottery)
            mass = previous - remaining
            capacity_terms[program_id].append(student.count * mass)
            welfare_terms.append(ceil_scale(utility, scale) * mass)
            previous = remaining
        bound += lottery * max(
            (ceil_scale(value, scale) for value in student.utility_sums),
            default=0,
        )

    for program_id, program in programs.items():
        terms = capacity_terms[program_id]
        if terms:
            model.Add(
                sum(terms) <= lottery * program.capacity * x[program.school_node]
            )

    welfare = bind_welfare(
        model, welfare_terms, bound, name=f"{tag}zone_welfare"
    )
    return welfare, {
        "types": len(market.types),
        "programs": len(programs),
        "access_indicators": len(access),
        "cutoffs": len(cutoffs),
        "thresholds": len(thresholds),
    }


def add_matching_welfare(
    model,
    x,
    objective,
    scale,
    *,
    non_wastefulness=True,
    aggregate_stability=True,
    tag="",
):
    """Welfare of the zone's stable matching, under one tie-breaking draw.

    The rows are the access polytope's, made integral: ``y`` seats a student,
    ``z`` says the student clears the program's realized cutoff, and the
    monotone chain down the drawn priority order is what makes the clearing set
    a prefix of that order -- i.e. a cutoff -- with no cutoff variable. The
    weakly-preferred prefix is declared Boolean, which states "at most one seat
    per student" as a domain rather than a row.

    Every variable here is Boolean except the per-program seat count and the
    running ``(S2)`` prefixes, whose domains are a program's quota. That is why
    this block prices in seconds, and it is preserved verbatim when two labels
    are redrawn at once: the block is per label either way.
    """

    prepared = objective.prepared
    market = prepared.market
    members = frozenset(x)
    available = {
        program.program_id: program
        for program in market.programs
        if program.school_node in members and program.capacity > 0
    }
    retained = {}
    for index, student in enumerate(market.students):
        if student.node not in members:
            continue
        ranks = [
            rank
            for rank, program_id in enumerate(student.programs)
            if program_id in available
        ]
        if ranks:
            retained[index] = ranks

    access: dict[tuple[int, int], object] = {}
    seat: dict[tuple[int, str], object] = {}
    clears: dict[tuple[int, str], object] = {}
    prefix: dict[tuple[int, str], object] = {}
    welfare_terms = []
    bound = 0
    for index, ranks in retained.items():
        student = market.students[index]
        previous = None
        best = 0
        for rank in ranks:
            program_id = student.programs[rank]
            program = available[program_id]
            key = (index, program_id)
            indicator = access_indicator(
                model, x, student.node, program.school_node, access, tag=tag
            )
            y = model.NewBoolVar(f"{tag}y_{index}_{program_id}")
            z = model.NewBoolVar(f"{tag}z_{index}_{program_id}")
            seat[key] = y
            clears[key] = z
            model.Add(y <= indicator)                                   # (F3)
            model.Add(y <= z)                                           # (F4)
            # The weakly-preferred prefix. Boolean, so (F1) is its domain.
            weakly = model.NewBoolVar(f"{tag}prefix_{index}_{rank}")
            model.Add(weakly == (y if previous is None else previous + y))
            prefix[key] = weakly
            model.Add(weakly >= indicator + z - 1)                      # (F6)
            coefficient = ceil_scale(student.utilities[rank], scale)
            welfare_terms.append(coefficient * y)
            best = max(best, coefficient)
            previous = weakly
        bound += best

    chain_rows = 0
    stability_rows = 0
    for program_index, program in enumerate(market.programs):
        program_id = program.program_id
        if program_id not in available:
            continue
        quota = int(program.capacity)
        order = [
            index
            for index in prepared.sample.school_orders[program_index]
            if (index, program_id) in seat
        ]
        if not order:
            continue
        seats = model.NewIntVar(0, quota, f"{tag}seats_{program_id}")
        model.Add(seats == sum(seat[(index, program_id)] for index in order))
        model.Add(seats <= quota * x[program.school_node])              # (F2)
        running = 0
        previous_clears = None
        for position, index in enumerate(order):
            key = (index, program_id)
            if previous_clears is not None:
                model.Add(clears[key] <= previous_clears)               # (F5)
                chain_rows += 1
            previous_clears = clears[key]
            if non_wastefulness:
                model.Add(seats + quota * clears[key] >= quota)         # (S1)
            if aggregate_stability:
                indicator = access_indicator(
                    model,
                    x,
                    market.students[index].node,
                    program.school_node,
                    access,
                    tag=tag,
                )
                model.Add(
                    quota * prefix[key] + running >= quota * indicator
                )                                                       # (S2)
                stability_rows += 1
                if position + 1 < len(order):
                    # Running prefix over the students the program *strictly*
                    # prefers to this one, extended a position at a time so
                    # (S2) stays O(|Gamma(s)|) rows.
                    tail = model.NewIntVar(
                        0, quota, f"{tag}better_{program_id}_{position + 1}"
                    )
                    model.Add(tail == running + seat[key])
                    running = tail

    welfare = bind_welfare(
        model, welfare_terms, bound, name=f"{tag}zone_welfare"
    )
    return welfare, {
        "students": len(retained),
        "programs": len(available),
        "access_indicators": len(access),
        "pairs": len(seat),
        "chain_rows": chain_rows,
        "aggregate_stability_rows": stability_rows,
    }


def welfare_allowance_units(objective, stats) -> int:
    """Scaled units by which a ceiling-rounded welfare block can overstate.

    Ceiling a utility inflates the objective by under one scaled unit per
    welfare term: a MID type's masses sum to at most ``L``, and a matching
    student takes at most one seat.
    """

    if objective.kind == "mid":
        return objective.lottery_scale * stats["types"]
    if objective.kind == "stable_matching":
        return stats["students"]
    if objective.kind == "boundary":
        # Integer weights times an integer scale: no rounding at all.
        return 0
    raise ValueError(f"Unknown DW objective {objective.kind!r}.")


def add_welfare_block(
    model,
    x,
    objective,
    scale,
    *,
    cut_terms=(),
    non_wastefulness=True,
    aggregate_stability=True,
    tag="",
):
    """One label's welfare, whichever definition this run decomposes."""

    if objective.kind == "boundary":
        return boundary_welfare(
            model, cut_terms, scale, name=f"{tag}zone_welfare"
        ), {}
    if objective.kind == "mid":
        return add_mid_welfare(model, x, objective, scale, tag=tag)
    if objective.kind == "stable_matching":
        return add_matching_welfare(
            model,
            x,
            objective,
            scale,
            non_wastefulness=non_wastefulness,
            aggregate_stability=aggregate_stability,
            tag=tag,
        )
    raise ValueError(f"Unknown DW objective {objective.kind!r}.")


def price_zone(pool, zone, master, decisions, *, deadline, phase_one=False, **options):
    """One-shot pricing. Inside a loop, hold a :class:`ZonePricer` instead."""

    with ZonePricer(pool, **options) as pricer:
        return pricer(
            pool, zone, master, decisions, deadline=deadline, phase_one=phase_one
        )
