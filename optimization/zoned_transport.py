"""Zone-aware transportation bounds on attainable stable-matching welfare.

:mod:`optimization.welfare_bounds` develops two *constants* that bound
``W_psi(x)`` uniformly over zonings: the first-choice sum, and the
capacity-feasible transportation value ``W_TP``. Neither prices the zoning
restriction itself -- ``W_TP`` hands every applicant a globally best available
seat, while a real zoning confines them to the programs co-zoned with their
residence.

This module prices that confinement without giving up a single constant. Write
the zoning model itself as a linear program -- the same assignment, anchor,
contiguity, school-count, capacity and diversity rows the masters use, with
``x[z, v]`` relaxed to ``[0, 1]`` so a geographic unit may sit fractionally in
several zones -- reify the co-zoning indicators on top of it, and then maximise
the *zone-restricted* transportation value over that relaxed zoning polytope::

    max   sum_is u_is y_is
    s.t.  x in Omega_LP                       (relaxed zoning model)
          both[v,u,z] <= x[z,v],  <= x[z,u]   (co-zoning linearization)
          a[v,u] <= sum_z both[v,u,z]
          sum_s y_is <= 1                     (one seat per applicant)
          sum_i y_is <= q_s                   (program capacity)
          0 <= y_is <= a[v(i), l(s)]          (access)

Because every integral zoning the master can return is feasible for the
relaxation, and because ``W_psi(x) <= T(x)`` for each of them, the optimal
value bounds sample stable-matching welfare at *every* zoning -- so it is a
valid ``Welfare_max`` -- and it is never above ``W_TP``, which is the same LP
with the access rows deleted.

Two variants differ only in how connectedness is written. ``neighbors`` relaxes
the closer-neighbor support rows the masters ship with; ``flow`` replaces them
with the rooted single-commodity flow description. Both contain every
closer-neighbor-feasible integral zoning, so both are valid, but they are not
nested at fractional points, which is why both are offered rather than one.

Where the tightening comes from is worth naming, because it is not the
contiguity rows. The near-uniform point ``x[z, v] = 1/k`` satisfies every
ratio constraint exactly (each zone receives a ``1/k`` share of every
quantity) and drives ``a[v, u]`` to 1 on every pair whose endpoints are both
free, so what pins the bound below ``W_TP`` is whatever pins zone membership:
the centroid anchors and their ``centroid_neighbor_radius`` neighborhoods, the
distance-derived candidate sets from ``max_distance``, and vertices whose
closer-neighbor support is a single node. With ``max_distance`` unset and
schools away from the anchors, expect the bound to land close to ``W_TP``.

The LP is large and its value depends only on the zoning model and the market,
never on how many threads solved it, so results are cached content-addressed by
exactly those inputs with the worker count excluded from the key.

It also defaults to a single thread. Gurobi runs concurrent LP, racing several
algorithms against each other, and on a model this size they mostly just
contend for memory bandwidth: sweeping 1/2/4/6/8 threads on two Block_2
instances, no count above one was reliably faster and ``flow`` degraded from
2.4s to 4.4s. The whole spread is a couple of seconds against a 300s master
solve, so this is not a speed decision so much as a reason not to take cores
the master can use.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loaders import CacheNamespace, CacheStore, DataScenario
from optimization.data.initial_solutions import feasibility_fingerprint
from optimization.problem import ZoneProblem
from optimization.saa_oracle import access_state

if TYPE_CHECKING:
    from optimization.data.mid import MidProgram, MidStudent

# v2 added `zoned_transport_workers` to the stored metadata. As with the
# feasible-hint cache, cached metadata describes the solve that produced the
# value, not the run reading it -- `zoned_transport_cache: hit` is the flag that
# says so.
ZONED_TRANSPORT_CACHE_SCHEMA_VERSION = 2
ZONED_TRANSPORT_ARTIFACT = "zoned_transport_bound"
ZONED_TRANSPORT_PAYLOAD = "bound.pickle"

# Which connectedness description the relaxed zoning model carries. Mirrors
# ``optimization.solvers.mip.CONTIGUITY_MODELS``.
ZONED_TRANSPORT_MODELS = ("neighbors", "flow")


@dataclass(frozen=True)
class ZonedTransportBound:
    """One solved zoned transportation relaxation.

    ``objective`` is the bound itself. ``prices`` are the capacity duals, which
    are non-negative and therefore an admissible price vector for the
    congestion-priced bound of :mod:`choice.priced_access` -- unlike the
    zone-blind duals, they are the marginal value of a seat *after* confinement
    is accounted for.
    """

    objective: float
    prices: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


def zoned_transport_bound(
    programs: Sequence["MidProgram"],
    students: Sequence["MidStudent"],
    problem: ZoneProblem,
    *,
    contiguity_model: str = "neighbors",
    workers: int = 1,
    centroid_neighbor_radius: int = 0,
) -> ZonedTransportBound:
    """Solve (or reuse) the zoned transportation relaxation for one instance.

    ``workers`` is the Gurobi thread count and defaults to one; see the module
    docstring for why more is not better here. It never enters the cache key.
    """

    if contiguity_model not in ZONED_TRANSPORT_MODELS:
        raise ValueError(
            "zoned transport contiguity_model must be one of: "
            f"{', '.join(ZONED_TRANSPORT_MODELS)}."
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("zoned transport workers must be a positive integer.")
    if isinstance(centroid_neighbor_radius, bool) or not isinstance(
        centroid_neighbor_radius, int
    ):
        raise ValueError("centroid_neighbor_radius must be a non-negative integer.")
    if centroid_neighbor_radius < 0:
        raise ValueError("centroid_neighbor_radius must be a non-negative integer.")

    namespace = _namespace(
        problem,
        programs,
        students,
        contiguity_model=contiguity_model,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    if namespace is not None:
        cached = _validated_payload(namespace.load_pickle(ZONED_TRANSPORT_PAYLOAD))
        if cached is not None:
            return ZonedTransportBound(
                objective=cached["objective"],
                prices=cached["prices"],
                metadata={
                    **cached["metadata"],
                    "zoned_transport_cache": "hit",
                    "zoned_transport_cache_key": namespace.key,
                },
            )

    result = _solve_zoned_transport(
        programs,
        students,
        problem,
        contiguity_model=contiguity_model,
        workers=workers,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    if namespace is None:
        return result

    namespace.save_pickle(
        ZONED_TRANSPORT_PAYLOAD,
        {
            "objective": result.objective,
            "prices": result.prices,
            "metadata": result.metadata,
        },
    )
    return ZonedTransportBound(
        objective=result.objective,
        prices=result.prices,
        metadata={
            **result.metadata,
            "zoned_transport_cache": "miss",
            "zoned_transport_cache_key": namespace.key,
        },
    )


def _solve_zoned_transport(
    programs: Sequence["MidProgram"],
    students: Sequence["MidStudent"],
    problem: ZoneProblem,
    *,
    contiguity_model: str,
    workers: int,
    centroid_neighbor_radius: int,
) -> ZonedTransportBound:
    """Build and solve the relaxation. No time limit, by design.

    A time-limited LP returns a primal-feasible point, which is a *lower*
    bound on the LP value and therefore useless here: validity of
    ``Welfare_max`` needs the optimum.
    """

    import gurobipy as gp
    from gurobipy import GRB

    from optimization.solvers.mip import add_gurobi_zoning_geography

    program_by_id = {program.program_id: program for program in programs}
    if len(program_by_id) != len(programs):
        raise ValueError("Zoned transport program identities must be unique.")

    started = time.perf_counter()
    with gp.Env(params={"OutputFlag": 0}) as env:
        with gp.Model("zoned_transport", env=env) as model:
            model.Params.Threads = int(workers)
            x = add_gurobi_zoning_geography(
                model,
                problem,
                centroid_neighbor_radius=centroid_neighbor_radius,
                contiguity_model=contiguity_model,
            )
            # The zoning rows are built by the shared MIP builder, which declares
            # its membership and boundary variables binary. Relaxing them here --
            # rather than duplicating the builder -- is what keeps this model and
            # the master provably the same set of constraints.
            model.update()
            relaxed = 0
            for variable in model.getVars():
                if variable.VType != GRB.CONTINUOUS:
                    variable.VType = GRB.CONTINUOUS
                    relaxed += 1

            access: dict[tuple[int, int], gp.Var] = {}

            def access_variable(student_node: int, school_node: int) -> gp.Var:
                key = (student_node, school_node)
                existing = access.get(key)
                if existing is not None:
                    return existing
                shared = problem.candidate_zones(
                    student_node
                ) & problem.candidate_zones(school_node)
                joint = []
                for zone in sorted(shared):
                    both = model.addVar(
                        lb=0.0, ub=1.0, name=f"both_{student_node}_{school_node}_{zone}"
                    )
                    model.addConstr(both <= x[(zone, student_node)])
                    model.addConstr(both <= x[(zone, school_node)])
                    joint.append(both)
                variable = model.addVar(
                    lb=0.0, ub=1.0, name=f"a_{student_node}_{school_node}"
                )
                # ``<=`` rather than ``==``: the objective only ever pushes
                # access up, so the two have the same optimum and this keeps the
                # row one-sided.
                model.addConstr(variable <= gp.quicksum(joint))
                access[key] = variable
                return variable

            seats: dict[str, list] = {program_id: [] for program_id in program_by_id}
            assignment_rows = 0
            for student in students:
                share = []
                for program_id, utility in zip(student.programs, student.utilities):
                    program = program_by_id.get(program_id)
                    if program is None:
                        raise ValueError(
                            f"Unknown program {program_id!r} in preferences."
                        )
                    pair, fixed = access_state(problem, student.node, program)
                    seat = model.addVar(lb=0.0, ub=1.0, obj=float(utility))
                    if pair is None:
                        # Citywide programs and schools sited at the applicant's
                        # own vertex are always accessible; a pair whose
                        # endpoints share no candidate zone never is.
                        if not fixed:
                            seat.UB = 0.0
                    else:
                        model.addConstr(seat <= access_variable(*pair))
                    share.append(seat)
                    seats[program_id].append(seat)
                if share:
                    model.addConstr(gp.quicksum(share) <= 1.0)
                    assignment_rows += 1

            capacity_rows = {}
            for program_id, variables in seats.items():
                if not variables:
                    continue
                capacity_rows[program_id] = model.addConstr(
                    gp.quicksum(variables) <= float(program_by_id[program_id].capacity),
                    name=f"capacity_{program_id}",
                )

            model.ModelSense = GRB.MAXIMIZE
            model.optimize()
            if model.Status == GRB.INFEASIBLE:
                # The relaxation contains every feasible zoning, so an empty
                # relaxation means an empty feasible set: the master this bound
                # was computed for cannot be solved either.
                raise RuntimeError(
                    "The relaxed zoning model is infeasible, so no zoning "
                    "satisfies these constraints at all. Loosen the tolerances, "
                    "raise max_distance, or choose different centroids."
                )
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(
                    "Zoned transportation relaxation did not solve to optimality "
                    f"(status {model.Status}); the bound is only valid at the "
                    "LP optimum."
                )
            objective = float(model.ObjVal)
            prices = {
                program_id: max(0.0, float(row.Pi))
                for program_id, row in capacity_rows.items()
            }
            metadata = {
                "zoned_transport_contiguity_model": contiguity_model,
                "zoned_transport_objective": objective,
                "zoned_transport_variables": int(model.NumVars),
                "zoned_transport_constraints": int(model.NumConstrs),
                "zoned_transport_relaxed_variables": relaxed,
                "zoned_transport_access_pairs": len(access),
                "zoned_transport_applicant_rows": assignment_rows,
                "zoned_transport_centroid_neighbor_radius": centroid_neighbor_radius,
                "zoned_transport_workers": int(workers),
                "zoned_transport_solve_seconds": time.perf_counter() - started,
            }
    return ZonedTransportBound(objective=objective, prices=prices, metadata=metadata)


# ---------------------------------------------------------------------- #
# cache
# ---------------------------------------------------------------------- #
def market_fingerprint(
    programs: Sequence["MidProgram"], students: Sequence["MidStudent"]
) -> str:
    """Hash exactly the market inputs the relaxation reads.

    Priorities and scaled utilities are excluded: the transportation relaxation
    discards stability, so neither can change its value.
    """

    payload = {
        "programs": [
            [
                str(program.program_id),
                float(program.capacity),
                bool(program.citywide),
                None if program.school_node is None else int(program.school_node),
            ]
            for program in programs
        ],
        "students": [
            [
                int(student.node),
                [str(program_id) for program_id in student.programs],
                [float(utility) for utility in student.utilities],
            ]
            for student in students
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _namespace(
    problem: ZoneProblem,
    programs: Sequence["MidProgram"],
    students: Sequence["MidStudent"],
    *,
    contiguity_model: str,
    centroid_neighbor_radius: int,
) -> CacheNamespace | None:
    """Resolve the cache namespace, or ``None`` when the problem carries no scenario.

    The key is the zoning feasible set, the market, and how contiguity is
    written -- every input that can move the optimum. The worker count is
    deliberately absent: threads change how long the LP takes and nothing about
    its value, so two runs that differ only in parallelism share one solve.
    """

    scenario = _scenario(problem)
    if scenario is None:
        return None
    return CacheStore(scenario).namespace(
        ZONED_TRANSPORT_ARTIFACT,
        {
            # The warm start is irrelevant to an LP solved to optimality, so it
            # is excluded rather than fragmenting the key across hint methods.
            "problem": feasibility_fingerprint(problem, include_hint=False),
            "market": market_fingerprint(programs, students),
            "contiguity_model": contiguity_model,
            "centroid_neighbor_radius": int(centroid_neighbor_radius),
        },
        schema_version=ZONED_TRANSPORT_CACHE_SCHEMA_VERSION,
        # The fingerprints already pin every source-derived model input.
        roles=(),
    )


def _scenario(problem: ZoneProblem) -> DataScenario | None:
    config = problem.optimization_config
    if config is None:
        return None
    scenario = getattr(config, "data_scenario", None)
    return scenario if isinstance(scenario, DataScenario) else None


def _validated_payload(payload: object) -> dict[str, object] | None:
    """Return a cached bound only when it is structurally intact and finite."""

    if not isinstance(payload, Mapping):
        return None
    objective = payload.get("objective")
    if not isinstance(objective, (int, float)) or isinstance(objective, bool):
        return None
    objective = float(objective)
    if not math.isfinite(objective):
        return None
    raw_prices = payload.get("prices")
    if not isinstance(raw_prices, Mapping):
        return None
    prices: dict[str, float] = {}
    for program_id, price in raw_prices.items():
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            return None
        if not math.isfinite(float(price)) or float(price) < 0.0:
            return None
        prices[str(program_id)] = float(price)
    metadata = payload.get("metadata")
    return {
        "objective": objective,
        "prices": prices,
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }
