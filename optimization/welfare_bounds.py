"""A-priori upper bounds on stable-matching welfare, valid for every zoning.

The SAA master needs a constant that bounds ``W_s(x)`` for every zoning ``x``
and every tie-breaking scenario ``s``, because it is declared as the domain of
the master's objective variable. That constant matters more than it looks:
whenever the solver fails to improve its own dual bound -- which is what happens
on these instances -- the constant *is* the optimality gap that gets reported.

Two zoning-independent bounds live here. ``first_choice_upper_bound`` is the
one the code shipped with: give every student their top-ranked program.
``transport_upper_bound`` additionally respects program capacities, which is
what makes it strictly smaller whenever any program is oversubscribed.

Neither prices the zoning restriction itself. The two ``zoned_transport_*``
kinds do, by maximising the zone-restricted transportation value over the
*linear relaxation* of the zoning model; they live in
:mod:`optimization.zoned_transport` because they read the whole
:class:`~optimization.problem.ZoneProblem` rather than just the market, and
they are cached because the LP is large.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimization.data.mid import MidProgram, MidStudent
    from optimization.problem import ZoneProblem


def first_choice_upper_bound(students: Sequence["MidStudent"]) -> float:
    """Welfare if every student were handed their top-ranked program.

    Ignores capacity, so it is loose by exactly the amount of oversubscription.
    """
    return sum(
        student.utilities[0] if student.utilities else 0.0 for student in students
    )


def transport_upper_bound(
    programs: Sequence["MidProgram"], students: Sequence["MidStudent"]
) -> float:
    """Welfare of the best capacity-feasible assignment, ignoring zones.

    Every stable matching gives each student at most one seat and each program at
    most its capacity, so it is a feasible point of the transportation polytope

        max  sum_ip u_ip y_ip
        s.t. sum_p y_ip <= 1        (one seat per student)
             sum_i y_ip <= c_p      (program capacity)
             0 <= y <= 1

    Relaxing the stability and zone-access constraints can only raise the
    optimum, so this bounds ``W_s(x)`` for every zoning ``x`` and every scenario
    ``s``. The polytope is integral, so the LP value is attained by an integral
    assignment and the bound is the tightest one obtainable without reasoning
    about zones or stability.
    """
    import gurobipy as gp
    from gurobipy import GRB

    capacity = {program.program_id: program.capacity for program in programs}
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as model:
        model.ModelSense = GRB.MAXIMIZE
        seats: dict[str, list] = {program_id: [] for program_id in capacity}
        for student in students:
            if not student.programs:
                continue
            share = []
            for program_id, utility in zip(student.programs, student.utilities):
                if program_id not in capacity:
                    raise ValueError(f"Unknown program {program_id!r} in preferences.")
                variable = model.addVar(lb=0.0, ub=1.0, obj=float(utility))
                share.append(variable)
                seats[program_id].append(variable)
            model.addConstr(gp.quicksum(share) <= 1.0)
        for program_id, variables in seats.items():
            if variables:
                model.addConstr(gp.quicksum(variables) <= float(capacity[program_id]))
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Transportation upper bound did not solve (status {model.Status})."
            )
        return float(model.ObjVal)


WELFARE_BOUNDS = (
    "first_choice",
    "transport",
    "zoned_transport_neighbors",
    "zoned_transport_flow",
)

# The two zone-aware kinds need the zoning model, so they cannot be evaluated
# from the market alone. The suffix names the contiguity description.
ZONED_WELFARE_BOUNDS = {
    "zoned_transport_neighbors": "neighbors",
    "zoned_transport_flow": "flow",
}


def welfare_upper_bound(
    kind: str,
    programs: Sequence["MidProgram"],
    students: Sequence["MidStudent"],
    *,
    problem: "ZoneProblem | None" = None,
    workers: int = 1,
    centroid_neighbor_radius: int = 0,
) -> float:
    """Dispatch to the named a-priori welfare bound.

    ``problem`` is required by the ``zoned_transport_*`` kinds and ignored by
    the others. ``workers`` only sets Gurobi's thread count and never the value,
    which is why it is not part of the zoned bound's cache key; it defaults to
    one because concurrent LP measured slower on that model.
    """
    if kind == "first_choice":
        return first_choice_upper_bound(students)
    if kind == "transport":
        return transport_upper_bound(programs, students)
    contiguity_model = ZONED_WELFARE_BOUNDS.get(kind)
    if contiguity_model is not None:
        if problem is None:
            raise ValueError(
                f"Welfare bound {kind!r} needs the zoning problem; none was given."
            )
        from optimization.zoned_transport import zoned_transport_bound

        return zoned_transport_bound(
            programs,
            students,
            problem,
            contiguity_model=contiguity_model,
            workers=workers,
            centroid_neighbor_radius=centroid_neighbor_radius,
        ).objective
    raise ValueError(
        f"Unknown welfare bound {kind!r}; expected one of {WELFARE_BOUNDS}."
    )
