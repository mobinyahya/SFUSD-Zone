"""A-priori upper bounds on stable-matching welfare, valid for every zoning.

The SAA master needs a constant that bounds ``W_s(x)`` for every zoning ``x``
and every tie-breaking scenario ``s``, because it is declared as the domain of
the master's objective variable. That constant matters more than it looks:
whenever the solver fails to improve its own dual bound -- which is what happens
on these instances -- the constant *is* the optimality gap that gets reported.

Two bounds live here. ``first_choice_upper_bound`` is the one the code shipped
with: give every student their top-ranked program. ``transport_upper_bound``
additionally respects program capacities, which is what makes it strictly
smaller whenever any program is oversubscribed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimization.data.mid import MidProgram, MidStudent


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


WELFARE_BOUNDS = ("first_choice", "transport")


def welfare_upper_bound(
    kind: str,
    programs: Sequence["MidProgram"],
    students: Sequence["MidStudent"],
) -> float:
    """Dispatch to the named a-priori welfare bound."""
    if kind == "first_choice":
        return first_choice_upper_bound(students)
    if kind == "transport":
        return transport_upper_bound(programs, students)
    raise ValueError(f"Unknown welfare bound {kind!r}; expected one of {WELFARE_BOUNDS}.")
