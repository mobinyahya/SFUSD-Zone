"""LP recourse oracle shared by the SAA zoning masters."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from optimization.data.mid import MidProgram
from optimization.data.saa import SaaMarket, SaaSample
from optimization.problem import ZoneProblem


AccessPair = tuple[int, int]
_OUTSIDE = "__saa_outside__"


@dataclass(frozen=True)
class SaaCut:
    sample_index: int
    constant: float
    coefficients: tuple[tuple[AccessPair, float], ...]
    anchor_access: tuple[tuple[AccessPair, int], ...] = ()

    def value(self, access: dict[AccessPair, int]) -> float:
        return self.constant + sum(
            coefficient * access[pair] for pair, coefficient in self.coefficients
        )


@dataclass(frozen=True)
class SaaOracleResult:
    welfare: float
    cut: SaaCut


class SaaOracle:
    """Continuous stable-matching LP for one strict-priority sample."""

    def __init__(
        self,
        market: SaaMarket,
        sample: SaaSample,
        sample_index: int,
        problem: ZoneProblem,
        workers: int | None = None,
    ) -> None:
        if _OUTSIDE in market.program_by_id:
            raise ValueError(f"Program identity {_OUTSIDE!r} is reserved by SAA.")
        if len(sample.school_orders) != len(market.programs):
            raise ValueError("SAA school orders must align with programs.")
        self.market = market
        self.sample_index = sample_index
        self.problem = problem
        self.programs = market.program_by_id
        self.capacities = {
            program.program_id: program.capacity for program in market.programs
        }
        self.capacities[_OUTSIDE] = len(market.students)
        self.preferences = tuple(
            student.programs + (_OUTSIDE,) for student in market.students
        )

        model = gp.Model(f"saa_recourse_{sample_index}")
        model.Params.OutputFlag = 0
        if workers is not None:
            model.Params.Threads = int(workers)
        variables = {
            (student_index, program_id): model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"d_{student_index}_{program_id}",
            )
            for student_index, preferences in enumerate(self.preferences)
            for program_id in preferences
        }
        model.update()

        self.assignment_constraints = {}
        self.capacity_constraints = {}
        self.access_constraints = {}
        self.stability_constraints = {}
        preferred = {}
        for student_index, preferences in enumerate(self.preferences):
            self.assignment_constraints[student_index] = model.addConstr(
                gp.quicksum(
                    variables[(student_index, program_id)] for program_id in preferences
                )
                == 1.0,
                name=f"student_{student_index}",
            )
            previous = None
            for rank, program_id in enumerate(preferences):
                cumulative = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"preferred_{student_index}_{rank}",
                )
                if previous is None:
                    model.addConstr(
                        cumulative == variables[(student_index, program_id)]
                    )
                else:
                    model.addConstr(
                        cumulative == previous + variables[(student_index, program_id)]
                    )
                preferred[(student_index, program_id)] = cumulative
                previous = cumulative

        interested: dict[str, tuple[int, ...]] = {
            program.program_id: tuple(
                student_index
                for student_index, student in enumerate(market.students)
                if program.program_id in student.programs
            )
            for program in market.programs
        }
        interested[_OUTSIDE] = tuple(range(len(market.students)))
        school_orders = {
            program.program_id: sample.school_orders[program_index]
            for program_index, program in enumerate(market.programs)
        }
        school_orders[_OUTSIDE] = interested[_OUTSIDE]

        better = {}
        for program_id, order in school_orders.items():
            if set(order) != set(interested[program_id]) or len(order) != len(
                interested[program_id]
            ):
                raise ValueError(
                    f"SAA school order for {program_id!r} is not a permutation."
                )
            previous = None
            for rank, student_index in enumerate(order):
                cumulative = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"better_{program_id}_{rank}",
                )
                if previous is None:
                    model.addConstr(cumulative == 0.0)
                else:
                    previous_student = order[rank - 1]
                    model.addConstr(
                        cumulative
                        == previous + variables[(previous_student, program_id)]
                    )
                better[(student_index, program_id)] = cumulative
                previous = cumulative

        for program_id, capacity in self.capacities.items():
            self.capacity_constraints[program_id] = model.addConstr(
                gp.quicksum(
                    variables[(student_index, program_id)]
                    for student_index in interested[program_id]
                )
                <= capacity,
                name=f"capacity_{program_id}",
            )

        for student_index, preferences in enumerate(self.preferences):
            for program_id in preferences:
                key = (student_index, program_id)
                capacity = self.capacities[program_id]
                self.access_constraints[key] = model.addConstr(
                    variables[key] <= 1.0,
                    name=f"access_{student_index}_{program_id}",
                )
                self.stability_constraints[key] = model.addConstr(
                    better[key] + capacity * preferred[key] >= capacity,
                    name=f"stability_{student_index}_{program_id}",
                )

        objective = gp.quicksum(
            utility * variables[(student_index, program_id)]
            for student_index, student in enumerate(market.students)
            for program_id, utility in zip(student.programs, student.utilities)
        )
        model.setObjective(objective, GRB.MAXIMIZE)
        self.model = model

    def solve(self, zoning: dict[int, int]) -> SaaOracleResult:
        access = access_values(self.market, self.problem, zoning)
        for student_index, preferences in enumerate(self.preferences):
            student = self.market.students[student_index]
            for program_id in preferences:
                value = _edge_access(
                    student.node, self.programs.get(program_id), access
                )
                key = (student_index, program_id)
                self.access_constraints[key].RHS = value
                self.stability_constraints[key].RHS = (
                    self.capacities[program_id] * value
                )
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"SAA recourse LP {self.sample_index} did not solve to optimality "
                f"(status={self.model.Status})."
            )

        constant = sum(
            constraint.Pi for constraint in self.assignment_constraints.values()
        ) + sum(
            self.capacities[program_id] * constraint.Pi
            for program_id, constraint in self.capacity_constraints.items()
        )
        coefficients: defaultdict[AccessPair, float] = defaultdict(float)
        for student_index, preferences in enumerate(self.preferences):
            student = self.market.students[student_index]
            for program_id in preferences:
                key = (student_index, program_id)
                coefficient = self.access_constraints[key].Pi + (
                    self.capacities[program_id] * self.stability_constraints[key].Pi
                )
                pair, fixed = access_state(
                    self.problem,
                    student.node,
                    self.programs.get(program_id),
                )
                if pair is None:
                    constant += coefficient * fixed
                else:
                    coefficients[pair] += coefficient

        cut = SaaCut(
            sample_index=self.sample_index,
            constant=float(constant),
            coefficients=tuple(sorted(coefficients.items())),
            anchor_access=tuple((pair, access[pair]) for pair in sorted(coefficients)),
        )
        welfare = float(self.model.ObjVal)
        if not math.isclose(cut.value(access), welfare, rel_tol=1e-7, abs_tol=1e-6):
            raise RuntimeError("SAA recourse primal and dual objectives disagree.")
        return SaaOracleResult(welfare=welfare, cut=cut)


def access_values(
    market: SaaMarket,
    problem: ZoneProblem,
    zoning: dict[int, int],
) -> dict[AccessPair, int]:
    values = {}
    for student in market.students:
        for program_id in student.programs:
            program = market.program_by_id[program_id]
            pair, fixed = access_state(problem, student.node, program)
            if pair is not None:
                values[pair] = int(zoning[pair[0]] == zoning[pair[1]])
            elif not fixed and not program.citywide:
                values[(student.node, program.school_node)] = 0
    return values


def access_state(
    problem: ZoneProblem,
    student_node: int,
    program: MidProgram | None,
) -> tuple[AccessPair | None, int]:
    if program is None or program.citywide:
        return None, 1
    return restricted_access_state(problem, student_node, program.school_node)


def restricted_access_state(
    problem: ZoneProblem,
    student_node: int,
    school_node: int,
) -> tuple[AccessPair | None, int]:
    if student_node == school_node:
        return None, 1
    pair = (student_node, school_node)
    if not (
        problem.candidate_zones(student_node) & problem.candidate_zones(school_node)
    ):
        return None, 0
    return pair, 0


def required_access_pairs(market: SaaMarket) -> set[AccessPair]:
    return {
        (student.node, market.program_by_id[program_id].school_node)
        for student in market.students
        for program_id in student.programs
        if not market.program_by_id[program_id].citywide
    }


def _edge_access(
    student_node: int,
    program: MidProgram | None,
    access: dict[AccessPair, int],
) -> int:
    if program is None or program.citywide or student_node == program.school_node:
        return 1
    return access.get((student_node, program.school_node), 0)
