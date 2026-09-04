"""Finite student markets and sampled school preferences for SAA."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from optimization.data.mid import (
    MidMarket,
    MidProgram,
    MidStudent,
    build_mid_student_market,
    compress_mid_students,
)
from optimization.problem import ZoneProblem


SAA_TIE_BREAKING_METHODS = frozenset({"STB", "MTB"})


@dataclass(frozen=True)
class SaaMarket:
    programs: tuple[MidProgram, ...]
    students: tuple[MidStudent, ...]
    utility_student_count: int
    utility_handling: str

    def __post_init__(self) -> None:
        program_ids = [program.program_id for program in self.programs]
        if len(program_ids) != len(set(program_ids)):
            raise ValueError("SAA program identities must be unique.")
        known = set(program_ids)
        for student in self.students:
            lengths = {
                len(student.programs),
                len(student.priorities),
                len(student.utilities),
                len(student.scaled_utilities),
            }
            if lengths != {len(student.programs)}:
                raise ValueError("SAA student preference fields must align.")
            if len(student.programs) != len(set(student.programs)):
                raise ValueError("SAA student preferences cannot repeat a program.")
            if not set(student.programs) <= known:
                raise ValueError("SAA students contain unknown programs.")

    @property
    def program_by_id(self) -> dict[str, MidProgram]:
        return {program.program_id: program for program in self.programs}

    @property
    def welfare_upper_bound(self) -> float:
        return sum(
            student.utilities[0] if student.utilities else 0.0
            for student in self.students
        )

    @property
    def preference_count(self) -> int:
        return sum(len(student.programs) for student in self.students)


@dataclass(frozen=True)
class SaaSample:
    seed: int
    school_orders: tuple[tuple[int, ...], ...]


def build_saa_market(problem: ZoneProblem, optimization_config) -> SaaMarket:
    """Build the individual market using MID's utility and access semantics."""
    source = build_mid_student_market(problem, optimization_config)
    return preprocess_saa_market(
        SaaMarket(
            programs=source.programs,
            students=source.students,
            utility_student_count=source.utility_student_count,
            utility_handling=source.utility_handling,
        ),
        problem,
    )


def saa_market_to_mid_market(market: SaaMarket) -> MidMarket:
    """Convert an SAA market into a compressed MID market."""
    return MidMarket(
        programs=market.programs,
        types=compress_mid_students(market.students),
        student_count=len(market.students),
        outside_only_student_count=sum(
            not student.programs for student in market.students
        ),
        utility_student_count=market.utility_student_count,
        utility_handling=market.utility_handling,
    )


def preprocess_saa_market(market: SaaMarket, problem: ZoneProblem) -> SaaMarket:
    """Remove zero-capacity and permanently inaccessible alternatives."""
    programs = {
        program.program_id: program
        for program in market.programs
        if program.capacity > 0
    }
    possible_access: dict[tuple[int, int], bool] = {}
    students = []
    referenced_programs = set()
    for student in market.students:
        kept = []
        for rank, program_id in enumerate(student.programs):
            program = programs.get(program_id)
            if program is None:
                continue
            if not program.citywide:
                key = (student.node, program.school_node)
                if key not in possible_access:
                    possible_access[key] = bool(
                        problem.candidate_zones(student.node)
                        & problem.candidate_zones(program.school_node)
                    )
                if not possible_access[key]:
                    continue
            kept.append(rank)
            referenced_programs.add(program_id)
        students.append(
            MidStudent(
                node=student.node,
                programs=tuple(student.programs[rank] for rank in kept),
                priorities=tuple(student.priorities[rank] for rank in kept),
                utilities=tuple(student.utilities[rank] for rank in kept),
                scaled_utilities=tuple(student.scaled_utilities[rank] for rank in kept),
            )
        )
    return SaaMarket(
        programs=tuple(
            program
            for program in market.programs
            if program.program_id in referenced_programs
        ),
        students=tuple(students),
        utility_student_count=market.utility_student_count,
        utility_handling=market.utility_handling,
    )


def sample_school_preferences(
    market: SaaMarket,
    num_seeds: int,
    tie_breaking_method: str,
    base_seed: int,
) -> tuple[SaaSample, ...]:
    """Draw strict program priorities under single or multiple tie-breaking."""
    if isinstance(num_seeds, bool) or not isinstance(num_seeds, int) or num_seeds <= 0:
        raise ValueError("SAA num_seeds must be a positive integer.")
    method = str(tie_breaking_method).upper()
    if method not in SAA_TIE_BREAKING_METHODS:
        raise ValueError("SAA tie_breaking_method must be one of: MTB, STB.")

    program_number = {
        program.program_id: index for index, program in enumerate(market.programs)
    }
    priorities = [
        dict(zip(student.programs, student.priorities)) for student in market.students
    ]
    interested = {
        program.program_id: tuple(
            student_index
            for student_index, student in enumerate(market.students)
            if program.program_id in priorities[student_index]
        )
        for program in market.programs
    }

    samples = []
    for sample_index in range(num_seeds):
        seed = _sample_seed(base_seed, sample_index)
        random = np.random.RandomState(seed)
        if method == "STB":
            lotteries = random.random_sample((len(market.students), 1))
        else:
            lotteries = random.random_sample(
                (len(market.students), len(market.programs))
            )
        orders = []
        for program in market.programs:
            column = 0 if method == "STB" else program_number[program.program_id]
            orders.append(
                tuple(
                    sorted(
                        interested[program.program_id],
                        key=lambda student_index: (
                            -priorities[student_index][program.program_id],
                            -lotteries[student_index, column],
                            student_index,
                        ),
                    )
                )
            )
        samples.append(SaaSample(seed=seed, school_orders=tuple(orders)))
    return tuple(samples)


def _sample_seed(base_seed: int, sample_index: int) -> int:
    payload = f"sfusd-saa-v1:{int(base_seed)}:{sample_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
