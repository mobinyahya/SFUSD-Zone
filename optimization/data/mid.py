"""Program market data for the MID welfare strategy."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from loaders import (
    load_program_records,
    load_school_records,
    load_student_records,
    read_csv_source,
)

from assignment.student_assignment.data_interfaces import Programs, Schools, Students
from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)
from assignment.student_assignment.market_generator.utility_model import UtilityModel
from optimization.problem import ZoneProblem
from optimization.solvers.cpsat import CP_SAT_SCALE


MID_UTILITY_HANDLING = frozenset({"omit_nonpositive", "exponentiate"})


@dataclass(frozen=True)
class MidProgram:
    program_id: str
    school_id: int
    capacity: int
    citywide: bool
    school_node: int | None


@dataclass(frozen=True)
class MidStudent:
    node: int
    programs: tuple[str, ...]
    priorities: tuple[int, ...]
    utilities: tuple[float, ...]
    scaled_utilities: tuple[int, ...]


@dataclass(frozen=True)
class MidType:
    node: int
    count: int
    programs: tuple[str, ...]
    priorities: tuple[int, ...]
    utility_sums: tuple[float, ...]
    scaled_utility_sums: tuple[int, ...]


@dataclass(frozen=True)
class MidMarket:
    programs: tuple[MidProgram, ...]
    types: tuple[MidType, ...]
    student_count: int
    outside_only_student_count: int
    utility_student_count: int
    utility_handling: str
    utility_scale: int = CP_SAT_SCALE

    def __post_init__(self) -> None:
        program_ids = [program.program_id for program in self.programs]
        if len(program_ids) != len(set(program_ids)):
            raise ValueError("MID program identities must be unique.")
        known = set(program_ids)
        for program in self.programs:
            if (
                isinstance(program.capacity, bool)
                or not isinstance(program.capacity, int)
                or program.capacity < 0
            ):
                raise ValueError(
                    "MID program capacities must be non-negative integers."
                )
            if not program.citywide and program.school_node is None:
                raise ValueError("Restricted MID programs require a school node.")
        for student_type in self.types:
            lengths = {
                len(student_type.programs),
                len(student_type.priorities),
                len(student_type.utility_sums),
                len(student_type.scaled_utility_sums),
            }
            if lengths != {len(student_type.programs)}:
                raise ValueError("MID type preference fields must align.")
            if (
                isinstance(student_type.count, bool)
                or not isinstance(student_type.count, int)
                or student_type.count <= 0
            ):
                raise ValueError("MID type counts must be positive integers.")
            if not set(student_type.programs) <= known:
                raise ValueError("MID types contain unknown programs.")
            if len(student_type.programs) != len(set(student_type.programs)):
                raise ValueError("MID type preferences cannot repeat a program.")
            if any(
                isinstance(priority, bool)
                or not isinstance(priority, int)
                or priority < 0
                for priority in student_type.priorities
            ):
                raise ValueError("MID priority tiers must be non-negative integers.")
            if any(
                not math.isfinite(utility) or utility <= 0
                for utility in student_type.utility_sums
            ):
                raise ValueError("MID utility sums must be positive and finite.")
            if any(
                isinstance(utility, bool)
                or not isinstance(utility, int)
                or utility <= 0
                for utility in student_type.scaled_utility_sums
            ):
                raise ValueError("MID scaled utility sums must be positive integers.")
            if any(
                first < second
                for first, second in zip(
                    student_type.utility_sums,
                    student_type.utility_sums[1:],
                )
            ):
                raise ValueError("MID utility sums must be non-increasing by rank.")
            if any(
                first < second
                for first, second in zip(
                    student_type.scaled_utility_sums,
                    student_type.scaled_utility_sums[1:],
                )
            ):
                raise ValueError(
                    "MID scaled utility sums must be non-increasing by rank."
                )
        if sum(student_type.count for student_type in self.types) != self.student_count:
            raise ValueError("MID type counts must sum to the student count.")

    @property
    def program_by_id(self) -> dict[str, MidProgram]:
        return {program.program_id: program for program in self.programs}

    @property
    def preference_count(self) -> int:
        return sum(len(student_type.programs) for student_type in self.types)


@dataclass(frozen=True)
class MidStudentMarket:
    programs: tuple[MidProgram, ...]
    students: tuple[MidStudent, ...]
    utility_student_count: int
    utility_handling: str


class _StaticZones:
    def __init__(self) -> None:
        self._zone_priority_matrix = None

    @property
    def zone_priority_matrix(self):
        return self._zone_priority_matrix


def build_mid_student_market(
    problem: ZoneProblem, optimization_config
) -> MidStudentMarket:
    """Load the assignment cohort before MID type compression."""
    handling = optimization_config.mid_utility_handling
    if handling not in MID_UTILITY_HANDLING:
        raise ValueError(f"Unknown MID utility handling {handling!r}.")

    assignment_market = _assignment_market(optimization_config)
    priority_config = assignment_market.config
    policies = priority_config["policies"]
    ctip_options = priority_config["ctip-options"]
    if len(policies) != 1 or len(ctip_options) != 1:
        raise ValueError(
            "Matching welfare requires one status-quo policy and CTIP option."
        )
    priorities = PriorityGenerator(assignment_market).get_base_policy_priorities(
        policies[0],
        ctip_options[0],
        zone_priority_matrix=_attendance_area_priority_matrix(assignment_market),
    )

    utility = _load_utility_table(
        optimization_config.data_scenario,
        assignment_market.students.student_data.index,
        assignment_market.programs.program_df["program_id"],
    )
    program_rows = assignment_market.programs.program_df.copy()
    available_programs = sorted(set(program_rows["program_id"]) & set(utility.columns))
    program_rows = program_rows.set_index("program_id")
    if not program_rows.index.is_unique:
        raise ValueError("Assignment program identities are not unique.")
    citywide_schools = {
        int(value) for value in assignment_market.schools.citywide_schools
    }
    school_to_node = _school_to_node(problem)
    area_to_node = _area_to_node(problem)

    programs = []
    for program_id in available_programs:
        row = program_rows.loc[program_id]
        school_id = _integer(row["school_id"], f"school for program {program_id}")
        capacity = _integer(row["capacity"], f"capacity for program {program_id}")
        if capacity < 0:
            raise ValueError(f"Program {program_id} has negative capacity.")
        citywide = school_id in citywide_schools
        school_node = None if citywide else school_to_node.get(school_id)
        if not citywide and school_node is None:
            school_area = _area_key(row.get(problem.level.unit))
            school_node = area_to_node.get(school_area)
        if not citywide and school_node is None:
            raise ValueError(
                f"Restricted program {program_id} school {school_id} is absent "
                f"from graph {problem.level.name}."
            )
        programs.append(
            MidProgram(program_id, school_id, capacity, citywide, school_node)
        )

    area_column = _student_area_column(problem)
    students_frame = assignment_market.students.student_data
    if area_column not in students_frame:
        raise ValueError(f"Assignment students lack {area_column!r}.")
    program_indices = assignment_market.programs.indices
    students = []
    utility_students = 0
    for student_id, row in students_frame.iterrows():
        area = _area_key(row[area_column])
        node = area_to_node.get(area)
        if node is None:
            raise ValueError(
                f"Assignment student {student_id} geography {area!r} is absent "
                f"from graph {problem.level.name}."
            )
        student_key = _identity_text(student_id)
        if student_key not in utility.index:
            student = MidStudent(node, (), (), (), ())
        else:
            utility_students += 1
            values = utility.loc[student_key, available_programs].to_numpy(dtype=float)
            tiers = np.asarray(
                [
                    priorities[
                        assignment_market.students.studentno2idx[student_id],
                        program_indices[p] - 1,
                    ]
                    for p in available_programs
                ],
                dtype=float,
            )
            student = make_mid_student(
                node, available_programs, values, tiers, handling
            )
        students.append(student)

    return MidStudentMarket(
        programs=tuple(programs),
        students=tuple(students),
        utility_student_count=utility_students,
        utility_handling=handling,
    )


def build_mid_market(problem: ZoneProblem, optimization_config) -> MidMarket:
    """Load the assignment cohort and compress it into MID market types."""
    student_market = build_mid_student_market(problem, optimization_config)
    return MidMarket(
        programs=student_market.programs,
        types=compress_mid_students(student_market.students),
        student_count=len(student_market.students),
        outside_only_student_count=sum(
            not student.programs for student in student_market.students
        ),
        utility_student_count=student_market.utility_student_count,
        utility_handling=student_market.utility_handling,
    )


def make_mid_student(
    node: int,
    program_ids,
    utility_values,
    priority_values,
    handling: str,
) -> MidStudent:
    """Transform one utility row into deterministic preferences."""
    if handling not in MID_UTILITY_HANDLING:
        raise ValueError(f"Unknown MID utility handling {handling!r}.")
    program_ids = [str(value) for value in program_ids]
    utilities = np.asarray(utility_values, dtype=float)
    priorities = np.asarray(priority_values, dtype=float)
    if utilities.shape != (len(program_ids),) or priorities.shape != utilities.shape:
        raise ValueError("MID utility, priority, and program rows must align.")
    if np.isnan(utilities).any() or np.isposinf(utilities).any():
        raise ValueError("MID utilities cannot contain NaN or positive infinity.")

    finite = np.isfinite(utilities)
    if handling == "omit_nonpositive":
        included = finite & (utilities > 0)
        transformed = utilities
    else:
        included = finite
        transformed = np.zeros_like(utilities)
        if included.any():
            transformed[included] = np.maximum(
                np.exp(utilities[included] - utilities[included].max()),
                np.nextafter(0.0, 1.0),
            )

    alternatives = []
    for index in np.flatnonzero(included):
        priority = _integer(priorities[index], f"priority for {program_ids[index]}")
        if priority < 0:
            raise ValueError(f"Program {program_ids[index]} has negative priority.")
        value = float(transformed[index])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Included MID utilities must be positive and finite.")
        alternatives.append(
            (program_ids[index], float(utilities[index]), value, priority)
        )

    alternatives.sort(key=lambda item: (-item[1], item[0]))
    return MidStudent(
        node=int(node),
        programs=tuple(item[0] for item in alternatives),
        priorities=tuple(item[3] for item in alternatives),
        utilities=tuple(item[2] for item in alternatives),
        scaled_utilities=tuple(
            max(1, round(CP_SAT_SCALE * item[2])) for item in alternatives
        ),
    )


def compress_mid_students(students) -> tuple[MidType, ...]:
    """Aggregate students with identical assignment recurrences."""
    grouped: dict[tuple, list] = {}
    for student in students:
        key = (student.node, student.programs, student.priorities)
        if key not in grouped:
            grouped[key] = [
                0,
                [0.0] * len(student.programs),
                [0] * len(student.programs),
            ]
        aggregate = grouped[key]
        aggregate[0] += 1
        for rank, utility in enumerate(student.utilities):
            aggregate[1][rank] += utility
            aggregate[2][rank] += student.scaled_utilities[rank]

    types = []
    for (node, programs, priorities), (count, utilities, scaled) in grouped.items():
        types.append(
            MidType(
                node=node,
                count=count,
                programs=programs,
                priorities=priorities,
                utility_sums=tuple(utilities),
                scaled_utility_sums=tuple(scaled),
            )
        )
    return tuple(
        sorted(types, key=lambda item: (item.node, item.programs, item.priorities))
    )


def preprocess_mid_market(market: MidMarket, problem: ZoneProblem) -> MidMarket:
    """Remove alternatives that cannot receive positive assignment mass."""

    programs = {
        program.program_id: program
        for program in market.programs
        if program.capacity > 0
    }
    possible_access: dict[tuple[int, int], bool] = {}
    grouped: dict[tuple, list] = {}
    referenced_programs = set()

    for student_type in market.types:
        kept = []
        for rank, program_id in enumerate(student_type.programs):
            program = programs.get(program_id)
            if program is None:
                continue
            if not program.citywide:
                access_key = (student_type.node, program.school_node)
                if access_key not in possible_access:
                    possible_access[access_key] = bool(
                        problem.candidate_zones(student_type.node)
                        & problem.candidate_zones(program.school_node)
                    )
                if not possible_access[access_key]:
                    continue
            kept.append(rank)
            referenced_programs.add(program_id)

        program_ids = tuple(student_type.programs[rank] for rank in kept)
        priorities = tuple(student_type.priorities[rank] for rank in kept)
        key = (student_type.node, program_ids, priorities)
        if key not in grouped:
            grouped[key] = [
                0,
                [0.0] * len(kept),
                [0] * len(kept),
            ]
        aggregate = grouped[key]
        aggregate[0] += student_type.count
        for target_rank, source_rank in enumerate(kept):
            aggregate[1][target_rank] += student_type.utility_sums[source_rank]
            aggregate[2][target_rank] += student_type.scaled_utility_sums[source_rank]

    types = tuple(
        MidType(
            node=node,
            count=values[0],
            programs=program_ids,
            priorities=priorities,
            utility_sums=tuple(values[1]),
            scaled_utility_sums=tuple(values[2]),
        )
        for (node, program_ids, priorities), values in sorted(grouped.items())
    )
    return MidMarket(
        programs=tuple(
            program
            for program in market.programs
            if program.program_id in referenced_programs
        ),
        types=types,
        student_count=market.student_count,
        outside_only_student_count=sum(
            student_type.count for student_type in types if not student_type.programs
        ),
        utility_student_count=market.utility_student_count,
        utility_handling=market.utility_handling,
        utility_scale=market.utility_scale,
    )


def _assignment_market(optimization_config):
    root = Path(__file__).resolve().parents[2]
    with (root / "assignment/configs/base_config.yaml").open(
        encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    with (root / "assignment/configs/policy_configs/status_quo.yaml").open(
        encoding="utf-8"
    ) as stream:
        config.update(yaml.safe_load(stream))
    data = optimization_config.data_scenario
    filters = data.filters["assignment"]
    config["year"] = int(filters["year"][:2])
    config["grade"] = filters["grades"][0]
    config["special_programs"] = filters["special_programs"]

    ctip_sources = data.source_map("assignment.ctip")
    config.setdefault("paths", {}).update(
        {
            "new-ctip-path": str(ctip_sources["block"].path),
            "new-ctip-blockgroup-path": str(ctip_sources["blockgroup"].path),
        }
    )
    program_data = load_program_records(
        data, "assignment.programs", filter_group="assignment"
    )
    program_codes = None
    if "programno" not in program_data:
        program_codes = read_csv_source(data.source("assignment.program_codes"))
    programs = Programs(program_data, program_codes, config)
    schools = Schools(
        load_school_records(data, "assignment.schools", filter_group="assignment"),
        programs,
    )
    students = Students(
        student_data_file=load_student_records(
            data,
            "assignment.students",
            filter_group="assignment",
            low_memory=False,
        ),
        programs=programs,
        school_data_file=load_school_records(
            data,
            "assignment.school_coordinates",
            filter_group="assignment",
        ),
        block_data_file=ctip_sources["workbook"].path,
        config=config,
        data_scenario=data,
    )
    return SimpleNamespace(
        config=config,
        programs=programs,
        schools=schools,
        students=students,
        n=students.n,
        num_programs=programs.num_programs,
        zones=_StaticZones(),
    )


def _attendance_area_priority_matrix(market) -> np.ndarray:
    """Build the status-quo singleton attendance-area GE priority component."""
    matrix = np.zeros((market.n, market.num_programs), dtype=int)
    grade = market.config["grade"]
    for student_id, row in market.students.student_data.iterrows():
        school_id = row.get("idschoolattendance")
        if pd.isna(school_id):
            continue
        program_id = f"{int(school_id)}-GE-{grade}"
        program_index = market.programs.indices.get(program_id)
        if program_index is not None:
            matrix[market.students.studentno2idx[student_id], program_index - 1] = 1
    return matrix


def _load_utility_table(data, student_ids, program_ids) -> pd.DataFrame:
    source = data.source("choice.estimate")
    UtilityModel._validate_csv_header(source.path)
    frame = read_csv_source(source, dtype={"studentno": "string"})
    student_keys = [_identity_text(value) for value in student_ids]
    program_keys = [_identity_text(value) for value in program_ids]
    if len(student_keys) != len(set(student_keys)):
        raise ValueError(
            "Assignment student identities are ambiguous after normalization."
        )
    if len(program_keys) != len(set(program_keys)):
        raise ValueError(
            "Assignment program identities are ambiguous after normalization."
        )
    required_students = set(student_keys)
    required_programs = set(program_keys)

    student_labels = [
        _normalize_labeled_identity(value, required_students)
        for value in frame.pop("studentno")
    ]
    if len(student_labels) != len(set(student_labels)):
        raise ValueError("Choice utility CSV has duplicate student rows.")
    frame.index = pd.Index(student_labels, name="studentno")

    program_labels = [
        _normalize_labeled_identity(value, required_programs) for value in frame.columns
    ]
    if len(program_labels) != len(set(program_labels)):
        raise ValueError("Choice utility CSV has duplicate program columns.")
    frame.columns = program_labels
    selected_programs = sorted(required_programs & set(program_labels))
    selected_students = sorted(required_students & set(student_labels))
    numeric = frame.loc[selected_students, selected_programs].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.to_numpy(dtype=float)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Choice utility CSV contains NaN or positive infinity.")
    return numeric


def _identity_text(value) -> str:
    if pd.isna(value):
        raise ValueError("MID identities cannot be null.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    value = str(value).strip()
    if not value:
        raise ValueError("MID identities cannot be empty.")
    return value


def _normalize_labeled_identity(value, required: set[str]) -> str:
    identity = _identity_text(value)
    if identity in required:
        return identity
    match = re.fullmatch(r"(?:\d{2}|\d{4})-(.+)", identity)
    return match.group(1) if match else identity


def _integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"MID {label} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MID {label} must be an integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"MID {label} must be an integer.")
    return int(numeric)


def _student_area_column(problem: ZoneProblem) -> str:
    return {
        "Block": "census_block",
        "BlockGroup": "census_blockgroup",
        "Tract": "census_tract",
    }.get(problem.level.unit, f"census_{problem.level.unit.lower()}")


def _area_key(value) -> str | None:
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        value = str(value).strip()
        return value or None


def _area_to_node(problem: ZoneProblem) -> dict[str, int]:
    result = {}
    for node, attrs in problem.G.nodes(data=True):
        values = attrs.get("block_ids", [attrs.get("area_id")])
        for value in values:
            key = _area_key(value)
            if key is not None:
                result[key] = int(node)
    return result


def _school_to_node(problem: ZoneProblem) -> dict[int, int]:
    return {
        int(school_id): int(node)
        for node, attrs in problem.G.nodes(data=True)
        for school_id in attrs.get("school_ids", [])
    }
