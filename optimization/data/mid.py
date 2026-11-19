"""Program market data for the MID welfare strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)
from assignment.student_assignment.market_generator.school_choice_market import (
    load_market_data,
)
from choice.estimate import area_key as _area_key
from choice.estimate import area_to_node as _area_to_node
from choice.estimate import identity_text as _identity_text
from choice.estimate import load_estimate
from choice.estimate import school_to_node as _school_to_node
from choice.estimate import student_area_column as _student_area_column
from optimization.mid_options import MID_PREFERENCE_SOURCES
from optimization.problem import ZoneProblem
from optimization.solvers.cpsat import CP_SAT_SCALE


MID_UTILITY_HANDLING = frozenset({"omit_nonpositive", "exponentiate"})


@dataclass(frozen=True)
class MidProgram:
    program_id: str
    school_id: int
    capacity: int
    #: Open to every zone: the program holds no attendance-area seats, even
    #: when its school is an attendance-area school.
    citywide: bool
    school_node: int | None
    #: How many of ``capacity`` are attendance-area seats (the program table's
    #: ``attendance_area_capacity``); ``None`` when the table does not say.
    attendance_area_capacity: int | None = None


@dataclass(frozen=True)
class MidStudent:
    node: int
    programs: tuple[str, ...]
    priorities: tuple[int, ...]
    utilities: tuple[float, ...]
    scaled_utilities: tuple[int, ...]
    #: The student's free/reduced-lunch probability (``Students.frl``, after
    #: any ``frl_estimate``). It rides along with the type so school-level FRL
    #: composition can be read off a match; it never changes the match.
    frl: float = 0.0
    #: Policy access (``OptimizationConfig.matching_policy``): per alternative,
    #: the priority tier the student holds at a zoned program *outside* their
    #: zone, or -1 when the policy gives them no access there. ``priorities``
    #: is then the in-zone tier. ``None`` (no policy): zoned programs are
    #: reachable only in zone, at ``priorities``.
    open_priorities: tuple[int, ...] | None = None
    #: Set only in reserve markets, which keep one type per student.
    student_id: str | None = None
    reserve_class: int = -1


@dataclass(frozen=True)
class MidType:
    node: int
    count: int
    programs: tuple[str, ...]
    priorities: tuple[int, ...]
    utility_sums: tuple[float, ...]
    scaled_utility_sums: tuple[int, ...]
    #: Sum of the members' ``MidStudent.frl``: the type's expected FRL count.
    #: Every member follows the same path through the match, so a school's FRL
    #: count is ``sum_t frl_sum_t * mass_t / lottery_scale``.
    frl_sum: float = 0.0
    #: See ``MidStudent``; types differing in any of these never merge.
    open_priorities: tuple[int, ...] | None = None
    student_id: str | None = None
    reserve_class: int = -1


@dataclass(frozen=True)
class ReserveSpec:
    """A matching policy's reserves (``guard-rails`` >= 0), as the DA runs them.

    ``fractions`` maps each reserved program to its per-class seat fractions
    (``assignment.student_assignment.da.guardrail_setup``); a program absent
    from it has none. Every type of a reserve market is one student, matched
    by a discrete deferred acceptance on one lottery drawn from ``seed``
    (``optimization.reserve_da``).
    """

    fractions: dict[str, tuple[float, ...]]
    num_classes: int
    strict: bool
    seed: int


@dataclass(frozen=True)
class MidMarket:
    programs: tuple[MidProgram, ...]
    types: tuple[MidType, ...]
    student_count: int
    outside_only_student_count: int
    utility_student_count: int
    utility_handling: str
    utility_scale: int = CP_SAT_SCALE
    #: The ``matching_policy`` the market was built for (``None``: status quo).
    policy: str | None = None
    reserves: ReserveSpec | None = None

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
            if student_type.open_priorities is not None and (
                len(student_type.open_priorities) != len(student_type.programs)
                or any(
                    isinstance(tier, bool) or not isinstance(tier, int) or tier < -1
                    for tier in student_type.open_priorities
                )
            ):
                raise ValueError(
                    "MID open priorities must align with the preferences and be "
                    "integers of at least -1."
                )
            if self.reserves is not None and (
                student_type.count != 1
                or student_type.student_id is None
                or not 0 <= student_type.reserve_class < self.reserves.num_classes
            ):
                raise ValueError(
                    "A reserve market keeps one student per type, each with a "
                    "reserve class."
                )
        if sum(student_type.count for student_type in self.types) != self.student_count:
            raise ValueError("MID type counts must sum to the student count.")

    @property
    def program_by_id(self) -> dict[str, MidProgram]:
        return {program.program_id: program for program in self.programs}

    @property
    def preference_count(self) -> int:
        return sum(len(student_type.programs) for student_type in self.types)

    @property
    def policy_access(self) -> bool:
        """Whether access or priority depends on more than the seat label."""
        return self.reserves is not None or any(
            student_type.open_priorities is not None for student_type in self.types
        )


def require_label_access(market: MidMarket, label: str) -> None:
    """Fail for a model that reads access off the seat label alone.

    The CP/MIP welfare models, SAA and the zoned transport bound know
    attendance-area seats only as "in zone or not"; a policy market's
    out-of-zone access and reserves are modelled by the least-cutoff oracle
    and the reserve DA, which the ReCom scorers and capacity references use.
    """
    if getattr(market, "policy_access", False):
        raise ValueError(
            f"{label} does not model matching policy {market.policy!r} "
            "(out-of-zone access or reserves); use it with ReCom objectives."
        )


@dataclass(frozen=True)
class MidStudentMarket:
    programs: tuple[MidProgram, ...]
    students: tuple[MidStudent, ...]
    utility_student_count: int
    utility_handling: str
    policy: str | None = None
    reserves: ReserveSpec | None = None


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

    policy_name = getattr(optimization_config, "matching_policy", None)
    assignment_market = _assignment_market(optimization_config)
    priority_config = assignment_market.config
    policies = priority_config["policies"]
    ctip_options = priority_config["ctip-options"]
    if len(policies) != 1 or len(ctip_options) != 1:
        raise ValueError(
            "The MID market requires a policy with one zone policy and one CTIP option."
        )
    open_priorities = None
    if policy_name is None:
        priorities = PriorityGenerator(assignment_market).get_base_policy_priorities(
            policies[0],
            ctip_options[0],
            zone_priority_matrix=_attendance_area_priority_matrix(assignment_market),
        )
    else:
        priorities, open_priorities = policy_priorities(assignment_market)

    source = getattr(optimization_config, "mid_preference_source", "estimate")
    if source not in MID_PREFERENCE_SOURCES:
        raise ValueError(f"Unknown MID preference source {source!r}.")
    load = _load_utility_table if source == "estimate" else _stated_utility_table
    utility = load(
        optimization_config.data_scenario,
        assignment_market.students.student_data.index,
        assignment_market.programs.program_df["program_id"],
        assignment_market,
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

    # The MID market is drawn from the assignment program table, which has no
    # citywide selector of its own, so the optimization filter is applied here.
    # This is the *choice* selector, not the zoning one: a citywide program
    # keeps ``school_node=None`` and every oracle reads that as reachable from
    # every zone, which is what a citywide school is. Whether it also occupies
    # a graph node and contributes to the capacity and school-count balance is
    # ``include_citywide_zoning``, applied in ``loaders._attach_capacity``.
    # The two are deliberately independent -- the default pairing is a citywide
    # school that no zone owns but every student can still choose.
    keep_citywide = bool(optimization_config.include_citywide_choice_opt)

    programs = []
    retained_programs = []
    for program_id in available_programs:
        row = program_rows.loc[program_id]
        school_id = _integer(row["school_id"], f"school for program {program_id}")
        capacity = _integer(row["capacity"], f"capacity for program {program_id}")
        if capacity < 0:
            raise ValueError(f"Program {program_id} has negative capacity.")
        if school_id in citywide_schools and not keep_citywide:
            continue
        # The seat label, not the school, decides access: attendance-area
        # seats are tied to the zone owning the school, citywide seats are open
        # to every zone, even at an attendance-area school. The loaders reject
        # a program holding both (loaders.tables.attach_attendance_area_capacity).
        attendance_seats = row.get("attendance_area_capacity")
        if attendance_seats is None or pd.isna(attendance_seats):
            raise ValueError(f"Program {program_id} has no attendance-area seat count.")
        attendance_seats = _integer(
            attendance_seats, f"attendance-area seats for {program_id}"
        )
        if 0 < attendance_seats < capacity:
            raise ValueError(
                f"Program {program_id} mixes {attendance_seats} attendance-area "
                f"and {capacity - attendance_seats} citywide seats, which MID "
                "does not model."
            )
        citywide = attendance_seats == 0
        retained_programs.append(program_id)
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
            MidProgram(
                program_id,
                school_id,
                capacity,
                citywide,
                school_node,
                attendance_seats,
            )
        )
    # Preference lists are read positionally off ``available_programs``, so the
    # utility and priority columns have to be narrowed in the same order.
    available_programs = retained_programs

    reserves, classes = None, None
    if policy_name is not None and priority_config.get("guard-rails", -1) != -1:
        reserves, classes = policy_reserves(
            assignment_market, programs, optimization_config
        )

    area_column = _student_area_column(problem)
    students_frame = assignment_market.students.student_data
    if area_column not in students_frame:
        raise ValueError(f"Assignment students lack {area_column!r}.")
    program_indices = assignment_market.programs.indices
    frl_values = assignment_market.students.frl
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
        frl = float(frl_values.loc[student_id])
        row_index = assignment_market.students.studentno2idx[student_id]
        if student_key not in utility.index:
            student = MidStudent(
                node, (), (), (), (), frl, None if open_priorities is None else ()
            )
        else:
            utility_students += 1
            values = utility.loc[student_key, available_programs].to_numpy(dtype=float)
            columns = [program_indices[p] - 1 for p in available_programs]
            tiers = np.asarray(priorities[row_index, columns], dtype=float)
            open_tiers = (
                None
                if open_priorities is None
                else np.asarray(open_priorities[row_index, columns], dtype=float)
            )
            student = replace(
                make_mid_student(
                    node, available_programs, values, tiers, handling, open_tiers
                ),
                frl=frl,
            )
        if reserves is not None:
            student = replace(
                student,
                student_id=student_key,
                reserve_class=int(classes[row_index]),
            )
        students.append(student)

    return MidStudentMarket(
        programs=tuple(programs),
        students=tuple(students),
        utility_student_count=utility_students,
        utility_handling=handling,
        policy=policy_name,
        reserves=reserves,
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
        policy=student_market.policy,
        reserves=student_market.reserves,
    )


def make_mid_student(
    node: int,
    program_ids,
    utility_values,
    priority_values,
    handling: str,
    open_priority_values=None,
) -> MidStudent:
    """Transform one utility row into deterministic preferences.

    ``open_priority_values``, aligned with ``priority_values``, are a policy's
    out-of-zone tiers (-1: no access); see ``MidStudent.open_priorities``.
    """
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
        open_tier = (
            None
            if open_priority_values is None
            else _integer(open_priority_values[index], "open priority")
        )
        alternatives.append(
            (program_ids[index], float(utilities[index]), value, priority, open_tier)
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
        open_priorities=(
            None
            if open_priority_values is None
            else tuple(item[4] for item in alternatives)
        ),
    )


def compress_mid_students(students) -> tuple[MidType, ...]:
    """Aggregate students with identical assignment recurrences."""
    grouped: dict[tuple, list] = {}
    for student in students:
        key = (
            student.node,
            student.programs,
            student.priorities,
            student.open_priorities,
            student.student_id,
            student.reserve_class,
        )
        if key not in grouped:
            grouped[key] = [
                0,
                [0.0] * len(student.programs),
                [0] * len(student.programs),
                0.0,
            ]
        aggregate = grouped[key]
        aggregate[0] += 1
        aggregate[3] += student.frl
        for rank, utility in enumerate(student.utilities):
            aggregate[1][rank] += utility
            aggregate[2][rank] += student.scaled_utilities[rank]

    types = []
    for (node, programs, priorities, open_priorities, student_id, reserve_class), (
        count,
        utilities,
        scaled,
        frl_sum,
    ) in grouped.items():
        types.append(
            MidType(
                node=node,
                count=count,
                programs=programs,
                priorities=priorities,
                utility_sums=tuple(utilities),
                scaled_utility_sums=tuple(scaled),
                frl_sum=frl_sum,
                open_priorities=open_priorities,
                student_id=student_id,
                reserve_class=reserve_class,
            )
        )
    return tuple(sorted(types, key=_type_order))


def _type_order(student_type) -> tuple:
    return (
        student_type.node,
        student_type.programs,
        student_type.priorities,
        student_type.open_priorities or (),
        student_type.student_id or "",
        student_type.reserve_class,
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
        open_tiers = student_type.open_priorities
        for rank, program_id in enumerate(student_type.programs):
            program = programs.get(program_id)
            if program is None:
                continue
            if not program.citywide and not (
                open_tiers is not None and open_tiers[rank] >= 0
            ):
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
        open_kept = (
            None if open_tiers is None else tuple(open_tiers[rank] for rank in kept)
        )
        key = (
            student_type.node,
            program_ids,
            priorities,
            open_kept,
            student_type.student_id,
            student_type.reserve_class,
        )
        if key not in grouped:
            grouped[key] = [
                0,
                [0.0] * len(kept),
                [0] * len(kept),
                0.0,
            ]
        aggregate = grouped[key]
        aggregate[0] += student_type.count
        aggregate[3] += student_type.frl_sum
        for target_rank, source_rank in enumerate(kept):
            aggregate[1][target_rank] += student_type.utility_sums[source_rank]
            aggregate[2][target_rank] += student_type.scaled_utility_sums[source_rank]

    types = tuple(
        sorted(
            (
                MidType(
                    node=node,
                    count=values[0],
                    programs=program_ids,
                    priorities=priorities,
                    utility_sums=tuple(values[1]),
                    scaled_utility_sums=tuple(values[2]),
                    frl_sum=values[3],
                    open_priorities=open_kept,
                    student_id=student_id,
                    reserve_class=reserve_class,
                )
                for (
                    node,
                    program_ids,
                    priorities,
                    open_kept,
                    student_id,
                    reserve_class,
                ), values in grouped.items()
            ),
            key=_type_order,
        )
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
        policy=market.policy,
        reserves=market.reserves,
    )


POLICY_CONFIG_DIR = Path(__file__).resolve().parents[2] / (
    "assignment/configs/policy_configs"
)


def policy_config_path(name: str) -> Path:
    """The policy config a ``matching_policy`` names, which must exist."""
    path = POLICY_CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(
            f"matching_policy {name!r} names no policy config: {path} is missing."
        )
    return path


def _assignment_market(optimization_config):
    root = Path(__file__).resolve().parents[2]
    with (root / "assignment/configs/base_config.yaml").open(
        encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    policy = getattr(optimization_config, "matching_policy", None)
    policy_path = (
        POLICY_CONFIG_DIR / "status_quo.yaml"
        if policy is None
        else policy_config_path(policy)
    )
    with policy_path.open(encoding="utf-8") as stream:
        config.update(yaml.safe_load(stream))
    data = optimization_config.data_scenario
    programs, schools, students = load_market_data(data)
    return SimpleNamespace(
        config=config,
        data_scenario=data,
        grade=data.filter("assignment", "grade"),
        programs=programs,
        schools=schools,
        students=students,
        n=students.n,
        num_programs=programs.num_programs,
        zones=_StaticZones(),
    )


def policy_priorities(market) -> tuple[np.ndarray, np.ndarray]:
    """In-zone and out-of-zone priority tiers of the market's matching policy.

    The assignment simulator adds the policy's ``zone`` weight at every
    program in the student's zone (under ``restrict-zone``, every citywide
    program too) and masks every program outside it unless the policy opens
    it (``Zones.zone_eligibility_matrix``). So the two tiers are the policy's
    priorities with every program in zone, and with none; the second is kept
    only where an exception opens the program -- every program without
    ``restrict-zone``, the student's CTIP status under ``CTIP_access``, a
    sibling's school under ``sibling-access``, a TK promote's feeder under
    ``tk-access`` -- and is -1 elsewhere.
    """
    config = market.config
    generator = PriorityGenerator(market)
    shape = (market.n, market.num_programs)
    policy, ctip = config["policies"][0], config["ctip-options"][0]
    in_zone = generator.get_base_policy_priorities(
        policy, ctip, zone_priority_matrix=np.ones(shape, dtype=int)
    )
    out_of_zone = generator.get_base_policy_priorities(
        policy, ctip, zone_priority_matrix=np.zeros(shape, dtype=int)
    )
    restrict = config.get("restrict-zone", False)
    students = market.students
    if not restrict:
        access = np.ones(shape, dtype=bool)
    else:
        access = np.zeros(shape, dtype=bool)
        if restrict == "CTIP_access":
            ctip_students = students.student_data["ctip1"].fillna(0).to_numpy() > 0
            access |= ctip_students[:, None]
        if config.get("sibling-access", False):
            access |= np.asarray(students.sibling(market.programs)) > 0
        if config.get("tk-access", False):
            access |= np.asarray(students.promotion(market.programs.indices)) > 0
    return in_zone, np.where(access, out_of_zone, -1)


def policy_reserves(market, programs, optimization_config):
    """The policy's reserves and each student's class (by assignment row).

    Mirrors ``GuardrailSetup``: classes split ``reserve-settings.column`` at its
    thresholds (missing values take the mean), every GE program -- or every
    citywide-school GE program under ``citywide_only`` -- reserves
    ``reserve_fraction``, and citywide-school GE programs take
    ``citywide-reserve-ratios`` unless ``citywide-separate-reserves`` is off.
    Zone-share reserves (no ``reserve_fraction``) depend on the zoning and are
    not modelled.
    """
    config = market.config
    settings = config.get("reserve-settings") or {}
    data = market.students.student_data
    column = settings["column"]
    if isinstance(column, dict):
        index = sum(weight * data[name] for name, weight in column.items())
    else:
        index = data[column]
    values = pd.to_numeric(index, errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=np.nanmean(values))
    if "num_categories" in settings:
        count = int(settings["num_categories"])
        thresholds = np.percentile(values, 100 * np.arange(1, count) / count)
    elif "thresholds" in settings:
        thresholds = settings["thresholds"]
        if isinstance(thresholds, str) and thresholds.startswith("percentile:"):
            thresholds = np.atleast_1d(
                np.percentile(values, float(thresholds.split(":")[1]))
            )
        thresholds = np.asarray(thresholds, dtype=float)
    else:
        raise ValueError("reserve-settings needs num_categories or thresholds.")
    classes = np.searchsorted(thresholds, values)
    if not settings["lower_disadvantaged"]:
        classes = len(thresholds) - classes
    num_classes = len(thresholds) + 1

    fraction = settings.get("reserve_fraction", -1)
    if fraction == -1:
        raise ValueError(
            f"matching_policy {optimization_config.matching_policy!r}: reserves "
            "sized by each zone's class shares change with the zoning and are "
            "not modelled; give reserve-settings.reserve_fraction."
        )
    fraction = tuple(float(value) for value in fraction)
    ratios = tuple(
        float(value) for value in config.get("citywide-reserve-ratios", [0.57, 0.43])
    )
    if len(fraction) != num_classes or len(ratios) != num_classes:
        raise ValueError("Reserve fractions must give one share per class.")
    citywide_schools = {int(value) for value in market.schools.citywide_schools}
    separate = config.get("citywide-separate-reserves", True)
    citywide_only = settings.get("citywide_only", False)
    fractions = {}
    for program in programs:
        parts = program.program_id.split("-")
        if len(parts) < 2 or parts[1] != "GE":
            continue
        citywide_ge = program.school_id in citywide_schools
        if separate and citywide_ge:
            fractions[program.program_id] = ratios
        elif citywide_ge or not citywide_only:
            fractions[program.program_id] = fraction
    reserves = ReserveSpec(
        fractions=fractions,
        num_classes=num_classes,
        strict=int(config["guard-rails"]) == 1,
        seed=int(getattr(optimization_config, "mid_reserve_seed", 0)),
    )
    return reserves, classes


def _attendance_area_priority_matrix(market) -> np.ndarray:
    """Build the status-quo singleton attendance-area GE priority component."""
    matrix = np.zeros((market.n, market.num_programs), dtype=int)
    grade = market.grade
    for student_id, row in market.students.student_data.iterrows():
        school_id = row.get("idschoolattendance")
        if pd.isna(school_id):
            continue
        program_id = f"{int(school_id)}-GE-{grade}"
        program_index = market.programs.indices.get(program_id)
        if program_index is not None:
            matrix[market.students.studentno2idx[student_id], program_index - 1] = 1
    return matrix


def _stated_utility_table(data, student_ids, program_ids, market) -> pd.DataFrame:
    """Score each student's submitted list ``length - rank``; unranked is -inf.

    Students who ranked nothing get no row, so they stay outside-only exactly as
    a student missing from a fitted estimate does.
    """
    del data
    program_ids = [str(value) for value in program_ids]
    by_index = {
        int(index): str(code) for code, index in market.programs.indices.items()
    }
    prefs = market.students.selected_preferences(market.programs.index_list)
    column = {program_id: position for position, program_id in enumerate(program_ids)}
    rows, labels = [], []
    for student_id, ranked in zip(student_ids, prefs, strict=True):
        listed = [by_index[int(index)] for index in ranked if int(index) > 0]
        if not listed:
            continue
        row = np.full(len(program_ids), -np.inf)
        for rank, program_id in enumerate(listed):
            if program_id in column and not np.isfinite(row[column[program_id]]):
                row[column[program_id]] = float(len(listed) - rank)
        rows.append(row)
        labels.append(_identity_text(student_id))
    return pd.DataFrame(
        np.asarray(rows).reshape(len(rows), len(program_ids)),
        index=pd.Index(labels, name="studentno"),
        columns=program_ids,
    )


def _load_utility_table(data, student_ids, program_ids, market=None) -> pd.DataFrame:
    del market
    return load_estimate(data, student_ids, program_ids)


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
