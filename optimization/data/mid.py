"""Program market data for the MID welfare strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from assignment.student_assignment.market_generator.policy_market import (
    PolicyMarket,
    ctip_key,
    load_policy_market,
    unmodelled_policy_settings,
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
    #: Set only in lottery markets (``LotterySpec``), which keep one type per
    #: student: the student, their reserve class, and per alternative its
    #: kind -- 0 listed, 1 their zone's GE program appended (``add_aa_schools``),
    #: 2 a designation (``designate``).
    student_id: str | None = None
    reserve_class: int = -1
    kinds: tuple[int, ...] | None = None


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
    kinds: tuple[int, ...] | None = None


LISTED, APPENDED, DESIGNATED = 0, 1, 2


@dataclass(frozen=True)
class LotterySpec:
    """How a matching-policy market is matched (``optimization.lottery_da``).

    Deferred acceptance on ``lotteries`` multiple-tie-breaking lotteries
    drawn from ``seed``, as the assignment simulator runs the policy: each
    student's list is what it builds (``MidType.kinds``), and ``fractions``
    maps each reserved program to its per-class seat fractions (empty without
    reserves; ``guardrail_setup``). The draws are indexed by ``student_ids``
    and ``program_ids``, the market's full sets, so they do not move when
    preprocessing drops alternatives.
    """

    fractions: dict[str, tuple[float, ...]]
    num_classes: int
    strict: bool
    seed: int
    lotteries: int
    add_aa_schools: bool
    student_ids: tuple[str, ...]
    program_ids: tuple[str, ...]


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
    lottery: LotterySpec | None = None

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
            # A lottery market may hold a zoned GE program twice: appended for
            # the student's own zone and as a designation for open access.
            entries = (
                student_type.programs
                if student_type.kinds is None
                else tuple(zip(student_type.programs, student_type.kinds))
            )
            if len(entries) != len(set(entries)):
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
            if self.lottery is not None and (
                student_type.count != 1
                or student_type.student_id is None
                or not 0 <= student_type.reserve_class < self.lottery.num_classes
                or student_type.kinds is None
                or len(student_type.kinds) != len(student_type.programs)
            ):
                raise ValueError(
                    "A lottery market keeps one student per type, each with a "
                    "reserve class and a kind per alternative."
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
        return self.lottery is not None or any(
            student_type.open_priorities is not None for student_type in self.types
        )


def require_label_access(market: MidMarket, label: str) -> None:
    """Fail for a model that reads access off the seat label alone.

    The CP/MIP welfare models, SAA and the zoned transport bound know
    attendance-area seats only as "in zone or not"; a policy market's
    out-of-zone access, reserves and built lists are modelled by the lottery
    DA, which the ReCom scorers and capacity references use.
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
    lottery: LotterySpec | None = None


def build_mid_student_market(
    problem: ZoneProblem, optimization_config
) -> MidStudentMarket:
    """Load the assignment cohort before MID type compression.

    Everything zoning-independent comes from the shared, cached
    ``PolicyMarket`` of the data scenario and the policy
    (``assignment.student_assignment.market_generator.policy_market``).
    """
    handling = optimization_config.mid_utility_handling
    if handling not in MID_UTILITY_HANDLING:
        raise ValueError(f"Unknown MID utility handling {handling!r}.")

    policy_name = getattr(optimization_config, "matching_policy", None)
    priority_config = policy_config(optimization_config)
    if policy_name is not None:
        _require_modelled_policy(priority_config, policy_name)
    policies = priority_config["policies"]
    ctip_options = priority_config["ctip-options"]
    if len(policies) != 1 or len(ctip_options) != 1:
        raise ValueError(
            "The MID market requires a policy with one zone policy and one CTIP option."
        )
    source = getattr(optimization_config, "mid_preference_source", "estimate")
    if source not in MID_PREFERENCE_SOURCES:
        raise ValueError(f"Unknown MID preference source {source!r}.")
    if policy_name is not None:
        if source != "stated":
            raise ValueError(
                "A matching_policy market matches the submitted lists the "
                "simulator matches; set mid_preference_source: stated."
            )
    market = load_policy_market(optimization_config.data_scenario, priority_config)
    if policy_name is None:
        # Status quo: zone priority at the student's own area GE program only.
        priorities = market.policy_priorities(
            market.attendance_area_zone_priority(), ctip_options[0]
        )
        load = _load_utility_table if source == "estimate" else _stated_utility_table
        utility = load(
            optimization_config.data_scenario,
            market.student_index,
            market.program_table["program_id"],
            market,
        )
        utility_programs = set(utility.columns)
    else:
        # A policy market scores the submitted lists itself; the stated table
        # it would otherwise read has a column for every program.
        utility_programs = set(market.program_ids)
    program_rows = market.program_table.copy()
    available_programs = sorted(set(program_rows["program_id"]) & utility_programs)
    program_rows = program_rows.set_index("program_id")
    if not program_rows.index.is_unique:
        raise ValueError("Assignment program identities are not unique.")
    citywide_schools = set(market.citywide_schools)
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
    if policy_name is not None:
        return policy_student_market(
            problem, optimization_config, market, tuple(programs)
        )

    area_column = _student_area_column(problem)
    attributes = market.student_attributes
    if area_column not in attributes:
        raise ValueError(f"Assignment students lack {area_column!r}.")
    program_indices = market.program_index
    columns = [program_indices[p] - 1 for p in available_programs]
    students = []
    utility_students = 0
    for row_index, (student_id, area_value, frl) in enumerate(
        zip(market.student_index, attributes[area_column], attributes["frl"])
    ):
        area = _area_key(area_value)
        node = area_to_node.get(area)
        if node is None:
            raise ValueError(
                f"Assignment student {student_id} geography {area!r} is absent "
                f"from graph {problem.level.name}."
            )
        student_key = _identity_text(student_id)
        frl = float(frl)
        if student_key not in utility.index:
            student = MidStudent(node, (), (), (), (), frl)
        else:
            utility_students += 1
            values = utility.loc[student_key, available_programs].to_numpy(dtype=float)
            tiers = np.asarray(priorities[row_index, columns], dtype=float)
            student = replace(
                make_mid_student(node, available_programs, values, tiers, handling),
                frl=frl,
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
        policy=student_market.policy,
        lottery=student_market.lottery,
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
            student.kinds,
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
    for (
        node,
        programs,
        priorities,
        open_priorities,
        student_id,
        reserve_class,
        kinds,
    ), (
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
                kinds=kinds,
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
        student_type.kinds or (),
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

    add_aa = market.lottery is not None and market.lottery.add_aa_schools
    for student_type in market.types:
        kept = []
        open_tiers = student_type.open_priorities
        kinds = student_type.kinds
        for rank, program_id in enumerate(student_type.programs):
            program = programs.get(program_id)
            if program is None:
                continue
            if not program.citywide:
                kind = LISTED if kinds is None else kinds[rank]
                opened = (
                    kind != APPENDED
                    and open_tiers is not None
                    and open_tiers[rank] >= 0
                )
                # With add_aa_schools a designation at the student's own zone
                # is the appended copy of that program, so it is reached out
                # of zone (open access) or not at all.
                own_zone_counts = not (kind == DESIGNATED and add_aa)
                if not opened:
                    if not own_zone_counts:
                        continue
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
        kinds_kept = None if kinds is None else tuple(kinds[rank] for rank in kept)
        key = (
            student_type.node,
            program_ids,
            priorities,
            open_kept,
            student_type.student_id,
            student_type.reserve_class,
            kinds_kept,
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
                    kinds=kinds_kept,
                )
                for (
                    node,
                    program_ids,
                    priorities,
                    open_kept,
                    student_id,
                    reserve_class,
                    kinds_kept,
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
        lottery=market.lottery,
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


def policy_config(optimization_config) -> dict:
    """The assignment config a MID market is built for: the base config and
    the ``matching_policy`` (status quo when unset)."""
    root = Path(__file__).resolve().parents[2]
    with (root / "assignment/configs/base_config.yaml").open(
        encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    # base_config leaves utility-model.enable unset (null), which a MID market
    # reads as the submitted lists. A policy that sets it true is rejected by
    # the shared-market whitelist (policy_market.unmodelled_policy_settings),
    # as is every other setting the market does not model.
    policy = getattr(optimization_config, "matching_policy", None)
    policy_path = (
        POLICY_CONFIG_DIR / "status_quo.yaml"
        if policy is None
        else policy_config_path(policy)
    )
    with policy_path.open(encoding="utf-8") as stream:
        config.update(yaml.safe_load(stream))
    return config


def _require_modelled_policy(config, name) -> None:
    """Reject a policy setting the lottery market does not model, naming each
    (``policy_market.unmodelled_policy_settings``, consumer ``mid``)."""
    unmodelled = unmodelled_policy_settings(config, "mid")
    if unmodelled:
        raise ValueError(
            f"matching_policy {name!r} sets what the lottery market does not "
            "model: " + "; ".join(unmodelled) + "."
        )


def _is_ge(program_id: str) -> bool:
    parts = str(program_id).split("-")
    return len(parts) >= 2 and parts[1] == "GE"


def policy_student_market(
    problem, optimization_config, market: PolicyMarket, programs
) -> MidStudentMarket:
    """The students of a matching-policy market, each with the simulator's list.

    ``PreferenceGenerator.initialize_real_preferences`` builds each list as the
    student's submitted programs, then (``add_aa_schools``) the GE program of
    each school their zone owns, then (``designate``) the designation order
    (``PolicyMarket.designation``). Every entry must be one the student
    qualifies for (program type) and may enter (zone eligibility). The zone
    parts depend on the zoning, so each list here holds every candidate,
    tagged by kind, and the lottery DA keeps those the zoning allows
    (``optimization.lottery_da``), as ``PolicyMarket.apply`` does for the
    simulator.

    Priorities follow ``PriorityGenerator.get_priorities_without_lottery``:
    the round priority, plus -- for listed and appended programs only -- the
    policy's tiers and the ``non_designation_boost``. A designation carries
    the round priority alone. ``priorities`` are the in-zone tiers and
    ``open_priorities`` those out of zone where the policy opens the program
    (``PolicyMarket.access``; -1: closed).
    """
    config = policy_config(optimization_config)
    index_of = market.program_index
    id_of = market.program_ids
    retained = {program.program_id for program in programs}
    in_zone, out_of_zone = market.tiers[ctip_key(config["ctip-options"][0])]
    open_tiers = np.where(market.access > 0, out_of_zone, -1)
    rounds = market.round_boost.astype(float)
    boost = float(market.non_designation_boost)
    qualifies = market.qualifies
    if market.reserve_fractions is None:
        raise ValueError(
            f"matching_policy {optimization_config.matching_policy!r}: reserves "
            "sized by each zone's class shares change with the zoning and are "
            "not modelled; give reserve-settings.reserve_fraction."
        )
    fractions = {
        program.program_id: market.reserve_fractions[program.program_id]
        for program in programs
        if program.program_id in market.reserve_fractions
    }
    add_aa = bool(config.get("add_aa_schools", False))
    designate = bool(config.get("designate", False))

    zoned_ge = [
        program.program_id
        for program in sorted(programs, key=lambda p: index_of[p.program_id])
        if not program.citywide and _is_ge(program.program_id)
    ]

    area_to_node = _area_to_node(problem)
    area_column = _student_area_column(problem)
    attributes = market.student_attributes
    if area_column not in attributes:
        raise ValueError(f"Assignment students lack {area_column!r}.")
    result = []
    with_lists = 0
    for row, (studentno, area_value, frl, ranked) in enumerate(
        zip(
            market.student_index,
            attributes[area_column],
            attributes["frl"],
            market.selected,
            strict=True,
        )
    ):
        node = area_to_node.get(_area_key(area_value))
        if node is None:
            raise ValueError(
                f"Assignment student {studentno} geography is absent from graph "
                f"{problem.level.name}."
            )
        listed = list(
            dict.fromkeys(id_of[int(index) - 1] for index in ranked if int(index) > 0)
        )
        excluded = set(listed)
        entries = []  # (program, tier, open tier, kind, utility)

        for rank, program_id in enumerate(listed):
            j = index_of[program_id] - 1
            if program_id not in retained or not qualifies[row, j]:
                continue
            tier = rounds[row] + in_zone[row, j] + boost
            open_tier = (
                rounds[row] + open_tiers[row, j] + boost
                if open_tiers[row, j] >= 0
                else -1
            )
            entries.append(
                (program_id, tier, open_tier, LISTED, float(len(listed) - rank))
            )
        if add_aa:
            for program_id in zoned_ge:
                j = index_of[program_id] - 1
                if program_id in excluded or not qualifies[row, j]:
                    continue
                entries.append(
                    (
                        program_id,
                        rounds[row] + in_zone[row, j] + boost,
                        -1,
                        APPENDED,
                        0.5,
                    )
                )
        if designate:
            for index in market.designation_order(row):
                j = int(index) - 1
                program_id = id_of[j]
                if program_id in excluded or program_id not in retained:
                    continue
                if not qualifies[row, j]:
                    continue
                open_tier = rounds[row] if open_tiers[row, j] >= 0 else -1
                entries.append((program_id, rounds[row], open_tier, DESIGNATED, 0.25))
        if listed:
            with_lists += 1
        result.append(
            MidStudent(
                node=int(node),
                programs=tuple(entry[0] for entry in entries),
                priorities=tuple(_integer(entry[1], "priority") for entry in entries),
                utilities=tuple(entry[4] for entry in entries),
                scaled_utilities=tuple(
                    max(1, round(CP_SAT_SCALE * entry[4])) for entry in entries
                ),
                frl=float(frl),
                open_priorities=tuple(
                    _integer(entry[2], "open priority") for entry in entries
                ),
                student_id=_identity_text(studentno),
                reserve_class=int(market.reserve_classes[row]),
                kinds=tuple(entry[3] for entry in entries),
            )
        )
    lottery = LotterySpec(
        fractions=fractions,
        num_classes=market.num_classes,
        strict=market.strict_reserves,
        seed=int(optimization_config.mid_lottery_seed),
        lotteries=int(optimization_config.mid_lotteries),
        add_aa_schools=add_aa,
        student_ids=tuple(sorted(student.student_id for student in result)),
        program_ids=tuple(sorted(retained)),
    )
    return MidStudentMarket(
        programs=programs,
        students=tuple(result),
        utility_student_count=with_lists,
        utility_handling="omit_nonpositive",
        policy=optimization_config.matching_policy,
        lottery=lottery,
    )


def _stated_utility_table(data, student_ids, program_ids, market) -> pd.DataFrame:
    """Score each student's submitted list ``length - rank``; unranked is -inf.

    Students who ranked nothing get no row, so they stay outside-only exactly as
    a student missing from a fitted estimate does.
    """
    del data
    program_ids = [str(value) for value in program_ids]
    column = {program_id: position for position, program_id in enumerate(program_ids)}
    rows, labels = [], []
    for student_id, ranked in zip(student_ids, market.selected, strict=True):
        listed = [
            market.program_ids[int(index) - 1] for index in ranked if int(index) > 0
        ]
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
