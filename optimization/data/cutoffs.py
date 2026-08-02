"""Build the year-23 school market used by the cutoff zoning strategy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from assignment.scripts.export.export_ordinal_matrices import (
    aggregate_best_eligible_by_school,
    build_school_capacities,
    load_config,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)
from optimization.data import loaders
from optimization.problem import CutoffMarket, CutoffStudent, ZoneProblem


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_cutoff_market(
    dataset,
    problem: ZoneProblem,
    *,
    assignment_config: str,
    ctip_path: str,
    lottery_scale: int,
    gumbel_scale: float,
    preference_seed: int,
    remove_city_wide: bool,
) -> CutoffMarket:
    """Construct individual school preferences after optimization filtering."""
    config_path = _resolve_project_path(assignment_config)
    resolved_ctip_path = str(Path(ctip_path).expanduser().resolve())
    config = load_config(config_path, "status_quo")
    config["ctip-options"] = ["new_ctip"]
    config["paths"]["new-ctip-path"] = resolved_ctip_path

    market = MarketGenerator(config=config)
    citywide_school_ids = frozenset(map(int, market.schools.citywide_schools))
    if remove_city_wide:
        citywide_centroids = sorted(
            citywide_school_ids & set(map(int, problem.centroid_school_ids))
        )
        if citywide_centroids:
            raise ValueError(
                "remove_city_wide cannot use city-wide centroid schools: "
                f"{citywide_centroids}."
            )
    policy = config["policies"][0]
    market.umodel.draw_utility_model_randomness(
        rows_to_keep=market.students.only_keep_rows,
        cols_to_keep=market.programs.only_keep_cols,
        gumbel_scale=0,
    )
    market.priority_generator.generate_base_priorities(policy)

    program_eligibility = market.preference_generator._get_eligibility().astype(bool)
    program_priorities = market.priority_generator._set_policy_priorities(
        "new_ctip", policy
    )
    school_priorities, school_ids = aggregate_best_eligible_by_school(
        program_priorities,
        program_eligibility,
        market.programs.school_to_indices,
    )
    school_utilities, utility_school_ids = aggregate_best_eligible_by_school(
        market.umodel.original_utilities,
        program_eligibility,
        market.programs.school_to_indices,
    )
    if school_ids != utility_school_ids:
        raise RuntimeError("Priority and utility school columns do not align.")
    excluded_citywide_school_ids = []
    if remove_city_wide:
        (
            school_ids,
            school_priorities,
            school_utilities,
            excluded_citywide_school_ids,
        ) = _exclude_citywide_school_columns(
            citywide_school_ids,
            school_ids,
            school_priorities,
            school_utilities,
        )
        if not school_ids:
            raise ValueError("remove_city_wide removed every school from the market.")

    school_eligibility = np.column_stack(
        [
            program_eligibility[
                :,
                np.asarray(market.programs.school_to_indices[school_id]) - 1,
            ].any(axis=1)
            for school_id in school_ids
        ]
    )
    rng = np.random.RandomState(preference_seed)
    shocks = rng.gumbel(0.0, gumbel_scale, school_utilities.shape)
    shocked_utilities = np.where(
        school_eligibility & np.isfinite(school_utilities),
        school_utilities + shocks,
        -np.inf,
    )

    capacities = build_school_capacities(market)["all_program_capacity"]
    school_nodes = _school_nodes(problem, market, school_ids)
    school_capacities = {
        int(school_id): _nonnegative_integer(
            capacities.loc[school_id], f"capacity for school {school_id}"
        )
        for school_id in school_ids
    }
    zone_restricted_schools = _zone_restricted_schools(
        market,
        school_ids,
        restrict_all=remove_city_wide,
    )

    optimization_students = loaders.load_students(dataset.ingest)
    if optimization_students["studentno"].duplicated().any():
        raise ValueError(
            "cutoffs requires unique studentno values after optimization filtering."
        )
    market_rows = {
        int(studentno): int(row)
        for studentno, row in market.students.studentno2idx.items()
    }
    area_to_node = _area_to_node(problem)

    students = []
    missing_studentnos = []
    empty_preference_studentnos = []
    for row in optimization_students.itertuples(index=False):
        studentno = int(row.studentno)
        market_row = market_rows.get(studentno)
        if market_row is None:
            missing_studentnos.append(studentno)
            continue

        area_id = int(getattr(row, dataset.ingest.unit))
        if area_id not in area_to_node:
            raise ValueError(
                f"Optimization student {studentno} has unmapped "
                f"{dataset.ingest.unit} {area_id}."
            )

        eligible_columns = np.flatnonzero(
            school_eligibility[market_row] & np.isfinite(shocked_utilities[market_row])
        )
        utility_values = shocked_utilities[market_row, eligible_columns]
        if np.unique(utility_values).size != utility_values.size:
            raise ValueError(
                f"Gumbel-shocked school utilities are not strict for student {studentno}."
            )
        order = eligible_columns[np.argsort(-utility_values)]
        preferences = tuple(int(school_ids[column]) for column in order)
        if not preferences:
            empty_preference_studentnos.append(studentno)

        priorities = {
            int(school_ids[column]): _integer_priority(
                school_priorities[market_row, column],
                studentno,
                int(school_ids[column]),
            )
            for column in order
        }
        students.append(
            CutoffStudent(
                studentno=studentno,
                node=area_to_node[area_id],
                preferences=preferences,
                priorities=priorities,
            )
        )

    metadata = {
        "assignment_config": str(config_path),
        "ctip_path": resolved_ctip_path,
        "gumbel_scale": float(gumbel_scale),
        "preference_seed": int(preference_seed),
        "optimization_student_count": int(len(optimization_students)),
        "matched_student_count": len(students),
        "missing_preference_student_count": len(missing_studentnos),
        "missing_preference_studentnos": missing_studentnos,
        "empty_preference_student_count": len(empty_preference_studentnos),
        "empty_preference_studentnos": empty_preference_studentnos,
        "school_count": len(school_ids),
        "remove_city_wide": bool(remove_city_wide),
        "excluded_citywide_school_count": len(excluded_citywide_school_ids),
        "excluded_citywide_schools": excluded_citywide_school_ids,
        "zone_restricted_school_count": len(zone_restricted_schools),
        "zone_restricted_schools": sorted(zone_restricted_schools),
        "zone_access_definition": (
            "City-wide schools are excluded and every remaining school is zone-gated."
            if remove_city_wide
            else "Only schools categorized as Attendance that offer GE are zone-gated; "
            "schools categorized as Citywide remain accessible from every zone."
        ),
        "preference_definition": (
            "All policy-eligible schools ordered by best eligible-program utility "
            "plus one student-school Gumbel shock."
        ),
        "priority_definition": (
            "Best eligible-program status-quo base policy score using ETB CTIP; "
            "lottery, round, and listed/designation boosts excluded."
        ),
    }
    return CutoffMarket(
        students=tuple(students),
        school_nodes=school_nodes,
        school_capacities=school_capacities,
        zone_restricted_schools=zone_restricted_schools,
        lottery_scale=lottery_scale,
        metadata=metadata,
    )


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _area_to_node(problem: ZoneProblem) -> dict[int, int]:
    mapping = {}
    for node, attrs in problem.G.nodes(data=True):
        area_ids = (
            [attrs["area_id"]] if "area_id" in attrs else attrs.get("block_ids", [])
        )
        for area_id in area_ids:
            mapping[int(area_id)] = int(node)
    return mapping


def _school_nodes(
    problem: ZoneProblem,
    market: MarketGenerator,
    school_ids: list[int],
) -> dict[int, int]:
    area_to_node = _area_to_node(problem)
    school_nodes = {}
    missing = []
    for school_id in school_ids:
        school_row = market.schools.school_df.loc[school_id]
        area_id = int(school_row[problem.level.unit])
        node = area_to_node.get(area_id)
        if node is None:
            missing.append((int(school_id), area_id))
        else:
            school_nodes[int(school_id)] = node
    if missing:
        raise ValueError(
            f"Cutoff schools are missing from the {problem.level.unit} graph: {missing}."
        )
    return school_nodes


def _zone_restricted_schools(
    market: MarketGenerator,
    school_ids: list[int],
    *,
    restrict_all: bool = False,
) -> frozenset[int]:
    """Return attendance-area schools whose GE access is controlled by zones."""
    if restrict_all:
        return frozenset(map(int, school_ids))
    attendance_schools = {
        int(school_id)
        for school_id in market.schools.school_df.index[
            market.schools.school_df["category"] == "Attendance"
        ]
    }
    ge_schools = {
        int(school_id)
        for school_id in market.programs.program_df.loc[
            market.programs.program_df["program_type"] == "GE", "school_id"
        ]
    }
    return frozenset(map(int, school_ids)) & attendance_schools & ge_schools


def _exclude_citywide_school_columns(
    citywide_school_ids: frozenset[int],
    school_ids: list[int],
    school_priorities: np.ndarray,
    school_utilities: np.ndarray,
) -> tuple[list[int], np.ndarray, np.ndarray, list[int]]:
    retained_columns = [
        column
        for column, school_id in enumerate(school_ids)
        if int(school_id) not in citywide_school_ids
    ]
    excluded_school_ids = sorted(
        int(school_id)
        for school_id in school_ids
        if int(school_id) in citywide_school_ids
    )
    return (
        [int(school_ids[column]) for column in retained_columns],
        school_priorities[:, retained_columns],
        school_utilities[:, retained_columns],
        excluded_school_ids,
    )


def _nonnegative_integer(value, label: str) -> int:
    number = float(value)
    rounded = round(number)
    if not np.isfinite(number) or number < 0 or not np.isclose(number, rounded):
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    return int(rounded)


def _integer_priority(value, studentno: int, school_id: int) -> int:
    number = float(value)
    rounded = round(number)
    if not np.isfinite(number) or number < 0 or not np.isclose(number, rounded):
        raise ValueError(
            f"Priority for student {studentno} at school {school_id} must be a "
            f"non-negative integer, got {value!r}."
        )
    return int(rounded)
