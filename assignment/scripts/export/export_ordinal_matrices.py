"""Export student-by-school ordinal priority and preference matrices.

Priority tiers are dense-ranked within each school, while preference ranks are
dense-ranked within each student. Rank 1 is best and equal values remain tied.

Usage:
    uv run python -m assignment.scripts.export.export_ordinal_matrices \
        --config assignment/configs/kumar.config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loaders import anchor_data_config

from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def load_config(config_path: Path, subconfig_name: str | None) -> dict:
    """Load a user config and merge its selected policy subconfig."""
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)

    if subconfig_name is None:
        subconfigs = config.get("subconfigs", [])
        if len(subconfigs) != 1:
            raise ValueError(
                "Specify --subconfig when the config does not select exactly one."
            )
        subconfig_name = subconfigs[0]

    subconfig_path = config_path.parent / "policy_configs" / f"{subconfig_name}.yaml"
    with subconfig_path.open() as subconfig_file:
        subconfig = yaml.safe_load(subconfig_file)

    merged = {**config, **subconfig}
    if isinstance(merged.get("data"), dict):
        merged["data"] = anchor_data_config(merged["data"], config_path.parent)
    merged["subconfig-name"] = subconfig_name
    merged["save-assignment"] = True
    return merged


def aggregate_best_eligible_by_school(
    values: np.ndarray,
    eligibility: np.ndarray,
    school_to_indices: dict[int, list[int]],
) -> tuple[np.ndarray, list[int]]:
    """Take each student's best eligible program value at each school."""
    school_ids = sorted(school_to_indices)
    school_values = np.full((values.shape[0], len(school_ids)), -np.inf)

    for school_column, school_id in enumerate(school_ids):
        program_columns = np.asarray(school_to_indices[school_id], dtype=int) - 1
        eligible_values = np.where(
            eligibility[:, program_columns],
            values[:, program_columns],
            -np.inf,
        )
        school_values[:, school_column] = eligible_values.max(axis=1)

    return school_values, school_ids


def dense_rank(
    values: np.ndarray,
    student_ids: list[int],
    school_ids: list[int],
    axis: int,
) -> pd.DataFrame:
    """Dense-rank descending values along one DataFrame axis."""
    frame = pd.DataFrame(values, index=student_ids, columns=school_ids)
    frame.index.name = "studentno"
    frame.columns.name = "school_id"
    return frame.rank(axis=axis, ascending=False, method="dense").astype("int16")


def build_school_capacities(market: MarketGenerator) -> pd.DataFrame:
    """Aggregate configured program capacities by school and program type."""
    capacities = market.programs.program_df.pivot_table(
        index="school_id",
        columns="program_type",
        values="capacity",
        aggfunc="sum",
        fill_value=0,
    )
    capacities.columns.name = None
    capacities["all_program_capacity"] = capacities.sum(axis=1)

    school_names = market.schools.school_df["school_name"]
    capacities.insert(0, "school_name", school_names.reindex(capacities.index))
    capacities.index.name = "school_id"
    return capacities.sort_index()


def export_matrices(
    config_path: Path,
    output_dir: Path | None,
    subconfig_name: str | None,
) -> tuple[Path, Path, Path, Path]:
    """Build and write school priority tiers and student preference ranks."""
    config = load_config(config_path, subconfig_name)
    policies = config["policies"]
    ctip_options = config["ctip-options"]
    if len(policies) != 1 or len(ctip_options) != 1:
        raise ValueError("The export requires exactly one policy and one CTIP option.")

    market = MarketGenerator(config=config)
    policy = policies[0]
    ctip = ctip_options[0]

    market.umodel.draw_utility_model_randomness(
        rows_to_keep=market.students.only_keep_rows,
        cols_to_keep=market.programs.only_keep_cols,
        gumbel_scale=0,
    )
    market.priority_generator.generate_base_priorities(policy)

    # Eligibility setup is part of the simulator's normal policy construction.
    eligibility = market.preference_generator._get_eligibility().astype(bool)
    program_priorities = market.priority_generator._set_policy_priorities(ctip, policy)
    program_utilities = market.umodel.original_utilities

    school_priorities, school_ids = aggregate_best_eligible_by_school(
        program_priorities,
        eligibility,
        market.programs.school_to_indices,
    )
    school_utilities, utility_school_ids = aggregate_best_eligible_by_school(
        program_utilities,
        eligibility,
        market.programs.school_to_indices,
    )
    if school_ids != utility_school_ids:
        raise RuntimeError("Priority and utility school columns do not align.")

    student_ids = [
        int(market.students.idx2studentno[index]) for index in range(market.n)
    ]
    priority_tiers = dense_rank(school_priorities, student_ids, school_ids, axis=0)
    preference_ranks = dense_rank(school_utilities, student_ids, school_ids, axis=1)
    school_eligibility = np.column_stack(
        [
            eligibility[
                :,
                np.asarray(market.programs.school_to_indices[school_id]) - 1,
            ].any(axis=1)
            for school_id in school_ids
        ]
    )

    if output_dir is None:
        output_dir = Path(config["output_dir"])
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    priority_path = output_dir / "school_priority_tiers.csv"
    preference_path = output_dir / "student_school_preference_ranks.csv"
    capacity_path = output_dir / "school_program_capacities.csv"
    metadata_path = output_dir / "ordinal_matrices_metadata.json"
    priority_tiers.to_csv(priority_path)
    preference_ranks.to_csv(preference_path)
    school_capacities = build_school_capacities(market)
    school_capacities.to_csv(capacity_path)

    metadata = {
        "config": str(config_path.resolve()),
        "subconfig": config["subconfig-name"],
        "policy": policy,
        "ctip": ctip,
        "students": market.n,
        "schools": len(school_ids),
        "capacity_definition": (
            "Configured capacity summed by school and program type; "
            "all_program_capacity is the sum across program types."
        ),
        "priority_definition": (
            "Best eligible program policy score at each school; lottery, "
            "round, and submitted-list/designation boosts excluded; dense-ranked "
            "within school with 1 as highest priority."
        ),
        "preference_definition": (
            "Best eligible program deterministic estimated utility at each "
            "school; no Gumbel noise; dense-ranked within student with 1 as "
            "highest preference."
        ),
        "policy_ineligible_student_school_pairs": int((~school_eligibility).sum()),
        "eligible_pairs_without_finite_estimated_utility": int(
            (school_eligibility & np.isneginf(school_utilities)).sum()
        ),
        "unrankable_student_school_pairs": int(np.isneginf(school_utilities).sum()),
        "priority_columns_with_ties": int(
            sum(priority_tiers[column].nunique() < market.n for column in school_ids)
        ),
        "preference_rows_with_ties": int(
            sum(
                preference_ranks.loc[student].nunique() < len(school_ids)
                for student in student_ids
            )
        ),
    }
    with metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    return priority_path, preference_path, capacity_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subconfig")
    args = parser.parse_args()

    paths = export_matrices(args.config, args.output_dir, args.subconfig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
