#!/usr/bin/env python3
"""Extract the Pareto front for selected assignment metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("~/Downloads/clean_summary_verified.csv").expanduser()
DEFAULT_OUTPUT = Path("~/Downloads/clean_summary_verified_pareto.csv").expanduser()

# True means minimize; False means maximize.
OBJECTIVES = {
    "assignment_distance_av_all_assigned": True,
    "assignment_distance_lt_0_5_all_assigned": False,
    "assignment_distance_gt_3_all_assigned": True,
    "assignment_number_schools_above_15pct_district_frl": True,
    "assignment_dissimilarity_high_frl": True,
    "assignment_prop_top_1_choice_all_assigned": False,
    "assignment_prop_top_3_choice_all_assigned": False,
    "assignment_designated": True,
    "assignment_unassigned": True,
    "assignment_aalpi_in_school_with_plus_15pct_frl": True,
}
PARETO_SUBSET = (
    "assignment_distance_av_all_assigned",
    "assignment_dissimilarity_high_frl",
)


def pareto_front(
    frame: pd.DataFrame,
    objectives: dict[str, bool] = OBJECTIVES,
) -> tuple[pd.DataFrame, int]:
    """Return complete rows not dominated across all selected objectives."""
    missing = [column for column in objectives if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing Pareto metric columns: {missing}")

    metrics = frame[list(objectives)].apply(pd.to_numeric, errors="coerce")
    complete = metrics.notna().all(axis=1)
    metrics = metrics.loc[complete]
    values = metrics.to_numpy(dtype=float, copy=True)
    values[:, [not minimize for minimize in objectives.values()]] *= -1

    not_worse = (values[:, None, :] <= values[None, :, :]).all(axis=2)
    strictly_better = (values[:, None, :] < values[None, :, :]).any(axis=2)
    dominated = (not_worse & strictly_better).any(axis=0)

    output_columns = [
        column
        for column in frame.columns
        if not column.startswith("assignment_") or column in OBJECTIVES
    ]
    frontier = frame.loc[complete].loc[~dominated, output_columns]
    frontier = frontier.sort_values(next(iter(OBJECTIVES)), kind="stable")
    return frontier, int((~complete).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--filter-subset",
        action="store_true",
        help="Also retain only the Pareto front for PARETO_SUBSET.",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    summary = pd.read_csv(input_path)
    frontier, incomplete = pareto_front(summary)
    if args.filter_subset:
        unknown = [column for column in PARETO_SUBSET if column not in OBJECTIVES]
        if unknown:
            raise ValueError(f"PARETO_SUBSET contains unknown objectives: {unknown}")
        subset_objectives = {column: OBJECTIVES[column] for column in PARETO_SUBSET}
        frontier, _ = pareto_front(frontier, subset_objectives)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(output_path, index=False)
    print(
        f"Wrote {len(frontier)} Pareto-optimal rows to {output_path} "
        f"({incomplete} rows excluded for missing metrics)."
    )


if __name__ == "__main__":
    main()
