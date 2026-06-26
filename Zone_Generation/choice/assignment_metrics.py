"""Assignment-output choice metrics backed by student-assignment."""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy


CHOICE_AVG_STUDENT_DISTANCE = "choice_avg_student_distance"
CHOICE_DISTANCE_LT_HALF_MILE = "choice_percent_distance_lt_0_5"
CHOICE_DISTANCE_GT_3_MILES = "choice_percent_distance_gt_3"
CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL = "choice_schools_above_10pct_district_frl"
CHOICE_SCHOOLS_ABOVE_15PCT_DISTRICT_FRL = "choice_schools_above_15pct_district_frl"
CHOICE_AALPI_IN_SCHOOL_WITH_10PCT_FRL = "choice_aalpi_in_school_with_10pct_frl"
CHOICE_AALPI_IN_SCHOOL_WITH_15PCT_FRL = "choice_aalpi_in_school_with_15pct_frl"
CHOICE_AALPI_DISSIMILARITY = "choice_aalpi_dissimilarity"
CHOICE_FRL_DISSIMILARITY = "choice_frl_dissimilarity"
CHOICE_SES3_DISSIMILARITY = "choice_ses3_dissimilarity"
CHOICE_PROGRAMS_WITH_1_4_AA = "choice_programs_with_1_4_aa"
CHOICE_PERCENT_UNASSIGNED = "choice_percent_unassigned"
CHOICE_PERCENT_DESIGNATED = "choice_percent_designated"
CHOICE_PERCENT_TOP_1 = "choice_percent_top_1"
CHOICE_PERCENT_TOP_3 = "choice_percent_top_3"
CHOICE_PERCENT_TOP_1_IN_ZONE = "choice_percent_top_1_in_zone"
CHOICE_PERCENT_TOP_3_IN_ZONE = "choice_percent_top_3_in_zone"
CHOICE_PERCENT_DIST_GE_3_RANK_GE_5 = "choice_percent_dist_ge_3_rank_ge_5"
CHOICE_AVG_MNL_UTILITY = "choice_avg_mnl_utility"
CHOICE_TOTAL_MNL_UTILITY = "choice_total_mnl_utility"
CHOICE_BG_COHESION_3 = "choice_bg_cohesion_3"

PAPER_METRIC_NAMES = [
    "Distance Av",
    "Distance < 0.5",
    "Distance > 3",
    "Schools above 10% district FRL",
    "Schools above 15% district FRL",
    "AALPI in school with +10% FRL",
    "AALPI in school with +15% FRL",
    "Dissimilarity AALPI",
    "Dissimilarity SES3",
    "Programs with 1-4 AA",
    "Unassigned",
    "Designated",
    "Top 3 choice",
    "Top 1 choice",
    "Top 3 in-zone choice",
    "Top 1 in-zone choice",
    "Dist >= 3, Rank >= 5",
    "Avg utility",
    "BG Cohesion (3)",
]

for _group in [
    "Black or African American",
    "Asian",
    "Hispanic/Latino",
    "Pacific Islander",
    "White",
    "High FRL",
    "Low FRL",
]:
    PAPER_METRIC_NAMES.extend(
        [
            f"Top 3 choice {_group}",
            f"Distance Av {_group}",
            f"{_group} in school with +15% FRL",
            f"{_group} Dist >= 3, Rank >= 5",
        ]
    )

PAPER_METRIC_COLUMN_OVERRIDES = {
    "Distance Av": CHOICE_AVG_STUDENT_DISTANCE,
    "Distance < 0.5": CHOICE_DISTANCE_LT_HALF_MILE,
    "Distance > 3": CHOICE_DISTANCE_GT_3_MILES,
    "Schools above 10% district FRL": CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL,
    "Schools above 15% district FRL": CHOICE_SCHOOLS_ABOVE_15PCT_DISTRICT_FRL,
    "AALPI in school with +10% FRL": CHOICE_AALPI_IN_SCHOOL_WITH_10PCT_FRL,
    "AALPI in school with +15% FRL": CHOICE_AALPI_IN_SCHOOL_WITH_15PCT_FRL,
    "Dissimilarity AALPI": CHOICE_AALPI_DISSIMILARITY,
    "Dissimilarity SES3": CHOICE_SES3_DISSIMILARITY,
    "Programs with 1-4 AA": CHOICE_PROGRAMS_WITH_1_4_AA,
    "Unassigned": CHOICE_PERCENT_UNASSIGNED,
    "Designated": CHOICE_PERCENT_DESIGNATED,
    "Top 1 choice": CHOICE_PERCENT_TOP_1,
    "Top 3 choice": CHOICE_PERCENT_TOP_3,
    "Top 1 in-zone choice": CHOICE_PERCENT_TOP_1_IN_ZONE,
    "Top 3 in-zone choice": CHOICE_PERCENT_TOP_3_IN_ZONE,
    "Dist >= 3, Rank >= 5": CHOICE_PERCENT_DIST_GE_3_RANK_GE_5,
    "Avg utility": CHOICE_AVG_MNL_UTILITY,
    "BG Cohesion (3)": CHOICE_BG_COHESION_3,
}


def dependency_metric_column(name: str) -> str:
    normalized = name.lower()
    normalized = normalized.replace(">=", " ge ").replace("<=", " le ")
    normalized = normalized.replace(">", " gt ").replace("<", " lt ")
    normalized = normalized.replace("+", " plus ").replace("%", "pct")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    return f"choice_{normalized}"


DEPENDENCY_CHOICE_METRIC_COLUMNS = {
    name: PAPER_METRIC_COLUMN_OVERRIDES.get(name, dependency_metric_column(name))
    for name in PAPER_METRIC_NAMES
}
ADDITIONAL_CHOICE_METRIC_COLUMNS = [
    CHOICE_FRL_DISSIMILARITY,
    CHOICE_TOTAL_MNL_UTILITY,
]
CHOICE_METRIC_COLUMNS = list(
    dict.fromkeys(
        list(DEPENDENCY_CHOICE_METRIC_COLUMNS.values())
        + ADDITIONAL_CHOICE_METRIC_COLUMNS
    )
)


@dataclass
class _StudentsAdapter:
    student_data: pd.DataFrame
    distance_data: pd.DataFrame
    round_participation: np.ndarray


def prepare_assignment_df(
    assignments: pd.DataFrame,
    student_data: pd.DataFrame,
    distance_data: pd.DataFrame | None,
) -> pd.DataFrame:
    out = assignments.copy()
    out = ensure_studentno(out)
    out = ensure_school_id(out)
    out = ensure_frl(out, student_data)
    out = ensure_assignment_distance(out, distance_data)

    for column in [
        "programno",
        "rank",
        "In-Zone Rank",
        "designation",
        "assignment_dist",
        "frl",
        "assigned_utility",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out.attrs["choice_student_data"] = _student_data_for_assignment(out, student_data)
    out.attrs["choice_distance_data"] = _distance_data_for_assignment(out, distance_data)
    return out


def choice_metrics_for_assignment(assignments: pd.DataFrame) -> dict[str, Any]:
    evaluator = _match_evaluator_for_assignment(assignments)
    with _dependency_numeric_groupby_mean():
        _guard_empty_dependency_methods(evaluator)
        paper_metrics = evaluator.eval_assignment_paper_metrics()
    metrics = dependency_metrics_to_choice_metrics(paper_metrics)
    metrics.update(additional_choice_metrics(evaluator.student_data))
    return metrics


def dependency_metrics_to_choice_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {column: None for column in CHOICE_METRIC_COLUMNS}
    for name, value in dict(metrics).items():
        column = DEPENDENCY_CHOICE_METRIC_COLUMNS.get(name, dependency_metric_column(name))
        out[column] = scalar_metric_value(value)
    return out


def additional_choice_metrics(student_data: pd.DataFrame) -> dict[str, Any]:
    if "programno" not in student_data.columns:
        assigned_students = student_data.iloc[0:0].copy()
    else:
        assigned_students = student_data[
            pd.to_numeric(student_data["programno"], errors="coerce").fillna(0) > 0
        ].copy()
    return {
        CHOICE_FRL_DISSIMILARITY: frl_dissimilarity(assigned_students),
        CHOICE_TOTAL_MNL_UTILITY: total_mnl_utility(assigned_students),
    }


def choice_metric_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    preferred = [column for column in CHOICE_METRIC_COLUMNS if column in frame.columns]
    dynamic = [
        column
        for column in frame.columns
        if str(column).startswith("choice_")
        and column not in preferred
    ]
    return preferred + dynamic


def _match_evaluator_for_assignment(assignments: pd.DataFrame):
    from student_assignment.evaluation.match_evaluator import MatchEvaluator

    assignment_df = _dependency_assignment_df(assignments)
    student_df = assignments.attrs.get("choice_student_data")
    if not isinstance(student_df, pd.DataFrame):
        student_df = _student_data_for_assignment(assignments, pd.DataFrame())
    distance_df = assignments.attrs.get("choice_distance_data")
    if not isinstance(distance_df, pd.DataFrame):
        distance_df = _distance_data_for_assignment(assignments, None)

    assignment_df.attrs = {}
    student_df = student_df.copy()
    student_df.attrs = {}
    distance_df = distance_df.copy()
    distance_df.attrs = {}

    adapter = _StudentsAdapter(
        student_data=student_df,
        distance_data=distance_df,
        round_participation=np.ones((len(student_df), 1), dtype=int),
    )
    return MatchEvaluator(adapter, assignment_df, adapter.distance_data)


def _dependency_assignment_df(assignments: pd.DataFrame) -> pd.DataFrame:
    out = ensure_studentno(assignments).copy()
    if "studentno" not in out.columns:
        raise ValueError("Assignment metrics require a studentno column.")
    out["studentno"] = pd.to_numeric(out["studentno"], errors="coerce")
    out = out.dropna(subset=["studentno"]).copy()
    out["studentno"] = out["studentno"].astype(int)

    if "programcodes" not in out.columns:
        out["programcodes"] = pd.NA
    out["programcodes"] = out["programcodes"].replace("", pd.NA)
    if "programno" not in out.columns:
        out["programno"] = out["programcodes"].notna().astype(int)
    if "rank" not in out.columns:
        out["rank"] = np.nan
    if "designation" not in out.columns:
        out["designation"] = 0
    if "In-Zone Rank" not in out.columns:
        out["In-Zone Rank"] = out["rank"]

    for column in ["programno", "rank", "designation", "In-Zone Rank", "assigned_utility"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    keep = [
        column
        for column in [
            "studentno",
            "programno",
            "programcodes",
            "rank",
            "designation",
            "In-Zone Rank",
            "assigned_utility",
        ]
        if column in out.columns
    ]
    return out[keep].set_index("studentno", drop=True)


def _student_data_for_assignment(
    assignments: pd.DataFrame, student_data: pd.DataFrame
) -> pd.DataFrame:
    assignment_students = ensure_studentno(assignments)
    if "studentno" not in assignment_students.columns:
        return pd.DataFrame()

    base = assignment_students[["studentno"]].copy()
    base["studentno"] = pd.to_numeric(base["studentno"], errors="coerce")
    base = base.dropna(subset=["studentno"]).drop_duplicates("studentno")
    base["studentno"] = base["studentno"].astype(int)

    students = ensure_studentno(student_data) if not student_data.empty else pd.DataFrame()
    if not students.empty and "studentno" in students.columns:
        drop_assignment_columns = [
            column
            for column in [
                "programno",
                "programcodes",
                "rank",
                "designation",
                "In-Zone Rank",
                "assignment_dist",
                "assigned_utility",
            ]
            if column in students.columns
        ]
        students = students.drop(columns=drop_assignment_columns)
        students = students.dropna(subset=["studentno"]).drop_duplicates("studentno")
        students["studentno"] = pd.to_numeric(students["studentno"], errors="coerce")
        students = students.dropna(subset=["studentno"])
        students["studentno"] = students["studentno"].astype(int)
        base = base.merge(students, on="studentno", how="left")

    base = _fill_missing_student_columns(base, assignment_students)
    return base.set_index("studentno", drop=True)


def _fill_missing_student_columns(
    student_data: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    out = student_data.copy()
    assignment_lookup = assignments.copy()
    if "studentno" in assignment_lookup.columns:
        assignment_lookup["studentno"] = pd.to_numeric(
            assignment_lookup["studentno"], errors="coerce"
        )
        assignment_lookup = assignment_lookup.dropna(subset=["studentno"])
        assignment_lookup["studentno"] = assignment_lookup["studentno"].astype(int)
        assignment_lookup = assignment_lookup.drop_duplicates("studentno")

    for column in [
        "freelunch_prob",
        "free_lunch_prob",
        "reducedlunch_prob",
        "reduced_lunch_prob",
        "FRL Score",
        "FRL",
        "frl",
        "resolved_ethnicity",
        "grade",
        "census_blockgroup",
        "N'hood SES Score",
        "SES_category",
    ]:
        if column not in out.columns and column in assignment_lookup.columns:
            out = out.merge(
                assignment_lookup[["studentno", column]], on="studentno", how="left"
            )

    frl = frl_series(out)
    if "freelunch_prob" not in out.columns:
        out["freelunch_prob"] = frl if frl is not None else 0.0
    if "reducedlunch_prob" not in out.columns:
        out["reducedlunch_prob"] = 0.0
    if "resolved_ethnicity" not in out.columns:
        out["resolved_ethnicity"] = ""
    if "grade" not in out.columns:
        out["grade"] = ""
    if "census_blockgroup" not in out.columns:
        out["census_blockgroup"] = out["studentno"]
    if "SES_category" not in out.columns:
        out["SES_category"] = _ses_category(out)

    for column in ["freelunch_prob", "reducedlunch_prob", "SES_category"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["freelunch_prob"] = out["freelunch_prob"].fillna(0)
    out["reducedlunch_prob"] = out["reducedlunch_prob"].fillna(0)
    out["resolved_ethnicity"] = out["resolved_ethnicity"].fillna("")
    return out


def _ses_category(student_data: pd.DataFrame) -> pd.Series:
    if "N'hood SES Score" in student_data.columns and "FRL Score" in student_data.columns:
        score = (
            0.25 * pd.to_numeric(student_data["N'hood SES Score"], errors="coerce")
            + 0.25 * pd.to_numeric(student_data["FRL Score"], errors="coerce")
        )
    else:
        frl = frl_series(student_data)
        score = pd.to_numeric(frl, errors="coerce") if frl is not None else pd.Series(0, index=student_data.index)
    numeric = pd.to_numeric(score, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return pd.Series(1, index=student_data.index)
    thresh33, thresh66 = np.percentile(non_null, [33, 66])
    return numeric.apply(lambda x: 1 if x < thresh33 else (2 if x < thresh66 else 3))


def _distance_data_for_assignment(
    assignments: pd.DataFrame, distance_data: pd.DataFrame | None
) -> pd.DataFrame:
    if distance_data is not None:
        out = distance_data.copy()
        if out.index.name != "studentno":
            out.index.name = "studentno"
        numeric_index = pd.to_numeric(out.index, errors="coerce")
        if not pd.isna(numeric_index).any():
            out.index = numeric_index.astype(int)
            out.index.name = "studentno"
        return out

    students = ensure_studentno(assignments)
    if "studentno" not in students.columns:
        return pd.DataFrame(index=pd.Index([], name="studentno"))
    students = students.copy()
    students["studentno"] = pd.to_numeric(students["studentno"], errors="coerce")
    students = students.dropna(subset=["studentno"])
    students["studentno"] = students["studentno"].astype(int)
    if "programcodes" not in students.columns or "assignment_dist" not in students.columns:
        index = pd.Index(students["studentno"].drop_duplicates(), name="studentno")
        return pd.DataFrame(index=index)

    rows = students.dropna(subset=["programcodes"])
    rows = rows.loc[rows["programcodes"].astype(str).str.strip() != ""]
    if rows.empty:
        index = pd.Index(students["studentno"].drop_duplicates(), name="studentno")
        return pd.DataFrame(index=index)
    table = rows.pivot_table(
        values="assignment_dist",
        index="studentno",
        columns="programcodes",
        aggfunc="first",
    )
    table.index.name = "studentno"
    return table


@contextmanager
def _dependency_numeric_groupby_mean():
    original = DataFrameGroupBy.mean

    def mean(self, *args, **kwargs):
        kwargs.setdefault("numeric_only", True)
        return original(self, *args, **kwargs)

    DataFrameGroupBy.mean = mean
    try:
        yield
    finally:
        DataFrameGroupBy.mean = original


def _guard_empty_dependency_methods(evaluator) -> None:
    original_metric_frl_concentration = evaluator.metric_FRL_concentration
    original_metric_dissimilarity = evaluator.metric_dissimilarity
    original_dissimilarity = evaluator.dissimilarity

    def metric_frl_concentration(all_students, group_students, threshold):
        if len(group_students) == 0:
            return np.nan
        return original_metric_frl_concentration(all_students, group_students, threshold)

    def metric_dissimilarity(group_students, total_enrollment):
        if len(group_students) == 0 or pd.to_numeric(total_enrollment, errors="coerce").sum() == 0:
            return np.nan
        return original_metric_dissimilarity(group_students, total_enrollment)

    def dissimilarity(group_students, total_enrollment):
        if len(group_students) == 0 or pd.to_numeric(total_enrollment, errors="coerce").sum() == 0:
            return np.nan
        return original_dissimilarity(group_students, total_enrollment)

    def metric_bg_cohesion(assigned_students, num):
        if len(assigned_students) == 0:
            return np.nan
        cohesion = assigned_students.groupby("census_blockgroup").apply(
            lambda group: evaluator._bgcohesion(group, num)
        )
        values = cohesion.to_numpy().reshape(-1) if isinstance(cohesion, pd.DataFrame) else cohesion
        numeric = pd.to_numeric(values, errors="coerce")
        numeric = pd.Series(numeric).dropna()
        return float(numeric.sum() / len(assigned_students))

    evaluator.metric_FRL_concentration = metric_frl_concentration
    evaluator.metric_dissimilarity = metric_dissimilarity
    evaluator.dissimilarity = dissimilarity
    evaluator.metric_BG_cohesion = metric_bg_cohesion


def ensure_studentno(df: pd.DataFrame) -> pd.DataFrame:
    if "studentno" in df.columns:
        out = df.copy()
    else:
        out = df.copy()
        unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
        if unnamed:
            out.rename(columns={unnamed[0]: "studentno"}, inplace=True)
    if "studentno" in out.columns:
        out["studentno"] = pd.to_numeric(out["studentno"], errors="coerce").astype(
            "Int64"
        )
    return out


def ensure_school_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "school_id" in out.columns:
        out["school_id"] = pd.to_numeric(out["school_id"], errors="coerce")
        return out
    programcodes = out.get("programcodes")
    if programcodes is None:
        out["school_id"] = pd.NA
        return out
    school = programcodes.fillna("").astype(str).str.split("-", n=1).str[0]
    out["school_id"] = pd.to_numeric(school, errors="coerce")
    return out


def ensure_frl(df: pd.DataFrame, student_data: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "frl" not in out.columns:
        frl = frl_series(out)
        if frl is not None:
            out["frl"] = frl

    if "frl" in out.columns or student_data.empty or "studentno" not in out.columns:
        return out

    students = ensure_studentno(student_data)
    frl = frl_series(students)
    if frl is None or "studentno" not in students.columns:
        return out
    students = students[["studentno"]].copy().assign(frl=frl)
    return out.merge(students.dropna(subset=["studentno"]), how="left", on="studentno")


def ensure_assignment_distance(
    df: pd.DataFrame,
    distance_data: pd.DataFrame | None,
) -> pd.DataFrame:
    out = df.copy()
    if "assignment_dist" in out.columns or distance_data is None:
        return out
    if "studentno" not in out.columns or "programcodes" not in out.columns:
        out["assignment_dist"] = pd.NA
        return out
    out["assignment_dist"] = out.apply(
        lambda row: distance_for_assignment(row, distance_data),
        axis=1,
    )
    return out


def frl_series(df: pd.DataFrame) -> pd.Series | None:
    free_cols = [c for c in ["freelunch_prob", "free_lunch_prob"] if c in df.columns]
    reduced_cols = [
        c for c in ["reducedlunch_prob", "reduced_lunch_prob"] if c in df.columns
    ]
    if free_cols or reduced_cols:
        free = pd.to_numeric(df[free_cols[0]], errors="coerce") if free_cols else 0
        reduced = (
            pd.to_numeric(df[reduced_cols[0]], errors="coerce") if reduced_cols else 0
        )
        return pd.Series(free, index=df.index).fillna(0) + pd.Series(
            reduced, index=df.index
        ).fillna(0)
    for column in ["FRL Score", "FRL", "frl"]:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return None


def assigned_mask(df: pd.DataFrame) -> pd.Series:
    if "programno" in df.columns:
        return pd.to_numeric(df["programno"], errors="coerce").fillna(0) > 0
    if "programcodes" in df.columns:
        return df["programcodes"].fillna("").astype(str).str.strip() != ""
    return pd.Series([False] * len(df), index=df.index)


def frl_dissimilarity(assigned_students: pd.DataFrame) -> float | None:
    if "frl" not in assigned_students.columns or "assigned school" not in assigned_students.columns:
        return None
    students = assigned_students.dropna(subset=["assigned school", "frl"]).copy()
    if students.empty:
        return None

    students["frl"] = pd.to_numeric(students["frl"], errors="coerce")
    students = students.dropna(subset=["frl"])
    if students.empty:
        return None

    by_school = students.groupby("assigned school")["frl"].agg(["sum", "count"])
    by_school["non_frl"] = by_school["count"] - by_school["sum"]
    total_frl = float(by_school["sum"].sum())
    total_non_frl = float(by_school["non_frl"].sum())
    if total_frl <= 0 or total_non_frl <= 0:
        return None
    return float(
        0.5
        * (
            (by_school["sum"] / total_frl)
            - (by_school["non_frl"] / total_non_frl)
        )
        .abs()
        .sum()
    )


def total_mnl_utility(assigned_students: pd.DataFrame) -> float | None:
    if "assigned_utility" not in assigned_students.columns:
        return None
    numeric = pd.to_numeric(assigned_students["assigned_utility"], errors="coerce")
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return float(numeric.sum()) if not numeric.empty else None


def distance_for_assignment(row: pd.Series, distance_data: pd.DataFrame) -> Any:
    program = row.get("programcodes")
    student = row.get("studentno")
    if pd.isna(program) or str(program).strip() == "" or pd.isna(student):
        return None
    program = str(program)
    if program not in distance_data.columns:
        return None

    keys = [student]
    try:
        student_int = int(student)
    except (TypeError, ValueError):
        student_int = None
    if student_int is not None:
        keys.extend([student_int, str(student_int)])

    for key in keys:
        if key is not None and key in distance_data.index:
            return distance_data.at[key, program]
    return None


def mean(values) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def scalar_metric_value(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return mean(value)
    if isinstance(value, np.ndarray):
        return mean(value.reshape(-1))
    if isinstance(value, (list, tuple, set)):
        return mean(list(value))
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return scalar_metric_value(value.item())
    return float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
