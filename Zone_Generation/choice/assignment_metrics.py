"""Assignment-output choice metrics shared by benchmark and UI code."""

from __future__ import annotations

from typing import Any

import pandas as pd


CHOICE_AVG_STUDENT_DISTANCE = "choice_avg_student_distance"
CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL = "choice_schools_above_10pct_district_frl"
CHOICE_FRL_DISSIMILARITY = "choice_frl_dissimilarity"
CHOICE_PERCENT_UNASSIGNED = "choice_percent_unassigned"
CHOICE_PERCENT_DESIGNATED = "choice_percent_designated"
CHOICE_PERCENT_TOP_1 = "choice_percent_top_1"
CHOICE_PERCENT_TOP_3 = "choice_percent_top_3"
CHOICE_TOTAL_MNL_UTILITY = "choice_total_mnl_utility"

CHOICE_METRIC_COLUMNS = [
    CHOICE_AVG_STUDENT_DISTANCE,
    CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL,
    CHOICE_FRL_DISSIMILARITY,
    CHOICE_PERCENT_UNASSIGNED,
    CHOICE_PERCENT_DESIGNATED,
    CHOICE_PERCENT_TOP_1,
    CHOICE_PERCENT_TOP_3,
    CHOICE_TOTAL_MNL_UTILITY,
]


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
    return out


def choice_metrics_for_assignment(assignments: pd.DataFrame) -> dict[str, Any]:
    assigned = assigned_mask(assignments)
    assigned_students = assignments.loc[assigned].copy()
    total = len(assignments)

    metrics: dict[str, Any] = {column: None for column in CHOICE_METRIC_COLUMNS}
    metrics[CHOICE_PERCENT_UNASSIGNED] = (
        float((~assigned).sum() / total) if total else None
    )
    if assigned_students.empty:
        return metrics

    if "assignment_dist" in assigned_students.columns:
        metrics[CHOICE_AVG_STUDENT_DISTANCE] = mean(assigned_students["assignment_dist"])
    if "designation" in assigned_students.columns:
        metrics[CHOICE_PERCENT_DESIGNATED] = mean(assigned_students["designation"])
    if "rank" in assigned_students.columns:
        metrics[CHOICE_PERCENT_TOP_1] = float((assigned_students["rank"] <= 1).mean())
        metrics[CHOICE_PERCENT_TOP_3] = float((assigned_students["rank"] <= 3).mean())
    if "assigned_utility" in assignments.columns:
        metrics[CHOICE_TOTAL_MNL_UTILITY] = sum_utility(assignments["assigned_utility"])
    if "frl" in assignments.columns and "school_id" in assigned_students.columns:
        metrics[CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL] = schools_above_district_frl(
            assignments,
            assigned_students,
            threshold=0.10,
        )
        metrics[CHOICE_FRL_DISSIMILARITY] = frl_dissimilarity(assigned_students)
    return metrics


def schools_above_district_frl(
    all_students: pd.DataFrame,
    assigned_students: pd.DataFrame,
    *,
    threshold: float,
) -> float | None:
    frl = pd.to_numeric(all_students["frl"], errors="coerce").dropna()
    if frl.empty:
        return None
    school_frl = (
        assigned_students.dropna(subset=["school_id"])
        .groupby("school_id")["frl"]
        .mean()
        .dropna()
    )
    if school_frl.empty:
        return None
    district_avg = float(frl.mean())
    return float((school_frl >= district_avg + threshold).mean())


def frl_dissimilarity(assigned_students: pd.DataFrame) -> float | None:
    if "frl" not in assigned_students.columns or "school_id" not in assigned_students.columns:
        return None
    students = assigned_students.dropna(subset=["school_id", "frl"]).copy()
    if students.empty:
        return None

    students["frl"] = pd.to_numeric(students["frl"], errors="coerce")
    students = students.dropna(subset=["frl"])
    if students.empty:
        return None

    by_school = students.groupby("school_id")["frl"].agg(["sum", "count"])
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


def sum_utility(values) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return float(numeric.sum()) if not numeric.empty else None
