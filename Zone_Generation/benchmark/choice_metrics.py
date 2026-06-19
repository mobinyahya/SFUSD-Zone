"""Assignment-based choice metrics for benchmark outputs."""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from Zone_Generation.benchmark.config import ChoiceMetricsRunConfig, json_ready


MATCHING_DIRNAME = "matching"
ASSIGNMENTS_RAW_DIR = "assignments_raw"
GENERATED_CONFIG = "config.generated.yaml"
CHOICE_BY_ASSIGNMENT_CSV = "choice_metrics_by_assignment.csv"
CHOICE_SUMMARY_JSON = "choice_metrics_summary.json"

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


@dataclass
class ChoiceMetricsResult:
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "enabled": True,
            "status": self.status,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "run": self.run,
        }
        if self.error_message:
            payload["error_message"] = self.error_message
        return json_ready(payload)


@dataclass
class ChoiceMetricsTaskResult:
    run_dir: str
    status: str
    error_message: str | None = None
    skipped: bool = False


@dataclass
class ChoiceMetricsBatchResult:
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[ChoiceMetricsTaskResult] = field(default_factory=list)

    def add(self, result: ChoiceMetricsTaskResult) -> None:
        self.results.append(result)
        if result.skipped:
            self.skipped += 1
        elif result.status == "ERROR":
            self.failed += 1
        else:
            self.successful += 1


def compute_choice_metrics_for_run(
    output_dir: str,
    choice_metrics: ChoiceMetricsRunConfig,
) -> ChoiceMetricsResult | None:
    """Compute choice metrics for one benchmark run if assignment artifacts exist."""

    if not choice_metrics.enabled:
        return None

    output_root = Path(os.path.expanduser(output_dir)).resolve()
    matching_dir = output_root / MATCHING_DIRNAME
    assignments_dir = matching_dir / ASSIGNMENTS_RAW_DIR
    if not _assignment_files(assignments_dir):
        return None

    return compute_choice_metrics_from_assignments(
        assignments_dir=assignments_dir,
        matching_dir=matching_dir,
        output_root=output_root,
    )


def compute_choice_metrics_from_assignments(
    *,
    assignments_dir: Path,
    matching_dir: Path,
    output_root: Path,
) -> ChoiceMetricsResult:
    """Compute assignment outcome metrics and write per-assignment artifacts."""

    assignment_files = _assignment_files(assignments_dir)
    if not assignment_files:
        raise ValueError(f"No assignment CSVs found under {assignments_dir}.")

    config_path = matching_dir / GENERATED_CONFIG
    matching_config = _load_yaml(config_path)
    student_data = _load_student_data(matching_config)
    distance_data = _load_distance_data(matching_config, matching_dir)

    rows: list[dict[str, Any]] = []
    for assignment_file in assignment_files:
        raw = pd.read_csv(assignment_file)
        prepared = _prepare_assignment_df(raw, student_data, distance_data)
        row = _choice_metrics_for_assignment(prepared)
        row.update(
            {
                "assignment_file": _relpath(assignment_file, assignments_dir),
                "students_total": int(len(prepared)),
                "students_assigned": int(_assigned_mask(prepared).sum()),
            }
        )
        rows.append(row)

    by_assignment = pd.DataFrame(rows)
    by_assignment_path = matching_dir / CHOICE_BY_ASSIGNMENT_CSV
    by_assignment.to_csv(by_assignment_path, index=False)

    metrics = {
        column: _mean(by_assignment[column])
        for column in CHOICE_METRIC_COLUMNS
        if column in by_assignment.columns
    }
    result = ChoiceMetricsResult(
        status="OK",
        metrics=metrics,
        artifacts={
            "by_assignment": _relpath(by_assignment_path, output_root),
        },
        run={
            "assignment_files": len(assignment_files),
            "student_data_loaded": not student_data.empty,
            "distance_data_loaded": distance_data is not None,
            "config": _relpath(config_path, output_root) if config_path.exists() else None,
        },
    )
    summary_path = matching_dir / CHOICE_SUMMARY_JSON
    result.artifacts["summary"] = _relpath(summary_path, output_root)
    _write_json(summary_path, result.to_payload())
    return result


def run_choice_metrics_for_existing_runs(
    root_folder: str,
    choice_metrics: ChoiceMetricsRunConfig,
    *,
    fail_fast: bool = False,
) -> ChoiceMetricsBatchResult:
    """Run choice-metrics-only regeneration for saved benchmark run folders."""

    from Zone_Generation.benchmark.results import discover_run_dirs
    from Zone_Generation.benchmark.runner import (
        MANIFEST_FILENAME,
        RESULT_FILENAME,
        load_manifest,
        write_json,
    )

    batch = ChoiceMetricsBatchResult()
    run_dirs = discover_run_dirs(root_folder)
    batch.total = len(run_dirs)
    if not choice_metrics.enabled:
        for run_dir in run_dirs:
            batch.add(
                ChoiceMetricsTaskResult(run_dir=run_dir, status="SKIPPED", skipped=True)
            )
        return batch

    for run_dir in run_dirs:
        try:
            result = compute_choice_metrics_for_run(run_dir, choice_metrics)
            if result is None:
                batch.add(
                    ChoiceMetricsTaskResult(
                        run_dir=run_dir,
                        status="SKIPPED",
                        skipped=True,
                    )
                )
                continue

            result_path = os.path.join(run_dir, RESULT_FILENAME)
            payload = _load_json(result_path)
            merge_choice_metrics_result(payload, result)
            write_json(result_path, payload)

            manifest = load_manifest(run_dir)
            manifest["choice_metrics_regenerated"] = True
            write_json(os.path.join(run_dir, MANIFEST_FILENAME), manifest)
            batch.add(ChoiceMetricsTaskResult(run_dir=run_dir, status="OK"))
        except Exception as exc:
            error_message = str(exc) or exc.__class__.__name__
            _mark_choice_metrics_error(run_dir, error_message, traceback.format_exc())
            batch.add(
                ChoiceMetricsTaskResult(
                    run_dir=run_dir,
                    status="ERROR",
                    error_message=error_message,
                )
            )
            if fail_fast:
                raise
    return batch


def merge_choice_metrics_result(
    payload: dict[str, Any], choice_result: ChoiceMetricsResult | None
) -> dict[str, Any]:
    if choice_result is None:
        return payload
    payload["choice_metrics"] = choice_result.to_payload()
    payload.setdefault("metrics", {}).update(choice_result.metrics)
    return payload


def preserve_choice_metrics_payload(
    new_payload: dict[str, Any], previous_payload: Mapping[str, Any]
) -> dict[str, Any]:
    choice_payload = previous_payload.get("choice_metrics")
    if choice_payload is not None:
        new_payload["choice_metrics"] = choice_payload
    previous_metrics = previous_payload.get("metrics") or {}
    choice_metric_values = {
        key: value
        for key, value in previous_metrics.items()
        if str(key).startswith("choice_")
    }
    if choice_metric_values:
        new_payload.setdefault("metrics", {}).update(choice_metric_values)
    return new_payload


def _prepare_assignment_df(
    assignments: pd.DataFrame,
    student_data: pd.DataFrame,
    distance_data: pd.DataFrame | None,
) -> pd.DataFrame:
    out = assignments.copy()
    out = _ensure_studentno(out)
    out = _ensure_school_id(out)
    out = _ensure_frl(out, student_data)
    out = _ensure_assignment_distance(out, distance_data)

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


def _choice_metrics_for_assignment(assignments: pd.DataFrame) -> dict[str, Any]:
    assigned_mask = _assigned_mask(assignments)
    assigned = assignments.loc[assigned_mask].copy()
    total = len(assignments)

    metrics: dict[str, Any] = {column: None for column in CHOICE_METRIC_COLUMNS}
    metrics[CHOICE_PERCENT_UNASSIGNED] = (
        float((~assigned_mask).sum() / total) if total else None
    )
    if assigned.empty:
        return metrics

    if "assignment_dist" in assigned.columns:
        metrics[CHOICE_AVG_STUDENT_DISTANCE] = _mean(assigned["assignment_dist"])
    if "designation" in assigned.columns:
        metrics[CHOICE_PERCENT_DESIGNATED] = _mean(assigned["designation"])
    if "rank" in assigned.columns:
        metrics[CHOICE_PERCENT_TOP_1] = float((assigned["rank"] <= 1).mean())
        metrics[CHOICE_PERCENT_TOP_3] = float((assigned["rank"] <= 3).mean())
    if "assigned_utility" in assignments.columns:
        metrics[CHOICE_TOTAL_MNL_UTILITY] = _sum_utility(assignments["assigned_utility"])
    if "frl" in assignments.columns and "school_id" in assigned.columns:
        metrics[CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL] = _schools_above_district_frl(
            assignments,
            assigned,
            threshold=0.10,
        )
        metrics[CHOICE_FRL_DISSIMILARITY] = _frl_dissimilarity(assigned)
    return metrics


def _schools_above_district_frl(
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


def _frl_dissimilarity(assigned_students: pd.DataFrame) -> float | None:
    if (
        "frl" not in assigned_students.columns
        or "school_id" not in assigned_students.columns
    ):
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


def _ensure_studentno(df: pd.DataFrame) -> pd.DataFrame:
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


def _ensure_school_id(df: pd.DataFrame) -> pd.DataFrame:
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


def _ensure_frl(df: pd.DataFrame, student_data: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "frl" not in out.columns:
        frl = _frl_series(out)
        if frl is not None:
            out["frl"] = frl

    if "frl" in out.columns or student_data.empty or "studentno" not in out.columns:
        return out

    students = student_data.copy()
    students = _ensure_studentno(students)
    frl = _frl_series(students)
    if frl is None or "studentno" not in students.columns:
        return out
    students = students[["studentno"]].copy().assign(frl=frl)
    return out.merge(students.dropna(subset=["studentno"]), how="left", on="studentno")


def _ensure_assignment_distance(
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
        lambda row: _distance_for_assignment(row, distance_data),
        axis=1,
    )
    return out


def _frl_series(df: pd.DataFrame) -> pd.Series | None:
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


def _assigned_mask(df: pd.DataFrame) -> pd.Series:
    if "programno" in df.columns:
        return pd.to_numeric(df["programno"], errors="coerce").fillna(0) > 0
    if "programcodes" in df.columns:
        return df["programcodes"].fillna("").astype(str).str.strip() != ""
    return pd.Series([False] * len(df), index=df.index)


def _load_student_data(config: Mapping[str, Any]) -> pd.DataFrame:
    path = _student_data_path(config)
    if path is None or not path.exists():
        return pd.DataFrame()
    students = pd.read_csv(path, low_memory=False)
    grade = config.get("grade")
    if grade is not None and "grade" in students.columns:
        students = students.loc[students["grade"].astype(str) == str(grade)].copy()
    return students


def _load_distance_data(
    config: Mapping[str, Any], matching_dir: Path
) -> pd.DataFrame | None:
    for path in _distance_candidates(config, matching_dir):
        if not path.exists():
            continue
        distance = pd.read_csv(path)
        if "studentno" in distance.columns:
            distance.set_index("studentno", inplace=True)
        else:
            first_col = distance.columns[0]
            distance.set_index(first_col, inplace=True)
            distance.index.name = "studentno"
        numeric_index = pd.to_numeric(distance.index, errors="coerce")
        if not pd.isna(numeric_index).any():
            distance.index = numeric_index
        return distance
    return None


def _distance_candidates(config: Mapping[str, Any], matching_dir: Path) -> list[Path]:
    paths = dict(config.get("paths") or {})
    candidates: list[Path] = []
    precomputed = paths.get("student-save")
    if precomputed:
        precomputed_dir = Path(os.path.expanduser(str(precomputed)))
        grade = str(config.get("grade", "KG"))
        year = config.get("year")
        if year is not None:
            year = int(year)
            prefix = "student_program_distances"
            student_path = str(paths.get("student-data", ""))
            if Path(student_path).name.startswith("drop_optout"):
                prefix = "student_program_distances_dropoptout"
            candidates.append(
                precomputed_dir / f"{prefix}_{grade}_{year}{year + 1}.csv"
            )
        candidates.extend(sorted(precomputed_dir.glob("student_program_distances*.csv")))

    candidates.extend(
        sorted((matching_dir / "precomputed").glob("student_program_distances*.csv"))
    )
    return _dedupe_paths(candidates)


def _student_data_path(config: Mapping[str, Any]) -> Path | None:
    paths = dict(config.get("paths") or {})
    value = paths.get("student-data")
    if not value:
        return None
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path
    root = paths.get("sfusd")
    if root:
        return Path(os.path.expanduser(str(root))) / path
    return path


def _distance_for_assignment(row: pd.Series, distance_data: pd.DataFrame) -> Any:
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


def _assignment_files(assignments_dir: Path) -> list[Path]:
    if not assignments_dir.exists():
        return []
    return sorted(assignments_dir.rglob("*.csv"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Choice metrics config {path} must be a YAML mapping.")
    return dict(data)


def _load_json(path: str | Path) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(data), f, indent=2, sort_keys=True)


def _mark_choice_metrics_error(run_dir: str, error_message: str, trace: str) -> None:
    from Zone_Generation.benchmark.runner import RESULT_FILENAME, write_json

    result_path = os.path.join(run_dir, RESULT_FILENAME)
    payload = _load_json(result_path)
    payload["choice_metrics"] = {
        "enabled": True,
        "status": "ERROR",
        "error_message": error_message,
        "traceback": trace,
    }
    write_json(result_path, payload)


def _mean(values) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def _sum_utility(values) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return float(numeric.sum()) if not numeric.empty else None


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _relpath(path: str | Path, root: str | Path) -> str:
    return os.path.relpath(os.path.expanduser(str(path)), os.path.expanduser(str(root)))
