"""Assignment-based choice metrics for benchmark outputs."""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from Zone_Generation.benchmark.config import ChoiceMetricsRunConfig, json_ready
from Zone_Generation.choice.assignment_metrics import (
    CHOICE_AVG_STUDENT_DISTANCE,
    CHOICE_METRIC_COLUMNS,
    CHOICE_PERCENT_DESIGNATED,
    CHOICE_PERCENT_TOP_1,
    CHOICE_PERCENT_TOP_3,
    CHOICE_PERCENT_UNASSIGNED,
    CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL,
    CHOICE_AVG_MNL_UTILITY,
    CHOICE_SES3_DISSIMILARITY,
    CHOICE_FRL_DISSIMILARITY,
    CHOICE_TOTAL_MNL_UTILITY,
    assigned_mask,
    choice_metric_columns_from_frame,
    choice_metrics_for_assignment,
    mean,
    prepare_assignment_df,
)


MATCHING_DIRNAME = "matching"
ASSIGNMENTS_RAW_DIR = "assignments_raw"
GENERATED_CONFIG = "config.generated.yaml"
CHOICE_BY_ASSIGNMENT_CSV = "choice_metrics_by_assignment.csv"
CHOICE_SUMMARY_JSON = "choice_metrics_summary.json"
ELIGIBLE_SOLUTION_STATUSES = {"OPTIMAL", "FEASIBLE"}
_CHOICE_METRIC_SUFFIXES = tuple(
    column.removeprefix("choice_") for column in CHOICE_METRIC_COLUMNS
)

@dataclass
class ChoiceMetricsResult:
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
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
    matching_runs = _matching_assignment_runs(matching_dir)
    if not matching_runs:
        return None

    if len(matching_runs) == 1 and matching_runs[0][0] is None:
        _, assignments_dir, run_matching_dir = matching_runs[0]
        return compute_choice_metrics_from_assignments(
            assignments_dir=assignments_dir,
            matching_dir=run_matching_dir,
            output_root=output_root,
        )

    results = []
    for name, assignments_dir, run_matching_dir in matching_runs:
        result = compute_choice_metrics_from_assignments(
            assignments_dir=assignments_dir,
            matching_dir=run_matching_dir,
            output_root=output_root,
        )
        results.append((str(name), result))
    return _combined_choice_metrics_result(results, matching_dir, output_root)


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
        prepared = prepare_assignment_df(raw, student_data, distance_data)
        row = choice_metrics_for_assignment(prepared)
        row.update(
            {
                "assignment_file": _relpath(assignment_file, assignments_dir),
                "students_total": int(len(prepared)),
                "students_assigned": int(assigned_mask(prepared).sum()),
            }
        )
        rows.append(row)

    by_assignment = pd.DataFrame(rows)
    by_assignment_path = matching_dir / CHOICE_BY_ASSIGNMENT_CSV
    by_assignment.to_csv(by_assignment_path, index=False)

    metrics = {
        column: mean(by_assignment[column])
        for column in choice_metric_columns_from_frame(by_assignment)
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
    """Regenerate choice metrics for saved benchmark run folders."""

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
            manifest = load_manifest(run_dir)
            result_path = os.path.join(run_dir, RESULT_FILENAME)
            payload = _load_json(result_path)
            result = None
            if _status_is_eligible(_selected_final_status(manifest, payload)):
                result = compute_choice_metrics_for_run(run_dir, choice_metrics)
            stage_result = compute_choice_metrics_for_stages(
                run_dir,
                choice_metrics,
                manifest.get("stages", []),
            )
            clear_choice_metrics_payload(payload)
            if result is None and not (stage_result and stage_result.get("stages")):
                write_json(result_path, payload)
                batch.add(
                    ChoiceMetricsTaskResult(
                        run_dir=run_dir,
                        status="SKIPPED",
                        skipped=True,
                    )
                )
                continue

            merge_choice_metrics_result(payload, result)
            merge_stage_choice_metrics_result(payload, stage_result)
            write_json(result_path, payload)

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


def compute_choice_metrics_for_stages(
    output_dir: str,
    choice_metrics: ChoiceMetricsRunConfig,
    stage_records: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Optionally compute assignment outcome metrics for saved stage folders."""

    if not choice_metrics.enabled or not choice_metrics.compute_stage_metrics:
        return None

    output_root = Path(os.path.expanduser(output_dir)).resolve()
    stages: dict[str, Any] = {}
    for stage in stage_records:
        if not _status_is_eligible(stage.get("status")):
            continue
        stage_name = str(stage.get("name"))
        stage_dir = output_root / str(stage.get("path"))
        result = compute_choice_metrics_for_run(str(stage_dir), choice_metrics)
        if result is not None:
            stages[stage_name] = {"choice_metrics": result.to_payload()}

    return {"enabled": True, "stages": stages}


def merge_choice_metrics_result(
    payload: dict[str, Any], choice_result: ChoiceMetricsResult | None
) -> dict[str, Any]:
    if choice_result is None:
        return payload
    payload["choice_metrics"] = choice_result.to_payload()
    payload.setdefault("metrics", {}).update(choice_result.metrics)
    return payload


def clear_choice_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("choice_metrics", None)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in list(metrics):
            if _is_assignment_choice_metric(key):
                metrics.pop(key, None)

    stage_choice_metrics = payload.get("stage_choice_metrics")
    if isinstance(stage_choice_metrics, dict):
        for stage_payload in (stage_choice_metrics.get("stages") or {}).values():
            if isinstance(stage_payload, dict):
                stage_payload.pop("choice_metrics", None)
    stage_matching = payload.get("stage_matching")
    if isinstance(stage_matching, dict):
        for stage_payload in (stage_matching.get("stages") or {}).values():
            if isinstance(stage_payload, dict):
                stage_payload.pop("choice_metrics", None)
    for stage in (payload.get("run") or {}).get("stages", []):
        if isinstance(stage, dict):
            stage.pop("choice_metrics", None)
            stage.pop("choice_metrics_metrics", None)
    return payload


def merge_stage_choice_metrics_result(
    payload: dict[str, Any], stage_choice_result: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not stage_choice_result:
        return payload
    payload["stage_choice_metrics"] = json_ready(stage_choice_result)
    run_stages = {
        stage.get("name"): stage
        for stage in (payload.get("run") or {}).get("stages", [])
    }
    for stage_name, stage_payload in stage_choice_result.get("stages", {}).items():
        row = run_stages.get(stage_name)
        if row is None:
            continue
        choice_payload = stage_payload.get("choice_metrics")
        if choice_payload is None:
            continue
        row["choice_metrics"] = choice_payload
        row["choice_metrics_metrics"] = choice_payload.get("metrics", {})
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


def _status_is_eligible(status: Any) -> bool:
    return str(status or "").upper() in ELIGIBLE_SOLUTION_STATUSES


def _selected_final_status(
    manifest: Mapping[str, Any], payload: Mapping[str, Any]
) -> Any:
    strategy = str((manifest.get("config") or {}).get("strategy", "")).lower()
    stages = manifest.get("stages") or []
    if "iterative" in strategy:
        choice_stages = [
            stage
            for stage in stages
            if _status_is_eligible(stage.get("status"))
            and (stage.get("metadata") or {}).get("choice_utility") is not None
        ]
        if choice_stages:
            return max(
                choice_stages,
                key=lambda stage: (stage.get("metadata") or {})["choice_utility"],
            ).get("status")
        return (payload.get("run") or {}).get("final_status") or payload.get("status")

    if stages:
        return stages[-1].get("status")
    return (payload.get("run") or {}).get("final_status") or payload.get("status")


def _is_assignment_choice_metric(key: Any) -> bool:
    key = str(key)
    if key in CHOICE_METRIC_COLUMNS:
        return True
    return key.startswith("choice_") and any(
        key.endswith(f"_{suffix}") for suffix in _CHOICE_METRIC_SUFFIXES
    )


def _combined_choice_metrics_result(
    results: list[tuple[str, ChoiceMetricsResult]],
    matching_dir: Path,
    output_root: Path,
) -> ChoiceMetricsResult:
    metrics: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    artifacts: dict[str, Any] = {"runs": {}}
    for name, result in results:
        safe_name = _safe_name(name)
        runs[safe_name] = result.to_payload()
        artifacts["runs"][safe_name] = result.artifacts
        metrics.update(_prefix_choice_metrics(safe_name, result.metrics))

    summary_path = matching_dir / CHOICE_SUMMARY_JSON
    combined = ChoiceMetricsResult(
        status="OK" if all(result.status == "OK" for _, result in results) else "ERROR",
        metrics=metrics,
        artifacts=artifacts,
        run={"configs": list(runs), "runs": runs},
    )
    combined.artifacts["summary"] = _relpath(summary_path, output_root)
    _write_json(summary_path, combined.to_payload())
    return combined


def _matching_assignment_runs(
    matching_dir: Path,
) -> list[tuple[str | None, Path, Path]]:
    legacy_assignments_dir = matching_dir / ASSIGNMENTS_RAW_DIR
    if _assignment_files(legacy_assignments_dir):
        return [(None, legacy_assignments_dir, matching_dir)]

    runs: list[tuple[str | None, Path, Path]] = []
    if not matching_dir.exists():
        return runs
    for child in sorted(matching_dir.iterdir()):
        if not child.is_dir():
            continue
        assignments_dir = child / ASSIGNMENTS_RAW_DIR
        if _assignment_files(assignments_dir):
            runs.append((child.name, assignments_dir, child))
    return runs


def _prefix_choice_metrics(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    prefix = f"choice_{_safe_name(name)}"
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        key_str = str(key)
        suffix = key_str.removeprefix("choice_")
        out[f"{prefix}_{suffix}"] = value
    return out


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return safe or "default"


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
