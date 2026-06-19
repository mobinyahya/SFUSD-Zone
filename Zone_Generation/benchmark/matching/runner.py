"""Run student-assignment simulations from saved zoning outputs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from Zone_Generation.benchmark.config import (
    ChoiceMetricsRunConfig,
    MatchingRunConfig,
    json_ready,
)
from Zone_Generation.metrics.base import MetricsContext
from Zone_Generation.optimization.solution import ZoneSolution


DEFAULT_MATCHING_TEMPLATE = Path(__file__).with_name(
    "medium_zones_no_reserves_no_sib.yaml"
)
GENERATED_POLICY_NAME = "generated_zones"
MATCHING_DIRNAME = "matching"
ZONE_CSV = "zones.csv"
GENERATED_CONFIG = "config.generated.yaml"
ASSIGNMENTS_RAW_DIR = "assignments_raw"
STUDENT_ASSIGNMENTS_CSV = "student_school_assignments.csv"
SCHOOL_POPULATIONS_CSV = "school_populations.csv"
PROGRAM_POPULATIONS_CSV = "program_populations.csv"
SUMMARY_JSON = "summary.json"


@dataclass
class MatchingResult:
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "enabled": True,
            "status": self.status,
            "artifacts": self.artifacts,
            "run": self.run,
        }
        if self.error_message:
            payload["error_message"] = self.error_message
        return json_ready(payload)


@dataclass
class MatchingTaskResult:
    run_dir: str
    status: str
    error_message: str | None = None
    skipped: bool = False


@dataclass
class MatchingBatchResult:
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[MatchingTaskResult] = field(default_factory=list)

    def add(self, result: MatchingTaskResult) -> None:
        self.results.append(result)
        if result.skipped:
            self.skipped += 1
        elif result.status == "ERROR":
            self.failed += 1
        else:
            self.successful += 1


def run_matching_for_solution(
    solution: ZoneSolution,
    output_dir: str,
    matching: MatchingRunConfig,
) -> MatchingResult | None:
    """Run matching for one final zoning solution and write run artifacts."""

    if not matching.enabled:
        return None
    if not solution.assignment:
        raise ValueError("Cannot run matching without a final zone assignment.")

    output_root = Path(os.path.expanduser(output_dir)).resolve()
    matching_dir = output_root / MATCHING_DIRNAME
    assignments_dir = matching_dir / ASSIGNMENTS_RAW_DIR
    _reset_matching_dir(matching_dir)
    assignments_dir.mkdir(parents=True, exist_ok=True)

    zone_csv = matching_dir / ZONE_CSV
    zone_id_map = write_matching_zone_csv(solution.area_assignment(), zone_csv)

    config_template = resolve_matching_template(matching.config)
    simulation_config = build_simulation_config(
        template_path=config_template,
        zone_csv=zone_csv,
        assignments_dir=assignments_dir,
        precomputed_dir=matching_dir / "precomputed",
        solution=solution,
    )
    generated_config = matching_dir / GENERATED_CONFIG
    with open(generated_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(json_ready(simulation_config), f, sort_keys=True)

    _run_student_assignment(simulation_config, assignments_dir)
    result = summarize_assignment_outputs(
        assignments_dir=assignments_dir,
        matching_dir=matching_dir,
        output_root=output_root,
    )
    result.artifacts.update(
        {
            "zone_csv": _relpath(zone_csv, output_root),
            "generated_config": _relpath(generated_config, output_root),
        }
    )
    result.run.update(
        {
            "config_template": str(config_template),
            "policy_name": GENERATED_POLICY_NAME,
            "zone_id_map": {str(k): v for k, v in zone_id_map.items()},
            "zone_building_blocks": simulation_config.get("zone-building-blocks"),
            "level": solution.level.name,
        }
    )
    result.artifacts["summary"] = _relpath(matching_dir / SUMMARY_JSON, output_root)
    _write_json(matching_dir / SUMMARY_JSON, result.to_payload())
    return result


def run_matching_for_existing_runs(
    root_folder: str,
    matching: MatchingRunConfig,
    *,
    choice_metrics: ChoiceMetricsRunConfig | None = None,
    fail_fast: bool = False,
    dataset_factory=None,
) -> MatchingBatchResult:
    """Run matching-only regeneration for saved benchmark run folders."""

    from Zone_Generation.benchmark.results import discover_run_dirs
    from Zone_Generation.benchmark.runner import (
        MANIFEST_FILENAME,
        RESULT_FILENAME,
        load_manifest,
        load_solutions,
        write_json,
    )

    batch = MatchingBatchResult()
    run_dirs = discover_run_dirs(root_folder)
    batch.total = len(run_dirs)
    if not matching.enabled:
        for run_dir in run_dirs:
            batch.add(MatchingTaskResult(run_dir=run_dir, status="SKIPPED", skipped=True))
        return batch

    for run_dir in run_dirs:
        try:
            dataset = None
            if dataset_factory is not None:
                from Zone_Generation.benchmark.config import optimization_config_from_dict

                manifest_for_dataset = load_manifest(run_dir)
                config_for_dataset = optimization_config_from_dict(
                    manifest_for_dataset["config"]
                )
                dataset = dataset_factory(config_for_dataset, manifest_for_dataset)

            solutions, config, manifest = load_solutions(run_dir, dataset=dataset)
            if not solutions:
                batch.add(MatchingTaskResult(run_dir=run_dir, status="SKIPPED", skipped=True))
                continue
            final_solution = MetricsContext(solutions, config=config).solution
            matching_result = run_matching_for_solution(final_solution, run_dir, matching)
            result_path = os.path.join(run_dir, RESULT_FILENAME)
            payload = _load_json(result_path)
            merge_matching_result(payload, matching_result)
            if choice_metrics and choice_metrics.enabled:
                from Zone_Generation.benchmark.choice_metrics import (
                    compute_choice_metrics_for_run,
                    merge_choice_metrics_result,
                )

                choice_result = compute_choice_metrics_for_run(
                    run_dir,
                    choice_metrics,
                )
                merge_choice_metrics_result(payload, choice_result)
            write_json(result_path, payload)

            manifest["matching_regenerated"] = True
            write_json(os.path.join(run_dir, MANIFEST_FILENAME), manifest)
            batch.add(MatchingTaskResult(run_dir=run_dir, status="OK"))
        except Exception as exc:
            error_message = str(exc) or exc.__class__.__name__
            _mark_matching_error(run_dir, error_message, traceback.format_exc())
            batch.add(
                MatchingTaskResult(
                    run_dir=run_dir,
                    status="ERROR",
                    error_message=error_message,
                )
            )
            if fail_fast:
                raise
    return batch


def write_matching_zone_csv(
    area_assignment: Mapping[int, int], path: str | Path
) -> dict[int, int]:
    """Write matching's row-per-zone CSV from ``{area_id: zone_id}``."""

    if not area_assignment:
        raise ValueError("Cannot write matching zones from an empty assignment.")
    zones: dict[int, list[int]] = {}
    for area_id, zone_id in area_assignment.items():
        zones.setdefault(int(zone_id), []).append(int(area_id))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    zone_id_map: dict[int, int] = {}
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row_idx, zone_id in enumerate(sorted(zones)):
            zone_id_map[zone_id] = row_idx
            writer.writerow(sorted(zones[zone_id]))
    return zone_id_map


def resolve_matching_template(path: str | None) -> Path:
    if not path:
        return DEFAULT_MATCHING_TEMPLATE
    expanded = Path(os.path.expanduser(path))
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def build_simulation_config(
    *,
    template_path: Path,
    zone_csv: Path,
    assignments_dir: Path,
    precomputed_dir: Path,
    solution: ZoneSolution,
) -> dict[str, Any]:
    with open(template_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f) or {}
    if not isinstance(template, Mapping):
        raise ValueError(f"Matching config {template_path} must be a YAML mapping.")

    config = _default_matching_config()
    _deep_update(config, dict(template))

    paths = dict(config.get("paths") or {})
    paths["zone-files"] = {GENERATED_POLICY_NAME: str(zone_csv.resolve())}
    paths["assignment-folder"] = str(assignments_dir.resolve())
    paths["student-save"] = str(precomputed_dir.resolve())
    _absolutize_direct_matching_paths(paths)
    config["paths"] = paths

    config["policies"] = [GENERATED_POLICY_NAME]
    config["subconfig-name"] = template_path.stem
    config["subconfigs"] = [template_path.stem]
    config["save-assignment"] = True
    config["zone-building-blocks"] = _zone_building_blocks(solution.level.unit)
    return config


def summarize_assignment_outputs(
    *,
    assignments_dir: Path,
    matching_dir: Path,
    output_root: Path,
) -> MatchingResult:
    assignment_files = sorted(assignments_dir.rglob("*.csv"))
    if not assignment_files:
        raise ValueError(f"No assignment CSVs were written under {assignments_dir}.")

    frames = []
    summaries: list[dict[str, Any]] = []
    for assignment_file in assignment_files:
        df = pd.read_csv(assignment_file)
        df = _normalize_assignment_df(df, assignment_file, assignments_dir)
        frames.append(df)
        summaries.append(_assignment_summary(df, assignment_file, assignments_dir))

    combined = pd.concat(frames, ignore_index=True)
    student_path = matching_dir / STUDENT_ASSIGNMENTS_CSV
    school_path = matching_dir / SCHOOL_POPULATIONS_CSV
    program_path = matching_dir / PROGRAM_POPULATIONS_CSV

    combined.to_csv(student_path, index=False)
    _school_populations(combined).to_csv(school_path, index=False)
    _program_populations(combined).to_csv(program_path, index=False)

    metrics = _matching_metrics(summaries)
    artifacts = {
        "assignments_raw": _relpath(assignments_dir, output_root),
        "student_school_assignments": _relpath(student_path, output_root),
        "school_populations": _relpath(school_path, output_root),
        "program_populations": _relpath(program_path, output_root),
    }
    return MatchingResult(
        status="OK",
        metrics=metrics,
        artifacts=artifacts,
        run={"assignments": summaries},
    )


def merge_matching_result(
    payload: dict[str, Any], matching_result: MatchingResult | None
) -> dict[str, Any]:
    if matching_result is None:
        return payload
    payload["matching"] = matching_result.to_payload()
    payload.setdefault("metrics", {}).update(matching_result.metrics)
    return payload


def preserve_matching_payload(
    new_payload: dict[str, Any], previous_payload: Mapping[str, Any]
) -> dict[str, Any]:
    matching_payload = previous_payload.get("matching")
    if matching_payload is not None:
        new_payload["matching"] = matching_payload
    previous_metrics = previous_payload.get("metrics") or {}
    matching_metrics = {
        key: value
        for key, value in previous_metrics.items()
        if str(key).startswith("matching_")
    }
    if matching_metrics:
        new_payload.setdefault("metrics", {}).update(matching_metrics)
    return new_payload


def _run_student_assignment(config: dict[str, Any], assignments_dir: Path) -> None:
    from student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    configurator = _StaticConfigurator(config)
    market = MarketGenerator(
        configurator=configurator,
        assignment_path=str(assignments_dir),
    )
    MarketGenerator.execute_generator(market.create_iterations_generator())


class _StaticConfigurator:
    """Minimal Configerator-compatible object for generated in-memory configs."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._original_config = None


def _default_matching_config() -> dict[str, Any]:
    return {
        "desig_after_mainround": False,
        "grade": "KG",
        "iterations": {"start": 0, "end": 1},
        "paths": {
            "sfusd": "/share/data/school_choice",
            "student-data": "Data/Cleaned/r1_filter_student_without_specialprogs_2324.csv",
            "program-data": "Data/Cleaned/programs_without_specialprogs_2324.csv",
            "school-data": "Data/Cleaned/schools_rehauled_withMissionBay_2324.csv",
            "estimate-path": "simulation-files/choice-model/estimates_2324_exp8_0514.csv",
            "zone-files": {},
            "citywide-or-lp-zones": {},
        },
        "r1-only": True,
        "random-seed": 2023,
        "remove-special-lps": True,
        "rounds-merged-options": [0],
        "save-assignment": True,
        "subconfigs": [],
        "utility-model": {
            "designate-lp-for-all": False,
            "enable": True,
            "list-length": "0.8*round(real_length)",
        },
        "year": 23,
    }


def _absolutize_direct_matching_paths(paths: dict[str, Any]) -> None:
    sfusd_root = paths.get("sfusd")
    if not sfusd_root:
        return
    sfusd_root = os.path.expanduser(str(sfusd_root))
    for key in ["estimate-path"]:
        value = paths.get(key)
        if value and not os.path.isabs(os.path.expanduser(str(value))):
            paths[key] = os.path.abspath(os.path.join(sfusd_root, str(value)))

    citywide = paths.get("citywide-or-lp-zones") or {}
    paths["citywide-or-lp-zones"] = {
        name: (
            os.path.abspath(os.path.join(sfusd_root, str(path)))
            if path and not os.path.isabs(os.path.expanduser(str(path)))
            else path
        )
        for name, path in citywide.items()
    }


def _normalize_assignment_df(
    df: pd.DataFrame, assignment_file: Path, assignments_dir: Path
) -> pd.DataFrame:
    out = df.copy()
    out["assignment_file"] = _relpath(assignment_file, assignments_dir)
    out["assignment_name"] = assignment_file.stem

    programcodes = out.get("programcodes")
    if programcodes is None:
        programcodes = pd.Series([None] * len(out), index=out.index)
    parsed = programcodes.fillna("").astype(str).str.split("-", expand=True)
    out["school_id"] = pd.to_numeric(parsed.get(0), errors="coerce").astype("Int64")
    out["program_type"] = parsed.get(1) if 1 in parsed else pd.NA
    out["grade"] = parsed.get(2) if 2 in parsed else pd.NA
    return out


def _assignment_summary(
    df: pd.DataFrame, assignment_file: Path, assignments_dir: Path
) -> dict[str, Any]:
    assigned = df["programno"].fillna(0).astype(int) > 0
    total = int(len(df))
    assigned_count = int(assigned.sum())
    return {
        "assignment_file": _relpath(assignment_file, assignments_dir),
        "students_total": total,
        "students_assigned": assigned_count,
        "students_unassigned": total - assigned_count,
        "unassigned_rate": (total - assigned_count) / total if total else None,
        "schools_with_assignments": int(df.loc[assigned, "school_id"].nunique()),
        "programs_with_assignments": int(df.loc[assigned, "programcodes"].nunique()),
    }


def _school_populations(assignments: pd.DataFrame) -> pd.DataFrame:
    assigned = assignments[assignments["programno"].fillna(0).astype(int) > 0]
    if assigned.empty:
        return pd.DataFrame(columns=["assignment_file", "school_id", "assigned_count"])
    return (
        assigned.groupby(["assignment_file", "school_id"], dropna=False)
        .size()
        .reset_index(name="assigned_count")
    )


def _program_populations(assignments: pd.DataFrame) -> pd.DataFrame:
    assigned = assignments[assignments["programno"].fillna(0).astype(int) > 0]
    columns = [
        "assignment_file",
        "programno",
        "programcodes",
        "school_id",
        "program_type",
        "grade",
        "assigned_count",
    ]
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    return (
        assigned.groupby(
            [
                "assignment_file",
                "programno",
                "programcodes",
                "school_id",
                "program_type",
                "grade",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="assigned_count")
    )


def _matching_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [int(row["students_total"]) for row in summaries]
    assigned = [int(row["students_assigned"]) for row in summaries]
    unassigned = [int(row["students_unassigned"]) for row in summaries]
    rates = [row["unassigned_rate"] for row in summaries if row["unassigned_rate"] is not None]
    return {
        "matching_assignment_files": len(summaries),
        "matching_students_total": totals[0] if len(set(totals)) == 1 else sum(totals),
        "matching_students_assigned_mean": sum(assigned) / len(assigned),
        "matching_students_unassigned_mean": sum(unassigned) / len(unassigned),
        "matching_unassigned_rate_mean": sum(rates) / len(rates) if rates else None,
        "matching_unassigned_rate_max": max(rates) if rates else None,
    }


def _zone_building_blocks(unit: str) -> str:
    if unit == "BlockGroup":
        return "block_group"
    if unit == "Block":
        return "block"
    if unit == "attendance_area":
        return "attendance_area"
    raise ValueError(f"Unsupported matching unit: {unit}")


def _reset_matching_dir(matching_dir: Path) -> None:
    if matching_dir.exists():
        shutil.rmtree(matching_dir)
    matching_dir.mkdir(parents=True, exist_ok=True)


def _mark_matching_error(run_dir: str, error_message: str, trace: str) -> None:
    from Zone_Generation.benchmark.runner import (
        MANIFEST_FILENAME,
        RESULT_FILENAME,
        load_manifest,
        write_json,
    )

    result_path = os.path.join(run_dir, RESULT_FILENAME)
    payload = _load_json(result_path)
    payload["status"] = "ERROR"
    payload["error_message"] = error_message
    payload["matching"] = {
        "enabled": True,
        "status": "ERROR",
        "error_message": error_message,
        "traceback": trace,
    }
    write_json(result_path, payload)

    try:
        manifest = load_manifest(run_dir)
    except Exception:
        return
    manifest["status"] = "ERROR"
    manifest["error_message"] = error_message
    manifest["matching_error"] = True
    write_json(os.path.join(run_dir, MANIFEST_FILENAME), manifest)


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _load_json(path: str | Path) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(data), f, indent=2, sort_keys=True)


def _relpath(path: str | Path, root: str | Path) -> str:
    return os.path.relpath(os.path.expanduser(str(path)), os.path.expanduser(str(root)))
