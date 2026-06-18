"""Result discovery and aggregation for optimization-native benchmarks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from Zone_Generation.benchmark.config import json_ready
from Zone_Generation.benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_manifest,
)


def discover_run_dirs(root_folder: str) -> list[str]:
    """Return benchmark run folders containing a manifest."""

    root = os.path.expanduser(root_folder)
    if os.path.isfile(os.path.join(root, MANIFEST_FILENAME)):
        return [root]

    run_dirs: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        if MANIFEST_FILENAME in filenames:
            run_dirs.append(dirpath)
    return sorted(run_dirs)


def load_run_result(folder_path: str) -> dict[str, Any]:
    result_path = os.path.join(os.path.expanduser(folder_path), RESULT_FILENAME)
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_results(
    root_folder: str,
    *,
    summary_csv: str | None = None,
    stages_csv: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate run-level and stage-level benchmark outputs."""

    root = os.path.expanduser(root_folder)
    run_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []

    for run_dir in discover_run_dirs(root):
        try:
            manifest = load_manifest(run_dir)
        except (OSError, json.JSONDecodeError):
            continue

        result = _load_result_if_present(run_dir)
        run_rows.append(_run_row(root, run_dir, manifest, result))
        stage_rows.extend(_stage_rows(root, run_dir, manifest, result))

    run_df = pd.DataFrame(run_rows)
    stage_df = pd.DataFrame(stage_rows)

    if summary_csv:
        _write_csv(run_df, root, summary_csv)
    if stages_csv:
        _write_csv(stage_df, root, stages_csv)
    return run_df, stage_df


def _run_row(
    root: str, run_dir: str, manifest: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    config = dict(manifest.get("config") or result.get("config") or {})
    row = {
        "task_id": manifest.get("task_id"),
        "config_hash": manifest.get("config_hash"),
        "path": run_dir,
        "rel_path": os.path.relpath(run_dir, root),
        "status": result.get("status") or manifest.get("status"),
        "error_message": result.get("error_message") or manifest.get("error_message"),
        "total_wall_time": result.get("total_wall_time", manifest.get("total_wall_time")),
        "final_stage": (result.get("run") or {}).get("final_stage") or manifest.get("final_stage"),
        "levels": _join(result.get("levels") or [s.get("level") for s in manifest.get("stages", [])]),
        "num_stages": len(manifest.get("stages", [])),
    }
    row.update({f"config_{k}": _cell(v) for k, v in config.items()})
    row.update(result.get("metrics") or {})
    return row


def _stage_rows(
    root: str, run_dir: str, manifest: Mapping[str, Any], result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    run_stage_by_name = {
        stage.get("name"): stage for stage in (result.get("run") or {}).get("stages", [])
    }
    rows: list[dict[str, Any]] = []
    for stage in manifest.get("stages", []):
        metrics_stage = run_stage_by_name.get(stage.get("name"), {})
        row = {
            "task_id": manifest.get("task_id"),
            "config_hash": manifest.get("config_hash"),
            "path": run_dir,
            "rel_path": os.path.relpath(run_dir, root),
            "stage_name": stage.get("name"),
            "stage_path": stage.get("path"),
            "stage_index": stage.get("index"),
            "level": stage.get("level"),
            "status": stage.get("status"),
            "objective": stage.get("objective"),
            "cut_edges": metrics_stage.get("cut_edges"),
            "fractional_cut_edges": metrics_stage.get("fractional_cut_edges"),
            "avg_reock_score": metrics_stage.get("avg_reock_score"),
            "avg_polsby_popper_score": metrics_stage.get("avg_polsby_popper_score"),
            "wall_time": stage.get("wall_time"),
            "contiguous": stage.get("contiguous"),
            "num_nodes": metrics_stage.get("num_nodes"),
            "num_zones": stage.get("num_zones"),
        }
        rows.append(row)
    return rows


def _load_result_if_present(run_dir: str) -> dict[str, Any]:
    result_path = os.path.join(run_dir, RESULT_FILENAME)
    if not os.path.exists(result_path):
        return {}
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_csv(df: pd.DataFrame, root: str, path: str) -> None:
    output = Path(path)
    if not output.is_absolute():
        output = Path(root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def _cell(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _join(values) -> str:
    return "-".join(str(v) for v in values if v is not None)
