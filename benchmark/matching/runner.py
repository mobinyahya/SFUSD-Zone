"""Run student-assignment simulations from saved zoning outputs."""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from benchmark.config import (
    ChoiceMetricsRunConfig,
    MatchingConfigSpec,
    MatchingRunConfig,
    json_ready,
)
from metrics.base import MetricsContext
from optimization.solution import ZoneSolution


DEFAULT_MATCHING_TEMPLATE = Path("benchmark/matching/zones+hard_reserves_06frl.yaml")
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


@dataclass(frozen=True)
class _PreparedMatchingRun:
    name: str
    config_template: str
    simulation_config: dict[str, Any]
    matching_dir: str
    assignments_dir: str
    output_root: str
    generated_config: str
    zone_csv: str
    zone_id_map: dict[int, int]
    level: str
    workers: int


def run_matching_for_solution(
    solution: ZoneSolution,
    output_dir: str,
    matching: MatchingRunConfig,
    *,
    student_assignment_session: StudentAssignmentSession | None = None,
    precomputed_dir: str | Path | None = None,
    workers: int = 1,
) -> MatchingResult | None:
    """Run matching for one final zoning solution and write run artifacts."""

    if not matching.enabled:
        return None
    if not solution.feasible:
        return None
    if not solution.assignment:
        raise ValueError("Cannot run matching without a final zone assignment.")

    output_root = Path(os.path.expanduser(output_dir)).resolve()
    matching_dir = output_root / MATCHING_DIRNAME
    _reset_matching_dir(matching_dir)

    zone_csv = matching_dir / ZONE_CSV
    zone_id_map = write_matching_zone_csv(solution.area_assignment(), zone_csv)
    precomputed_base_dir = (
        Path(os.path.expanduser(str(precomputed_dir))).resolve()
        if precomputed_dir is not None
        else matching_dir / "precomputed"
    )

    config_specs = matching.config_specs()
    named_configs = _named_matching_configs(config_specs)
    legacy_layout = len(named_configs) == 1
    worker_count = max(1, int(workers or 1))
    prepared_runs = [
        _prepare_matching_run(
            solution=solution,
            output_root=output_root,
            root_matching_dir=matching_dir,
            config_name=config_name,
            config_spec=config_spec,
            zone_csv=zone_csv,
            zone_id_map=zone_id_map,
            precomputed_base_dir=precomputed_base_dir,
            legacy_layout=legacy_layout,
            workers=worker_count,
        )
        for config_name, config_spec in named_configs
    ]

    if legacy_layout:
        return _execute_prepared_matching_run(
            prepared_runs[0],
            student_assignment_session=student_assignment_session,
        )

    if worker_count > 1:
        results = _execute_prepared_matching_runs_parallel(prepared_runs, worker_count)
    else:
        results = [
            _execute_prepared_matching_run(
                prepared_run,
                student_assignment_session=student_assignment_session,
            )
            for prepared_run in prepared_runs
        ]

    result = _combined_matching_result(
        results=results,
        output_root=output_root,
        matching_dir=matching_dir,
        zone_csv=zone_csv,
        zone_id_map=zone_id_map,
        solution=solution,
        workers=worker_count,
    )
    result.artifacts["summary"] = _relpath(matching_dir / SUMMARY_JSON, output_root)
    _write_json(matching_dir / SUMMARY_JSON, result.to_payload())
    return result


def _prepare_matching_run(
    *,
    solution: ZoneSolution,
    output_root: Path,
    root_matching_dir: Path,
    config_name: str,
    config_spec: MatchingConfigSpec,
    zone_csv: Path,
    zone_id_map: dict[int, int],
    precomputed_base_dir: Path,
    legacy_layout: bool,
    workers: int,
) -> _PreparedMatchingRun:
    matching_dir = (
        root_matching_dir if legacy_layout else root_matching_dir / config_name
    )
    assignments_dir = matching_dir / ASSIGNMENTS_RAW_DIR
    assignments_dir.mkdir(parents=True, exist_ok=True)

    precomputed_dir = (
        precomputed_base_dir if legacy_layout else precomputed_base_dir / config_name
    )
    config_template = resolve_matching_template(config_spec.config)
    simulation_config = build_simulation_config(
        template_path=config_template,
        zone_csv=zone_csv,
        assignments_dir=assignments_dir,
        precomputed_dir=precomputed_dir,
        solution=solution,
    )
    simulation_config["workers"] = max(1, int(workers or 1))

    return _PreparedMatchingRun(
        name=config_name,
        config_template=str(config_template),
        simulation_config=simulation_config,
        matching_dir=str(matching_dir),
        assignments_dir=str(assignments_dir),
        output_root=str(output_root),
        generated_config=str(matching_dir / GENERATED_CONFIG),
        zone_csv=str(zone_csv),
        zone_id_map=zone_id_map,
        level=solution.level.name,
        workers=max(1, int(workers or 1)),
    )


def _execute_prepared_matching_runs_parallel(
    prepared_runs: list[_PreparedMatchingRun], workers: int
) -> list[MatchingResult]:
    max_workers = min(max(1, int(workers or 1)), len(prepared_runs))
    results_by_name: dict[str, MatchingResult] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _execute_prepared_matching_run, prepared_run
            ): prepared_run.name
            for prepared_run in prepared_runs
        }
        for future in as_completed(futures):
            name = futures[future]
            results_by_name[name] = future.result()
    return [results_by_name[prepared_run.name] for prepared_run in prepared_runs]


def _execute_prepared_matching_run(
    prepared_run: _PreparedMatchingRun,
    student_assignment_session: StudentAssignmentSession | None = None,
) -> MatchingResult:
    matching_dir = Path(prepared_run.matching_dir)
    assignments_dir = Path(prepared_run.assignments_dir)
    output_root = Path(prepared_run.output_root)
    generated_config = Path(prepared_run.generated_config)
    zone_csv = Path(prepared_run.zone_csv)

    matching_dir.mkdir(parents=True, exist_ok=True)
    assignments_dir.mkdir(parents=True, exist_ok=True)
    with open(generated_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(json_ready(prepared_run.simulation_config), f, sort_keys=True)

    if student_assignment_session is None:
        _run_student_assignment(
            prepared_run.simulation_config,
            assignments_dir,
            workers=prepared_run.workers,
        )
    else:
        student_assignment_session.run(
            prepared_run.simulation_config,
            assignments_dir,
            workers=prepared_run.workers,
        )
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
            "config_name": prepared_run.name,
            "config_template": prepared_run.config_template,
            "policy_name": GENERATED_POLICY_NAME,
            "zone_id_map": {str(k): v for k, v in prepared_run.zone_id_map.items()},
            "zone_building_blocks": prepared_run.simulation_config.get(
                "zone-building-blocks"
            ),
            "level": prepared_run.level,
            "workers": prepared_run.workers,
        }
    )
    result.artifacts["summary"] = _relpath(matching_dir / SUMMARY_JSON, output_root)
    _write_json(matching_dir / SUMMARY_JSON, result.to_payload())
    return result


def _combined_matching_result(
    *,
    results: list[MatchingResult],
    output_root: Path,
    matching_dir: Path,
    zone_csv: Path,
    zone_id_map: dict[int, int],
    solution: ZoneSolution,
    workers: int,
) -> MatchingResult:
    metrics: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    artifacts: dict[str, Any] = {
        "zone_csv": _relpath(zone_csv, output_root),
        "runs": {},
    }
    for result in results:
        name = str(result.run.get("config_name") or "default")
        runs[name] = result.to_payload()
        artifacts["runs"][name] = result.artifacts
        metrics.update(_prefix_matching_metrics(name, result.metrics))

    return MatchingResult(
        status="OK" if all(result.status == "OK" for result in results) else "ERROR",
        metrics=metrics,
        artifacts=artifacts,
        run={
            "configs": list(runs),
            "runs": runs,
            "workers": max(1, int(workers or 1)),
            "policy_name": GENERATED_POLICY_NAME,
            "zone_id_map": {str(k): v for k, v in zone_id_map.items()},
            "zone_building_blocks": _zone_building_blocks(solution.level.unit),
            "level": solution.level.name,
        },
    )


def run_matching_for_existing_runs(
    root_folder: str,
    matching: MatchingRunConfig,
    *,
    choice_metrics: ChoiceMetricsRunConfig | None = None,
    fail_fast: bool = False,
    dataset_factory=None,
) -> MatchingBatchResult:
    """Run matching-only regeneration for saved benchmark run folders."""

    from benchmark.results import discover_run_dirs
    from benchmark.runner import (
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
            batch.add(
                MatchingTaskResult(run_dir=run_dir, status="SKIPPED", skipped=True)
            )
        return batch

    for run_dir in run_dirs:
        try:
            dataset = None
            if dataset_factory is not None:
                from benchmark.config import (
                    optimization_config_from_dict,
                )

                manifest_for_dataset = load_manifest(run_dir)
                config_for_dataset = optimization_config_from_dict(
                    manifest_for_dataset["config"]
                )
                dataset = dataset_factory(config_for_dataset, manifest_for_dataset)

            solutions, config, manifest = load_solutions(run_dir, dataset=dataset)
            if not solutions:
                batch.add(
                    MatchingTaskResult(run_dir=run_dir, status="SKIPPED", skipped=True)
                )
                continue
            matching_workers = max(1, int(config.workers or 1))
            student_assignment_session = _new_student_assignment_session()
            shared_precomputed_dir = (
                Path(os.path.expanduser(run_dir)).resolve()
                / MATCHING_DIRNAME
                / "precomputed"
            )
            final_solution = MetricsContext(solutions, config=config).solution
            matching_result = None
            if final_solution.feasible:
                matching_result = run_matching_for_solution(
                    final_solution,
                    run_dir,
                    matching,
                    student_assignment_session=student_assignment_session,
                    precomputed_dir=shared_precomputed_dir,
                    workers=matching_workers,
                )
            stage_matching_result = run_matching_for_stages(
                solutions,
                manifest.get("stages", []),
                run_dir,
                matching,
                choice_metrics=choice_metrics,
                student_assignment_session=student_assignment_session,
                precomputed_dir=shared_precomputed_dir,
                workers=matching_workers,
            )
            result_path = os.path.join(run_dir, RESULT_FILENAME)
            payload = _load_json(result_path)
            clear_matching_payload(payload)
            if (
                choice_metrics and choice_metrics.enabled
            ) or not final_solution.feasible:
                from benchmark.choice_metrics import (
                    clear_choice_metrics_payload,
                )

                clear_choice_metrics_payload(payload)
            merge_matching_result(payload, matching_result)
            merge_stage_matching_result(payload, stage_matching_result)
            if choice_metrics and choice_metrics.enabled:
                from benchmark.choice_metrics import (
                    compute_choice_metrics_for_run,
                    compute_choice_metrics_for_stages,
                    merge_choice_metrics_result,
                    merge_stage_choice_metrics_result,
                )

                choice_result = None
                if final_solution.feasible:
                    choice_result = compute_choice_metrics_for_run(
                        run_dir,
                        choice_metrics,
                    )
                stage_choice_result = None
                if not (stage_matching_result and choice_metrics.compute_stage_metrics):
                    stage_choice_result = compute_choice_metrics_for_stages(
                        run_dir,
                        choice_metrics,
                        manifest.get("stages", []),
                    )
                merge_choice_metrics_result(payload, choice_result)
                merge_stage_choice_metrics_result(payload, stage_choice_result)
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


def run_matching_for_stages(
    solutions: list[ZoneSolution],
    stage_records: list[dict[str, Any]],
    output_dir: str,
    matching: MatchingRunConfig,
    *,
    choice_metrics: ChoiceMetricsRunConfig | None = None,
    student_assignment_session: StudentAssignmentSession | None = None,
    precomputed_dir: str | Path | None = None,
    workers: int = 1,
) -> dict[str, Any] | None:
    """Optionally run matching and choice metrics for every saved stage."""

    if not matching.enabled or not matching.compute_stage_assignments:
        return None

    output_root = Path(os.path.expanduser(output_dir)).resolve()
    stages: dict[str, Any] = {}
    for solution, stage in zip(solutions, stage_records):
        if not solution.feasible or solution.metadata.get("partial_assignment"):
            continue
        stage_name = str(stage.get("name"))
        stage_dir = output_root / str(stage.get("path"))
        matching_result = run_matching_for_solution(
            solution,
            str(stage_dir),
            matching,
            student_assignment_session=student_assignment_session,
            precomputed_dir=precomputed_dir,
            workers=workers,
        )
        stage_payload: dict[str, Any] = {}
        if matching_result is not None:
            stage_payload["matching"] = matching_result.to_payload()
            stage["matching"] = stage_payload["matching"]

        if (
            choice_metrics
            and choice_metrics.enabled
            and choice_metrics.compute_stage_metrics
        ):
            from benchmark.choice_metrics import (
                compute_choice_metrics_for_run,
            )

            choice_result = compute_choice_metrics_for_run(
                str(stage_dir), choice_metrics
            )
            if choice_result is not None:
                stage_payload["choice_metrics"] = choice_result.to_payload()
                stage["choice_metrics"] = stage_payload["choice_metrics"]

        stages[stage_name] = stage_payload

    return {"enabled": True, "stages": stages}


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

    precomputed_dir.mkdir(parents=True, exist_ok=True)
    utility_model = dict(config.get("utility-model") or {})
    utility_model["save-path"] = str((precomputed_dir / "utility_matrix.npy").resolve())
    config["utility-model"] = utility_model

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


def clear_matching_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("matching", None)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in list(metrics):
            if str(key).startswith("matching_"):
                metrics.pop(key, None)

    stage_matching = payload.get("stage_matching")
    if isinstance(stage_matching, dict):
        for stage_payload in (stage_matching.get("stages") or {}).values():
            if isinstance(stage_payload, dict):
                stage_payload.pop("matching", None)
    for stage in (payload.get("run") or {}).get("stages", []):
        if isinstance(stage, dict):
            stage.pop("matching", None)
            stage.pop("matching_metrics", None)
    return payload


def merge_stage_matching_result(
    payload: dict[str, Any], stage_matching_result: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not stage_matching_result:
        return payload
    payload["stage_matching"] = json_ready(stage_matching_result)
    run_stages = {
        stage.get("name"): stage
        for stage in (payload.get("run") or {}).get("stages", [])
    }
    for stage_name, stage_payload in stage_matching_result.get("stages", {}).items():
        row = run_stages.get(stage_name)
        if row is None:
            continue
        matching_payload = stage_payload.get("matching")
        if matching_payload is not None:
            row["matching"] = matching_payload
            row["matching_metrics"] = matching_payload.get("metrics", {})
        choice_payload = stage_payload.get("choice_metrics")
        if choice_payload is not None:
            row["choice_metrics"] = choice_payload
            row["choice_metrics_metrics"] = choice_payload.get("metrics", {})
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


def _run_student_assignment(
    config: dict[str, Any], assignments_dir: Path, *, workers: int = 1
) -> None:
    MarketGenerator = _market_generator_class()

    config["workers"] = max(1, int(workers or 1))
    _install_student_assignment_config(config)
    market = MarketGenerator(
        assignment_path=str(assignments_dir),
    )
    MarketGenerator.execute_generator(market.create_iterations_generator())


class StudentAssignmentSession:
    """Reuse one student-assignment market while swapping zoning artifacts."""

    def __init__(self) -> None:
        self.market = None
        self.configurator: _StaticConfigurator | None = None
        self._static_signature: str | None = None

    def run(
        self, config: dict[str, Any], assignments_dir: Path, *, workers: int = 1
    ) -> None:
        run_config = copy.deepcopy(config)
        run_config["workers"] = max(1, int(workers or 1))
        static_signature = _student_assignment_static_signature(run_config)
        if self.market is None or static_signature != self._static_signature:
            self._initialize_market(run_config, assignments_dir, static_signature)
        else:
            self._update_market(run_config, assignments_dir)

        MarketGenerator = self.market.__class__
        MarketGenerator.execute_generator(self.market.create_iterations_generator())

    def _initialize_market(
        self,
        config: dict[str, Any],
        assignments_dir: Path,
        static_signature: str,
    ) -> None:
        MarketGenerator = _market_generator_class()
        self.configurator = _install_student_assignment_config(config)
        self.market = MarketGenerator(
            assignment_path=str(assignments_dir),
        )
        self._static_signature = static_signature

    def _update_market(self, config: dict[str, Any], assignments_dir: Path) -> None:
        if self.configurator is None or self.market is None:
            raise RuntimeError("StudentAssignmentSession has not been initialized.")

        self.configurator.config = config
        self.market.config = config
        self.market._set_up_save_folder(str(assignments_dir))

        # These generators hold zone-dependent caches, so reset them per zoning.
        self.market.priority_generator = self.market.priority_generator.__class__(
            self.market
        )
        self.market.preference_generator = self.market.preference_generator.__class__(
            self.market
        )


class _StaticConfigurator:
    """Minimal Configerator-compatible object for generated in-memory configs."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._original_config = None


def _new_student_assignment_session() -> StudentAssignmentSession:
    return StudentAssignmentSession()


def _install_student_assignment_config(config: dict[str, Any]) -> _StaticConfigurator:
    from student_assignment.configerator import Configerator

    configurator = _StaticConfigurator(config)
    Configerator.instance = configurator
    return configurator


def _patch_student_assignment_guardrail_pandas_compat() -> None:
    """Allow reserve zone fractions to stay fractional on newer pandas."""

    from student_assignment.da.guardrail_setup import GuardrailSetup

    if getattr(GuardrailSetup._calculate_zone_fractions, "_sfusd_pandas_compat", False):
        return

    def _calculate_zone_fractions(self):
        data = self.students.student_data
        data["count"] = 1.0
        data["zone_id"] = [
            self.student2zone[x] if x in self.student2zone else float("nan")
            for x in data.index
        ]
        count_per_zone = (
            data[["zone_id", "diversity_category", "count"]]
            .groupby(["zone_id", "diversity_category"], as_index=False)
            .sum()
        )
        zone_total = data[["zone_id", "count"]].groupby("zone_id", as_index=False).sum()
        count_per_zone = count_per_zone.merge(
            zone_total, how="left", on="zone_id", suffixes=("", "_tot")
        )
        count_per_zone["count"] = (
            count_per_zone["count"].astype(float) / count_per_zone["count_tot"]
        )
        return pd.pivot_table(
            count_per_zone,
            index="zone_id",
            columns="diversity_category",
            values="count",
            fill_value=0,
        )

    _calculate_zone_fractions._sfusd_pandas_compat = True
    GuardrailSetup._calculate_zone_fractions = _calculate_zone_fractions


def _patch_student_assignment_empty_excess_match_compat() -> None:
    """Avoid strict guardrail evictions when the virtual school is empty."""

    from student_assignment.da.da import School

    if getattr(School.has_excess_matches, "_sfusd_empty_excess_compat", False):
        return

    def has_excess_matches(self):
        return bool(self.matches) and self.capacity < len(self.matches)

    has_excess_matches._sfusd_empty_excess_compat = True
    School.has_excess_matches = has_excess_matches


def _market_generator_class():
    from student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    _patch_student_assignment_guardrail_pandas_compat()
    _patch_student_assignment_empty_excess_match_compat()
    return MarketGenerator


def _student_assignment_static_signature(config: Mapping[str, Any]) -> str:
    signature_config = copy.deepcopy(dict(config))
    paths = dict(signature_config.get("paths") or {})
    paths.pop("assignment-folder", None)
    paths.pop("zone-files", None)
    signature_config["paths"] = paths
    return json.dumps(
        json_ready(signature_config),
        sort_keys=True,
        separators=(",", ":"),
    )


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
    rates = [
        row["unassigned_rate"]
        for row in summaries
        if row["unassigned_rate"] is not None
    ]
    return {
        "matching_assignment_files": len(summaries),
        "matching_students_total": totals[0] if len(set(totals)) == 1 else sum(totals),
        "matching_students_assigned_mean": sum(assigned) / len(assigned),
        "matching_students_unassigned_mean": sum(unassigned) / len(unassigned),
        "matching_unassigned_rate_mean": sum(rates) / len(rates) if rates else None,
        "matching_unassigned_rate_max": max(rates) if rates else None,
    }


def _named_matching_configs(
    config_specs: list[MatchingConfigSpec],
) -> list[tuple[str, MatchingConfigSpec]]:
    used: dict[str, int] = {}
    named: list[tuple[str, MatchingConfigSpec]] = []
    for idx, config_spec in enumerate(config_specs):
        base = _safe_name(config_spec.name) or f"config_{idx}"
        count = used.get(base, 0)
        used[base] = count + 1
        name = base if count == 0 else f"{base}_{count + 1}"
        named.append((name, config_spec))
    return named


def _prefix_matching_metrics(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    prefix = f"matching_{_safe_name(name)}"
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        key_str = str(key)
        suffix = key_str.removeprefix("matching_")
        out[f"{prefix}_{suffix}"] = value
    return out


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return safe or "default"


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
    from benchmark.runner import (
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
