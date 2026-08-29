"""Simulation sweep configuration for optimization-native benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from loaders import DataScenario, anchor_data_config
from optimization.config import OptimizationConfig


SEQUENCE_OPTIMIZATION_FIELDS = {
    "levels",
    "solve_time_limits",
    "gap_limits",
}
SPECIAL_FLOATS = {"Infinity": math.inf, "-Infinity": -math.inf}
OPTIMIZATION_SOURCE_ROLES = (
    "optimization.students",
    "optimization.frl_estimate",
    "optimization.schools",
    "optimization.programs",
    "optimization.census",
    "optimization.crosswalk",
    "optimization.adjacency",
    "optimization.centroids",
    "optimization.manual_edges",
)
MNL_ASSIGNMENT_FILTERS = (
    "year",
    "grades",
    "student_population",
    "rounds",
    "special_programs",
    "capacity_profile",
    "capacity_scenario",
    "include_mission_bay",
    "geography_vintage",
    "frl_estimate",
    "outside_district_students",
)


@dataclass(frozen=True)
class ExecutionConfig:
    """How a simulation sweep should be executed."""

    output_dir: str = "./benchmark_output"
    capacity: int | None = None
    max_workers: int | None = None
    max_tasks_per_worker: int | None = 25
    queue_multiplier: int = 2
    skip_existing: bool = True
    rerun_failed: bool = True
    sequential: bool = False
    fail_fast: bool = False
    task_capacity: int | None = None
    output_template: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ExecutionConfig":
        return _dataclass_from_dict(cls, data or {})


@dataclass(frozen=True)
class MetricsRunConfig:
    """Metric and aggregation settings for a simulation sweep."""

    strict: bool = True
    compute_stage_metrics: bool = False
    summary_csv: str = "summary.csv"
    stages_csv: str = "stages.csv"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MetricsRunConfig":
        return _dataclass_from_dict(cls, data or {})


@dataclass(frozen=True)
class VisualizationRunConfig:
    """Solution-map settings for every run in a simulation sweep."""

    enabled: bool = False
    stages: str = "final"
    artifact_dir: str | None = None

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, *, base_dir: Path | None = None
    ) -> "VisualizationRunConfig":
        raw = dict(data or {})
        allowed = {"enabled", "stages", "artifact_dir"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")

        stages = str(raw.get("stages", "final"))
        if stages not in {"final", "all"}:
            raise ValueError("visualization.stages must be 'final' or 'all'.")

        artifact_dir = raw.get("artifact_dir")
        if artifact_dir is not None:
            path = Path(str(artifact_dir)).expanduser()
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            artifact_dir = str(path.resolve())

        return cls(
            enabled=bool(raw.get("enabled", False)),
            stages=stages,
            artifact_dir=artifact_dir,
        )


@dataclass(frozen=True)
class MatchingRunConfig:
    """Assignment base config used with generated benchmark zones."""

    enabled: bool = False
    config: str | None = None
    compute_stage_assignments: bool = False

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any] | None, *, base_dir: Path | None = None
    ) -> "MatchingRunConfig":
        raw = dict(data or {})
        allowed = {"enabled", "config", "compute_stage_assignments"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")

        config = raw.get("config")
        if config is not None:
            path = Path(str(config)).expanduser()
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            config = str(path.resolve())
        if raw.get("enabled") and config is None:
            raise ValueError("matching.config is required when matching is enabled.")

        return cls(
            enabled=bool(raw.get("enabled", False)),
            config=config,
            compute_stage_assignments=bool(raw.get("compute_stage_assignments", False)),
        )


@dataclass(frozen=True)
class BenchmarkTask:
    """One concrete optimization run generated from a simulation sweep."""

    task_id: str
    config_hash: str
    config: dict[str, Any]
    output_dir: str
    capacity_slots: int

    def optimization_config(self) -> OptimizationConfig:
        return optimization_config_from_dict(self.config)


@dataclass(frozen=True)
class SimulationSweep:
    """Top-level YAML-backed benchmark description."""

    name: str = "simulation_sweep"
    mode: str = "run"
    optimization_defaults: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    metrics: MetricsRunConfig = field(default_factory=MetricsRunConfig)
    visualization: VisualizationRunConfig = field(
        default_factory=VisualizationRunConfig
    )
    matching: MatchingRunConfig = field(default_factory=MatchingRunConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationSweep":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("Simulation sweep YAML must be a mapping.")
        allowed_top_level = {
            "name",
            "mode",
            "optimization_defaults",
            "sweep",
            "tasks",
            "execution",
            "metrics",
            "visualization",
            "matching",
        }
        unknown_top_level = set(raw) - allowed_top_level
        if unknown_top_level:
            raise ValueError(f"Unknown sweep YAML keys: {sorted(unknown_top_level)}")

        name = str(raw.get("name") or config_path.stem)
        mode = str(raw.get("mode", "run"))
        if mode not in {"run", "metrics", "matching"}:
            raise ValueError("mode must be one of: run, metrics, matching")

        optimization_defaults = dict(raw.get("optimization_defaults") or {})
        sweep = dict(raw.get("sweep") or {})
        raw_tasks = list(raw.get("tasks") or [])
        _validate_optimization_keys(optimization_defaults, "optimization_defaults")
        _validate_optimization_keys(sweep, "sweep")
        _anchor_section_data(optimization_defaults, config_path.parent)
        _anchor_section_data(sweep, config_path.parent, sweep_values=True)

        tasks = []
        for idx, task in enumerate(raw_tasks):
            if not isinstance(task, Mapping):
                raise ValueError(f"tasks[{idx}] must be a mapping.")
            _validate_optimization_keys(task, f"tasks[{idx}]")
            task = dict(task)
            _anchor_section_data(task, config_path.parent)
            tasks.append(task)

        return cls(
            name=name,
            mode=mode,
            optimization_defaults=_restore_special_values(optimization_defaults),
            sweep=_restore_special_values(sweep),
            tasks=[_restore_special_values(dict(task)) for task in tasks],
            execution=ExecutionConfig.from_dict(raw.get("execution")),
            metrics=MetricsRunConfig.from_dict(raw.get("metrics")),
            visualization=VisualizationRunConfig.from_dict(
                raw.get("visualization"), base_dir=config_path.parent
            ),
            matching=MatchingRunConfig.from_dict(
                raw.get("matching"), base_dir=config_path.parent
            ),
        )

    def generate_tasks(self) -> list[BenchmarkTask]:
        overrides = list(_sweep_overrides(self.sweep)) or [{}]
        explicit_tasks = self.tasks or [{}]
        tasks: list[BenchmarkTask] = []
        scenarios: dict[str, DataScenario] = {}
        source_manifests: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for sweep_values, task_values in product(overrides, explicit_tasks):
            config_data = dict(self.optimization_defaults)
            config_data.update(sweep_values)
            config_data.update(task_values)
            data_key = stable_hash(
                config_data.get("data", {"scenario": "legacy", "overrides": {}})
            )
            config = optimization_config_from_dict(
                config_data, data_scenario=scenarios.get(data_key)
            )
            scenarios.setdefault(data_key, config.data_scenario)
            config_dict = optimization_config_to_dict(config)
            manifest_key = (
                data_key,
                config.strategy,
                config.choice_model,
                config.capacity_scenario,
            )
            source_manifest = source_manifests.get(manifest_key)
            if source_manifest is None:
                source_manifest = _benchmark_source_manifest(config)
                source_manifests[manifest_key] = source_manifest
            config_hash = optimization_config_hash(
                config, source_manifest=source_manifest
            )
            output_dir = os.path.join(
                os.path.expanduser(self.execution.output_dir),
                format_output_path(
                    config_dict, config_hash, self.execution.output_template
                ),
            )
            tasks.append(
                BenchmarkTask(
                    task_id=config_hash[:12],
                    config_hash=config_hash,
                    config=config_dict,
                    output_dir=output_dir,
                    capacity_slots=capacity_slots(config_dict, self.execution),
                )
            )
        return tasks


def optimization_config_from_dict(
    data: Mapping[str, Any], *, data_scenario: DataScenario | None = None
) -> OptimizationConfig:
    """Construct a :class:`OptimizationConfig` from a saved config snapshot."""

    restored = _restore_special_values(dict(data))
    field_names = _optimization_field_names()
    unknown = set(restored) - field_names - {"unit"}
    if unknown:
        raise ValueError(f"Unknown optimization config keys: {sorted(unknown)}")
    kwargs = {k: v for k, v in restored.items() if k in field_names}
    return OptimizationConfig(**kwargs, _resolved_data_scenario=data_scenario)


def optimization_config_to_dict(
    config: OptimizationConfig | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(config, OptimizationConfig):
        data = asdict(config)
    else:
        data = dict(config)
        data.pop("unit", None)
    return _restore_special_values(data)


def config_snapshot(config: OptimizationConfig | Mapping[str, Any]) -> dict[str, Any]:
    optimization = optimization_config_to_dict(config)
    cfg = optimization_config_from_dict(optimization)
    snapshot = optimization_config_to_dict(cfg)
    snapshot["unit"] = cfg.unit
    return snapshot


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def optimization_config_hash(
    config: OptimizationConfig | Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any] | None = None,
) -> str:
    """Hash optimization semantics and resolved source contents deterministically."""
    resolved = (
        config
        if isinstance(config, OptimizationConfig)
        else optimization_config_from_dict(config)
    )
    semantic = copy.deepcopy(optimization_config_to_dict(resolved))
    semantic.pop("data", None)
    if semantic.get("weight_edges") is False:
        semantic.pop("weight_edges")
    if semantic.get("enumerated_solutions", -1) <= 0:
        semantic.pop("enumerated_solutions", None)
    if semantic.get("hints") != "feasible":
        semantic.pop("feasible_hint_time_limit", None)
    if source_manifest is None:
        source_manifest = _benchmark_source_manifest(resolved)
    return stable_hash(
        {
            "optimization": semantic,
            "source_manifest": source_manifest,
        }
    )


def _benchmark_source_manifest(config: OptimizationConfig) -> dict[str, Any]:
    scenario = config.data_scenario
    roles = [
        role for role in OPTIMIZATION_SOURCE_ROLES if _scenario_has_role(scenario, role)
    ]
    if config.capacity_scenario != "programs" and _scenario_has_role(
        scenario, "optimization.capacity"
    ):
        roles.append("optimization.capacity")
    filter_groups = ["optimization"]
    matching_strategy = config.strategy in {"mid", "mid_decomp", "saa"}
    if config.choice_model == "mnl" or matching_strategy:
        roles.extend(
            role
            for role in (
                "assignment.students",
                "assignment.frl_estimate",
                "assignment.geography.blocks",
                "assignment.geography.crosswalk",
                "choice.estimate",
            )
            if _scenario_has_role(scenario, role)
        )
        if scenario.filter("assignment", "capacity_scenario") != "programs" and (
            _scenario_has_role(scenario, "assignment.capacity")
        ):
            roles.append("assignment.capacity")
        filter_groups.append("assignment")
    if matching_strategy:
        roles.extend(
            role
            for role in (
                "assignment.programs",
                "assignment.schools",
                "assignment.school_coordinates",
                "assignment.program_codes",
                "assignment.ctip",
            )
            if _scenario_has_role(scenario, role)
        )

    manifest = scenario.source_manifest(dict.fromkeys(roles))
    manifest["filters"] = {"optimization": json_ready(scenario.filters["optimization"])}
    if "assignment" in filter_groups:
        manifest["filters"]["assignment"] = {
            key: json_ready(scenario.filters["assignment"][key])
            for key in MNL_ASSIGNMENT_FILTERS
        }
    if matching_strategy:
        root = Path(__file__).resolve().parents[1]
        policy_files = (
            root / "assignment/configs/base_config.yaml",
            root / "assignment/configs/policy_configs/status_quo.yaml",
        )
        manifest["matching_policy"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in policy_files
        }
    return manifest


def _scenario_has_role(scenario: Any, role: str) -> bool:
    try:
        scenario.resolved(role)
    except KeyError:
        return False
    return True


def json_ready(value: Any) -> Any:
    """Return a JSON-safe value while preserving infinity round-trips."""

    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def capacity_slots(config: Mapping[str, Any], execution: ExecutionConfig) -> int:
    if execution.task_capacity is not None:
        return max(1, int(execution.task_capacity))
    return max(1, int(config.get("workers", 1) or 1))


def format_output_path(
    config: Mapping[str, Any], config_hash: str, template: str | None = None
) -> str:
    levels_key = "-".join(str(v) for v in config.get("levels", []))
    times_key = "-".join(str(v) for v in config.get("solve_time_limits", []))
    format_values = {
        **config,
        "config_hash": config_hash[:12],
        "levels_key": levels_key,
        "solve_time_limits_key": times_key,
    }
    if template:
        return _safe_path(template.format(**format_values))

    parts = [
        _safe_path(str(config.get("centroids_type", "centroids"))),
        f"seed{_safe_path(str(config.get('seed', 'na')))}",
        (
            f"frl{_safe_path(str(config.get('frl_dev', 'na')))}_"
            f"racial{_safe_path(str(config.get('racial_dev', 'na')))}"
        ),
        (
            f"overage{_safe_path(str(config.get('overage', 'na')))}_"
            f"shortage{_safe_path(str(config.get('shortage', 'na')))}"
        ),
        _safe_path(
            f"{config.get('strategy', 'strategy')}_{config.get('solver', 'solver')}_"
            f"{levels_key}_tl_{times_key}_{config_hash[:8]}"
        ),
    ]
    return os.path.join(*parts)


def _sweep_overrides(sweep: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    if not sweep:
        return
    keys = list(sweep)
    value_lists = [_normalize_sweep_values(key, sweep[key]) for key in keys]
    for combo in product(*value_lists):
        yield dict(zip(keys, combo))


def _normalize_sweep_values(key: str, value: Any) -> list[Any]:
    if key in SEQUENCE_OPTIMIZATION_FIELDS:
        if not isinstance(value, list):
            return [value]
        if not value or all(not isinstance(v, list) for v in value):
            return [value]
        return value
    if isinstance(value, list):
        return value
    return [value]


def _dataclass_from_dict(cls, data: Mapping[str, Any]):
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in field_names})


def _validate_optimization_keys(data: Mapping[str, Any], section: str) -> None:
    unknown = set(data) - _optimization_field_names()
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")


def _anchor_section_data(
    section: dict[str, Any],
    base_dir: Path,
    *,
    sweep_values: bool = False,
) -> None:
    if "data" not in section:
        return
    value = section["data"]
    if sweep_values and isinstance(value, list):
        section["data"] = [anchor_data_config(item, base_dir) for item in value]
        return
    section["data"] = anchor_data_config(value, base_dir)


def _optimization_field_names() -> set[str]:
    return {f.name for f in fields(OptimizationConfig)}


def _restore_special_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _restore_special_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_special_values(v) for v in value]
    if isinstance(value, str) and value in SPECIAL_FLOATS:
        return SPECIAL_FLOATS[value]
    return value


def _safe_path(value: str) -> str:
    parts = []
    for part in value.split(os.sep):
        clean = re.sub(r"[^A-Za-z0-9_.=-]+", "-", part).strip("-.")
        parts.append(clean or "value")
    return os.sep.join(parts)
