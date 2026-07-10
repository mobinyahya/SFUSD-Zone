"""Simulation sweep configuration for optimization-native benchmarks."""

from __future__ import annotations

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

from optimization.config import OptimizationConfig


SEQUENCE_OPTIMIZATION_FIELDS = {"levels", "solve_time_limits", "gap_limits", "years"}
SPECIAL_FLOATS = {"Infinity": math.inf, "-Infinity": -math.inf}


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
class ChoiceMetricsRunConfig:
    """Student-assignment outcome metrics for benchmark runs."""

    enabled: bool = False
    compute_stage_metrics: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ChoiceMetricsRunConfig":
        return _dataclass_from_dict(cls, data or {})


@dataclass(frozen=True)
class MatchingConfigSpec:
    """One student-assignment template to run for each zoning solution."""

    name: str
    config: str | None = None

    @classmethod
    def from_value(cls, value: Any, index: int = 0) -> "MatchingConfigSpec":
        if value is None:
            return cls(name=f"default" if index == 0 else f"default_{index}")
        if isinstance(value, str):
            return cls(name=_matching_name_from_path(value, index), config=value)
        if not isinstance(value, Mapping):
            raise ValueError(
                "matching.configs entries must be strings or mappings with name/config."
            )

        allowed = {"name", "config", "path"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown MatchingConfigSpec keys: {sorted(unknown)}")
        config = value.get("config", value.get("path"))
        if config is not None:
            config = str(config)
        raw_name = value.get("name")
        name = str(raw_name) if raw_name else _matching_name_from_path(config, index)
        return cls(name=name, config=config)


@dataclass(frozen=True)
class MatchingRunConfig:
    """Student-assignment simulation settings for benchmark runs."""

    enabled: bool = False
    config: str | None = None
    configs: list[MatchingConfigSpec] = field(default_factory=list)
    compute_stage_assignments: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MatchingRunConfig":
        raw = dict(data or {})
        allowed = {"enabled", "config", "configs", "compute_stage_assignments"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")

        config = raw.get("config")
        if config is not None:
            config = str(config)
        configs = [
            MatchingConfigSpec.from_value(value, idx)
            for idx, value in enumerate(raw.get("configs") or [])
        ]
        if not configs and (config is not None or raw.get("enabled")):
            configs = [MatchingConfigSpec.from_value(config, 0)]

        return cls(
            enabled=bool(raw.get("enabled", False)),
            config=config,
            configs=configs,
            compute_stage_assignments=bool(raw.get("compute_stage_assignments", False)),
        )

    def config_specs(self) -> list[MatchingConfigSpec]:
        if self.configs:
            return list(self.configs)
        return [MatchingConfigSpec.from_value(self.config, 0)]


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
    matching: MatchingRunConfig = field(default_factory=MatchingRunConfig)
    choice_metrics: ChoiceMetricsRunConfig = field(
        default_factory=ChoiceMetricsRunConfig
    )

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationSweep":
        with open(path, "r", encoding="utf-8") as f:
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
            "matching",
            "choice_metrics",
        }
        unknown_top_level = set(raw) - allowed_top_level
        if unknown_top_level:
            raise ValueError(f"Unknown sweep YAML keys: {sorted(unknown_top_level)}")

        name = str(raw.get("name") or Path(path).stem)
        mode = str(raw.get("mode", "run"))
        if mode not in {"run", "metrics", "matching", "choice_metrics"}:
            raise ValueError(
                "mode must be one of: run, metrics, matching, choice_metrics"
            )

        optimization_defaults = dict(raw.get("optimization_defaults") or {})
        sweep = dict(raw.get("sweep") or {})
        tasks = list(raw.get("tasks") or [])
        _validate_optimization_keys(optimization_defaults, "optimization_defaults")
        _validate_optimization_keys(sweep, "sweep")
        for idx, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                raise ValueError(f"tasks[{idx}] must be a mapping.")
            _validate_optimization_keys(task, f"tasks[{idx}]")

        return cls(
            name=name,
            mode=mode,
            optimization_defaults=_restore_special_values(optimization_defaults),
            sweep=_restore_special_values(sweep),
            tasks=[_restore_special_values(dict(task)) for task in tasks],
            execution=ExecutionConfig.from_dict(raw.get("execution")),
            metrics=MetricsRunConfig.from_dict(raw.get("metrics")),
            matching=MatchingRunConfig.from_dict(raw.get("matching")),
            choice_metrics=ChoiceMetricsRunConfig.from_dict(raw.get("choice_metrics")),
        )

    def generate_tasks(self) -> list[BenchmarkTask]:
        overrides = list(_sweep_overrides(self.sweep)) or [{}]
        explicit_tasks = self.tasks or [{}]
        tasks: list[BenchmarkTask] = []
        for sweep_values, task_values in product(overrides, explicit_tasks):
            config_data = dict(self.optimization_defaults)
            config_data.update(sweep_values)
            config_data.update(task_values)
            config = optimization_config_from_dict(config_data)
            config_dict = optimization_config_to_dict(config)
            config_hash = stable_hash(config_dict)
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


def optimization_config_from_dict(data: Mapping[str, Any]) -> OptimizationConfig:
    """Construct a :class:`OptimizationConfig` from a saved config snapshot."""

    restored = _restore_special_values(dict(data))
    field_names = _optimization_field_names()
    unknown = set(restored) - field_names - {"unit"}
    if unknown:
        raise ValueError(f"Unknown optimization config keys: {sorted(unknown)}")
    kwargs = {k: v for k, v in restored.items() if k in field_names}
    return OptimizationConfig(**kwargs)


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


def _matching_name_from_path(path: str | None, index: int) -> str:
    if path:
        return Path(str(path)).stem or f"config_{index}"
    return "default" if index == 0 else f"default_{index}"


def _validate_optimization_keys(data: Mapping[str, Any], section: str) -> None:
    unknown = set(data) - _optimization_field_names()
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")


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
