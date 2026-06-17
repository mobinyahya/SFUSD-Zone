"""Simulation sweep configuration for pipeline-native benchmarks."""

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

from Zone_Generation.pipeline.config import PipelineConfig


SEQUENCE_PIPELINE_FIELDS = {"levels", "solve_time_limits", "gap_limits", "years"}
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
    summary_csv: str = "summary.csv"
    stages_csv: str = "stages.csv"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MetricsRunConfig":
        return _dataclass_from_dict(cls, data or {})


@dataclass(frozen=True)
class BenchmarkTask:
    """One concrete pipeline run generated from a simulation sweep."""

    task_id: str
    config_hash: str
    config: dict[str, Any]
    output_dir: str
    capacity_slots: int

    def pipeline_config(self) -> PipelineConfig:
        return pipeline_config_from_dict(self.config)


@dataclass(frozen=True)
class SimulationSweep:
    """Top-level YAML-backed benchmark description."""

    name: str = "simulation_sweep"
    mode: str = "run"
    pipeline_defaults: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    metrics: MetricsRunConfig = field(default_factory=MetricsRunConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationSweep":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("Simulation sweep YAML must be a mapping.")
        allowed_top_level = {
            "name",
            "mode",
            "pipeline_defaults",
            "sweep",
            "tasks",
            "execution",
            "metrics",
        }
        unknown_top_level = set(raw) - allowed_top_level
        if unknown_top_level:
            raise ValueError(f"Unknown sweep YAML keys: {sorted(unknown_top_level)}")

        name = str(raw.get("name") or Path(path).stem)
        mode = str(raw.get("mode", "run"))
        if mode not in {"run", "metrics"}:
            raise ValueError("mode must be one of: run, metrics")

        pipeline_defaults = dict(raw.get("pipeline_defaults") or {})
        sweep = dict(raw.get("sweep") or {})
        tasks = list(raw.get("tasks") or [])
        _validate_pipeline_keys(pipeline_defaults, "pipeline_defaults")
        _validate_pipeline_keys(sweep, "sweep")
        for idx, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                raise ValueError(f"tasks[{idx}] must be a mapping.")
            _validate_pipeline_keys(task, f"tasks[{idx}]")

        return cls(
            name=name,
            mode=mode,
            pipeline_defaults=_restore_special_values(pipeline_defaults),
            sweep=_restore_special_values(sweep),
            tasks=[_restore_special_values(dict(task)) for task in tasks],
            execution=ExecutionConfig.from_dict(raw.get("execution")),
            metrics=MetricsRunConfig.from_dict(raw.get("metrics")),
        )

    def generate_tasks(self) -> list[BenchmarkTask]:
        overrides = list(_sweep_overrides(self.sweep)) or [{}]
        explicit_tasks = self.tasks or [{}]
        tasks: list[BenchmarkTask] = []
        for sweep_values, task_values in product(overrides, explicit_tasks):
            config_data = dict(self.pipeline_defaults)
            config_data.update(sweep_values)
            config_data.update(task_values)
            config = pipeline_config_from_dict(config_data)
            config_dict = pipeline_config_to_dict(config)
            config_hash = stable_hash(config_dict)
            output_dir = os.path.join(
                os.path.expanduser(self.execution.output_dir),
                format_output_path(config_dict, config_hash, self.execution.output_template),
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


def pipeline_config_from_dict(data: Mapping[str, Any]) -> PipelineConfig:
    """Construct a :class:`PipelineConfig` from a saved config snapshot."""

    restored = _restore_special_values(dict(data))
    field_names = _pipeline_field_names()
    unknown = set(restored) - field_names - {"unit"}
    if unknown:
        raise ValueError(f"Unknown pipeline config keys: {sorted(unknown)}")
    kwargs = {k: v for k, v in restored.items() if k in field_names}
    return PipelineConfig(**kwargs)


def pipeline_config_to_dict(config: PipelineConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, PipelineConfig):
        data = asdict(config)
    else:
        data = dict(config)
        data.pop("unit", None)
    return _restore_special_values(data)


def config_snapshot(config: PipelineConfig | Mapping[str, Any]) -> dict[str, Any]:
    pipeline = pipeline_config_to_dict(config)
    cfg = pipeline_config_from_dict(pipeline)
    snapshot = pipeline_config_to_dict(cfg)
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
    if key in SEQUENCE_PIPELINE_FIELDS:
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


def _validate_pipeline_keys(data: Mapping[str, Any], section: str) -> None:
    unknown = set(data) - _pipeline_field_names()
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")


def _pipeline_field_names() -> set[str]:
    return {f.name for f in fields(PipelineConfig)}


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
