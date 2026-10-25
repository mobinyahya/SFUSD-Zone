"""Optimization-native metric data structures.

Metrics now operate on :class:`ZoneSolution` objects rather than legacy
``zone_dict``/graph pairs. A run can contain one solution, recursive stages, or
iterative attempts; the context exposes the selected final solution plus all
stages for run-level analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, is_dataclass, asdict
from functools import cached_property
from typing import Any, Callable, Mapping, Sequence

from Config.Constants import SUPPLEMENTARY_ETHNICITIES
from loaders import DataScenario, load_scenario
from metrics.spatial import SpatialMetrics, compute_spatial_metrics
from optimization.solution import ZoneSolution

MetricValue = float | int | str | bool | None
MetricFn = Callable[["MetricsContext"], "MetricOutput"]


@dataclass
class MetricOutput:
    """One metric module's contribution."""

    metrics: dict[str, MetricValue] = field(default_factory=dict)
    zone_data: dict[int, dict[str, Any]] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsResult:
    """Complete metric output for a optimization run."""

    metrics: dict[str, MetricValue] = field(default_factory=dict)
    zone_data: dict[int, dict[str, Any]] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)

    def update(self, output: MetricOutput) -> None:
        self.metrics.update(output.metrics)
        _deep_update(self.run, output.run)
        for zone_id, values in output.zone_data.items():
            self.zone_data.setdefault(int(zone_id), {}).update(values)

    def to_full_dict(self) -> dict[str, Any]:
        """JSON-serializable metrics, per-zone data, and run metadata."""
        return _jsonable(
            {
                "metrics": self.metrics,
                "zone_data": {str(k): v for k, v in self.zone_data.items()},
                "run": self.run,
            }
        )


class MetricsContext:
    """Shared derived views for metric modules."""

    def __init__(
        self,
        solutions: ZoneSolution | list[ZoneSolution],
        config: Mapping[str, Any] | Any | None = None,
        compute_stage_metrics: bool = False,
    ):
        if isinstance(solutions, ZoneSolution):
            self.stages = [solutions]
        else:
            self.stages = list(solutions)
        if not self.stages:
            raise ValueError("MetricsContext requires at least one ZoneSolution.")
        if not all(isinstance(stage, ZoneSolution) for stage in self.stages):
            raise TypeError(
                "MetricsCalculator is optimization-only; pass a ZoneSolution or a "
                "sequence of ZoneSolution objects."
            )

        self.config = _config_dict(config)
        # An OptimizationConfig already holds its loaded scenario; reuse it.
        self._data_scenario = getattr(config, "data_scenario", None)
        self.solution = self._select_final_solution()
        self.compute_stage_metrics = bool(compute_stage_metrics)
        self._spatial: dict[int, SpatialMetrics] = {}

    @property
    def G(self):
        return self.solution.problem.G

    @property
    def assignment(self) -> dict[int, int]:
        return self.solution.assignment

    @property
    def problem(self):
        return self.solution.problem

    @property
    def level_name(self) -> str:
        return self.solution.level.name

    def spatial_metrics(self, solution: ZoneSolution) -> SpatialMetrics:
        """Block_0 spatial metrics for one stage, computed once per stage."""
        key = id(solution)
        if key not in self._spatial:
            self._spatial[key] = compute_spatial_metrics(solution, self.config)
        return self._spatial[key]

    @cached_property
    def data_scenario(self) -> DataScenario | None:
        """The run's data scenario, or ``None`` when the config names no data."""
        if self._data_scenario is not None:
            return self._data_scenario
        data = self.config.get("data")
        return None if data is None else load_scenario(data)

    @cached_property
    def school_programs(self) -> dict[int, list[str]]:
        """Sorted program types offered by each school."""
        from metrics.programs import school_programs

        return school_programs(self)

    @cached_property
    def supplementary_ethnicity_counts(self) -> dict[int, dict[str, float]] | None:
        """Per-node counts of the groups graph nodes omit, or ``None`` without data.

        Read from the run's student area table at the solution's unit, so they
        carry the node counts' population filters and weights. A node sums the
        areas in its ``area_id`` or ``block_ids``.
        """
        if self.data_scenario is None:
            return None
        from optimization.data.loaders import IngestConfig, load_area_table

        unit = self.solution.level.unit
        area = load_area_table(IngestConfig(unit=unit, data=self.data_scenario))
        by_area = area.set_index(unit)[SUPPLEMENTARY_ETHNICITIES].to_dict("index")
        counts: dict[int, dict[str, float]] = {}
        for node, attrs in self.G.nodes(data=True):
            area_ids = (
                [attrs["area_id"]] if "area_id" in attrs else attrs.get("block_ids", [])
            )
            totals = dict.fromkeys(SUPPLEMENTARY_ETHNICITIES, 0.0)
            for area_id in area_ids:
                for ethnicity, value in by_area.get(int(area_id), {}).items():
                    totals[ethnicity] += float(value)
            counts[int(node)] = totals
        return counts

    @cached_property
    def zone_nodes(self) -> dict[int, list[int]]:
        zones: dict[int, list[int]] = {}
        for node, zone_id in self.assignment.items():
            zones.setdefault(int(zone_id), []).append(int(node))
        return zones

    @cached_property
    def zone_schools(self) -> dict[int, list[int]]:
        schools: dict[int, list[int]] = {z: [] for z in self.zone_nodes}
        for zone_id, nodes in self.zone_nodes.items():
            seen = set()
            for node in nodes:
                for sid in self.G.nodes[node].get("school_ids", []):
                    if sid not in seen:
                        schools[zone_id].append(sid)
                        seen.add(sid)
        return schools

    @cached_property
    def school_to_node(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for node, attrs in self.G.nodes(data=True):
            for sid in attrs.get("school_ids", []):
                out.setdefault(sid, node)
        return out

    @cached_property
    def area_assignment(self) -> dict[int, int]:
        return self.solution.area_assignment()

    @cached_property
    def stage_names(self) -> list[str]:
        return stage_names(
            [stage.level.name for stage in self.stages],
            iterative=self._is_iterative_run(),
        )

    @cached_property
    def final_stage_index(self) -> int:
        for idx, stage in enumerate(self.stages):
            if stage is self.solution:
                return idx
        for idx, stage in enumerate(self.stages):
            if stage == self.solution:
                return idx
        return len(self.stages) - 1

    @cached_property
    def final_stage_name(self) -> str:
        return self.stage_names[self.final_stage_index]

    def _select_final_solution(self) -> ZoneSolution:
        if not self._is_iterative_run():
            return self.stages[-1]

        choice_candidates = [
            sol
            for sol in self.stages
            if sol.feasible
            and sol.assignment
            and sol.metadata.get("choice_utility") is not None
        ]
        if choice_candidates:
            return max(
                choice_candidates, key=lambda sol: sol.metadata["choice_utility"]
            )
        return self.stages[-1]

    def _is_iterative_run(self) -> bool:
        strategy = str(self.config.get("strategy", ""))
        if strategy:
            return is_iterative_strategy(strategy)
        levels = [stage.level.name for stage in self.stages]
        return len(levels) > 1 and len(set(levels)) == 1


def is_iterative_strategy(strategy: str) -> bool:
    """Whether a strategy's stages are attempts at one level, not refinements."""
    strategy = str(strategy).lower()
    return "iterative" in strategy or strategy == "saa"


def stage_names(levels: Sequence[str], *, iterative: bool) -> list[str]:
    """Name a run's stages: the folder under ``stages/`` and the metric suffix."""
    prefix = "iteration" if iterative else "stage"
    return [f"{prefix}_{index:02d}_{level}" for index, level in enumerate(levels)]


def _config_dict(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if is_dataclass(config):
        out = asdict(config)
        if hasattr(config, "unit"):
            out["unit"] = getattr(config, "unit")
        return out
    return dict(vars(config))


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
