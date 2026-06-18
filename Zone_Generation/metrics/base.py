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
from typing import Any, Callable, Mapping

from Zone_Generation.optimization.solution import ZoneSolution

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

    def to_flat_dict(self) -> dict[str, MetricValue]:
        """Flat metrics only, for CSV-like consumers."""
        return dict(self.metrics)

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
        final_solution: ZoneSolution | None = None,
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
        self.solution = final_solution or self._select_final_solution()

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
        prefix = "iteration" if self._is_iterative_run() else "stage"
        return [f"{prefix}_{idx:02d}_{sol.level.name}" for idx, sol in enumerate(self.stages)]

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
        choice_candidates = [
            sol
            for sol in self.stages
            if sol.assignment and sol.metadata.get("choice_utility") is not None
        ]
        if choice_candidates:
            return max(choice_candidates, key=lambda sol: sol.metadata["choice_utility"])
        for sol in reversed(self.stages):
            if sol.assignment:
                return sol
        return self.stages[-1]

    def _is_iterative_run(self) -> bool:
        strategy = str(self.config.get("strategy", "")).lower()
        if "iterative" in strategy:
            return True
        if strategy:
            return False
        levels = [stage.level.name for stage in self.stages]
        return len(levels) > 1 and len(set(levels)) == 1


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
