"""Optimization configuration and factories.

``OptimizationConfig`` is the single typed description of a run: which levels, which
solver, which strategy, and all the data/optimization parameters. Its factory
methods build the concrete :class:`Dataset`, :class:`Solver` and
:class:`Strategy`, wiring the three layers together from a string config.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from loaders import DataScenario, anchor_data_config, load_scenario
from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec


_STRATEGIES = {"single", "recursive", "iterative_choice"}


def _legacy_data_config() -> dict[str, Any]:
    return {"scenario": "legacy", "overrides": {}}


@dataclass
class OptimizationConfig:
    # --- what to solve ------------------------------------------------- #
    centroids_type: str = "5-zone-AF"
    levels: list[str] = field(default_factory=lambda: ["BlockGroup_0"])
    solver: str = "cp_int"
    strategy: str = "single"

    # --- optimization parameters -------------------------------------- #
    frl_dev: float = 0.3
    racial_dev: float = 0.3
    overage: float = 0.8
    shortage: float = 0.2
    looseness: float = 1.0
    max_distance: float = float("inf")
    centroid_neighbor_radius: int = 0
    solve_time_limits: list[float] = field(default_factory=lambda: [60.0])
    carry_over_compute: bool = False
    gap_limits: list[float] = field(default_factory=lambda: [0.0])
    hints: str = "voronoi"
    save_solver_logs: bool = False
    save_solver_progress: bool = False
    secondary_objective: bool = False
    seed: int = 42
    workers: int = 8
    linearization_level: int | None = None
    cp_model_probing_level: int | None = None
    symmetry_level: int | None = None
    cp_sat_search_strategy: str | None = None
    recom_iterations: int = 1000
    short_bursts_length: int = 25
    short_bursts_method: str = "recom"
    # --- strategy-specific -------------------------------------------- #
    boundary_radius: int = 1
    boundary_prop: float = -1.0
    max_iterations: int = 5
    choice_model: str = "mnl"
    choice_model_method: str = "logsum"
    choice_utility_scale: float = 100.0
    choice_utility_hints: bool = False
    tolerance: float = 1e-6

    # --- data ingestion ----------------------------------------------- #
    data: dict[str, Any] = field(default_factory=_legacy_data_config)

    def __post_init__(self):
        if not isinstance(self.data, Mapping):
            raise ValueError("data must be a {scenario, overrides} map.")
        self.data = deepcopy(dict(self.data))
        self._data_scenario = load_scenario(self.data)

        # All levels in a run share one unit (the base graph is built per unit).
        specs = [LevelSpec.parse(level) for level in self.levels]
        units = {level.unit for level in specs}
        if len(units) != 1:
            raise ValueError(f"All levels must share one unit; got {sorted(units)}.")
        self.unit = units.pop()
        unsupported = [
            level.name
            for level in specs
            if not level.is_base
            and level.depth not in LEVEL_NODE_TARGETS.get(level.unit, {})
        ]
        if unsupported:
            raise ValueError(
                f"No predefined graph size for levels: {', '.join(unsupported)}."
            )
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of: {', '.join(sorted(_STRATEGIES))}."
            )
        if self.strategy == "recursive" and self.looseness < 1.0:
            raise ValueError("looseness must be >= 1.0 for recursive runs.")
        if self.solver == "cp_single_zone" and self.strategy != "single":
            raise ValueError("cp_single_zone requires strategy='single'.")
        if isinstance(self.boundary_prop, bool):
            raise ValueError("boundary_prop must be at most 1; negative disables it.")
        try:
            self.boundary_prop = float(self.boundary_prop)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "boundary_prop must be at most 1; negative disables it."
            ) from exc
        if math.isnan(self.boundary_prop) or self.boundary_prop > 1:
            raise ValueError("boundary_prop must be at most 1; negative disables it.")
        # Resolve the strict scenario-backed selectors eagerly.
        self.years
        self.grades
        self.student_population
        self.rounds
        self.special_programs
        self.program_population
        self.capacity_scenario
        self.include_k8
        self.include_citywide
        self.include_mission_bay
        self.frl_estimate
        self.outside_district_students
        if (
            isinstance(self.centroid_neighbor_radius, bool)
            or not isinstance(self.centroid_neighbor_radius, int)
            or self.centroid_neighbor_radius < 0
        ):
            raise ValueError("centroid_neighbor_radius must be a non-negative integer.")
        if self.hints not in {"voronoi", "none"}:
            raise ValueError("hints must be one of: voronoi, none.")
        if self.recom_iterations < 0 and not self.solve_time_limits:
            raise ValueError(
                "solve_time_limits must include a value when recom_iterations is negative."
            )
        if self.short_bursts_length <= 0:
            raise ValueError("short_bursts_length must be positive.")
        if self.short_bursts_method not in {"recom", "relaxed_recom"}:
            raise ValueError(
                "short_bursts_method must be one of: recom, relaxed_recom."
            )

    # ------------------------------------------------------------------ #
    # scenario-backed data settings
    # ------------------------------------------------------------------ #
    @property
    def data_scenario(self) -> DataScenario:
        """The immutable scenario loaded from the serializable ``data`` field."""
        return self._data_scenario

    @property
    def years(self) -> tuple[str, ...]:
        return tuple(self._data_scenario.filter("optimization", "years"))

    @property
    def grades(self) -> tuple[str, ...]:
        return tuple(self._data_scenario.filter("optimization", "grades"))

    @property
    def student_population(self) -> str:
        return self._data_scenario.filter("optimization", "student_population")

    @property
    def rounds(self) -> str | tuple[int, ...]:
        value = self._data_scenario.filter("optimization", "rounds")
        return value if value == "all" else tuple(value)

    @property
    def special_programs(self) -> str:
        return self._data_scenario.filter("optimization", "special_programs")

    @property
    def program_population(self) -> str:
        return self._data_scenario.filter("optimization", "program_population")

    @property
    def capacity_scenario(self) -> str:
        return self._data_scenario.filter("optimization", "capacity_scenario")

    @property
    def include_k8(self) -> bool:
        return self._data_scenario.filter("optimization", "include_k8")

    @property
    def include_citywide(self) -> bool:
        return self._data_scenario.filter("optimization", "include_citywide")

    @property
    def include_mission_bay(self) -> bool:
        return self._data_scenario.filter("optimization", "include_mission_bay")

    @property
    def outside_district_students(self) -> str:
        return self._data_scenario.filter(
            "optimization", "outside_district_students"
        )

    @property
    def frl_estimate(self) -> str | None:
        return self._data_scenario.filter("optimization", "frl_estimate")

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "OptimizationConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Optimization config YAML must contain a map.")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}.")
        if "data" in raw:
            raw["data"] = anchor_data_config(raw["data"], config_path.parent)
        return cls(**raw)

    # ------------------------------------------------------------------ #
    # factories
    # ------------------------------------------------------------------ #
    def make_dataset(self):
        from optimization.data.dataset import Dataset

        return Dataset(self)

    def make_solver(self, output_dir: str | None = None):
        from optimization.solvers import get_solver

        options = {
            "solve_time_limit": self.solve_time_limits[0],
            "relative_gap_limit": self.gap_limits[0],
            "seed": self.seed,
            "workers": self.workers,
            "linearization_level": self.linearization_level,
            "cp_model_probing_level": self.cp_model_probing_level,
            "symmetry_level": self.symmetry_level,
            "cp_sat_search_strategy": self.cp_sat_search_strategy,
            "hints": self.hints,
            "save_solver_logs": self.save_solver_logs,
            "save_solver_progress": self.save_solver_progress,
            "secondary_objective": self.secondary_objective,
            "centroid_neighbor_radius": self.centroid_neighbor_radius,
            "recom_iterations": self.recom_iterations,
            "short_bursts_length": self.short_bursts_length,
            "short_bursts_method": self.short_bursts_method,
        }
        if output_dir is not None:
            options["output_dir"] = output_dir
        return get_solver(self.solver, **options)

    def make_strategy(self):
        from optimization.strategies import get_strategy

        return get_strategy(
            self.strategy,
            levels=self.levels,
            solve_time_limits=self.solve_time_limits,
            carry_over_compute=self.carry_over_compute,
            gap_limits=self.gap_limits,
            hints=self.hints,
            looseness=self.looseness,
            boundary_radius=self.boundary_radius,
            boundary_prop=self.boundary_prop,
            max_iterations=self.max_iterations,
            choice_model=self.choice_model,
            choice_model_method=self.choice_model_method,
            choice_utility_scale=self.choice_utility_scale,
            choice_utility_hints=self.choice_utility_hints,
            tolerance=self.tolerance,
        )
