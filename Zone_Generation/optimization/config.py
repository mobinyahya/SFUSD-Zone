"""Optimization configuration and factories.

``OptimizationConfig`` is the single typed description of a run: which levels, which
solver, which strategy, and all the data/optimization parameters. Its factory
methods build the concrete :class:`Dataset`, :class:`Solver` and
:class:`Strategy`, wiring the three layers together from a string config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

import yaml

from Zone_Generation.Config.Constants import get_dropbox_path
from Zone_Generation.optimization.levels import LevelSpec


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
    max_distance: float = float("inf")
    solve_time_limits: list[float] = field(default_factory=lambda: [60.0])
    gap_limits: list[float] = field(default_factory=lambda: [0.0])
    use_hints: bool = True
    seed: int = 42
    workers: int = 8

    # --- strategy-specific -------------------------------------------- #
    boundary_radius: int = 1
    max_iterations: int = 5
    choice_model: str = "distance"
    choice_model_method: str = "logsum"
    choice_utility_scale: float = 100.0
    tolerance: float = 1e-6

    # --- data ingestion ----------------------------------------------- #
    years: list[int] = field(
        default_factory=lambda: [14, 15, 16, 17, 18, 21, 22]
    )
    population_type: str = "GE"
    drop_optout: bool = True
    capacity_scenario: str = "A"
    new_schools: bool = True
    include_k8: bool = False
    level_to_split: dict[int, int] = field(
        default_factory=lambda: {1: 2, 2: 1}
    )
    graphs_dir: str = ""

    def __post_init__(self):
        # All levels in a run share one unit (the base graph is built per unit).
        units = {LevelSpec.parse(l).unit for l in self.levels}
        if len(units) != 1:
            raise ValueError(
                f"All levels must share one unit; got {sorted(units)}."
            )
        self.unit = units.pop()
        if not self.graphs_dir:
            self.graphs_dir = os.path.join(
                get_dropbox_path(False),
                "Optimization",
                "Zones",
                "Graphs",
                "optimization",
            )
        # Levels in float keys can arrive from YAML as strings.
        self.level_to_split = {int(k): int(v) for k, v in self.level_to_split.items()}

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "OptimizationConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}.")
        return cls(**raw)

    # ------------------------------------------------------------------ #
    # factories
    # ------------------------------------------------------------------ #
    def make_dataset(self):
        from Zone_Generation.optimization.data.dataset import Dataset

        return Dataset(self)

    def make_solver(self):
        from Zone_Generation.optimization.solvers import get_solver

        return get_solver(
            self.solver,
            solve_time_limit=self.solve_time_limits[0],
            relative_gap_limit=self.gap_limits[0],
            seed=self.seed,
            workers=self.workers,
        )

    def make_strategy(self):
        from Zone_Generation.optimization.strategies import get_strategy

        return get_strategy(
            self.strategy,
            levels=self.levels,
            solve_time_limits=self.solve_time_limits,
            gap_limits=self.gap_limits,
            use_hints=self.use_hints,
            boundary_radius=self.boundary_radius,
            max_iterations=self.max_iterations,
            choice_model=self.choice_model,
            choice_model_method=self.choice_model_method,
            choice_utility_scale=self.choice_utility_scale,
            tolerance=self.tolerance,
        )
