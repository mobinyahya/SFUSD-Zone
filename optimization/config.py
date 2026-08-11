"""Optimization configuration and factories.

``OptimizationConfig`` is the single typed description of a run: which levels, which
solver, which strategy, and all the data/optimization parameters. Its factory
methods build the concrete :class:`Dataset`, :class:`Solver` and
:class:`Strategy`, wiring the three layers together from a string config.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields

import yaml

from Config.Constants import get_sfusd_path
from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec


def migrate_legacy_zoned_recom_seed_runs(raw: dict) -> None:
    """Collapse strategy-specific saved ReCom counts into the shared setting."""
    legacy = {
        "zoned_column_generation": raw.pop("zoned_cg_recom_seed_runs", None),
        "zoned_benders": raw.pop("zoned_benders_recom_seed_runs", None),
    }
    if "zoned_recom_seed_runs" in raw:
        return
    strategy = raw.get("strategy")
    if strategy in legacy and legacy[strategy] is not None:
        raw["zoned_recom_seed_runs"] = legacy[strategy]
        return
    values = {value for value in legacy.values() if value is not None}
    if len(values) == 1:
        raw["zoned_recom_seed_runs"] = values.pop()


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
    school_solve_time_limit: float = 60.0
    max_iterations: int = 5
    choice_model: str = "mnl"
    choice_model_method: str = "logsum"
    choice_utility_scale: float = 100.0
    choice_utility_hints: bool = False
    tolerance: float = 1e-6
    cutoff_assignment_config: str = "assignment/configs/kumar.config.yaml"
    cutoff_ctip_path: str = (
        "/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/ETB_2024.npy"
    )
    cutoff_lottery_scale: int = 20
    cutoff_gumbel_scale: float = 1.0
    cutoff_preference_seed: int = 2023
    cutoff_method: str = "decomposition"
    decomposition_generate_assigned_pairs: bool = True
    remove_city_wide: bool = False
    welfare_utility_scale: int = 1_000_000
    welfare_initial_assignment_path: str = ""
    welfare_prefix_depth: int = 10
    welfare_decomposition_round_time_limit: float = 180.0
    welfare_decomposition_theta_enabled: bool = True
    welfare_assignment_relaxation_enabled: bool = True
    welfare_submodular_access_start_enabled: bool = False
    welfare_adjacent_zone_subset_improvement_enabled: bool = False
    welfare_branch_price_enabled: bool = False
    welfare_recom_time_limit: float = 600.0
    welfare_branch_price_time_limit: float = 45.0
    welfare_method: str = "decomposition"
    decomposition_pressure_starts_enabled: bool = False
    decomposition_local_moves_enabled: bool = False
    zoned_recom_seed_runs: int = 0
    zoned_cg_wall_time_limit: float = 2700.0
    zoned_cg_max_rounds: int = 100
    zoned_cg_pricing_time_limit: float = 300.0
    zoned_cg_pricing_node_limit: int = 10_000
    zoned_cg_columns_per_label: int = 10
    zoned_cg_reduced_cost_tolerance: float = 1e-7
    zoned_cg_menu_tolerance: float = 1e-9
    zoned_cg_master_feasibility_tolerance: float = 1e-8
    zoned_cg_optimality_tolerance: float = 1e-6
    zoned_cg_mip_time_limit: float = 300.0
    zoned_cg_seed_paths: list[str] = field(default_factory=list)
    zoned_cg_local_move_rounds: int = 0
    zoned_cg_save_mechanism: bool = True
    zoned_cg_evaluate_stable_diagnostics: bool = True
    zoned_benders_wall_time_limit: float = 2700.0
    zoned_benders_max_rounds: int = 100
    zoned_benders_master_time_limit: float = 180.0
    zoned_benders_menu_tolerance: float = 1e-9
    zoned_benders_menu_max_rounds: int = 1000
    zoned_benders_master_feasibility_tolerance: float = 1e-8
    zoned_benders_optimality_tolerance: float = 1e-6
    zoned_benders_seed_paths: list[str] = field(default_factory=list)
    zoned_benders_local_move_rounds: int = 0
    zoned_benders_save_mechanism: bool = True
    zoned_benders_evaluate_stable_diagnostics: bool = True

    # --- data ingestion ----------------------------------------------- #
    years: list[int] = field(default_factory=lambda: [14, 15, 16, 17, 18, 21, 22])
    population_type: str = "GE"
    drop_optout: bool = True
    capacity_scenario: str = "A"
    new_schools: bool = True
    include_k8: bool = False
    graphs_dir: str = ""

    def __post_init__(self):
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
        if not self.graphs_dir:
            self.graphs_dir = os.path.join(
                get_sfusd_path(False),
                "Zones",
                "Optimization",
                "Graphs",
            )
        if self.strategy == "recursive" and self.looseness < 1.0:
            raise ValueError("looseness must be >= 1.0 for recursive runs.")
        if self.solver == "cp_single_zone" and self.strategy != "single":
            raise ValueError("cp_single_zone requires strategy='single'.")
        if self.strategy in {"cutoffs", "welfare", "approximate_welfare"}:
            if self.solver != "cp_bool":
                raise ValueError(f"{self.strategy} requires solver='cp_bool'.")
            if self.years != [23]:
                raise ValueError(f"{self.strategy} currently requires years: [23].")
            if self.population_type != "All":
                raise ValueError(f"{self.strategy} requires population_type: 'All'.")
        if (
            self.strategy in {"welfare", "approximate_welfare"}
            and not self.remove_city_wide
        ):
            raise ValueError(
                f"{self.strategy} currently requires remove_city_wide: true."
            )
        if self.strategy == "zoned_column_generation":
            self._validate_zoned_column_generation()
        if self.strategy == "zoned_benders":
            self._validate_zoned_benders()
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
        if (
            isinstance(self.cutoff_lottery_scale, bool)
            or not isinstance(self.cutoff_lottery_scale, int)
            or self.cutoff_lottery_scale <= 0
        ):
            raise ValueError("cutoff_lottery_scale must be a positive integer.")
        if (
            not math.isfinite(float(self.cutoff_gumbel_scale))
            or self.cutoff_gumbel_scale < 0
        ):
            raise ValueError("cutoff_gumbel_scale must be finite and non-negative.")
        if isinstance(self.cutoff_preference_seed, bool) or not isinstance(
            self.cutoff_preference_seed, int
        ):
            raise ValueError("cutoff_preference_seed must be an integer.")
        if self.cutoff_method not in {
            "decomposition",
            "pair_generation",
            "conditional_demand",
        }:
            raise ValueError(
                "cutoff_method must be one of: decomposition, pair_generation, "
                "conditional_demand."
            )
        if not isinstance(self.remove_city_wide, bool):
            raise ValueError("remove_city_wide must be a boolean.")
        if (
            isinstance(self.welfare_utility_scale, bool)
            or not isinstance(self.welfare_utility_scale, int)
            or self.welfare_utility_scale <= 0
        ):
            raise ValueError("welfare_utility_scale must be a positive integer.")
        if (
            isinstance(self.welfare_prefix_depth, bool)
            or not isinstance(self.welfare_prefix_depth, int)
            or self.welfare_prefix_depth <= 0
        ):
            raise ValueError("welfare_prefix_depth must be a positive integer.")
        if isinstance(self.welfare_decomposition_round_time_limit, bool):
            raise ValueError(
                "welfare_decomposition_round_time_limit must be positive and finite."
            )
        try:
            self.welfare_decomposition_round_time_limit = float(
                self.welfare_decomposition_round_time_limit
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "welfare_decomposition_round_time_limit must be positive and finite."
            ) from exc
        if (
            not math.isfinite(self.welfare_decomposition_round_time_limit)
            or self.welfare_decomposition_round_time_limit <= 0
        ):
            raise ValueError(
                "welfare_decomposition_round_time_limit must be positive and finite."
            )
        for name in (
            "decomposition_generate_assigned_pairs",
            "welfare_decomposition_theta_enabled",
            "welfare_assignment_relaxation_enabled",
            "welfare_submodular_access_start_enabled",
            "welfare_adjacent_zone_subset_improvement_enabled",
            "welfare_branch_price_enabled",
            "decomposition_pressure_starts_enabled",
            "decomposition_local_moves_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if (
            self.strategy == "welfare"
            and self.welfare_method == "decomposition"
            and not self.welfare_decomposition_theta_enabled
            and not self.decomposition_generate_assigned_pairs
        ):
            raise ValueError(
                "welfare_decomposition_theta_enabled=false requires "
                "decomposition_generate_assigned_pairs=true."
            )
        if (
            isinstance(self.zoned_recom_seed_runs, bool)
            or not isinstance(self.zoned_recom_seed_runs, int)
            or self.zoned_recom_seed_runs < 0
        ):
            raise ValueError("zoned_recom_seed_runs must be a non-negative integer.")
        recom_time_limit_error = (
            "welfare_recom_time_limit must be non-negative and finite, and "
            "positive when zoned_recom_seed_runs > 0."
        )
        if isinstance(self.welfare_recom_time_limit, bool):
            raise ValueError(recom_time_limit_error)
        try:
            self.welfare_recom_time_limit = float(self.welfare_recom_time_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(recom_time_limit_error) from exc
        if (
            not math.isfinite(self.welfare_recom_time_limit)
            or self.welfare_recom_time_limit < 0
            or (self.zoned_recom_seed_runs > 0 and self.welfare_recom_time_limit == 0)
        ):
            raise ValueError(recom_time_limit_error)
        if isinstance(self.welfare_branch_price_time_limit, bool):
            raise ValueError(
                "welfare_branch_price_time_limit must be positive and finite."
            )
        try:
            self.welfare_branch_price_time_limit = float(
                self.welfare_branch_price_time_limit
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "welfare_branch_price_time_limit must be positive and finite."
            ) from exc
        if (
            not math.isfinite(self.welfare_branch_price_time_limit)
            or self.welfare_branch_price_time_limit <= 0
        ):
            raise ValueError(
                "welfare_branch_price_time_limit must be positive and finite."
            )
        if self.welfare_branch_price_enabled and self.zoned_recom_seed_runs == 0:
            raise ValueError(
                "welfare_branch_price_enabled requires zoned_recom_seed_runs > 0."
            )
        if self.welfare_method not in {
            "budget",
            "decomposition",
            "direct",
            "lbbd",
        }:
            raise ValueError(
                "welfare_method must be one of: budget, decomposition, direct, lbbd."
            )
        if (
            not math.isfinite(float(self.school_solve_time_limit))
            or self.school_solve_time_limit <= 0
        ):
            raise ValueError("school_solve_time_limit must be positive and finite.")
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

    def _validate_zoned_column_generation(self) -> None:
        if self.years != [23]:
            raise ValueError("zoned_column_generation currently requires years: [23].")
        if self.population_type != "All":
            raise ValueError("zoned_column_generation requires population_type: 'All'.")
        if not self.remove_city_wide:
            raise ValueError("zoned_column_generation requires remove_city_wide: true.")
        accepted_solvers = {
            "cp_int",
            "cp_bool",
            "mip",
            "recom",
            "relaxed_recom",
            "short_bursts",
        }
        if self.solver not in accepted_solvers:
            raise ValueError(
                "zoned_column_generation seed solver must be one of: "
                + ", ".join(sorted(accepted_solvers))
                + "."
            )
        if (
            not math.isfinite(float(self.cutoff_gumbel_scale))
            or self.cutoff_gumbel_scale <= 0
        ):
            raise ValueError(
                "zoned_column_generation requires a positive finite cutoff_gumbel_scale."
            )
        positive_finite = {
            "zoned_cg_wall_time_limit": self.zoned_cg_wall_time_limit,
            "zoned_cg_pricing_time_limit": self.zoned_cg_pricing_time_limit,
            "zoned_cg_reduced_cost_tolerance": self.zoned_cg_reduced_cost_tolerance,
            "zoned_cg_menu_tolerance": self.zoned_cg_menu_tolerance,
            "zoned_cg_master_feasibility_tolerance": self.zoned_cg_master_feasibility_tolerance,
            "zoned_cg_optimality_tolerance": self.zoned_cg_optimality_tolerance,
            "zoned_cg_mip_time_limit": self.zoned_cg_mip_time_limit,
        }
        for name, value in positive_finite.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be positive and finite.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be positive and finite.") from exc
            if not math.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        nonnegative_counts = {
            "zoned_cg_max_rounds": self.zoned_cg_max_rounds,
            "zoned_cg_pricing_node_limit": self.zoned_cg_pricing_node_limit,
            "zoned_cg_local_move_rounds": self.zoned_cg_local_move_rounds,
        }
        for name, value in nonnegative_counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            isinstance(self.zoned_cg_columns_per_label, bool)
            or not isinstance(self.zoned_cg_columns_per_label, int)
            or self.zoned_cg_columns_per_label <= 0
        ):
            raise ValueError("zoned_cg_columns_per_label must be a positive integer.")
        if not isinstance(self.zoned_cg_seed_paths, list) or any(
            not isinstance(path, str) or not path.strip()
            for path in self.zoned_cg_seed_paths
        ):
            raise ValueError("zoned_cg_seed_paths must be a list of nonempty paths.")
        for name in (
            "zoned_cg_save_mechanism",
            "zoned_cg_evaluate_stable_diagnostics",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    def _validate_zoned_benders(self) -> None:
        if self.years != [23]:
            raise ValueError("zoned_benders currently requires years: [23].")
        if self.population_type != "All":
            raise ValueError("zoned_benders requires population_type: 'All'.")
        if not self.remove_city_wide:
            raise ValueError("zoned_benders requires remove_city_wide: true.")
        accepted_solvers = {
            "cp_int",
            "cp_bool",
            "mip",
            "recom",
            "relaxed_recom",
            "short_bursts",
        }
        if self.solver not in accepted_solvers:
            raise ValueError(
                "zoned_benders seed solver must be one of: "
                + ", ".join(sorted(accepted_solvers))
                + "."
            )
        if (
            not math.isfinite(float(self.cutoff_gumbel_scale))
            or self.cutoff_gumbel_scale <= 0
        ):
            raise ValueError(
                "zoned_benders requires a positive finite cutoff_gumbel_scale."
            )
        positive_finite = {
            "zoned_benders_wall_time_limit": self.zoned_benders_wall_time_limit,
            "zoned_benders_master_time_limit": self.zoned_benders_master_time_limit,
            "zoned_benders_menu_tolerance": self.zoned_benders_menu_tolerance,
            "zoned_benders_master_feasibility_tolerance": (
                self.zoned_benders_master_feasibility_tolerance
            ),
            "zoned_benders_optimality_tolerance": (
                self.zoned_benders_optimality_tolerance
            ),
        }
        for name, value in positive_finite.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be positive and finite.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be positive and finite.") from exc
            if not math.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        nonnegative_counts = {
            "zoned_benders_max_rounds": self.zoned_benders_max_rounds,
            "zoned_benders_local_move_rounds": self.zoned_benders_local_move_rounds,
        }
        for name, value in nonnegative_counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            isinstance(self.zoned_benders_menu_max_rounds, bool)
            or not isinstance(self.zoned_benders_menu_max_rounds, int)
            or self.zoned_benders_menu_max_rounds <= 0
        ):
            raise ValueError(
                "zoned_benders_menu_max_rounds must be a positive integer."
            )
        if not isinstance(self.zoned_benders_seed_paths, list) or any(
            not isinstance(path, str) or not path.strip()
            for path in self.zoned_benders_seed_paths
        ):
            raise ValueError(
                "zoned_benders_seed_paths must be a list of nonempty paths."
            )
        for name in (
            "zoned_benders_save_mechanism",
            "zoned_benders_evaluate_stable_diagnostics",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "OptimizationConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        # Persisted pre-KaHIP configs included this obsolete partition setting.
        raw.pop("level_to_split", None)
        migrate_legacy_zoned_recom_seed_runs(raw)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}.")
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
            school_solve_time_limit=self.school_solve_time_limit,
            max_iterations=self.max_iterations,
            choice_model=self.choice_model,
            choice_model_method=self.choice_model_method,
            choice_utility_scale=self.choice_utility_scale,
            choice_utility_hints=self.choice_utility_hints,
            tolerance=self.tolerance,
            cutoff_assignment_config=self.cutoff_assignment_config,
            cutoff_ctip_path=self.cutoff_ctip_path,
            cutoff_lottery_scale=self.cutoff_lottery_scale,
            cutoff_gumbel_scale=self.cutoff_gumbel_scale,
            cutoff_preference_seed=self.cutoff_preference_seed,
            cutoff_method=self.cutoff_method,
            decomposition_generate_assigned_pairs=(
                self.decomposition_generate_assigned_pairs
            ),
            remove_city_wide=self.remove_city_wide,
            welfare_utility_scale=self.welfare_utility_scale,
            welfare_initial_assignment_path=self.welfare_initial_assignment_path,
            welfare_prefix_depth=self.welfare_prefix_depth,
            welfare_decomposition_round_time_limit=(
                self.welfare_decomposition_round_time_limit
            ),
            welfare_decomposition_theta_enabled=(
                self.welfare_decomposition_theta_enabled
            ),
            welfare_assignment_relaxation_enabled=(
                self.welfare_assignment_relaxation_enabled
            ),
            welfare_submodular_access_start_enabled=(
                self.welfare_submodular_access_start_enabled
            ),
            welfare_adjacent_zone_subset_improvement_enabled=(
                self.welfare_adjacent_zone_subset_improvement_enabled
            ),
            welfare_branch_price_enabled=self.welfare_branch_price_enabled,
            welfare_recom_time_limit=self.welfare_recom_time_limit,
            welfare_branch_price_time_limit=self.welfare_branch_price_time_limit,
            welfare_method=self.welfare_method,
            decomposition_pressure_starts_enabled=(
                self.decomposition_pressure_starts_enabled
            ),
            decomposition_local_moves_enabled=self.decomposition_local_moves_enabled,
            zoned_recom_seed_runs=self.zoned_recom_seed_runs,
            centroid_neighbor_radius=self.centroid_neighbor_radius,
            seed=self.seed,
            workers=self.workers,
            recom_iterations=self.recom_iterations,
            short_bursts_length=self.short_bursts_length,
            short_bursts_method=self.short_bursts_method,
            zoned_cg_wall_time_limit=self.zoned_cg_wall_time_limit,
            zoned_cg_max_rounds=self.zoned_cg_max_rounds,
            zoned_cg_pricing_time_limit=self.zoned_cg_pricing_time_limit,
            zoned_cg_pricing_node_limit=self.zoned_cg_pricing_node_limit,
            zoned_cg_columns_per_label=self.zoned_cg_columns_per_label,
            zoned_cg_reduced_cost_tolerance=self.zoned_cg_reduced_cost_tolerance,
            zoned_cg_menu_tolerance=self.zoned_cg_menu_tolerance,
            zoned_cg_master_feasibility_tolerance=self.zoned_cg_master_feasibility_tolerance,
            zoned_cg_optimality_tolerance=self.zoned_cg_optimality_tolerance,
            zoned_cg_mip_time_limit=self.zoned_cg_mip_time_limit,
            zoned_cg_seed_paths=self.zoned_cg_seed_paths,
            zoned_cg_local_move_rounds=self.zoned_cg_local_move_rounds,
            zoned_cg_save_mechanism=self.zoned_cg_save_mechanism,
            zoned_cg_evaluate_stable_diagnostics=self.zoned_cg_evaluate_stable_diagnostics,
            zoned_benders_wall_time_limit=self.zoned_benders_wall_time_limit,
            zoned_benders_max_rounds=self.zoned_benders_max_rounds,
            zoned_benders_master_time_limit=self.zoned_benders_master_time_limit,
            zoned_benders_menu_tolerance=self.zoned_benders_menu_tolerance,
            zoned_benders_menu_max_rounds=self.zoned_benders_menu_max_rounds,
            zoned_benders_master_feasibility_tolerance=self.zoned_benders_master_feasibility_tolerance,
            zoned_benders_optimality_tolerance=self.zoned_benders_optimality_tolerance,
            zoned_benders_seed_paths=self.zoned_benders_seed_paths,
            zoned_benders_local_move_rounds=self.zoned_benders_local_move_rounds,
            zoned_benders_save_mechanism=self.zoned_benders_save_mechanism,
            zoned_benders_evaluate_stable_diagnostics=self.zoned_benders_evaluate_stable_diagnostics,
        )
