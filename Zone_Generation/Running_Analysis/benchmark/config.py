"""
Benchmark configuration module.

Provides unified configuration classes for both recursive and non-recursive
zoning optimization runs, with support for batch scenario generation.
"""
from dataclasses import dataclass, field, fields, asdict
from itertools import product
from typing import Iterator
import yaml


@dataclass
class BenchmarkConfig:
    """
    Unified configuration for zoning optimization benchmarks.
    
    Supports both recursive (multi-level) and non-recursive (single-level) modes.
    Use `generate_scenarios()` to create multiple configs from parameter sweeps.
    """
    # Core settings
    centroids_type: str
    frl_dev: float
    racial_dev: float
    random_seed: int = 42
    
    # Recursive settings - if recursive_levels is set, runs in recursive mode
    recursive_levels: list[str] | None = None
    solve_time_limits: list[int] | None = None
    relative_gap_limits: list[float] | None = None
    
    # Single-level settings (used when not recursive)
    level: str = "BlockGroup_0"
    solve_time_limit: int = 600
    relative_gap_limit: float = 0.0
    
    # Common settings
    overage: float = 0.8
    shortage: float = 0.2
    optimizer: str = "cp_int"
    use_hints: bool = True
    is_local: bool = False
    
    # Output settings
    log_folder: str | None = None
    
    # Additional settings from config.yaml
    years: list[int] = field(default_factory=lambda: [14, 15, 16, 17, 18, 21, 22])
    population_type: str = "GE"
    drop_optout: bool = True
    capacity_scenario: str = "A"
    new_schools: bool = True
    include_k8: bool = False
    all_cap_shortage: float = float('inf')
    max_distance: float = 5.0
    population_dev: float = float('inf')
    
    @property
    def is_recursive(self) -> bool:
        """Check if this config uses recursive (multi-level) zoning."""
        return self.recursive_levels is not None and len(self.recursive_levels) > 1
    
    @property
    def num_zones(self) -> int:
        """Extract the number of zones from centroids_type (e.g., '5-zone-AF' -> 5)."""
        parts = self.centroids_type.split('-')
        if parts and parts[0].isdigit():
            return int(parts[0])
        return 0
    
    def to_optimizer_config(self) -> dict:
        """Convert to the dict format expected by Optimizer class."""
        config = asdict(self)
        # Remove None values for cleaner config
        return {k: v for k, v in config.items() if v is not None}
    
    def get_output_folder_name(self) -> str:
        """Generate a descriptive folder name for this config's output."""
        if self.is_recursive:
            levels_str = '-'.join(self.recursive_levels)
            times_str = '-'.join(str(t) for t in self.solve_time_limits)
            return (
                f"{self.centroids_type}/seed{self.random_seed}/"
                f"frl{self.frl_dev}_racial{self.racial_dev}/"
                f"overage{self.overage}_shortage{self.shortage}/"
                f"{levels_str}_tl_{times_str}"
            )
        else:
            return (
                f"time{self.solve_time_limit}_seed{self.random_seed}_"
                f"centroids{self.centroids_type}_level{self.level}_"
                f"frl{self.frl_dev}_racial{self.racial_dev}_opt{self.optimizer}"
            )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BenchmarkConfig":
        """Load config from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to a YAML file."""
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


@dataclass
class ScenarioSweep:
    """
    Define parameter sweeps for generating multiple benchmark scenarios.
    
    Example:
        sweep = ScenarioSweep(
            centroids_types=['5-zone-AF', '6-zone-3'],
            frl_devs=[0.15, 0.2, 0.25],
            racial_devs=[0.15, 0.2, 0.25],
            random_seeds=[42, 14],
        )
        for config in sweep.generate_configs():
            run_benchmark(config)
    """
    centroids_types: list[str]
    frl_devs: list[float]
    racial_devs: list[float]
    random_seeds: list[int] = field(default_factory=lambda: [42])
    
    # Recursive mode parameters (optional)
    recursive_computations: list[list[tuple[str, float]]] | None = None
    total_times: list[int] | None = None
    overages: list[float] = field(default_factory=lambda: [0.8])
    shortages: list[float] = field(default_factory=lambda: [0.2])
    
    # Non-recursive mode parameters (optional)
    levels: list[str] | None = None
    solve_time_limits: list[int] | None = None
    
    # Common settings
    optimizer: str = "cp_int"
    is_local: bool = False
    
    def generate_configs(self) -> Iterator[BenchmarkConfig]:
        """Generate all config combinations from the sweep parameters."""
        if self.recursive_computations is not None:
            yield from self._generate_recursive_configs()
        else:
            yield from self._generate_single_level_configs()
    
    def _generate_recursive_configs(self) -> Iterator[BenchmarkConfig]:
        """Generate configs for recursive (multi-level) runs."""
        for (centroids, frl, racial, overage, shortage, 
             total_time, seed, computation) in product(
            self.centroids_types,
            self.frl_devs,
            self.racial_devs,
            self.overages,
            self.shortages,
            self.total_times or [240],
            self.random_seeds,
            self.recursive_computations,
        ):
            levels = [level for level, _ in computation]
            time_limits = [int(total_time * proportion) for _, proportion in computation]
            gap_limits = [0.05] * len(computation)
            
            yield BenchmarkConfig(
                centroids_type=centroids,
                frl_dev=frl,
                racial_dev=racial,
                random_seed=seed,
                recursive_levels=levels,
                solve_time_limits=time_limits,
                relative_gap_limits=gap_limits,
                overage=overage,
                shortage=shortage,
                optimizer=self.optimizer,
                is_local=self.is_local,
            )
    
    def _generate_single_level_configs(self) -> Iterator[BenchmarkConfig]:
        """Generate configs for single-level runs."""
        levels = self.levels or ["BlockGroup_0"]
        time_limits = self.solve_time_limits or [600]
        
        for centroids, frl, racial, level, time_limit, seed in product(
            self.centroids_types,
            self.frl_devs,
            self.racial_devs,
            levels,
            time_limits,
            self.random_seeds,
        ):
            yield BenchmarkConfig(
                centroids_type=centroids,
                frl_dev=frl,
                racial_dev=racial,
                random_seed=seed,
                level=level,
                solve_time_limit=time_limit,
                optimizer=self.optimizer,
                is_local=self.is_local,
            )
    
    def count_scenarios(self) -> int:
        """Count total number of scenarios without generating them."""
        return sum(1 for _ in self.generate_configs())
