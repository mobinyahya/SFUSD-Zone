"""
Benchmark results module.

Provides standardized result storage and aggregation for zoning benchmarks.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from Zone_Generation.Running_Analysis.zoning_metrics import ZoneMetrics


@dataclass
class LevelResult:
    """Result for a single optimization level."""
    level: str
    status: str
    wall_time: float
    boundary_cost: float
    zone_dict: dict[int, int]
    config: dict
    
    def save(self, folder_path: str) -> None:
        """Save level result as solution_info_{level}.json."""
        save_path = os.path.expanduser(folder_path)
        os.makedirs(save_path, exist_ok=True)
        
        # Save zone dict
        zone_dict_file = os.path.join(save_path, f"zone_dict_{self.level}.json")
        with open(zone_dict_file, "w") as f:
            # Convert int keys to strings for JSON
            json.dump({str(k): v for k, v in self.zone_dict.items()}, f)
        
        # Save solution info
        info = {
            "level": self.level,
            "status": self.status,
            "wall_time": self.wall_time,
            "boundary_cost": self.boundary_cost,
            "config": self.config,
        }
        info_file = os.path.join(save_path, f"solution_info_{self.level}.json")
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)


@dataclass
class BenchmarkResult:
    """
    Standardized result for a complete benchmark run.
    
    Contains results for all levels (for recursive runs) and aggregated metrics.
    """
    # Status
    status: str
    error_message: str | None = None
    
    # Timing
    total_wall_time: float = 0.0
    level_results: list[LevelResult] = field(default_factory=list)
    
    # Final solution quality (from last level)
    boundary_cost: float = -1
    zone_dict: dict[int, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Configuration snapshot
    config: dict = field(default_factory=dict)
    
    def add_level_result(self, level_result: LevelResult) -> None:
        """Add a level result and update aggregates."""
        self.level_results.append(level_result)
        self.total_wall_time += level_result.wall_time
        # Update final values from latest level
        self.status = level_result.status
        self.boundary_cost = level_result.boundary_cost
        self.zone_dict = level_result.zone_dict
    
    def compute_metrics(self, G) -> None:
        """Compute zoning metrics for the final solution."""
        if self.zone_dict:
            zm = ZoneMetrics(self.zone_dict.copy(), G)
            self.metrics = zm.get_metrics()
    
    def save(self, folder_path: str) -> None:
        """
        Save benchmark result to folder.
        
        Creates:
        - solution_info_{level}.json for each level
        - zone_dict_{level}.json for each level
        - result.json with aggregated results (duplicating final level info)
        """
        save_path = os.path.expanduser(folder_path)
        os.makedirs(save_path, exist_ok=True)
        
        # Save each level's results
        for level_result in self.level_results:
            level_result.save(save_path)
        
        # Save aggregated result.json (includes final level info)
        result_data = {
            "status": self.status,
            "error_message": self.error_message,
            "total_wall_time": self.total_wall_time,
            "level_wall_times": {
                lr.level: lr.wall_time for lr in self.level_results
            },
            "boundary_cost": self.boundary_cost,
            "metrics": self.metrics,
            "config": self.config,
            "levels": [lr.level for lr in self.level_results],
        }
        
        result_file = os.path.join(save_path, "result.json")
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Also save solution_info.json as alias for final level
        if self.level_results:
            final_level = self.level_results[-1]
            info = {
                "level": final_level.level,
                "status": self.status,
                "wall_time": self.total_wall_time,
                "boundary_cost": self.boundary_cost,
                "config": self.config,
            }
            info_file = os.path.join(save_path, "solution_info.json")
            with open(info_file, "w") as f:
                json.dump(info, f, indent=2)
    
    @classmethod
    def from_error(cls, error: Exception, config: dict) -> "BenchmarkResult":
        """Create a result representing a failed run."""
        return cls(
            status="ERROR",
            error_message=str(error),
            config=config,
        )
    
    @classmethod
    def load(cls, folder_path: str) -> "BenchmarkResult":
        """Load a benchmark result from folder."""
        save_path = os.path.expanduser(folder_path)
        result_file = os.path.join(save_path, "result.json")
        
        with open(result_file, "r") as f:
            data = json.load(f)
        
        result = cls(
            status=data["status"],
            error_message=data.get("error_message"),
            total_wall_time=data["total_wall_time"],
            boundary_cost=data["boundary_cost"],
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
        )
        
        # Load level results
        for level in data.get("levels", []):
            zone_dict_file = os.path.join(save_path, f"zone_dict_{level}.json")
            info_file = os.path.join(save_path, f"solution_info_{level}.json")
            
            if os.path.exists(zone_dict_file) and os.path.exists(info_file):
                with open(zone_dict_file, "r") as f:
                    zone_dict = {int(k): v for k, v in json.load(f).items()}
                with open(info_file, "r") as f:
                    info = json.load(f)
                
                level_result = LevelResult(
                    level=level,
                    status=info["status"],
                    wall_time=info["wall_time"],
                    boundary_cost=info["boundary_cost"],
                    zone_dict=zone_dict,
                    config=info.get("config", {}),
                )
                result.level_results.append(level_result)
        
        if result.level_results:
            result.zone_dict = result.level_results[-1].zone_dict
        
        return result


def aggregate_results(root_folder: str, output_file: str | None = None) -> pd.DataFrame:
    """
    Aggregate all benchmark results from a folder tree into a DataFrame.
    
    Walks through root_folder looking for result.json files and combines them.
    """
    root_path = os.path.expanduser(root_folder)
    results = []
    
    for root, dirs, files in os.walk(root_path):
        if "result.json" in files:
            try:
                result = BenchmarkResult.load(root)
                row = {
                    "path": root,
                    "status": result.status,
                    "total_wall_time": result.total_wall_time,
                    "boundary_cost": result.boundary_cost,
                    **result.metrics,
                    **{f"config_{k}": v for k, v in result.config.items() 
                       if not isinstance(v, (list, dict))},
                }
                
                # Add level-specific wall times
                for lr in result.level_results:
                    row[f"wall_time_{lr.level}"] = lr.wall_time
                
                results.append(row)
            except Exception as e:
                print(f"Error loading result from {root}: {e}")
    
    df = pd.DataFrame(results)
    
    if output_file:
        df.to_csv(os.path.expanduser(output_file), index=False)
        print(f"Saved aggregated results to {output_file}")
    
    return df
