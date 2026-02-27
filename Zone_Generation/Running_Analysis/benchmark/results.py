"""
Benchmark results module.

Provides standardized result storage and aggregation for zoning benchmarks.
"""
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from Zone_Generation.Running_Analysis.metrics import ZoneMetricsCalculator


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
    zone_data: dict = field(default_factory=dict)  # Per-zone detailed data
    
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
    
    def compute_metrics(self, G, config: dict | None = None) -> None:
        """Compute zoning metrics for the final solution."""
        if self.zone_dict:
            calc = ZoneMetricsCalculator(self.zone_dict.copy(), G, config)
            result = calc.compute_all()
            self.metrics = result.to_flat_dict()
            self.zone_data = {zid: zd.to_dict() for zid, zd in result.zone_data.items()}
    
    def export_zone_data(self, folder_path: str) -> str:
        """
        Export detailed per-zone data to a CSV file.
        
        Returns:
            Path to the saved CSV file
        """
        if not self.zone_data:
            return ""
            
        save_path = os.path.expanduser(folder_path)
        os.makedirs(save_path, exist_ok=True)
        
        # Use config to generate a filename
        filename = "zone_data.csv"
        # If we have a config, try to make it more descriptive
        if self.config:
            # Generate a unique filename from config parameters
            parts = [
                f"ct_{self.config.get('centroids_type', 'X')}",
                f"l_{self.config.get('level', 'X')}",
                f"s_{self.config.get('random_seed', 'X')}",
                f"f_{self.config.get('frl_dev', 'X')}",
                f"r_{self.config.get('racial_dev', 'X')}",
                f"o_{self.config.get('overage', 'X')}",
                f"sh_{self.config.get('shortage', 'X')}"
            ]
            filename = "_".join(parts) + ".csv"
            
        csv_path = os.path.join(save_path, filename)
        
        # Convert zone_data dict to DataFrame
        # Flatten nested dicts (ethnicity_pcts, programs)
        rows = []
        for zd in self.zone_data.values():
            row = zd.copy()
            # Flatten ethnicity
            eth = row.pop('ethnicity_pcts', {})
            for e, p in eth.items():
                row[f'eth_{e}'] = p
            # Flatten programs
            prog = row.pop('programs', {})
            for p, c in prog.items():
                row[f'prog_{p}'] = c
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        return csv_path

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
            "zone_data": self.zone_data,
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
            zone_data=data.get("zone_data", {}),
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


def aggregate_results(
    root_folder: str, 
    output_file: str | None = None,
    recompute_metrics: bool = True,
    include_choice: bool = False,
    zone_data_folder: str | None = None
) -> pd.DataFrame:
    """
    Aggregate all benchmark results from a folder tree into a DataFrame.
    
    Walks through root_folder looking for result.json files and combines them.
    
    Args:
        root_folder: Root folder to search for results
        output_file: Optional CSV file to save results
        recompute_metrics: If True, recompute metrics using ZoneMetricsCalculator
        include_choice: If True, include choice metrics (slower)
        zone_data_folder: Optional folder to export detailed per-zone CSVs
    """
    import pickle
    from Zone_Generation.Running_Analysis.metrics import ZoneMetricsCalculator
    from Zone_Generation.Config.Constants import get_dropbox_path
    
    root_path = os.path.expanduser(root_folder)
    results = []
    
    # Cache graphs by level to avoid reloading
    graph_cache = {}
    
    def get_graph(level: str, is_local: bool = False):
        if level not in graph_cache:
            graph_path = f"{get_dropbox_path(is_local)}/Optimization/Zones/Graphs/{level}.pickle"
            with open(graph_path, 'rb') as f:
                graph_cache[level] = pickle.load(f)
        return graph_cache[level]
    
    for root, _, files in os.walk(root_path):
        if "result.json" not in files:
            continue
            
        try:
            result = BenchmarkResult.load(root)
            
            # Base row data
            row = {
                "path": root,
                "status": result.status,
                "total_wall_time": result.total_wall_time,
                "boundary_cost": result.boundary_cost,
                "num_zones": len(set(result.zone_dict.values())) if result.zone_dict else 0,
            }
            
            # Recompute metrics if requested and we have a zone_dict
            if recompute_metrics and result.zone_dict:
                # Determine level from config or last level result
                level = result.config.get('level', 'BlockGroup_0')
                if result.level_results:
                    level = result.level_results[-1].level
                
                is_local = result.config.get('is_local', False)
                
                try:
                    G = get_graph(level, is_local)
                    calc = ZoneMetricsCalculator(
                        result.zone_dict, G, 
                        {'is_local': is_local, 'compute_choice': include_choice}
                    )
                    metrics_result = calc.compute_all(include_choice=include_choice)
                    row.update(metrics_result.to_flat_dict())
                    
                    # Update result with new data for potential export
                    result.metrics = metrics_result.to_flat_dict()
                    result.zone_data = {zid: zd.to_dict() for zid, zd in metrics_result.zone_data.items()}
                except Exception as e:
                    print(f"  Warning: Could not recompute metrics for {root}: {e}")
                    row.update(result.metrics)
            else:
                row.update(result.metrics)
            
            # Export detailed zone data if requested (now has data if recomputed)
            if zone_data_folder:
                result.export_zone_data(zone_data_folder)
            
            # Add config values
            row.update({
                f"config_{k}": v for k, v in result.config.items() 
                if not isinstance(v, (list, dict))
            })
            
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

