"""
Benchmark runner module.

Provides a unified entry point for running zoning optimization benchmarks.
"""
import copy
import datetime
import os
from typing import Callable

from Zone_Generation.Optimization.optimizer import Optimizer
from Zone_Generation.Optimization.recursive_zoning import solve_level
from Zone_Generation.Running_Analysis.benchmark.config import BenchmarkConfig
from Zone_Generation.Running_Analysis.benchmark.results import (
    BenchmarkResult,
    LevelResult,
)


def run_benchmark(
    config: BenchmarkConfig,
    output_folder: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """
    Run a complete benchmark with unified handling for both modes.
    
    Args:
        config: Benchmark configuration
        output_folder: Where to save results (optional, uses config if not set)
        progress_callback: Optional callback for progress updates
        
    Returns:
        BenchmarkResult with all level results and aggregated metrics
    """
    # Determine output folder
    if output_folder is None:
        output_folder = config.log_folder
    if output_folder is None:
        output_folder = f"/tmp/benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_folder = os.path.expanduser(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert to optimizer config format
    opt_config = config.to_optimizer_config()
    opt_config['log_folder'] = output_folder
    
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    try:
        if config.is_recursive:
            result = _run_recursive(config, opt_config, output_folder, log)
        else:
            result = _run_single_level(config, opt_config, output_folder, log)
        
        # Compute metrics on final solution
        if result.zone_dict:
            optimizer = Optimizer.get_optimizer(opt_config)
            result.compute_metrics(optimizer.G)
        
        # Save results
        result.save(output_folder)
        log(f"Results saved to {output_folder}")
        
        return result
        
    except Exception as e:
        log(f"Error: {e}")
        result = BenchmarkResult.from_error(e, opt_config)
        result.save(output_folder)
        return result


def _run_recursive(
    config: BenchmarkConfig,
    opt_config: dict,
    output_folder: str,
    log: Callable[[str], None],
) -> BenchmarkResult:
    """Run recursive (multi-level) zoning optimization."""
    result = BenchmarkResult(status="RUNNING", config=opt_config)
    cur_block_zone_dict = None
    
    for i, level in enumerate(config.recursive_levels):
        log(f"Solving level {i+1}/{len(config.recursive_levels)}: {level}")
        
        # Create level-specific config
        level_config = copy.deepcopy(opt_config)
        level_config['level'] = level
        level_config['solve_time_limit'] = config.solve_time_limits[i]
        level_config['relative_gap_limit'] = config.relative_gap_limits[i]
        
        # Solve this level
        solution_output = solve_level(level_config, config.is_local, cur_block_zone_dict)
        
        # Create level result
        level_result = LevelResult(
            level=level,
            status=solution_output.status,
            wall_time=solution_output.wall_time,
            boundary_cost=solution_output.get_boundary_cost(),
            zone_dict=solution_output.zone_dict or {},
            config=level_config,
        )
        result.add_level_result(level_result)
        
        # Update for next iteration
        cur_block_zone_dict = solution_output.block_zone_dict
        
        log(f"  Status: {solution_output.status}, Time: {solution_output.wall_time:.1f}s, "
            f"Boundary: {level_result.boundary_cost}")
        
        # Stop if infeasible
        if solution_output.status in ['INFEASIBLE', 'MODEL_INVALID', 'UNKNOWN']:
            log(f"Stopping: {solution_output.status} at level {level}")
            break
    
    return result


def _run_single_level(
    config: BenchmarkConfig,
    opt_config: dict,
    output_folder: str,
    log: Callable[[str], None],
) -> BenchmarkResult:
    """Run single-level zoning optimization."""
    result = BenchmarkResult(status="RUNNING", config=opt_config)
    
    log(f"Solving single level: {config.level}")
    
    optimizer = Optimizer.get_optimizer(opt_config)
    optimizer.add_variables()
    optimizer.add_constraints()
    optimizer.add_boundary_objective()
    
    solution_output = optimizer.solve()
    
    level_result = LevelResult(
        level=config.level,
        status=solution_output.status,
        wall_time=solution_output.wall_time,
        boundary_cost=solution_output.get_boundary_cost(),
        zone_dict=solution_output.zone_dict or {},
        config=opt_config,
    )
    result.add_level_result(level_result)
    
    log(f"Status: {solution_output.status}, Time: {solution_output.wall_time:.1f}s, "
        f"Boundary: {level_result.boundary_cost}")
    
    return result


def run_batch(
    configs: list[BenchmarkConfig],
    base_output_folder: str,
    continue_on_error: bool = True,
) -> list[BenchmarkResult]:
    """
    Run multiple benchmark configurations.
    
    Args:
        configs: List of benchmark configurations
        base_output_folder: Base folder for all outputs
        continue_on_error: Whether to continue if a run fails
        
    Returns:
        List of BenchmarkResults
    """
    results = []
    total = len(configs)
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Running benchmark {i+1}/{total}")
        print(f"Config: {config.centroids_type}, frl={config.frl_dev}, racial={config.racial_dev}")
        print(f"{'='*60}")
        
        output_folder = os.path.join(
            os.path.expanduser(base_output_folder),
            config.get_output_folder_name()
        )
        
        try:
            result = run_benchmark(config, output_folder)
            results.append(result)
        except Exception as e:
            print(f"Error in benchmark {i+1}: {e}")
            if not continue_on_error:
                raise
            results.append(BenchmarkResult.from_error(e, config.to_optimizer_config()))
    
    return results
