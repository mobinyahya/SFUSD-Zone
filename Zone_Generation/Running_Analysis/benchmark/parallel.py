"""
Parallel benchmark execution module.

Provides utilities for running multiple benchmark scenarios in parallel
using ProcessPoolExecutor, optimized for long-running batch jobs.
"""
import os
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from Zone_Generation.Running_Analysis.benchmark.config import BenchmarkConfig
from Zone_Generation.Running_Analysis.benchmark.results import BenchmarkResult


# Default workers: CP-SAT uses ~6 cores per solver, so 5 workers = ~30 cores
DEFAULT_MAX_WORKERS = 5


@dataclass
class ParallelConfig:
    """Configuration for parallel batch execution."""
    max_workers: int = DEFAULT_MAX_WORKERS
    skip_existing: bool = True
    continue_on_error: bool = True
    max_tasks_per_worker: int | None = 50  # Recycle workers to prevent memory leaks


@dataclass
class ProgressUpdate:
    """Progress information for callbacks."""
    completed: int
    total: int
    current_scenario: str
    status: str  # 'running', 'success', 'error', 'skipped'
    elapsed_seconds: float
    estimated_remaining_seconds: float | None = None


@dataclass
class BatchResult:
    """Summary of batch execution."""
    results: list[BenchmarkResult] = field(default_factory=list)
    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_wall_time: float = 0.0

    def add_result(self, result: BenchmarkResult, was_skipped: bool = False):
        """Add a result and update counts."""
        self.results.append(result)
        if was_skipped:
            self.skipped += 1
        elif result.status in ['ERROR', 'INFEASIBLE']:
            self.failed += 1
        else:
            self.successful += 1


def _run_single_scenario(
    config: BenchmarkConfig,
    output_folder: str,
    project_root: str,
) -> BenchmarkResult:
    """
    Entry point for worker processes to run a single benchmark.
    
    This function is called in a separate process, so it has its own
    memory space and can safely load graphs, create solvers, etc.
    
    Args:
        config: Benchmark configuration
        output_folder: Where to save results
        project_root: Project root directory (for relative paths in Optimizer)
    """
    # Change to project root for relative path compatibility
    # (Optimizer uses ../Config/centroids.yaml which requires being in Zone_Generation/Optimization)
    import os
    optimization_dir = os.path.join(project_root, 'Zone_Generation', 'Optimization')
    if os.path.exists(optimization_dir):
        os.chdir(optimization_dir)
    
    # Import here to avoid pickling issues
    from Zone_Generation.Running_Analysis.benchmark.runner import run_benchmark
    return run_benchmark(config, output_folder)


def _check_existing_result(output_folder: str) -> bool:
    """Check if a result already exists for this scenario."""
    result_file = os.path.join(os.path.expanduser(output_folder), "result.json")
    return os.path.exists(result_file)


class ParallelRunner:
    """
    Manages parallel benchmark execution with progress tracking.
    
    Example:
        runner = ParallelRunner(ParallelConfig(max_workers=5))
        result = runner.run(configs, "/path/to/output")
        print(f"Completed {result.successful}/{result.total}")
    """
    
    def __init__(self, config: ParallelConfig | None = None):
        self.config = config or ParallelConfig()
        self._shutdown_event = threading.Event()
        self._original_sigint = None
    
    def _setup_signal_handler(self):
        """Setup graceful shutdown on Ctrl+C."""
        def handler(signum, frame):
            print("\n[ParallelRunner] Shutdown requested, finishing current tasks...")
            self._shutdown_event.set()
            # Restore original handler for second Ctrl+C to force quit
            signal.signal(signal.SIGINT, self._original_sigint)
        
        self._original_sigint = signal.signal(signal.SIGINT, handler)
    
    def _restore_signal_handler(self):
        """Restore original signal handler."""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
    
    def run(
        self,
        configs: list[BenchmarkConfig],
        base_output_folder: str,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> BatchResult:
        """
        Run multiple benchmarks in parallel.
        
        Args:
            configs: List of benchmark configurations to run
            base_output_folder: Base folder for all outputs
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchResult with summary statistics
        """
        base_folder = os.path.expanduser(base_output_folder)
        os.makedirs(base_folder, exist_ok=True)
        
        batch_result = BatchResult(total=len(configs))
        start_time = time.time()
        completed = 0
        
        # Compute project root for worker processes
        # This file is at Zone_Generation/Running_Analysis/benchmark/parallel.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        
        # Prepare output folders and filter existing if skip_existing
        tasks: list[tuple[BenchmarkConfig, str]] = []
        for config in configs:
            output_folder = os.path.join(base_folder, config.get_output_folder_name())
            
            if self.config.skip_existing and _check_existing_result(output_folder):
                # Load existing result
                try:
                    result = BenchmarkResult.load(output_folder)
                    batch_result.add_result(result, was_skipped=True)
                except Exception:
                    # If can't load, re-run
                    tasks.append((config, output_folder))
                
                completed += 1
                if progress_callback:
                    progress_callback(ProgressUpdate(
                        completed=completed,
                        total=len(configs),
                        current_scenario=config.get_output_folder_name(),
                        status='skipped',
                        elapsed_seconds=time.time() - start_time,
                    ))
            else:
                tasks.append((config, output_folder))
        
        if not tasks:
            batch_result.total_wall_time = time.time() - start_time
            return batch_result
        
        # Setup signal handler for graceful shutdown
        self._setup_signal_handler()
        
        try:
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                max_tasks_per_child=self.config.max_tasks_per_worker,
            ) as executor:
                # Use a throttled submission approach to avoid deadlocking the executor
                # when faced with thousands of tasks and worker recycling.
                future_to_config = {}
                task_iter = iter(tasks)
                
                # Submit initial batch (2x workers to keep queue full but not overloaded)
                initial_batch_size = self.config.max_workers * 2
                for _ in range(initial_batch_size):
                    try:
                        config, output_folder = next(task_iter)
                        if self._shutdown_event.is_set():
                            break
                        future = executor.submit(_run_single_scenario, config, output_folder, project_root)
                        future_to_config[future] = (config, output_folder)
                    except StopIteration:
                        break
                
                # Collect results and feed new tasks
                scenario_times = []
                from concurrent.futures import wait, FIRST_COMPLETED
                
                while future_to_config:
                    if self._shutdown_event.is_set():
                        # Cancel remaining futures
                        for f in future_to_config:
                            f.cancel()
                        break
                    
                    # Wait for at least one future to complete
                    done, _ = wait(future_to_config.keys(), return_when=FIRST_COMPLETED)
                    
                    for future in done:
                        config, output_folder = future_to_config.pop(future)
                        scenario_start = time.time()
                        
                        try:
                            # Use a short timeout to be safe, but result() should be ready
                            result = future.result()
                            batch_result.add_result(result)
                            status = 'success' if result.status not in ['ERROR', 'INFEASIBLE'] else 'error'
                        except (Exception, BaseException) as e:
                            # Catch BaseException (like SystemExit) to prevent parent hang
                            if self.config.continue_on_error:
                                error_msg = str(e) or e.__class__.__name__
                                error_result = BenchmarkResult.from_error(Exception(error_msg), config.to_optimizer_config())
                                batch_result.add_result(error_result)
                                status = 'error'
                            else:
                                raise
                        
                        completed += 1
                        scenario_time = time.time() - scenario_start
                        scenario_times.append(scenario_time)
                        
                        # Estimate remaining time
                        avg_time = sum(scenario_times) / len(scenario_times)
                        remaining = batch_result.total - completed
                        estimated_remaining = avg_time * remaining / max(self.config.max_workers, 1)
                        
                        if progress_callback:
                            progress_callback(ProgressUpdate(
                                completed=completed,
                                total=len(configs),
                                current_scenario=config.get_output_folder_name(),
                                status=status,
                                elapsed_seconds=time.time() - start_time,
                                estimated_remaining_seconds=estimated_remaining,
                            ))
                        else:
                            # Default progress output
                            print(f"[{completed}/{batch_result.total}] {status.upper()}: "
                                  f"{config.centroids_type} frl={config.frl_dev} "
                                  f"(ETA: {estimated_remaining/60:.1f}min)", flush=True)
                        
                        # Submit next task if we have more
                        if not self._shutdown_event.is_set():
                            try:
                                next_config, next_output_folder = next(task_iter)
                                next_future = executor.submit(_run_single_scenario, next_config, next_output_folder, project_root)
                                future_to_config[next_future] = (next_config, next_output_folder)
                            except StopIteration:
                                pass
        
        finally:
            self._restore_signal_handler()
        
        batch_result.total_wall_time = time.time() - start_time
        return batch_result
    
    def shutdown(self):
        """Request graceful shutdown of running tasks."""
        self._shutdown_event.set()
