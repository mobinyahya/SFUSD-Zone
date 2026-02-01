"""
Comparative Benchmarks CLI

Unified entry point for running zoning optimization benchmarks.
Uses the benchmark package for configuration, running, and result aggregation.
"""
import argparse
import os
import sys

import yaml

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Zone_Generation.Running_Analysis.benchmark.config import BenchmarkConfig, ScenarioSweep
from Zone_Generation.Running_Analysis.benchmark.runner import run_benchmark, run_batch
from Zone_Generation.Running_Analysis.benchmark.results import aggregate_results


# ============================================================================
# Predefined Scenario Sweeps
# ============================================================================

CENTROID_TYPES = [
    '4-zone-rec-4', '4-zone-rec-3',
    '5-zone-AF', '5-zone-AF-relocated',
    '6-zone-2', '6-zone-3',
    '7-zone-14', '7-zone-19',
    '8-zone-25', '8-zone-22',
    '10-zone-11', '10-zone-3',
    '13-zone-6', '13-zone-5',
]

# Recursive computation patterns: (level, time_proportion)
RECURSIVE_COMPUTATIONS = [
    [('BlockGroup_1', 0.5), ('BlockGroup_0', 0.5)],
]


def get_recursive_sweep() -> ScenarioSweep:
    """Get the predefined recursive zoning sweep."""
    return ScenarioSweep(
        centroids_types=CENTROID_TYPES,
        frl_devs=[0.12, 0.15, 0.2, 0.25],
        racial_devs=[0.12, 0.15, 0.2, 0.25],
        random_seeds=[42],
        recursive_computations=RECURSIVE_COMPUTATIONS,
        total_times=[4 * 60],
        overages=[0.7, 0.8, 0.9],
        shortages=[0.15, 0.2, 0.25],
    )


def get_single_level_sweep() -> ScenarioSweep:
    """Get the predefined single-level zoning sweep."""
    return ScenarioSweep(
        centroids_types=CENTROID_TYPES,
        frl_devs=[0.15, 0.2, 0.25, 0.3],
        racial_devs=[0.15, 0.2, 0.25, 0.3],
        random_seeds=[42, 14],
        levels=['Block_0'],
        solve_time_limits=[10 * 60],
    )


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_run_single(args):
    """Run a single benchmark from config file or arguments."""
    if args.config_file:
        config = BenchmarkConfig.from_yaml(args.config_file)
    else:
        # Load base config and override with args
        with open("../Config/config.yaml", "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        
        config = BenchmarkConfig(
            centroids_type=args.centroids or base['centroids_type'],
            frl_dev=args.frl_dev or base['frl_dev'],
            racial_dev=args.racial_dev or base['racial_dev'],
            random_seed=args.seed or base.get('random_seed', 42),
            level=args.level or base.get('level', 'BlockGroup_0'),
            solve_time_limit=args.time_limit or base.get('solve_time_limit', 600),
            recursive_levels=base.get('recursive_levels') if args.recursive else None,
            solve_time_limits=base.get('solve_time_limits') if args.recursive else None,
            relative_gap_limits=base.get('relative_gap_limits') if args.recursive else None,
            is_local=base.get('is_local', False),
        )
    
    result = run_benchmark(config, args.output)
    print(f"\nResult: {result.status}")
    print(f"Wall time: {result.total_wall_time:.1f}s")
    print(f"Boundary cost: {result.boundary_cost}")


def cmd_run_batch(args):
    """Run a batch of benchmarks from a sweep."""
    if args.mode == 'recursive':
        sweep = get_recursive_sweep()
    else:
        sweep = get_single_level_sweep()
    
    print(f"Generating {sweep.count_scenarios()} scenarios...")
    configs = list(sweep.generate_configs())
    
    if args.limit:
        configs = configs[:args.limit]
        print(f"Limited to {len(configs)} scenarios")
    
    results = run_batch(configs, args.output)
    
    # Summary
    success = sum(1 for r in results if r.status not in ['ERROR', 'INFEASIBLE'])
    print(f"\n{'='*60}")
    print(f"Completed: {success}/{len(results)} successful")


def cmd_aggregate(args):
    """Aggregate results from a folder into CSV."""
    df = aggregate_results(args.input, args.output)
    print(f"Aggregated {len(df)} results")
    if not df.empty:
        print("\nSummary:")
        print(df[['status', 'total_wall_time', 'boundary_cost']].describe())


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Zoning Optimization Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run single
    run_parser = subparsers.add_parser('run', help='Run a single benchmark')
    run_parser.add_argument('--config-file', '-c', help='YAML config file')
    run_parser.add_argument('--output', '-o', required=True, help='Output folder')
    run_parser.add_argument('--centroids', help='Centroid type (e.g., 5-zone-AF)')
    run_parser.add_argument('--frl-dev', type=float, help='FRL deviation limit')
    run_parser.add_argument('--racial-dev', type=float, help='Racial deviation limit')
    run_parser.add_argument('--seed', type=int, help='Random seed')
    run_parser.add_argument('--level', help='Optimization level')
    run_parser.add_argument('--time-limit', type=int, help='Solve time limit (seconds)')
    run_parser.add_argument('--recursive', action='store_true', help='Use recursive mode')
    run_parser.set_defaults(func=cmd_run_single)
    
    # Run batch
    batch_parser = subparsers.add_parser('batch', help='Run batch benchmarks')
    batch_parser.add_argument('--mode', choices=['recursive', 'single'], 
                              default='recursive', help='Benchmark mode')
    batch_parser.add_argument('--output', '-o', required=True, help='Base output folder')
    batch_parser.add_argument('--limit', type=int, help='Limit number of scenarios')
    batch_parser.set_defaults(func=cmd_run_batch)
    
    # Aggregate
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate results to CSV')
    agg_parser.add_argument('--input', '-i', required=True, help='Root folder with results')
    agg_parser.add_argument('--output', '-o', help='Output CSV file')
    agg_parser.set_defaults(func=cmd_aggregate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
