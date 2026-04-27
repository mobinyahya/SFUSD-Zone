"""
Aggregate local recursive optimization runs into a summary CSV with full metrics.

Walks the recursive-runs directory structure:
  {centroid_type}/{seed}/{time_budget}/{hierarchy_config}/
    solution_info_Block_0.json, zone_dict_Block_0.json, ...

Computes metrics via ZoneMetricsCalculator for each run with a valid Block_0 zone_dict.
"""

import argparse
import json
import os
import pickle
import re
import sys
import time

import pandas as pd

from Zone_Generation.Config.Constants import get_dropbox_path
from Zone_Generation.Running_Analysis.metrics import ZoneMetricsCalculator


def parse_hierarchy_config(dir_name: str) -> tuple[list[str], list[int]]:
    """
    Parse a hierarchy config directory name into levels and time limits.

    Examples:
        'Block_0_tl_300' -> (['Block_0'], [300])
        'Block_1-Block_0_tl_450-150' -> (['Block_1', 'Block_0'], [450, 150])
        'Block_2-Block_1-Block_0_tl_300-150-150' -> (['Block_2', 'Block_1', 'Block_0'], [300, 150, 150])
    """
    parts = dir_name.split('_tl_')
    if len(parts) != 2:
        raise ValueError(f"Cannot parse hierarchy config: {dir_name}")

    levels = parts[0].split('-')
    time_limits = [int(t) for t in parts[1].split('-')]
    return levels, time_limits


def discover_runs(root_path: str) -> list[dict]:
    """
    Discover all runs under root_path by looking for zone_dict_Block_0.json
    or error.txt files within the expected directory structure.

    Returns list of dicts with path info and whether the run has a solution.
    """
    runs = []
    root_path = os.path.expanduser(root_path)

    # Walk to find leaf directories with zone_dict_Block_0.json or error.txt
    for dirpath, dirnames, filenames in os.walk(root_path):
        has_zone_dict = 'zone_dict_Block_0.json' in filenames
        has_error = 'error.txt' in filenames

        if not has_zone_dict and not has_error:
            continue

        # Extract path components: root/centroid_type/seed/time_budget/hierarchy_config
        rel_path = os.path.relpath(dirpath, root_path)
        parts = rel_path.split(os.sep)

        if len(parts) != 4:
            continue

        centroid_type, seed_str, time_budget_str, hierarchy_config = parts

        try:
            seed = int(seed_str)
            time_budget = int(time_budget_str)
            levels, time_limits = parse_hierarchy_config(hierarchy_config)
        except (ValueError, TypeError):
            print(f"  Skipping unparseable path: {rel_path}")
            continue

        runs.append({
            'path': dirpath,
            'rel_path': rel_path,
            'centroid_type': centroid_type,
            'seed': seed,
            'time_budget': time_budget,
            'hierarchy_config': hierarchy_config,
            'levels': levels,
            'time_limits': time_limits,
            'num_levels': len(levels),
            'has_zone_dict': has_zone_dict,
        })

    runs.sort(key=lambda r: r['rel_path'])
    return runs


def load_json_with_infinity(filepath: str) -> dict:
    """Load a JSON file, handling Infinity values."""
    with open(filepath, 'r') as f:
        text = f.read()
    # Replace JSON-invalid Infinity with a large number placeholder
    text = re.sub(r'\bInfinity\b', '1e308', text)
    text = re.sub(r'-Infinity\b', '-1e308', text)
    return json.loads(text)


def load_solution_infos(run_path: str, levels: list[str]) -> dict[str, dict]:
    """Load solution_info_{level}.json for each level in the run."""
    infos = {}
    for level in levels:
        info_file = os.path.join(run_path, f'solution_info_{level}.json')
        if os.path.exists(info_file):
            infos[level] = load_json_with_infinity(info_file)
    return infos


def aggregate_local_runs(
    root_path: str,
    output: str | None = None,
    include_choice: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame | None:
    """
    Aggregate local recursive runs into a summary DataFrame/CSV.

    Args:
        root_path: Root directory containing recursive runs
        output: Output CSV path (default: {root_path}/summary.csv)
        include_choice: Whether to compute choice metrics
        dry_run: If True, just list discovered runs without computing metrics
    """
    root_path = os.path.expanduser(root_path)

    if output is None:
        output = os.path.join(root_path, 'summary.csv')

    print(f"Discovering runs in {root_path}...")
    runs = discover_runs(root_path)

    n_with_solution = sum(1 for r in runs if r['has_zone_dict'])
    n_errors = sum(1 for r in runs if not r['has_zone_dict'])
    print(f"Found {len(runs)} runs ({n_with_solution} with solutions, {n_errors} errors)")

    if dry_run:
        for run in runs:
            status = "OK" if run['has_zone_dict'] else "ERROR"
            print(f"  [{status}] {run['rel_path']}")
        return None

    # Load Block_0 graph once
    graph_path = os.path.join(
        get_dropbox_path(is_local=False),
        'Optimization', 'Zones', 'Graphs', 'Block_0.pickle'
    )
    print(f"Loading Block_0 graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"  Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    rows = []
    start_time = time.time()

    for i, run in enumerate(runs, 1):
        rel = run['rel_path']

        # Load solution infos for all levels
        infos = load_solution_infos(run['path'], run['levels'])

        # Base row with path/structure info
        row = {
            'path': run['path'],
            'centroid_type': run['centroid_type'],
            'seed': run['seed'],
            'time_budget': run['time_budget'],
            'levels': '-'.join(run['levels']),
            'time_limits': '-'.join(str(t) for t in run['time_limits']),
            'num_levels': run['num_levels'],
        }

        # Determine status from Block_0 solution_info
        block0_info = infos.get('Block_0', {})
        status = block0_info.get('status', 'ERROR') if run['has_zone_dict'] else 'ERROR'
        row['status'] = status

        # Aggregate wall time and objective from all levels
        total_wall_time = sum(info.get('wall_time', 0) for info in infos.values())
        row['total_wall_time'] = total_wall_time
        row['objective_value'] = block0_info.get('boundary_cost', None)

        # Per-level status and wall time
        for level in run['levels']:
            info = infos.get(level, {})
            row[f'status_{level}'] = info.get('status', 'MISSING')
            row[f'wall_time_{level}'] = info.get('wall_time', None)
            row[f'boundary_cost_{level}'] = info.get('boundary_cost', None)

        # Extract config from Block_0 solution_info (or any available)
        config = block0_info.get('config', {})
        if not config:
            for info in infos.values():
                config = info.get('config', {})
                if config:
                    break

        # Add config values (scalar only)
        for k, v in config.items():
            if not isinstance(v, (list, dict)):
                row[f'config_{k}'] = v

        # Compute metrics if we have a zone_dict
        if run['has_zone_dict'] and status in ('FEASIBLE', 'OPTIMAL'):
            zone_dict_file = os.path.join(run['path'], 'zone_dict_Block_0.json')
            with open(zone_dict_file, 'r') as f:
                zone_dict = {int(k): v for k, v in json.load(f).items()}

            row['num_zones'] = len(set(zone_dict.values()))

            try:
                calc = ZoneMetricsCalculator(
                    zone_dict, G,
                    {'is_local': False, 'compute_choice': include_choice}
                )
                metrics_result = calc.compute_all(include_choice=include_choice)
                row.update(metrics_result.to_flat_dict())
            except Exception as e:
                print(f"  Warning: metrics failed for {rel}: {e}")

        rows.append(row)

        if i % 50 == 0 or i == len(runs):
            elapsed = time.time() - start_time
            print(f"  Processed {i}/{len(runs)} runs ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\nSaved {len(df)} rows to {output}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate local recursive optimization runs into a summary CSV'
    )
    parser.add_argument(
        '--input', '-i',
        default='~/sfusd-local-data/zones/SFUSD/local_runs/recursive-runs',
        help='Root directory containing recursive runs'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output CSV path (default: {input}/summary.csv)'
    )
    parser.add_argument(
        '--include-choice',
        action='store_true',
        help='Include choice/utility metrics (slower)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List discovered runs without computing metrics'
    )

    args = parser.parse_args()

    aggregate_local_runs(
        root_path=args.input,
        output=args.output,
        include_choice=args.include_choice,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
