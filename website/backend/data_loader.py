"""Data loading utilities for the SFUSD Zoning Dashboard."""
import json
import os
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from LLM.exploration.pareto import (
    load_solutions,
    normalize_metrics,
    compute_pareto_frontier,
    METRIC_CONFIG,
    get_metric_columns,
)
from LLM.exploration.clusters import (
    vectorize_solutions,
    cluster_solutions,
    compute_cluster_directions,
    get_representative_solution,
)
from Zone_Generation.Config.Constants import AREA_ETHNICITIES, zone_colors
from LLM.exploration.metrics_config import CORE_METRICS, get_core_metric_columns

# Paths
CSV_PATH = Path("/home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/new_benchmarks_test/summary.csv")
GRAPH_PATH = Path("/home/kumarc/sfusd-local-data/zones/SFUSD/Optimization/Zones/Graphs/BlockGroup_0.pickle")
SHAPEFILE_PATH = Path("/share/data/school_choice/shapefiles/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp")
GEOJSON_PATH = Path(__file__).parent.parent / "data" / "sf_blockgroups.geojson"

# Cache
_graph_cache = None
_geojson_cache = None
_clusters_cache = None
_solution_space_stats_cache = None


def load_graph() -> nx.Graph:
    """Load the BlockGroup graph and cache it."""
    global _graph_cache
    if _graph_cache is None:
        with open(GRAPH_PATH, "rb") as f:
            _graph_cache = pickle.load(f)
    return _graph_cache


def get_node_to_blockgroup_map(G: nx.Graph) -> dict[int, int]:
    """Build mapping from graph node ID to BlockGroup ID."""
    return {node: data["area_id"] for node, data in G.nodes(data=True)}


def load_solution_result(solution_path: str) -> dict:
    """
    Load result.json from a solution directory.

    Args:
        solution_path: Path to solution folder containing result.json

    Returns:
        Dict with keys: status, metrics, zone_data, boundary_cost, total_wall_time, etc.
    """
    result_path = os.path.join(solution_path, "result.json")
    with open(result_path, "r") as f:
        result = json.load(f)
    return result


def load_zone_dict(solution_path: str) -> dict[int, int]:
    """
    Load zone_dict for a solution and convert to blockgroup->zone mapping.

    Args:
        solution_path: Path to solution folder containing zone_dict_BlockGroup_0.json

    Returns:
        Dict mapping blockgroup_id to zone_id
    """
    zone_dict_path = os.path.join(solution_path, "zone_dict_BlockGroup_0.json")
    with open(zone_dict_path, "r") as f:
        node_zone_dict = json.load(f)

    # Convert string keys to int
    node_zone_dict = {int(k): v for k, v in node_zone_dict.items()}

    # Map node IDs to blockgroup IDs
    G = load_graph()
    node_to_bg = get_node_to_blockgroup_map(G)

    bg_zone_dict = {}
    for node_id, zone_id in node_zone_dict.items():
        if node_id in node_to_bg:
            bg_zone_dict[node_to_bg[node_id]] = zone_id

    return bg_zone_dict


def get_zone_demographics(solution_path: str) -> dict[int, dict]:
    """
    Load pre-calculated zone demographics from result.json.

    NOTE: This function now reads from pre-calculated result.json instead of
    recalculating from the graph. This is MUCH faster and avoids redundant computation.

    Args:
        solution_path: Path to solution folder containing result.json

    Returns:
        Dict mapping zone_id to comprehensive zone data with:
        - zone_id: zone identifier
        - ge_students: total general education students
        - FRL_pct: Free/Reduced Lunch percentage (0-100, normalized for frontend)
        - frl_pct: Free/Reduced Lunch percentage (0-1, from result.json)
        - ethnicity_pcts: dict of ethnicity percentages by ethnicity name
        - programs: dict of program counts (GE, SA, CN, AF, etc.)
        - total_programs: total number of programs
        - language_immersion_count: count of language immersion programs
        - special_ed_count: count of special education programs
        - avg_greatschools_rating: average GreatSchools rating
        - avg_math_score: average math test score
        - avg_eng_score: average English test score
        - avg_suspension_index: average suspension index
        - avg_closest_school_distance: average distance to closest school
        - schools_in_attendance_area: number of schools in zone
        - avg_max_utility: average maximum utility
        - avg_logsum_utility: average logsum utility
    """
    try:
        result = load_solution_result(solution_path)
        # result.json has zone_data with string keys, convert to int
        zone_data = result.get("zone_data", {})

        # Normalize field names for frontend compatibility
        normalized_data = {}
        for zone_id, data in zone_data.items():
            zone_dict = data.copy()
            # Frontend expects FRL_pct in 0-100 range (uppercase FRL)
            # result.json has frl_pct in 0-1 range (lowercase frl)
            if "frl_pct" in zone_dict:
                zone_dict["FRL_pct"] = zone_dict["frl_pct"] * 100
            normalized_data[int(zone_id)] = zone_dict

        return normalized_data
    except FileNotFoundError:
        # Fallback for older solutions without result.json
        # (could remove this if all solutions have result.json)
        raise FileNotFoundError(
            f"result.json not found in {solution_path}. "
            "This solution may be from an older run without pre-calculated metrics."
        )


def convert_shapefile_to_geojson():
    """Convert shapefile to GeoJSON with BlockGroup IDs. Run once."""
    print(f"Loading shapefile from {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)

    # Ensure correct CRS
    gdf = gdf.to_crs(epsg=4326)

    # Clean up blockgroup IDs
    gdf["geoid10"] = gdf["geoid10"].fillna(0).astype("int64")

    # Load block-to-blockgroup mapping
    block_bg_path = Path("/home/kumarc/sfusd-local-data/zones/SFUSD/Optimization/block_blockgroup_tract.csv")
    if block_bg_path.exists():
        df = pd.read_csv(block_bg_path)
        df["Block"] = df["Block"].fillna(0).astype("int64")
        gdf = gdf.merge(df, how="left", left_on="geoid10", right_on="Block")
    else:
        # Fallback: derive BlockGroup from Block ID (remove last 3 digits)
        gdf["BlockGroup"] = (gdf["geoid10"] // 1000).astype("int64")

    # Keep only needed columns
    gdf = gdf[["BlockGroup", "geometry"]].copy()
    gdf["BlockGroup"] = gdf["BlockGroup"].fillna(0).astype("int64")

    # Filter out BlockGroup 0 (unassigned blocks)
    gdf = gdf[gdf["BlockGroup"] != 0]

    # Dissolve by BlockGroup to combine blocks
    gdf = gdf.dissolve(by="BlockGroup", as_index=False)

    # Save
    GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(GEOJSON_PATH, driver="GeoJSON")
    print(f"Saved GeoJSON to {GEOJSON_PATH}")
    return gdf


def load_geojson() -> dict:
    """Load GeoJSON, converting if needed."""
    global _geojson_cache
    if _geojson_cache is not None:
        return _geojson_cache

    if not GEOJSON_PATH.exists():
        convert_shapefile_to_geojson()

    with open(GEOJSON_PATH, "r") as f:
        _geojson_cache = json.load(f)

    return _geojson_cache


def get_solution_space_stats() -> dict:
    """
    Get statistics for core metrics across all solutions in the solution space.

    Returns dict mapping metric column name to:
    - min: minimum value
    - max: maximum value
    - p10: 10th percentile
    - p25: 25th percentile
    - p50: median (50th percentile)
    - p75: 75th percentile
    - p90: 90th percentile
    - direction: 'minimize' or 'maximize'
    - display_name: user-friendly name
    - description: brief description
    - category: metric category
    """
    global _solution_space_stats_cache
    if _solution_space_stats_cache is not None:
        return _solution_space_stats_cache

    # Load all solutions
    all_solutions = load_solutions(CSV_PATH)

    # Filter to only valid solutions (OPTIMAL or FEASIBLE)
    if 'status' in all_solutions.columns:
        valid_solutions = all_solutions[
            all_solutions['status'].isin(['OPTIMAL', 'FEASIBLE'])
        ]
    else:
        valid_solutions = all_solutions

    stats = {}
    for metric in CORE_METRICS:
        col = metric.column
        if col not in valid_solutions.columns:
            continue

        values = valid_solutions[col].dropna()
        if len(values) == 0:
            continue

        stats[col] = {
            'min': float(values.min()),
            'max': float(values.max()),
            'p10': float(values.quantile(0.10)),
            'p25': float(values.quantile(0.25)),
            'p50': float(values.quantile(0.50)),
            'p75': float(values.quantile(0.75)),
            'p90': float(values.quantile(0.90)),
            'direction': metric.direction,
            'display_name': metric.display_name,
            'description': metric.description,
            'category': metric.category,
        }

    _solution_space_stats_cache = stats
    return stats


def _interpolate_percentile(value: float, stats: dict) -> float:
    """Interpolate where a value falls in the distribution (0-100 raw percentile)."""
    min_val = stats['min']
    max_val = stats['max']
    p10 = stats['p10']
    p25 = stats['p25']
    p50 = stats['p50']
    p75 = stats['p75']
    p90 = stats['p90']

    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 100.0

    if value <= p10:
        return 10 * (value - min_val) / (p10 - min_val) if p10 != min_val else 0.0
    if value <= p25:
        return 10 + 15 * (value - p10) / (p25 - p10) if p25 != p10 else 10.0
    if value <= p50:
        return 25 + 25 * (value - p25) / (p50 - p25) if p50 != p25 else 25.0
    if value <= p75:
        return 50 + 25 * (value - p50) / (p75 - p50) if p75 != p50 else 50.0
    if value <= p90:
        return 75 + 15 * (value - p75) / (p90 - p75) if p90 != p75 else 75.0
    return 90 + 10 * (value - p90) / (max_val - p90) if max_val != p90 else 90.0


def _get_ranking_class(normalized_percentile: float) -> str:
    """Map a normalized percentile (higher=better) to a CSS ranking class."""
    if normalized_percentile >= 80:
        return 'excellent'
    if normalized_percentile >= 60:
        return 'good'
    if normalized_percentile >= 40:
        return 'average'
    if normalized_percentile >= 20:
        return 'below-avg'
    return 'poor'


def compute_percentile_ranks(metrics: dict) -> dict:
    """Compute normalized percentile ranks for a solution's metrics.

    Returns dict mapping metric column -> {percentile, ranking, display_name, category}
    where percentile is 0-100 (higher = better) and ranking is the CSS class.
    """
    stats = get_solution_space_stats()
    ranks = {}
    for col, stat in stats.items():
        if col not in metrics:
            continue
        value = metrics[col]
        if value is None:
            continue
        raw = _interpolate_percentile(value, stat)
        normalized = (100 - raw) if stat['direction'] == 'minimize' else raw
        ranks[col] = {
            'percentile': round(normalized),
            'ranking': _get_ranking_class(normalized),
            'display_name': stat['display_name'],
            'category': stat['category'],
        }
    return ranks


def get_clusters(n_clusters: int = 5) -> list[dict]:
    """
    Get clustered solutions with labels and representative paths.

    Returns list of dicts with:
    - id: cluster ID (1-indexed)
    - label: interpretable direction label
    - count: number of solutions in cluster
    - path: path to representative solution
    - metrics: dict of metric values for representative
    """
    global _clusters_cache
    if _clusters_cache is not None:
        return _clusters_cache

    # Load and process solutions
    all_solutions = load_solutions(CSV_PATH)
    normalized = normalize_metrics(all_solutions)
    pareto = compute_pareto_frontier(normalized)

    # Get original values for Pareto solutions
    pareto_original = all_solutions.loc[pareto.index].copy()

    if len(pareto_original) < n_clusters * 2:
        n_clusters = max(2, len(pareto_original) // 2)

    # Cluster
    vectors = vectorize_solutions(pareto_original)
    labels, centers = cluster_solutions(vectors, n_clusters)
    directions = compute_cluster_directions(vectors, centers)

    clusters = []
    metric_cols = get_metric_columns()

    for cluster_id in range(n_clusters):
        cluster_size = int((labels == cluster_id).sum())
        direction_info = directions[cluster_id]

        # Get representative solution
        rep_solution, _ = get_representative_solution(
            pareto_original, vectors, centers, labels, cluster_id
        )

        # Build metrics dict
        metrics = {}
        for name, config in METRIC_CONFIG.items():
            col = config["column"]
            metrics[name] = float(rep_solution[col])

        clusters.append({
            "id": cluster_id + 1,
            "label": direction_info["direction_label"],
            "count": cluster_size,
            "path": rep_solution["path"],
            "metrics": metrics,
        })

    _clusters_cache = clusters
    return clusters


def get_zone_color(zone_id: int) -> str:
    """Get color for a zone ID."""
    return zone_colors.get(zone_id, "#808080")


if __name__ == "__main__":
    # Generate GeoJSON on first run
    convert_shapefile_to_geojson()

    # Test clustering
    print("\nTesting clusters...")
    clusters = get_clusters()
    for c in clusters:
        print(f"Cluster {c['id']}: {c['label']} ({c['count']} solutions)")
