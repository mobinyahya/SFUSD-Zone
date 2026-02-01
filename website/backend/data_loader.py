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

# Paths
CSV_PATH = Path("/home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/llm_bg_runs/recursive_metrics_flattened.csv")
GRAPH_PATH = Path("/home/kumarc/sfusd-local-data/zones/SFUSD/Optimization/Zones/Graphs/BlockGroup_0.pickle")
SHAPEFILE_PATH = Path("/share/data/school_choice/shapefiles/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp")
GEOJSON_PATH = Path(__file__).parent.parent / "data" / "sf_blockgroups.geojson"

# Cache
_graph_cache = None
_geojson_cache = None
_clusters_cache = None


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


def get_zone_demographics(bg_zone_dict: dict[int, int]) -> dict[int, dict]:
    """
    Aggregate demographics per zone from graph node attributes.

    Returns dict mapping zone_id to demographics dict with:
    - ge_students: total students
    - FRL: total FRL count (to be normalized)
    - Ethnicity_*: counts per ethnicity
    """
    G = load_graph()
    node_to_bg = get_node_to_blockgroup_map(G)
    bg_to_node = {v: k for k, v in node_to_bg.items()}

    zone_stats = {}

    for bg_id, zone_id in bg_zone_dict.items():
        if zone_id not in zone_stats:
            zone_stats[zone_id] = {
                "ge_students": 0,
                "FRL": 0,
            }
            for eth in AREA_ETHNICITIES:
                zone_stats[zone_id][eth] = 0

        node_id = bg_to_node.get(bg_id)
        if node_id is not None and node_id in G.nodes:
            node_data = G.nodes[node_id]
            zone_stats[zone_id]["ge_students"] += node_data.get("ge_students", 0)
            zone_stats[zone_id]["FRL"] += node_data.get("FRL", 0)
            for eth in AREA_ETHNICITIES:
                zone_stats[zone_id][eth] += node_data.get(eth, 0)

    # Normalize to percentages
    for zone_id, stats in zone_stats.items():
        total = stats["ge_students"]
        if total > 0:
            stats["FRL_pct"] = (stats["FRL"] / total) * 100
            for eth in AREA_ETHNICITIES:
                stats[f"{eth}_pct"] = (stats[eth] / total) * 100
        else:
            stats["FRL_pct"] = 0
            for eth in AREA_ETHNICITIES:
                stats[f"{eth}_pct"] = 0

    return zone_stats


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
