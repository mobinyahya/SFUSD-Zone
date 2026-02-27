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
    get_centroid_solution,
)
from Zone_Generation.Config.Constants import zone_colors
from Zone_Generation.Config.metrics_config import CORE_METRICS, ALL_METRICS, METRIC_BY_COLUMN
from LLM.exploration.filters import FilterState, FilterBounds, apply_filters, find_relaxation_needed

# Paths
CSV_PATH = Path("/share/data/school_choice/local_runs/kumar_website_test/new_benchmarks_test/summary.csv")
GRAPH_PATH = Path("/share/data/school_choice/Data/Computed/Graphs/BlockGroup_0.pickle")
SHAPEFILE_PATH = Path("/share/data/school_choice/shapefiles/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp")
GEOJSON_PATH = Path(__file__).parent.parent / "data" / "sf_blockgroups.geojson"

# Cache
_graph_cache = None
_geojson_cache = None
_school_locations_cache = None
_pareto_cache = None
_all_metrics_stats_cache = None
_pareto_percentiles_cache = None
_category_percentiles_cache = None


def load_graph() -> nx.Graph:
    """Load the BlockGroup graph and cache it."""
    global _graph_cache
    if _graph_cache is None:
        with open(GRAPH_PATH, "rb") as f:
            _graph_cache = pickle.load(f)
    return _graph_cache


def get_school_locations() -> list[dict]:
    """
    Extract school locations from the graph.

    Returns:
        List of dicts with school_id, name, lat, lon, category
    """
    global _school_locations_cache
    if _school_locations_cache is not None:
        return _school_locations_cache

    G = load_graph()
    school_data = G.graph.get('school_data', {})

    valid_capacity = {}
    try:
        from Zone_Generation.Config.Constants import get_dropbox_path
        csv_path = f"{get_dropbox_path(False)}/Data/Cleaned/stanford_capacities_12.23.21.csv"
        cap_df = pd.read_csv(csv_path)
        for _, row in cap_df.iterrows():
            sch_num = int(row['SchNum'])
            code = str(row['PathwayCode'])
            cap = int(row.get('Scenario_A_Capacity', 0))
            if sch_num not in valid_capacity:
                valid_capacity[sch_num] = {}
            if code not in valid_capacity[sch_num]:
                valid_capacity[sch_num][code] = 0
            valid_capacity[sch_num][code] += cap
    except Exception as e:
        print(f"Failed to load capacity data: {e}")

    schools = []
    for school_id, school_info in school_data.items():
        # Only include schools with valid lat/lon
        lat = school_info.get('lat')
        lon = school_info.get('lon')
        if lat is not None and lon is not None:
            programs_dict = valid_capacity.get(int(school_id), {})
            total_cap = sum(programs_dict.values())
            schools.append({
                'school_id': school_id,
                'name': school_info.get('school_name', f'School {school_id}'),
                'lat': float(lat),
                'lon': float(lon),
                'category': school_info.get('category', 'Unknown'),
                'total_capacity': total_cap,
                'programs': programs_dict
            })

    _school_locations_cache = schools
    return schools


def get_node_to_blockgroup_map(G: nx.Graph) -> dict[int, int]:
    """Build mapping from graph node ID to BlockGroup ID."""
    return {node: data["area_id"] for node, data in G.nodes(data=True)}


def load_solution_result(solution_path: str) -> dict:
    """
    Load result.json from a solution directory.

    Args:
        solution_path: Path to solution folder containing result.json

    Returns:
        Dict with keys: status, metrics, zone_data, total_wall_time, etc.
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
        - avg_math_score: average math test score
        - avg_eng_score: average English test score
        - avg_closest_school_distance: average distance to closest school
        - schools_in_attendance_area: number of schools in zone
        - avg_max_utility: average maximum utility
        - avg_logsum_utility: average logsum utility
    """
    try:
        result = load_solution_result(solution_path)
        # result.json has zone_data with string keys, convert to int
        zone_data = result.get("zone_data", {})

        # Quality metric fields where 0.0 means "no data" (not a real value).
        # Scores like avg_math_score are in the 2400+ range; 0 means the zone has
        # no schools with score data. Convert to None so the frontend can distinguish
        # "no data" from an actual zero. This also handles older result.json files
        # that stored 0.0 instead of null for missing values.
        _QUALITY_ZERO_MEANS_NO_DATA = {
            'avg_math_score', 'avg_eng_score',
        }

        # Normalize field names for frontend compatibility
        normalized_data = {}
        for zone_id, data in zone_data.items():
            zone_dict = data.copy()
            # Frontend expects FRL_pct in 0-100 range (uppercase FRL)
            # result.json has frl_pct in 0-1 range (lowercase frl)
            if "frl_pct" in zone_dict:
                zone_dict["FRL_pct"] = zone_dict["frl_pct"] * 100
            # Convert quality 0.0 → None so frontend knows there is no data
            for field in _QUALITY_ZERO_MEANS_NO_DATA:
                if zone_dict.get(field) == 0:
                    zone_dict[field] = None
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
    block_bg_path = Path("/share/data/school_choice/Zones/Optimization/block_blockgroup_tract.csv")
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


def precompute_pareto_percentiles() -> dict:
    """Pre-compute true empirical percentiles for all Pareto solutions.

    Called once (lazy, on first access). For each core metric column, uses
    pandas rank(pct=True) to compute the true empirical percentile for every
    solution. For 'minimize' metrics, the percentile is inverted so that
    higher percentile always means 'better'.

    Returns dict mapping solution_path -> {metric_col -> {percentile, raw_value,
    ranking, display_name, category}}.
    """
    global _pareto_percentiles_cache
    if _pareto_percentiles_cache is not None:
        return _pareto_percentiles_cache

    pareto = get_pareto_solutions()
    cache = {}

    # Pre-compute ranks for each metric column
    metric_ranks = {}
    for metric in CORE_METRICS:
        if metric.direction is None:
            continue
        col = metric.column
        if col not in pareto.columns:
            continue
        values = pareto[col]
        raw_pct = values.rank(pct=True) * 100
        if metric.direction == 'minimize':
            raw_pct = 100 - raw_pct
        metric_ranks[col] = raw_pct

    # Build per-solution cache
    for idx, row in pareto.iterrows():
        path = row.get('path', '')
        if not path:
            continue
        solution_ranks = {}
        for metric in CORE_METRICS:
            col = metric.column
            if col not in metric_ranks:
                continue
            raw_value = row[col]
            if pd.isna(raw_value):
                continue
            pct = round(metric_ranks[col].loc[idx])
            solution_ranks[col] = {
                'percentile': pct,
                'raw_value': float(raw_value),
                'ranking': _get_ranking_class(pct),
                'display_name': metric.display_name,
                'category': metric.category,
            }
        cache[path] = solution_ranks

    _pareto_percentiles_cache = cache
    return cache


def precompute_category_percentiles() -> dict:
    """Pre-compute category-level percentiles for all Pareto solutions.

    For each category (diversity, distance, programs, quality), computes the
    average of the individual metric percentiles, then takes the percentile
    rank of that average across all solutions. This gives a true percentile
    for each category rather than a raw average of percentiles.

    These are display-only metrics and must NOT be used for Pareto filtering.

    Returns dict mapping solution_path -> {category_short -> percentile (0-100)}.
    """
    global _category_percentiles_cache
    if _category_percentiles_cache is not None:
        return _category_percentiles_cache

    per_solution_ranks = precompute_pareto_percentiles()

    # Map metric category to short display key
    CATEGORY_SHORT = {
        'diversity': 'Div',
        'distance': 'Dist',
        'programs': 'Prog',
        'quality': 'Perf',
        'structure': 'Struct',
    }

    # Group core directed metric columns by category
    category_metrics = {}
    for metric in CORE_METRICS:
        if metric.direction is None:
            continue
        short = CATEGORY_SHORT.get(metric.category)
        if short:
            category_metrics.setdefault(short, []).append(metric.column)

    # Step 1: Compute raw average-of-percentiles per category per solution
    solution_cat_avgs = {}  # {path: {cat_short: avg_percentile}}
    for path, ranks in per_solution_ranks.items():
        cat_avgs = {}
        for cat_short, columns in category_metrics.items():
            pcts = [ranks[col]['percentile'] for col in columns if col in ranks]
            if pcts:
                cat_avgs[cat_short] = sum(pcts) / len(pcts)
        solution_cat_avgs[path] = cat_avgs

    # Step 2: For each category, rank the averages across all solutions
    all_paths = list(solution_cat_avgs.keys())
    cache = {path: {} for path in all_paths}

    for cat_short in category_metrics:
        # Collect all average values for this category
        values = []
        paths_with_values = []
        for path in all_paths:
            avg = solution_cat_avgs[path].get(cat_short)
            if avg is not None:
                values.append(avg)
                paths_with_values.append(path)

        if not values:
            continue

        # Compute percentile rank of each average
        series = pd.Series(values, index=paths_with_values)
        ranked = series.rank(pct=True) * 100

        for path in paths_with_values:
            cache[path][cat_short] = round(ranked[path])

    _category_percentiles_cache = cache
    return cache


def compute_percentile_ranks(metrics: dict, solution_path: str = None) -> dict:
    """Compute normalized percentile ranks for a solution's metrics.

    If solution_path is provided and found in the precomputed Pareto cache,
    returns the cached true empirical percentiles (with raw_value included).
    Otherwise falls back to rank-based computation against the Pareto set.

    Returns dict mapping metric column -> {percentile, raw_value, ranking,
    display_name, category} where percentile is 0-100 (higher = better).
    """
    # Try precomputed cache first
    if solution_path:
        cache = precompute_pareto_percentiles()
        if solution_path in cache:
            return cache[solution_path]

    # Fallback: compute empirical percentile against Pareto set
    pareto = get_pareto_solutions()
    ranks = {}
    for metric in CORE_METRICS:
        if metric.direction is None:
            continue
        col = metric.column
        if col not in metrics or col not in pareto.columns:
            continue
        value = metrics[col]
        if value is None:
            continue

        all_values = pareto[col].dropna()
        if len(all_values) == 0:
            continue

        raw_pct = (all_values <= value).sum() / len(all_values) * 100
        if metric.direction == 'minimize':
            normalized = 100 - raw_pct
        else:
            normalized = raw_pct

        ranks[col] = {
            'percentile': round(normalized),
            'raw_value': float(value),
            'ranking': _get_ranking_class(normalized),
            'display_name': metric.display_name,
            'category': metric.category,
        }
    return ranks


def get_category_percentiles(solution_path: str = None, percentile_ranks: dict = None) -> dict:
    """Get category-level percentiles for a solution.

    If solution_path is in the precomputed cache, returns cached values.
    Otherwise computes from the given percentile_ranks using a fallback
    that ranks the average against all Pareto solutions.

    Returns dict mapping category short name -> percentile (0-100).
    """
    # Try precomputed cache first
    if solution_path:
        cache = precompute_category_percentiles()
        if solution_path in cache:
            return cache[solution_path]

    # Fallback: compute from percentile_ranks against Pareto distribution
    if not percentile_ranks:
        return {}

    CATEGORY_SHORT = {
        'diversity': 'Div',
        'distance': 'Dist',
        'programs': 'Prog',
        'quality': 'Perf',
        'structure': 'Struct',
    }

    # Group directed metrics by category
    category_metrics = {}
    for metric in CORE_METRICS:
        if metric.direction is None:
            continue
        short = CATEGORY_SHORT.get(metric.category)
        if short:
            category_metrics.setdefault(short, []).append(metric.column)

    # Compute this solution's category averages
    result = {}
    pareto_cache = precompute_category_percentiles()

    for cat_short, columns in category_metrics.items():
        pcts = [percentile_ranks[col]['percentile'] for col in columns
                if col in percentile_ranks]
        if not pcts:
            continue
        avg = sum(pcts) / len(pcts)

        # Rank against all Pareto solutions' category averages
        all_cat_values = [v.get(cat_short, 0) for v in precompute_pareto_percentiles().values()]
        # Get all raw category averages to rank against
        all_raw_avgs = []
        for path_ranks in precompute_pareto_percentiles().values():
            cat_pcts = [path_ranks[col]['percentile'] for col in columns if col in path_ranks]
            if cat_pcts:
                all_raw_avgs.append(sum(cat_pcts) / len(cat_pcts))

        if all_raw_avgs:
            rank_pct = sum(1 for v in all_raw_avgs if v <= avg) / len(all_raw_avgs) * 100
            result[cat_short] = round(rank_pct)
        else:
            result[cat_short] = round(avg)

    return result


def get_zone_color(zone_id: int) -> str:
    """Get color for a zone ID."""
    return zone_colors.get(zone_id, "#808080")


# ============================================================================
# Admin console helpers
# ============================================================================

def get_pareto_solutions() -> pd.DataFrame:
    """Load solutions, compute Pareto frontier, cache and return original-scale Pareto set."""
    global _pareto_cache
    if _pareto_cache is not None:
        return _pareto_cache

    all_solutions = load_solutions(CSV_PATH)
    all_solutions = all_solutions.dropna(subset=[m.column for m in ALL_METRICS if m.column in all_solutions.columns])
    all_solutions = all_solutions.drop_duplicates(subset="path")
    normalized = normalize_metrics(all_solutions)
    pareto_norm = compute_pareto_frontier(normalized)
    _pareto_cache = all_solutions.loc[pareto_norm.index].copy()
    return _pareto_cache


def get_all_metrics_stats() -> dict:
    """Like get_solution_space_stats but for ALL metrics (not just core)."""
    global _all_metrics_stats_cache
    if _all_metrics_stats_cache is not None:
        return _all_metrics_stats_cache

    pareto = get_pareto_solutions()
    stats = {}
    for metric in ALL_METRICS:
        col = metric.column
        if col not in pareto.columns:
            continue
        values = pareto[col].dropna()
        if len(values) == 0:
            continue
        stats[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "p25": float(values.quantile(0.25)),
            "p50": float(values.quantile(0.50)),
            "p75": float(values.quantile(0.75)),
            "direction": metric.direction,
            "display_name": metric.display_name,
            "description": metric.description,
            "category": metric.category,
            "is_core": metric.is_core,
            "short_name": metric.short_name or metric.display_name[:4],
        }

    _all_metrics_stats_cache = stats
    return stats


def filter_and_centroid(bounds: dict) -> dict:
    """
    Apply filter bounds to Pareto solutions and return centroid + feasible stats.

    Args:
        bounds: {metric_column: {min_bound, max_bound}} where values can be None.

    Returns dict with solution_count, centroid_path, centroid_metrics, feasible_stats.
    """
    pareto = get_pareto_solutions()

    fs = FilterState()
    for col, b in bounds.items():
        metric = METRIC_BY_COLUMN.get(col)
        if not metric:
            continue
        fb = FilterBounds(
            min_bound=b.get("min_bound"),
            max_bound=b.get("max_bound"),
        )
        fs.bounds[metric.display_name] = fb

    filtered = apply_filters(pareto, fs)

    result = {"solution_count": len(filtered), "total_pareto": len(pareto)}

    if len(filtered) == 0:
        result["centroid_path"] = None
        result["centroid_metrics"] = None
        result["feasible_stats"] = {}
        result["available_stats"] = {}
        return result

    norm_filtered = normalize_metrics(filtered)
    centroid_row, _ = get_centroid_solution(filtered, norm_filtered)

    result["centroid_path"] = centroid_row["path"]
    result["centroid_metrics"] = {
        m.column: float(centroid_row[m.column])
        for m in ALL_METRICS if m.column in centroid_row.index
    }

    fstats = {}
    for m in ALL_METRICS:
        col = m.column
        if col not in filtered.columns:
            continue
        vals = filtered[col].dropna()
        if len(vals) == 0:
            continue
        fstats[col] = {"min": float(vals.min()), "max": float(vals.max())}
    result["feasible_stats"] = fstats

    # Available stats: for each active filter, compute metric range without
    # that filter (but with all others). Tells frontend how far loosening can go.
    active_cols = [
        c for c, b in bounds.items()
        if b.get("min_bound") is not None or b.get("max_bound") is not None
    ]
    available = {}
    for target_col in active_cols:
        target_metric = METRIC_BY_COLUMN.get(target_col)
        if not target_metric:
            continue
        relaxed_fs = FilterState()
        for c, b in bounds.items():
            if c == target_col:
                continue
            m = METRIC_BY_COLUMN.get(c)
            if not m:
                continue
            relaxed_fs.bounds[m.display_name] = FilterBounds(
                min_bound=b.get("min_bound"),
                max_bound=b.get("max_bound"),
            )
        relaxed = apply_filters(pareto, relaxed_fs)
        if len(relaxed) > 0 and target_col in relaxed.columns:
            vals = relaxed[target_col].dropna()
            if len(vals) > 0:
                available[target_col] = {"min": float(vals.min()), "max": float(vals.max())}
    result["available_stats"] = available

    return result


def suggest_relaxation(bounds: dict) -> dict:
    """Find minimal relaxations to restore feasibility."""
    pareto = get_pareto_solutions()

    fs = FilterState()
    for col, b in bounds.items():
        metric = METRIC_BY_COLUMN.get(col)
        if not metric:
            continue
        fb = FilterBounds(min_bound=b.get("min_bound"), max_bound=b.get("max_bound"))
        fs.bounds[metric.display_name] = fb

    suggestions_by_name = find_relaxation_needed(pareto, fs)

    # Convert display_name keys back to column keys
    result = {}
    for display_name, bound_val in suggestions_by_name.items():
        for m in ALL_METRICS:
            if m.display_name == display_name:
                result[m.column] = bound_val
                break
    return result


if __name__ == "__main__":
    convert_shapefile_to_geojson()
