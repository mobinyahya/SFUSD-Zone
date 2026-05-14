"""
Distance metrics for zoning.

Includes:
- Average distance to any GE school in zone (mean of all pairwise distances)
- Average distance to farthest GE school in zone (mean of per-area max distances)
- Average number of out-of-zone GE schools within 0.5 miles
- Schools in attendance area metric
"""

import networkx as nx

from Zone_Generation.Config.metrics_config import MetricColumns

GE_PROXIMITY_RADIUS = 0.5  # miles, matches the in-zone GE proximity metric


def compute_distance_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_blocks: dict[int, list[int]],
    zone_schools: dict[int, list[int]],
    ge_schools: set[int]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute distance-related metrics.

    Args:
        zone_dict: Area to zone mapping
        G: Graph with distance_dict and school_data
        zone_blocks: Zone to list of blocks mapping
        zone_schools: Zone to list of school IDs mapping
        ge_schools: Set of school IDs that have a GE program

    Returns:
        Tuple of (aggregated_metrics, per_zone_distance_data)
    """
    distance_dict = G.graph.get('distance_dict', {})
    school_data = G.graph.get('school_data', {})

    per_zone_data = {}
    all_avg_any: list[float] = []
    all_avg_farthest: list[float] = []
    all_avg_out_of_zone: list[float] = []
    all_aa_counts: list[int] = []

    # Build school_id -> node lookup
    school_to_node: dict[int, int] = {}
    for node in G.nodes():
        for sid in G.nodes[node].get('school_ids', []):
            school_to_node[sid] = node

    # Build GE school -> zone lookup (for out-of-zone metric)
    ge_school_to_zone: dict[int, int] = {}
    for zone_id, schools in zone_schools.items():
        for sid in schools:
            if sid in ge_schools:
                ge_school_to_zone[sid] = zone_id

    for zone_id, blocks in zone_blocks.items():
        schools_in_zone = zone_schools.get(zone_id, [])
        ge_in_zone = [s for s in schools_in_zone if s in ge_schools]

        # --- Metric 1: avg distance to any in-zone GE school ---
        # --- Metric 2: avg distance to farthest in-zone GE school ---
        if ge_in_zone and blocks:
            sum_avg_any = 0.0
            sum_max_dist = 0.0
            count_blocks = 0

            for block in blocks:
                if block not in distance_dict:
                    continue

                dists = []
                for sid in ge_in_zone:
                    school_node = school_to_node.get(sid)
                    if school_node is not None and school_node in distance_dict[block]:
                        dists.append(distance_dict[block][school_node])

                if dists:
                    sum_avg_any += sum(dists) / len(dists)
                    sum_max_dist += max(dists)
                    count_blocks += 1

            avg_any = sum_avg_any / count_blocks if count_blocks > 0 else 0.0
            avg_farthest = sum_max_dist / count_blocks if count_blocks > 0 else 0.0
        else:
            avg_any = 0.0
            avg_farthest = 0.0

        # --- Metric 3: avg out-of-zone GE schools within 0.5 miles ---
        if blocks:
            total_out_of_zone = 0.0
            count_blocks_ooz = 0

            for block in blocks:
                if block not in distance_dict:
                    continue

                out_of_zone_count = 0
                for sid, sid_zone in ge_school_to_zone.items():
                    if sid_zone == zone_id:
                        continue
                    school_node = school_to_node.get(sid)
                    if school_node is not None and school_node in distance_dict[block]:
                        if distance_dict[block][school_node] <= GE_PROXIMITY_RADIUS:
                            out_of_zone_count += 1

                total_out_of_zone += out_of_zone_count
                count_blocks_ooz += 1

            avg_out_of_zone = total_out_of_zone / count_blocks_ooz if count_blocks_ooz > 0 else 0.0
        else:
            avg_out_of_zone = 0.0

        # --- Schools in attendance area (unchanged) ---
        zone_attendance_areas = set()
        for sid in schools_in_zone:
            if sid in school_data:
                aa = school_data[sid].get('attendance_area')
                if aa:
                    zone_attendance_areas.add(aa)

        schools_in_aa = 0
        for sid in schools_in_zone:
            if sid in school_data:
                school_aa = school_data[sid].get('attendance_area')
                if school_aa in [s for s in schools_in_zone]:
                    schools_in_aa += 1

        per_zone_data[zone_id] = {
            'avg_any_ge_school_distance': avg_any,
            'avg_farthest_ge_school_distance': avg_farthest,
            'avg_out_of_zone_ge_schools': avg_out_of_zone,
            'schools_in_attendance_area': schools_in_aa,
        }

        all_avg_any.append(avg_any)
        all_avg_farthest.append(avg_farthest)
        all_avg_out_of_zone.append(avg_out_of_zone)
        all_aa_counts.append(schools_in_aa)

    n = len(zone_blocks) if zone_blocks else 1
    aggregated = {
        MetricColumns.AVG_ANY_ZONE_GE_SCHOOL_DISTANCE: (
            sum(all_avg_any) / n if all_avg_any else 0.0
        ),
        MetricColumns.AVG_FARTHEST_ZONE_GE_SCHOOL_DISTANCE: (
            sum(all_avg_farthest) / n if all_avg_farthest else 0.0
        ),
        MetricColumns.AVG_OUT_OF_ZONE_GE_SCHOOLS: (
            sum(all_avg_out_of_zone) / n if all_avg_out_of_zone else 0.0
        ),
        MetricColumns.AVG_SCHOOLS_IN_ATTENDANCE_AREA: (
            sum(all_aa_counts) / n if all_aa_counts else 0.0
        ),
    }

    return aggregated, per_zone_data
