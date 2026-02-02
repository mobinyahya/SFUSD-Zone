"""
Distance metrics for zoning.

Includes:
- Average closest school distance within zone (bug fix from original)
- Schools in attendance area metric
"""

import networkx as nx


def compute_distance_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_blocks: dict[int, list[int]],
    zone_schools: dict[int, list[int]]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute distance-related metrics.
    
    Args:
        zone_dict: Area to zone mapping
        G: Graph with distance_dict and school_data
        zone_blocks: Zone to list of blocks mapping
        zone_schools: Zone to list of school IDs mapping
    
    Returns:
        Tuple of (aggregated_metrics, per_zone_distance_data)
    """
    distance_dict = G.graph.get('distance_dict', {})
    school_data = G.graph.get('school_data', {})
    
    per_zone_data = {}
    all_avg_distances = []
    all_aa_counts = []
    
    # Build school_node lookup: school_id -> node containing that school
    school_to_node = {}
    for node in G.nodes():
        for sid in G.nodes[node].get('school_ids', []):
            school_to_node[sid] = node
    
    for zone_id, blocks in zone_blocks.items():
        schools_in_zone = zone_schools.get(zone_id, [])
        
        # 1. Average closest school distance WITHIN ZONE
        if schools_in_zone and blocks:
            total_distance = 0.0
            count = 0
            
            for block in blocks:
                if block not in distance_dict:
                    continue
                
                # Find closest school in this zone
                min_dist = float('inf')
                for school_id in schools_in_zone:
                    school_node = school_to_node.get(school_id)
                    if school_node is not None and school_node in distance_dict.get(block, {}):
                        dist = distance_dict[block][school_node]
                        if dist < min_dist:
                            min_dist = dist
                
                if min_dist < float('inf'):
                    total_distance += min_dist
                    count += 1
            
            avg_closest_dist = total_distance / count if count > 0 else 0.0
        else:
            avg_closest_dist = 0.0
        
        # 2. Count schools in attendance area
        # For each school in zone, check if zone's blocks are in its attendance area
        schools_in_aa = 0
        
        # Get the attendance areas of schools in this zone
        zone_attendance_areas = set()
        for sid in schools_in_zone:
            if sid in school_data:
                aa = school_data[sid].get('attendance_area')
                if aa:
                    zone_attendance_areas.add(aa)
        
        # Count schools whose attendance area matches this zone's blocks
        for sid in schools_in_zone:
            if sid in school_data:
                school_aa = school_data[sid].get('attendance_area')
                # A school is "in attendance area" if its AA matches the zone
                # This check considers if the school's AA is one of the schools in zone
                if school_aa in [s for s in schools_in_zone]:
                    schools_in_aa += 1
        
        per_zone_data[zone_id] = {
            'avg_closest_school_distance': avg_closest_dist,
            'schools_in_attendance_area': schools_in_aa
        }
        
        all_avg_distances.append(avg_closest_dist)
        all_aa_counts.append(schools_in_aa)
    
    # Aggregate metrics
    aggregated = {
        'avg_closest_zone_school_distance': (
            sum(all_avg_distances) / len(all_avg_distances) 
            if all_avg_distances else 0.0
        ),
        'avg_schools_in_attendance_area': (
            sum(all_aa_counts) / len(all_aa_counts) 
            if all_aa_counts else 0.0
        )
    }
    
    return aggregated, per_zone_data
