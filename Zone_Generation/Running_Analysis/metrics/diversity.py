"""
Diversity and demographic metrics for zoning.
"""

import networkx as nx

from Zone_Generation.Config.Constants import AREA_ETHNICITIES


def compute_diversity_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_blocks: dict[int, list[int]]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute demographic deviation metrics.
    
    Returns:
        Tuple of (aggregated_metrics, per_zone_demographics)
    """
    # Get area-wide averages from graph
    area_frl_pct = G.graph['F']
    area_ethnicities = {eth: G.graph['R'][eth] for eth in AREA_ETHNICITIES}
    
    # Track deviations per zone
    deviations = {}
    per_zone_demos = {}
    
    for zone_id, blocks in zone_blocks.items():
        zone_demo = {'ge_students': 0.0, 'FRL': 0.0}
        for ethnicity in AREA_ETHNICITIES:
            zone_demo[ethnicity] = 0.0
        
        for block in blocks:
            if block not in G.nodes:
                continue
            zone_demo['ge_students'] += G.nodes[block]['ge_students']
            zone_demo['FRL'] += G.nodes[block]['FRL']
            for ethnicity in AREA_ETHNICITIES:
                zone_demo[ethnicity] += G.nodes[block][ethnicity]
        
        # Store per-zone data
        if zone_demo['ge_students'] > 0:
            frl_pct = zone_demo['FRL'] / zone_demo['ge_students']
            ethnicity_pcts = {
                eth: zone_demo[eth] / zone_demo['ge_students']
                for eth in AREA_ETHNICITIES
            }
        else:
            frl_pct = 0.0
            ethnicity_pcts = {eth: 0.0 for eth in AREA_ETHNICITIES}
        
        per_zone_demos[zone_id] = {
            'ge_students': zone_demo['ge_students'],
            'frl_pct': frl_pct,
            'ethnicity_pcts': ethnicity_pcts
        }
        
        # Compute deviations from area averages
        if zone_demo['ge_students'] > 0:
            if 'FRL' not in deviations:
                deviations['FRL'] = []
            frl_deviation = abs(frl_pct - area_frl_pct)
            deviations['FRL'].append(frl_deviation)
            
            for ethnicity in AREA_ETHNICITIES:
                if ethnicity not in deviations:
                    deviations[ethnicity] = []
                eth_deviation = abs(ethnicity_pcts[ethnicity] - area_ethnicities[ethnicity])
                deviations[ethnicity].append(eth_deviation)
    
    # Average deviations across zones
    avg_deviations = {}
    for key, dev_list in deviations.items():
        if dev_list:
            avg_deviations[key] = sum(dev_list) / len(dev_list)
    
    return avg_deviations, per_zone_demos


def compute_seat_disparity(
    zone_blocks: dict[int, list[int]],
    G: nx.Graph
) -> float:
    """
    Compute average % shortage/overage across zones.
    """
    if not zone_blocks:
        return 0.0
    
    total_diff = 0.0
    valid_zones = 0
    
    for zone_id, blocks in zone_blocks.items():
        seats = 0.0
        students = 0.0
        for block in blocks:
            if block not in G.nodes:
                continue
            seats += G.nodes[block]['ge_capacity']
            students += G.nodes[block]['ge_students']
        
        if students == 0:
            continue
        
        diff = abs(seats - students) / students
        total_diff += diff
        valid_zones += 1
    
    return total_diff / valid_zones if valid_zones > 0 else 0.0
