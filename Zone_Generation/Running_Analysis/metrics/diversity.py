"""
Diversity and demographic metrics for zoning.
"""

import math
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


def compute_theil_index(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_blocks: dict[int, list[int]]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute Theil's Information Theory Index for ethnic segregation.

    The Theil Index measures segregation by comparing zone-level entropy
    to district-wide entropy:

        T = Σ (t_j / T) × (E - E_j) / E

    where:
        - T = total students, t_j = zone j students
        - E = district-wide Shannon entropy, E_j = zone j entropy

    Lower values indicate less segregation (zones match district composition).
    Range: 0 (no segregation) to ~1 (high segregation)

    Returns:
        Tuple of (aggregated_metrics, per_zone_data)
    """
    # 1. Calculate district-wide entropy E using 4 main ethnic groups
    district_eth_props = {eth: G.graph['R'][eth] for eth in AREA_ETHNICITIES}
    E_district = -sum(
        p * math.log(p) for p in district_eth_props.values() if p > 0
    )

    # 2. Calculate per-zone entropy and populations
    zone_entropies = {}
    zone_populations = {}
    total_students = 0.0

    for zone_id, blocks in zone_blocks.items():
        zone_demo = {eth: 0.0 for eth in AREA_ETHNICITIES}
        zone_demo['ge_students'] = 0.0

        for block in blocks:
            if block not in G.nodes:
                continue
            zone_demo['ge_students'] += G.nodes[block]['ge_students']
            for eth in AREA_ETHNICITIES:
                zone_demo[eth] += G.nodes[block][eth]

        zone_populations[zone_id] = zone_demo['ge_students']
        total_students += zone_demo['ge_students']

        # Zone entropy
        if zone_demo['ge_students'] > 0:
            zone_eth_props = {
                eth: zone_demo[eth] / zone_demo['ge_students']
                for eth in AREA_ETHNICITIES
            }
            zone_entropies[zone_id] = -sum(
                p * math.log(p) for p in zone_eth_props.values() if p > 0
            )
        else:
            zone_entropies[zone_id] = 0.0

    # 3. Compute Theil Index (student-weighted)
    theil_index = 0.0
    if E_district > 0 and total_students > 0:
        for zone_id in zone_blocks:
            t_j = zone_populations.get(zone_id, 0)
            E_j = zone_entropies.get(zone_id, 0)
            weight = t_j / total_students
            theil_index += weight * (E_district - E_j) / E_district

    # 4. Build per-zone data
    per_zone_data = {
        zone_id: {'entropy': zone_entropies.get(zone_id, 0.0)}
        for zone_id in zone_blocks
    }

    return {'theil_index': theil_index}, per_zone_data
