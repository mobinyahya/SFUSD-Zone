"""
Diversity and demographic metrics for zoning.
"""

import math
import networkx as nx

from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.Config.metrics_config import MetricColumns


def compute_diversity_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_blocks: dict[int, list[int]]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute dissimilarity index for FRL and each ethnicity group.

    Dissimilarity index D for a group:
        D = Σ |n_i - T_i × (n / T)| / (2 × n)
    where n_i = group count in zone i, T_i = total students in zone i,
    n = total group students district-wide, T = total students district-wide.
    Range: 0 (perfectly integrated) to 1 (completely segregated).

    Per-zone demographics (frl_pct, ethnicity_pcts) are unchanged.

    Returns:
        Tuple of (aggregated_metrics, per_zone_demographics)
    """
    # Map AREA_ETHNICITIES node attributes to MetricColumns dissim keys
    eth_to_column = {
        "Ethnicity_Black_or_African_American": MetricColumns.BLACK_DISSIM,
        "Ethnicity_Hispanic/Latinx": MetricColumns.HISPANIC_DISSIM,
        "Ethnicity_White": MetricColumns.WHITE_DISSIM,
        "Ethnicity_Asian": MetricColumns.ASIAN_DISSIM,
    }

    # Accumulate per-zone counts
    per_zone_demos = {}
    zone_counts: list[dict] = []  # list of {ge_students, FRL, eth1, eth2, ...}

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

        # Per-zone proportions (unchanged)
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
            'ethnicity_pcts': ethnicity_pcts,
        }
        zone_counts.append(zone_demo)

    # District totals
    T = sum(z['ge_students'] for z in zone_counts)
    total_frl = sum(z['FRL'] for z in zone_counts)
    total_eth = {eth: sum(z[eth] for z in zone_counts) for eth in AREA_ETHNICITIES}

    # Dissimilarity index: D = Σ |n_i - T_i × (n/T)| / (2 × n)
    def _dissim(group_counts: list[float], zone_totals: list[float], n: float) -> float:
        if n <= 0 or T <= 0:
            return 0.0
        return sum(abs(ni - ti * (n / T)) for ni, ti in zip(group_counts, zone_totals)) / (2 * n)

    zone_totals = [z['ge_students'] for z in zone_counts]

    metrics: dict[str, float] = {}
    # FRL dissimilarity
    frl_counts = [z['FRL'] for z in zone_counts]
    metrics[MetricColumns.FRL_DISSIM] = _dissim(frl_counts, zone_totals, total_frl)

    # Per-ethnicity dissimilarity
    for eth, col in eth_to_column.items():
        eth_counts = [z[eth] for z in zone_counts]
        metrics[col] = _dissim(eth_counts, zone_totals, total_eth[eth])

    return metrics, per_zone_demos


def compute_seat_disparity(
    zone_blocks: dict[int, list[int]],
    G: nx.Graph
) -> tuple[float, dict[int, dict]]:
    """
    Compute average % shortage/overage across zones.

    Returns:
        Tuple of (solution_level_value, per_zone_data) where per_zone_data maps
        zone_id -> {'seat_disparity': float | None}
    """
    if not zone_blocks:
        return 0.0, {}

    total_diff = 0.0
    valid_zones = 0
    per_zone_data: dict[int, dict] = {}

    for zone_id, blocks in zone_blocks.items():
        seats = 0.0
        students = 0.0
        for block in blocks:
            if block not in G.nodes:
                continue
            seats += G.nodes[block]['ge_capacity']
            students += G.nodes[block]['ge_students']

        if students == 0:
            per_zone_data[zone_id] = {'seat_disparity': None, 'ge_capacity': seats}
            continue

        signed_diff = (seats - students) / students
        total_diff += abs(signed_diff)
        valid_zones += 1
        per_zone_data[zone_id] = {'seat_disparity': signed_diff, 'ge_capacity': seats}

    solution_value = total_diff / valid_zones if valid_zones > 0 else 0.0
    return solution_value, per_zone_data


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

    return {MetricColumns.THEIL_INDEX: theil_index}, per_zone_data
