"""
Diversity and demographic metrics for zoning.
"""

import networkx as nx

from Zone_Generation.Config.Constants import AALPI_ETHNICITIES, AREA_ETHNICITIES
from Zone_Generation.Config.metrics_config import MetricColumns


def compute_diversity_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_blocks: dict[int, list[int]]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute mean absolute deviation (MAD) of zone-level group shares from the
    district-wide share, for FRL, each individual ethnicity, and AALPI
    (Black + Hispanic/Latinx + Pacific Islander).

        MAD_g = mean over non-empty zones of |zone_g_proportion - district_g_proportion|

    Lower = zones match the district composition for group g. Range 0-1.

    Returns:
        Tuple of (aggregated_metrics, per_zone_demographics) where per-zone
        demographics include frl_pct, ethnicity_pcts (per individual ethnicity),
        and aalpi_pct (combined AALPI share).
    """
    eth_to_column = {
        "Ethnicity_Black_or_African_American": MetricColumns.BLACK_MAD,
        "Ethnicity_Hispanic/Latinx": MetricColumns.HISPANIC_MAD,
        "Ethnicity_White": MetricColumns.WHITE_MAD,
        "Ethnicity_Asian": MetricColumns.ASIAN_MAD,
        "Ethnicity_PacificIslander": MetricColumns.PACIFIC_ISLANDER_MAD,
    }

    per_zone_demos = {}
    zone_counts: list[dict] = []

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

        zone_demo['AALPI'] = sum(zone_demo[eth] for eth in AALPI_ETHNICITIES)

        if zone_demo['ge_students'] > 0:
            frl_pct = zone_demo['FRL'] / zone_demo['ge_students']
            ethnicity_pcts = {
                eth: zone_demo[eth] / zone_demo['ge_students']
                for eth in AREA_ETHNICITIES
            }
            aalpi_pct = zone_demo['AALPI'] / zone_demo['ge_students']
        else:
            frl_pct = 0.0
            ethnicity_pcts = {eth: 0.0 for eth in AREA_ETHNICITIES}
            aalpi_pct = 0.0

        per_zone_demos[zone_id] = {
            'ge_students': zone_demo['ge_students'],
            'frl_pct': frl_pct,
            'ethnicity_pcts': ethnicity_pcts,
            'aalpi_pct': aalpi_pct,
        }
        zone_counts.append(zone_demo)

    # District-wide shares (student-weighted)
    T = sum(z['ge_students'] for z in zone_counts)
    district_frl_pct = (sum(z['FRL'] for z in zone_counts) / T) if T > 0 else 0.0
    district_eth_pct = {
        eth: G.graph['R'][eth] for eth in AREA_ETHNICITIES
    }
    district_aalpi_pct = sum(district_eth_pct[eth] for eth in AALPI_ETHNICITIES)

    nonempty = [z for z in zone_counts if z['ge_students'] > 0]

    def _mad(zone_props: list[float], district_prop: float) -> float:
        if not zone_props:
            return 0.0
        return sum(abs(p - district_prop) for p in zone_props) / len(zone_props)

    def _range(zone_props: list[float]) -> float:
        if len(zone_props) < 2:
            return 0.0
        return max(zone_props) - min(zone_props)

    eth_to_range_column = {
        "Ethnicity_Black_or_African_American": MetricColumns.BLACK_RANGE,
        "Ethnicity_Hispanic/Latinx": MetricColumns.HISPANIC_RANGE,
        "Ethnicity_White": MetricColumns.WHITE_RANGE,
        "Ethnicity_Asian": MetricColumns.ASIAN_RANGE,
        "Ethnicity_PacificIslander": MetricColumns.PACIFIC_ISLANDER_RANGE,
    }

    metrics: dict[str, float] = {}
    frl_props = [z['FRL'] / z['ge_students'] for z in nonempty]
    metrics[MetricColumns.FRL_MAD] = _mad(frl_props, district_frl_pct)
    metrics[MetricColumns.FRL_RANGE] = _range(frl_props)

    for eth, col in eth_to_column.items():
        eth_props = [z[eth] / z['ge_students'] for z in nonempty]
        metrics[col] = _mad(eth_props, district_eth_pct[eth])
        metrics[eth_to_range_column[eth]] = _range(eth_props)

    aalpi_props = [z['AALPI'] / z['ge_students'] for z in nonempty]
    metrics[MetricColumns.AALPI_MAD] = _mad(aalpi_props, district_aalpi_pct)
    metrics[MetricColumns.AALPI_RANGE] = _range(aalpi_props)

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
