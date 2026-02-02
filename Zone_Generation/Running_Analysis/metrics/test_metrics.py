"""
Tests for the new metrics package.
"""

import json
import pickle
import sys
import os

# Ensure imports work
sys.path.insert(0, '/home/kumarc/sfusd/SFUSD-Zone')

from Zone_Generation.Running_Analysis.metrics import ZoneMetricsCalculator, MetricsResult, ZoneData
from Zone_Generation.Config.Constants import get_dropbox_path


def test_metrics_calculator():
    """Test the full metrics calculator with real data."""
    print("Loading graph...")
    graph_path = f'{get_dropbox_path(False)}/Optimization/Zones/Graphs/BlockGroup_0.pickle'
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Load test zone dict
    zd_path = '/home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/new_benchmarks_test/4-zone-rec-4/seed42/frl0.12_racial0.12/overage0.7_shortage0.15/BlockGroup_1-BlockGroup_0_tl_120-120/zone_dict_BlockGroup_0.json'
    print(f"Loading zone dict from {zd_path}...")
    with open(zd_path) as f:
        zone_dict = {int(k): v for k, v in json.load(f).items()}
    print(f"Zone dict loaded: {len(zone_dict)} areas, {len(set(zone_dict.values()))} zones")
    
    # Create calculator
    print("\nCreating ZoneMetricsCalculator...")
    calc = ZoneMetricsCalculator(zone_dict, G, {'is_local': False, 'compute_choice': False})
    
    # Compute all metrics (without choice for speed)
    print("Computing all metrics (excluding choice for speed)...")
    result = calc.compute_all(include_choice=False)
    
    print("\n" + "="*60)
    print("AGGREGATED METRICS:")
    print("="*60)
    for key, value in sorted(result.metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("PER-ZONE DATA (first 3 zones):")
    print("="*60)
    for zone_id, zone_data in list(result.zone_data.items())[:3]:
        print(f"\nZone {zone_id}:")
        zd = zone_data
        print(f"  Students: {zd.ge_students:.0f}")
        print(f"  FRL %: {zd.frl_pct:.3f}")
        print(f"  Programs: {zd.total_programs}")
        print(f"  Language immersion: {zd.language_immersion_count}")
        print(f"  Special ed: {zd.special_ed_count}")
        print(f"  Avg GS rating: {zd.avg_greatschools_rating:.2f}")
        print(f"  Avg math score: {zd.avg_math_score:.0f}")
        print(f"  Avg closest school dist: {zd.avg_closest_school_distance:.3f}")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_metrics_calculator()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
