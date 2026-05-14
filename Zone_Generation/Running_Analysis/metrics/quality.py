"""
School quality metrics for zoning.

Computes per-zone capacity-weighted average math/english scores,
then aggregates as Mean Absolute Deviation (MAD) across zones.
Lower MAD = more equitable quality distribution.
"""

import networkx as nx

from Zone_Generation.Config.metrics_config import MetricColumns


def compute_quality_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_schools: dict[int, list[int]]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute school quality metrics.
    
    Per-zone: capacity-weighted average math/english scores.
    Aggregate: mean absolute deviation of per-zone averages.
    
    Returns:
        Tuple of (aggregated_metrics, per_zone_quality_data)
    """
    school_data = G.graph.get('school_data', {})
    
    per_zone_data = {}
    zone_math_avgs = []
    zone_eng_avgs = []
    
    for zone_id, schools in zone_schools.items():
        weighted_math = 0.0
        weighted_eng = 0.0
        cap_math = 0.0
        cap_eng = 0.0
        
        for sid in schools:
            if sid not in school_data:
                continue
            
            sdata = school_data[sid]
            cap = sdata.get('cap_lb', 0) or 0
            if cap <= 0:
                cap = 1
            
            math = sdata.get('math_scores_1819')
            if math and math > 0:
                weighted_math += math * cap
                cap_math += cap
            
            eng = sdata.get('eng_scores_1819')
            if eng and eng > 0:
                weighted_eng += eng * cap
                cap_eng += cap
        
        avg_math = weighted_math / cap_math if cap_math > 0 else 0
        avg_eng = weighted_eng / cap_eng if cap_eng > 0 else 0
        
        per_zone_data[zone_id] = {
            'avg_math_score': avg_math,
            'avg_eng_score': avg_eng,
        }
        
        if avg_math > 0:
            zone_math_avgs.append(avg_math)
        if avg_eng > 0:
            zone_eng_avgs.append(avg_eng)
    
    def mad(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum(abs(v - mean) for v in values) / len(values)

    def val_range(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return max(values) - min(values)
    
    aggregated = {
        MetricColumns.MAD_MATH_SCORE: mad(zone_math_avgs),
        MetricColumns.MAD_ENG_SCORE: mad(zone_eng_avgs),
        MetricColumns.MATH_SCORE_RANGE: val_range(zone_math_avgs),
        MetricColumns.ENG_SCORE_RANGE: val_range(zone_eng_avgs),
    }
    
    return aggregated, per_zone_data
