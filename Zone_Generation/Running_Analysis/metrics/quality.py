"""
School quality metrics for zoning.

Uses school data from the graph to compute quality metrics per zone.
"""

import networkx as nx

from Zone_Generation.Running_Analysis.metrics.base import COLOR_TO_INDEX


def compute_quality_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    zone_schools: dict[int, list[int]]
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute school quality metrics.
    
    Args:
        zone_dict: Area to zone mapping
        G: Graph with school_data attribute
        zone_schools: Zone to list of school IDs mapping
    
    Returns:
        Tuple of (aggregated_metrics, per_zone_quality_data)
    """
    school_data = G.graph.get('school_data', {})
    
    per_zone_data = {}
    
    all_gs_ratings = []
    all_math_scores = []
    all_eng_scores = []
    all_suspension_indices = []
    
    for zone_id, schools in zone_schools.items():
        # Compute weighted averages for schools in zone
        total_cap = 0.0
        weighted_gs = 0.0
        weighted_math = 0.0
        weighted_eng = 0.0
        weighted_susp = 0.0
        
        for sid in schools:
            if sid not in school_data:
                continue
            
            sdata = school_data[sid]
            cap = sdata.get('cap_lb', 0) or 0
            if cap <= 0:
                cap = 1  # Fallback weight
            
            total_cap += cap
            
            # GreatSchools rating
            gs = sdata.get('greatschools_rating')
            if gs and gs > 0:
                weighted_gs += gs * cap
            
            # Math scores
            math = sdata.get('math_scores_1819')
            if math and math > 0:
                weighted_math += math * cap
            
            # English scores
            eng = sdata.get('eng_scores_1819')
            if eng and eng > 0:
                weighted_eng += eng * cap
            
            # Suspension color -> index
            susp_color = sdata.get('suspension_color', '')
            susp_idx = COLOR_TO_INDEX.get(susp_color, 0)
            if susp_idx > 0:
                weighted_susp += susp_idx * cap
        
        # Compute averages
        if total_cap > 0:
            avg_gs = weighted_gs / total_cap
            avg_math = weighted_math / total_cap
            avg_eng = weighted_eng / total_cap
            avg_susp = weighted_susp / total_cap
        else:
            avg_gs = 0.0
            avg_math = 0.0
            avg_eng = 0.0
            avg_susp = 0.0
        
        per_zone_data[zone_id] = {
            'avg_greatschools_rating': avg_gs,
            'avg_math_score': avg_math,
            'avg_eng_score': avg_eng,
            'avg_suspension_index': avg_susp
        }
        
        if avg_gs > 0:
            all_gs_ratings.append(avg_gs)
        if avg_math > 0:
            all_math_scores.append(avg_math)
        if avg_eng > 0:
            all_eng_scores.append(avg_eng)
        if avg_susp > 0:
            all_suspension_indices.append(avg_susp)
    
    # Aggregate across zones
    aggregated = {
        'avg_greatschools_rating': (
            sum(all_gs_ratings) / len(all_gs_ratings) 
            if all_gs_ratings else 0.0
        ),
        'avg_math_score': (
            sum(all_math_scores) / len(all_math_scores) 
            if all_math_scores else 0.0
        ),
        'avg_eng_score': (
            sum(all_eng_scores) / len(all_eng_scores) 
            if all_eng_scores else 0.0
        ),
        'avg_suspension_index': (
            sum(all_suspension_indices) / len(all_suspension_indices) 
            if all_suspension_indices else 0.0
        )
    }
    
    return aggregated, per_zone_data
