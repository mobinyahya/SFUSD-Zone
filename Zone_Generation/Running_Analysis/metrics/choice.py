"""
Choice/utility metrics for zoning.

Uses the UtilityEvaluator to compute max and logsum utilities.
"""

import functools
import networkx as nx

from Zone_Generation.Optimization.utility_evaluation import UtilityEvaluator
from Zone_Generation.Config.Constants import get_sfusd_path, get_dropbox_path


@functools.lru_cache(maxsize=1)
def get_utility_evaluator(utility_path: str, student_path: str) -> UtilityEvaluator:
    """Cached factory for UtilityEvaluator."""
    return UtilityEvaluator(utility_path, student_path)


def compute_choice_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    config: dict | None = None
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute choice/utility metrics.
    
    Args:
        zone_dict: Area to zone mapping
        G: Graph with school_ids per node
        config: Optional config with paths and settings
    
    Returns:
        Tuple of (aggregated_metrics, per_zone_utility_data)
    """
    config = config or {}
    is_local = config.get('is_local', False)
    
    # Default paths
    sfusd_path = get_sfusd_path(is_local)
    utility_path = config.get(
        'utility_path',
        f"{sfusd_path}/simulation-files/choice-model/estimates_2324_exp8_0514.csv"
    )
    student_path = config.get(
        'student_path',
        f"{sfusd_path}/Data/Cleaned/r1_filter_student_without_specialprogs_2324.csv"
    )
    
    evaluator = get_utility_evaluator(utility_path, student_path)
    
    # Compute max utility
    max_result = evaluator.evaluate(zone_dict, G, method='max')
    max_utilities = max_result['student_utilities']
    
    # Compute logsum utility
    logsum_result = evaluator.evaluate(zone_dict, G, method='logsum')
    logsum_utilities = logsum_result['student_utilities']
    
    # Per-zone aggregation
    per_zone_data = {}
    
    for zone_id in max_utilities['assigned_zone'].unique():
        zone_mask = max_utilities['assigned_zone'] == zone_id
        
        zone_max_utils = max_utilities.loc[zone_mask, 'utility']
        zone_logsum_utils = logsum_utilities.loc[zone_mask, 'utility']
        
        # Filter out invalid values (-inf)
        valid_max = zone_max_utils[zone_max_utils > -1e9]
        valid_logsum = zone_logsum_utils[zone_logsum_utils > -1e9]
        
        per_zone_data[zone_id] = {
            'avg_max_utility': valid_max.mean() if len(valid_max) > 0 else 0.0,
            'avg_logsum_utility': valid_logsum.mean() if len(valid_logsum) > 0 else 0.0
        }
    
    # Global averages
    valid_max_all = max_utilities['utility'][max_utilities['utility'] > -1e9]
    valid_logsum_all = logsum_utilities['utility'][logsum_utilities['utility'] > -1e9]
    
    aggregated = {
        'avg_max_utility': valid_max_all.mean() if len(valid_max_all) > 0 else 0.0,
        'avg_logsum_utility': valid_logsum_all.mean() if len(valid_logsum_all) > 0 else 0.0
    }
    
    return aggregated, per_zone_data
