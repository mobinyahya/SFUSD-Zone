"""
Program access metrics for zoning.

Tracks which programs are available in each zone.
"""

import functools
import pandas as pd
import networkx as nx

from Zone_Generation.Config.Constants import get_sfusd_path, PROGRAM_CATEGORIES
from Zone_Generation.Config.metrics_config import MetricColumns

LANGUAGE_PROGRAMS = set(PROGRAM_CATEGORIES["Language Programs"])
SPECIAL_EDUCATION = set(PROGRAM_CATEGORIES["Special Education"])


@functools.lru_cache(maxsize=1)
def load_programs_data(is_local: bool = False) -> pd.DataFrame:
    """Load the programs CSV file."""
    programs_path = f"{get_sfusd_path(is_local)}/Data/Cleaned/programs_withMissionBay_2324.csv"
    return pd.read_csv(programs_path)


def compute_program_metrics(
    _zone_dict: dict[int, int],  # unused, kept for API consistency
    _G: nx.Graph,  # unused, kept for API consistency
    zone_schools: dict[int, list[int]],
    is_local: bool = False
) -> tuple[dict[str, float], dict[int, dict]]:
    """
    Compute program access metrics.
    
    Args:
        zone_dict: Area to zone mapping
        G: Graph with school_ids per node
        zone_schools: Zone to list of school IDs mapping
        is_local: Whether running locally
    
    Returns:
        Tuple of (aggregated_metrics, per_zone_program_data)
    """
    programs_df = load_programs_data(is_local)
    
    # Build school_id -> list of program_types mapping
    school_programs = {}
    for _, row in programs_df.iterrows():
        sid = row['school_id']
        ptype = row['program_type']
        if sid not in school_programs:
            school_programs[sid] = []
        school_programs[sid].append(ptype)
    
    per_zone_data = {}
    
    # Aggregate counts for averaging
    all_program_counts = []
    all_lang_counts = []
    all_sped_counts = []
    program_type_counts = {}  # {program_type: [count_per_zone]}
    
    for zone_id, schools in zone_schools.items():
        # Count programs in this zone
        zone_programs = {}
        lang_count = 0
        sped_count = 0
        
        for sid in schools:
            if sid in school_programs:
                for ptype in school_programs[sid]:
                    zone_programs[ptype] = zone_programs.get(ptype, 0) + 1
                    
                    if ptype in LANGUAGE_PROGRAMS:
                        lang_count += 1
                    if ptype in SPECIAL_EDUCATION:
                        sped_count += 1
        
        total_programs = sum(zone_programs.values())
        
        per_zone_data[zone_id] = {
            'programs': zone_programs,
            'total_programs': total_programs,
            'language_immersion_count': lang_count,
            'special_ed_count': sped_count
        }
        
        all_program_counts.append(total_programs)
        all_lang_counts.append(lang_count)
        all_sped_counts.append(sped_count)
        
        # Track per-program-type counts
        for ptype, count in zone_programs.items():
            if ptype not in program_type_counts:
                program_type_counts[ptype] = []
            program_type_counts[ptype].append(count)
    
    # Compute aggregated metrics
    num_zones = len(zone_schools) if zone_schools else 1
    aggregated = {
        MetricColumns.AVG_TOTAL_PROGRAMS: sum(all_program_counts) / num_zones if all_program_counts else 0,
        MetricColumns.AVG_LANGUAGE_IMMERSION: sum(all_lang_counts) / num_zones if all_lang_counts else 0,
        MetricColumns.AVG_SPECIAL_ED: sum(all_sped_counts) / num_zones if all_sped_counts else 0,
    }

    # Add per-program-type averages
    for ptype, counts in program_type_counts.items():
        avg = sum(counts) / num_zones
        aggregated[MetricColumns.program_column(ptype)] = avg
    
    return aggregated, per_zone_data
