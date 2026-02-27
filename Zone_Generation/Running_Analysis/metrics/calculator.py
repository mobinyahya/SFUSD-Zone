"""
Main metrics calculator that combines all metric modules.
"""

import networkx as nx
from typing import Optional

from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.Running_Analysis.metrics.base import MetricsResult, ZoneData
from Zone_Generation.Running_Analysis.metrics.diversity import (
    compute_diversity_metrics, compute_seat_disparity, compute_theil_index
)
from Zone_Generation.Running_Analysis.metrics.distance import compute_distance_metrics
from Zone_Generation.Running_Analysis.metrics.programs import compute_program_metrics, compute_ge_proximity_metrics
from Zone_Generation.Running_Analysis.metrics.quality import compute_quality_metrics
from Zone_Generation.Running_Analysis.metrics.choice import compute_choice_metrics


class ZoneMetricsCalculator:
    """
    Main entry point for computing zone metrics.
    
    Computes all metrics and returns a structured MetricsResult with
    both aggregated metrics and per-zone data.
    """
    
    def __init__(
        self, 
        zone_dict: dict[int, int], 
        G: nx.Graph, 
        config: Optional[dict] = None
    ):
        """
        Initialize the calculator.
        
        Args:
            zone_dict: Mapping from area/node ID to zone ID
            G: NetworkX graph with node attributes and graph-level data
            config: Optional configuration dict with settings like 'is_local'
        """
        # Normalize zone_dict keys to int
        self.zone_dict = {int(k): v for k, v in zone_dict.items()}
        self.G = G
        self.config = config or {}
        
        # Build helper mappings
        self._build_zone_mappings()
    
    def _build_zone_mappings(self) -> None:
        """Build zone_blocks and zone_schools mappings."""
        self.zone_blocks: dict[int, list[int]] = {}
        self.zone_schools: dict[int, list[int]] = {}
        
        for node_id, zone_id in self.zone_dict.items():
            # Zone blocks
            if zone_id not in self.zone_blocks:
                self.zone_blocks[zone_id] = []
            self.zone_blocks[zone_id].append(node_id)
            
            # Zone schools
            if zone_id not in self.zone_schools:
                self.zone_schools[zone_id] = []
            if node_id in self.G.nodes:
                school_ids = self.G.nodes[node_id].get('school_ids', [])
                self.zone_schools[zone_id].extend(school_ids)
    
    def compute_all(self, include_choice: bool = True) -> MetricsResult:
        """
        Compute all metrics.
        
        Args:
            include_choice: Whether to compute choice/utility metrics
                           (requires loading utility data, may be slow)
        
        Returns:
            MetricsResult with aggregated metrics and per-zone data
        """
        result = MetricsResult()
        
        # Initialize zone data structures
        for zone_id in self.zone_blocks.keys():
            result.zone_data[zone_id] = ZoneData(zone_id=zone_id)
        
        # 1. Diversity metrics
        diversity_metrics, per_zone_demos = compute_diversity_metrics(
            self.zone_dict, self.G, self.zone_blocks
        )
        result.update(diversity_metrics)
        
        # Add seat disparity (solution-level and per-zone)
        seat_disparity, per_zone_seat = compute_seat_disparity(self.zone_blocks, self.G)
        result.update({MetricColumns.SEAT_DISPARITY: seat_disparity})
        for zone_id, sd_data in per_zone_seat.items():
            if zone_id in result.zone_data:
                result.zone_data[zone_id].seat_disparity = sd_data.get('seat_disparity')

        # Add Theil segregation index
        theil_metrics, per_zone_entropy = compute_theil_index(
            self.zone_dict, self.G, self.zone_blocks
        )
        result.update(theil_metrics)

        # Update zone data with demographics
        for zone_id, demos in per_zone_demos.items():
            if zone_id in result.zone_data:
                result.zone_data[zone_id].ge_students = demos['ge_students']
                result.zone_data[zone_id].frl_pct = demos['frl_pct']
                result.zone_data[zone_id].ethnicity_pcts = demos['ethnicity_pcts']

        # Update zone data with entropy from Theil calculation
        for zone_id, ent_data in per_zone_entropy.items():
            if zone_id in result.zone_data:
                result.zone_data[zone_id].ethnicity_entropy = ent_data['entropy']

        # 2. Distance metrics
        distance_metrics, per_zone_dist = compute_distance_metrics(
            self.zone_dict, self.G, self.zone_blocks, self.zone_schools
        )
        result.update(distance_metrics)
        
        for zone_id, dist_data in per_zone_dist.items():
            if zone_id in result.zone_data:
                result.zone_data[zone_id].avg_closest_school_distance = (
                    dist_data['avg_closest_school_distance']
                )
                result.zone_data[zone_id].schools_in_attendance_area = (
                    dist_data['schools_in_attendance_area']
                )
        
        # 3. Program metrics
        is_local = self.config.get('is_local', False)
        program_metrics, per_zone_prog = compute_program_metrics(
            self.zone_dict, self.G, self.zone_schools, is_local
        )
        result.update(program_metrics)
        
        for zone_id, prog_data in per_zone_prog.items():
            if zone_id in result.zone_data:
                result.zone_data[zone_id].programs = prog_data['programs']
                result.zone_data[zone_id].total_programs = prog_data['total_programs']
                result.zone_data[zone_id].language_immersion_count = (
                    prog_data['language_immersion_count']
                )
                result.zone_data[zone_id].special_ed_count = prog_data['special_ed_count']

        # 3b. GE proximity metrics
        ge_prox_metrics, per_zone_ge = compute_ge_proximity_metrics(
            self.zone_dict, self.G, self.zone_blocks, self.zone_schools, is_local
        )
        result.update(ge_prox_metrics)

        for zone_id, ge_data in per_zone_ge.items():
            if zone_id in result.zone_data:
                result.zone_data[zone_id].ge_schools_within_half_mile = (
                    ge_data['ge_schools_within_half_mile']
                )

        # 4. Quality metrics
        quality_metrics, per_zone_qual = compute_quality_metrics(
            self.zone_dict, self.G, self.zone_schools
        )
        result.update(quality_metrics)
        
        for zone_id, qual_data in per_zone_qual.items():
            if zone_id in result.zone_data:
                result.zone_data[zone_id].avg_math_score = qual_data['avg_math_score']
                result.zone_data[zone_id].avg_eng_score = qual_data['avg_eng_score']
        
        # 5. Structure metrics
        result.update({MetricColumns.NUM_ZONES: len(self.zone_blocks)})

        # 6. Choice metrics (optional)
        if include_choice and self.config.get('compute_choice', True):
            try:
                choice_metrics, per_zone_choice = compute_choice_metrics(
                    self.zone_dict, self.G, self.config
                )
                result.update(choice_metrics)
                
                for zone_id, choice_data in per_zone_choice.items():
                    if zone_id in result.zone_data:
                        result.zone_data[zone_id].avg_max_utility = (
                            choice_data['avg_max_utility']
                        )
                        result.zone_data[zone_id].avg_logsum_utility = (
                            choice_data['avg_logsum_utility']
                        )
            except Exception as e:
                print(f"Warning: Could not compute choice metrics: {e}")
                result.update({MetricColumns.AVG_MAX_UTILITY: 0.0, MetricColumns.AVG_LOGSUM_UTILITY: 0.0})
        
        return result
