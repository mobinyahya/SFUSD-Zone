"""
Base classes and data structures for zoning metrics.
"""

from dataclasses import dataclass, field
from typing import Any


# Program type groupings based on metrics_plan.md
LANGUAGE_IMMERSION = {
    'CB', 'FB', 'IMMC', 'IMMK', 'IMMM', 'IMMS', 'JB',
    'NC', 'NS', 'NX', 'SB', 'SDLC', 'SDLM', 'SDLS'
}
SPECIAL_EDUCATION = {
    'AF', 'AO', 'CA', 'SOAR', 'MM', 'MS', 'RSP', 'SA', 'TC'
}

# Color to numeric mapping for school quality metrics
COLOR_TO_INDEX = {
    'Blue': 5,
    'Green': 4,
    'Yellow': 3,
    'Orange': 2,
    'Red': 1,
    None: 0,
    '': 0
}


@dataclass
class ZoneData:
    """Per-zone data storage for all metrics."""
    zone_id: int
    
    # Demographics
    ge_students: float = 0.0
    frl_pct: float = 0.0
    ethnicity_pcts: dict[str, float] = field(default_factory=dict)
    
    # Programs available in zone
    programs: dict[str, int] = field(default_factory=dict)
    total_programs: int = 0
    language_immersion_count: int = 0
    special_ed_count: int = 0
    
    # Quality (weighted by school capacity)
    avg_greatschools_rating: float = 0.0
    avg_math_score: float = 0.0
    avg_eng_score: float = 0.0
    avg_suspension_index: float = 0.0
    
    # Distance
    avg_closest_school_distance: float = 0.0
    schools_in_attendance_area: int = 0
    
    # Utility
    avg_max_utility: float = 0.0
    avg_logsum_utility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'zone_id': self.zone_id,
            'ge_students': self.ge_students,
            'frl_pct': self.frl_pct,
            'ethnicity_pcts': self.ethnicity_pcts,
            'programs': self.programs,
            'total_programs': self.total_programs,
            'language_immersion_count': self.language_immersion_count,
            'special_ed_count': self.special_ed_count,
            'avg_greatschools_rating': self.avg_greatschools_rating,
            'avg_math_score': self.avg_math_score,
            'avg_eng_score': self.avg_eng_score,
            'avg_suspension_index': self.avg_suspension_index,
            'avg_closest_school_distance': self.avg_closest_school_distance,
            'schools_in_attendance_area': self.schools_in_attendance_area,
            'avg_max_utility': self.avg_max_utility,
            'avg_logsum_utility': self.avg_logsum_utility,
        }


@dataclass
class MetricsResult:
    """Aggregated metrics result with per-zone data."""
    
    # Aggregated metrics (averages across zones)
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Per-zone detailed data
    zone_data: dict[int, ZoneData] = field(default_factory=dict)
    
    def update(self, new_metrics: dict[str, float]) -> None:
        """Update aggregated metrics."""
        self.metrics.update(new_metrics)
    
    def to_flat_dict(self) -> dict[str, float]:
        """Return flat dictionary of aggregated metrics for backward compatibility."""
        return self.metrics.copy()
    
    def to_full_dict(self) -> dict[str, Any]:
        """Return full dictionary including zone data."""
        return {
            'metrics': self.metrics,
            'zone_data': {zid: zd.to_dict() for zid, zd in self.zone_data.items()}
        }
