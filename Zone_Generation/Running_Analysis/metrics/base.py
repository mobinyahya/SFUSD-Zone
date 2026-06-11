"""
Base classes and data structures for zoning metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ZoneData:
    """Per-zone data storage for all metrics."""
    zone_id: int
    
    # Demographics
    ge_students: float = 0.0
    ge_capacity: float = 0.0
    frl_pct: float = 0.0
    aalpi_pct: float = 0.0
    ethnicity_pcts: dict[str, float] = field(default_factory=dict)
    
    # Programs available in zone
    programs: dict[str, int] = field(default_factory=dict)
    total_programs: int = 0
    language_immersion_count: int = 0
    special_ed_count: int = 0
    
    # GE proximity
    ge_schools_within_half_mile: float = 0.0

    # Seat disparity (per-zone capacity vs enrollment ratio)
    seat_disparity: Optional[float] = None

    # Quality (capacity-weighted per-zone averages); None means no schools with data
    avg_math_score: Optional[float] = None
    avg_eng_score: Optional[float] = None
    
    # Distance
    avg_any_ge_school_distance: float = 0.0
    avg_farthest_ge_school_distance: float = 0.0
    avg_out_of_zone_ge_schools: float = 0.0
    schools_in_attendance_area: int = 0
    
    # Utility
    avg_max_utility: float = 0.0
    avg_logsum_utility: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'zone_id': self.zone_id,
            'ge_students': self.ge_students,
            'ge_capacity': self.ge_capacity,
            'frl_pct': self.frl_pct,
            'aalpi_pct': self.aalpi_pct,
            'ethnicity_pcts': self.ethnicity_pcts,
            'programs': self.programs,
            'total_programs': self.total_programs,
            'language_immersion_count': self.language_immersion_count,
            'special_ed_count': self.special_ed_count,
            'ge_schools_within_half_mile': self.ge_schools_within_half_mile,
            'seat_disparity': self.seat_disparity,
            'avg_math_score': self.avg_math_score,
            'avg_eng_score': self.avg_eng_score,
            'avg_any_ge_school_distance': self.avg_any_ge_school_distance,
            'avg_farthest_ge_school_distance': self.avg_farthest_ge_school_distance,
            'avg_out_of_zone_ge_schools': self.avg_out_of_zone_ge_schools,
            'schools_in_attendance_area': self.schools_in_attendance_area,
            'avg_max_utility': self.avg_max_utility,
            'avg_logsum_utility': self.avg_logsum_utility,
        }


@dataclass
class MetricsResult:
    """Aggregated metrics result with per-zone data."""
    
    # Aggregated metrics (averages across zones; string values allowed for ids)
    metrics: dict[str, float | str] = field(default_factory=dict)

    # Per-zone detailed data
    zone_data: dict[int, ZoneData] = field(default_factory=dict)

    def update(self, new_metrics: dict[str, float | str]) -> None:
        """Update aggregated metrics."""
        self.metrics.update(new_metrics)

    def to_flat_dict(self) -> dict[str, float | str]:
        """Return flat dictionary of aggregated metrics for backward compatibility."""
        return self.metrics.copy()
    
    def to_full_dict(self) -> dict[str, Any]:
        """Return full dictionary including zone data."""
        return {
            'metrics': self.metrics,
            'zone_data': {zid: zd.to_dict() for zid, zd in self.zone_data.items()}
        }
