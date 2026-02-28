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
    frl_pct: float = 0.0
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
    avg_closest_school_distance: float = 0.0
    schools_in_attendance_area: int = 0
    
    # Utility
    avg_max_utility: float = 0.0
    avg_logsum_utility: float = 0.0

    # Segregation
    ethnicity_entropy: float = 0.0  # Shannon entropy of zone ethnic composition

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
            'ge_schools_within_half_mile': self.ge_schools_within_half_mile,
            'seat_disparity': self.seat_disparity,
            'avg_math_score': self.avg_math_score,
            'avg_eng_score': self.avg_eng_score,
            'avg_closest_school_distance': self.avg_closest_school_distance,
            'schools_in_attendance_area': self.schools_in_attendance_area,
            'avg_max_utility': self.avg_max_utility,
            'avg_logsum_utility': self.avg_logsum_utility,
            'ethnicity_entropy': self.ethnicity_entropy,
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
