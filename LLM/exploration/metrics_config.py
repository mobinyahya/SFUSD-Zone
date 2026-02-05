"""
Centralized Metric Configuration for Zoning Agent.

Defines all metrics available for filtering zoning solutions,
organized by category with descriptions for LLM understanding.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric."""
    column: str                              # CSV column name
    display_name: str                        # User-friendly name
    description: str                         # Brief description for LLM
    category: str                            # diversity/distance/programs/quality
    direction: Literal["minimize", "maximize"]
    is_core: bool = True                     # Show in main prompt vs on-demand


# ============================================================================
# METRIC CATEGORIES
# ============================================================================

CATEGORIES = {
    "diversity": "Demographics and Economic Balance",
    "distance": "Geographic Access and Proximity",
    "programs": "Educational Program Availability",
    "quality": "School Quality Indicators",
}

CATEGORY_DESCRIPTIONS = {
    "diversity": "Measures how evenly demographics are distributed across zones. Lower deviation = more balanced.",
    "distance": "Measures geographic access to schools within zones.",
    "programs": "Counts of educational programs available in each zone. Higher = more options.",
    "quality": "Aggregated school quality indicators. Higher = better outcomes.",
}


# ============================================================================
# DIVERSITY METRICS (minimize deviation from district average)
# ============================================================================

DIVERSITY_METRICS = [
    MetricSpec(
        column="theil_index",
        display_name="Ethnic Segregation Index",
        description="Theil index measuring ethnic segregation (0=integrated, higher=segregated)",
        category="diversity",
        direction="minimize",
        is_core=True,
    ),
    MetricSpec(
        column="FRL",
        display_name="FRL Deviation",
        description="Free/reduced lunch % deviation from district average",
        category="diversity",
        direction="minimize",
        is_core=True,
    ),
    MetricSpec(
        column="Ethnicity_Black_or_African_American",
        display_name="Black Population Deviation",
        description="Black student % deviation from district average",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="Ethnicity_Hispanic/Latinx",
        display_name="Hispanic Population Deviation",
        description="Hispanic/Latinx student % deviation from district average",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="Ethnicity_White",
        display_name="White Population Deviation",
        description="White student % deviation from district average",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="Ethnicity_Asian",
        display_name="Asian Population Deviation",
        description="Asian student % deviation from district average",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="seat_disparity",
        display_name="Seat Disparity",
        description="Imbalance between seats and students per zone",
        category="diversity",
        direction="minimize",
        is_core=True,
    ),
]


# ============================================================================
# DISTANCE METRICS
# ============================================================================

DISTANCE_METRICS = [
    MetricSpec(
        column="avg_closest_zone_school_distance",
        display_name="Avg Distance to Closest School",
        description="Average distance students travel to nearest school in zone",
        category="distance",
        direction="minimize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_schools_in_attendance_area",
        display_name="Schools in Attendance Area",
        description="Avg number of zone schools in students' attendance area",
        category="distance",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="boundary_cost",
        display_name="Boundary Cost (Compactness)",
        description="Zone boundary complexity; lower = more compact zones",
        category="distance",
        direction="minimize",
        is_core=True,
    ),
]


# ============================================================================
# PROGRAM METRICS (higher = more access to programs)
# ============================================================================

PROGRAM_METRICS = [
    MetricSpec(
        column="avg_total_programs_per_zone",
        display_name="Total Programs",
        description="Average total program count per zone",
        category="programs",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_language_immersion_per_zone",
        display_name="Language Immersion Programs",
        description="Avg language immersion programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_special_ed_per_zone",
        display_name="Special Education Programs",
        description="Avg special education programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_GE_per_zone",
        display_name="General Education Programs",
        description="Avg general education (GE) programs per zone",
        category="programs",
        direction="maximize",
        is_core=True,
    ),
    # Individual program types (non-core, available on demand)
    MetricSpec(
        column="avg_SA_per_zone",
        display_name="Self-Contained Autism (SA)",
        description="Self-contained autism programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_CN_per_zone",
        display_name="Cantonese Immersion (CN)",
        description="Cantonese immersion programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_AF_per_zone",
        display_name="Autism Focus (AF)",
        description="Autism focus programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_CB_per_zone",
        display_name="Cantonese Bilingual (CB)",
        description="Cantonese bilingual programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_MM_per_zone",
        display_name="Mild-Moderate SDC (MM)",
        description="Mild-moderate special day class programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_SB_per_zone",
        display_name="Spanish Bilingual (SB)",
        description="Spanish bilingual programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_FB_per_zone",
        display_name="Filipino Bilingual (FB)",
        description="Filipino bilingual programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_NC_per_zone",
        display_name="Newcomer (NC)",
        description="Newcomer programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_MS_per_zone",
        display_name="Moderate-Severe SDC (MS)",
        description="Moderate-severe special day class programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_TC_per_zone",
        display_name="Transitional Classroom (TC)",
        description="Transitional classroom programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_AO_per_zone",
        display_name="Autism Overlay (AO)",
        description="Autism overlay programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
]


# ============================================================================
# QUALITY METRICS (higher = better)
# ============================================================================

QUALITY_METRICS = [
    MetricSpec(
        column="avg_greatschools_rating",
        display_name="GreatSchools Rating",
        description="Average GreatSchools rating (1-10 scale)",
        category="quality",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_math_score",
        display_name="Math Scores",
        description="Average math proficiency scores",
        category="quality",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_eng_score",
        display_name="English Scores",
        description="Average English proficiency scores",
        category="quality",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_suspension_index",
        display_name="Suspension Index",
        description="Suspension index (1-5, higher = fewer suspensions = better)",
        category="quality",
        direction="maximize",
        is_core=True,
    ),
]


# ============================================================================
# AGGREGATED METRIC REGISTRY
# ============================================================================

ALL_METRICS: list[MetricSpec] = (
    DIVERSITY_METRICS + DISTANCE_METRICS + PROGRAM_METRICS + QUALITY_METRICS
)

# Build lookup dictionaries
METRIC_BY_COLUMN: dict[str, MetricSpec] = {m.column: m for m in ALL_METRICS}
METRIC_BY_NAME: dict[str, MetricSpec] = {m.display_name: m for m in ALL_METRICS}

# Core metrics for default LLM exposure
CORE_METRICS: list[MetricSpec] = [m for m in ALL_METRICS if m.is_core]


def get_metric_columns() -> list[str]:
    """Get list of all metric CSV column names."""
    return [m.column for m in ALL_METRICS]


def get_core_metric_columns() -> list[str]:
    """Get list of core metric CSV column names."""
    return [m.column for m in CORE_METRICS]


def get_metrics_by_category(category: str) -> list[MetricSpec]:
    """Get all metrics in a category."""
    return [m for m in ALL_METRICS if m.category == category]


def get_metric_summary() -> str:
    """Generate a summary of all metrics for LLM context."""
    lines = ["## Available Metrics\n"]
    
    for cat_key, cat_name in CATEGORIES.items():
        metrics = get_metrics_by_category(cat_key)
        if not metrics:
            continue
        
        lines.append(f"### {cat_name}")
        lines.append(CATEGORY_DESCRIPTIONS[cat_key])
        lines.append("")
        
        for m in metrics:
            direction = "lower is better" if m.direction == "minimize" else "higher is better"
            core_marker = "" if m.is_core else " [detailed]"
            lines.append(f"- **{m.display_name}**{core_marker}: {m.description} ({direction})")
        
        lines.append("")
    
    return "\n".join(lines)


def search_metrics(query: str) -> list[MetricSpec]:
    """Search metrics by name or description (case-insensitive)."""
    query_lower = query.lower()
    return [
        m for m in ALL_METRICS
        if query_lower in m.display_name.lower() or query_lower in m.description.lower()
    ]
