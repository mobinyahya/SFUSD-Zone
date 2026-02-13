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
        display_name="Racial Diversity Index",
        description="Theil Entropy index measuring racial diversity (0=Highly-Diverse, 1=Non-Diverse). This uses the following racial groups: Black, Hispanic/Latinx, White, Asian.",
        category="diversity",
        direction="minimize",
        is_core=True,
    ),
    MetricSpec(
        column="FRL",
        display_name="FRL Representation",
        description="Average percentage that the free/reduced lunch population deviates from the district average across zones. Lower indicates that zones are more balanced in terms of FRL representation.",
        category="diversity",
        direction="minimize",
        is_core=True,
    ),
    MetricSpec(
        column="Ethnicity_Black_or_African_American",
        display_name="Black Representation",
        description="Average percentage that the black population deviates from the district average across zones. Lower indicates that zones are more balanced in terms of black representation.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="Ethnicity_Hispanic/Latinx",
        display_name="Hispanic/Latinx Representation",
        description="Average percentage that the Hispanic/Latinx population deviates from the district average across zones. Lower indicates that zones are more balanced in terms of Hispanic/Latinx representation.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="Ethnicity_White",
        display_name="White Representation",
        description="Average percentage that the white population deviates from the district average across zones. Lower indicates that zones are more balanced in terms of white representation.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="Ethnicity_Asian",
        display_name="Asian Representation",
        description="Average percentage that the Asian population deviates from the district average across zones. Lower indicates that zones are more balanced in terms of Asian representation.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="seat_disparity",
        display_name="Student Seat Imbalance",
        description="Average percentage that the number of seats deviates from the number of students per zone. Lower indicates that zones are more balanced in terms of seat-student imbalance.",
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
        description="Average distance to nearest school in zone (miles) across all zones.",
        category="distance",
        direction="minimize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_schools_in_attendance_area",
        display_name="Schools in Attendance Area",
        description="Avg number of schools in the a students attendance area and zone. Higher indicates that students have more access to nearby schools.",
        category="distance",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="boundary_cost",
        display_name="Boundary Cost (Compactness)",
        description="Zone boundary complexity. This basically measures how jagged and weird the zones look. Lower indicates that zones are more compact and \"nicer\" looking.",
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
        description="Avg language immersion programs per zone.",
        category="programs",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_special_ed_per_zone",
        display_name="Special Education Programs",
        description="Avg special education programs per zone",
        category="programs",
        direction="maximize",
        is_core=True,
    ),
    MetricSpec(
        column="avg_GE_per_zone",
        display_name="General Education Programs",
        description="Avg general education (GE) programs per zone",
        category="programs",
        direction="maximize",
        is_core=True,
    ),
    # Individual program types (non-core, available on demand)\
    # MetricSpec(
    #     column="avg_CA_per_zone",
    #     display_name="Community Access (CA)",
    #     description="Community Access and Transition programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_SOAR_per_zone",
    #     display_name="Success, Opportunity, Achievement, Resiliency (SOAR)",
    #     description="Success, Opportunity, Achievement, Resiliency programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_RSP_per_zone",
    #     display_name="Resource Specialist Program (RSP)",
    #     description="Resource Specialist Program programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    MetricSpec(
        column="avg_SA_per_zone",
        display_name="Extensive Services Autism (SA)",
        description="Extensive Services Autism Focus programs per zone",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    # MetricSpec(
    #     column="avg_CN_per_zone",
    #     display_name="Cantonese Immersion (CN)",
    #     description="Cantonese immersion programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    MetricSpec(
        column="avg_AF_per_zone",
        display_name="Autism Focus (AF)",
        description="Autism Focus programs per zone. These are programs for students with autism spectrum disorder (ASD).",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_CB_per_zone",
        display_name="Cantonese Biliteracy (CB)",
        description="Cantonese biliteracy programs per zone. These programs are targeted for english language learners.",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_MM_per_zone",
        display_name="Mild-Moderate (MM)",
        description="Mild-moderate special education programs per zone. These are programs for students with mild to moderate autism spectrum disorder (ASD) needs or other developmental disabilities.",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_SB_per_zone",
        display_name="Spanish Biliteracy (SB)",
        description="Spanish biliteracy programs per zone. These programs are targeted for english language learners.",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_FB_per_zone",
        display_name="Filipino Biliteracy (FB)",
        description="Filipino biliteracy programs per zone. These programs are targeted for english language learners.",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_NC_per_zone",
        display_name="Newcomer Chinese (NC)",
        description="Newcomer Chinese programs per zone. These programs are targeted for students who are recent immigrants to the US and need extra support.",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_MS_per_zone",
        display_name="Moderate-Severe (MS)",
        description="Moderate-severe special education programs per zone. These are programs for students with moderate to severe autism spectrum disorder (ASD) or other developmental disabilities.",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_TC_per_zone",
        display_name="Total Communication (TC)",
        description="Deaf/Hard of Hearing, Total Communication programs per zone. " + \
                    "Total Communication employs a multi-modal approach that simultaneously combines speech, formal signs, and gestures " + \
                    "to ensure the child has every tool available to understand and express themselves.",
        category="programs",
        direction="maximize",
        is_core=False,
    ),
    MetricSpec(
        column="avg_AO_per_zone",
        display_name="Auditory Oral (AO)",
        description="Deaf/Hard of Hearing, Auditory Oral programs per zone. " + \
                    "Auditory-Oral programs focus exclusively on developing spoken language and listening skills " + \
                    "by utilizing residual hearing and lip-reading while strictly avoiding sign language.",
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

#     MetricSpec(
#         column="avg_greatschools_rating",
#         display_name="GreatSchools Rating",
#         description="Average GreatSchools rating (1-10 scale)",
#         category="quality",
#         direction="maximize",
#         is_core=True,
#     ),
#     MetricSpec(
#         column="avg_suspension_index",
#         display_name="Suspension Index",
#         description="Suspension index (1-5, higher = fewer suspensions = better)",
#         category="quality",
#         direction="maximize",
#         is_core=True,
#     ),
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
