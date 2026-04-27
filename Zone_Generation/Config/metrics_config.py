"""
Centralized Metric Configuration — Single Source of Truth.

Defines all metrics available for filtering zoning solutions,
organized by category with descriptions for LLM understanding.
Also includes chart visualization hints for the website frontend
and column-name constants for computation modules.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric."""
    column: str                              # CSV column name
    display_name: str                        # User-friendly name
    description: str                         # Brief description for LLM
    category: str                            # diversity/distance/programs/quality/structure
    direction: Literal["minimize", "maximize"] | None = None  # None = informational only
    is_core: bool = True                     # Show in main prompt vs on-demand
    short_name: str = ""                     # Short label for badges/compact display
    # Chart visualization (for website frontend)
    chart_type: str = "none"                 # "bar", "ethnicity", "none"
    chart_field: str = ""                    # zone_data field name if different from column
    chart_unit: str = ""                     # "%", "miles", "Count", "Score", etc.
    chart_max: float | None = None           # max value for chart scale
    chart_title: str = ""                    # chart title override
    chart_normalize: bool = False            # show as % deviation from zone average


# ============================================================================
# COLUMN NAME CONSTANTS (for use in computation modules)
# ============================================================================

class MetricColumns:
    """Column name constants for use in computation modules."""
    THEIL_INDEX = "theil_index"
    FRL_DISSIM = "frl_dissim"
    BLACK_DISSIM = "black_dissim"
    HISPANIC_DISSIM = "hispanic_dissim"
    WHITE_DISSIM = "white_dissim"
    ASIAN_DISSIM = "asian_dissim"
    SEAT_DISPARITY = "seat_disparity"
    AVG_CLOSEST_ZONE_SCHOOL_DISTANCE = "avg_closest_zone_school_distance"
    AVG_SCHOOLS_IN_ATTENDANCE_AREA = "avg_schools_in_attendance_area"
    BOUNDARY_COST = "boundary_cost"
    COMPACTNESS = "compactness"
    AVG_TOTAL_PROGRAMS = "avg_total_programs_per_zone"
    AVG_LANGUAGE_IMMERSION = "avg_language_immersion_per_zone"
    AVG_SPECIAL_ED = "avg_special_ed_per_zone"
    AVG_GE = "avg_GE_per_zone"
    MAD_MATH_SCORE = "mad_math_score"
    MAD_ENG_SCORE = "mad_eng_score"
    AVG_MAX_UTILITY = "avg_max_utility"
    AVG_LOGSUM_UTILITY = "avg_logsum_utility"
    AVG_GE_SCHOOLS_WITHIN_HALF_MILE = "avg_ge_schools_within_half_mile"
    NUM_ZONES = "num_zones"

    @staticmethod
    def program_column(ptype: str) -> str:
        """Generate column name for a specific program type."""
        return f"avg_{ptype}_per_zone"


# ============================================================================
# ETHNICITY DISPLAY LABELS (for website frontend)
# ============================================================================

ETHNICITY_DISPLAY_LABELS = {
    "Ethnicity_Black_or_African_American": "Black/African American",
    "Ethnicity_Hispanic/Latinx": "Hispanic/Latinx",
    "Ethnicity_White": "White",
    "Ethnicity_Asian": "Asian",
}


# ============================================================================
# METRIC CATEGORIES
# ============================================================================

CATEGORIES = {
    "diversity": "Demographics and Economic Balance",
    "proximity": "Geographic Access and Proximity",
    "programs": "Educational Program Availability",
    "quality": "School Quality Indicators",
    "structure": "Zone Structure and Shape",
}

CATEGORY_DESCRIPTIONS = {
    "diversity": "Measures how evenly demographics are distributed across zones.",
    "proximity": "Measures geographic access to schools within zones.",
    "programs": "Measure how hard it is for students to access programs within their zone.",
    "quality": "Measures how evenly school quality is distributed across zones.",
    "structure": "Structural properties of the zone configuration including shape compactness and zone count.",
}


# ============================================================================
# DIVERSITY METRICS
# ============================================================================

DIVERSITY_METRICS = [
    MetricSpec(
        column="theil_index",
        display_name="Racial Diversity",
        description="Theil Entropy index measuring racial diversity across zones. Compares the racial composition (Black, Hispanic/Latinx, White, Asian) within each zone to the district-wide composition, weighted by zone population. Lower = more diverse (zones reflect the district mix), higher = less diverse (racial groups concentrated in certain zones). Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=True,
        short_name="Racial",
        chart_type="ethnicity",
        chart_title="Racial Diversity",
    ),
    MetricSpec(
        column="frl_dissim",
        display_name="Soceconomic Diversity",
        description="Dissimilarity index for free/reduced lunch students. This represents the proportion of FRL students that would need to move between zones for even distribution. 0 = perfectly integrated, 1 = completely segregated.",
        category="diversity",
        direction="minimize",
        is_core=True,
        short_name="SES",
        chart_type="bar",
        chart_field="frl_pct",
        chart_unit="%",
        chart_max=100,
        chart_title="Socioeconomic Diversity",
    ),
    MetricSpec(
        column="black_dissim",
        display_name="Black Representation",
        description="Dissimilarity index for Black/African American students. This represents the proportion of Black/African American students that would need to move between zones for even distribution. 0 = perfectly integrated, 1 = completely segregated.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="hispanic_dissim",
        display_name="Hispanic/Latinx Representation",
        description="Dissimilarity index for Hispanic/Latinx students. This represents the proportion of Hispanic/Latinx students that would need to move between zones for even distribution. 0 = perfectly integrated, 1 = completely segregated.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="white_dissim",
        display_name="White Representation",
        description="Dissimilarity index for White students. This represents the proportion of white students that would need to move between zones for even distribution. 0 = perfectly integrated, 1 = completely segregated.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="asian_dissim",
        display_name="Asian Representation",
        description="Dissimilarity index for Asian students. This represents the proportion of asian students that would need to move between zones for even distribution. 0 = perfectly integrated, 1 = completely segregated.",
        category="diversity",
        direction="minimize",
        is_core=False,
    )
]


# ============================================================================
# PROXIMITY METRICS
# ============================================================================

PROXIMITY_METRICS = [
    MetricSpec(
        column="avg_closest_zone_school_distance",
        display_name="Avg Distance to Closest In-Zone School",
        description="Average distance to nearest in-zone school (miles) across all students.",
        category="proximity",
        direction="minimize",
        is_core=True,
        short_name="Avg Distance",
        chart_type="bar",
        chart_field="avg_closest_school_distance",
        chart_unit="miles",
        chart_title="Avg Distance to Closest School",
    ),

    MetricSpec(
        column="avg_ge_schools_within_half_mile",
        display_name="In-Zone GE Programs Within 0.5 Miles",
        description="Average number of General Education programs within 0.5 miles of each student. Higher means students have more nearby GE program options.",
        category="proximity",
        direction="maximize",
        is_core=True,
        short_name="Walkable Schools",
        chart_type="bar",
        chart_field="ge_schools_within_half_mile",
        chart_unit="Schools",
        chart_title="Schools with General Education programs within 0.5 Miles",
    ),
]


# ============================================================================
# PROGRAM METRICS (higher = more access to programs)
# ============================================================================

PROGRAM_METRICS = [
    # MetricSpec(
    #     column="avg_total_programs_per_zone",
    #     display_name="Total Programs",
    #     description="Average total program count per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=True,
    #     chart_type="bar",
    #     chart_field="total_programs",
    #     chart_unit="Count",
    #     chart_title="Total Programs by Zone",
    # ),
    # MetricSpec(
    #     column="avg_language_immersion_per_zone",
    #     display_name="Language Immersion Programs",
    #     description="Avg language immersion programs per zone.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=True,
    #     chart_type="bar",
    #     chart_field="language_immersion_count",
    #     chart_unit="Count",
    #     chart_title="Language Immersion by Zone",
    # ),
    # MetricSpec(
    #     column="avg_special_ed_per_zone",
    #     display_name="Special Education Programs",
    #     description="Avg special education programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=True,
    #     chart_type="bar",
    #     chart_field="special_ed_count",
    #     chart_unit="Count",
    #     chart_title="Special Ed by Zone",
    # ),

    MetricSpec(
        column="seat_disparity",
        display_name="Student Seat Imbalance",
        description="Average % difference in GE seats over students in each zone. Negative numbers for each zone show that this zone have more students than seats, \
        positive numbers show that this zone have more seats than students. The overall value for this mapping is the average % total difference in GE seats over students across all zones. \
         Lower indicates that no one zone has a large imbalance of either having too few students or too few seats.",
        category="programs",
        direction="minimize",
        is_core=True,
        short_name="Seats",
        chart_type="bar",
        chart_field="seat_disparity",
        chart_title="Seat Disparity by Zone",
        chart_unit="%",
    ),
    # MetricSpec(
    #     column="avg_GE_per_zone",
    #     display_name="General Education Programs",
    #     description="Avg general education (GE) programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=True,
    # ),
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
    # MetricSpec(
    #     column="avg_SA_per_zone",
    #     display_name="Extensive Services Autism (SA)",
    #     description="Extensive Services Autism Focus programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_CN_per_zone",
    #     display_name="Cantonese Immersion (CN)",
    #     description="Cantonese immersion programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_AF_per_zone",
    #     display_name="Autism Focus (AF)",
    #     description="Autism Focus programs per zone. These are programs for students with autism spectrum disorder (ASD).",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_CB_per_zone",
    #     display_name="Cantonese Biliteracy (CB)",
    #     description="Cantonese biliteracy programs per zone. These programs are targeted for english language learners.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_MM_per_zone",
    #     display_name="Mild-Moderate (MM)",
    #     description="Mild-moderate special education programs per zone. These are programs for students with mild to moderate autism spectrum disorder (ASD) needs or other developmental disabilities.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_SB_per_zone",
    #     display_name="Spanish Biliteracy (SB)",
    #     description="Spanish biliteracy programs per zone. These programs are targeted for english language learners.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_FB_per_zone",
    #     display_name="Filipino Biliteracy (FB)",
    #     description="Filipino biliteracy programs per zone. These programs are targeted for english language learners.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_NC_per_zone",
    #     display_name="Newcomer Chinese (NC)",
    #     description="Newcomer Chinese programs per zone. These programs are targeted for students who are recent immigrants to the US and need extra support.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_MS_per_zone",
    #     display_name="Moderate-Severe (MS)",
    #     description="Moderate-severe special education programs per zone. These are programs for students with moderate to severe autism spectrum disorder (ASD) or other developmental disabilities.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_TC_per_zone",
    #     display_name="Total Communication (TC)",
    #     description="Deaf/Hard of Hearing, Total Communication programs per zone. " + \
    #                 "Total Communication employs a multi-modal approach that simultaneously combines speech, formal signs, and gestures " + \
    #                 "to ensure the child has every tool available to understand and express themselves.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
    # MetricSpec(
    #     column="avg_AO_per_zone",
    #     display_name="Auditory Oral (AO)",
    #     description="Deaf/Hard of Hearing, Auditory Oral programs per zone. " + \
    #                 "Auditory-Oral programs focus exclusively on developing spoken language and listening skills " + \
    #                 "by utilizing residual hearing and lip-reading while strictly avoiding sign language.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=False,
    # ),
]


# ============================================================================
# STRUCTURE METRICS (zone shape and configuration)
# ============================================================================

STRUCTURE_METRICS = [
    # MetricSpec(
    #     column="boundary_cost",
    #     display_name="Boundary Cost",
    #     description="Total number of edges that cross zone boundaries in the graph. Higher means more complex, jagged zone borders.",
    #     category="structure",
    #     direction="minimize",
    #     is_core=False,
    #     short_name="Boundary",
    # ),
    MetricSpec(
        column="compactness",
        display_name="Compactness",
        description="The overall compactness of the zones in this mapping. This basically measures how jagged and weird the zones look. Lower indicates \
        that zones are more compact and \"nicer\" looking. This is calculated by normalizing the number of neighbors in different zones by the number of zones.",
        category="structure",
        direction="minimize",
        is_core=False,
        short_name="Compactness",
    ),
    MetricSpec(
        column="num_zones",
        display_name="Number of Zones",
        description="Total number of zones in this map.",
        category="structure",
        is_core=False,
        short_name="Zones",
    ),
]


# ============================================================================
# QUALITY METRICS (lower MAD = more equitable distribution)
# ============================================================================

QUALITY_METRICS = [
    MetricSpec(
        column="mad_math_score",
        display_name="Math Score Equity",
        description="Mean absolute deviation of average school math scores across zones. Lower means more equitable math quality distribution.",
        category="quality",
        direction="minimize",
        is_core=True,
        short_name="Math",
        chart_type="bar",
        chart_field="avg_math_score",
        chart_unit="Score",
        chart_title="Math Scores by Zone",
        chart_normalize=True,
    ),
    MetricSpec(
        column="mad_eng_score",
        display_name="English Score Equity",
        description="Mean absolute deviation of average school English scores across zones. Lower means more equitable English quality distribution.",
        category="quality",
        direction="minimize",
        is_core=True,
        short_name="Eng",
        chart_type="bar",
        chart_field="avg_eng_score",
        chart_unit="Score",
        chart_title="English Scores by Zone",
        chart_normalize=True,
    ),
]


# ============================================================================
# AGGREGATED METRIC REGISTRY
# ============================================================================

ALL_METRICS: list[MetricSpec] = (
    DIVERSITY_METRICS + PROXIMITY_METRICS + PROGRAM_METRICS + QUALITY_METRICS + STRUCTURE_METRICS
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
            if m.direction == "minimize":
                direction = "lower is better"
            elif m.direction == "maximize":
                direction = "higher is better"
            else:
                direction = "informational"
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


def resolve_metric_identifiers(identifiers: list[str]) -> list[str]:
    """Resolve a mix of category names and metric display/column names to column names."""
    columns = []
    for ident in identifiers:
        ident_lower = ident.lower()
        if ident_lower in CATEGORIES:
            columns.extend(m.column for m in get_metrics_by_category(ident_lower))
        elif ident in METRIC_BY_NAME:
            columns.append(METRIC_BY_NAME[ident].column)
        elif ident in METRIC_BY_COLUMN:
            columns.append(ident)
    return columns


def get_chart_hints() -> dict[str, dict]:
    """Build chart hints dict from MetricSpec chart fields (for website API)."""
    hints = {}
    for m in ALL_METRICS:
        if m.chart_type == "none":
            hints[m.column] = {"type": "none"}
        else:
            hint: dict = {"type": m.chart_type, "title": m.chart_title}
            if m.chart_field:
                hint["field"] = m.chart_field
            if m.chart_unit:
                hint["unit"] = m.chart_unit
            if m.chart_max is not None:
                hint["max"] = m.chart_max
            if m.chart_normalize:
                hint["normalize"] = True
            hints[m.column] = hint
    return hints
