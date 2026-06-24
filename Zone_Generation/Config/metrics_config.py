"""
Centralized Metric Configuration — Single Source of Truth.

Defines all metrics available for filtering zoning solutions, organized by
category, with column-name constants for computation modules.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a single metric."""
    column: str                              # CSV column name
    display_name: str                        # User-friendly name
    description: str                         # Brief metric description
    category: str                            # diversity/distance/programs/quality/structure
    direction: Literal["minimize", "maximize"] | None = None  # None = informational only
    is_core: bool = True                     # Include in the default metric subset
    short_name: str = ""                     # Short label for badges/compact display


# ============================================================================
# COLUMN NAME CONSTANTS (for use in computation modules)
# ============================================================================

class MetricColumns:
    """Column name constants for use in computation modules."""
    AALPI_MAD = "aalpi_mad"
    FRL_MAD = "frl_mad"
    BLACK_MAD = "black_mad"
    HISPANIC_MAD = "hispanic_mad"
    WHITE_MAD = "white_mad"
    ASIAN_MAD = "asian_mad"
    PACIFIC_ISLANDER_MAD = "pacific_islander_mad"
    AALPI_RANGE = "aalpi_range"
    FRL_RANGE = "frl_range"
    BLACK_RANGE = "black_range"
    HISPANIC_RANGE = "hispanic_range"
    WHITE_RANGE = "white_range"
    ASIAN_RANGE = "asian_range"
    PACIFIC_ISLANDER_RANGE = "pacific_islander_range"
    SEAT_DISPARITY = "seat_disparity"
    AVG_ANY_ZONE_GE_SCHOOL_DISTANCE = "avg_any_zone_ge_school_distance"
    AVG_FARTHEST_ZONE_GE_SCHOOL_DISTANCE = "avg_farthest_zone_ge_school_distance"
    AVG_OUT_OF_ZONE_GE_SCHOOLS = "avg_out_of_zone_ge_schools_within_half_mile"
    AVG_SCHOOLS_IN_ATTENDANCE_AREA = "avg_schools_in_attendance_area"
    CUT_EDGES = "cut_edges"
    NORMALIZED_CUT_EDGES = "normalized_cut_edges"
    AVG_REOCK_SCORE = "avg_reock_score"
    AVG_POLSBY_POPPER_SCORE = "avg_polsby_popper_score"
    AVG_TOTAL_PROGRAMS = "avg_total_programs_per_zone"
    AVG_LANGUAGE_IMMERSION = "avg_language_immersion_per_zone"
    AVG_SPECIAL_ED = "avg_special_ed_per_zone"
    AVG_GE = "avg_GE_per_zone"
    MAD_MATH_SCORE = "mad_math_score"
    MAD_ENG_SCORE = "mad_eng_score"
    MATH_SCORE_RANGE = "math_score_range"
    ENG_SCORE_RANGE = "eng_score_range"
    AVG_MAX_UTILITY = "avg_max_utility"
    AVG_LOGSUM_UTILITY = "avg_logsum_utility"
    AVG_GE_SCHOOLS_WITHIN_HALF_MILE = "avg_ge_schools_within_half_mile"
    NUM_ZONES = "num_zones"
    CONTIGUOUS = "contiguous"
    SOLUTION_CODE = "solution_code"
    FINAL_OBJECTIVE = "final_objective"
    FINAL_CUT_EDGES = "final_cut_edges"
    FINAL_STATUS = "final_status"
    FINAL_WALL_TIME = "final_wall_time"
    TOTAL_WALL_TIME = "total_wall_time"
    TIME_TO_CONVERGENCE = "time_to_convergence"
    FINAL_STAGE_INDEX = "final_stage_index"
    FINAL_CHOICE_UTILITY = "final_choice_utility"
    CHOICE_TOTAL_PREASSIGNMENT_UTILITY = "choice_total_preassignment_utility"
    CHOICE_AVG_STUDENT_DISTANCE = "choice_avg_student_distance"
    CHOICE_SCHOOLS_ABOVE_10PCT_DISTRICT_FRL = "choice_schools_above_10pct_district_frl"
    CHOICE_FRL_DISSIMILARITY = "choice_frl_dissimilarity"
    CHOICE_PERCENT_UNASSIGNED = "choice_percent_unassigned"
    CHOICE_PERCENT_DESIGNATED = "choice_percent_designated"
    CHOICE_PERCENT_TOP_1 = "choice_percent_top_1"
    CHOICE_PERCENT_TOP_3 = "choice_percent_top_3"
    CHOICE_TOTAL_MNL_UTILITY = "choice_total_mnl_utility"

    @staticmethod
    def program_column(ptype: str) -> str:
        """Generate column name for a specific program type."""
        return f"avg_{ptype}_per_zone"


# ============================================================================
# METRIC CATEGORIES
# ============================================================================

CATEGORIES = {
    "diversity": "Demographics and Economic Balance",
    "proximity": "Geographic Access and Proximity",
    "programs": "Educational Program Availability",
    "quality": "School Quality Indicators",
    "structure": "Zone Structure and Shape",
    "choice": "Student Choice and Assignment Outcomes",
    "run": "Optimization Run Metadata",
}

CATEGORY_DESCRIPTIONS = {
    "diversity": "Measures how evenly demographics are distributed across zones.",
    "proximity": "Measures geographic access to schools within zones.",
    "programs": "Measure how hard it is for students to access programs within their zone.",
    "quality": "Measures how evenly school quality is distributed across zones.",
    "structure": "Structural properties of the zone configuration including shape compactness and zone count.",
    "choice": "Measures preassignment school-choice utility and assignment outcomes after students are matched to schools under a matching policy.",
    "run": "Solver and strategy outputs for the optimization run, including objectives and timing.",
}


# ============================================================================
# DIVERSITY METRICS
# ============================================================================

DIVERSITY_METRICS = [
    MetricSpec(
        column="aalpi_mad",
        display_name="Racial Diversity",
        description="Mean absolute deviation of zone-level AALPI share (Black + Hispanic/Latinx + Pacific Islander) from the district-wide AALPI share. For each non-empty zone we compute |zone_AALPI_proportion - district_AALPI_proportion|, then average across zones. 0 = every zone matches the district AALPI composition; higher = AALPI students concentrated in fewer zones. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=True,
        short_name="Racial",
    ),
    MetricSpec(
        column="frl_mad",
        display_name="Socioeconomic Diversity",
        description="Mean absolute deviation of zone-level FRL share from the district-wide FRL share. For each zone we compute |zone_FRL_proportion - district_FRL_proportion|, then average across zones. 0 = every zone matches the district FRL composition, higher = larger imbalance. Range 0-1 (proportion points).",
        category="diversity",
        direction="minimize",
        is_core=True,
        short_name="SES",
    ),
    MetricSpec(
        column="black_mad",
        display_name="Black Representation",
        description="Mean absolute deviation of zone-level Black/African American share from the district-wide share. For each zone we compute |zone_proportion - district_proportion|, then average across zones. 0 = every zone matches the district Black/African American composition, higher = larger imbalance. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="hispanic_mad",
        display_name="Hispanic/Latinx Representation",
        description="Mean absolute deviation of zone-level Hispanic/Latinx share from the district-wide share. For each zone we compute |zone_proportion - district_proportion|, then average across zones. 0 = every zone matches the district Hispanic/Latinx composition, higher = larger imbalance. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="white_mad",
        display_name="White Representation",
        description="Mean absolute deviation of zone-level White share from the district-wide share. For each zone we compute |zone_proportion - district_proportion|, then average across zones. 0 = every zone matches the district White composition, higher = larger imbalance. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="asian_mad",
        display_name="Asian Representation",
        description="Mean absolute deviation of zone-level Asian share from the district-wide share. For each zone we compute |zone_proportion - district_proportion|, then average across zones. 0 = every zone matches the district Asian composition, higher = larger imbalance. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="pacific_islander_mad",
        display_name="Pacific Islander Representation",
        description="Mean absolute deviation of zone-level Pacific Islander share from the district-wide share. For each zone we compute |zone_proportion - district_proportion|, then average across zones. 0 = every zone matches the district Pacific Islander composition, higher = larger imbalance. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    # Range metrics (max - min percentage across zones)
    MetricSpec(
        column="aalpi_range",
        display_name="AALPI Range",
        description="Range (max - min) of zone-level AALPI share (Black + Hispanic/Latinx + Pacific Islander) across all non-empty zones. Shows the full spread of AALPI representation. 0 = all zones identical; higher = wider gap between most and least AALPI-concentrated zones. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="frl_range",
        display_name="SES Range",
        description="Range (max - min) of zone-level Free/Reduced Lunch share across all non-empty zones. Shows the full spread of socioeconomic composition. 0 = all zones identical; higher = wider gap between most and least FRL-concentrated zones. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="black_range",
        display_name="Black Range",
        description="Range (max - min) of zone-level Black/African American share across all non-empty zones. 0 = all zones identical; higher = wider gap. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="hispanic_range",
        display_name="Hispanic/Latinx Range",
        description="Range (max - min) of zone-level Hispanic/Latinx share across all non-empty zones. 0 = all zones identical; higher = wider gap. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="white_range",
        display_name="White Range",
        description="Range (max - min) of zone-level White share across all non-empty zones. 0 = all zones identical; higher = wider gap. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="asian_range",
        display_name="Asian Range",
        description="Range (max - min) of zone-level Asian share across all non-empty zones. 0 = all zones identical; higher = wider gap. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="pacific_islander_range",
        display_name="Pacific Islander Range",
        description="Range (max - min) of zone-level Pacific Islander share across all non-empty zones. 0 = all zones identical; higher = wider gap. Range 0-1.",
        category="diversity",
        direction="minimize",
        is_core=False,
    ),
]


# ============================================================================
# PROXIMITY METRICS
# ============================================================================

PROXIMITY_METRICS = [
    MetricSpec(
        column="avg_any_zone_ge_school_distance",
        display_name="Avg Distance to Any In-Zone GE School",
        description="Average distance (miles) from each area to all GE schools in its zone, averaged across all areas. Lower means GE schools are closer on average.",
        category="proximity",
        direction="minimize",
        is_core=True,
        short_name="Avg GE Distance",
    ),
    MetricSpec(
        column="avg_farthest_zone_ge_school_distance",
        display_name="Avg Distance to Farthest In-Zone GE School",
        description="Average distance (miles) from each area to the farthest GE school in its zone, averaged across all areas. Lower means zones are more compact around GE schools.",
        category="proximity",
        direction="minimize",
        is_core=True,
        short_name="Farthest GE Distance",
    ),
    MetricSpec(
        column="avg_out_of_zone_ge_schools_within_half_mile",
        display_name="Nearby Out-of-Zone GE Schools",
        description="Average number of GE schools within 0.5 miles of each area that are in a different zone. Lower means nearby GE schools are in the same zone as the area.",
        category="proximity",
        direction="minimize",
        is_core=True,
        short_name="Out-of-Zone GE",
    ),

    MetricSpec(
        column="avg_ge_schools_within_half_mile",
        display_name="In-Zone GE Programs Within 0.5 Miles",
        description="Average number of General Education programs within 0.5 miles of each student. Higher means students have more nearby GE program options.",
        category="proximity",
        direction="maximize",
        is_core=True,
        short_name="Walkable Schools",
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
    # ),
    # MetricSpec(
    #     column="avg_language_immersion_per_zone",
    #     display_name="Language Immersion Programs",
    #     description="Avg language immersion programs per zone.",
    #     category="programs",
    #     direction="maximize",
    #     is_core=True,
    # ),
    # MetricSpec(
    #     column="avg_special_ed_per_zone",
    #     display_name="Special Education Programs",
    #     description="Avg special education programs per zone",
    #     category="programs",
    #     direction="maximize",
    #     is_core=True,
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
    MetricSpec(
        column="cut_edges",
        display_name="Cut Edges",
        description="Number of Block_0 graph edges whose endpoints are assigned to different zones after converting the selected final solution to Block_0. Lower means fewer boundaries between adjacent base areas.",
        category="structure",
        direction="minimize",
        is_core=False,
        short_name="Cuts",
    ),
    MetricSpec(
        column="normalized_cut_edges",
        display_name="Normalized Cut Edges",
        description="Number of Block_0 graph edges cut by zone boundaries after converting the selected final solution to Block_0, divided by the number of zones. Lower is better.",
        category="structure",
        direction="minimize",
        is_core=False,
        short_name="Cuts/Zone",
    ),
    MetricSpec(
        column="avg_reock_score",
        display_name="Average Reock Score",
        description="Average Reock compactness score across zones. For each zone, area is divided by the area of its minimum enclosing circle. Higher means zones are more compact. Range 0-1.",
        category="structure",
        direction="maximize",
        is_core=False,
        short_name="Reock",
    ),
    MetricSpec(
        column="avg_polsby_popper_score",
        display_name="Average Polsby-Popper Score",
        description="Average Polsby-Popper compactness score across zones. For each zone, 4*pi*area/perimeter^2 is computed from projected zone geometry. Higher means zones are more compact. Range 0-1.",
        category="structure",
        direction="maximize",
        is_core=False,
        short_name="PP",
    ),
    MetricSpec(
        column="num_zones",
        display_name="Number of Zones",
        description="Total number of zones in this map.",
        category="structure",
        is_core=False,
        short_name="Zones",
    ),
    MetricSpec(
        column="contiguous",
        display_name="Contiguous",
        description="1 if every zone forms a single connected geographic region anchored on its centroid school; 0 otherwise. A non-contiguous solution typically means the solver returned a relaxed or partially infeasible result.",
        category="structure",
        direction="maximize",
        is_core=False,
        short_name="Contig",
    ),
    MetricSpec(
        column="solution_code",
        display_name="Solution Code",
        description="7-character base36 hash uniquely identifying this zoning partition. Deterministic from the area-to-zone mapping; useful for citing or comparing specific solutions.",
        category="structure",
        direction=None,
        is_core=False,
        short_name="Code",
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
    ),
    MetricSpec(
        column="mad_eng_score",
        display_name="English Score Equity",
        description="Mean absolute deviation of average school English scores across zones. Lower means more equitable English quality distribution.",
        category="quality",
        direction="minimize",
        is_core=True,
        short_name="Eng",
    ),
    # Range metrics for quality scores
    MetricSpec(
        column="math_score_range",
        display_name="Math Score Range",
        description="Range (max - min) of capacity-weighted average math scores across zones. Shows the full spread of math quality. Lower means more equitable distribution.",
        category="quality",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="eng_score_range",
        display_name="English Score Range",
        description="Range (max - min) of capacity-weighted average English scores across zones. Shows the full spread of English quality. Lower means more equitable distribution.",
        category="quality",
        direction="minimize",
        is_core=False,
    ),
]


# ============================================================================
# CHOICE METRICS (student-assignment outcomes)
# ============================================================================

CHOICE_METRICS = [
    MetricSpec(
        column="choice_total_preassignment_utility",
        display_name="Preassignment Choice Utility",
        description="Total student utility from the schools available in each student's assigned zone before any assignment/matching simulation is run. Computed with the configured choice_model and, for MNL, choice_model_method (logsum or max). Higher is better.",
        category="choice",
        direction="maximize",
        is_core=False,
        short_name="Preassign Utility",
    ),
    MetricSpec(
        column="choice_avg_student_distance",
        display_name="Average Assigned Student Distance",
        description="Average distance in miles from assigned students to their matched school/program, averaged across assignment simulations when multiple assignment CSVs are present. Lower means students are assigned closer to home.",
        category="choice",
        direction="minimize",
        is_core=False,
        short_name="Choice Dist",
    ),
    MetricSpec(
        column="choice_schools_above_10pct_district_frl",
        display_name="Schools Above District FRL +10%",
        description="Fraction of schools with assigned-student FRL at least 10 percentage points above participating-student district FRL, averaged across assignment simulations. Lower means fewer schools concentrate higher-FRL students.",
        category="choice",
        direction="minimize",
        is_core=False,
        short_name="High FRL Schools",
    ),
    MetricSpec(
        column="choice_frl_dissimilarity",
        display_name="FRL Dissimilarity Across Schools",
        description="Weighted FRL dissimilarity across assigned school populations using freelunch_prob + reducedlunch_prob as each student's FRL weight. Lower means FRL students are more evenly distributed across schools.",
        category="choice",
        direction="minimize",
        is_core=False,
        short_name="FRL Dissimil.",
    ),
    MetricSpec(
        column="choice_percent_unassigned",
        display_name="Unassigned Students",
        description="Fraction of participating students not assigned to any program, averaged across assignment simulations. Lower is better.",
        category="choice",
        direction="minimize",
        is_core=False,
        short_name="Unassigned",
    ),
    MetricSpec(
        column="choice_percent_designated",
        display_name="Designated Assignments",
        description="Fraction of assigned students who were assigned through designation, averaged across assignment simulations. Lower means fewer assigned students needed designation to receive a placement.",
        category="choice",
        direction="minimize",
        is_core=False,
        short_name="Designated",
    ),
    MetricSpec(
        column="choice_percent_top_1",
        display_name="Top 1 Choice",
        description="Fraction of assigned students who received their first-ranked choice, averaged across assignment simulations. Higher is better.",
        category="choice",
        direction="maximize",
        is_core=False,
        short_name="Top 1",
    ),
    MetricSpec(
        column="choice_percent_top_3",
        display_name="Top 3 Choice",
        description="Fraction of assigned students who received one of their top three ranked choices, averaged across assignment simulations. Higher is better.",
        category="choice",
        direction="maximize",
        is_core=False,
        short_name="Top 3",
    ),
    MetricSpec(
        column="choice_total_mnl_utility",
        display_name="Total MNL Utility",
        description="Sum of assigned_utility from the MNL utility model across students in each assignment simulation, averaged across assignment simulations when multiple assignment CSVs are present. Higher is better. Available only when matching writes assigned_utility.",
        category="choice",
        direction="maximize",
        is_core=False,
        short_name="MNL Utility",
    ),
]


# ============================================================================
# RUN METRICS (solver/strategy metadata)
# ============================================================================

RUN_METRICS = [
    MetricSpec(
        column="final_objective",
        display_name="Final Objective",
        description="Objective value reported by the solver for the selected final solution. This is solver metadata; use cut_edges for the normalized structural boundary metric.",
        category="run",
        direction=None,
        is_core=False,
        short_name="Obj",
    ),
    MetricSpec(
        column="final_cut_edges",
        display_name="Final Cut Edges",
        description="Block_0-normalized cut edge count for the selected final solution, repeated in run metadata for comparing against solver-reported objectives.",
        category="run",
        direction=None,
        is_core=False,
        short_name="Final Cuts",
    ),
    MetricSpec(
        column="total_wall_time",
        display_name="Total Wall Time",
        description="Total solver wall time across all stages in the run.",
        category="run",
        direction="minimize",
        is_core=False,
        short_name="Time",
    ),
    MetricSpec(
        column="final_wall_time",
        display_name="Final Stage Wall Time",
        description="Solver wall time for the selected final stage.",
        category="run",
        direction="minimize",
        is_core=False,
    ),
    MetricSpec(
        column="time_to_convergence",
        display_name="Time To Convergence",
        description="Time until the solver first found a feasible solution. For recursive runs this sums stage convergence times; for iterative-choice runs this uses the first iteration only.",
        category="run",
        direction="minimize",
        is_core=False,
        short_name="Converge",
    ),
    MetricSpec(
        column="final_status",
        display_name="Final Status",
        description="Solver status for the selected final solution.",
        category="run",
        direction=None,
        is_core=False,
    ),
    MetricSpec(
        column="final_choice_utility",
        display_name="Final Choice Utility",
        description="Choice utility attached by an iterative choice strategy, when available. Higher means better according to that strategy's choice model.",
        category="run",
        direction="maximize",
        is_core=False,
    ),
]


# ============================================================================
# AGGREGATED METRIC REGISTRY
# ============================================================================

ALL_METRICS: list[MetricSpec] = (
    DIVERSITY_METRICS
    + PROXIMITY_METRICS
    + PROGRAM_METRICS
    + QUALITY_METRICS
    + STRUCTURE_METRICS
    + CHOICE_METRICS
    + RUN_METRICS
)

# Build lookup dictionaries
METRIC_BY_COLUMN: dict[str, MetricSpec] = {m.column: m for m in ALL_METRICS}
METRIC_BY_NAME: dict[str, MetricSpec] = {m.display_name: m for m in ALL_METRICS}

# Core metrics for default summaries and filters.
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
    """Generate a markdown summary of all metrics."""
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
