"""Column names of the flat metrics each metric module writes.

The spatial columns are the field names of
:class:`metrics.spatial.SpatialMetrics` (``SPATIAL_FIELDS``).
"""


class MetricColumns:
    """Flat metric column names; metrics/ is their single owner."""

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
    AVG_TOTAL_PROGRAMS = "avg_total_programs_per_zone"
    AVG_LANGUAGE_IMMERSION = "avg_language_immersion_per_zone"
    AVG_SPECIAL_ED = "avg_special_ed_per_zone"
    MAD_MATH_SCORE = "mad_math_score"
    MAD_ENG_SCORE = "mad_eng_score"
    MATH_SCORE_RANGE = "math_score_range"
    ENG_SCORE_RANGE = "eng_score_range"
    AVG_GE_SCHOOLS_WITHIN_HALF_MILE = "avg_ge_schools_within_half_mile"
    GE_WEIGHTED_AVG_NEAREST_GE_SCHOOL_DISTANCE = (
        "ge_weighted_avg_nearest_ge_school_distance"
    )
    MAX_GE_POPULATED_NODE_NEAREST_GE_SCHOOL_DISTANCE = (
        "max_ge_populated_node_nearest_ge_school_distance"
    )
    GE_STUDENTS_WITHOUT_IN_ZONE_GE_SCHOOL = "ge_students_without_in_zone_ge_school"
    WEIGHTED_AVG_NEAREST_GE_SCHOOL_DISTANCE = "weighted_avg_nearest_ge_school_distance"
    MAX_POPULATED_NODE_NEAREST_GE_SCHOOL_DISTANCE = (
        "max_populated_node_nearest_ge_school_distance"
    )
    STUDENTS_WITHOUT_IN_ZONE_GE_SCHOOL = "students_without_in_zone_ge_school"
    NUM_ZONES = "num_zones"
    CONTIGUOUS = "contiguous"
    SOLUTION_CODE = "solution_code"
    FINAL_OBJECTIVE = "final_objective"
    FINAL_CUT_EDGES = "final_cut_edges"
    FINAL_STATUS = "final_status"
    FINAL_WALL_TIME = "final_wall_time"
    TOTAL_WALL_TIME = "total_wall_time"
    FINAL_STAGE_INDEX = "final_stage_index"
    FINAL_CHOICE_UTILITY = "final_choice_utility"
    CHOICE_TOTAL_PREASSIGNMENT_UTILITY = "choice_total_preassignment_utility"

    @staticmethod
    def program_column(ptype: str) -> str:
        """Generate column name for a specific program type."""
        return f"avg_{ptype}_per_zone"
