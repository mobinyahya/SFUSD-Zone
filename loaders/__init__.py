"""Shared, scenario-driven data loading and caching."""

from loaders.cache import CacheNamespace, CacheStore, identity_fingerprint
from loaders.config import (
    DataScenario,
    ResolvedSource,
    anchor_data_config,
    load_scenario,
)
from loaders.edge_overrides import (
    apply_block_edge_overrides,
    block_edge_override_fingerprint,
    load_block_edge_overrides,
)
from loaders.geography import (
    GEOGRAPHY_COLUMNS,
    GEOGRAPHY_UNITS,
    load_census_geometry,
    load_geography_crosswalk,
    match_points_to_census,
    normalize_census_geography,
    selected_geography_vintage,
)
from loaders.tables import (
    SPECIAL_PROGRAMS,
    apply_capacity_scenario,
    filter_outside_district_students,
    load_program_records,
    load_school_records,
    load_student_records,
    normalize_school_records,
    normalize_student_records,
    normalize_grade,
    parse_ranked_programs,
    parse_ranked_schools,
    read_csv,
    read_csv_source,
    school_id_aliases,
)

__all__ = [
    "CacheNamespace",
    "CacheStore",
    "DataScenario",
    "GEOGRAPHY_COLUMNS",
    "GEOGRAPHY_UNITS",
    "ResolvedSource",
    "SPECIAL_PROGRAMS",
    "anchor_data_config",
    "apply_capacity_scenario",
    "apply_block_edge_overrides",
    "block_edge_override_fingerprint",
    "filter_outside_district_students",
    "identity_fingerprint",
    "load_block_edge_overrides",
    "load_census_geometry",
    "load_geography_crosswalk",
    "load_program_records",
    "load_scenario",
    "load_school_records",
    "load_student_records",
    "match_points_to_census",
    "normalize_census_geography",
    "normalize_school_records",
    "normalize_student_records",
    "normalize_grade",
    "parse_ranked_programs",
    "parse_ranked_schools",
    "read_csv",
    "read_csv_source",
    "school_id_aliases",
    "selected_geography_vintage",
]
