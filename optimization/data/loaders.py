"""Scenario-driven ingestion and optimization-specific table transformations.

Raw source selection, path resolution, CSV reading, common normalization, and
cache storage come from the top-level :mod:`loaders` package. This module owns
only the transformations needed to build optimization area tables and graphs.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from Config.Constants import (
    AREA_COLS,
    AREA_ETHNICITIES,
    AUX_BG,
    BUILDING_BLOCKS,
    ETHNICITY_COLS,
    ETHNICITY_DICT,
    IMPORTANT_COLS,
    K8_SCHOOLS,
)
from loaders import (
    CacheNamespace,
    CacheStore,
    DataScenario,
    GEOGRAPHY_UNITS,
    apply_student_frl_estimate,
    filter_outside_district_students,
    load_census_geometry,
    load_scenario,
    normalize_census_geography,
)
from loaders.edge_overrides import load_block_edge_overrides
from loaders.tables import (
    load_program_records,
    load_school_records,
    normalize_student_records,
    read_csv,
    read_csv_source,
    school_id_aliases,
)
from optimization.data.geography import EARTH_RADIUS_MILES

PROJECTED_CENTROID_CRS = "EPSG:32610"  # San Francisco is in UTM zone 10N.
OUTPUT_LATLON_CRS = "EPSG:4326"
STUDENT_CACHE_SCHEMA_VERSION = 8
AREA_DISTANCE_CACHE_SCHEMA_VERSION = 3

STUDENT_ROLE = "optimization.students"
FRL_ESTIMATE_ROLE = "optimization.frl_estimate"
SCHOOL_ROLE = "optimization.schools"
PROGRAM_ROLE = "optimization.programs"
CAPACITY_ROLE = "optimization.capacity"
CENSUS_ROLE = "optimization.census"
CROSSWALK_ROLE = "optimization.crosswalk"
ADJACENCY_ROLE = "optimization.adjacency"
CENTROID_ROLE = "optimization.centroids"
MANUAL_EDGE_ROLE = "optimization.manual_edges"

_ACADEMIC_YEAR = re.compile(r"(?<!\d)(\d{2})(\d{2})(?!\d)")


def census_geometry_roles(data: DataScenario, unit: str) -> list[str]:
    """Return every source role used to build one unit's Census geometry."""
    roles = [CENSUS_ROLE, CROSSWALK_ROLE]
    parent_role = {
        "BlockGroup": "optimization.geography.blockgroups",
        "Tract": "optimization.geography.tracts",
    }.get(unit)
    if parent_role is not None:
        try:
            data.resolved(parent_role)
        except KeyError:
            pass
        else:
            roles.append(parent_role)
    return roles


def capacity_source_roles(cfg: IngestConfig) -> list[str]:
    """Return capacity inputs that affect this optimization graph."""
    roles = [PROGRAM_ROLE]
    if cfg.capacity_scenario != "programs":
        roles.append(CAPACITY_ROLE)
    return roles


# Maps post-dummy ethnicity columns to the graph's canonical names.
_ETHNICITY_RENAME = {
    "resolved_ethnicity_American Indian": "Ethnicity_American_Indian",
    "resolved_ethnicity_Asian": "Ethnicity_Asian",
    "resolved_ethnicity_Black or African American": (
        "Ethnicity_Black_or_African_American"
    ),
    "resolved_ethnicity_Filipino": "Ethnicity_Filipino",
    "resolved_ethnicity_Pacific Islander": "Ethnicity_PacificIslander",
    "resolved_ethnicity_Hispanic/Latinx": "Ethnicity_Hispanic/Latinx",
    "resolved_ethnicity_Two or More Races": "Ethnicity_Two_or_More_Races",
    "resolved_ethnicity_White": "Ethnicity_White",
}


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """The geographic unit and immutable data scenario used by ingestion."""

    unit: str
    data: DataScenario

    def __post_init__(self) -> None:
        if self.unit not in ("Block", "BlockGroup", "Tract"):
            raise ValueError(
                f"Ingestion supports Block/BlockGroup/Tract, got {self.unit!r}."
            )
        if not isinstance(self.data, DataScenario):
            raise TypeError("IngestConfig.data must be a DataScenario.")

    @property
    def filters(self) -> Mapping[str, Any]:
        return self.data.filters["optimization"]

    @property
    def years(self) -> tuple[str, ...]:
        return tuple(self.data.filter("optimization", "years"))

    @property
    def grades(self) -> tuple[str, ...]:
        return tuple(self.data.filter("optimization", "grades"))

    @property
    def student_population(self) -> str:
        return self.data.filter("optimization", "student_population")

    @property
    def rounds(self) -> Any:
        return self.data.filter("optimization", "rounds", None)

    @property
    def special_programs(self) -> str:
        return self.data.filter("optimization", "special_programs")

    @property
    def program_population(self) -> str:
        return self.data.filter("optimization", "program_population")

    @property
    def capacity_scenario(self) -> str:
        return self.data.filter("optimization", "capacity_scenario")

    @property
    def include_k8(self) -> bool:
        return self.data.filter("optimization", "include_k8")

    @property
    def include_citywide(self) -> bool:
        return self.data.filter("optimization", "include_citywide")

    @property
    def include_mission_bay(self) -> bool:
        return self.data.filter("optimization", "include_mission_bay")

    @property
    def outside_district_students(self) -> str:
        return self.data.filter("optimization", "outside_district_students")

    @property
    def frl_estimate(self) -> str | None:
        return self.data.filter("optimization", "frl_estimate")


def _legacy_scenario() -> DataScenario:
    return load_scenario({"scenario": "legacy", "overrides": {}})


# ====================================================================== #
# Students
# ====================================================================== #
def load_students(cfg: IngestConfig) -> pd.DataFrame:
    """Load, transform, and cache student rows across configured school years."""
    sources = _student_sources_by_year(cfg)
    namespace = _student_cache_namespace(cfg)
    cached = namespace.load_dataframe("students.csv", low_memory=False)
    if cached is not None:
        return cached

    frames = [_load_students_for_year(cfg, source, year) for year, source in sources]
    students = pd.concat(frames, ignore_index=True, sort=False)
    namespace.save_dataframe("students.csv", students)
    return students


def _student_cache_namespace(cfg: IngestConfig) -> CacheNamespace:
    return CacheStore(cfg.data).namespace(
        "students",
        {
            "source_role": STUDENT_ROLE,
            "optimization_filters": cfg.filters,
        },
        schema_version=STUDENT_CACHE_SCHEMA_VERSION,
        roles=student_source_roles(cfg),
        classification="restricted-derived",
    )


def _student_cache_path(cfg: IngestConfig) -> str:
    """Return the v8 student CSV path for cache introspection."""
    return str(_student_cache_namespace(cfg).payload_path("students.csv"))


def student_source_roles(cfg: IngestConfig) -> list[str]:
    """Return student inputs that affect normalized optimization rows."""
    roles = [STUDENT_ROLE]
    if cfg.frl_estimate is not None:
        roles.append(FRL_ESTIMATE_ROLE)
    return roles


def _student_sources_by_year(cfg: IngestConfig):
    sources = cfg.data.sources(STUDENT_ROLE)
    if len(sources) != len(cfg.years):
        raise ValueError(
            f"{STUDENT_ROLE} has {len(sources)} sources for "
            f"{len(cfg.years)} configured years."
        )

    aligned = []
    for year, source in zip(cfg.years, sources):
        labels = [source.path.name]
        if source.catalog_id is not None:
            labels.append(source.catalog_id)
        recognized = {
            match.group(0)
            for label in labels
            for match in _ACADEMIC_YEAR.finditer(label)
            if int(match.group(2)) == (int(match.group(1)) + 1) % 100
        }
        if recognized != {year}:
            raise ValueError(
                f"Student source {source.path} does not align with configured "
                f"canonical year {year!r}; found school years {sorted(recognized)}."
            )
        aligned.append((year, source))
    return tuple(aligned)


def _load_students_for_year(cfg: IngestConfig, source, year: str) -> pd.DataFrame:
    frame = read_csv_source(source, low_memory=False)
    # Some legacy exports repeat rows with complementary null fields. Coalesce
    # only compatible copies; shared normalization still rejects conflicts.
    frame = frame.drop_duplicates(ignore_index=True)
    if "studentno" in frame.columns:
        duplicate_rows = frame.loc[frame["studentno"].duplicated(keep=False)]
        drop_indices = []
        for _, group in duplicate_rows.groupby("studentno", sort=False, dropna=False):
            merged = group.iloc[0].copy()
            compatible = True
            for column in frame.columns:
                values = group[column].dropna().unique()
                if len(values) > 1:
                    compatible = False
                    break
                if len(values) == 1:
                    merged[column] = values[0]
            if compatible:
                frame.loc[group.index[0]] = merged
                drop_indices.extend(group.index[1:])
        frame = frame.drop(index=drop_indices).reset_index(drop=True)
    frame = normalize_student_records(frame, cfg.data, "optimization")
    frame = normalize_census_geography(
        frame,
        cfg.data,
        "optimization",
        source_vintage=source.geography_vintage,
        style="student",
    )
    frame = apply_student_frl_estimate(frame, cfg.data, "optimization")
    frame = filter_outside_district_students(frame, cfg.data, "optimization")
    frame.rename(
        columns={
            "census_block": "Block",
            "census_blockgroup": "BlockGroup",
            "census_tract": "Tract",
            "idschoolattendance": "attendance_area",
            "FRL Score": "FRL",
        },
        inplace=True,
    )
    missing = set(BUILDING_BLOCKS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Student source {source.path} is missing columns: {sorted(missing)}."
        )
    frame["enrolled_students"] = np.where(frame["enrolled_idschool"].isna(), 0, 1)
    frame["participating_programs"] = frame["selected_programs"].map(list)
    frame["program_types"] = frame["selected_programs"].map(
        lambda programs: np.unique(programs)
    )
    frame["all_prog_students"] = 1
    frame["ge_students"] = frame.apply(
        lambda row: (
            sum(program == "GE" for program in row["participating_programs"])
            / len(row["participating_programs"])
            if row["enrolled_students"] == 1 and row["participating_programs"]
            else 0
        ),
        axis=1,
    )

    if "resolved_ethnicity" not in frame.columns:
        raise ValueError(
            "Student data is missing required column 'resolved_ethnicity'."
        )
    frame["resolved_ethnicity"] = frame["resolved_ethnicity"].replace(ETHNICITY_DICT)
    frame = pd.get_dummies(frame, columns=["resolved_ethnicity"])

    population_weight = (
        frame["ge_students"]
        if cfg.program_population == "GE"
        else frame["all_prog_students"]
    )
    for column in ETHNICITY_COLS + ["FRL"]:
        if column in frame.columns:
            frame[column] = frame[column] * population_weight

    frame = _filter_to_population(frame, cfg.program_population, year)
    keep = [
        column for column in IMPORTANT_COLS + ETHNICITY_COLS if column in frame.columns
    ]
    frame = frame[keep]
    for column in ["FRL", "AALPI Score"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna(frame[column].mean())
    fill_columns = [column for column in frame.columns if column not in GEOGRAPHY_UNITS]
    frame[fill_columns] = frame[fill_columns].fillna(value=0)
    frame.rename(columns=_ETHNICITY_RENAME, inplace=True)
    for ethnicity in AREA_ETHNICITIES:
        if ethnicity not in frame.columns:
            frame[ethnicity] = 0.0
    return frame


def _filter_to_population(
    frame: pd.DataFrame, program_population: str, year: str
) -> pd.DataFrame:
    if program_population == "All":
        return frame
    if program_population == "GE":
        if year != "1819":
            frame["filter"] = frame["program_types"].map(
                lambda programs: int("GE" in programs)
            )
        else:

            def keep(row):
                if (
                    row["r1_idschool"] == row["enrolled_idschool"]
                    and row["r1_programcode"] == "GE"
                ):
                    return 1
                if (
                    row["r3_idschool"] == row["enrolled_idschool"]
                    and row["r3_programcode"] == "GE"
                ):
                    return 1
                if (
                    row["r1_idschool"] != row["enrolled_idschool"]
                    and row["r3_idschool"] != row["enrolled_idschool"]
                ):
                    return 1
                return 0

            frame["filter"] = frame.apply(keep, axis=1)
    else:
        frame["filter"] = frame["program_types"].map(
            lambda programs: int(program_population in programs)
        )
    return frame.loc[frame["filter"] == 1]


# ====================================================================== #
# Schools
# ====================================================================== #
def load_schools(cfg: IngestConfig) -> pd.DataFrame:
    """Return scenario schools with capacities and optimization attributes."""
    frame = load_school_records(
        cfg.data,
        SCHOOL_ROLE,
        filter_group="optimization",
        low_memory=False,
    )
    missing = {"school_id", cfg.unit} - set(frame.columns)
    if missing:
        raise ValueError(f"School table is missing columns: {sorted(missing)}.")
    frame[cfg.unit] = frame[cfg.unit].astype("Int64")
    frame["school_id"] = frame["school_id"].astype(int)
    frame["K-8"] = frame["school_id"].map(lambda school: int(school in K8_SCHOOLS))
    frame = _attach_capacity(frame, cfg)
    frame["num_schools"] = 1
    frame.rename(
        columns={
            "eng_scores_1819": "english_score",
            "math_scores_1819": "math_score",
        },
        inplace=True,
    )
    return frame


def load_school_locations(cfg: IngestConfig) -> pd.DataFrame:
    """Return canonical school IDs and raw locations before capacity filtering."""
    frame = load_school_records(
        cfg.data,
        SCHOOL_ROLE,
        filter_group="optimization",
        low_memory=False,
    )
    missing = {"school_id", cfg.unit} - set(frame.columns)
    if missing:
        raise ValueError(
            f"School location table is missing columns: {sorted(missing)}."
        )
    frame = frame[["school_id", cfg.unit]].dropna().copy()
    frame["school_id"] = frame["school_id"].astype(int)
    frame[cfg.unit] = frame[cfg.unit].astype("int64")
    return frame


def load_school_coordinates(data: DataScenario | None = None) -> pd.DataFrame:
    """Return canonical scenario school points for geometry artifacts."""
    scenario = data or _legacy_scenario()
    frame = load_school_records(
        scenario,
        SCHOOL_ROLE,
        filter_group="optimization",
        low_memory=False,
    )
    required = {"school_id", "lat", "lon"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"School coordinate table is missing columns: {sorted(missing)}."
        )
    coordinates = frame[["school_id", "lat", "lon"]].dropna().copy()
    coordinates["school_id"] = coordinates["school_id"].astype(int)
    if coordinates["school_id"].duplicated().any():
        duplicates = sorted(
            coordinates.loc[
                coordinates["school_id"].duplicated(False), "school_id"
            ].unique()
        )
        raise ValueError(f"Duplicate school coordinates for IDs: {duplicates}.")
    return coordinates


def _attach_capacity(frame: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    programs = load_program_records(
        cfg.data,
        PROGRAM_ROLE,
        filter_group="optimization",
        low_memory=False,
    )
    missing = {"school_id", "program_type", "capacity"} - set(programs.columns)
    if missing:
        raise ValueError(
            f"Optimization program table is missing columns: {sorted(missing)}."
        )
    programs["school_id"] = programs["school_id"].astype(int)
    programs["capacity"] = pd.to_numeric(programs["capacity"], errors="coerce").fillna(
        0
    )

    ge = (
        programs.loc[programs["program_type"] == "GE", ["school_id", "capacity"]]
        .groupby("school_id", as_index=False)
        .sum()
        .rename(columns={"capacity": "ge_capacity"})
    )
    all_programs = programs[["school_id", "capacity"]].rename(
        columns={"capacity": "all_prog_capacity"}
    )
    all_programs = all_programs.groupby("school_id", as_index=False).sum()

    frame = frame.merge(all_programs, how="outer", on="school_id")
    frame = frame.merge(ge, how="outer", on="school_id")
    if cfg.capacity_scenario == "Closure" or cfg.program_population == "All":
        frame = frame.loc[frame["all_prog_capacity"] > 0]
    else:
        frame = frame.loc[frame["ge_capacity"] > 0]
    if cfg.program_population != "All" and not cfg.include_k8:
        frame = frame.loc[frame["K-8"] == 0]
    if cfg.program_population != "All" or not cfg.include_citywide:
        frame = frame.loc[frame["category"] != "Citywide"]
    return frame


# ====================================================================== #
# Per-area aggregation
# ====================================================================== #
def load_area_table(cfg: IngestConfig) -> pd.DataFrame:
    """Build one optimization row per scenario-backed geographic area."""
    students = load_students(cfg)
    missing_geography = students[cfg.unit].isna()
    if missing_geography.any():
        identities = students.loc[missing_geography, "studentno"].head(10).tolist()
        raise ValueError(
            f"Cannot construct a {cfg.unit} graph with "
            f"{int(missing_geography.sum())} included students that have no "
            f"district {cfg.unit}; students include {identities}. Set "
            "outside_district_students to 'ignore' to filter them."
        )
    schools = load_schools(cfg)
    student_stats = _aggregate_students(students, cfg)
    school_stats = _aggregate_schools(schools, cfg.unit)

    student_stats[cfg.unit] = student_stats[cfg.unit].astype(int)
    area = student_stats.merge(school_stats, how="outer", on=cfg.unit)
    area = _add_missing_areas(area, cfg)
    area.fillna(value=0, inplace=True)
    if cfg.capacity_scenario == "Closure":
        area = _apply_closure_scaling(area)
    area[cfg.unit] = area[cfg.unit].astype("int64")
    area = area.reset_index(drop=True)

    latlon = load_area_latlon(cfg)
    area = area.merge(latlon, how="left", left_on=cfg.unit, right_index=True)
    school_ids = (
        schools.dropna(subset=[cfg.unit]).groupby(cfg.unit)["school_id"].apply(list)
    )
    area["school_ids"] = (
        area[cfg.unit]
        .map(school_ids)
        .map(lambda value: value if isinstance(value, list) else [])
    )
    area[["Lat", "Lon"]] = area[["Lat", "Lon"]].fillna(0.0)
    return area


def _aggregate_students(students: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    stats = students.groupby(cfg.unit, as_index=False).sum(numeric_only=True)
    columns = [
        column
        for column in AREA_COLS + [cfg.unit] + AREA_ETHNICITIES
        if column in stats.columns
    ]
    stats = stats[columns]
    for column in stats.columns:
        if column not in BUILDING_BLOCKS:
            stats[column] = stats[column] / len(cfg.years)
    return stats


def _aggregate_schools(schools: pd.DataFrame, unit: str) -> pd.DataFrame:
    sum_columns = [
        unit,
        "all_prog_capacity",
        "ge_capacity",
        "num_schools",
        "english_score",
        "math_score",
        "greatschools_rating",
        "AvgColorIndex",
    ]
    mean_columns = [unit, "MetStandards"]
    sum_columns = [column for column in sum_columns if column in schools.columns]
    mean_columns = [column for column in mean_columns if column in schools.columns]
    summed = schools[sum_columns].groupby(unit, as_index=False).sum(numeric_only=True)
    if len(mean_columns) > 1:
        meaned = (
            schools[mean_columns].groupby(unit, as_index=False).mean(numeric_only=True)
        )
        return meaned.merge(summed, how="left", on=unit)
    return summed


def _add_missing_areas(area: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    """Append census areas that have neither students nor schools."""
    crosswalk = read_csv(cfg.data, CROSSWALK_ROLE, low_memory=False)
    valid = set(crosswalk[cfg.unit].dropna().astype("int64"))
    census = set(load_census_shapefile(cfg.unit, cfg.data)[cfg.unit]) - set(AUX_BG)
    have = set(area[cfg.unit].astype(int))
    missing = (census & valid) - have
    if missing:
        extra = pd.DataFrame({cfg.unit: sorted(missing)})
        area[cfg.unit] = area[cfg.unit].astype(int)
        extra[cfg.unit] = extra[cfg.unit].astype(int)
        area = pd.merge(area, extra, how="outer", on=cfg.unit)
        area.fillna(value=0, inplace=True)
    return area


def _apply_closure_scaling(area: pd.DataFrame) -> pd.DataFrame:
    ratio = 3700 / area["all_prog_students"].sum()
    area["all_prog_students"] = area["all_prog_students"] * ratio
    area["FRL"] = (3700 / 2460) * area["FRL"]
    for ethnicity in AREA_ETHNICITIES:
        if ethnicity in area.columns:
            area[ethnicity] = (3700 / 2460) * area[ethnicity]
    return area


# ====================================================================== #
# Geometry, distance, and adjacency
# ====================================================================== #
def load_census_shapefile(
    level: str, data: DataScenario | None = None
) -> gpd.GeoDataFrame:
    """Load selected-vintage Census geometry for one optimization unit."""
    scenario = data or _legacy_scenario()
    return load_census_geometry(scenario, "optimization", level)


def load_area_latlon(cfg: IngestConfig) -> pd.DataFrame:
    """Return projected-centroid Lat/Lon for every configured census area."""
    census = load_census_shapefile(cfg.unit, cfg.data)
    dissolved = census.dissolve(by=cfg.unit, as_index=False)
    centroids = _projected_centroids_latlon(dissolved)
    dissolved["Lat"] = centroids.y
    dissolved["Lon"] = centroids.x
    output = dissolved[[cfg.unit, "Lat", "Lon"]].copy()
    output[cfg.unit] = output[cfg.unit].astype("int64")
    return output.set_index(cfg.unit)


def _projected_centroids_latlon(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Return projected centroids represented as WGS84 points."""
    if gdf.crs is None:
        gdf = gdf.set_crs(OUTPUT_LATLON_CRS)
    projected_centroids = gdf.to_crs(PROJECTED_CENTROID_CRS).centroid
    return gpd.GeoSeries(projected_centroids, crs=PROJECTED_CENTROID_CRS).to_crs(
        OUTPUT_LATLON_CRS
    )


def _distance_cache_namespace(
    cfg: IngestConfig, area_ids: Sequence[int]
) -> CacheNamespace:
    roles = census_geometry_roles(cfg.data, cfg.unit)
    if cfg.unit == "Block":
        roles.append(SCHOOL_ROLE)
    return CacheStore(cfg.data).namespace(
        "area_distances",
        {
            "unit": cfg.unit,
            "destination_area_ids": sorted(int(area_id) for area_id in area_ids),
            "optimization_filters": cfg.filters,
            "distance": "great_circle_from_projected_centroids",
            "block_rows": "raw_school_location_areas",
        },
        schema_version=AREA_DISTANCE_CACHE_SCHEMA_VERSION,
        roles=roles,
    )


def load_distance_dict(
    cfg: IngestConfig, area2idx: dict[int, int]
) -> dict[int, dict[int, float]]:
    """Load or build source-aware area distances keyed by graph node index.

    BlockGroup and Tract payloads are complete square matrices. Block payloads
    contain only raw school-location Block rows against every destination Block.
    """
    area_ids = [int(area_id) for area_id in area2idx]
    namespace = _distance_cache_namespace(cfg, area_ids)
    matrix = namespace.load_dataframe("distances.csv", index_col=cfg.unit)
    if matrix is None:
        matrix = _build_distance_matrix(cfg, area_ids)
        namespace.save_dataframe("distances.csv", matrix, index=True)

    try:
        matrix.index = [_area_id(value) for value in matrix.index]
        matrix.columns = [_area_id(value) for value in matrix.columns]
    except ValueError as exc:
        raise ValueError(
            f"Area-distance cache {namespace.path} has invalid area IDs."
        ) from exc
    missing_columns = set(area_ids) - set(matrix.columns)
    if missing_columns:
        raise ValueError(
            f"Area-distance cache {namespace.path} is missing {cfg.unit} IDs: "
            f"{sorted(missing_columns)}."
        )

    source_ids = [area_id for area_id in matrix.index if area_id in area2idx]
    if not source_ids:
        raise ValueError(
            f"Area-distance cache {namespace.path} has no {cfg.unit} rows used "
            "by the graph."
        )
    selected = matrix.loc[source_ids, area_ids]
    if selected.isna().any().any():
        raise ValueError(f"Area-distance cache {namespace.path} contains null values.")

    distances = {index: {} for index in area2idx.values()}
    for area_i, row in zip(source_ids, selected.to_numpy()):
        index_i = area2idx[area_i]
        for area_j, distance in zip(area_ids, row):
            index_j = area2idx[area_j]
            value = float(distance)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Area-distance cache {namespace.path} contains invalid "
                    f"distance {value!r}."
                )
            distances[index_i][index_j] = value
            distances[index_j][index_i] = value
    return distances


def _build_distance_matrix(cfg: IngestConfig, area_ids: Sequence[int]) -> pd.DataFrame:
    locations = load_area_latlon(cfg)
    missing_destinations = set(area_ids) - set(locations.index)
    if missing_destinations:
        raise ValueError(
            f"Missing {cfg.unit} centroid locations for {sorted(missing_destinations)}."
        )

    if cfg.unit == "Block":
        source_ids = sorted(set(load_school_locations(cfg)[cfg.unit].astype(int)))
        if not source_ids:
            raise ValueError("No raw school-location Blocks are configured.")
    else:
        source_ids = list(area_ids)
    missing_sources = set(source_ids) - set(locations.index)
    if missing_sources:
        raise ValueError(
            f"Missing {cfg.unit} centroid locations for distance source rows: "
            f"{sorted(missing_sources)}."
        )

    source_coordinates = np.radians(
        locations.loc[source_ids, ["Lat", "Lon"]].to_numpy(dtype=float)
    )
    destination_coordinates = np.radians(
        locations.loc[list(area_ids), ["Lat", "Lon"]].to_numpy(dtype=float)
    )
    source_latitudes = source_coordinates[:, 0, np.newaxis]
    destination_latitudes = destination_coordinates[np.newaxis, :, 0]
    longitude_deltas = (
        source_coordinates[:, 1, np.newaxis] - destination_coordinates[np.newaxis, :, 1]
    )
    cosines = np.sin(source_latitudes) * np.sin(destination_latitudes) + np.cos(
        source_latitudes
    ) * np.cos(destination_latitudes) * np.cos(longitude_deltas)
    distances = EARTH_RADIUS_MILES * np.arccos(np.clip(cosines, -1.0, 1.0))
    distances[
        np.asarray(source_ids)[:, np.newaxis] == np.asarray(area_ids)[np.newaxis, :]
    ] = 0.0
    return pd.DataFrame(
        distances,
        index=pd.Index(source_ids, name=cfg.unit),
        columns=list(area_ids),
    )


def _area_id(value: Any) -> int:
    number = float(str(value))
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid area ID {value!r}.")
    return int(number)


def load_neighbors(cfg: IngestConfig, area2idx: dict[int, int]) -> dict[int, list[int]]:
    """Return symmetric adjacency from the scenario's unit-specific role."""
    key = {"Block": "block", "BlockGroup": "blockgroup", "Tract": "tract"}[cfg.unit]
    source_map = cfg.data.source_map(ADJACENCY_ROLE)
    try:
        source = source_map[key]
    except KeyError as exc:
        raise ValueError(f"{ADJACENCY_ROLE} has no {key!r} source.") from exc
    neighbors: dict[int, list[int]] = {}
    with source.path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            area_ids = [int(value) for value in row if str(value).strip()]
            if not area_ids or area_ids[0] not in area2idx:
                continue
            node = area2idx[area_ids[0]]
            adjacent = [
                area2idx[area]
                for area in area_ids
                if area in area2idx and area2idx[area] != node
            ]
            neighbors.setdefault(node, [])
            for neighbor in adjacent:
                if neighbor not in neighbors[node]:
                    neighbors[node].append(neighbor)
                neighbors.setdefault(neighbor, [])
                if node not in neighbors[neighbor]:
                    neighbors[neighbor].append(node)
    for index in area2idx.values():
        neighbors.setdefault(index, [])
    return neighbors


# ====================================================================== #
# Centroids and manual edges
# ====================================================================== #
def load_centroid_schools(
    centroids_type: str, data: DataScenario | None = None
) -> list[int]:
    """Return canonical school IDs for a scenario centroid configuration."""
    scenario = data or _legacy_scenario()
    path = scenario.source(CENTROID_ROLE).path
    with path.open("r", encoding="utf-8") as stream:
        configurations = yaml.safe_load(stream)
    if not isinstance(configurations, dict) or centroids_type not in configurations:
        raise ValueError(
            f"centroids_type {centroids_type!r} not found in {Path(path).name}."
        )
    aliases = school_id_aliases(scenario, "optimization")
    return [
        aliases.get(int(school_id), int(school_id))
        for school_id in configurations[centroids_type]
    ]


def load_manual_block_edges(cfg: IngestConfig) -> list[tuple[int, int]]:
    """Load and merge the scenario's reviewed and explicit Block edges."""
    edges = {
        edge
        for source in cfg.data.sources(MANUAL_EDGE_ROLE)
        for edge in load_block_edge_overrides(source.path)
    }
    return sorted(edges)


__all__ = [
    "ADJACENCY_ROLE",
    "AREA_DISTANCE_CACHE_SCHEMA_VERSION",
    "CAPACITY_ROLE",
    "capacity_source_roles",
    "census_geometry_roles",
    "CENSUS_ROLE",
    "CENTROID_ROLE",
    "CROSSWALK_ROLE",
    "FRL_ESTIMATE_ROLE",
    "IngestConfig",
    "MANUAL_EDGE_ROLE",
    "OUTPUT_LATLON_CRS",
    "PROJECTED_CENTROID_CRS",
    "PROGRAM_ROLE",
    "SCHOOL_ROLE",
    "STUDENT_CACHE_SCHEMA_VERSION",
    "STUDENT_ROLE",
    "load_area_latlon",
    "load_area_table",
    "load_census_shapefile",
    "load_centroid_schools",
    "load_distance_dict",
    "load_manual_block_edges",
    "load_neighbors",
    "load_school_coordinates",
    "load_school_locations",
    "load_schools",
    "load_students",
    "student_source_roles",
]
