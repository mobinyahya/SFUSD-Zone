"""Raw data ingestion.

A clean reimplementation of the legacy ``Students`` / ``Schools`` /
``DesignZones`` ingestion. The job is to turn the raw source files into the
tidy, per-area tables the graph builder needs:

* :func:`load_area_table`  -- one row per geographic area (students, capacity,
  FRL, ethnicity counts, school ids/locations),
* :func:`load_neighbors`   -- adjacency from the precomputed matrices,
* :func:`load_distance_dict`-- area-to-area distances,
* :func:`load_area_latlon` -- area centroids from the census shapefile,
* :func:`load_centroid_schools` -- centroid school ids for a ``centroids_type``.

All paths are derived from :class:`IngestConfig`; nothing is hardcoded to a
single unit, so the same code builds Block and BlockGroup tables.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field

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
    get_dropbox_path,
    get_sfusd_path,
)
from optimization.data.geography import great_circle_miles

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Config")
CENTROIDS_YAML = os.path.abspath(os.path.join(CONFIG_DIR, "centroids.yaml"))
SFUSD_PATH = get_sfusd_path(False)
DROPBOX_PATH = get_dropbox_path(False)
PROJECTED_CENTROID_CRS = "EPSG:32610"  # San Francisco is in UTM zone 10N.
OUTPUT_LATLON_CRS = "EPSG:4326"
STUDENT_CACHE_SCHEMA_VERSION = 2

# Maps the post-dummies ethnicity columns to the canonical Ethnicity_* names.
_ETHNICITY_RENAME = {
    "resolved_ethnicity_American Indian": "Ethnicity_American_Indian",
    "resolved_ethnicity_Asian": "Ethnicity_Asian",
    "resolved_ethnicity_Black or African American": "Ethnicity_Black_or_African_American",
    "resolved_ethnicity_Filipino": "Ethnicity_Filipino",
    "resolved_ethnicity_Pacific Islander": "Ethnicity_PacificIslander",
    "resolved_ethnicity_Hispanic/Latinx": "Ethnicity_Hispanic/Latinx",
    "resolved_ethnicity_Two or More Races": "Ethnicity_Two_or_More_Races",
    "resolved_ethnicity_White": "Ethnicity_White",
}


@dataclass
class IngestConfig:
    """Everything ingestion needs to locate and shape the raw data."""

    unit: str  # 'Block' or 'BlockGroup'
    years: list[int] = field(default_factory=lambda: [14, 15, 16, 17, 18, 21, 22])
    population_type: str = "GE"
    drop_optout: bool = True
    capacity_scenario: str = "A"
    new_schools: bool = True
    include_k8: bool = False

    def __post_init__(self):
        if self.unit not in ("Block", "BlockGroup"):
            raise ValueError(f"Ingestion supports Block/BlockGroup, got {self.unit!r}.")


# ====================================================================== #
# Students
# ====================================================================== #
def load_students(cfg: IngestConfig) -> pd.DataFrame:
    """Concatenated, filtered, per-student rows across ``cfg.years``.

    A cleaned cache is reused when present. It is keyed by the student filters
    that change row membership so generated graph populations match the run.
    """
    cache = _student_cache_path(cfg)
    if os.path.exists(cache):
        return pd.read_csv(cache, low_memory=False)

    frames = [_load_students_for_year(cfg, year) for year in cfg.years]
    students = pd.concat(frames, ignore_index=True)
    tmp_cache = f"{cache}.{os.getpid()}.tmp"
    students.to_csv(tmp_cache, index=False)
    os.replace(tmp_cache, cache)
    return students


def _student_cache_path(cfg: IngestConfig) -> str:
    years = "_".join(str(y) for y in cfg.years)
    population = _safe_cache_value(cfg.population_type)
    return os.path.join(
        f"{SFUSD_PATH}/Data/Cleaned",
        f"Cleaned_Students_v{STUDENT_CACHE_SCHEMA_VERSION}_{years}_"
        f"pop{population}_drop{int(cfg.drop_optout)}.csv",
    )


def _safe_cache_value(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _load_students_for_year(cfg: IngestConfig, year: int) -> pd.DataFrame:
    if cfg.drop_optout:
        path = f"{SFUSD_PATH}/Data/Cleaned/enrolled_{year}{year + 1}.csv"
    else:
        path = f"{SFUSD_PATH}/Data/Cleaned/student_{year}{year + 1}.csv"
    df = pd.read_csv(path, low_memory=False)

    df = df.loc[df["grade"] == "KG"].copy()
    df["resolved_ethnicity"] = df["resolved_ethnicity"].replace(ETHNICITY_DICT)
    df.rename(
        columns={
            "census_block": "Block",
            "census_blockgroup": "BlockGroup",
            "idschoolattendance": "attendance_area",
            "FRL Score": "FRL",
        },
        inplace=True,
    )
    df.dropna(subset=BUILDING_BLOCKS, inplace=True)
    df["enrolled_students"] = np.where(df["enrolled_idschool"].isna(), 0, 1)

    # Coalesce program lists into r1_programs, then keep only students who
    # ranked at least one program.
    df["r1_programs"] = df["r1_programs"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    for rnd in range(2, 6):
        col = f"r{rnd}_programs"
        if col in df.columns:
            empty = df["r1_programs"].apply(lambda x: len(x) <= 0)
            is_list = df[col].apply(lambda x: isinstance(x, list))
            df.loc[empty & is_list, "r1_programs"] = df.loc[empty & is_list, col]
    df = df.loc[df["r1_programs"].apply(lambda x: x != [])]

    # Each student is one all-program student and a (fractional) GE student.
    df["all_prog_students"] = 1
    df["ge_students"] = df.apply(
        lambda x: (
            sum(p == "GE" for p in x["r1_programs"]) / len(x["r1_programs"])
            if x["enrolled_students"] == 1 and len(x["r1_programs"]) > 0
            else 0
        ),
        axis=1,
    )

    df = pd.get_dummies(df, columns=["resolved_ethnicity"])

    # Count demographics in the same population units used by the optimization.
    population_weight = (
        df["ge_students"] if cfg.population_type == "GE" else df["all_prog_students"]
    )
    for col in ETHNICITY_COLS + ["FRL"]:
        if col in df.columns:
            df[col] = df[col] * population_weight

    df = _tag_program_types(df)
    df = _filter_to_population(df, cfg.population_type, year)

    keep = [c for c in IMPORTANT_COLS + ETHNICITY_COLS if c in df.columns]
    df = df[keep]

    for col in ["FRL", "AALPI Score"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    df.fillna(value=0, inplace=True)

    df.rename(columns=_ETHNICITY_RENAME, inplace=True)
    return df


def _tag_program_types(df: pd.DataFrame) -> pd.DataFrame:
    for rnd in range(1, 4):
        col = f"r{rnd}_programs"
        if col not in df.columns:
            continue
        if rnd == 1:
            df["program_types"] = df[col]
        else:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
            df["program_types"] = df["program_types"] + df[col]
    df["program_types"] = df["program_types"].apply(lambda x: np.unique(x))
    return df


def _filter_to_population(
    df: pd.DataFrame, population_type: str, year: int
) -> pd.DataFrame:
    if population_type == "All":
        return df
    if population_type == "GE":
        if year != 18:
            df["filter"] = df["program_types"].apply(lambda pt: 1 if "GE" in pt else 0)
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

            df["filter"] = df.apply(keep, axis=1)
    else:  # language program populations (SB, CB, ...)
        df["filter"] = df["program_types"].apply(
            lambda pt: 1 if population_type in pt else 0
        )
    return df.loc[df["filter"] == 1]


# ====================================================================== #
# Schools
# ====================================================================== #
def load_schools(cfg: IngestConfig) -> pd.DataFrame:
    """Per-school rows with capacities, quality scores and the unit column."""
    if cfg.new_schools:
        df = pd.read_csv(
            f"{SFUSD_PATH}/Data/Cleaned/schools_table_for_zone_development_updated.csv"
        )
    else:
        df = pd.read_csv(f"{SFUSD_PATH}/Data/Cleaned/schools_rehauled_1819.csv")

    df[cfg.unit] = df[cfg.unit].astype("Int64")
    df["K-8"] = df["school_id"].apply(lambda x: 1 if x in K8_SCHOOLS else 0)
    df = _attach_capacity(df, cfg)
    df["num_schools"] = 1
    df.rename(
        columns={
            "eng_scores_1819": "english_score",
            "math_scores_1819": "math_score",
        },
        inplace=True,
    )
    return df


def load_school_locations(cfg: IngestConfig) -> pd.DataFrame:
    """Raw school ids and locations before capacity/K-8/citywide filtering.

    This is used only to locate centroid anchors geographically. Graph school
    and capacity metrics continue to come from :func:`load_schools`.
    """
    if cfg.new_schools:
        df = pd.read_csv(
            f"{SFUSD_PATH}/Data/Cleaned/schools_table_for_zone_development_updated.csv"
        )
    else:
        df = pd.read_csv(f"{SFUSD_PATH}/Data/Cleaned/schools_rehauled_1819.csv")

    missing = {"school_id", cfg.unit} - set(df.columns)
    if missing:
        raise ValueError(f"School location table missing columns: {sorted(missing)}.")

    df = df[["school_id", cfg.unit]].dropna().copy()
    df["school_id"] = df["school_id"].astype(int)
    df[cfg.unit] = df[cfg.unit].astype("int64")
    return df


def load_school_coordinates() -> pd.DataFrame:
    """Canonical school points used by config-independent geometry artifacts."""
    path = f"{SFUSD_PATH}/Data/Cleaned/schools_table_for_zone_development_updated.csv"
    df = pd.read_csv(path)
    required = {"school_id", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"School coordinate table missing columns: {sorted(missing)}.")
    return df[["school_id", "lat", "lon"]].copy()


def _attach_capacity(df: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    programs = pd.read_csv(
        f"{SFUSD_PATH}/Data/Cleaned/stanford_capacities_12.23.21.csv"
    )
    programs.rename(
        columns={
            "SchNum": "school_id",
            "PathwayCode": "program_type",
            f"Scenario_{cfg.capacity_scenario}_Capacity": "Capacity",
        },
        inplace=True,
    )
    ge = programs.loc[programs["program_type"] == "GE"][["school_id", "Capacity"]]
    ge = ge.rename(columns={"Capacity": "ge_capacity"})
    allp = programs[["school_id", "Capacity"]].rename(
        columns={"Capacity": "all_prog_capacity"}
    )
    allp = allp.groupby("school_id", as_index=False).sum()

    df = df.merge(allp, how="outer", on="school_id")
    df = df.merge(ge, how="outer", on="school_id")

    if cfg.capacity_scenario == "Closure" or cfg.population_type == "All":
        df = df[df["all_prog_capacity"] > 0]
    else:
        df = df.loc[df["ge_capacity"] > 0]
    if cfg.population_type != "All":
        if not cfg.include_k8:
            df = df.loc[df["K-8"] == 0]
        df = df.loc[df["category"] != "Citywide"]
    return df


# ====================================================================== #
# Per-area aggregation
# ====================================================================== #
def load_area_table(cfg: IngestConfig) -> pd.DataFrame:
    """One row per area, indexed ``0..A-1``, with the unit id in column ``unit``.

    Columns: ``ge_students, ge_capacity, all_prog_students, all_prog_capacity,
    num_schools, FRL, <AREA_ETHNICITIES>, Lat, Lon`` plus a ``school_ids`` list.
    """
    students = load_students(cfg)
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

    # location + school id list per area
    latlon = load_area_latlon(cfg)
    area = area.merge(latlon, how="left", left_on=cfg.unit, right_index=True)
    school_ids = (
        schools.dropna(subset=[cfg.unit]).groupby(cfg.unit)["school_id"].apply(list)
    )
    area["school_ids"] = (
        area[cfg.unit].map(school_ids).apply(lambda x: x if isinstance(x, list) else [])
    )
    area[["Lat", "Lon"]] = area[["Lat", "Lon"]].fillna(0.0)
    return area


def _aggregate_students(students: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    stats = students.groupby(cfg.unit, as_index=False).sum(numeric_only=True)
    cols = [c for c in AREA_COLS + [cfg.unit] + AREA_ETHNICITIES if c in stats.columns]
    stats = stats[cols]
    # Counts were summed across years; normalize back to a per-year average.
    for col in stats.columns:
        if col not in BUILDING_BLOCKS:
            stats[col] = stats[col] / len(cfg.years)
    return stats


def _aggregate_schools(schools: pd.DataFrame, unit: str) -> pd.DataFrame:
    sum_cols = [
        unit,
        "all_prog_capacity",
        "ge_capacity",
        "num_schools",
        "english_score",
        "math_score",
        "greatschools_rating",
        "AvgColorIndex",
    ]
    mean_cols = [unit, "MetStandards"]
    sum_cols = [c for c in sum_cols if c in schools.columns]
    mean_cols = [c for c in mean_cols if c in schools.columns]
    summed = schools[sum_cols].groupby(unit, as_index=False).sum(numeric_only=True)
    if len(mean_cols) > 1:
        meaned = (
            schools[mean_cols].groupby(unit, as_index=False).mean(numeric_only=True)
        )
        return meaned.merge(summed, how="left", on=unit)
    return summed


def _add_missing_areas(area: pd.DataFrame, cfg: IngestConfig) -> pd.DataFrame:
    """Append census areas that had neither students nor schools."""
    valid = set(
        pd.read_csv(f"{DROPBOX_PATH}/Optimization/block_blockgroup_tract.csv")[cfg.unit]
    )
    census = set(load_census_shapefile(cfg.unit, False)[cfg.unit]) - set(AUX_BG)
    have = set(area[cfg.unit].astype(int))
    missing = (census & valid) - have
    if missing:
        extra = pd.DataFrame({cfg.unit: list(missing)})
        area[cfg.unit] = area[cfg.unit].astype(int)
        extra[cfg.unit] = extra[cfg.unit].astype(int)
        area = pd.merge(area, extra, how="outer", on=cfg.unit)
        area.fillna(value=0, inplace=True)
    return area


def _apply_closure_scaling(area: pd.DataFrame) -> pd.DataFrame:
    """Closure-scenario rescaling (mirrors the legacy construct_datastructures)."""
    ratio = 3700 / area["all_prog_students"].sum()
    area["all_prog_students"] = area["all_prog_students"] * ratio
    area["FRL"] = (3700 / 2460) * area["FRL"]
    for eth in AREA_ETHNICITIES:
        if eth in area.columns:
            area[eth] = (3700 / 2460) * area[eth]
    return area


# ====================================================================== #
# Geometry / distance / adjacency
# ====================================================================== #
def load_census_shapefile(level: str, is_local: bool = False) -> gpd.GeoDataFrame:
    """Load census geometry enriched with Block and BlockGroup identifiers."""
    if is_local:
        path = os.path.join(
            get_sfusd_path(is_local),
            "drive-download-20200216T210200Z-001",
            "2013 ESAAs SFUSD.shp",
        )
    else:
        path = os.path.join(
            get_sfusd_path(is_local),
            "shapefiles",
            "geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp",
        )

    census = gpd.read_file(path)
    census["Block"] = census["geoid10"].fillna(0).astype("int64")

    crosswalk = pd.read_csv(
        os.path.join(
            get_dropbox_path(is_local),
            "Optimization",
            "block_blockgroup_tract.csv",
        )
    )
    crosswalk["Block"] = crosswalk["Block"].fillna(0).astype("int64")
    census = census.merge(crosswalk, how="left", on="Block")
    census.dropna(subset=["BlockGroup", "Block"], inplace=True)
    census[level] = census[level].astype("int64")
    return census


def load_area_latlon(cfg: IngestConfig) -> pd.DataFrame:
    """Centroid Lat/Lon per area, indexed by the unit id."""
    census = load_census_shapefile(cfg.unit, False)
    dissolved = census.dissolve(by=cfg.unit, as_index=False)
    centroids = _projected_centroids_latlon(dissolved)
    dissolved["Lat"] = centroids.y
    dissolved["Lon"] = centroids.x
    out = dissolved[[cfg.unit, "Lat", "Lon"]].copy()
    out[cfg.unit] = out[cfg.unit].astype("int64")
    return out.set_index(cfg.unit)


def _projected_centroids_latlon(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Return centroids computed in a projected CRS, expressed as WGS84 points."""
    if gdf.crs is None:
        gdf = gdf.set_crs(OUTPUT_LATLON_CRS)
    projected_centroids = gdf.to_crs(PROJECTED_CENTROID_CRS).centroid
    return gpd.GeoSeries(projected_centroids, crs=PROJECTED_CENTROID_CRS).to_crs(
        OUTPUT_LATLON_CRS
    )


def load_distance_dict(
    cfg: IngestConfig, area2idx: dict[int, int]
) -> dict[int, dict[int, float]]:
    """Distance lookup for cached source areas and their destinations.

    Block caches are rectangular: their rows are school/centroid blocks and
    their columns are all blocks. BlockGroup caches are square. Reuse either
    format when present, otherwise calculate and atomically cache a complete
    matrix from projected area centroids.
    """
    filename = (
        "distances_b2b_schools.csv" if cfg.unit == "Block" else "distances_bg2bg.csv"
    )
    cache_path = os.path.join(DROPBOX_PATH, "Optimization", filename)
    area_ids = list(area2idx)

    if os.path.exists(cache_path):
        matrix = pd.read_csv(cache_path, index_col=cfg.unit)
        matrix.index = [int(float(area_id)) for area_id in matrix.index]
        matrix.columns = [int(float(area_id)) for area_id in matrix.columns]
    else:
        locations = load_area_latlon(cfg)
        missing_locations = set(area_ids) - set(locations.index)
        if missing_locations:
            raise ValueError(
                f"Missing {cfg.unit} centroid locations for "
                f"{sorted(missing_locations)}."
            )

        matrix = pd.DataFrame(
            0.0,
            index=pd.Index(area_ids, name=cfg.unit),
            columns=area_ids,
        )
        for i, area_i in enumerate(area_ids):
            lat_i = float(locations.loc[area_i, "Lat"])
            lon_i = float(locations.loc[area_i, "Lon"])
            for area_j in area_ids[i + 1 :]:
                distance = great_circle_miles(
                    lat_i,
                    lon_i,
                    float(locations.loc[area_j, "Lat"]),
                    float(locations.loc[area_j, "Lon"]),
                )
                matrix.loc[area_i, area_j] = distance
                matrix.loc[area_j, area_i] = distance

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_cache = f"{cache_path}.{os.getpid()}.tmp"
        matrix.to_csv(tmp_cache)
        os.replace(tmp_cache, cache_path)

    missing_columns = set(area_ids) - set(matrix.columns)
    if missing_columns:
        raise ValueError(
            f"Distance cache {cache_path} is missing {cfg.unit} IDs: "
            f"{sorted(missing_columns)}."
        )

    source_ids = [area_id for area_id in matrix.index if area_id in area2idx]
    if not source_ids:
        raise ValueError(
            f"Distance cache {cache_path} has no {cfg.unit} rows used by the graph."
        )

    distances = {idx: {} for idx in area2idx.values()}
    values = matrix.loc[source_ids, area_ids].to_numpy()
    for area_i, row in zip(source_ids, values):
        idx_i = area2idx[area_i]
        for area_j, distance in zip(area_ids, row):
            idx_j = area2idx[area_j]
            value = float(distance)
            distances[idx_i][idx_j] = value
            distances[idx_j][idx_i] = value
    return distances


def load_neighbors(cfg: IngestConfig, area2idx: dict[int, int]) -> dict[int, list[int]]:
    """Symmetric adjacency ``{area_idx: [neighbor_idx, ...]}`` from the matrix."""
    import csv

    fname = (
        "adjacency_matrix_b.csv" if cfg.unit == "Block" else "adjacency_matrix_bg.csv"
    )
    path = os.path.expanduser(f"{DROPBOX_PATH}/Optimization/{fname}")
    with open(path, "r") as f:
        rows = list(csv.reader(f))

    neighbors: dict[int, list[int]] = {}
    for row in rows:
        if not row or int(row[0]) not in area2idx:
            continue
        u = area2idx[int(row[0])]
        adj = [area2idx[int(n)] for n in row if n != "" and int(n) in area2idx]
        if u in adj:
            adj.remove(u)
        neighbors.setdefault(u, [])
        for n in adj:
            if n not in neighbors[u]:
                neighbors[u].append(n)
            neighbors.setdefault(n, [])
            if u not in neighbors[n]:
                neighbors[n].append(u)
    # ensure every area has an entry
    for idx in area2idx.values():
        neighbors.setdefault(idx, [])
    return neighbors


# ====================================================================== #
# Centroids
# ====================================================================== #
def load_centroid_schools(centroids_type: str) -> list[int]:
    """School ids anchoring each zone for a ``centroids_type`` key."""
    with open(CENTROIDS_YAML, "r") as f:
        configs = yaml.safe_load(f)
    if centroids_type not in configs:
        raise ValueError(
            f"centroids_type {centroids_type!r} not found in centroids.yaml."
        )
    return list(configs[centroids_type])
