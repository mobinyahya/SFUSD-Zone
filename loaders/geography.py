"""Vintage-aware Census geometry loading and point assignment."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from loaders.config import DataScenario
from loaders.tables import read_csv

GEOGRAPHY_UNITS = ("Block", "BlockGroup", "Tract")
GEOGRAPHY_COLUMNS = {
    "student": {
        "Block": "census_block",
        "BlockGroup": "census_blockgroup",
        "Tract": "census_tract",
    },
    "school": {
        "Block": "Block",
        "BlockGroup": "BlockGroup",
        "Tract": "Tract",
    },
}
GEOMETRY_ROLE_KEYS = {
    "Block": "blocks",
    "BlockGroup": "blockgroups",
    "Tract": "tracts",
}
COORDINATE_COLUMNS = {
    "student": ("latitude", "longitude"),
    "school": ("lat", "lon"),
}
WGS84_CRS = "EPSG:4326"


def selected_geography_vintage(scenario: DataScenario, group: str) -> str:
    """Return the configured Census vintage for one consumer group."""
    if group not in {"optimization", "assignment"}:
        raise ValueError(f"Unknown filter group {group!r}.")
    return str(scenario.filter(group, "geography_vintage"))


def _geography_role(group: str, suffix: str) -> str:
    if group == "optimization" and suffix == "blocks":
        return "optimization.census"
    if group == "optimization" and suffix == "crosswalk":
        return "optimization.crosswalk"
    return f"{group}.geography.{suffix}"


def _normalize_geoid(values: pd.Series, *, length: int, label: str) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    present = text.notna() & text.ne("")
    invalid = present & (~text.str.fullmatch(r"\d+").fillna(False))
    if invalid.any():
        examples = text.loc[invalid].head(5).tolist()
        raise ValueError(f"{label} contains non-numeric GEOIDs: {examples}.")
    text.loc[present] = text.loc[present].str.zfill(length)
    wrong_length = present & text.str.len().ne(length)
    if wrong_length.any():
        examples = text.loc[wrong_length].head(5).tolist()
        raise ValueError(
            f"{label} contains GEOIDs that are not {length} digits: {examples}."
        )
    return text


def _geoid_column(frame: pd.DataFrame, vintage: str) -> str:
    candidates = (f"GEOID{vintage[-2:]}", f"geoid{vintage[-2:]}", "GEOID", "geoid")
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        f"Census {vintage} geometry has no GEOID column; tried {list(candidates)}."
    )


def load_geography_crosswalk(
    scenario: DataScenario,
    group: str,
) -> pd.DataFrame:
    """Load and validate the selected Block-to-parent geography crosswalk."""
    role = _geography_role(group, "crosswalk")
    crosswalk = read_csv(scenario, role, low_memory=False)
    required = set(GEOGRAPHY_UNITS)
    missing = required - set(crosswalk.columns)
    if missing:
        raise ValueError(
            f"Census geography crosswalk is missing columns: {sorted(missing)}."
        )

    result = crosswalk[list(GEOGRAPHY_UNITS)].copy()
    lengths = {"Block": 15, "BlockGroup": 12, "Tract": 11}
    for unit, length in lengths.items():
        normalized = _normalize_geoid(
            result[unit], length=length, label=f"Census crosswalk {unit}"
        )
        if normalized.isna().any():
            raise ValueError(f"Census crosswalk contains blank {unit} GEOIDs.")
        result[unit] = normalized.astype("int64")

    result = result.drop_duplicates().copy()
    duplicates = result.loc[result["Block"].duplicated(False), "Block"].unique()
    if len(duplicates):
        raise ValueError(
            "Census crosswalk contains duplicate Block GEOIDs, including "
            f"{duplicates[:5].tolist()}."
        )
    if selected_geography_vintage(scenario, group) == "2020":
        expected_bg = (
            result["Block"].astype(str).str.zfill(15).str[:12].astype("int64")
        )
        expected_tract = (
            result["Block"].astype(str).str.zfill(15).str[:11].astype("int64")
        )
        inconsistent = (result["BlockGroup"] != expected_bg) | (
            result["Tract"] != expected_tract
        )
        if inconsistent.any():
            examples = result.loc[inconsistent].head(5).to_dict("records")
            raise ValueError(
                "Census crosswalk contains inconsistent Block parent GEOIDs: "
                f"{examples}."
            )
    return result.sort_values("Block").reset_index(drop=True)


def _validate_geometry(
    geometry: gpd.GeoDataFrame,
    *,
    source_label: str,
) -> None:
    if geometry.crs is None:
        raise ValueError(f"{source_label} has no coordinate reference system.")
    if geometry.empty:
        raise ValueError(f"{source_label} contains no rows.")
    if geometry.geometry.isna().any() or geometry.geometry.is_empty.any():
        raise ValueError(f"{source_label} contains empty geometries.")
    invalid = ~geometry.geometry.is_valid
    if invalid.any():
        geometry.loc[invalid, "geometry"] = geometry.loc[invalid, "geometry"].make_valid()
    if (~geometry.geometry.is_valid).any():
        raise ValueError(f"{source_label} contains invalid geometries.")


def load_census_geometry(
    scenario: DataScenario,
    group: str,
    unit: str,
) -> gpd.GeoDataFrame:
    """Load one official Census layer, deriving legacy parents when necessary."""
    if unit not in GEOGRAPHY_UNITS:
        raise ValueError(f"Unsupported census unit {unit!r}.")
    vintage = selected_geography_vintage(scenario, group)
    role = _geography_role(group, GEOMETRY_ROLE_KEYS[unit])
    try:
        source = scenario.source(role)
    except KeyError:
        if unit == "Block":
            raise
        blocks = load_census_geometry(scenario, group, "Block")
        crosswalk = load_geography_crosswalk(scenario, group)
        derived = blocks.merge(crosswalk[["Block", unit]], on="Block", validate="1:1")
        return derived.dissolve(by=unit, as_index=False)[[unit, "geometry"]]

    geometry = gpd.read_file(source.path)
    source_label = f"Census {vintage} {unit} geometry {source.path}"
    _validate_geometry(geometry, source_label=source_label)
    geoid_column = _geoid_column(geometry, vintage)
    length = {"Block": 15, "BlockGroup": 12, "Tract": 11}[unit]
    geoids = _normalize_geoid(
        geometry[geoid_column], length=length, label=f"Census {vintage} {unit}"
    )
    if geoids.isna().any():
        raise ValueError(f"{source_label} contains blank GEOIDs.")
    result = geometry[["geometry"]].copy()
    result[unit] = geoids.astype("int64")
    if result[unit].duplicated().any():
        duplicates = result.loc[result[unit].duplicated(False), unit].head(5).tolist()
        raise ValueError(f"{source_label} contains duplicate GEOIDs: {duplicates}.")

    if unit == "Block":
        crosswalk = load_geography_crosswalk(scenario, group)
        missing = set(crosswalk["Block"]) - set(result["Block"])
        if missing:
            raise ValueError(
                f"Census {vintage} Block geometry/crosswalk mismatch: "
                f"missing geometry={sorted(missing)[:5]}."
            )
        result = result.loc[result["Block"].isin(crosswalk["Block"])].copy()
    return result[[unit, "geometry"]].sort_values(unit).reset_index(drop=True)


def match_points_to_census(
    frame: pd.DataFrame,
    scenario: DataScenario,
    group: str,
    *,
    latitude_column: str,
    longitude_column: str,
    row_mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Map WGS84 points inside the selected Blocks to their Census GEOIDs."""
    required = {latitude_column, longitude_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Point data is missing coordinate columns: {sorted(missing)}.")
    latitude = pd.to_numeric(frame[latitude_column], errors="coerce")
    longitude = pd.to_numeric(frame[longitude_column], errors="coerce")
    requested = (
        pd.Series(True, index=frame.index)
        if row_mask is None
        else pd.Series(np.asarray(row_mask, dtype=bool), index=frame.index)
    )
    valid = (
        requested
        & latitude.between(-90, 90, inclusive="both")
        & longitude.between(-180, 180, inclusive="both")
    )
    result = pd.DataFrame(
        {unit: pd.Series(pd.NA, index=frame.index, dtype="Int64") for unit in GEOGRAPHY_UNITS}
    )
    if not valid.any():
        return result

    blocks = load_census_geometry(scenario, group, "Block")
    crosswalk = load_geography_crosswalk(scenario, group)
    block_parents = crosswalk.set_index("Block")[["BlockGroup", "Tract"]]
    positions = np.flatnonzero(valid.to_numpy())
    points = gpd.GeoDataFrame(
        {"row_position": positions},
        geometry=gpd.points_from_xy(
            longitude.loc[valid].to_numpy(), latitude.loc[valid].to_numpy()
        ),
        crs=WGS84_CRS,
    ).to_crs(blocks.crs)
    matches = gpd.sjoin(
        points,
        blocks[["Block", "geometry"]],
        how="left",
        predicate="intersects",
    ).dropna(subset=["Block"])
    matches["Block"] = matches["Block"].astype("int64")
    matches = matches.sort_values(["row_position", "Block"]).drop_duplicates(
        "row_position", keep="first"
    )

    matches = matches[["row_position", "Block"]]
    matches = matches.merge(block_parents, left_on="Block", right_index=True, validate="m:1")
    matched_rows = matches["row_position"].astype(int).to_numpy()
    for unit in GEOGRAPHY_UNITS:
        result.iloc[matched_rows, result.columns.get_loc(unit)] = (
            matches[unit].astype("int64").to_numpy()
        )
    return result


def _column_style(frame: pd.DataFrame) -> str | None:
    if set(COORDINATE_COLUMNS["student"]) <= set(frame.columns) or any(
        column in frame.columns for column in GEOGRAPHY_COLUMNS["student"].values()
    ):
        return "student"
    if set(COORDINATE_COLUMNS["school"]) <= set(frame.columns) or any(
        column in frame.columns for column in GEOGRAPHY_COLUMNS["school"].values()
    ):
        return "school"
    return None


def normalize_census_geography(
    frame: pd.DataFrame,
    scenario: DataScenario,
    group: str,
    *,
    source_vintage: str | None,
    style: str | None = None,
) -> pd.DataFrame:
    """Return a table with selected-vintage Block, BlockGroup, and Tract IDs.

    Catalog sources declare their Census vintage. Ad-hoc sources without that
    metadata retain their existing behavior; callers can opt in by setting the
    direct source's ``geography_vintage`` field.
    """
    style = style or _column_style(frame)
    if style is None:
        return frame.copy()
    if style not in GEOGRAPHY_COLUMNS:
        raise ValueError(f"Unknown geography column style {style!r}.")
    if source_vintage is None:
        return frame.copy()

    target_vintage = selected_geography_vintage(scenario, group)
    columns = GEOGRAPHY_COLUMNS[style]
    latitude_column, longitude_column = COORDINATE_COLUMNS[style]
    result = frame.copy()
    attrs = dict(frame.attrs)
    for column in columns.values():
        if column not in result.columns:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    same_vintage = str(source_vintage) == target_vintage
    if same_vintage:
        for column in columns.values():
            values = pd.to_numeric(result[column], errors="coerce").astype("Int64")
            result[column] = values
        result.attrs.update(attrs)
        result.attrs["geography_vintage"] = target_vintage
        return result
    else:
        for column in columns.values():
            result[column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        needs_coordinates = pd.Series(True, index=result.index)

    if needs_coordinates.any():
        if {latitude_column, longitude_column} <= set(result.columns):
            matched = match_points_to_census(
                result,
                scenario,
                group,
                latitude_column=latitude_column,
                longitude_column=longitude_column,
                row_mask=needs_coordinates,
            )
            for unit, column in columns.items():
                result[column] = result[column].fillna(matched[unit]).astype("Int64")
        elif not same_vintage:
            raise ValueError(
                f"Cannot convert {style} data from Census {source_vintage} to "
                f"{target_vintage}: missing coordinate columns "
                f"{[latitude_column, longitude_column]}."
            )

    result.attrs.update(attrs)
    result.attrs["geography_vintage"] = target_vintage
    return result


def source_geography_vintage(source: Any) -> str | None:
    """Return optional source-vintage metadata without coupling callers to type."""
    value = getattr(source, "geography_vintage", None)
    return str(value) if value is not None else None


__all__ = [
    "GEOGRAPHY_COLUMNS",
    "GEOGRAPHY_UNITS",
    "load_census_geometry",
    "load_geography_crosswalk",
    "match_points_to_census",
    "normalize_census_geography",
    "selected_geography_vintage",
    "source_geography_vintage",
]
