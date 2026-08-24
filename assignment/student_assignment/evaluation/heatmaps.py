import csv
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from loaders import (
    CacheStore,
    DataScenario,
    load_census_geometry,
    load_school_records,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patheffects import Normal, withStroke


HEATMAP_CONTEXT_COLUMNS = ["config_name", "policy", "building_block", "zone_file"]
HEATMAP_COLUMNS = [
    *HEATMAP_CONTEXT_COLUMNS,
    "area_id",
    "capacity",
    "assigned",
    "unassigned",
]
ATTENDANCE_AREA_CACHE_SCHEMA_VERSION = 1
ATTENDANCE_AREA_PAYLOAD = "geometry.pkl"
ATTENDANCE_AREA_SOURCE_ROLES = (
    "assignment.attendance_areas",
    "assignment.schools",
)
SEAT_BALANCE_COLORMAP = LinearSegmentedColormap.from_list(
    "assignment_seat_balance",
    ["#ed6a5a", "#f2efe4", "#69bd63"],
)


def average_heatmap_data(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=HEATMAP_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    contexts = combined[HEATMAP_CONTEXT_COLUMNS].drop_duplicates()
    repeated = contexts["config_name"].duplicated(keep=False)
    if repeated.any():
        names = sorted(contexts.loc[repeated, "config_name"].astype(str).unique())
        raise ValueError(f"Heatmap configs have inconsistent zone contexts: {names}.")
    return (
        combined.groupby(
            [*HEATMAP_CONTEXT_COLUMNS, "area_id"], as_index=False, sort=True
        )[["capacity", "assigned", "unassigned"]]
        .mean()
        .reset_index(drop=True)
    )


def _short_school_name(name: str) -> str:
    return re.sub(r"\s+ES(?:\s+\([^)]*\))?$", "", str(name), flags=re.I)


def build_attendance_area_geometry(
    areas: gpd.GeoDataFrame,
    schools: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Map attendance schools to polygons once for reuse across policies."""
    required_school_columns = {"school_id", "school_name", "category", "lat", "lon"}
    missing = sorted(required_school_columns - set(schools.columns))
    if missing:
        raise ValueError(f"Attendance-area heatmaps require school columns: {missing}.")
    if areas.crs is None:
        raise ValueError("Attendance-area geometry must define a CRS.")
    if areas.empty or areas.geometry.isna().any() or areas.geometry.is_empty.any():
        raise ValueError("Attendance-area geometry is empty or invalid.")

    attendance_schools = schools.loc[
        schools["category"].astype("string").str.casefold().eq("attendance")
    ].copy()
    points = gpd.GeoDataFrame(
        attendance_schools,
        geometry=gpd.points_from_xy(
            attendance_schools["lon"], attendance_schools["lat"]
        ),
        crs="EPSG:4326",
    ).to_crs(areas.crs)

    indexed_areas = areas[["geometry"]].reset_index(drop=True)
    indexed_areas.index.name = "attendance_area_index"
    joined = gpd.sjoin(points, indexed_areas, how="left", predicate="within")
    if joined["attendance_area_index"].isna().any():
        missing_names = joined.loc[
            joined["attendance_area_index"].isna(), "school_name"
        ].tolist()
        raise ValueError(
            "Attendance schools fall outside the attendance-area geometry: "
            f"{missing_names}."
        )

    joined["school_id"] = pd.to_numeric(joined["school_id"], errors="raise").astype(
        "int64"
    )
    polygon_data = (
        joined.assign(_name=joined["school_name"].map(_short_school_name))
        .groupby("attendance_area_index", sort=True)
        .agg(
            attendance_area_name=(
                "_name",
                lambda values: " / ".join(dict.fromkeys(values)),
            ),
            school_ids=(
                "school_id",
                lambda values: tuple(dict.fromkeys(int(value) for value in values)),
            ),
        )
    )
    if len(polygon_data) != len(indexed_areas):
        raise ValueError(
            "Every attendance-area polygon must contain an attendance school."
        )
    return gpd.GeoDataFrame(
        indexed_areas.join(polygon_data), geometry="geometry", crs=areas.crs
    )


def _valid_cached_geometry(value: Any) -> bool:
    required = {"attendance_area_name", "school_ids", "geometry"}
    return (
        isinstance(value, gpd.GeoDataFrame)
        and required <= set(value.columns)
        and value.crs is not None
        and not value.empty
        and value.index.name == "attendance_area_index"
        and not value.index.duplicated().any()
        and not value["attendance_area_name"].isna().any()
        and value["school_ids"]
        .map(lambda ids: isinstance(ids, tuple) and bool(ids))
        .all()
        and not value.geometry.isna().any()
        and not value.geometry.is_empty.any()
    )


class AttendanceAreaArtifactStore:
    """Build and reuse source-aware attendance-area geometry."""

    def __init__(
        self,
        scenario: DataScenario,
        area_loader: Callable[[], gpd.GeoDataFrame] | None = None,
        school_loader: Callable[[], pd.DataFrame] | None = None,
    ):
        if not isinstance(scenario, DataScenario):
            raise TypeError("AttendanceAreaArtifactStore requires a DataScenario.")
        self.scenario = scenario
        self.area_loader = area_loader or (
            lambda: gpd.read_file(
                self.scenario.source("assignment.attendance_areas").path
            )
        )
        self.school_loader = school_loader or (
            lambda: load_school_records(
                self.scenario,
                "assignment.schools",
                filter_group="assignment",
            )
        )

    def geometry(self) -> tuple[gpd.GeoDataFrame, Path]:
        namespace = CacheStore(self.scenario).namespace(
            "attendance_area_heatmap_geometry",
            {"operation": "map_attendance_schools_to_polygons"},
            schema_version=ATTENDANCE_AREA_CACHE_SCHEMA_VERSION,
            roles=ATTENDANCE_AREA_SOURCE_ROLES,
            classification="internal-derived",
        )
        path = namespace.payload_path(ATTENDANCE_AREA_PAYLOAD)
        geometry = namespace.load_pickle(ATTENDANCE_AREA_PAYLOAD)
        if not _valid_cached_geometry(geometry):
            geometry = build_attendance_area_geometry(
                self.area_loader(), self.school_loader()
            )
            path = namespace.save_pickle(ATTENDANCE_AREA_PAYLOAD, geometry)
        return geometry.copy(), path


def _seat_label(value: float) -> str:
    rounded = round(value)
    if np.isclose(value, rounded, atol=0.05):
        return str(rounded)
    return f"{value:.1f}"


def _zone_plan(zone_file: str | Path) -> pd.DataFrame:
    rows = []
    with Path(zone_file).expanduser().open(newline="", encoding="utf-8-sig") as stream:
        for zone_id, row in enumerate(csv.reader(stream)):
            for raw_area in row:
                token = str(raw_area).strip()
                if token:
                    rows.append((int(token), zone_id))
    result = pd.DataFrame(rows, columns=["area_id", "zone_id"])
    if result.empty:
        raise ValueError(f"Zone file contains no areas: {zone_file}.")
    duplicates = result.loc[result["area_id"].duplicated(False), "area_id"].unique()
    if len(duplicates):
        raise ValueError(
            f"Zone file assigns areas to multiple zones: {duplicates[:10].tolist()}."
        )
    return result


def _dissolve_zones(
    geometry: gpd.GeoDataFrame,
    *,
    zone_column: str = "zone_id",
) -> gpd.GeoDataFrame:
    result = geometry.dissolve(
        by=zone_column,
        as_index=False,
        aggfunc={
            "area_ids": lambda values: tuple(
                dict.fromkeys(area for ids in values for area in ids)
            )
        },
    )
    result["geographic_area_name"] = result[zone_column].map(
        lambda value: f"Zone {int(value) + 1}"
    )
    return result[["geographic_area_name", "area_ids", "geometry"]]


def build_heatmap_geometry(
    scenario: DataScenario,
    building_block: str,
    zone_file: str | Path | None,
    *,
    attendance_area_geometry: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Build individual attendance areas or dissolve them into a policy's zones."""
    if building_block == "attendance_area":
        areas = (
            attendance_area_geometry.copy()
            if attendance_area_geometry is not None
            else AttendanceAreaArtifactStore(scenario).geometry()[0]
        )
        areas = areas.reset_index(drop=True).rename(
            columns={
                "attendance_area_name": "geographic_area_name",
                "school_ids": "area_ids",
            }
        )
        if not zone_file:
            return areas[["geographic_area_name", "area_ids", "geometry"]]

        plan = _zone_plan(zone_file).rename(columns={"area_id": "school_id"})
        school_zones = (
            areas[["area_ids"]]
            .reset_index(names="area_index")
            .explode("area_ids")
            .merge(
                plan,
                left_on="area_ids",
                right_on="school_id",
                how="left",
            )
        )
        zone_counts = school_zones.groupby("area_index", sort=True)["zone_id"].nunique()
        if (zone_counts > 1).any():
            raise ValueError("An attendance-area polygon spans multiple policy zones.")
        areas["zone_id"] = school_zones.groupby("area_index", sort=True)[
            "zone_id"
        ].first()
        if areas["zone_id"].isna().any():
            missing = areas.loc[
                areas["zone_id"].isna(), "geographic_area_name"
            ].tolist()
            raise ValueError(f"Policy zones omit attendance areas: {missing}.")
        return _dissolve_zones(areas)

    census_units = {
        "block": "Block",
        "block_group": "BlockGroup",
        "tract": "Tract",
    }
    if building_block not in census_units:
        raise ValueError(f"Unsupported heatmap building block {building_block!r}.")
    if not zone_file:
        raise ValueError(f"{building_block} heatmaps require a zone file.")

    unit = census_units[building_block]
    plan = _zone_plan(zone_file)
    geometry = load_census_geometry(scenario, "assignment", unit).rename(
        columns={unit: "area_id"}
    )
    geometry = geometry.merge(plan, on="area_id", how="inner", validate="one_to_one")
    missing_zones = sorted(set(plan["zone_id"]) - set(geometry["zone_id"]))
    if missing_zones:
        raise ValueError(
            f"Policy zones without renderable geometry: {missing_zones[:10]}."
        )
    geometry["area_ids"] = geometry["area_id"].map(lambda value: (int(value),))
    return _dissolve_zones(geometry)


def attach_heatmap_metrics(
    area_geometry: gpd.GeoDataFrame,
    area_metrics: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Attach open GE seats and resident unassigned students to map polygons."""
    area_mapping = (
        area_geometry[["area_ids"]]
        .explode("area_ids")
        .reset_index(names="geographic_area_index")
        .rename(columns={"area_ids": "area_id"})
    )
    metrics = area_metrics[["area_id", "capacity", "assigned", "unassigned"]].copy()
    for frame in (area_mapping, metrics):
        frame["area_id"] = pd.to_numeric(frame["area_id"], errors="raise").astype(
            "int64"
        )
    grouped = (
        metrics.merge(area_mapping, on="area_id", how="inner")
        .groupby("geographic_area_index", sort=True)[
            ["capacity", "assigned", "unassigned"]
        ]
        .sum(min_count=1)
    )
    result = area_geometry.drop(columns="area_ids").copy().join(grouped)
    result[["capacity", "assigned", "unassigned"]] = result[
        ["capacity", "assigned", "unassigned"]
    ].fillna(0)
    result["unfilled_spots"] = (result["capacity"] - result["assigned"]).clip(lower=0)
    result["seat_balance"] = result["unfilled_spots"].where(
        result["unassigned"].eq(0), -result["unassigned"]
    )
    return gpd.GeoDataFrame(result, geometry="geometry", crs=area_geometry.crs)


def _output_path(root: Path, config_name: str) -> Path:
    parts = [
        re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("-.") or "policy"
        for part in str(config_name).split("/")
    ]
    path = root.joinpath("heatmaps", *parts)
    return path.parent / f"{path.name}.png"


def render_assignment_heatmap(
    geographic_data: gpd.GeoDataFrame,
    config_name: str,
    output_path: str | Path,
) -> Path:
    values = geographic_data["seat_balance"].to_numpy(dtype=float)
    limit = max(float(np.max(np.abs(values))) if len(values) else 0, 1)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    figure = Figure(figsize=(12, 12), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    geographic_data.plot(
        ax=axis,
        column="seat_balance",
        cmap=SEAT_BALANCE_COLORMAP,
        norm=norm,
        edgecolor="#505050",
        linewidth=0.55,
        missing_kwds={"color": "#d9d9d9"},
    )
    axis.set_axis_off()
    axis.set_title(str(config_name), fontsize=14, fontweight="bold", pad=10)

    for row in geographic_data.itertuples():
        point = row.geometry.representative_point()
        count = abs(row.seat_balance)
        label = f"{row.geographic_area_name}\n{_seat_label(count)}"
        axis.text(
            point.x,
            point.y,
            label,
            ha="center",
            va="center",
            fontsize=6.2,
            color="#202020",
            linespacing=1.05,
            path_effects=[withStroke(linewidth=1.6, foreground="white"), Normal()],
        )

    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=SEAT_BALANCE_COLORMAP),
        ax=axis,
        location="bottom",
        shrink=0.55,
        pad=0.015,
        aspect=35,
    )
    colorbar.set_label(
        "Unassigned resident students (red) | Unfilled GE spots (green)", fontsize=9
    )
    axis.text(
        0.5,
        -0.015,
        "Labels show the count represented by each area's color",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#404040",
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.png")
    try:
        figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def export_assignment_heatmaps(
    assignment_path: str | Path,
    scenario: DataScenario,
    heatmap_data: pd.DataFrame,
) -> list[Path]:
    if heatmap_data.empty:
        return []
    geometry_cache = {}
    attendance_area_geometry = None
    outputs = []
    for config_name, metrics in heatmap_data.groupby("config_name", sort=True):
        context = metrics[HEATMAP_CONTEXT_COLUMNS[1:]].drop_duplicates()
        if len(context) != 1:
            raise ValueError(
                f"Heatmap config {config_name!r} has multiple zone contexts."
            )
        building_block = str(context.iloc[0]["building_block"])
        zone_file = str(context.iloc[0]["zone_file"])
        geometry_key = (building_block, zone_file)
        if geometry_key not in geometry_cache:
            if building_block == "attendance_area" and attendance_area_geometry is None:
                attendance_area_geometry, _ = AttendanceAreaArtifactStore(
                    scenario
                ).geometry()
            geometry_cache[geometry_key] = build_heatmap_geometry(
                scenario,
                building_block,
                zone_file or None,
                attendance_area_geometry=attendance_area_geometry,
            )
        geographic_data = attach_heatmap_metrics(geometry_cache[geometry_key], metrics)
        outputs.append(
            render_assignment_heatmap(
                geographic_data,
                str(config_name),
                _output_path(Path(assignment_path).expanduser(), str(config_name)),
            )
        )
    return outputs
