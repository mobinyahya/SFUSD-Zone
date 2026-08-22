import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from loaders import CacheStore, DataScenario, load_school_records
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patheffects import Normal, withStroke


HEATMAP_COLUMNS = ["config_name", "school_id", "capacity", "assigned"]
ATTENDANCE_AREA_CACHE_SCHEMA_VERSION = 1
ATTENDANCE_AREA_PAYLOAD = "geometry.pkl"
ATTENDANCE_AREA_SOURCE_ROLES = (
    "assignment.attendance_areas",
    "assignment.schools",
)
UTILIZATION_COLORMAP = LinearSegmentedColormap.from_list(
    "attendance_area_utilization",
    ["#69bd63", "#f2efe4", "#ed6a5a"],
)


def average_heatmap_data(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=HEATMAP_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby(["config_name", "school_id"], as_index=False, sort=True)[
            ["capacity", "assigned"]
        ]
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


def attach_attendance_area_utilization(
    area_geometry: gpd.GeoDataFrame,
    school_utilization: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Attach one policy's GE utilization to pre-mapped area geometry."""
    utilization = school_utilization[HEATMAP_COLUMNS[1:]].copy()
    utilization["school_id"] = pd.to_numeric(
        utilization["school_id"], errors="raise"
    ).astype("int64")
    school_to_polygon = (
        area_geometry[["school_ids"]]
        .explode("school_ids")
        .reset_index()
        .rename(columns={"school_ids": "school_id"})
    )
    school_to_polygon["school_id"] = pd.to_numeric(
        school_to_polygon["school_id"], errors="raise"
    ).astype("int64")

    by_attendance_area = (
        utilization.merge(school_to_polygon, on="school_id", how="inner")
        .groupby("attendance_area_index", as_index=True, sort=True)[
            ["capacity", "assigned"]
        ]
        .sum(min_count=1)
    )

    result = area_geometry.drop(columns="school_ids").copy().join(by_attendance_area)
    positive_capacity = result["capacity"] > 0
    result["utilization"] = (result["assigned"] / result["capacity"]).where(
        positive_capacity
    )
    result["seat_difference"] = (result["capacity"] - result["assigned"]).where(
        positive_capacity
    )
    return gpd.GeoDataFrame(result, geometry="geometry", crs=area_geometry.crs)


def build_attendance_area_data(
    areas: gpd.GeoDataFrame,
    schools: pd.DataFrame,
    school_utilization: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Build and populate attendance-area geometry without using the cache."""
    geometry = build_attendance_area_geometry(areas, schools)
    return attach_attendance_area_utilization(geometry, school_utilization)


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


def _output_path(root: Path, config_name: str) -> Path:
    parts = [
        re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("-.") or "policy"
        for part in str(config_name).split("/")
    ]
    path = root.joinpath("heatmaps", *parts)
    return path.parent / f"{path.name}.png"


def render_attendance_area_heatmap(
    by_attendance_area: gpd.GeoDataFrame,
    config_name: str,
    output_path: str | Path,
) -> Path:
    values = by_attendance_area["utilization"].dropna().to_numpy(dtype=float)
    max_deviation = max(float(np.max(np.abs(values - 1))) if len(values) else 0, 0.1)
    norm = TwoSlopeNorm(
        vmin=max(0, 1 - max_deviation),
        vcenter=1,
        vmax=1 + max_deviation,
    )

    figure = Figure(figsize=(12, 12), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    by_attendance_area.plot(
        ax=axis,
        column="utilization",
        cmap=UTILIZATION_COLORMAP,
        norm=norm,
        edgecolor="#505050",
        linewidth=0.55,
        missing_kwds={"color": "#d9d9d9"},
    )
    axis.set_axis_off()
    axis.set_title(str(config_name), fontsize=14, fontweight="bold", pad=10)

    for row in by_attendance_area.itertuples():
        point = row.geometry.representative_point()
        difference = row.seat_difference
        value = "No GE capacity" if pd.isna(difference) else _seat_label(difference)
        label = f"{row.attendance_area_name}\n{value}"
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
        ScalarMappable(norm=norm, cmap=UTILIZATION_COLORMAP),
        ax=axis,
        location="bottom",
        shrink=0.55,
        pad=0.015,
        aspect=35,
    )
    colorbar.set_label("GE utilization (100% = capacity)", fontsize=9)
    axis.text(
        0.5,
        -0.015,
        "Labels show seats under capacity; negative values are seats over capacity",
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


def export_attendance_area_heatmaps(
    assignment_path: str | Path,
    scenario: DataScenario,
    heatmap_data: pd.DataFrame,
) -> list[Path]:
    if heatmap_data.empty:
        return []
    area_geometry, _ = AttendanceAreaArtifactStore(scenario).geometry()
    outputs = []
    for config_name, utilization in heatmap_data.groupby("config_name", sort=True):
        by_attendance_area = attach_attendance_area_utilization(
            area_geometry, utilization
        )
        outputs.append(
            render_attendance_area_heatmap(
                by_attendance_area,
                str(config_name),
                _output_path(Path(assignment_path).expanduser(), str(config_name)),
            )
        )
    return outputs
