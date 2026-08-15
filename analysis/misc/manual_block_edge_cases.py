"""Generate and compile manual reviews of missing Census Block adjacency.

The review treats every Block_0 node as a candidate for every eligible school.
Each case is a ``(focal node, centroid school)`` pair for which none of the
focal node's current neighbors is strictly closer to the school centroid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

from optimization.config import OptimizationConfig
from optimization.data import loaders
from optimization.data.closer_neighbors import (
    CLOSER_NEIGHBORS_GRAPH_KEY,
    SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY,
)
from optimization.data.dataset import Dataset
from optimization.solution import graph_fingerprint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "plots" / "manual_cases"
DEFAULT_MANIFEST = Path(__file__).with_name("manual_case_manifest.json")
DEFAULT_SUMMARY = Path(__file__).with_name("manual_case_summary.csv")
DEFAULT_SELECTIONS = Path(__file__).with_name("manual_case_selections.yaml")
DEFAULT_OVERRIDES = ROOT / "Config" / "manual_block_edges.yaml"
DEFAULT_EDGE_ADDITIONS = Path(__file__).with_name("manual_block_edge_additions.yaml")
DEFAULT_COMPILED_EDGE_ADDITIONS = (
    ROOT / "Config" / "manual_block_edge_additions.yaml"
)
BASE_RADIUS_MILES = 0.25
EARTH_RADIUS_MILES = 3958.7613
DISTANCE_TOLERANCE = 1e-9
MANIFEST_SCHEMA_VERSION = 3

ROLE_COLORS = {
    "nearby": "#3b82f6",
    "neighbor": "#22c55e",
    "centroid": "#facc15",
    "focal": "#ef4444",
}


def school_centroids(dataset: Dataset) -> dict[int, int]:
    """Resolve every eligible school to its Block_0 node."""
    return {
        school_id: dataset.centroids_for("Block_0", [school_id])[0]
        for school_id in dataset.school_ids_for("Block_0")
    }


def enumerate_cases(G, centroids_by_school: dict[int, int]) -> list[dict]:
    """Return every node-school pair with no strictly closer graph neighbor."""
    closer_neighbors = G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    geometry_distances = G.graph[SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY]
    cases = []
    for school_id, centroid in sorted(centroids_by_school.items()):
        for node in G.nodes():
            if node == centroid:
                continue
            node_distance = float(geometry_distances[node][school_id])
            if closer_neighbors[node][school_id]:
                continue
            cases.append(
                {
                    "focal_node": int(node),
                    "focal_area_id": int(G.nodes[node]["area_id"]),
                    "school_id": int(school_id),
                    "centroid_node": int(centroid),
                    "centroid_area_id": int(G.nodes[centroid]["area_id"]),
                    "distance_to_centroid_miles": node_distance,
                }
            )
    cases.sort(
        key=lambda case: (
            case["focal_area_id"],
            case["school_id"],
            case["centroid_area_id"],
        )
    )
    for case_number, case in enumerate(cases, start=1):
        case["case_number"] = case_number
    return cases


def build_manifest(
    G,
    centroids_by_school: dict[int, int],
    *,
    include_nearby_non_neighbors: bool = False,
    base_radius_miles: float = BASE_RADIUS_MILES,
) -> dict:
    """Build deterministic case metadata and per-case local node labels."""
    base_radius_miles = float(base_radius_miles)
    if not math.isfinite(base_radius_miles) or base_radius_miles < 0:
        raise ValueError("base_radius_miles must be finite and non-negative.")
    nodes = list(G.nodes())
    node_indices = {node: index for index, node in enumerate(nodes)}
    latitudes = np.radians(
        np.asarray([float(G.nodes[node]["lat"]) for node in nodes])
    )
    longitudes = np.radians(
        np.asarray([float(G.nodes[node]["lon"]) for node in nodes])
    )
    focal_distances: dict[int, np.ndarray] = {}
    cases = enumerate_cases(G, centroids_by_school)

    for case in cases:
        focal = case["focal_node"]
        centroid = case["centroid_node"]
        if focal not in focal_distances:
            focal_distances[focal] = _distances_from(
                node_indices[focal], latitudes, longitudes
            )
        local_distances = focal_distances[focal]
        school_distances = np.asarray(
            [
                float(
                    G.graph[SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY][node][
                        case["school_id"]
                    ]
                )
                for node in nodes
            ]
        )
        focal_school_distance = school_distances[node_indices[focal]]
        existing_neighbors = set(G.neighbors(focal))

        radius = None
        nearest_closer_distance = None
        within_radius = set()
        if include_nearby_non_neighbors:
            missing_closer = np.asarray(
                [
                    node != focal
                    and node not in existing_neighbors
                    and school_distances[index] < focal_school_distance
                    for index, node in enumerate(nodes)
                ],
                dtype=bool,
            )
            radius = base_radius_miles
            if missing_closer.any():
                nearest_closer_distance = float(
                    local_distances[missing_closer].min()
                )
                radius = max(radius, nearest_closer_distance)
            within_radius = {
                node
                for index, node in enumerate(nodes)
                if local_distances[index] <= radius + DISTANCE_TOLERANCE
            }

        label_nodes = [focal]
        label_nodes.extend(
            sorted(
                existing_neighbors - {focal},
                key=lambda node: int(G.nodes[node]["area_id"]),
            )
        )
        label_nodes.extend(
            sorted(
                within_radius - set(label_nodes),
                key=lambda node: (
                    float(local_distances[node_indices[node]]),
                    int(G.nodes[node]["area_id"]),
                ),
            )
        )
        labels = {}
        node_labels = {}
        closer_candidate_labels = []
        for label, node in enumerate(label_nodes, start=1):
            node_labels[node] = label
            roles = []
            if node == focal:
                roles.append("focal")
            if node == centroid:
                roles.append("centroid")
            if node in existing_neighbors:
                roles.append("existing_neighbor")
            if node in within_radius:
                roles.append("within_radius")
            is_closer = (
                school_distances[node_indices[node]] < focal_school_distance
            )
            if is_closer:
                roles.append("strictly_closer")
            if (
                node != focal
                and node not in existing_neighbors
                and node in within_radius
            ):
                roles.append("missing_edge")
                if is_closer:
                    closer_candidate_labels.append(label)

            labels[str(label)] = {
                "node": int(node),
                "area_id": int(G.nodes[node]["area_id"]),
                "distance_to_focal_miles": float(
                    local_distances[node_indices[node]]
                ),
                "distance_to_centroid_miles": float(
                    school_distances[node_indices[node]]
                ),
                "roles": roles,
            }

        filename = (
            f"case_{case['case_number']:04d}_block_{case['focal_area_id']}"
            f"_school_{case['school_id']}.png"
        )
        case.update(
            {
                "plot": filename,
                "plot_radius_miles": radius,
                "nearest_closer_endpoint_miles": nearest_closer_distance,
                "include_nearby_non_neighbors": include_nearby_non_neighbors,
                "focal_label": node_labels[focal],
                "centroid_label": node_labels.get(centroid),
                "existing_neighbor_labels": sorted(
                    node_labels[node] for node in existing_neighbors
                ),
                "within_radius_labels": sorted(
                    node_labels[node] for node in within_radius
                ),
                "closer_candidate_labels": sorted(closer_candidate_labels),
                "labels": labels,
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "level": "Block_0",
        "case_definition": (
            "Every Block_0 node against every eligible school centroid; "
            "max_distance and candidate restrictions are ignored."
        ),
        "base_radius_miles": base_radius_miles,
        "include_nearby_non_neighbors": include_nearby_non_neighbors,
        "graph_fingerprint": graph_fingerprint(G),
        "node_count": G.number_of_nodes(),
        "school_count": len(centroids_by_school),
        "case_count": len(cases),
        "school_centroids": {
            str(school_id): {
                "node": int(node),
                "area_id": int(G.nodes[node]["area_id"]),
            }
            for school_id, node in sorted(centroids_by_school.items())
        },
        "cases": cases,
    }


def compile_selections(manifest: dict, selections: dict) -> tuple[list, dict]:
    """Resolve user-facing case labels to stable, deduplicated Block GEOID edges."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Manual case manifest uses an obsolete closer-neighbor definition; "
            "regenerate it before compiling selections."
        )
    cases = {int(case["case_number"]): case for case in manifest["cases"]}
    edges_to_cases: dict[tuple[int, int], set[int]] = defaultdict(set)

    for raw_case_number, raw_labels in selections.items():
        if isinstance(raw_case_number, bool):
            raise ValueError("Selection case numbers must be integers.")
        try:
            case_number = int(raw_case_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid selection case number {raw_case_number!r}."
            ) from exc
        if case_number not in cases:
            raise ValueError(f"Selection references unknown case {case_number}.")
        labels = [] if raw_labels is None else raw_labels
        if not isinstance(labels, list):
            raise ValueError(f"Selection for case {case_number} must be a list.")

        case = cases[case_number]
        valid_labels = set(case["closer_candidate_labels"])
        for raw_label in labels:
            if isinstance(raw_label, bool):
                raise ValueError(
                    f"Selection for case {case_number} contains a Boolean label."
                )
            try:
                label = int(raw_label)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Selection {raw_label!r} for case {case_number} is not a label."
                ) from exc
            if label not in valid_labels:
                raise ValueError(
                    f"Label {label} in case {case_number} is not a displayed, "
                    "strictly closer non-neighbor."
                )
            target = case["labels"][str(label)]
            edge = tuple(
                sorted((int(case["focal_area_id"]), int(target["area_id"])))
            )
            edges_to_cases[edge].add(case_number)

    edges = [list(edge) for edge in sorted(edges_to_cases)]
    provenance = {
        f"{u}:{v}": sorted(edges_to_cases[(u, v)]) for u, v in sorted(edges_to_cases)
    }
    return edges, provenance


def render_case_plot(
    case: dict,
    G,
    geometry: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Render one local inspection figure with a school-direction arrow."""
    labels = {int(label): info for label, info in case["labels"].items()}
    local_labels = {
        label
        for label, info in labels.items()
        if "within_radius" in info["roles"]
        or "existing_neighbor" in info["roles"]
        or "focal" in info["roles"]
    }
    local_nodes = [labels[label]["node"] for label in sorted(local_labels)]
    local_geometry = geometry[geometry["node"].isin(local_nodes)].copy()
    role_by_node = {
        info["node"]: _primary_role(info["roles"]) for info in labels.values()
    }
    local_geometry["color"] = local_geometry["node"].map(
        lambda node: ROLE_COLORS[role_by_node[int(node)]]
    )
    closer_nodes = {
        info["node"]
        for info in labels.values()
        if "strictly_closer" in info["roles"] and "missing_edge" in info["roles"]
    }
    local_geometry["line_width"] = local_geometry["node"].map(
        lambda node: 1.5 if int(node) in closer_nodes else 0.35
    )

    fig, local_ax = plt.subplots(figsize=(9, 9))
    focal = case["focal_node"]
    centroid = case["centroid_node"]
    focal_xy = (float(G.nodes[focal]["lon"]), float(G.nodes[focal]["lat"]))
    centroid_xy = (
        float(G.nodes[centroid]["lon"]),
        float(G.nodes[centroid]["lat"]),
    )
    if not local_geometry.empty:
        local_geometry.plot(
            ax=local_ax,
            color=local_geometry["color"],
            edgecolor="#111827",
            linewidth=local_geometry["line_width"],
            zorder=1,
        )
    for neighbor in G.neighbors(focal):
        if neighbor not in local_nodes:
            continue
        local_ax.plot(
            [focal_xy[0], float(G.nodes[neighbor]["lon"])],
            [focal_xy[1], float(G.nodes[neighbor]["lat"])],
            color="#15803d",
            linewidth=0.8,
            zorder=2,
        )
    for label in sorted(local_labels):
        info = labels[label]
        node = info["node"]
        _number_label(
            local_ax,
            float(G.nodes[node]["lon"]),
            float(G.nodes[node]["lat"]),
            label,
        )
    _set_local_bounds(local_ax, local_geometry, G, local_nodes)
    _draw_school_direction(
        local_ax,
        focal_xy,
        centroid_xy,
        case["school_id"],
        case["distance_to_centroid_miles"],
    )
    if case["include_nearby_non_neighbors"]:
        subtitle = (
            f"Review radius: {case['plot_radius_miles']:.3f} miles | "
            "outlined blue blocks are strictly closer"
        )
    else:
        subtitle = "Existing graph neighbors only"
    local_ax.set_title(subtitle)
    local_ax.set_aspect("equal", adjustable="box")
    local_ax.axis("off")

    legend = [
        mpatches.Patch(color=ROLE_COLORS["focal"], label="Focal block"),
        mpatches.Patch(color=ROLE_COLORS["neighbor"], label="Existing neighbor"),
    ]
    if case["include_nearby_non_neighbors"]:
        legend.append(
            mpatches.Patch(
                color=ROLE_COLORS["nearby"], label="Nearby non-neighbor"
            )
        )
    if case["centroid_label"] is not None:
        legend.append(
            mpatches.Patch(color=ROLE_COLORS["centroid"], label="School centroid")
        )
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False)
    fig.suptitle(
        f"Case {case['case_number']} | focal label {case['focal_label']} "
        f"(Block {case['focal_area_id']}) | school {case['school_id']}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_block_geometry(G) -> gpd.GeoDataFrame:
    """Load one WGS84 polygon per graph node."""
    base = loaders.load_census_shapefile("Block")
    area_to_node = {
        int(attrs["area_id"]): int(node) for node, attrs in G.nodes(data=True)
    }
    geometry = base[["Block", "geometry"]].dropna().copy()
    geometry["Block"] = geometry["Block"].astype("int64")
    geometry["node"] = geometry["Block"].map(area_to_node)
    geometry = geometry.dropna(subset=["node"]).copy()
    geometry["node"] = geometry["node"].astype(int)
    geometry = geometry.dissolve(by="node", as_index=False)[["node", "geometry"]]
    if geometry.crs is None:
        geometry = geometry.set_crs(epsg=4326, allow_override=True)
    else:
        geometry = geometry.to_crs(epsg=4326)
    return geometry


def generate(
    config_path: Path | None,
    output_dir: Path,
    manifest_path: Path,
    summary_path: Path,
    selections_path: Path,
    *,
    plots: bool,
    overwrite: bool,
    include_nearby_non_neighbors: bool = False,
    base_radius_miles: float = BASE_RADIUS_MILES,
) -> dict:
    config = (
        OptimizationConfig.from_yaml(str(config_path))
        if config_path is not None
        else OptimizationConfig(levels=["Block_0"])
    )
    if config.unit != "Block":
        raise ValueError("Manual edge cases require a Block optimization config.")
    dataset = Dataset(config)
    G = dataset.graph_for("Block_0")
    dataset.closer_neighbors_for("Block_0")
    centroids = school_centroids(dataset)
    manifest = build_manifest(
        G,
        centroids,
        include_nearby_non_neighbors=include_nearby_non_neighbors,
        base_radius_miles=base_radius_miles,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    _write_summary(summary_path, manifest)
    if not selections_path.exists():
        selections_path.write_text(
            "# case_number: [local_node_label, ...]\n{}\n", encoding="utf-8"
        )

    if plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        geometry = load_block_geometry(G)
        for index, case in enumerate(manifest["cases"], start=1):
            path = output_dir / case["plot"]
            if overwrite or not path.exists():
                render_case_plot(case, G, geometry, path)
            if index % 100 == 0 or index == manifest["case_count"]:
                print(f"Rendered {index}/{manifest['case_count']} cases")
    return manifest


def compile_file(manifest_path: Path, selections_path: Path, output_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    with selections_path.open("r", encoding="utf-8") as file:
        selections = yaml.safe_load(file) or {}
    if not isinstance(selections, dict):
        raise ValueError("Manual case selections must be a case-to-label mapping.")
    edges, provenance = compile_selections(manifest, selections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "edges": edges,
                "source_cases": provenance,
                "review_graph_fingerprint": manifest["graph_fingerprint"],
            },
            file,
            sort_keys=False,
        )
    print(f"Compiled {len(edges)} unique manual Block edges to {output_path}")
    return len(edges)


def compile_edge_additions(additions: dict) -> list[list[int]]:
    """Normalize explicit focal-GEOID to neighbor-GEOID declarations."""
    if not isinstance(additions, dict):
        raise ValueError("Manual edge additions must be a GEOID-to-neighbors mapping.")

    edges = set()
    for raw_focal, raw_neighbors in additions.items():
        if isinstance(raw_focal, bool):
            raise ValueError("Manual edge addition focal GEOIDs must be integers.")
        try:
            focal = int(raw_focal)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid manual edge addition focal GEOID {raw_focal!r}."
            ) from exc

        neighbors = [] if raw_neighbors is None else raw_neighbors
        if not isinstance(neighbors, list):
            raise ValueError(
                f"Manual edge additions for Block {focal} must be a list."
            )
        for raw_neighbor in neighbors:
            if isinstance(raw_neighbor, bool):
                raise ValueError(
                    f"Manual edge additions for Block {focal} contain a Boolean."
                )
            try:
                neighbor = int(raw_neighbor)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid neighbor GEOID {raw_neighbor!r} for Block {focal}."
                ) from exc
            if focal == neighbor:
                raise ValueError(
                    f"Manual edge addition for Block {focal} is a self-edge."
                )
            edges.add(tuple(sorted((focal, neighbor))))
    return [list(edge) for edge in sorted(edges)]


def compile_edge_additions_file(additions_path: Path, output_path: Path) -> int:
    with additions_path.open("r", encoding="utf-8") as file:
        additions = yaml.safe_load(file) or {}
    edges = compile_edge_additions(additions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"edges": edges}, file, sort_keys=False)
    print(f"Compiled {len(edges)} explicit Block edges to {output_path}")
    return len(edges)


def _distances_from(
    source_index: int, latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    dlat = latitudes - latitudes[source_index]
    dlon = longitudes - longitudes[source_index]
    haversine = np.sin(dlat / 2) ** 2 + (
        np.cos(latitudes[source_index])
        * np.cos(latitudes)
        * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_MILES * 2 * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )


def _primary_role(roles: list[str]) -> str:
    if "focal" in roles:
        return "focal"
    if "centroid" in roles:
        return "centroid"
    if "existing_neighbor" in roles:
        return "neighbor"
    return "nearby"


def _number_label(ax, x: float, y: float, label: int) -> None:
    text = ax.text(
        x,
        y,
        str(label),
        fontsize=6.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=4,
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=1.7, foreground="#111827"), path_effects.Normal()]
    )


def _set_local_bounds(ax, geometry, G, nodes: list[int]) -> None:
    xs = [float(G.nodes[node]["lon"]) for node in nodes]
    ys = [float(G.nodes[node]["lat"]) for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if not geometry.empty:
        geo_min_x, geo_min_y, geo_max_x, geo_max_y = geometry.total_bounds
        min_x, min_y = min(min_x, geo_min_x), min(min_y, geo_min_y)
        max_x, max_y = max(max_x, geo_max_x), max(max_y, geo_max_y)
    width = max(max_x - min_x, 0.002)
    height = max(max_y - min_y, 0.002)
    ax.set_xlim(min_x - width * 0.08, max_x + width * 0.08)
    ax.set_ylim(min_y - height * 0.08, max_y + height * 0.08)


def _draw_school_direction(
    ax,
    focal_xy: tuple[float, float],
    centroid_xy: tuple[float, float],
    school_id: int,
    distance_miles: float,
) -> None:
    """Draw an arrow from the focal block toward an off-plot school."""
    dx = centroid_xy[0] - focal_xy[0]
    dy = centroid_xy[1] - focal_xy[1]
    if dx == 0 and dy == 0:
        return

    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    scales = []
    if dx > 0:
        scales.append((max_x - focal_xy[0]) / dx)
    elif dx < 0:
        scales.append((min_x - focal_xy[0]) / dx)
    if dy > 0:
        scales.append((max_y - focal_xy[1]) / dy)
    elif dy < 0:
        scales.append((min_y - focal_xy[1]) / dy)
    boundary_scale = min(scale for scale in scales if scale > 0)
    arrow_scale = min(1.0, boundary_scale * 0.88)
    endpoint = (
        focal_xy[0] + dx * arrow_scale,
        focal_xy[1] + dy * arrow_scale,
    )
    ax.annotate(
        "",
        xy=endpoint,
        xytext=focal_xy,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#ca8a04",
            "linewidth": 2.0,
            "mutation_scale": 14,
        },
        zorder=3,
    )
    ax.annotate(
        f"School {school_id}\n{distance_miles:.2f} mi",
        xy=endpoint,
        xytext=(-4 if dx > 0 else 4, -4 if dy > 0 else 4),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color="#854d0e",
        ha="right" if dx > 0 else "left",
        va="top" if dy > 0 else "bottom",
        zorder=4,
    )


def _write_summary(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_number",
        "plot",
        "focal_node",
        "focal_area_id",
        "school_id",
        "centroid_node",
        "centroid_area_id",
        "distance_to_centroid_miles",
        "plot_radius_miles",
        "closer_candidate_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case in manifest["cases"]:
            row = {field: case.get(field) for field in fields}
            row["closer_candidate_count"] = len(case["closer_candidate_labels"])
            writer.writerow(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path)
    generate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    generate_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    generate_parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    generate_parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    generate_parser.add_argument(
        "--include-nearby-non-neighbors",
        action="store_true",
        help="Include radius-based non-neighbors as blue edge candidates.",
    )
    generate_parser.add_argument(
        "--base-radius-miles",
        type=float,
        default=BASE_RADIUS_MILES,
        help="Initial radius used when nearby non-neighbors are enabled.",
    )
    generate_parser.add_argument("--skip-plots", action="store_true")
    generate_parser.add_argument("--overwrite", action="store_true")

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    compile_parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OVERRIDES)

    additions_parser = subparsers.add_parser("compile-additions")
    additions_parser.add_argument(
        "--additions", type=Path, default=DEFAULT_EDGE_ADDITIONS
    )
    additions_parser.add_argument(
        "--output", type=Path, default=DEFAULT_COMPILED_EDGE_ADDITIONS
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        manifest = generate(
            args.config,
            args.output_dir,
            args.manifest,
            args.summary,
            args.selections,
            plots=not args.skip_plots,
            overwrite=args.overwrite,
            include_nearby_non_neighbors=args.include_nearby_non_neighbors,
            base_radius_miles=args.base_radius_miles,
        )
        print(
            f"Generated {manifest['case_count']} cases for "
            f"{manifest['school_count']} schools."
        )
    elif args.command == "compile":
        compile_file(args.manifest, args.selections, args.output)
    else:
        compile_edge_additions_file(args.additions, args.output)


if __name__ == "__main__":
    main()
