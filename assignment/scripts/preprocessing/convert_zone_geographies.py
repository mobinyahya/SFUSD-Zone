#!/usr/bin/env python3
"""Convert 2010 Census zone CSVs to complete 2020 geography coverage.

Relationship files are published by the Census Bureau at:
https://www2.census.gov/geo/docs/maps-data/data/rel2020/blkgrp/
https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _read_zones(path: Path, width: int) -> tuple[dict[str, int], int]:
    area_to_zone: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as zone_file:
        rows = list(csv.reader(zone_file))

    if not rows:
        raise ValueError(f"Zone file is empty: {path}")
    for zone, row in enumerate(rows):
        area_ids = [value.strip() for value in row if value.strip()]
        if not area_ids:
            raise ValueError(f"Zone {zone} is empty in {path}")
        for value in area_ids:
            area_id = str(int(value)).zfill(width)
            if area_id in area_to_zone:
                raise ValueError(f"Geography {area_id} is repeated in {path}")
            area_to_zone[area_id] = zone
    return area_to_zone, len(rows)


def _read_geometry(path: Path, geoid_column: str) -> gpd.GeoDataFrame:
    geometry = gpd.read_file(path)[[geoid_column, "geometry"]].rename(
        columns={geoid_column: "target"}
    )
    geometry["target"] = geometry["target"].astype(str)
    if geometry["target"].duplicated().any():
        raise ValueError(f"Duplicate target geographies in {path}")
    return geometry.to_crs(32610).set_index("target")


def _read_block_group_relationship(path: Path) -> pd.DataFrame:
    relationship = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "GEOID_BLKGRP_20",
            "GEOID_BLKGRP_10",
            "AREALAND_PART",
            "AREAWATER_PART",
        ],
    ).rename(
        columns={
            "GEOID_BLKGRP_20": "target",
            "GEOID_BLKGRP_10": "source",
            "AREALAND_PART": "land",
            "AREAWATER_PART": "water",
        }
    )
    return relationship.loc[relationship["target"].str.startswith("06075")]


def _read_block_relationship(path: Path) -> pd.DataFrame:
    relationship = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "STATE_2010",
            "COUNTY_2010",
            "TRACT_2010",
            "BLK_2010",
            "STATE_2020",
            "COUNTY_2020",
            "TRACT_2020",
            "BLK_2020",
            "AREALAND_INT",
            "AREAWATER_INT",
        ],
    )
    relationship["source"] = (
        relationship["STATE_2010"]
        + relationship["COUNTY_2010"]
        + relationship["TRACT_2010"]
        + relationship["BLK_2010"]
    )
    relationship["target"] = (
        relationship["STATE_2020"]
        + relationship["COUNTY_2020"]
        + relationship["TRACT_2020"]
        + relationship["BLK_2020"]
    )
    relationship = relationship.rename(
        columns={"AREALAND_INT": "land", "AREAWATER_INT": "water"}
    )
    return relationship.loc[
        (relationship["STATE_2020"] == "06") & (relationship["COUNTY_2020"] == "075"),
        ["target", "source", "land", "water"],
    ]


def _convert(
    source_path: Path,
    output_path: Path,
    relationship: pd.DataFrame,
    geometry: gpd.GeoDataFrame,
    width: int,
) -> int:
    area_to_zone, zone_count = _read_zones(source_path, width)
    overlaps = relationship.loc[relationship["source"].isin(area_to_zone)].copy()
    overlaps["zone"] = overlaps["source"].map(area_to_zone)
    overlaps[["land", "water"]] = overlaps[["land", "water"]].apply(
        pd.to_numeric, errors="raise"
    )
    overlaps = (
        overlaps.groupby(["target", "zone"], as_index=False)[["land", "water"]]
        .sum()
        .sort_values(
            ["target", "land", "water", "zone"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("target")
    )
    target_to_zone = dict(zip(overlaps["target"], overlaps["zone"], strict=True))

    target_ids = set(geometry.index)
    extra = set(target_to_zone) - target_ids
    if extra:
        raise ValueError(
            f"Relationship has targets absent from geometry: {sorted(extra)}"
        )

    missing = sorted(target_ids - set(target_to_zone))
    assigned = geometry.loc[sorted(target_to_zone)].copy()
    assigned["zone"] = assigned.index.map(target_to_zone)
    assigned["target_id"] = assigned.index
    for target in missing:
        distances = assigned.geometry.distance(geometry.at[target, "geometry"])
        nearest = assigned.assign(distance=distances).sort_values(
            ["distance", "zone", "target_id"]
        )
        target_to_zone[target] = int(nearest.iloc[0]["zone"])

    zones: list[list[int]] = [[] for _ in range(zone_count)]
    for target, zone in target_to_zone.items():
        zones[int(zone)].append(int(target))
    if any(not zone for zone in zones):
        raise ValueError(f"Conversion produced an empty zone for {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerows(sorted(zone) for zone in zones)

    flattened = [area_id for zone in zones for area_id in zone]
    if len(flattened) != len(set(flattened)) or set(flattened) != {
        int(target) for target in target_ids
    }:
        raise AssertionError(f"Incomplete target coverage in {output_path}")
    return len(missing)


def _output_path(output_dir: Path, source_path: Path) -> Path:
    return output_dir / f"{source_path.stem}_2020.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-group-relationship", type=Path, required=True)
    parser.add_argument("--block-relationship", type=Path, required=True)
    parser.add_argument("--block-group-geometry", type=Path, required=True)
    parser.add_argument("--block-geometry", type=Path, required=True)
    parser.add_argument("--block-group-source-dir", type=Path, required=True)
    parser.add_argument("--block-group-source", type=Path, action="append", default=[])
    parser.add_argument("--block-source", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    block_group_relationship = _read_block_group_relationship(
        args.block_group_relationship
    )
    block_relationship = _read_block_relationship(args.block_relationship)
    block_group_geometry = _read_geometry(args.block_group_geometry, "GEOID20")
    block_geometry = _read_geometry(args.block_geometry, "GEOID20")

    block_group_sources = sorted(args.block_group_source_dir.glob("*.csv"))
    block_group_sources.extend(args.block_group_source)
    for source_path in block_group_sources:
        output_path = _output_path(args.output_dir, source_path)
        fallback_count = _convert(
            source_path,
            output_path,
            block_group_relationship,
            block_group_geometry,
            12,
        )
        print(
            f"converted {source_path.name} -> {output_path.name} "
            f"({len(block_group_geometry)} block groups, {fallback_count} fallbacks)"
        )

    for source_path in args.block_source:
        output_path = _output_path(args.output_dir, source_path)
        fallback_count = _convert(
            source_path,
            output_path,
            block_relationship,
            block_geometry,
            15,
        )
        print(
            f"converted {source_path.name} -> {output_path.name} "
            f"({len(block_geometry)} blocks, {fallback_count} fallbacks)"
        )


if __name__ == "__main__":
    main()
