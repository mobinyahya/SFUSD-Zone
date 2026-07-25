import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.visualize_selected_zones import (  # noqa: E402
    load_zone_assignment,
    render_comparison,
)


def test_load_zone_assignment_uses_row_index_as_zone_id(tmp_path):
    zone_file = tmp_path / "zones.csv"
    zone_file.write_text("100,101\n102,103\n", encoding="utf-8")

    assert load_zone_assignment(zone_file) == {100: 0, 101: 0, 102: 1, 103: 1}


def test_load_zone_assignment_rejects_duplicate_area(tmp_path):
    zone_file = tmp_path / "zones.csv"
    zone_file.write_text("100,101\n101,102\n", encoding="utf-8")

    with pytest.raises(ValueError, match="appears in zones 0 and 1"):
        load_zone_assignment(zone_file)


def test_render_comparison_writes_png(tmp_path):
    small = tmp_path / "Zones_2-FRL_Dev_0.25-Objective_10_BG.csv"
    medium = tmp_path / "Zones_2-FRL_Dev_0.10-Objective_20_BG.csv"
    small.write_text("100,101\n102,103\n", encoding="utf-8")
    medium.write_text("100,102\n101,103\n", encoding="utf-8")
    geometry = gpd.GeoDataFrame(
        {
            "BlockGroup": [100, 101, 102, 103],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
                box(0, 1, 1, 2),
                box(1, 1, 2, 2),
            ],
        },
        crs="EPSG:4326",
    )
    output = tmp_path / "plots" / "comparison.png"

    assert render_comparison(small, medium, output, geometry=geometry) == output
    assert output.is_file()
    assert output.stat().st_size > 0
