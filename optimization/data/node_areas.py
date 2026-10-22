"""Land area of every graph node, for area-weighted objectives.

Census block polygons are already clipped to land (they sum to ~124 km^2 for
San Francisco), so a node's area is the projected area of its base units. It
is computed when an objective needs it rather than stored on cached graphs,
so adding it changes no graph-cache key.
"""

from __future__ import annotations

import networkx as nx

from optimization.data import loaders
from optimization.data.edge_weights import BOUNDARY_CRS

LAND_AREA_ATTR = "land_area_m2"


def attach_land_areas(G: nx.Graph, unit: str, data) -> None:
    """Set ``LAND_AREA_ATTR`` (square metres) on every node of ``G``, once."""
    if all(LAND_AREA_ATTR in attrs for _, attrs in G.nodes(data=True)):
        return
    areas = _base_areas(unit, data)
    for node, attrs in G.nodes(data=True):
        base_ids = attrs.get("block_ids") or [attrs["area_id"]]
        missing = [int(a) for a in base_ids if int(a) not in areas]
        if missing:
            raise ValueError(f"No {unit} geometry for area ids {missing[:5]}.")
        attrs[LAND_AREA_ATTR] = float(sum(areas[int(a)] for a in base_ids))


# Keyed by scenario identity; the scenario is held so its id cannot be reused.
_AREAS: dict[tuple[str, int], tuple[object, dict[int, float]]] = {}


def _base_areas(unit: str, data) -> dict[int, float]:
    key = (unit, id(data))
    if key not in _AREAS:
        geometry = loaders.load_census_shapefile(unit, data).to_crs(BOUNDARY_CRS)
        geometry = geometry.assign(_area=geometry.geometry.area)
        totals = geometry.groupby(unit)["_area"].sum()
        _AREAS[key] = (data, {int(a): float(v) for a, v in totals.items()})
    return _AREAS[key][1]
