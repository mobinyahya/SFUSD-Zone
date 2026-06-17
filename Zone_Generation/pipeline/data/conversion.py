"""Cross-level assignment conversion.

Assignments are produced at one level but often consumed at another -- a coarse
solution seeds a finer one (recursive zoning), or a Block solution is reported
at BlockGroup resolution. The lingua franca is the *area assignment*: a
``{base_area_id: zone}`` dict at a unit's finest resolution, since every node at
every depth records the base ids it covers (``area_id`` at depth 0,
``block_ids`` on aggregated nodes).

:class:`LevelConverter` translates between any two levels:

* **same unit, different depth** -- direct id lookup,
* **different unit** (Block <-> BlockGroup) -- bridged by the
  ``block_blockgroup_tract.csv`` mapping, with majority voting when several
  source areas fall in one target area.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx

from Helper_Functions.util import load_b2bg
from Zone_Generation.pipeline.levels import LevelSpec


def base_area_assignment(G: nx.Graph, assignment: dict[int, int]) -> dict[int, int]:
    """Expand a node->zone map to ``{base_area_id: zone}`` for ``G``'s unit."""
    out: dict[int, int] = {}
    for node, zone in assignment.items():
        attrs = G.nodes[node]
        if "area_id" in attrs:
            out[attrs["area_id"]] = zone
        else:
            for block_id in attrs["block_ids"]:
                out[block_id] = zone
    return out


def _node_base_ids(G: nx.Graph, node: int) -> list[int]:
    attrs = G.nodes[node]
    if "area_id" in attrs:
        return [attrs["area_id"]]
    return list(attrs["block_ids"])


class LevelConverter:
    """Translate zone assignments between levels."""

    def __init__(self, is_local: bool = False):
        self.is_local = is_local
        self._b2bg: dict | None = None
        self._bg2blocks: dict | None = None

    # ------------------------------------------------------------------ #
    # unit bridging maps (lazy)
    # ------------------------------------------------------------------ #
    def b2bg(self) -> dict:
        if self._b2bg is None:
            self._b2bg = load_b2bg(self.is_local)
        return self._b2bg

    def bg2blocks(self) -> dict:
        if self._bg2blocks is None:
            bg2blocks: dict[int, list[int]] = {}
            for block, bg in self.b2bg().items():
                bg2blocks.setdefault(bg, []).append(block)
            self._bg2blocks = bg2blocks
        return self._bg2blocks

    # ------------------------------------------------------------------ #
    # conversion
    # ------------------------------------------------------------------ #
    def between(
        self,
        src_G: nx.Graph,
        src_assignment: dict[int, int],
        src_level: LevelSpec,
        dst_G: nx.Graph,
        dst_level: LevelSpec,
    ) -> dict[int, int]:
        """Return a node->zone assignment for ``dst_G``.

        Nodes the source does not cover are omitted; callers treat a missing
        node as "no hint".
        """
        area = base_area_assignment(src_G, src_assignment)
        lookup = self._zone_lookup(area, src_level.unit, dst_level.unit)

        result: dict[int, int] = {}
        for node in dst_G.nodes():
            votes = Counter()
            for base_id in _node_base_ids(dst_G, node):
                zone = lookup(base_id)
                if zone is not None:
                    votes[zone] += 1
            if votes:
                result[node] = votes.most_common(1)[0][0]
        return result

    def _zone_lookup(self, area: dict[int, int], src_unit: str, dst_unit: str):
        """Build a ``base_id(dst_unit) -> zone`` resolver."""
        if src_unit == dst_unit:
            return lambda base_id: area.get(base_id)

        if src_unit == "BlockGroup" and dst_unit == "Block":
            b2bg = self.b2bg()
            return lambda block_id: area.get(b2bg.get(block_id))

        if src_unit == "Block" and dst_unit == "BlockGroup":
            bg2blocks = self.bg2blocks()

            def lookup(bg_id: int):
                votes = Counter(
                    area[b] for b in bg2blocks.get(bg_id, []) if b in area
                )
                return votes.most_common(1)[0][0] if votes else None

            return lookup

        raise ValueError(
            f"Unsupported unit conversion {src_unit!r} -> {dst_unit!r}."
        )
