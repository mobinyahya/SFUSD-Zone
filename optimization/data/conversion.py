"""Cross-level assignment conversion.

Assignments are produced at one level but often consumed at another -- a coarse
solution seeds a finer one (recursive zoning), or a Block solution is reported
at BlockGroup resolution. The lingua franca is the *area assignment*: a
``{base_area_id: zone}`` dict at a unit's finest resolution, since every node at
every depth records the base ids it covers (``area_id`` at depth 0,
``block_ids`` on aggregated nodes).

:class:`LevelConverter` translates between any two levels:

* **same unit, different depth** -- direct id lookup,
* **different unit** (Block, BlockGroup, or Tract) -- bridged by the scenario's
  ``optimization.crosswalk`` role, with majority voting when several source
  areas fall in one target area.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import networkx as nx

from loaders import DataScenario, load_scenario
from loaders.tables import read_csv
from optimization.levels import LevelSpec


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


def _load_block_to_blockgroup(data: DataScenario | None = None) -> dict[int, int]:
    """Load the selected geographic Block-to-BlockGroup relationship."""
    scenario = data or load_scenario({"scenario": "legacy", "overrides": {}})
    crosswalk = read_csv(scenario, "optimization.crosswalk", low_memory=False).dropna(
        subset=["Block", "BlockGroup"]
    )
    return {
        int(row.Block): int(row.BlockGroup)
        for row in crosswalk[["Block", "BlockGroup"]].itertuples(index=False)
    }


def _load_block_to_tract(data: DataScenario | None = None) -> dict[int, int]:
    """Load the Block-to-Tract relationship from the selected crosswalk."""
    scenario = data or load_scenario({"scenario": "legacy", "overrides": {}})
    crosswalk = read_csv(scenario, "optimization.crosswalk", low_memory=False).dropna(
        subset=["Block", "Tract"]
    )
    return {
        int(row.Block): int(row.Tract)
        for row in crosswalk[["Block", "Tract"]].itertuples(index=False)
    }


class LevelConverter:
    """Translate zone assignments between levels."""

    def __init__(
        self,
        block_to_blockgroup: Mapping[int, int] | None = None,
        *,
        block_to_tract: Mapping[int, int] | None = None,
        data: DataScenario | None = None,
    ) -> None:
        self._b2bg = (
            dict(block_to_blockgroup) if block_to_blockgroup is not None else None
        )
        self.data = data
        self._b2tract = dict(block_to_tract) if block_to_tract is not None else None
        self._bg2blocks: dict[int, list[int]] | None = None
        self._tract2blocks: dict[int, list[int]] | None = None

    # ------------------------------------------------------------------ #
    # unit bridging maps (lazy)
    # ------------------------------------------------------------------ #
    def b2bg(self) -> dict[int, int]:
        if self._b2bg is None:
            self._b2bg = _load_block_to_blockgroup(self.data)
        return self._b2bg

    def bg2blocks(self) -> dict[int, list[int]]:
        if self._bg2blocks is None:
            bg2blocks: dict[int, list[int]] = {}
            for block, bg in self.b2bg().items():
                bg2blocks.setdefault(bg, []).append(block)
            self._bg2blocks = bg2blocks
        return self._bg2blocks

    def b2tract(self) -> dict[int, int]:
        if self._b2tract is None:
            self._b2tract = _load_block_to_tract(self.data)
        return self._b2tract

    def tract2blocks(self) -> dict[int, list[int]]:
        if self._tract2blocks is None:
            tract2blocks: dict[int, list[int]] = {}
            for block, tract in self.b2tract().items():
                tract2blocks.setdefault(tract, []).append(block)
            self._tract2blocks = tract2blocks
        return self._tract2blocks

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
        return self.from_area_assignment(area, src_level, dst_G, dst_level)

    def from_area_assignment(
        self,
        src_area_assignment: Mapping[int, int],
        src_level: LevelSpec | str,
        dst_G: nx.Graph,
        dst_level: LevelSpec | str,
    ) -> dict[int, int]:
        """Convert an already portable finest-area assignment to ``dst_G``."""
        src_level = LevelSpec.parse(src_level)
        dst_level = LevelSpec.parse(dst_level)
        area = {
            int(area_id): int(zone) for area_id, zone in src_area_assignment.items()
        }
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

        supported = {"Block", "BlockGroup", "Tract"}
        if src_unit not in supported or dst_unit not in supported:
            raise ValueError(
                f"Unsupported unit conversion {src_unit!r} -> {dst_unit!r}."
            )

        block_to_source = {
            "Block": lambda block: block,
            "BlockGroup": lambda block: self.b2bg().get(block),
            "Tract": lambda block: self.b2tract().get(block),
        }[src_unit]
        target_to_blocks = {
            "Block": lambda target: [target],
            "BlockGroup": lambda target: self.bg2blocks().get(target, []),
            "Tract": lambda target: self.tract2blocks().get(target, []),
        }[dst_unit]

        def lookup(target_id: int):
            source_ids = {
                source_id
                for block in target_to_blocks(target_id)
                if (source_id := block_to_source(block)) is not None
            }
            votes = Counter(
                area[source_id] for source_id in source_ids if source_id in area
            )
            return votes.most_common(1)[0][0] if votes else None

        return lookup
