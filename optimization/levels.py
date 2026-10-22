"""Level specifications.

A *level* identifies a geographic granularity in the hierarchy. It is the pair
``(unit, depth)``:

* ``unit``  -- the base geographic unit: ``"Block"``, ``"BlockGroup"``,
  ``"Tract"``, or ``"attendance_area"``.
* ``depth`` -- the aggregation depth, where ``0`` is the finest (one node per
  unit) and larger numbers are coarser aggregated graphs. A fractional depth
  (``Block_0.5``) names a level inserted between two integer depths.

The string form is ``"<unit>_<depth>"`` (e.g. ``"BlockGroup_0"``,
``"Block_2"``). This is the single source of truth for parsing/formatting level
names, which the legacy code scattered across modules and frequently
hardcoded. Switching levels is now purely a matter of naming a different
``LevelSpec`` -- no manual file edits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Recognized base units. ``attendance_area`` is supported for parsing parity
# with the legacy data. Census Block, BlockGroup, and Tract are active units.
KNOWN_UNITS = ("BlockGroup", "Block", "Tract", "attendance_area")
_DEPTH = re.compile(r"\d+(?:\.\d+)?")

# Requested graph sizes include school nodes, which remain singleton vertices.
# KaHIP may return fewer nonempty partitions, so these are upper targets.
LEVEL_NODE_TARGETS = {
    "Block": {0.5: 3000, 1: 1200, 2: 579, 3: 250, 4: 125},
    "BlockGroup": {1: 250, 2: 125},
    "Tract": {},
}

# Depths coarsened from somewhere other than ``depth - 1``. Block_0.5 is built
# from Block_0 (KaHIP returns ~1,850 nodes for its 3,000 target, with no
# aggregate above ~17 students against Block_1's ~46); Block_1 is still built
# from Block_0, so the integer chain and its graphs are unchanged.
LEVEL_PARENT_DEPTHS = {
    "Block": {0.5: 0},
}


@dataclass(frozen=True, order=True)
class LevelSpec:
    """An immutable ``(unit, depth)`` granularity identifier."""

    unit: str
    depth: int | float

    def __post_init__(self) -> None:
        if self.unit not in KNOWN_UNITS:
            raise ValueError(
                f"Unknown unit {self.unit!r}; expected one of {KNOWN_UNITS}."
            )
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}.")
        # Block_1.0 and Block_1 are the same level, and must hash and print alike.
        if isinstance(self.depth, float) and self.depth.is_integer():
            object.__setattr__(self, "depth", int(self.depth))

    # ------------------------------------------------------------------ #
    # parsing / formatting
    # ------------------------------------------------------------------ #
    @classmethod
    def parse(cls, value: "str | LevelSpec") -> "LevelSpec":
        """Parse a ``"<unit>_<depth>"`` string (or pass through a LevelSpec)."""
        if isinstance(value, LevelSpec):
            return value
        if not isinstance(value, str):
            raise TypeError(f"Cannot parse level from {type(value).__name__}.")
        # Units may themselves contain underscores (attendance_area), so split
        # off only the trailing depth component.
        unit, _, depth = value.rpartition("_")
        if not unit or not _DEPTH.fullmatch(depth):
            raise ValueError(
                f"Malformed level {value!r}; expected '<unit>_<depth>', "
                f"e.g. 'BlockGroup_0' or 'Block_0.5'."
            )
        return cls(unit=unit, depth=float(depth) if "." in depth else int(depth))

    def __str__(self) -> str:
        return f"{self.unit}_{self.depth}"

    @property
    def name(self) -> str:
        return str(self)

    @property
    def filename(self) -> str:
        """Pickle filename for this level's cached graph."""
        return f"{self}.pickle"

    @property
    def is_base(self) -> bool:
        """True if this is the finest (depth 0) graph for its unit."""
        return self.depth == 0

    def parent(self) -> "LevelSpec":
        """The level this one is coarsened from (see ``LEVEL_PARENT_DEPTHS``)."""
        if self.is_base:
            raise ValueError("Base level has no parent level within its unit.")
        depth = LEVEL_PARENT_DEPTHS.get(self.unit, {}).get(self.depth, self.depth - 1)
        return LevelSpec(self.unit, depth)
