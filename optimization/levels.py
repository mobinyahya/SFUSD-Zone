"""Level specifications.

A *level* identifies a geographic granularity in the hierarchy. It is the pair
``(unit, depth)``:

* ``unit``  -- the base geographic unit: ``"Block"``, ``"BlockGroup"``,
  ``"Tract"``, or ``"attendance_area"``.
* ``depth`` -- the aggregation depth, where ``0`` is the finest (one node per
  unit) and larger numbers are coarser aggregated graphs, each coarsened from
  the one before it.

The string form is ``"<unit>_<depth>"`` (e.g. ``"BlockGroup_0"``,
``"Block_2"``). This is the single source of truth for parsing/formatting level
names, which the legacy code scattered across modules and frequently
hardcoded. Switching levels is now purely a matter of naming a different
``LevelSpec`` -- no manual file edits.
"""

from __future__ import annotations

from dataclasses import dataclass

# Recognized base units. ``attendance_area`` is supported for parsing parity
# with the legacy data. Census Block, BlockGroup, and Tract are active units.
KNOWN_UNITS = ("BlockGroup", "Block", "Tract", "attendance_area")

# Requested graph sizes include school nodes, which remain singleton vertices.
# KaHIP may return fewer nonempty partitions, so these are upper targets.
LEVEL_NODE_TARGETS = {
    "Block": {1: 3000, 2: 1200, 3: 579, 4: 250},
    "BlockGroup": {1: 250, 2: 125},
    "Tract": {},
}


@dataclass(frozen=True, order=True)
class LevelSpec:
    """An immutable ``(unit, depth)`` granularity identifier."""

    unit: str
    depth: int

    def __post_init__(self) -> None:
        if self.unit not in KNOWN_UNITS:
            raise ValueError(
                f"Unknown unit {self.unit!r}; expected one of {KNOWN_UNITS}."
            )
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {self.depth}.")

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
        if not unit or not depth.isdigit():
            raise ValueError(
                f"Malformed level {value!r}; expected '<unit>_<depth>', "
                f"e.g. 'BlockGroup_0'."
            )
        return cls(unit=unit, depth=int(depth))

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
        """The level this one is coarsened from: the next finer depth."""
        if self.is_base:
            raise ValueError("Base level has no parent level within its unit.")
        return LevelSpec(self.unit, self.depth - 1)


def level_unit(value: "str | LevelSpec") -> str:
    """The unit of a level name, for readers that need nothing else.

    Saved zonings are keyed by their unit's base ids, so a reader that only
    looks up those ids also accepts level names this hierarchy no longer
    defines, such as the retired fractional ``Block_0.5``.
    """
    if isinstance(value, LevelSpec):
        return value.unit
    unit = str(value).rpartition("_")[0]
    if unit not in KNOWN_UNITS:
        raise ValueError(f"Malformed level {value!r}; expected '<unit>_<depth>'.")
    return unit
