"""Dependency-free option vocabularies for the Dantzig--Wolfe strategy.

``optimization.config`` validates these names, ``optimization.zone_columns``
turns a master method into a Gurobi parameter, and
``optimization.strategies.dantzig_wolfe`` dispatches on an objective. Keeping
the vocabularies here avoids importing a solver backend just to read a config,
and avoids three copies of the same tuple drifting apart. Same role as
``optimization/mid_options.py``.
"""

from __future__ import annotations


#: The welfare definitions the decomposition can price.
#:
#: ``mid``
#:     finite-grid least-cutoff MID welfare.
#: ``stable_matching``
#:     deferred-acceptance welfare of one fixed tie-breaking draw.
#: ``boundary``
#:     minus half the zone perimeter; the base compactness objective, kept
#:     because it makes the decomposition testable without a market.
DW_OBJECTIVES = ("mid", "stable_matching", "boundary")

#: Gurobi ``Method`` values for the master LP, named by what they mean for the
#: duals. ``barrier`` is paired with ``Crossover = 0`` so the duals stay
#: interior; every other setting returns a vertex of the dual polyhedron.
DW_MASTER_METHODS = {
    "auto": -1,
    "primal": 0,
    "dual": 1,
    "barrier": 2,
}
