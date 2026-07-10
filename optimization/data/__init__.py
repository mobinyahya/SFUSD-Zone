"""Data layer: ingestion, graph generation, level conversion, contiguity.

This layer owns everything between the raw source files and a ready-to-solve
:class:`~optimization.problem.ZoneProblem`:

* ``loaders``      -- read the raw census/student/school/distance/adjacency data
* ``graph_builder``-- build base graphs and aggregate them into a hierarchy
* ``dataset``      -- lazily expose graphs + centroids and emit ZoneProblems
* ``conversion``   -- map assignments between any two levels
* ``contiguity``   -- strict-contiguity primitives shared by solvers/strategies
* ``geography``    -- low-level geographic distance calculations
"""
