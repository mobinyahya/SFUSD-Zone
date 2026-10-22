"""Data layer: ingestion, graph generation, level conversion, contiguity.

This layer turns shared, scenario-resolved source tables into a ready-to-solve
:class:`~optimization.problem.ZoneProblem`:

* ``loaders``      -- optimization-specific transforms over top-level loaders
* ``graph_builder``-- build base graphs and aggregate them into a hierarchy
* ``dataset``      -- expose content-addressed graphs, centroids and problems
* ``conversion``   -- map assignments between any two levels
* ``contiguity``   -- strict-contiguity primitives shared by solvers/strategies
* ``geography``    -- low-level geographic distance calculations

Raw paths and generated-cache roots are never selected here; they come from a
top-level :class:`loaders.config.DataScenario`.
"""
