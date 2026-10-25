"""Execute named solves with explicit solution inputs."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import os
import time

import networkx as nx

from optimization.data import contiguity
from optimization.data.capacity_reference import (
    apply_capacity_reference,
    build_capacity_reference,
)
from optimization.data.conversion import (
    LevelConverter,
    _node_base_ids,
    base_area_assignment,
)
from optimization.objectives import get_objective, solve_objective
from optimization.levels import LevelSpec
from optimization.pipeline import step_config
from optimization.strategies.base import Strategy, register


def selected_nodes(graph, assignment, binding):
    """Select a boundary band in O(nodes + edges), independently of zone count."""
    selected = set(assignment)
    if binding.select != "all":
        band = set()
        for u, v in graph.edges:
            if u in assignment and v in assignment and assignment[u] != assignment[v]:
                band.update((u, v))
        queue = deque((node, 0) for node in band)
        while queue:
            node, distance = queue.popleft()
            if distance == binding.radius:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in band:
                    band.add(neighbor)
                    queue.append((neighbor, distance + 1))
        selected = selected - band if binding.select == "interior" else selected & band
    if binding.nodes is not None:
        unknown = set(binding.nodes) - set(graph)
        if unknown:
            raise ValueError(f"Input selects unknown target nodes: {sorted(unknown)}.")
        selected.intersection_update(binding.nodes)
    return selected


def detached_fixed_pieces(problem, fixed):
    """Hard assignments that reach their centroid only through free blocks.

    Releasing a boundary band can leave a piece of a zone's fixed interior
    whose every connection to the zone's centroid runs through the band. Kept
    fixed, it forces the solver to hold a same-zone path through blocks the
    step meant to free. Returns ``{node: zone}`` for every such piece, which the
    pipeline demotes to hints.
    """
    by_zone = {}
    for node, zone in fixed.items():
        by_zone.setdefault(zone, set()).add(node)
    detached = {}
    for zone, nodes in by_zone.items():
        centroid = problem.centroids[zone]
        for piece in nx.connected_components(problem.G.subgraph(nodes | {centroid})):
            if centroid not in piece:
                detached.update({node: zone for node in piece})
    return detached


def relax_unsupported_assignments(problem, fixed, assignment):
    """Loosen hard assignments the CP contiguity rows could never satisfy.

    The MIP/CP solvers require every non-centroid block to have a same-zone
    neighbor strictly closer to its school, which a prior step's zoning (ReCom
    zonings are only connected) need not give. As in the recursive strategy,
    :func:`relax_unsupported_candidates` frees each unsupported block and adds
    its zone as a candidate along its closer-neighbor paths, so a fixed block
    on such a path keeps its own zone and gains only the ones it may need.
    Returns ``(fixed, candidates)``: still-forced blocks, and blocks the
    relaxation widened, with their allowed zones.
    """
    if not fixed:
        return fixed, {}
    relaxed = contiguity.relax_unsupported_candidates(
        problem.G,
        assignment,
        list(problem.centroids),
        problem.centroid_school_ids,
        {node: {zone} for node, zone in fixed.items()},
    )
    kept, candidates = {}, {}
    for node, zone in fixed.items():
        zones = relaxed.get(node)
        if zones == {zone}:
            kept[node] = zone
        elif zones is not None:
            candidates[node] = zones
    return kept, candidates


def _base_area_candidates(G, candidates):
    """``{base_area_id: sorted zones}`` so a saved stage can rebuild them."""
    return {
        area: sorted(zones)
        for node, zones in candidates.items()
        for area in _node_base_ids(G, node)
    }


@register("pipeline")
class PipelineStrategy(Strategy):
    def __init__(self, *, config=None, **options):
        if config is None:
            from optimization.config import OptimizationConfig

            config = OptimizationConfig(strategy="pipeline", **options)
        super().__init__(config=config, boundary_prop=config.boundary_prop)
        self.config = config

    @property
    def reads_assignment_market(self):
        return any(
            step.capacity_reference is not None
            or get_objective(step.objective).reads_assignment_market
            for step in self.config.steps
        )

    def problem_for_stage(self, dataset, metadata):
        """Reconstruct the feasible set of a saved benchmark stage."""
        index = metadata["pipeline_step_index"]
        step = self.config.steps[index]
        fixed = {int(n): int(z) for n, z in metadata.get("pipeline_fixed", {}).items()}
        if "pipeline_fixed_areas" in metadata:
            graph = dataset.graph_for(step.level)
            areas = {
                int(n): int(z) for n, z in metadata["pipeline_fixed_areas"].items()
            }
            fixed = LevelConverter(data=dataset.data).from_area_assignment(
                areas, step.level, graph, step.level
            )
        candidates = {}
        if metadata.get("pipeline_candidates_areas"):
            graph = dataset.graph_for(step.level)
            areas = {
                int(a): {int(z) for z in zones}
                for a, zones in metadata["pipeline_candidates_areas"].items()
            }
            for node in graph:
                ids = [a for a in _node_base_ids(graph, node) if a in areas]
                if ids:
                    candidates[node] = areas[ids[0]]
        problem = dataset.problem_for(
            step.level,
            config=step_config(self.config, step, index),
            fixed=fixed or None,
            candidates=candidates or None,
        )
        if step.capacity_reference is not None:
            if "capacity_reference" not in metadata:
                raise ValueError(
                    "Saved stage is missing its frozen capacity reference."
                )
            apply_capacity_reference(problem, metadata["capacity_reference"])
        return problem

    def run(self, dataset, solver):
        converter = LevelConverter(data=dataset.data)
        results = []
        by_name = {}
        projected_cache = {}
        capacity_references = {}
        carry = 0.0
        output_dir = solver.options.get("output_dir")
        for index, step in enumerate(self.config.steps):
            started = time.perf_counter()
            effective = step_config(self.config, step, index)
            problem = dataset.problem_for(step.level, config=effective)
            reference = None
            if step.capacity_reference is not None:
                source = (
                    results[-1]
                    if step.capacity_reference == "previous"
                    else by_name[step.capacity_reference]
                )
                source_name = source.metadata["pipeline_step"]
                if source_name not in capacity_references:
                    if not source.feasible or set(source.assignment) != set(
                        source.problem.G
                    ):
                        raise ValueError(
                            "A capacity reference needs a complete feasible source."
                        )
                    source_config = source.problem.optimization_config
                    base = dataset.problem_for(
                        LevelSpec(source.level.unit, 0), config=source_config
                    )
                    zoning = converter.between(
                        source.problem.G,
                        source.assignment,
                        source.level,
                        base.G,
                        base.level,
                    )
                    from optimization.data.mid import build_mid_market

                    reference = build_capacity_reference(
                        base,
                        zoning,
                        build_mid_market(base, source_config),
                        source_config.mid_lottery_scale,
                    )
                    reference["source_step"] = source_name
                    reference["preference_source"] = source_config.mid_preference_source
                    capacity_references[source_name] = reference
                reference = capacity_references[source_name]
                apply_capacity_reference(problem, reference)
            fixed, hints = {}, dict(problem.hint or {})
            for binding in step.inputs:
                source = (
                    results[-1]
                    if binding.source == "previous"
                    else by_name[binding.source]
                )
                if not source.feasible or set(source.assignment) != set(
                    source.problem.G
                ):
                    raise ValueError(
                        f"Step {step.name!r} needs a complete feasible input from {binding.source!r}."
                    )
                key = (id(source), step.level)
                if key not in projected_cache:
                    projected_cache[key] = converter.between(
                        source.problem.G,
                        source.assignment,
                        source.level,
                        problem.G,
                        problem.level,
                    )
                projected = projected_cache[key]
                for node in selected_nodes(problem.G, projected, binding):
                    zone = projected[node]
                    if zone not in range(problem.Z):
                        raise ValueError(
                            f"Input supplies invalid zone {zone} for node {node}."
                        )
                    target = fixed if binding.mode == "assign" else hints
                    if (
                        binding.mode == "assign"
                        and node in fixed
                        and fixed[node] != zone
                    ):
                        raise ValueError(
                            f"Conflicting hard assignments for node {node} in step {step.name!r}."
                        )
                    target[node] = zone
            for zone, centroid in enumerate(problem.centroids):
                if centroid in fixed and fixed[centroid] != zone:
                    raise ValueError(
                        f"Hard assignment conflicts with centroid {centroid} in step {step.name!r}."
                    )
            # A source solved on coarser nodes measured max_distance from those
            # nodes, so a projected block can lie beyond the distance this step
            # allows its zone. Free it to its own distance candidates.
            out_of_range = {
                node: zone
                for node, zone in fixed.items()
                if zone not in problem.candidate_zones(node)
            }
            for node in out_of_range:
                del fixed[node]
                hints.pop(node, None)
            islands = detached_fixed_pieces(problem, fixed)
            for node, zone in islands.items():
                del fixed[node]
                hints[node] = zone
            hints.update(fixed)
            fixed, candidates = relax_unsupported_assignments(problem, fixed, hints)
            problem.fixed = fixed or None
            problem.candidates = candidates or None
            if hints:
                hints.update(
                    {centroid: zone for zone, centroid in enumerate(problem.centroids)}
                )
            problem.hint = hints or None
            step_solver = effective.make_solver(output_dir=output_dir)
            options = dict(step_solver.options)
            if output_dir:
                options["solver_log_dir"] = os.path.join(
                    output_dir, "solver_logs", f"step_{index:02d}"
                )
            # Objective adapters also need the existing strategy-specific knobs.
            options.update(
                {
                    name: getattr(effective, name)
                    for name in (
                        "mid_transport_bounds",
                        "mid_complementary_slackness",
                        "mid_complementary_slackness_slack",
                    )
                }
            )
            configured_time = float(options["solve_time_limit"])
            received = carry if self.config.carry_over_compute else 0.0
            options["solve_time_limit"] = configured_time + received
            self.add_initial_hint(problem, step_solver.with_options(**options))
            solution = solve_objective(
                problem, step.method, step.objective, options=options
            )
            elapsed = time.perf_counter() - started
            solution.metadata.update(
                pipeline_step=step.name,
                pipeline_step_index=index,
                pipeline_spec=asdict(step),
                pipeline_fixed=fixed,
                pipeline_fixed_areas=base_area_assignment(problem.G, fixed),
                pipeline_out_of_range_areas=base_area_assignment(
                    problem.G, out_of_range
                ),
                pipeline_detached_areas=base_area_assignment(problem.G, islands),
                pipeline_candidates_areas=_base_area_candidates(problem.G, candidates),
                method=step.method,
                objective_kind=step.objective,
                configured_time_limit_seconds=configured_time,
                pipeline_elapsed_seconds=elapsed,
                solver_reported_wall_time_seconds=solution.wall_time,
            )
            if reference is not None:
                solution.metadata["capacity_reference"] = reference
            solution.wall_time = elapsed
            if self.config.carry_over_compute:
                carry = max(0.0, configured_time + received - elapsed)
                solution.metadata.update(
                    carry_over_time_received_seconds=received,
                    unused_time_carried_forward_seconds=carry,
                )
            results.append(solution)
            by_name[step.name] = solution
            # Failed steps remain visible in benchmark output. Their values are
            # never passed to another solve as if they were a valid zoning.
            if not solution.feasible:
                break
        return results
