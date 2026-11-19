"""Execute named solves with explicit solution inputs."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
import os
import time

import networkx as nx

from optimization.data import contiguity
from optimization.data.capacity_reference import (
    apply_capacity_reference,
    build_capacity_reference,
)
from optimization.data.boundary_domain import restrict_boundary_domains
from optimization.data.contraction import contract_fixed
from optimization.data.feasibility import check_zoning
from optimization.data.conversion import (
    LevelConverter,
    _node_base_ids,
    base_area_assignment,
)
from optimization.objectives import get_objective, solve_objective
from optimization.levels import LevelSpec
from optimization.pipeline import step_config
from optimization.problem import DuplicateCentroidError
from optimization.solution import ZoneSolution
from optimization.solvers.cpsat import _assignment_objective
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


def relax_unsupported_assignments(problem, fixed, assignment, paths="all"):
    """Loosen hard assignments the CP contiguity rows could never satisfy.

    The MIP/CP solvers require every non-centroid block to have a same-zone
    neighbor strictly closer to its school, which a prior step's zoning (ReCom
    zonings are only connected) need not give.
    :func:`relax_unsupported_candidates` frees each unsupported block and adds
    its zone as a candidate along its closer-neighbor paths, so a fixed block
    on such a path keeps its own zone and gains only the ones it may need.
    Returns ``(fixed, candidates)``: still-forced blocks, and blocks the
    relaxation widened, with their allowed zones. ``paths`` is
    ``release_paths``: open every closer-neighbor path (``"all"``) or one
    shortest path (``"shortest"``).
    """
    if not fixed:
        return fixed, {}
    relaxed = contiguity.relax_unsupported_candidates(
        problem.G,
        assignment,
        list(problem.centroids),
        problem.centroid_school_ids,
        {node: {zone} for node, zone in fixed.items()},
        paths=paths,
    )
    kept, candidates = {}, {}
    for node, zone in fixed.items():
        zones = relaxed.get(node)
        if zones == {zone}:
            kept[node] = zone
        elif zones is not None:
            candidates[node] = zones
    return kept, candidates


def release_unreachable_assignments(problem, fixed, hints, paths="all"):
    """Free hard assignments until every fixed piece reaches its centroid.

    Alternates :func:`detached_fixed_pieces` (pieces cut off from their
    centroid become hints) and :func:`relax_unsupported_assignments` (blocks
    without closer-neighbor support are freed or widened). Widening can cut a
    piece off again, so one pass of each is not enough. ``hints`` is updated
    in place with every released block's zone. Returns ``(fixed, candidates,
    detached)``.
    """
    fixed = dict(fixed)
    detached_all, candidates = {}, {}
    while True:
        detached = detached_fixed_pieces(problem, fixed)
        for node, zone in detached.items():
            del fixed[node]
            hints[node] = zone
        detached_all.update(detached)
        hints.update(fixed)
        before = len(fixed)
        fixed, widened = relax_unsupported_assignments(problem, fixed, hints, paths)
        candidates.update(widened)
        # The relaxation frees unsupported blocks outright as well as widening
        # others, so repeat while either release shrank the fixed set.
        if not detached and len(fixed) == before:
            return fixed, candidates, detached_all


def centroid_neighborhood(G, centroids, zoning, radius):
    """``{node: zone}`` within ``radius`` hops of each zone's centroid.

    The walk only steps through blocks ``zoning`` puts in that zone, so the
    result is connected to the centroid within its zone.
    """
    keep = {}
    for zone, centroid in enumerate(centroids):
        seen, frontier = {centroid}, [centroid]
        for _ in range(radius):
            reached = []
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor not in seen and zoning.get(neighbor) == zone:
                        seen.add(neighbor)
                        reached.append(neighbor)
            frontier = reached
        keep.update({node: zone for node in seen})
    return keep


def _base_area_candidates(G, candidates):
    """``{base_area_id: sorted zones}`` so a saved stage can rebuild them."""
    return {
        area: sorted(zones)
        for node, zones in candidates.items()
        for area in _node_base_ids(G, node)
    }


@register("pipeline")
class PipelineStrategy(Strategy):
    """Run ``config.steps`` in order; every run, single or multi-level, is one.

    ``strategy: single`` and ``strategy: recursive`` are shorthand for steps
    (``optimization.pipeline.shorthand_steps``). Each step's solver reports its
    own ``wall_time``, and that solver time is what ``carry_over_compute``
    hands the next step: problem building and input projection are free.
    """

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

    def problem_for_stage(
        self, dataset, metadata, centroid_school_ids=None, stage_index=None
    ):
        """Reconstruct the feasible set of a saved benchmark stage.

        A stage names its step in ``pipeline_step_index``; one saved without it
        is taken to be step ``stage_index``, its position in the run.
        """
        index = metadata.get("pipeline_step_index", stage_index)
        if index is None:
            raise ValueError("Saved stage does not name its pipeline step.")
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
        extra = {}
        if centroid_school_ids is not None:
            extra["centroid_school_ids"] = centroid_school_ids
        problem = dataset.problem_for(
            step.level,
            config=step_config(self.config, step, index),
            fixed=fixed or None,
            candidates=candidates or None,
            **extra,
        )
        if step.capacity_reference is not None:
            if "capacity_reference" not in metadata:
                raise ValueError(
                    "Saved stage is missing its frozen capacity reference."
                )
            apply_capacity_reference(problem, metadata["capacity_reference"])
        return problem

    def run(self, dataset, solver):
        self._converter = LevelConverter(data=dataset.data)
        self._projected = {}
        self._capacity_references = {}
        results = []
        by_name = {}
        carry = 0.0
        output_dir = solver.options.get("output_dir")
        final = len(self.config.steps) - 1
        for index, step in enumerate(self.config.steps):
            started = time.perf_counter()
            effective = step_config(self.config, step, index)
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
            problem = dataset.problem_for(step.level, config=effective)
            extra = {}
            try:
                stages, extra = self._solve_step(
                    dataset,
                    problem,
                    step,
                    effective,
                    step_solver.with_options(**options),
                    results,
                    by_name,
                )
            except DuplicateCentroidError as exc:
                # A coarse level can merge two schools into one node. Only the
                # finest step has to place every centroid on its own node.
                if index == final:
                    raise
                stages = [_skipped_duplicate_centroid(problem, exc)]
            selected = stages[-1]
            used = 0.0 if selected.wall_time is None else float(selected.wall_time)
            metadata = dict(
                pipeline_step=step.name,
                pipeline_step_index=index,
                pipeline_spec=asdict(step),
                method=step.method,
                objective_kind=step.objective,
                configured_time_limit_seconds=configured_time,
                pipeline_elapsed_seconds=time.perf_counter() - started,
                **extra,
            )
            if self.config.carry_over_compute:
                carry = max(0.0, configured_time + received - used)
                metadata.update(
                    carry_over_time_received_seconds=received,
                    effective_time_limit_seconds=configured_time + received,
                    unused_time_carried_forward_seconds=carry,
                )
            for stage in stages:
                stage.metadata.update(metadata)
            results.extend(stages)
            if selected.status == "SKIPPED":
                continue
            by_name[step.name] = selected
            # Failed steps remain visible in benchmark output. Their values are
            # never passed to another solve as if they were a valid zoning.
            if not selected.feasible:
                break
        return results

    def _source(self, name, results, by_name):
        """The solution a step input names, or ``None``.

        ``previous`` is the latest step that was not skipped; with every
        earlier step skipped there is none, and the step starts from scratch.
        """
        if name != "previous":
            return by_name[name]
        return next((s for s in reversed(results) if s.status != "SKIPPED"), None)

    def _capacity_reference(self, dataset, step, results, by_name):
        source = self._source(step.capacity_reference, results, by_name)
        if source is None:
            raise ValueError("A capacity reference needs a complete feasible source.")
        source_name = source.metadata["pipeline_step"]
        if source_name not in self._capacity_references:
            if not source.feasible or set(source.assignment) != set(source.problem.G):
                raise ValueError(
                    "A capacity reference needs a complete feasible source."
                )
            source_config = source.problem.optimization_config
            base = dataset.problem_for(
                LevelSpec(source.level.unit, 0), config=source_config
            )
            zoning = self._converter.between(
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
            self._capacity_references[source_name] = reference
        return self._capacity_references[source_name]

    def _solve_step(
        self, dataset, problem, step, effective, step_solver, results, by_name
    ):
        """Apply the step's inputs to ``problem`` and solve it.

        Returns the step's solutions and the metadata describing its
        feasible set.
        """
        extra = {}
        if step.capacity_reference is not None:
            reference = self._capacity_reference(dataset, step, results, by_name)
            apply_capacity_reference(problem, reference)
            extra["capacity_reference"] = reference
        fixed, hints = {}, dict(problem.hint or {})
        assigned = {}  # every block's zone in the hard-assignment sources
        for binding in step.inputs:
            source = self._source(binding.source, results, by_name)
            if source is None:
                continue
            if not source.feasible or set(source.assignment) != set(source.problem.G):
                raise ValueError(
                    f"Step {step.name!r} needs a complete feasible input from {binding.source!r}."
                )
            key = (id(source), step.level)
            if key not in self._projected:
                self._projected[key] = self._converter.between(
                    source.problem.G,
                    source.assignment,
                    source.level,
                    problem.G,
                    problem.level,
                )
            projected = self._projected[key]
            if binding.mode == "assign":
                assigned.update(projected)
            for node in selected_nodes(problem.G, projected, binding):
                zone = projected[node]
                if zone not in range(problem.Z):
                    raise ValueError(
                        f"Input supplies invalid zone {zone} for node {node}."
                    )
                target = fixed if binding.mode == "assign" else hints
                if binding.mode == "assign" and node in fixed and fixed[node] != zone:
                    raise ValueError(
                        f"Conflicting hard assignments for node {node} in step {step.name!r}."
                    )
                target[node] = zone
        for zone, centroid in enumerate(problem.centroids):
            if centroid in fixed and fixed[centroid] != zone:
                raise ValueError(
                    f"Hard assignment conflicts with centroid {centroid} in step {step.name!r}."
                )
        # A school on its zone's boundary sits in the released band, which
        # would cut the zone's fixed interior off from it; keep the blocks
        # near each centroid that the source puts in its zone.
        radius = effective.keep_centroid_radius
        if radius is None:
            # Default: as far as the released band reaches; a step without one
            # (explicit or whole-zoning hard inputs) releases no band to keep.
            radius = max(
                (
                    b.radius
                    for b in step.inputs
                    if b.mode == "assign" and b.select == "interior"
                ),
                default=0,
            )
        if radius > 0 and assigned:
            kept = centroid_neighborhood(problem.G, problem.centroids, assigned, radius)
            for node, zone in kept.items():
                fixed.setdefault(node, zone)
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
        fixed, candidates, islands = release_unreachable_assignments(
            problem, fixed, hints, effective.release_paths
        )
        problem.fixed = fixed or None
        problem.candidates = candidates or None
        if hints:
            # Hint the fixed interior as well, so the hint is the whole source
            # zoning: CP-SAT takes a complete hint as its first solution, but
            # a partial one (only the free boundary) must be completed by
            # search first, which a single worker often fails to do.
            hints.update(fixed)
            hints.update(
                {centroid: zone for zone, centroid in enumerate(problem.centroids)}
            )
        problem.hint = hints or None
        extra.update(
            restrict_step_domains(
                problem,
                hints or assigned,
                step.method,
                effective.boundary_domain_hops,
                step_solver.options,
                effective.release_paths,
            )
        )
        extra.update(
            pipeline_fixed=fixed,
            pipeline_fixed_areas=base_area_assignment(problem.G, fixed),
            pipeline_out_of_range_areas=base_area_assignment(problem.G, out_of_range),
            pipeline_detached_areas=base_area_assignment(problem.G, islands),
            pipeline_candidates_areas=_base_area_candidates(
                problem.G, problem.candidates or {}
            ),
        )
        self.add_initial_hint(problem, step_solver)
        solution = solve_step_problem(
            problem,
            step.method,
            step.objective,
            step_solver.options,
            contract=effective.contract_fixed,
        )
        return [solution], extra


#: Methods whose models take explicit candidate domains and write contiguity
#: as closer-neighbour support rows (MIP unless ``contiguity_model: flow``).
BOUNDARY_DOMAIN_METHODS = frozenset({"cp_bool", "cp_int", "mip"})


def restrict_step_domains(problem, source, method, hops, options, paths):
    """Apply ``boundary_domain_hops`` to a refinement step's free nodes.

    Runs on the original problem after the step's split is final -- the
    out-of-range release and :func:`release_unreachable_assignments` have set
    ``problem.fixed`` and the relaxation's widened ``problem.candidates`` --
    and before the initial hint and :func:`solve_step_problem`'s contraction,
    which therefore see the restricted candidates (a free node pruned to one
    zone contracts with its fixed neighbours). ``source`` is the zoning the
    step starts from, the one its hints come from. Returns metadata.
    """
    if hops is None or not problem.fixed:
        return {}
    if method not in BOUNDARY_DOMAIN_METHODS:
        return {"boundary_domain_hops": hops, "boundary_domain_skipped": "method"}
    prune = not (method == "mip" and options.get("contiguity_model") == "flow")
    source = {n: z for n, z in source.items() if n in problem.G}
    return restrict_boundary_domains(problem, source, hops, prune=prune, paths=paths)


#: Methods whose solve contracts fixed groups: the CP-SAT models.
CONTRACTED_METHODS = frozenset({"cp_bool", "cp_int"})
#: Objectives the contraction preserves: boundary and distance costs. Market
#: objectives index their markets by node and are solved unreduced.
CONTRACTED_OBJECTIVES = frozenset(
    {"cut_edges", "boundary_perimeter", "school_distance", "average_distance"}
)


def contraction_skip_reason(problem, method, objective, options) -> str | None:
    """Why a solve cannot use :func:`contract_fixed`, or ``None`` if it can."""
    if not problem.fixed:
        return "no_fixed_nodes"
    if method not in CONTRACTED_METHODS:
        return "method"
    if objective not in CONTRACTED_OBJECTIVES or problem.choice_objective is not None:
        return "objective"
    if problem.boundary_prop >= 0:
        # The cap counts edges, and contraction merges parallel ones.
        return "boundary_prop"
    if options.get("centroid_neighbor_radius", 0):
        # Hop distances from a centroid change when groups contract.
        return "centroid_neighbor_radius"
    return None


def solve_step_problem(problem, method, objective, options, *, contract=True):
    """Solve one step, on the contracted problem when that is possible.

    The reduced solution is expanded to every original node and checked with
    ``check_zoning`` on the original problem; on failure the original problem
    is solved instead (``contraction_fallback`` in the metadata). The
    objective is recomputed on the original problem, so a distance objective
    includes the fixed groups' constant cost.
    """
    reason = contraction_skip_reason(problem, method, objective, options)
    if not contract or reason is not None:
        solution = solve_objective(problem, method, objective, options=options)
        if contract and reason != "no_fixed_nodes":
            solution.metadata["contraction_skipped"] = reason
        return solution
    started = time.perf_counter()
    weighted = problem.weight_edges or objective == "boundary_perimeter"
    contraction = contract_fixed(problem, weighted=weighted)
    contract_seconds = time.perf_counter() - started
    if contraction is None:
        solution = solve_objective(problem, method, objective, options=options)
        solution.metadata["contraction_skipped"] = "no_groups"
        return solution
    reduced = solve_objective(contraction.reduced, method, objective, options=options)
    started = time.perf_counter()
    view = replace(problem, weight_edges=weighted, objective=objective)
    assignment = contraction.expand(reduced.assignment) if reduced.feasible else {}
    report = check_zoning(view, assignment) if reduced.feasible else None
    expand_seconds = time.perf_counter() - started
    metadata = {
        **contraction.metadata(),
        "contraction_seconds": contract_seconds,
        "contraction_expand_seconds": expand_seconds,
    }
    if report is not None and not report.feasible:
        solution = solve_objective(problem, method, objective, options=options)
        solution.metadata.update(
            metadata,
            contraction_fallback=True,
            contraction_reduced_wall_time=reduced.wall_time,
            contraction_fallback_reason=report.describe(),
        )
        return solution
    metadata = {**reduced.metadata, **metadata}
    if not weighted and metadata.get("objective_kind") == "weighted_boundary_length":
        # The reduced edges count member edges; the solve itself is unweighted.
        metadata.pop("objective_kind")
        metadata.pop("objective_unit", None)
    objective_value = reduced.objective
    progress = list(reduced.solver_progress)
    if reduced.feasible and reduced.objective is not None:
        objective_value = _assignment_objective(view, assignment)
        offset = objective_value - reduced.objective
        progress = [
            replace(
                entry,
                objective=entry.objective + offset,
                assignment=contraction.expand_tuple(entry.assignment),
            )
            for entry in progress
        ]
        metadata["contraction_objective_offset"] = offset
    return ZoneSolution(
        problem=view,
        assignment=assignment,
        status=reduced.status,
        objective=objective_value,
        wall_time=reduced.wall_time,
        metadata=metadata,
        solver_progress=progress,
    )


def _skipped_duplicate_centroid(problem, exc: DuplicateCentroidError) -> ZoneSolution:
    return ZoneSolution(
        problem=problem,
        assignment={},
        status="SKIPPED",
        objective=None,
        wall_time=0.0,
        metadata={
            "skip_reason": "duplicate_centroid",
            "error_message": str(exc),
            "duplicate_centroid_node": exc.node,
            "duplicate_centroid_zones": list(exc.zones),
        },
    )
