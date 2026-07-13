"""Overlapping strategy: solve one candidate zone per school, then reconcile."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import networkx as nx

from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers import get_solver
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register


@register("overlapping")
class OverlappingStrategy(Strategy):
    """Use independent school zones to fix unambiguous interior assignments."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        target = LevelSpec.parse(self.options["levels"][-1])
        school_ids = dataset.school_ids_for(target)
        if not school_ids:
            raise ValueError(f"No eligible schools found at {target.name}.")

        centroids = dataset.centroids_for(target, school_ids)
        self._validate_school_centroids(
            dataset.graph_for(target), school_ids, centroids
        )
        school_problems = [
            dataset.problem_for(target, centroid_school_ids=[school_id])
            for school_id in school_ids
        ]

        worker_budget = max(1, int(solver.options.get("workers", 1)))
        solver.options["workers"] = worker_budget
        school_time_limit = float(self.options["school_solve_time_limit"])
        started = time.perf_counter()
        school_solutions = self._solve_schools(
            school_ids,
            school_problems,
            solver,
            worker_budget,
            school_time_limit,
        )
        school_phase_wall_time = time.perf_counter() - started

        fixed, boundary_band, membership_counts = self._fixed_assignments(
            dataset.graph_for(target),
            school_solutions,
            radius=self.options.get("boundary_radius", 1),
        )
        final_problem = dataset.problem_for(
            target,
            fixed=fixed,
            centroid_school_ids=school_ids,
        )
        final_solution = solver.solve(final_problem)
        final_solution.metadata.update(
            {
                "centroid_school_ids": school_ids,
                "school_solve_count": len(school_solutions),
                "school_solve_feasible_count": sum(
                    solution.feasible for solution in school_solutions
                ),
                "school_solve_time_limit_seconds": school_time_limit,
                "school_solve_parallelism": min(worker_budget, len(school_ids)),
                "school_solve_phase_wall_time_seconds": school_phase_wall_time,
                "fixed_node_count": len(fixed),
                "boundary_band_node_count": len(boundary_band),
                "unassigned_node_count": sum(
                    count == 0 for count in membership_counts.values()
                ),
                "overlapping_node_count": sum(
                    count > 1 for count in membership_counts.values()
                ),
            }
        )
        return [*school_solutions, final_solution]

    @staticmethod
    def _validate_school_centroids(G, school_ids, centroids) -> None:
        schools_by_node: dict[int, list[int]] = defaultdict(list)
        for school_id, centroid in zip(school_ids, centroids):
            schools_by_node[centroid].append(school_id)
        colocated = {node: ids for node, ids in schools_by_node.items() if len(ids) > 1}
        if colocated:
            details = ", ".join(
                f"node {node}: {ids}" for node, ids in sorted(colocated.items())
            )
            raise ValueError(
                "Overlapping strategy requires one school per centroid node; "
                "choose a graph representation with unique school nodes. "
                f"Colocated schools: {details}."
            )

        for school_id, centroid in zip(school_ids, centroids):
            node_school_ids = [
                int(sid) for sid in G.nodes[centroid].get("school_ids", [])
            ]
            school_count = int(G.nodes[centroid].get("num_schools", 0))
            if node_school_ids != [school_id] or school_count != 1:
                raise ValueError(
                    "Overlapping strategy requires each centroid node to contain "
                    f"exactly its target school; school {school_id} resolved to node "
                    f"{centroid} with school_ids={node_school_ids} and "
                    f"num_schools={school_count}. Choose a graph representation "
                    "with unique school nodes."
                )

    def _solve_schools(
        self,
        school_ids,
        school_problems,
        full_solver,
        worker_budget,
        school_time_limit,
    ) -> list[ZoneSolution]:
        def solve_one(item) -> ZoneSolution:
            zone, (school_id, problem) = item
            options = dict(full_solver.options)
            options.update(
                {
                    "workers": 1,
                    "solve_time_limit": school_time_limit,
                    "save_solver_progress": False,
                }
            )
            output_dir = options.get("output_dir")
            if output_dir:
                options["solver_log_dir"] = os.path.join(
                    str(output_dir), "solver_logs", "school_solves", str(school_id)
                )
            solution = get_solver("cp_single_zone", **options).solve(problem)
            solution.metadata.update(
                {
                    "overlapping_zone_id": zone,
                    "school_solve_workers": 1,
                    "school_solve_time_limit_seconds": school_time_limit,
                }
            )
            return solution

        items = list(enumerate(zip(school_ids, school_problems)))
        parallelism = min(worker_budget, len(items))
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            return list(executor.map(solve_one, items))

    @staticmethod
    def _fixed_assignments(G, school_solutions, radius):
        memberships: dict[int, list[int]] = defaultdict(list)
        boundary_sources: set[int] = set()
        for zone, solution in enumerate(school_solutions):
            if not solution.feasible:
                continue
            selected = set(solution.assignment)
            for node in selected:
                memberships[node].append(zone)
            for u, v in G.edges():
                if (u in selected) != (v in selected):
                    boundary_sources.update((u, v))

        boundary_band: set[int] = set()
        radius = int(radius)
        if radius != -1:
            cutoff = max(0, radius)
            for source in boundary_sources:
                boundary_band.update(
                    nx.single_source_shortest_path_length(G, source, cutoff=cutoff)
                )

        membership_counts = {node: len(memberships.get(node, ())) for node in G.nodes()}
        fixed = {
            node: zones[0]
            for node, zones in memberships.items()
            if len(zones) == 1 and node not in boundary_band
        }
        return fixed, boundary_band, membership_counts
