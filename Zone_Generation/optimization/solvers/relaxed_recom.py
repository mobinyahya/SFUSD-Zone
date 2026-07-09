"""Relaxed ReCom solver variant."""

from __future__ import annotations

import math
import random
import time
from typing import Mapping

import networkx as nx

from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.base import register
from Zone_Generation.optimization.solvers.recom import ReComSolver, _valid
from Zone_Generation.optimization.data.initial_solutions import (
    normalize_recom_balance_metric,
)


class _RelaxedReComMoveError(RuntimeError):
    """Raised when the relaxed ReCom walk cannot produce a valid tree move."""


@register("relaxed_recom")
class RelaxedReComSolver(ReComSolver):
    """Relaxed ReCom walk with post-hoc rejection sampling of valid plans."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        if problem.choice_objective is not None:
            raise NotImplementedError(
                "relaxed_recom does not support iterative choice objectives; "
                "use cp_int, cp_bool, or mip."
            )

        start = time.time()
        hint_error = self._hints_error_solution(problem, start)
        if hint_error is not None:
            return hint_error
        seed = int(self.options.get("seed", 42))
        rng = random.Random(seed)
        time_limit, max_iterations = self._recom_limits()
        cut_attempts = max(1, int(self.options.get("recom_cut_attempts", 100)))
        population_epsilon = self.options.get("recom_population_epsilon")
        balance_metric = normalize_recom_balance_metric(
            self.options.get("recom_balance_metric", "students")
        )
        min_boundary_edges = int(
            self.options.get("relaxed_recom_min_boundary_edges", 0)
        )
        log_path, progress_log = self._open_progress_log(problem)
        progress = self._new_recom_progress_tracker(problem)

        random_state = random.getstate()
        random.seed(seed)
        try:
            try:
                initial = self._initial_state_with_options(
                    problem,
                    cut_attempts,
                    population_epsilon=population_epsilon,
                    balance_metric=balance_metric,
                )
                current = self._prepare_relaxed_assignment(problem, initial.assignment)
                current_score = self._score(problem, current)
                initial_score = current_score
                trees = self._zone_spanning_trees(problem, current, rng)

                best = dict(current) if _valid(current_score) else None
                best_score = current_score if best is not None else None
                best_infeasible = None if best is not None else dict(current)
                best_infeasible_score = None if best is not None else current_score
                attempted = 0
                accepted = 0
                rejected_samples = 0 if best is not None else 1
                proposal_failures = 0
                last_proposal_error = None

                self._write_progress_log(
                    progress_log,
                    start=start,
                    event="initial",
                    iteration=0,
                    score=current_score,
                    best_score=best_score,
                )
                self._record_recom_progress(
                    progress,
                    start,
                    problem,
                    current,
                    current_score,
                    iteration=0,
                )

                while self._has_recom_iteration_budget(max_iterations, attempted):
                    if time.time() - start >= time_limit:
                        break
                    attempted += 1
                    try:
                        self._relaxed_recom_step(
                            problem,
                            current,
                            trees,
                            rng,
                            min_boundary_edges=min_boundary_edges,
                        )
                    except _RelaxedReComMoveError as exc:
                        proposal_failures += 1
                        last_proposal_error = str(exc)
                        continue

                    accepted += 1
                    current_score = self._score(problem, current)
                    if not _valid(current_score) and (
                        best_infeasible_score is None
                        or current_score < best_infeasible_score
                    ):
                        best_infeasible = dict(current)
                        best_infeasible_score = current_score
                    if _valid(current_score):
                        if (
                            best_score is None
                            or current_score.boundary < best_score.boundary
                        ):
                            best = dict(current)
                            best_score = current_score
                            self._record_recom_progress(
                                progress,
                                start,
                                problem,
                                current,
                                current_score,
                                iteration=attempted,
                            )
                    else:
                        rejected_samples += 1

                    self._write_progress_log(
                        progress_log,
                        start=start,
                        event="cut",
                        iteration=attempted,
                        score=current_score,
                        accepted=True,
                        best_score=best_score,
                    )
            finally:
                if progress_log is not None:
                    progress_log.close()
        finally:
            random.setstate(random_state)

        wall = time.time() - start
        repair_metadata = {}
        if best is None and best_infeasible is not None and best_infeasible_score:
            repaired, repaired_score, repair_metadata = self._repair_infeasible_solution(
                problem, best_infeasible, best_infeasible_score
            )
            if _valid(repaired_score):
                best = repaired
                best_score = repaired_score
                self._record_recom_progress(
                    progress,
                    start,
                    problem,
                    repaired,
                    repaired_score,
                    iteration=attempted,
                )
        if best is not None and best_score is not None:
            status = "FEASIBLE"
            assignment = best
            objective = float(best_score.boundary)
        else:
            status = "UNKNOWN"
            assignment = {}
            objective = None
        best_penalty = (
            best_score.penalty
            if best_score is not None
            else best_infeasible_score.penalty
            if best_infeasible_score is not None
            else current_score.penalty
        )

        metadata = {
            "solver": self.name,
            **self._progress_log_metadata(log_path),
            **self._solver_progress_metadata(progress),
            "hints": initial.metadata.get("hints", self._hints()),
            "iterations": max_iterations,
            "recom_balance_metric": balance_metric,
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "rejected_moves": proposal_failures,
            "rejected_samples": rejected_samples,
            "proposal_failures": proposal_failures,
            "initial_penalty": initial_score.penalty,
            "best_penalty": best_penalty,
            "relaxed_recom_min_boundary_edges": min_boundary_edges,
            "relaxed_recom_cut_weight": "log_inverse_configured_constraint_penalty",
            **repair_metadata,
            **initial.metadata,
        }
        if last_proposal_error is not None:
            metadata["last_proposal_error"] = last_proposal_error

        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status,
            objective=objective,
            wall_time=wall,
            metadata=metadata,
            solver_progress=list(progress.entries) if progress is not None else [],
        )

    def _prepare_relaxed_assignment(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> dict[int, int]:
        current = self._complete_assignment(problem, assignment)
        if contiguity.is_contiguous(problem.G, current, problem.centroids):
            return current
        repaired = contiguity.repair(problem.G, current, problem.centroids)
        # Candidate enforcement can undo connector nodes inserted by repair.
        # Keep the contiguous state and let scoring carry candidate violations.
        return repaired

    def _zone_spanning_trees(
        self,
        problem: ZoneProblem,
        assignment: Mapping[int, int],
        rng: random.Random,
    ) -> dict[int, nx.Graph]:
        trees: dict[int, nx.Graph] = {}
        for z in range(problem.Z):
            nodes = {node for node, zone in assignment.items() if zone == z}
            zone_graph = problem.G.subgraph(nodes).copy()
            if zone_graph.number_of_nodes() == 0:
                raise _RelaxedReComMoveError(f"zone {z} has no nodes")
            if not nx.is_connected(zone_graph):
                raise _RelaxedReComMoveError(f"zone {z} is not contiguous")
            trees[z] = self._random_spanning_tree(zone_graph, rng)
        return trees

    def _random_spanning_tree(self, graph: nx.Graph, rng: random.Random) -> nx.Graph:
        tree = nx.Graph()
        tree.add_nodes_from(graph.nodes(data=True))
        if graph.number_of_nodes() <= 1:
            return tree

        weighted = nx.Graph()
        weighted.add_nodes_from(graph.nodes(data=True))
        for u, v, attrs in graph.edges(data=True):
            weighted.add_edge(u, v, **attrs, _relaxed_recom_weight=rng.random())
        return nx.minimum_spanning_tree(
            weighted,
            weight="_relaxed_recom_weight",
            algorithm="kruskal",
        )

    def _relaxed_recom_step(
        self,
        problem: ZoneProblem,
        assignment: dict[int, int],
        trees: dict[int, nx.Graph],
        rng: random.Random,
        *,
        min_boundary_edges: int,
    ) -> None:
        (zone_a, zone_b), connecting_edge = self._relaxed_adjacent_zones(
            problem, assignment, rng, min_boundary_edges
        )
        merged = nx.Graph()
        merged.add_nodes_from(trees[zone_a].nodes(data=True))
        merged.add_nodes_from(trees[zone_b].nodes(data=True))
        merged.add_edges_from(trees[zone_a].edges(data=True))
        merged.add_edges_from(trees[zone_b].edges(data=True))
        merged.add_edge(*connecting_edge)

        nodes_a, nodes_b = self._sample_relaxed_cut(
            problem,
            merged,
            zone_a,
            zone_b,
            rng,
        )

        for node in nodes_a:
            assignment[node] = zone_a
        for node in nodes_b:
            assignment[node] = zone_b
        trees[zone_a] = merged.subgraph(nodes_a).copy()
        trees[zone_b] = merged.subgraph(nodes_b).copy()

    def _relaxed_adjacent_zones(
        self,
        problem: ZoneProblem,
        assignment: Mapping[int, int],
        rng: random.Random,
        min_boundary_edges: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        boundary_by_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for u, v in problem.G.edges():
            zone_u = assignment.get(u)
            zone_v = assignment.get(v)
            if zone_u is None or zone_v is None or zone_u == zone_v:
                continue
            pair = tuple(sorted((int(zone_u), int(zone_v))))
            boundary_by_pair.setdefault(pair, []).append((int(u), int(v)))

        eligible = [
            (pair, edges)
            for pair, edges in boundary_by_pair.items()
            if len(edges) > min_boundary_edges
        ]
        if not eligible:
            raise _RelaxedReComMoveError(
                "no adjacent zone pair exceeds relaxed_recom_min_boundary_edges"
            )

        pair, edges = rng.choice(eligible)
        return pair, edges[0]

    def _sample_relaxed_cut(
        self,
        problem: ZoneProblem,
        merged: nx.Graph,
        zone_a: int,
        zone_b: int,
        rng: random.Random,
    ) -> tuple[set[int], set[int]]:
        choices: list[tuple[tuple[int, int], set[int], set[int], float]] = []
        for edge in merged.edges():
            test_tree = merged.copy()
            test_tree.remove_edge(*edge)
            components = [
                set(component) for component in nx.connected_components(test_tree)
            ]
            if len(components) != 2:
                continue
            first, second = components
            if rng.random() < 0.5:
                nodes_a, nodes_b = first, second
            else:
                nodes_a, nodes_b = second, first
            choices.append(
                (
                    tuple(edge),
                    nodes_a,
                    nodes_b,
                    self._relaxed_cut_log_weight(problem, nodes_a, nodes_b),
                )
            )

        if not choices:
            raise _RelaxedReComMoveError("merged tree has no two-component cut")

        finite_logs = [
            log_weight for *_, log_weight in choices if math.isfinite(log_weight)
        ]
        if not finite_logs:
            _, nodes_a, nodes_b, _ = rng.choice(choices)
            return nodes_a, nodes_b

        max_log = max(finite_logs)
        weights = [
            math.exp(log_weight - max_log) if math.isfinite(log_weight) else 0.0
            for *_, log_weight in choices
        ]
        if sum(weights) <= 0:
            _, nodes_a, nodes_b, _ = rng.choice(choices)
            return nodes_a, nodes_b
        _, nodes_a, nodes_b, _ = rng.choices(choices, weights=weights, k=1)[0]
        return nodes_a, nodes_b

    def _relaxed_cut_log_weight(
        self,
        problem: ZoneProblem,
        nodes_a: set[int],
        nodes_b: set[int],
    ) -> float:
        context = self._penalty_context(problem)
        penalty = self._zone_constraint_penalty(
            problem, nodes_a, context
        ) + self._zone_constraint_penalty(problem, nodes_b, context)
        if not math.isfinite(penalty):
            return float("-inf")
        return math.log(1.0 / (1.0 + penalty))
