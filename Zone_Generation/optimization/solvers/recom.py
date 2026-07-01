"""ReCom local-search solver.

The solver uses recombination moves: choose two adjacent zones, sample a random
spanning tree on their union, cut one tree edge, and reassign the two resulting
components to the zones containing their centroids.  Intermediate assignments may
violate balance constraints while the search is moving, but only a fully valid
assignment is returned as feasible.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Mapping

import networkx as nx

from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.balance import balance_constraints
from Zone_Generation.optimization.solvers.base import Solver, register

_EPS = 1e-6


@dataclass(frozen=True, order=True)
class _Score:
    penalty: float
    boundary: int


@register("recom")
class ReComSolver(Solver):
    """Randomized ReCom solver honoring the ``ZoneProblem`` contract."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        if problem.choice_objective is not None:
            raise NotImplementedError(
                "recom does not support iterative choice objectives; use cp_int, cp_bool, or mip."
            )

        start = time.time()
        rng = random.Random(int(self.options.get("seed", 42)))
        time_limit = float(self.options.get("solve_time_limit", 60.0))
        max_iterations = max(0, int(self.options.get("recom_iterations", 1000)))
        cut_attempts = max(1, int(self.options.get("recom_cut_attempts", 100)))
        temperature = max(0.0, float(self.options.get("recom_temperature", 0.0)))

        current = self._initial_assignment(problem)
        current_score = self._score(problem, current)
        initial_score = current_score
        best = dict(current) if _valid(current_score) else None
        best_score = current_score if best is not None else None
        time_to_convergence = 0.0 if best is not None else None
        accepted = 0
        rejected = 0
        attempted = 0

        for iteration in range(max_iterations):
            if time.time() - start >= time_limit:
                break
            proposal = self._proposal(problem, current, rng, cut_attempts)
            attempted += 1
            if proposal is None:
                rejected += 1
                continue

            proposal_score = self._score(problem, proposal)
            if self._accept(current_score, proposal_score, temperature, rng):
                current = proposal
                current_score = proposal_score
                accepted += 1
            else:
                rejected += 1

            if _valid(proposal_score) and (
                best_score is None or proposal_score.boundary < best_score.boundary
            ):
                best = dict(proposal)
                best_score = proposal_score
                if time_to_convergence is None:
                    time_to_convergence = time.time() - start

        wall = time.time() - start
        if best is not None and best_score is not None:
            status = "FEASIBLE"
            assignment = best
            objective = float(best_score.boundary)
            if time_to_convergence is None:
                time_to_convergence = wall
        else:
            status = "UNKNOWN"
            assignment = {}
            objective = None

        metadata = {
            "solver": self.name,
            "iterations": max_iterations,
            "attempted_moves": attempted,
            "accepted_moves": accepted,
            "rejected_moves": rejected,
            "initial_penalty": initial_score.penalty,
            "best_penalty": best_score.penalty if best_score else current_score.penalty,
            "temperature": temperature,
        }
        cache_metadata = getattr(problem, "_recom_initial_cache", None)
        if cache_metadata is not None:
            metadata["initial_cache"] = dict(cache_metadata)

        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status,
            objective=objective,
            wall_time=wall,
            time_to_convergence=time_to_convergence,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Initial assignment
    # ------------------------------------------------------------------ #
    def _initial_assignment(self, problem: ZoneProblem) -> dict[int, int]:
        if problem.hint:
            return self._complete_assignment(problem, problem.hint)
        assignment = self._complete_assignment(problem, {})
        repaired = contiguity.repair(problem.G, assignment, problem.centroids)
        return self._complete_assignment(problem, repaired)

    def _complete_assignment(
        self, problem: ZoneProblem, seed: Mapping[int, int]
    ) -> dict[int, int]:
        assignment: dict[int, int] = {}
        for node in problem.nodes:
            zone = seed.get(node)
            candidates = problem.candidate_zones(node)
            if zone in candidates:
                assignment[node] = int(zone)
            else:
                assignment[node] = min(
                    candidates,
                    key=lambda z: problem.distance(problem.centroids[z], node),
                )
        for z, centroid in enumerate(problem.centroids):
            assignment[centroid] = z
        return assignment

    # ------------------------------------------------------------------ #
    # ReCom moves
    # ------------------------------------------------------------------ #
    def _proposal(
        self,
        problem: ZoneProblem,
        assignment: dict[int, int],
        rng: random.Random,
        cut_attempts: int,
    ) -> dict[int, int] | None:
        pair = self._choose_adjacent_pair(problem.G, assignment, rng)
        if pair is None:
            return None
        z1, z2 = pair
        union_nodes = [n for n, z in assignment.items() if z in pair]
        if (
            problem.centroids[z1] not in union_nodes
            or problem.centroids[z2] not in union_nodes
        ):
            return None
        union = problem.G.subgraph(union_nodes)
        if union.number_of_nodes() < 2 or not nx.is_connected(union):
            return None

        tree = _random_spanning_tree(union, rng)
        edges = list(tree.edges())
        rng.shuffle(edges)
        for edge in edges[:cut_attempts]:
            tree.remove_edge(*edge)
            components = [set(c) for c in nx.connected_components(tree)]
            tree.add_edge(*edge)
            if len(components) != 2:
                continue

            c1, c2 = components
            centroid1 = problem.centroids[z1]
            centroid2 = problem.centroids[z2]
            if centroid1 in c1 and centroid2 in c2:
                return _with_recom_cut(assignment, c1, c2, z1, z2)
            if centroid1 in c2 and centroid2 in c1:
                return _with_recom_cut(assignment, c2, c1, z1, z2)
        return None

    def _choose_adjacent_pair(
        self, G: nx.Graph, assignment: Mapping[int, int], rng: random.Random
    ) -> tuple[int, int] | None:
        weights: dict[tuple[int, int], int] = {}
        for u, v in G.edges():
            zu = assignment.get(u)
            zv = assignment.get(v)
            if zu is None or zv is None or zu == zv:
                continue
            pair = tuple(sorted((int(zu), int(zv))))
            weights[pair] = weights.get(pair, 0) + 1
        if not weights:
            return None

        total = sum(weights.values())
        draw = rng.uniform(0, total)
        upto = 0.0
        for pair, weight in weights.items():
            upto += weight
            if draw <= upto:
                return pair
        return next(iter(weights))

    def _accept(
        self,
        current: _Score,
        proposal: _Score,
        temperature: float,
        rng: random.Random,
    ) -> bool:
        if proposal <= current:
            return True
        if temperature <= 0:
            return False
        delta = (proposal.penalty - current.penalty) + (
            proposal.boundary - current.boundary
        )
        if delta <= 0:
            return True
        return rng.random() < math.exp(-delta / temperature)

    # ------------------------------------------------------------------ #
    # Constraint scoring and validation
    # ------------------------------------------------------------------ #
    def _score(self, problem: ZoneProblem, assignment: Mapping[int, int]) -> _Score:
        penalty = 0.0
        hard_penalty = float(problem.A + problem.Z + 1) * 1000.0

        if set(assignment) != set(problem.nodes):
            missing = set(problem.nodes) - set(assignment)
            extra = set(assignment) - set(problem.nodes)
            penalty += hard_penalty * (len(missing) + len(extra))

        for node in problem.nodes:
            zone = assignment.get(node)
            if zone not in problem.candidate_zones(node):
                penalty += hard_penalty

        for z, centroid in enumerate(problem.centroids):
            if assignment.get(centroid) != z:
                penalty += hard_penalty

        if set(assignment) >= set(problem.nodes) and not contiguity.is_contiguous(
            problem.G, dict(assignment), problem.centroids
        ):
            penalty += hard_penalty

        penalty += self._balance_penalty(problem, assignment)
        penalty += self._school_count_penalty(problem, assignment)
        return _Score(
            penalty=penalty,
            boundary=contiguity.boundary_edges(problem.G, dict(assignment)),
        )

    def _balance_penalty(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> float:
        penalty = 0.0
        for z in range(problem.Z):
            nodes = [n for n in problem.nodes if assignment.get(n) == z]
            students = sum(problem.students(n) for n in nodes)
            for constraint in balance_constraints(problem):
                value = sum(constraint.value(n) for n in nodes)
                lower = constraint.lower_ratio * students
                upper = constraint.upper_ratio * students
                if value < lower:
                    penalty += lower - value
                if value > upper:
                    penalty += value - upper
        return penalty

    def _school_count_penalty(
        self, problem: ZoneProblem, assignment: Mapping[int, int]
    ) -> float:
        total = sum(problem.num_schools(n) for n in problem.nodes)
        if total == 0:
            return 0.0
        avg = total / problem.Z
        lower = max(0.0, avg - 1.0)
        upper = avg + 1.0
        penalty = 0.0
        for z in range(problem.Z):
            schools = sum(
                problem.num_schools(n) for n in problem.nodes if assignment.get(n) == z
            )
            if schools < lower:
                penalty += lower - schools
            if schools > upper:
                penalty += schools - upper
        return penalty


def _with_recom_cut(
    assignment: Mapping[int, int],
    zone1_nodes: set[int],
    zone2_nodes: set[int],
    zone1: int,
    zone2: int,
) -> dict[int, int]:
    proposal = dict(assignment)
    for node in zone1_nodes:
        proposal[node] = zone1
    for node in zone2_nodes:
        proposal[node] = zone2
    return proposal


def _valid(score: _Score) -> bool:
    return score.penalty <= _EPS


def _random_spanning_tree(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Sample a spanning tree with loop-erased random walks."""
    nodes = list(G.nodes())
    root = rng.choice(nodes)
    covered = {root}
    tree = nx.Graph()
    tree.add_nodes_from(nodes)

    for start in nodes:
        if start in covered:
            continue
        path = [start]
        positions = {start: 0}
        current = start
        while current not in covered:
            nxt = rng.choice(list(G.neighbors(current)))
            if nxt in positions:
                cut = positions[nxt]
                path = path[: cut + 1]
                positions = {node: idx for idx, node in enumerate(path)}
            else:
                path.append(nxt)
                positions[nxt] = len(path) - 1
            current = nxt
        for u, v in zip(path, path[1:]):
            tree.add_edge(u, v)
            covered.add(u)
            covered.add(v)
    return tree
