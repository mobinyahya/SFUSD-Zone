"""Local-search solver -- interface stub.

Conforms to the :class:`Solver` interface so it is selectable today, but does
not yet perform real search. It seeds from the problem's ``hint`` (or a nearest
-centroid assignment) and runs one contiguity-repair pass. Greedy boundary
swaps / simulated annealing can later be implemented behind this same
interface without touching the rest of the optimization.
"""

from __future__ import annotations

import time

from optimization.data import contiguity
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver, register


@register("local_search")
class LocalSearchSolver(Solver):
    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        if problem.choice_objective is not None:
            raise NotImplementedError(
                "local_search does not support choice objective cuts; use cp_int, cp_bool, or mip."
            )

        start = time.time()

        if problem.hint:
            assignment = dict(problem.hint)
        else:
            # Nearest-candidate-centroid seed.
            assignment = {}
            for node in problem.nodes:
                cands = problem.candidate_zones(node)
                if not cands:
                    raise problem.no_candidate_zones_error(node)
                assignment[node] = min(
                    cands, key=lambda z: problem.distance(problem.centroids[z], node)
                )

        assignment = contiguity.repair(problem.G, assignment, problem.centroids)

        wall = time.time() - start
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status="STUB",
            objective=float(contiguity.boundary_edges(problem.G, assignment)),
            wall_time=wall,
            metadata={"solver": self.name, "note": "stub implementation"},
        )
