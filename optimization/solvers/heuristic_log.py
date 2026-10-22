"""Improvement-only progress logs for the heuristic (ReCom-family) solvers.

The math-programming backends already emit an incumbent-and-bound trajectory in
their native solver logs, so ``save_solver_logs`` just points Gurobi and CP-SAT
at a file.  The ReCom family has no such log, so the same flag routes it here:
one JSON object per line, a header followed by one record per strictly
improving state and a closing summary.

A heuristic has no bound, so the useful trajectory is the objective together
with the constraint residuals it is trading against.  Every record therefore
carries the boundary cost, each individual constraint penalty, and both
Lagrangians, which is enough to reconstruct any of the scalarizations the
solvers use internally without keeping a single assignment on disk.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

# Only used to compare two Lagrangian values; feasibility is decided by the
# caller so that this module never disagrees with ``_Snapshot.feasible``.
_EPS = 1e-9


class HeuristicProgressLog:
    """Append-only JSONL log of strictly improving heuristic states.

    States are ordered by ``(not feasible, objective)``: any feasible state
    beats any infeasible one, feasible states are ranked by boundary cost, and
    infeasible states by the unweighted Lagrangian.  Both components are
    independent of wall-clock time and of the running-max normalizer that
    drives burst selection, so the logged sequence is monotone by construction
    -- a record is written only when the search has genuinely beaten every
    state it has visited so far.
    """

    def __init__(
        self,
        path: str,
        *,
        solver: str,
        level: str,
        labels: Sequence[str],
        start: float,
        header: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = path
        self.labels = tuple(labels)
        self._start = float(start)
        self._count = 0
        self._best: tuple[int, float] | None = None
        self._file = open(path, "w", encoding="utf-8")
        self._write(
            {
                "record": "header",
                "solver": solver,
                "level": level,
                "violation_labels": list(self.labels),
                "objective": "boundary_cost",
                "definitions": {
                    "total_violation": "sum(penalties)",
                    "squared_violation": "sum(penalty**2)",
                    "unweighted_lagrangian": "boundary_cost + sum(penalty**2)",
                    "weighted_lagrangian": ("boundary_cost + sum(weight * penalty**2)"),
                    "improvement_key": (
                        "(not feasible, boundary_cost if feasible "
                        "else unweighted_lagrangian), minimized"
                    ),
                },
                **(dict(header) if header else {}),
            }
        )

    # ------------------------------------------------------------------ #
    # recording
    # ------------------------------------------------------------------ #
    def record(
        self,
        *,
        feasible: bool,
        boundary_cost: float,
        violations: Sequence[float],
        weights: Sequence[float] | None = None,
        iteration: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> bool:
        """Log ``violations`` if they beat every state seen so far.

        Returns whether a record was written.  The improvement test runs before
        the clock is read, so a non-improving state costs a handful of
        floating-point operations and no I/O.
        """

        if len(violations) != len(self.labels):
            raise ValueError(
                "Violation vector length does not match the constraint labels."
            )

        squared = sum(float(value) * float(value) for value in violations)
        unweighted = float(boundary_cost) + squared
        key = (0 if feasible else 1, float(boundary_cost) if feasible else unweighted)
        if not self._is_improvement(key):
            return False
        self._best = key

        row: dict[str, Any] = {
            "record": "improvement",
            "index": self._count,
            "elapsed_seconds": time.monotonic() - self._start,
            "feasible": bool(feasible),
            "boundary_cost": float(boundary_cost),
            "total_violation": float(sum(float(value) for value in violations)),
            "squared_violation": squared,
            "unweighted_lagrangian": unweighted,
            "penalties": {
                label: float(value)
                for label, value in zip(self.labels, violations, strict=True)
            },
        }
        if iteration is not None:
            row["iteration"] = int(iteration)
        if weights is not None:
            if len(weights) != len(self.labels):
                raise ValueError(
                    "Weight vector length does not match the constraint labels."
                )
            row["weighted_lagrangian"] = float(boundary_cost) + sum(
                float(weight) * float(value) * float(value)
                for weight, value in zip(weights, violations, strict=True)
            )
            row["weights"] = {
                label: float(weight)
                for label, weight in zip(self.labels, weights, strict=True)
            }
        if extra:
            row.update(dict(extra))

        self._write(row)
        self._count += 1
        return True

    def finish(self, **fields: Any) -> None:
        """Write the closing summary record and release the file."""

        if self._file.closed:
            return
        self._write(
            {
                "record": "summary",
                "elapsed_seconds": time.monotonic() - self._start,
                "improvements": self._count,
                **fields,
            }
        )
        self._file.close()

    @property
    def improvements(self) -> int:
        return self._count

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _is_improvement(self, key: tuple[int, float]) -> bool:
        if self._best is None:
            return True
        rank, value = key
        best_rank, best_value = self._best
        if rank != best_rank:
            return rank < best_rank
        return value < best_value - _EPS

    def _write(self, row: Mapping[str, Any]) -> None:
        json.dump(row, self._file, sort_keys=True)
        self._file.write("\n")
        self._file.flush()
