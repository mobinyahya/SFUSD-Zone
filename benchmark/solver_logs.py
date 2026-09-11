"""Parse solver logs into one tidy objective-and-bound trajectory table.

``save_solver_logs: true`` writes a per-solve log in whatever format the
backend speaks: Gurobi's node table, CP-SAT's search log, or the
improvement-only JSONL that the ReCom family emits
(:mod:`optimization.solvers.heuristic_log`).  This module reads all three and
flattens them into one long-format frame so a sweep can be plotted or
integrated without caring which solver produced a run.

Every row is one event on one solve.  ``elapsed_seconds`` is measured from the
start of that solve, and the two columns that are comparable across all
backends are:

``incumbent``
    Best objective value attained so far.  NaN for ReCom records that are still
    infeasible, since an infeasible partition has no objective value.
``bound``
    Best proven bound so far; NaN for the heuristics, which have none.

Caveats worth knowing before you plot:

* CP-SAT and Gurobi log values in the solver's own objective scale.  That scale
  is 1.0 whenever there is no choice objective (so the whole edges benchmark is
  already in cut edges); for choice runs divide CP-SAT values by
  ``choice_objective.scale`` via ``cpsat_objective_scale``.
* Gurobi prints its node table on a display interval (5s by default) plus one
  line per new incumbent, so its bound trajectory is sampled, not continuous.
* For the burst solvers a whole burst is sampled before any of it is logged, so
  ``elapsed_seconds`` is quantized to burst boundaries and ``iteration`` is the
  attempted-move count at the end of the burst that produced the state.

Command line::

    python -m benchmark.solver_logs <benchmark_root> --out solver_progress.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable, Iterator

import pandas as pd

from benchmark.results import _write_csv, discover_run_dirs
from benchmark.runner import MANIFEST_FILENAME

LOG_DIRNAME = "solver_logs"

_FILENAME = re.compile(r"^solver_(?P<index>\d+)_(?P<rest>.+)\.(?:log|jsonl)$")
# Units may contain underscores (attendance_area), so the level is the longest
# prefix that still ends in ``_<depth>``.
_LEVEL_AND_SOLVER = re.compile(r"^(?P<level>.+_\d+)_(?P<solver>.+)$")

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_GUROBI_FINAL = re.compile(
    rf"^Best objective (?P<incumbent>{_NUMBER}|-), "
    rf"best bound (?P<bound>{_NUMBER}|-), "
    rf"gap (?P<gap>{_NUMBER})%"
)
_GUROBI_EXPLORED = re.compile(rf"^Explored .*? in (?P<seconds>{_NUMBER}) seconds")
_GUROBI_ROOT = re.compile(
    rf"^Root relaxation: objective (?P<bound>{_NUMBER}), "
    rf".*?(?P<seconds>{_NUMBER}) seconds"
)
_CPSAT_EVENT = re.compile(
    rf"^#(?P<tag>Bound|\d+)\s+(?P<seconds>{_NUMBER})s\s+(?P<rest>.*)$"
)
_CPSAT_BEST = re.compile(r"best:(?P<best>\S+)")
_CPSAT_NEXT = re.compile(r"next:\[(?P<low>[^,\]]*)(?:,(?P<high>[^\]]*))?\]")
_CPSAT_SUMMARY_KEYS = ("status", "objective", "best_bound", "walltime", "gap_integral")


# ---------------------------------------------------------------------- #
# format detection
# ---------------------------------------------------------------------- #
def detect_format(path: str) -> str | None:
    """Return ``"recom"``, ``"cpsat"``, ``"gurobi"``, or None if unrecognized."""

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        head = handle.read(8192)
    if not head.strip():
        return None
    if head.lstrip().startswith("{"):
        return "recom"
    if "Starting CP-SAT solver" in head or "Parameters: " in head:
        return "cpsat"
    if "Gurobi Optimizer version" in head or "Gurobi" in head:
        return "gurobi"
    return None


def parse_log_name(filename: str) -> dict[str, Any]:
    """Recover the solve index, level, and solver name from a log filename."""

    match = _FILENAME.match(os.path.basename(filename))
    if match is None:
        return {"log_index": None, "level": None, "solver": None}
    rest = match.group("rest")
    parts = _LEVEL_AND_SOLVER.match(rest)
    if parts is None:
        level, _, solver = rest.rpartition("_")
        return {
            "log_index": int(match.group("index")),
            "level": level or None,
            "solver": solver or None,
        }
    return {
        "log_index": int(match.group("index")),
        "level": parts.group("level"),
        "solver": parts.group("solver"),
    }


# ---------------------------------------------------------------------- #
# per-format parsers
# ---------------------------------------------------------------------- #
def parse_recom_log(path: str) -> list[dict[str, Any]]:
    """Flatten a heuristic improvement log, exploding penalties into columns."""

    rows: list[dict[str, Any]] = []
    header: dict[str, Any] = {}
    for line in _lines(path):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = entry.get("record")
        if record == "header":
            header = entry
            continue
        if record == "summary":
            rows.append(
                {
                    "record": "summary",
                    "elapsed_seconds": entry.get("elapsed_seconds"),
                    "improvements": entry.get("improvements"),
                    "stop_reason": entry.get("stop_reason"),
                    "attempted_moves": entry.get("attempted_moves"),
                    "accepted_moves": entry.get("accepted_moves"),
                }
            )
            continue
        if record != "improvement":
            continue

        feasible = bool(entry.get("feasible"))
        boundary_cost = entry.get("boundary_cost")
        row = {
            "record": "improvement",
            "event_index": entry.get("index"),
            "elapsed_seconds": entry.get("elapsed_seconds"),
            "iteration": entry.get("iteration"),
            "burst": entry.get("burst"),
            "feasible": feasible,
            "incumbent": boundary_cost if feasible else math.nan,
            "boundary_cost": boundary_cost,
            "total_violation": entry.get("total_violation"),
            "squared_violation": entry.get("squared_violation"),
            "unweighted_lagrangian": entry.get("unweighted_lagrangian"),
            "weighted_lagrangian": entry.get("weighted_lagrangian"),
        }
        for label, value in (entry.get("penalties") or {}).items():
            row[f"pen_{label}"] = value
        for label, value in (entry.get("weights") or {}).items():
            row[f"w_{label}"] = value
        rows.append(row)

    for row in rows:
        row.setdefault("solver", header.get("solver"))
        row.setdefault("level", header.get("level"))
        row.setdefault("penalty_scale", header.get("penalty_scale"))
    return rows


def parse_cpsat_log(path: str, *, objective_scale: float = 1.0) -> list[dict[str, Any]]:
    """Read ``#N`` / ``#Bound`` search events and the response summary."""

    scale = float(objective_scale)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for line in _lines(path):
        event = _CPSAT_EVENT.match(line)
        if event is not None:
            rest = event.group("rest")
            best_match = _CPSAT_BEST.search(rest)
            next_match = _CPSAT_NEXT.search(rest)
            if best_match is None and next_match is None:
                continue
            incumbent = _number(best_match.group("best")) if best_match else None
            low = _number(next_match.group("low")) if next_match else None
            high = _number(next_match.group("high")) if next_match else None
            bound, sense = _cpsat_bound(incumbent, low, high)
            rows.append(
                {
                    "record": "bound" if event.group("tag") == "Bound" else "incumbent",
                    "elapsed_seconds": float(event.group("seconds")),
                    "incumbent": _scaled(incumbent, scale),
                    "bound": _scaled(bound, scale),
                    "bound_lower": _scaled(low, scale),
                    "bound_upper": _scaled(high, scale),
                    "sense": sense,
                    "solution_number": (
                        int(event.group("tag"))
                        if event.group("tag").isdigit()
                        else None
                    ),
                }
            )
            continue
        for key in _CPSAT_SUMMARY_KEYS:
            if line.startswith(f"{key}:"):
                summary[key] = line.split(":", 1)[1].strip()

    if summary:
        incumbent = _number(summary.get("objective"))
        bound = _number(summary.get("best_bound"))
        rows.append(
            {
                "record": "final",
                "elapsed_seconds": _number(summary.get("walltime")),
                "incumbent": _scaled(incumbent, scale),
                "bound": _scaled(bound, scale),
                "status": summary.get("status"),
                "gap_integral": _number(summary.get("gap_integral")),
            }
        )
    return [_with_gap(row) for row in rows]


def parse_gurobi_log(path: str) -> list[dict[str, Any]]:
    """Read the node table's Incumbent/BestBd/Gap columns plus the final line."""

    rows: list[dict[str, Any]] = []
    for line in _lines(path):
        root = _GUROBI_ROOT.match(line)
        if root is not None:
            rows.append(
                {
                    "record": "root_relaxation",
                    "elapsed_seconds": float(root.group("seconds")),
                    "bound": _number(root.group("bound")),
                }
            )
            continue
        final = _GUROBI_FINAL.match(line)
        if final is not None:
            rows.append(
                {
                    "record": "final",
                    "incumbent": _number(final.group("incumbent")),
                    "bound": _number(final.group("bound")),
                    "gap_rel": _number(final.group("gap")) / 100.0,
                }
            )
            continue
        explored = _GUROBI_EXPLORED.match(line)
        if explored is not None:
            rows.append(
                {
                    "record": "explored",
                    "elapsed_seconds": float(explored.group("seconds")),
                }
            )
            continue
        node = _gurobi_node_row(line)
        if node is not None:
            rows.append(node)

    # The final line carries no timestamp of its own; borrow the last one seen.
    last_elapsed = None
    for row in rows:
        if row.get("elapsed_seconds") is not None:
            last_elapsed = row["elapsed_seconds"]
        elif last_elapsed is not None:
            row["elapsed_seconds"] = last_elapsed
    return [_with_gap(row) for row in rows]


def _gurobi_node_row(line: str) -> dict[str, Any] | None:
    """Parse one node-table line by walking its fixed-width tail backwards."""

    tokens = line.split()
    if len(tokens) < 5 or not re.fullmatch(r"\d+s", tokens[-1]):
        return None
    gap = tokens[-3]
    if gap != "-" and not gap.endswith("%"):
        return None
    return {
        "record": "node",
        "elapsed_seconds": float(tokens[-1][:-1]),
        "incumbent": _number(tokens[-5]),
        "bound": _number(tokens[-4]),
        "gap_rel": (_number(gap[:-1]) / 100.0 if gap.endswith("%") else None),
    }


# ---------------------------------------------------------------------- #
# collection
# ---------------------------------------------------------------------- #
def parse_log(path: str, *, cpsat_objective_scale: float = 1.0) -> list[dict[str, Any]]:
    """Dispatch on the detected format and stamp identity columns on each row."""

    log_format = detect_format(path)
    if log_format is None:
        return []
    if log_format == "recom":
        rows = parse_recom_log(path)
    elif log_format == "cpsat":
        rows = parse_cpsat_log(path, objective_scale=cpsat_objective_scale)
    else:
        rows = parse_gurobi_log(path)

    identity = parse_log_name(path)
    identity["log_file"] = os.path.basename(path)
    identity["log_format"] = log_format
    stamped = []
    for row in rows:
        merged = {key: value for key, value in identity.items() if value is not None}
        merged.update({k: v for k, v in row.items() if v is not None})
        stamped.append(merged)
    return stamped


def parse_run_dir(
    run_dir: str, *, cpsat_objective_scale: float = 1.0
) -> list[dict[str, Any]]:
    """Parse every log under ``<run_dir>/solver_logs``."""

    log_dir = os.path.join(run_dir, LOG_DIRNAME)
    if not os.path.isdir(log_dir):
        return []

    manifest: dict[str, Any] = {}
    manifest_path = os.path.join(run_dir, MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            manifest = {}

    rows: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith((".log", ".jsonl")):
            continue
        for row in parse_log(
            os.path.join(log_dir, filename),
            cpsat_objective_scale=cpsat_objective_scale,
        ):
            rows.append(
                {
                    "task_id": manifest.get("task_id"),
                    "config_hash": manifest.get("config_hash"),
                    **row,
                }
            )
    return rows


def collect(
    root: str,
    *,
    out_csv: str | None = None,
    cpsat_objective_scale: float = 1.0,
) -> pd.DataFrame:
    """Parse every solver log under ``root`` into one long-format frame."""

    root_dir = os.path.abspath(os.path.expanduser(root))
    run_dirs = discover_run_dirs(root_dir) or [root_dir]

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for row in parse_run_dir(run_dir, cpsat_objective_scale=cpsat_objective_scale):
            rows.append({"rel_path": os.path.relpath(run_dir, root_dir) or ".", **row})

    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_keys = [
            key
            for key in ("rel_path", "log_index", "elapsed_seconds")
            if key in frame.columns
        ]
        frame = frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    if out_csv:
        _write_csv(frame, root_dir, out_csv)
    return frame


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _lines(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.rstrip("\n")


def _number(value: Any) -> float | None:
    """Parse one log field, tolerating ``-``, ``inf``, and 1'234 separators."""

    if value is None:
        return None
    text = str(value).strip().replace("'", "")
    if not text or text in {"-", "cutoff", "postponed", "infeasible"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return None if math.isinf(parsed) else parsed


def _scaled(value: float | None, scale: float) -> float | None:
    if value is None:
        return None
    return value / scale if scale and scale != 1.0 else value


def _cpsat_bound(
    incumbent: float | None, low: float | None, high: float | None
) -> tuple[float | None, str | None]:
    """Pick the bound endpoint that sits opposite the incumbent.

    CP-SAT prints the objective domain it has yet to explore, so for a
    minimization the bound is the low end and for a maximization the high end.
    """

    if incumbent is None:
        return low, None
    if high is not None and incumbent >= high:
        return low, "minimize"
    if low is not None and incumbent <= low:
        return high, "maximize"
    return low, None


def _with_gap(row: dict[str, Any]) -> dict[str, Any]:
    incumbent = row.get("incumbent")
    bound = row.get("bound")
    if incumbent is not None and bound is not None:
        row["gap_abs"] = abs(incumbent - bound)
        if row.get("gap_rel") is None and abs(incumbent) > 0:
            row["gap_rel"] = abs(incumbent - bound) / abs(incumbent)
    return row


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="Benchmark output directory (or one run dir).")
    parser.add_argument(
        "--out",
        default="solver_progress.csv",
        help="CSV path, relative to root unless absolute.",
    )
    parser.add_argument(
        "--cpsat-objective-scale",
        type=float,
        default=1.0,
        help="Divide CP-SAT log values by this (choice_objective.scale).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    frame = collect(
        args.root,
        out_csv=args.out,
        cpsat_objective_scale=args.cpsat_objective_scale,
    )
    if frame.empty:
        print(f"No solver logs found under {args.root}.")
        return 1
    solvers = ", ".join(
        sorted(frame.get("solver", pd.Series(dtype=str)).dropna().unique())
    )
    print(
        f"Parsed {len(frame)} events from "
        f"{frame.groupby(['rel_path', 'log_file']).ngroups} logs [{solvers}] -> "
        f"{args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
