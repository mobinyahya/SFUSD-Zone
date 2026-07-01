"""Initial solution helpers for heuristic solvers."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.config import OptimizationConfig
from Zone_Generation.optimization.data.conversion import LevelConverter
from Zone_Generation.optimization.data.dataset import Dataset
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers import get_solver


def math_prog_initial_hint(
    dataset: Dataset,
    problem: ZoneProblem,
    options: Mapping | None = None,
) -> dict[int, int] | None:
    """Return a starting hint converted from the cached ``Block_0`` seed.

    Seeds are keyed by data graph namespace and ``centroids_type``.  They are
    generated lazily on ``BlockGroup_1`` with loose constraints, converted to
    ``Block_0`` for storage, and converted from that canonical saved level to the
    requested problem level when needed.
    """

    options = options or {}
    config = getattr(dataset, "config", None)
    if config is None:
        return None

    initial_level = LevelSpec.parse(
        str(options.get("recom_initial_level", "BlockGroup_1"))
    )
    save_level = LevelSpec.parse(
        str(options.get("recom_initial_save_level", "Block_0"))
    )
    if save_level.name != "Block_0":
        raise ValueError("recom_initial_save_level must be Block_0.")

    block_dataset = _dataset_for_level(config, save_level)
    path = _cache_zone_dict_path(block_dataset, config.centroids_type, save_level)
    cache_hit = path.exists()
    if not cache_hit:
        _generate_math_prog_seed(
            config=config,
            initial_level=initial_level,
            save_level=save_level,
            output_path=path,
            options=options,
        )
    _set_cache_metadata(
        problem,
        path=path,
        cache_hit=cache_hit,
        initial_level=initial_level,
        save_level=save_level,
    )
    if not path.exists():
        return None

    block_assignment = _load_assignment(path)
    block_G = block_dataset.graph_for(save_level)
    if problem.level == save_level:
        hint = {
            node: block_assignment[node]
            for node in problem.nodes
            if node in block_assignment
        }
    else:
        hint = LevelConverter().between(
            block_G,
            block_assignment,
            save_level,
            problem.G,
            problem.level,
        )
    return _repair_hint(problem, hint)


def _generate_math_prog_seed(
    *,
    config: OptimizationConfig,
    initial_level: LevelSpec,
    save_level: LevelSpec,
    output_path: Path,
    options: Mapping,
) -> None:
    initial_dataset = _dataset_for_level(config, initial_level)
    save_dataset = _dataset_for_level(config, save_level)

    multiplier = float(options.get("recom_initial_constraint_multiplier", 10.0))
    initial_problem = initial_dataset.problem_for(
        initial_level,
        constraint_multiplier=multiplier,
    )
    seed_solver = get_solver(
        "cp_bool",
        solve_time_limit=float(options.get("recom_initial_time_limit", 60.0)),
        relative_gap_limit=float(options.get("relative_gap_limit", 0.0)),
        seed=int(options.get("seed", 42)),
        workers=int(options.get("workers", 1)),
    )
    solution = seed_solver.solve(initial_problem)
    if not solution.feasible:
        return

    save_problem = save_dataset.problem_for(save_level)
    save_assignment = LevelConverter().between(
        initial_problem.G,
        solution.assignment,
        initial_level,
        save_problem.G,
        save_level,
    )
    save_assignment = _repair_hint(save_problem, save_assignment)

    seed_solution = ZoneSolution(
        problem=save_problem,
        assignment=save_assignment,
        status="SEED",
        objective=None,
        wall_time=solution.wall_time,
        time_to_convergence=solution.time_to_convergence,
        metadata={
            "solver": "cp_bool",
            "initialization_method": "math_prog",
            "seed_source_level": initial_level.name,
            "seed_saved_level": save_level.name,
            "constraint_multiplier": multiplier,
            "source_status": solution.status,
            "generated_at": time.time(),
        },
    )
    _save_seed(seed_solution, output_path)


def _dataset_for_level(config: OptimizationConfig, level: LevelSpec) -> Dataset:
    return Dataset(replace(config, levels=[level.name]))


def _cache_zone_dict_path(
    dataset: Dataset, centroids_type: str, save_level: LevelSpec
) -> Path:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", centroids_type).strip("_")
    cache_dir = Path(dataset.graph_cache_dir) / "recom_initial_solutions" / key
    return cache_dir / f"zone_dict_{save_level.name}.json"


def _set_cache_metadata(
    problem: ZoneProblem,
    *,
    path: Path,
    cache_hit: bool,
    initial_level: LevelSpec,
    save_level: LevelSpec,
) -> None:
    problem._math_prog_initial_cache = {
        "cache_hit": bool(cache_hit),
        "cache_path": str(path),
        "source_level": initial_level.name,
        "saved_level": save_level.name,
        "available": path.exists(),
    }


def _load_assignment(path: Path) -> dict[int, int]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def _save_seed(solution: ZoneSolution, zone_dict_path: Path) -> None:
    zone_dict_path.parent.mkdir(parents=True, exist_ok=True)
    level = solution.level.name
    payload = {str(k): int(v) for k, v in solution.assignment.items()}
    _atomic_json(zone_dict_path, payload)

    area_path = zone_dict_path.parent / f"zone_dict_area_{level}.json"
    _atomic_json(
        area_path,
        {str(k): int(v) for k, v in solution.area_assignment().items()},
    )

    info_path = zone_dict_path.parent / f"solution_{level}.json"
    _atomic_json(
        info_path,
        {
            "level": level,
            "status": solution.status,
            "objective": solution.objective,
            "wall_time": solution.wall_time,
            "time_to_convergence": solution.time_to_convergence,
            "num_zones": solution.problem.Z,
            "centroids": list(solution.problem.centroids),
            "contiguous": None,
            "metadata": solution.metadata,
        },
        indent=2,
    )


def _atomic_json(path: Path, payload, indent: int | None = None) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
    os.replace(tmp_path, path)


def _complete_hint(problem: ZoneProblem, hint: Mapping[int, int]) -> dict[int, int]:
    assignment: dict[int, int] = {}
    for node in problem.nodes:
        zone = hint.get(node)
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


def _repair_hint(problem: ZoneProblem, hint: Mapping[int, int]) -> dict[int, int]:
    assignment = _complete_hint(problem, hint)
    repaired = contiguity.repair(problem.G, assignment, problem.centroids)
    return _complete_hint(problem, repaired)
