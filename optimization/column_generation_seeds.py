"""Portable seed collection for complete-zone analytical column generation."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from optimization.branch_price.analytical_patterns import (
    AnalyticalPatternValuator,
    AnalyticalZonePattern,
)
from optimization.branch_price.patterns import ZonePatternValidator, zone_perimeter
from optimization.data import contiguity
from optimization.data.conversion import LevelConverter
from optimization.data.initial_solutions import complete_assignment, initial_solution
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solution import graph_fingerprint
from optimization.solvers import get_solver
from optimization.solvers.base import Solver


@dataclass(frozen=True, slots=True)
class SeedProvenance:
    source: str
    accepted: bool
    reason: str | None = None
    source_path: str | None = None
    source_level: str | None = None
    source_solver: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeedCollectionResult:
    assignments: tuple[dict[int, int], ...]
    patterns: tuple[AnalyticalZonePattern, ...]
    best_assignment: dict[int, int]
    provenance: tuple[SeedProvenance, ...]
    rejected_count: int


def normalize_seed_labels(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
) -> dict[int, int]:
    """Relabel a complete partition by the centroid each raw label contains."""
    normalized_input = {int(node): int(zone) for node, zone in assignment.items()}
    missing_centroids = [
        centroid for centroid in problem.centroids if centroid not in normalized_input
    ]
    if missing_centroids:
        raise ValueError(f"Seed omits centroid nodes {missing_centroids}.")
    raw_to_label: dict[int, int] = {}
    for label, centroid in enumerate(problem.centroids):
        raw = normalized_input[centroid]
        if raw in raw_to_label:
            raise ValueError("Two centroids belong to the same raw seed label.")
        raw_to_label[raw] = label
    extra = set(normalized_input.values()) - set(raw_to_label)
    if extra:
        raise ValueError(f"Seed contains labels without centroids: {sorted(extra)}.")
    return {node: raw_to_label[zone] for node, zone in normalized_input.items()}


def validate_complete_seed(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
    *,
    centroid_neighbor_radius: int = 0,
    validator: ZonePatternValidator | None = None,
) -> dict[int, int]:
    """Validate exact local and district constraints for one complete partition."""
    assignment = {int(node): int(zone) for node, zone in assignment.items()}
    if set(assignment) != set(problem.nodes):
        missing = sorted(set(problem.nodes) - set(assignment))
        extra = sorted(set(assignment) - set(problem.nodes))
        raise ValueError(
            f"Seed must assign every graph node (missing={missing}, extra={extra})."
        )
    if set(assignment.values()) != set(range(problem.Z)):
        raise ValueError("Seed must represent every zone label and no extra labels.")
    validator = validator or ZonePatternValidator(
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    perimeters = 0
    for label in range(problem.Z):
        nodes = frozenset(
            node for node, assigned in assignment.items() if assigned == label
        )
        perimeter = zone_perimeter(problem.G, nodes)
        validator.validate_membership(
            label=label,
            nodes=nodes,
            perimeter=perimeter,
        )
        perimeters += perimeter
    cut_edges = sum(
        assignment[left] != assignment[right] for left, right in problem.G.edges
    )
    if perimeters != 2 * cut_edges:
        raise ValueError("Seed violates the exact factor-two perimeter identity.")
    if problem.boundary_prop >= 0:
        cap = math.floor(problem.boundary_prop * problem.G.number_of_edges())
        if cut_edges > cap:
            raise ValueError("Seed exceeds the district boundary cap.")
    return assignment


def extract_analytical_seed_patterns(
    problem: ZoneProblem,
    assignments: Sequence[Mapping[int, int]],
    valuator: AnalyticalPatternValuator,
    *,
    deadline: float | None = None,
) -> tuple[AnalyticalZonePattern, ...]:
    """Value every unique complete zone from accepted seed partitions."""
    patterns = {}
    for assignment in assignments:
        for label in range(problem.Z):
            nodes = frozenset(
                node for node, zone in assignment.items() if int(zone) == label
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Seed pattern valuation reached its deadline.")
            pattern = valuator.value(label, nodes, deadline=deadline)
            previous = patterns.get(pattern.key)
            if previous is not None and not math.isclose(
                previous.shi_welfare,
                pattern.shi_welfare,
                rel_tol=1e-7,
                abs_tol=1e-7,
            ):
                raise ValueError("A seed pattern key has conflicting Shi valuations.")
            patterns[pattern.key] = pattern
    return tuple(patterns.values())


def load_seed_assignment(
    path: str | Path,
    problem: ZoneProblem,
    *,
    target_level: LevelSpec | str | None = None,
    converter: LevelConverter | None = None,
) -> dict[int, int]:
    """Load raw, area, direct-output, stage, or benchmark-directory zoning JSON."""
    source = Path(path).expanduser().resolve()
    target_level = LevelSpec.parse(target_level or problem.level)
    converter = converter or LevelConverter()
    if source.is_dir():
        source = _assignment_file_from_directory(source, target_level.name)
    if not source.is_file():
        raise ValueError(f"Seed path does not exist: {source}.")
    with source.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise ValueError("Seed JSON must contain a node/area-to-zone mapping.")
    values = {int(key): int(value) for key, value in payload.items()}
    source_level = _level_from_assignment_name(source.name) or target_level
    is_area = "zone_dict_area_" in source.name
    if not is_area and source_level != target_level:
        companion_area = source.with_name(f"zone_dict_area_{source_level.name}.json")
        if companion_area.exists():
            return load_seed_assignment(
                companion_area,
                problem,
                target_level=target_level,
                converter=converter,
            )
        raise ValueError(
            "A raw cross-level seed requires its companion area assignment."
        )
    if not is_area and set(values) == set(problem.nodes):
        companion = source.with_name(f"solution_{source_level.name}.json")
        if companion.exists():
            with companion.open(encoding="utf-8") as input_file:
                info = json.load(input_file)
            saved = info.get("graph_fingerprint")
            if saved and saved != graph_fingerprint(problem.G):
                area_path = source.with_name(f"zone_dict_area_{source_level.name}.json")
                if not area_path.exists():
                    raise ValueError(
                        "Raw seed graph fingerprint differs and no area assignment exists."
                    )
                return load_seed_assignment(
                    area_path,
                    problem,
                    target_level=target_level,
                    converter=converter,
                )
        return values
    if not is_area:
        companion_area = source.with_name(f"zone_dict_area_{source_level.name}.json")
        if companion_area.exists():
            return load_seed_assignment(
                companion_area,
                problem,
                target_level=target_level,
                converter=converter,
            )
        raise ValueError(
            "A raw cross-level seed requires its companion area assignment."
        )
    converted = converter.from_area_assignment(
        values,
        source_level,
        problem.G,
        target_level,
    )
    if set(converted) != set(problem.nodes):
        missing = sorted(set(problem.nodes) - set(converted))
        raise ValueError(f"Converted area seed omits target nodes {missing}.")
    return converted


def generate_boundary_move_seeds(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
    *,
    rounds: int,
    centroid_neighbor_radius: int = 0,
    deadline: float | None = None,
) -> tuple[dict[int, int], ...]:
    """Generate deterministic one-node boundary moves accepted by exact validation."""
    if rounds <= 0:
        return ()
    validator = ZonePatternValidator(
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    seeds = []
    current = dict(assignment)
    centroids = set(problem.centroids)
    for node in sorted(problem.nodes):
        if len(seeds) >= rounds:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        if node in centroids:
            continue
        neighboring_labels = {
            current[neighbor]
            for neighbor in problem.neighbors(node)
            if current[neighbor] != current[node]
        }
        for label in sorted(neighboring_labels & problem.candidate_zones(node)):
            candidate = dict(current)
            candidate[node] = label
            candidate = complete_assignment(
                problem,
                contiguity.repair(problem.G, candidate, problem.centroids),
            )
            if candidate[node] != label:
                continue
            try:
                validate_complete_seed(
                    problem,
                    candidate,
                    validator=validator,
                )
            except ValueError:
                continue
            seeds.append(candidate)
            current = candidate
            break
    return tuple(seeds)


def generate_school_swap_seeds(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
    *,
    rounds: int,
    centroid_neighbor_radius: int = 0,
    deadline: float | None = None,
) -> tuple[dict[int, int], ...]:
    """Try exact-valid boundary moves of noncentroid school nodes."""
    school_nodes = {
        node
        for node in problem.nodes
        if problem.num_schools(node) > 0 and node not in set(problem.centroids)
    }
    if not school_nodes or rounds <= 0:
        return ()
    validator = ZonePatternValidator(
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    seeds = []
    current = dict(assignment)
    for node in sorted(school_nodes):
        if len(seeds) >= rounds:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        targets = {
            current[neighbor]
            for neighbor in problem.neighbors(node)
            if current[neighbor] != current[node]
        }
        for label in sorted(targets & problem.candidate_zones(node)):
            candidate = dict(current)
            candidate[node] = label
            candidate = complete_assignment(
                problem,
                contiguity.repair(problem.G, candidate, problem.centroids),
            )
            if candidate[node] != label:
                continue
            try:
                validate_complete_seed(problem, candidate, validator=validator)
            except ValueError:
                continue
            seeds.append(candidate)
            current = candidate
            break
    return tuple(seeds)


def collect_column_generation_seeds(
    problem: ZoneProblem,
    solver: Solver,
    valuator: AnalyticalPatternValuator,
    *,
    seed_paths: Sequence[str] = (),
    recom_seed_runs: int = 0,
    local_move_rounds: int = 0,
    centroid_neighbor_radius: int = 0,
    random_seed: int = 0,
    workers: int = 1,
    deadline: float | None = None,
) -> SeedCollectionResult:
    """Collect, relabel, validate, value, and deduplicate all configured seeds."""
    assignments: list[dict[int, int]] = []
    provenance: list[SeedProvenance] = []
    seen = set()

    def accept(raw: Mapping[int, int], source: str, **details) -> None:
        try:
            normalized = normalize_seed_labels(problem, raw)
            validated = validate_complete_seed(
                problem,
                normalized,
                centroid_neighbor_radius=centroid_neighbor_radius,
                validator=valuator.validator,
            )
        except (TypeError, ValueError) as exc:
            provenance.append(
                SeedProvenance(
                    source=source, accepted=False, reason=str(exc), **details
                )
            )
            return
        key = tuple(sorted(validated.items()))
        if key not in seen:
            assignments.append(validated)
            seen.add(key)
        provenance.append(SeedProvenance(source=source, accepted=True, **details))

    if problem.hint:
        accept(problem.hint, "problem_hint")
    try:
        configured_initial = initial_solution(
            problem,
            getattr(solver, "options", {}).get("hints", "voronoi"),
        )
    except ValueError as exc:
        provenance.append(
            SeedProvenance(
                source="configured_initial",
                accepted=False,
                reason=str(exc),
            )
        )
    else:
        if configured_initial is not None:
            accept(configured_initial.assignment, "configured_initial")
    for raw_path in seed_paths:
        try:
            loaded = load_seed_assignment(raw_path, problem)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            provenance.append(
                SeedProvenance(
                    source="saved",
                    source_path=str(raw_path),
                    accepted=False,
                    reason=str(exc),
                )
            )
        else:
            accept(loaded, "saved", source_path=str(raw_path))

    if deadline is None or time.monotonic() < deadline:
        if assignments:
            problem.hint = assignments[0]
        old_time_limit = getattr(solver, "options", {}).get("solve_time_limit")
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            configured = (
                float(old_time_limit) if old_time_limit is not None else remaining
            )
            solver.options["solve_time_limit"] = min(configured, remaining * 0.5)
        try:
            solution = solver.solve(problem)
        finally:
            if deadline is not None:
                if old_time_limit is None:
                    solver.options.pop("solve_time_limit", None)
                else:
                    solver.options["solve_time_limit"] = old_time_limit
        if solution.feasible:
            accept(
                solution.assignment,
                "primary_solver",
                source_solver=getattr(solver, "name", None),
            )
        else:
            provenance.append(
                SeedProvenance(
                    source="primary_solver",
                    source_solver=getattr(solver, "name", None),
                    accepted=False,
                    reason=f"solver status {solution.status}",
                )
            )

    for run in range(max(0, int(recom_seed_runs))):
        if not assignments or (deadline is not None and time.monotonic() >= deadline):
            break
        options = dict(getattr(solver, "options", {}))
        options.update(
            {
                "seed": random_seed + run,
                "workers": 1,
                "hints": "none",
                "save_solver_logs": False,
                "save_solver_progress": False,
            }
        )
        if deadline is not None:
            remaining = max(0.0, deadline - time.monotonic())
            options["solve_time_limit"] = remaining / (
                max(0, int(recom_seed_runs) - run) + 2
            )
        problem.hint = assignments[run % len(assignments)]
        recom = get_solver("recom", **options)
        solution = recom.solve(problem)
        if solution.feasible:
            accept(solution.assignment, "recom", source_solver="recom")
        else:
            provenance.append(
                SeedProvenance(
                    source="recom",
                    source_solver="recom",
                    accepted=False,
                    reason=f"solver status {solution.status}",
                )
            )

    if assignments and local_move_rounds > 0:
        for moved in generate_boundary_move_seeds(
            problem,
            assignments[0],
            rounds=local_move_rounds,
            centroid_neighbor_radius=centroid_neighbor_radius,
            deadline=deadline,
        ):
            accept(moved, "support_closed_boundary_move")
        for moved in generate_school_swap_seeds(
            problem,
            assignments[0],
            rounds=local_move_rounds,
            centroid_neighbor_radius=centroid_neighbor_radius,
            deadline=deadline,
        ):
            accept(moved, "school_node_swap")
    if not assignments:
        reasons = "; ".join(
            item.reason or item.source for item in provenance if not item.accepted
        )
        raise ValueError(
            "Zoned analytical optimization requires at least one valid complete seed"
            + (f": {reasons}" if reasons else ".")
        )
    valued_assignments = []
    pattern_map = {}
    for assignment in assignments:
        assignment_patterns = []
        try:
            for label in range(problem.Z):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError
                nodes = frozenset(
                    node for node, zone in assignment.items() if zone == label
                )
                assignment_patterns.append(
                    valuator.value(label, nodes, deadline=deadline)
                )
        except TimeoutError:
            if valued_assignments:
                provenance.append(
                    SeedProvenance(
                        source="seed_valuation",
                        accepted=False,
                        reason="optional seed valuation reached its deadline",
                    )
                )
                break
            raise
        valued_assignments.append(assignment)
        for pattern in assignment_patterns:
            pattern_map[pattern.key] = pattern
    assignments = valued_assignments
    patterns = tuple(pattern_map.values())
    pattern_by_key = {pattern.key: pattern for pattern in patterns}
    objectives = []
    for assignment in assignments:
        objective = 0.0
        for label in range(problem.Z):
            key = (
                label,
                frozenset(node for node, zone in assignment.items() if zone == label),
            )
            objective += pattern_by_key[key].shi_welfare
        objectives.append(objective)
    best_index = max(range(len(assignments)), key=lambda index: objectives[index])
    return SeedCollectionResult(
        assignments=tuple(assignments),
        patterns=patterns,
        best_assignment=dict(assignments[best_index]),
        provenance=tuple(provenance),
        rejected_count=sum(not item.accepted for item in provenance),
    )


def _assignment_file_from_directory(directory: Path, target_level: str) -> Path:
    manifest_path = directory / "benchmark_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as input_file:
            manifest = json.load(input_file)
        final_name = manifest.get("final_stage")
        stages = manifest.get("stages") or []
        final = next(
            (stage for stage in stages if stage.get("name") == final_name),
            stages[-1] if stages else None,
        )
        if final is None:
            raise ValueError("Benchmark seed directory has no saved stages.")
        directory = directory / final["path"]
    raw = directory / f"zone_dict_{target_level}.json"
    if raw.exists():
        return raw
    area = directory / f"zone_dict_area_{target_level}.json"
    if area.exists():
        return area
    area_candidates = sorted(directory.glob("zone_dict_area_*.json"))
    if area_candidates:
        return area_candidates[-1]
    raw_candidates = sorted(directory.glob("zone_dict_*.json"))
    raw_candidates = [
        path for path in raw_candidates if "zone_dict_area_" not in path.name
    ]
    if raw_candidates:
        return raw_candidates[-1]
    raise ValueError(f"Seed directory contains no zone assignment: {directory}.")


def _level_from_assignment_name(filename: str) -> LevelSpec | None:
    match = re.match(r"zone_dict_(?:area_)?(.+)\.json$", filename)
    if not match:
        return None
    try:
        return LevelSpec.parse(match.group(1))
    except ValueError:
        return None
