"""Serializable building blocks for optimization runs.

A step describes a solve; an input describes how an earlier solution supplies
values to that solve. Data selectors and graph construction belong to the run,
so every step shares a Dataset while constraints and search settings can vary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

from optimization.levels import LevelSpec
from optimization.solvers.recom_policy import ReComSettings

if TYPE_CHECKING:
    from optimization.config import OptimizationConfig


CONSTRAINT_FIELDS = frozenset(
    {
        "frl_dev",
        "racial_dev",
        "overage",
        "shortage",
        "max_distance",
        "school_count_tolerance",
        "boundary_prop",
        "centroid_neighbor_radius",
    }
)
OPTION_FIELDS = frozenset(
    {
        "seed",
        "workers",
        "hints",
        "feasible_hint_time_limit",
        "save_solver_logs",
        "save_solver_progress",
        "secondary_objective",
        "linearization_level",
        "cp_model_probing_level",
        "symmetry_level",
        "cp_sat_search_strategy",
        "recom_iterations",
        *(f"recom_{f.name}" for f in fields(ReComSettings)),
        "capacity_match_unassigned_weight",
        "capacity_match_designated_weight",
        "school_distance_area_weight",
        "school_distance_student_weight",
        "mid_lottery_scale",
        "mid_utility_handling",
        "mid_preference_source",
        "mid_transport_bounds",
        "mid_complementary_slackness",
        "mid_complementary_slackness_slack",
        "choice_access_triangle",
        "choice_access_cardinality",
        "choice_access_triangle_limit",
        "choice_access_complete",
        "choice_access_completion_limit",
        "max_iterations",
        "tolerance",
        "budget_accounting",
        "choice_model_method",
        "choice_utility_scale",
        "choice_utility_hints",
        "saa_num_seeds",
        "saa_tie_breaking_method",
        "saa_welfare_bound",
        "saa_relative_gap",
        "saa_multicut",
        "zoned_transport_workers",
    }
)


def checked_map(value, allowed, label) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a map.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label}: {sorted(unknown)}.")
    return dict(value)


def checked_constraints(value, label="constraints") -> dict:
    import math

    result = checked_map(value, CONSTRAINT_FIELDS, label)
    for name, raw in result.items():
        if name in {"max_distance", "centroid_neighbor_radius"}:
            continue  # Validated by OptimizationConfig, including 'auto'.
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(raw)
        ):
            raise ValueError(f"{label}.{name} must be a finite number.")
        if name == "school_count_tolerance" and raw < 0:
            raise ValueError(f"{label}.{name} must be non-negative.")
    return result


@dataclass(frozen=True)
class SolutionInput:
    """Use selected target nodes from a prior step as hints or hard assignments.

    ``interior`` selects everything outside the boundary band; ``boundary``
    selects its complement. The band includes both endpoints of every cut edge
    and extends ``radius`` graph hops. ``nodes`` intersects this selection with
    explicit target graph node indices. Hard assignments never become hints.
    """

    source: str = "previous"
    mode: str = "hint"
    select: str = "all"
    radius: int = 1
    nodes: list[int] | None = None

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("input.source must name an earlier step or 'previous'.")
        if self.mode not in {"hint", "assign"}:
            raise ValueError("input.mode must be 'hint' or 'assign'.")
        if self.select not in {"all", "interior", "boundary"}:
            raise ValueError("input.select must be all, interior, or boundary.")
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, int)
            or self.radius < 0
        ):
            raise ValueError("input.radius must be a non-negative integer.")
        if self.nodes is not None and (
            not isinstance(self.nodes, list)
            or any(isinstance(n, bool) or not isinstance(n, int) for n in self.nodes)
        ):
            raise ValueError("input.nodes must be a list of target node indices.")


@dataclass(frozen=True)
class SolveStep:
    name: str
    level: str
    method: str
    objective: str = "cut_edges"
    time_limit: float | None = None
    gap_limit: float | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    inputs: list[SolutionInput] | None = None

    def __post_init__(self):
        from optimization.objectives import validate_method_objective

        if not isinstance(self.name, str) or not self.name or self.name == "previous":
            raise ValueError("step.name must be nonempty and cannot be 'previous'.")
        object.__setattr__(self, "level", LevelSpec.parse(self.level).name)
        validate_method_objective(self.method, self.objective)
        for name in ("time_limit", "gap_limit"):
            value = getattr(self, name)
            if value is not None:
                import math

                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"step.{name} must be finite and non-negative.")
        object.__setattr__(
            self,
            "constraints",
            checked_constraints(self.constraints, "step constraints"),
        )
        object.__setattr__(
            self, "options", checked_map(self.options, OPTION_FIELDS, "step options")
        )
        if self.inputs is None:
            return  # Expanded with the step's position by normalize_steps.
        if not isinstance(self.inputs, list):
            raise ValueError("step.inputs must be a list.")
        object.__setattr__(
            self,
            "inputs",
            [
                value
                if isinstance(value, SolutionInput)
                else SolutionInput(
                    **checked_map(
                        value, {f.name for f in fields(SolutionInput)}, "input keys"
                    )
                )
                for value in self.inputs
            ],
        )


def normalize_steps(config: OptimizationConfig) -> list[SolveStep]:
    """Expand defaults once, validating names and backward-only dependencies."""
    raw_steps = config.steps
    if not isinstance(raw_steps, list):
        raise ValueError("steps must be a list of solve steps.")
    if not raw_steps:
        if len(config.levels) != 1:
            raise ValueError(
                "method/objective shorthand needs one level; use steps for multiple solves."
            )
        raw_steps = [{"level": config.levels[0]}]
    result = []
    known = set()
    for index, value in enumerate(raw_steps):
        if isinstance(value, SolveStep):
            step = value
        else:
            raw = checked_map(value, {f.name for f in fields(SolveStep)}, "step keys")
            raw.setdefault("name", f"step_{index:02d}")
            if "level" not in raw:
                if not config.levels:
                    raise ValueError("Each step needs a level when levels is empty.")
                raw["level"] = config.levels[-1]
            raw.setdefault(
                "method", config.method if config.method is not None else config.solver
            )
            raw.setdefault(
                "objective",
                config.objective if config.objective is not None else "cut_edges",
            )
            raw.setdefault("inputs", [{"source": "previous"}] if index else [])
            step = SolveStep(**raw)
        if step.inputs is None:
            step = replace(step, inputs=[SolutionInput()] if index else [])
        if step.name in known:
            raise ValueError(f"Duplicate pipeline step name {step.name!r}.")
        for binding in step.inputs:
            if (binding.source == "previous" and not index) or (
                binding.source != "previous" and binding.source not in known
            ):
                raise ValueError(
                    f"Step {step.name!r} input {binding.source!r} must reference an earlier step."
                )
        known.add(step.name)
        result.append(step)
    return result


def step_config(
    config: OptimizationConfig, step: SolveStep, index: int = 0
) -> OptimizationConfig:
    """Validated solve-local settings sharing the run's resolved data scenario."""
    limits = config.solve_time_limits
    gaps = config.gap_limits
    objective = step.objective
    from optimization.objectives import get_objective

    objective_spec = get_objective(objective)
    if objective in {"choice_welfare", "saa_welfare"}:
        import math

        iterations = step.options.get("max_iterations", config.max_iterations)
        tolerance = step.options.get("tolerance", config.tolerance)
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError(
                f"max_iterations must be a positive integer for {objective}."
            )
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError(
                f"tolerance must be finite and non-negative for {objective}."
            )
    return replace(
        config,
        method=None,
        objective=None,
        steps=[],
        constraints={},
        strategy="single",
        levels=[step.level],
        solver=step.method,
        weight_edges=config.weight_edges or objective == "boundary_perimeter",
        compactness_objective=objective_spec.compactness_objective
        if step.method in {"cp_bool", "cp_int", "mip"}
        else "cut_edges",
        recom_objective=(
            "welfare"
            if objective in {"matching_welfare", "welfare"}
            else "school_distance"
            if objective_spec.compactness_objective == "school_distance"
            else objective
            if objective in {"cut_edges", "school_distance", "capacity_match"}
            else "cut_edges"
        ),
        solve_time_limits=[
            step.time_limit
            if step.time_limit is not None
            else limits[min(index, len(limits) - 1)]
        ],
        gap_limits=[
            step.gap_limit
            if step.gap_limit is not None
            else gaps[min(index, len(gaps) - 1)]
        ],
        _resolved_data_scenario=config.data_scenario,
        **step.constraints,
        **step.options,
    )
