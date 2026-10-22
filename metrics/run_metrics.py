"""Run-level metrics across single, recursive, and iterative strategies."""

from __future__ import annotations

from metrics.base import MetricOutput, MetricsContext
from metrics.columns import MetricColumns
from metrics.spatial import SPATIAL_FIELDS


def compute(context: MetricsContext) -> MetricOutput:
    stage_rows = []
    compute_stage_metrics = context.compute_stage_metrics
    final_is_eligible = context.solution.feasible
    total_wall_time = _total_wall_time(context)
    flat = {}

    if final_is_eligible:
        flat.update(
            {
                MetricColumns.FINAL_OBJECTIVE: context.solution.objective,
                MetricColumns.FINAL_STATUS: context.solution.status,
                MetricColumns.FINAL_WALL_TIME: context.solution.wall_time,
                MetricColumns.TOTAL_WALL_TIME: total_wall_time,
                MetricColumns.FINAL_STAGE_INDEX: context.final_stage_index,
            }
        )

    if (
        final_is_eligible
        and context.solution.metadata.get("choice_utility") is not None
    ):
        flat[MetricColumns.FINAL_CHOICE_UTILITY] = context.solution.metadata[
            "choice_utility"
        ]

    level_counts: dict[str, int] = {}
    for stage in context.stages:
        level_counts[stage.level.name] = level_counts.get(stage.level.name, 0) + 1

    for idx, (name, solution) in enumerate(zip(context.stage_names, context.stages)):
        stage_is_eligible = solution.feasible and bool(solution.assignment)
        measured = compute_stage_metrics and stage_is_eligible
        spatial = context.spatial_metrics(solution) if measured else None
        row = {
            "name": name,
            "index": idx,
            "level": solution.level.name,
            "status": solution.status,
            "objective": solution.objective,
            **{
                field: getattr(spatial, field) if spatial else None
                for field in SPATIAL_FIELDS
            },
            "wall_time": solution.wall_time,
            "contiguous": solution.is_contiguous() if measured else None,
            "num_nodes": solution.problem.A,
            "num_zones": solution.problem.Z,
            "metadata": dict(solution.metadata),
        }
        if solution.metadata.get("choice_utility") is not None:
            row["choice_utility"] = solution.metadata["choice_utility"]
        stage_rows.append(row)

        # Every stage is keyed by its name; a level solved once is also keyed
        # by the level alone.
        suffixes = [name]
        if level_counts[solution.level.name] == 1:
            suffixes.append(solution.level.name)
        for suffix in suffixes:
            if final_is_eligible:
                flat[f"objective_{suffix}"] = solution.objective
                flat[f"wall_time_{suffix}"] = solution.wall_time
            if measured:
                flat.update(
                    {f"{field}_{suffix}": row[field] for field in SPATIAL_FIELDS}
                )

    final_stage = stage_rows[context.final_stage_index]
    if final_is_eligible and compute_stage_metrics:
        flat[MetricColumns.FINAL_CUT_EDGES] = final_stage["cut_edges"]

    run = {
        "strategy": _strategy_name(context),
        "selection": _selection_reason(context),
        "final_stage": context.final_stage_name,
        "final_stage_index": context.final_stage_index,
        "final_status": context.solution.status,
        "final_objective": context.solution.objective,
        "final_cut_edges": final_stage["cut_edges"],
        "total_wall_time": total_wall_time,
        "stages": stage_rows,
    }
    return MetricOutput(metrics=flat, run=run)


def _total_wall_time(context: MetricsContext) -> float:
    return sum(float(stage.wall_time or 0.0) for stage in context.stages)


def _strategy_name(context: MetricsContext) -> str:
    if context.config.get("strategy"):
        return str(context.config["strategy"])
    if context._is_iterative_run():
        return "iterative"
    if len(context.stages) > 1:
        return "recursive"
    return "single"


def _selection_reason(context: MetricsContext) -> str:
    if context.solution.metadata.get("enumerated_solution_selected"):
        return "seeded_random_enumerated_solution"
    if not context._is_iterative_run():
        return "literal_final_stage"
    if context.solution.metadata.get("choice_utility") is not None:
        best = max(
            (
                stage.metadata.get("choice_utility")
                for stage in context.stages
                if stage.feasible
                and stage.assignment
                and stage.metadata.get("choice_utility") is not None
            ),
            default=None,
        )
        if best == context.solution.metadata.get("choice_utility"):
            return "best_choice_utility"
    return "literal_final_stage"
