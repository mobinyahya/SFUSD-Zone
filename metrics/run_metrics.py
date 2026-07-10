"""Run-level metrics across single, recursive, and iterative strategies."""

from __future__ import annotations

from Config.metrics_config import MetricColumns
from metrics.base import MetricOutput, MetricsContext
from metrics.spatial import compute_spatial_metrics


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
        spatial = (
            compute_spatial_metrics(solution, context.config)
            if compute_stage_metrics and stage_is_eligible
            else None
        )
        contiguous = (
            solution.is_contiguous()
            if compute_stage_metrics and stage_is_eligible
            else None
        )
        row = {
            "name": name,
            "index": idx,
            "level": solution.level.name,
            "status": solution.status,
            "objective": solution.objective,
            "cut_edges": spatial.cut_edges if spatial else None,
            "normalized_cut_edges": spatial.normalized_cut_edges if spatial else None,
            "fractional_cut_edges": spatial.fractional_cut_edges if spatial else None,
            "avg_reock_score": spatial.avg_reock_score if spatial else None,
            "avg_polsby_popper_score": spatial.avg_polsby_popper_score
            if spatial
            else None,
            "wall_time": solution.wall_time,
            "contiguous": contiguous,
            "num_nodes": solution.problem.A,
            "num_zones": solution.problem.Z,
            "metadata": dict(solution.metadata),
        }
        if solution.metadata.get("choice_utility") is not None:
            row["choice_utility"] = solution.metadata["choice_utility"]
        stage_rows.append(row)

        if final_is_eligible:
            flat[f"objective_{name}"] = solution.objective
            flat[f"wall_time_{name}"] = solution.wall_time
        if compute_stage_metrics and stage_is_eligible:
            flat[f"cut_edges_{name}"] = row["cut_edges"]
            flat[f"normalized_cut_edges_{name}"] = row["normalized_cut_edges"]
            flat[f"fractional_cut_edges_{name}"] = row["fractional_cut_edges"]
            flat[f"avg_reock_score_{name}"] = row["avg_reock_score"]
            flat[f"avg_polsby_popper_score_{name}"] = row["avg_polsby_popper_score"]
        if final_is_eligible and level_counts[solution.level.name] == 1:
            flat[f"objective_{solution.level.name}"] = solution.objective
            flat[f"wall_time_{solution.level.name}"] = solution.wall_time
            if compute_stage_metrics and stage_is_eligible:
                flat[f"cut_edges_{solution.level.name}"] = row["cut_edges"]
                flat[f"normalized_cut_edges_{solution.level.name}"] = row[
                    "normalized_cut_edges"
                ]
                flat[f"fractional_cut_edges_{solution.level.name}"] = row[
                    "fractional_cut_edges"
                ]
                flat[f"avg_reock_score_{solution.level.name}"] = row["avg_reock_score"]
                flat[f"avg_polsby_popper_score_{solution.level.name}"] = row[
                    "avg_polsby_popper_score"
                ]

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
