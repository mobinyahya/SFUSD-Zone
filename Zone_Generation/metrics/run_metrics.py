"""Run-level metrics across single, recursive, and iterative strategies."""

from __future__ import annotations

from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.metrics.base import MetricOutput, MetricsContext
from Zone_Generation.metrics.spatial import compute_spatial_metrics


def compute(context: MetricsContext) -> MetricOutput:
    stage_rows = []
    compute_stage_metrics = context.compute_stage_metrics
    flat = {
        MetricColumns.FINAL_OBJECTIVE: context.solution.objective,
        MetricColumns.FINAL_STATUS: context.solution.status,
        MetricColumns.FINAL_WALL_TIME: context.solution.wall_time,
        MetricColumns.TOTAL_WALL_TIME: _total_wall_time(context),
        MetricColumns.TIME_TO_CONVERGENCE: _time_to_convergence(context),
        MetricColumns.FINAL_STAGE_INDEX: context.final_stage_index,
    }

    if context.solution.metadata.get("choice_utility") is not None:
        flat[MetricColumns.FINAL_CHOICE_UTILITY] = context.solution.metadata["choice_utility"]

    level_counts: dict[str, int] = {}
    for stage in context.stages:
        level_counts[stage.level.name] = level_counts.get(stage.level.name, 0) + 1

    for idx, (name, solution) in enumerate(zip(context.stage_names, context.stages)):
        spatial = (
            compute_spatial_metrics(solution, context.config)
            if compute_stage_metrics and solution.assignment
            else None
        )
        contiguous = (
            solution.is_contiguous()
            if compute_stage_metrics and solution.assignment
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
            "avg_reock_score": spatial.avg_reock_score if spatial else None,
            "avg_polsby_popper_score": spatial.avg_polsby_popper_score if spatial else None,
            "wall_time": solution.wall_time,
            "time_to_convergence": solution.time_to_convergence,
            "contiguous": contiguous,
            "num_nodes": solution.problem.A,
            "num_zones": solution.problem.Z,
            "metadata": dict(solution.metadata),
        }
        if solution.metadata.get("choice_utility") is not None:
            row["choice_utility"] = solution.metadata["choice_utility"]
        stage_rows.append(row)

        flat[f"objective_{name}"] = solution.objective
        flat[f"wall_time_{name}"] = solution.wall_time
        flat[f"time_to_convergence_{name}"] = solution.time_to_convergence
        if compute_stage_metrics:
            flat[f"cut_edges_{name}"] = row["cut_edges"]
            flat[f"normalized_cut_edges_{name}"] = row["normalized_cut_edges"]
            flat[f"avg_reock_score_{name}"] = row["avg_reock_score"]
            flat[f"avg_polsby_popper_score_{name}"] = row[
                "avg_polsby_popper_score"
            ]
        if level_counts[solution.level.name] == 1:
            flat[f"objective_{solution.level.name}"] = solution.objective
            flat[f"wall_time_{solution.level.name}"] = solution.wall_time
            flat[f"time_to_convergence_{solution.level.name}"] = (
                solution.time_to_convergence
            )
            if compute_stage_metrics:
                flat[f"cut_edges_{solution.level.name}"] = row["cut_edges"]
                flat[f"normalized_cut_edges_{solution.level.name}"] = row[
                    "normalized_cut_edges"
                ]
                flat[f"avg_reock_score_{solution.level.name}"] = row[
                    "avg_reock_score"
                ]
                flat[f"avg_polsby_popper_score_{solution.level.name}"] = row[
                    "avg_polsby_popper_score"
                ]

    final_stage = stage_rows[context.final_stage_index]
    if compute_stage_metrics:
        flat[MetricColumns.FINAL_CUT_EDGES] = final_stage["cut_edges"]

    run = {
        "strategy": _strategy_name(context),
        "selection": _selection_reason(context),
        "final_stage": context.final_stage_name,
        "final_stage_index": context.final_stage_index,
        "final_status": context.solution.status,
        "final_objective": context.solution.objective,
        "final_cut_edges": final_stage["cut_edges"],
        "total_wall_time": flat[MetricColumns.TOTAL_WALL_TIME],
        "time_to_convergence": flat[MetricColumns.TIME_TO_CONVERGENCE],
        "stages": stage_rows,
    }
    return MetricOutput(metrics=flat, run=run)


def _total_wall_time(context: MetricsContext) -> float:
    return sum(float(stage.wall_time or 0.0) for stage in context.stages)


def _time_to_convergence(context: MetricsContext) -> float | None:
    strategy = _strategy_name(context).lower()
    if "iterative" in strategy or context._is_iterative_run():
        return _stage_time_to_convergence(context.stages[0])
    if strategy == "recursive" or len(context.stages) > 1:
        total = 0.0
        for stage in context.stages:
            stage_time = _stage_time_to_convergence(stage)
            if stage_time is None:
                return None
            total += stage_time
        return total
    return _stage_time_to_convergence(context.solution)


def _stage_time_to_convergence(solution) -> float | None:
    if solution.time_to_convergence is None:
        return None
    return float(solution.time_to_convergence)


def _strategy_name(context: MetricsContext) -> str:
    if context.config.get("strategy"):
        return str(context.config["strategy"])
    if context._is_iterative_run():
        return "iterative"
    if len(context.stages) > 1:
        return "recursive"
    return "single"


def _selection_reason(context: MetricsContext) -> str:
    if context.solution.metadata.get("choice_utility") is not None:
        best = max(
            (
                stage.metadata.get("choice_utility")
                for stage in context.stages
                if stage.metadata.get("choice_utility") is not None
            ),
            default=None,
        )
        if best == context.solution.metadata.get("choice_utility"):
            return "best_choice_utility"
    return "last_solution_with_assignment"
