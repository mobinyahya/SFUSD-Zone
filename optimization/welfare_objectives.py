"""Reuse welfare cut loops as objectives of one fully specified problem.

The loops can perform several master solves internally. The pipeline still
receives one evaluated incumbent per step, with iteration diagnostics retained.
"""

from __future__ import annotations

from dataclasses import replace
import time

from optimization.solution import ZoneSolution


class _ProblemDataset:
    """A solve-local view that preserves the pipeline's domains and assignments."""

    def __init__(self, problem, config):
        self.problem = problem
        self.config = config
        self.data = config.data_scenario

    def problem_for(self, level, **kwargs):
        if str(level) != str(self.problem.level):
            raise ValueError("A welfare objective cannot change its step's level.")
        # A loop's absent hint means keep the explicit pipeline input. Fixed
        # assignments/candidates always stay on every internal master problem.
        if kwargs.get("hint") is None:
            kwargs.pop("hint", None)
        return replace(self.problem, **kwargs)


def solve_welfare_objective(problem, method, objective, options):
    """Return the best evaluated welfare, never the master's optimistic bound."""
    config = problem.optimization_config
    if config is None:
        raise ValueError(f"{objective} requires the originating optimization_config.")
    from optimization.solvers import get_solver

    strategy_name = "saa" if objective == "saa_welfare" else "iterative_choice"
    config = replace(
        config, strategy=strategy_name, _resolved_data_scenario=config.data_scenario
    )
    strategy = config.make_strategy()
    # The pipeline owns the feasible set. Welfare does not implicitly switch
    # off the run's capacity bands, as legacy standalone welfare strategies do.
    strategy.capacity_balance = True
    strategy.options["solve_time_limits"] = [options["solve_time_limit"]]
    strategy.options["gap_limits"] = [options.get("relative_gap_limit", 0.0)]
    dataset = _ProblemDataset(problem, config)
    started = time.perf_counter()
    stages = strategy.run(dataset, get_solver(method, **options))
    elapsed = time.perf_counter() - started
    score_key = "saa_welfare" if objective == "saa_welfare" else "choice_utility"
    feasible = [
        s for s in stages if s.feasible and s.metadata.get(score_key) is not None
    ]
    selected = (
        max(reversed(feasible), key=lambda s: s.metadata[score_key])
        if feasible
        else stages[-1]
        if stages
        else None
    )
    if selected is None:
        return ZoneSolution(
            problem,
            {},
            "UNKNOWN",
            wall_time=elapsed,
            metadata={"objective_kind": objective, "objective_iterations": []},
        )
    metadata = dict(selected.metadata)
    if stages:
        for key, value in stages[-1].metadata.items():
            if key.endswith(("termination_reason", "iteration_count")):
                metadata[key] = value
    metadata.update(
        objective_kind=objective,
        aggregate_capacity_overage_disabled=problem.overage < 0,
        aggregate_capacity_shortage_disabled=problem.shortage < 0,
        objective_iterations=[
            {
                "status": stage.status,
                "objective": stage.objective,
                "evaluated_objective": stage.metadata.get(score_key),
                "wall_time": stage.wall_time,
            }
            for stage in stages
        ],
    )
    status = selected.status
    if feasible:
        status = (
            "OPTIMAL"
            if objective == "saa_welfare"
            and stages[-1].metadata.get("saa_certified_optimal")
            else "FEASIBLE"
        )
    return replace(
        selected,
        status=status,
        objective=selected.metadata.get(score_key) if feasible else None,
        wall_time=elapsed,
        metadata=metadata,
    )
