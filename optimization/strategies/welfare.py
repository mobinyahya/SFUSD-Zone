"""Maximize expected utility at isolated student-optimal DA-STB outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from optimization.data.cutoffs import build_cutoff_market
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.solvers.budget_lbbd import BudgetSetLbbdSolver
from optimization.solvers.welfare import BooleanBudgetWelfareSolver, WelfareSolver
from optimization.solvers.welfare_decomposition import WelfareDecompositionSolver
from optimization.strategies.base import Strategy, register


@register("welfare")
class WelfareStrategy(Strategy):
    """Solve the finest level for stable finite-grid utilitarian welfare."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        problem = _build_isolated_welfare_problem(
            dataset,
            solver,
            self.options,
            strategy_name=self.name,
        )
        method = self.options.get("welfare_method", "decomposition")
        solver_class = {
            "budget": BooleanBudgetWelfareSolver,
            "decomposition": WelfareDecompositionSolver,
            "direct": WelfareSolver,
            "lbbd": BudgetSetLbbdSolver,
        }.get(method)
        if solver_class is None:
            raise ValueError(f"Unknown welfare_method: {method!r}.")
        solver_options = {"utility_scale": int(self.options["welfare_utility_scale"])}
        if method == "decomposition":
            solver_options["generate_assigned_pairs"] = self.options.get(
                "decomposition_generate_assigned_pairs", True
            )
            solver_options["prefix_depth"] = int(self.options["welfare_prefix_depth"])
            solver_options["round_time_limit"] = float(
                self.options.get("welfare_decomposition_round_time_limit", 180.0)
            )
            solver_options["theta_enabled"] = self.options.get(
                "welfare_decomposition_theta_enabled", True
            )
            solver_options["assignment_relaxation_enabled"] = self.options.get(
                "welfare_assignment_relaxation_enabled", True
            )
            solver_options["submodular_access_start_enabled"] = self.options.get(
                "welfare_submodular_access_start_enabled", False
            )
            solver_options["adjacent_zone_subset_improvement_enabled"] = (
                self.options.get(
                    "welfare_adjacent_zone_subset_improvement_enabled", False
                )
            )
            solver_options["pressure_starts_enabled"] = self.options.get(
                "decomposition_pressure_starts_enabled", False
            )
            solver_options["local_moves_enabled"] = self.options.get(
                "decomposition_local_moves_enabled", False
            )
            solver_options["recom_seed_runs"] = int(
                self.options.get("zoned_recom_seed_runs", 0)
            )
            solver_options["recom_time_limit"] = float(
                self.options.get("welfare_recom_time_limit", 600.0)
            )
            solver_options["branch_price_enabled"] = self.options.get(
                "welfare_branch_price_enabled", False
            )
            solver_options["branch_price_time_limit"] = float(
                self.options.get("welfare_branch_price_time_limit", 45.0)
            )
        return [
            solver_class(
                solver,
                **solver_options,
            ).solve(problem)
        ]


def _build_isolated_welfare_problem(
    dataset: Dataset,
    solver: Solver,
    options: dict,
    *,
    strategy_name: str,
):
    """Build the common year-23 finite-grid market and zoning problem."""
    if getattr(solver, "name", None) != "cp_bool":
        raise ValueError(f"{strategy_name} requires the cp_bool solver.")
    if list(dataset.config.years) != [23]:
        raise ValueError(f"{strategy_name} currently requires years: [23].")
    if dataset.config.population_type != "All":
        raise ValueError(f"{strategy_name} requires population_type: 'All'.")
    if not bool(options["remove_city_wide"]):
        raise ValueError(f"{strategy_name} currently requires remove_city_wide: true.")

    levels = [LevelSpec.parse(level) for level in options["levels"]]
    problem = dataset.problem_for(levels[-1])
    problem.boundary_prop = float(options.get("boundary_prop", -1.0))
    initial_assignment_path = options.get("welfare_initial_assignment_path")
    if initial_assignment_path:
        path = Path(initial_assignment_path).expanduser().resolve()
        with path.open(encoding="utf-8") as input_file:
            hint = {
                int(node): int(zone) for node, zone in json.load(input_file).items()
            }
        if set(hint) != set(problem.nodes):
            raise ValueError(
                "welfare_initial_assignment_path must assign every problem node."
            )
        invalid = {
            node: zone
            for node, zone in hint.items()
            if zone not in problem.candidate_zones(node)
        }
        if invalid:
            raise ValueError(
                "welfare_initial_assignment_path contains invalid node-zone "
                f"assignments: {invalid}."
            )
        problem.hint = hint
    problem.cutoff_market = build_cutoff_market(
        dataset,
        problem,
        assignment_config=options["cutoff_assignment_config"],
        ctip_path=options["cutoff_ctip_path"],
        lottery_scale=int(options["cutoff_lottery_scale"]),
        gumbel_scale=float(options["cutoff_gumbel_scale"]),
        preference_seed=int(options["cutoff_preference_seed"]),
        remove_city_wide=True,
        outside_option_utility=0.0,
    )
    return problem
