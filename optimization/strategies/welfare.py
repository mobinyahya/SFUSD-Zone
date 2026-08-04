"""Maximize expected utility at isolated student-optimal DA-STB outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from optimization.data.cutoffs import build_cutoff_market
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.solvers.welfare import WelfareSolver
from optimization.solvers.welfare_decomposition import WelfareDecompositionSolver
from optimization.strategies.base import Strategy, register


@register("welfare")
class WelfareStrategy(Strategy):
    """Solve the finest level for stable finite-grid utilitarian welfare."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        if getattr(solver, "name", None) != "cp_bool":
            raise ValueError("welfare requires the cp_bool solver.")
        if list(dataset.config.years) != [23]:
            raise ValueError("welfare currently requires years: [23].")
        if dataset.config.population_type != "All":
            raise ValueError("welfare requires population_type: 'All'.")
        if not bool(self.options["remove_city_wide"]):
            raise ValueError("welfare currently requires remove_city_wide: true.")

        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        problem = dataset.problem_for(levels[-1])
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        initial_assignment_path = self.options.get("welfare_initial_assignment_path")
        if initial_assignment_path:
            path = Path(initial_assignment_path).expanduser().resolve()
            with path.open(encoding="utf-8") as input_file:
                hint = {int(node): int(zone) for node, zone in json.load(input_file).items()}
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
            assignment_config=self.options["cutoff_assignment_config"],
            ctip_path=self.options["cutoff_ctip_path"],
            lottery_scale=int(self.options["cutoff_lottery_scale"]),
            gumbel_scale=float(self.options["cutoff_gumbel_scale"]),
            preference_seed=int(self.options["cutoff_preference_seed"]),
            remove_city_wide=True,
            outside_option_utility=0.0,
        )
        method = self.options.get("welfare_method", "decomposition")
        solver_class = {
            "decomposition": WelfareDecompositionSolver,
            "direct": WelfareSolver,
        }.get(method)
        if solver_class is None:
            raise ValueError(f"Unknown welfare_method: {method!r}.")
        solver_options = {"utility_scale": int(self.options["welfare_utility_scale"])}
        if method == "decomposition":
            solver_options["prefix_depth"] = int(self.options["welfare_prefix_depth"])
        return [
            solver_class(
                solver,
                **solver_options,
            ).solve(problem)
        ]
