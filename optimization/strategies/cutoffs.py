"""Single-shot zoning strategy with a DA-STB cutoff objective."""

from __future__ import annotations

from optimization.data.cutoffs import build_cutoff_market
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register


@register("cutoffs")
class CutoffsStrategy(Strategy):
    """Solve the finest configured level with the school cutoff objective."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        if getattr(solver, "name", None) != "cp_bool":
            raise ValueError("cutoffs requires the cp_bool solver.")
        if list(dataset.config.years) != [23]:
            raise ValueError("cutoffs currently requires years: [23].")
        if dataset.config.population_type != "All":
            raise ValueError("cutoffs requires population_type: 'All'.")

        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        problem = dataset.problem_for(levels[-1])
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        problem.cutoff_market = build_cutoff_market(
            dataset,
            problem,
            assignment_config=self.options["cutoff_assignment_config"],
            ctip_path=self.options["cutoff_ctip_path"],
            lottery_scale=int(self.options["cutoff_lottery_scale"]),
            gumbel_scale=float(self.options["cutoff_gumbel_scale"]),
            preference_seed=int(self.options["cutoff_preference_seed"]),
            remove_city_wide=bool(self.options["remove_city_wide"]),
        )
        return [solver.solve(problem)]
