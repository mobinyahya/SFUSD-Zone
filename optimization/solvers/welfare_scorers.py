"""Cached welfare evaluators shared by all ReCom search methods.

Each evaluator prepares its market once. SAA additionally keeps its recourse
LPs alive, letting Gurobi warm-start consecutive evaluations. Search methods
consume only the batch-score contract and know nothing about either model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class CachedWelfareScorer(ABC):
    def __init__(self):
        self.cache: dict[tuple[tuple[int, int], ...], float] = {}

    def __call__(self, assignments, base=None) -> tuple[float, ...]:
        scores = []
        for assignment in assignments:
            key = tuple(sorted(assignment.items()))
            if key not in self.cache:
                self.cache[key] = self.score(dict(assignment))
            scores.append(self.cache[key])
        return tuple(scores)

    @abstractmethod
    def score(self, assignment: dict[int, int]) -> float: ...

    def close(self) -> None:
        pass


class ChoiceWelfareScorer(CachedWelfareScorer):
    def __init__(self, problem, options: Mapping[str, Any]):
        from choice.models import build_mnl_choice_model

        super().__init__()
        config = problem.optimization_config
        if config is None:
            raise ValueError(
                "choice_welfare requires the originating optimization_config."
            )
        self.problem = problem
        self.method = options.get("choice_model_method", "logsum")
        self.model = build_mnl_choice_model(config.data_scenario, method=self.method)

    def score(self, assignment: dict[int, int]) -> float:
        return float(self.model.evaluate(self.problem, assignment))

    def describe(self, assignment) -> dict:
        return {
            "choice_model_method": self.method,
            "choice_utility": self((assignment,))[0],
        }


class SaaWelfareScorer(CachedWelfareScorer):
    def __init__(self, problem, options: Mapping[str, Any]):
        from optimization.data.saa import build_saa_market, sample_school_preferences
        from optimization.saa_oracle import SaaOracle

        super().__init__()
        config = problem.optimization_config
        if config is None:
            raise ValueError(
                "saa_welfare requires the originating optimization_config."
            )
        if problem.program_population != "All":
            raise ValueError("saa_welfare requires program_population='All'.")
        market = build_saa_market(problem, config)
        if market.utility_student_count == 0:
            raise ValueError("The saa_welfare market has no choice-utility rows.")
        self.tie_breaking_method = options.get("saa_tie_breaking_method", "MTB")
        self.samples = sample_school_preferences(
            market,
            options.get("saa_num_seeds", 5),
            self.tie_breaking_method,
            options.get("seed", 42),
        )
        self.oracles = []
        try:
            for index, sample in enumerate(self.samples):
                self.oracles.append(
                    SaaOracle(
                        market,
                        sample,
                        index,
                        problem,
                        workers=options.get("workers", 1),
                    )
                )
        except BaseException:
            self.close()
            raise

    def score(self, assignment: dict[int, int]) -> float:
        return sum(oracle.solve(assignment).welfare for oracle in self.oracles) / len(
            self.oracles
        )

    def describe(self, assignment) -> dict:
        return {
            "saa_welfare": self((assignment,))[0],
            "saa_num_seeds": len(self.samples),
            "saa_sample_seeds": [sample.seed for sample in self.samples],
            "saa_tie_breaking_method": self.tie_breaking_method,
        }

    def close(self) -> None:
        for oracle in self.oracles:
            oracle.model.dispose()
        self.oracles.clear()
