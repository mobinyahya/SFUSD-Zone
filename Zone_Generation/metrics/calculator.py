"""Optimization-native metrics calculator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.metrics import (
    choice,
    distance,
    diversity,
    programs,
    quality,
    run_metrics,
    structure,
)
from Zone_Generation.metrics.base import (
    MetricFn,
    MetricOutput,
    MetricsContext,
    MetricsResult,
)

DEFAULT_MODULES: tuple[MetricFn, ...] = (
    run_metrics.compute,
    diversity.compute,
    programs.compute,
    distance.compute,
    quality.compute,
    structure.compute,
    choice.compute,
)


class MetricsCalculator:
    """Compute modular metrics for one optimization run.

    ``solutions`` is the list returned by a optimization strategy. Single-shot runs
    pass one ``ZoneSolution``; recursive and iterative runs pass every stage so
    run-level metrics can analyze progression while final metrics are computed
    on the selected final solution.
    """

    def __init__(
        self,
        solutions: ZoneSolution | Sequence[ZoneSolution],
        config: Mapping[str, Any] | Any | None = None,
        *,
        final_solution: ZoneSolution | None = None,
        modules: Sequence[MetricFn] | None = None,
        strict: bool = True,
        compute_stage_metrics: bool = False,
    ):
        self.context = MetricsContext(
            solutions,
            config=config,
            final_solution=final_solution,
            compute_stage_metrics=compute_stage_metrics,
        )
        self.modules = tuple(modules or DEFAULT_MODULES)
        self.strict = strict

    def compute(self) -> MetricsResult:
        result = MetricsResult()
        errors = []
        for module in self.modules:
            if not self.context.solution.feasible and module is not run_metrics.compute:
                continue
            try:
                output = module(self.context)
            except Exception as exc:
                if self.strict:
                    raise
                errors.append({"module": _module_name(module), "error": str(exc)})
                continue
            if not isinstance(output, MetricOutput):
                raise TypeError(
                    f"Metric module {_module_name(module)} returned "
                    f"{type(output).__name__}, expected MetricOutput."
                )
            result.update(output)

        if errors:
            result.run.setdefault("metric_errors", errors)
        return result

    def compute_all(self) -> MetricsResult:
        """Alias for callers that want a calculator-style verb."""
        return self.compute()


def _module_name(module: MetricFn) -> str:
    return (
        getattr(module, "__module__", "")
        + "."
        + getattr(module, "__name__", repr(module))
    )


# Temporary name bridge for imports during the migration. The constructor is the
# optimization-only constructor above; legacy zone_dict/G inputs are intentionally not
# supported.
ZoneMetricsCalculator = MetricsCalculator
