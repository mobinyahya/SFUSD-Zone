"""Build a sweep's shared data caches once, before its workers fan out.

When 40 task processes start together on one node, each loads the same
tables, graphs, and policy market at the same moment, and NFS contention
slows every one of them many times over. Warming each distinct key once
first means the rest load what the first built (the caches lock, so a key
another allocation is already building is waited for, not rebuilt), and at
the least they read files the node's page cache already holds.

A warm key is a dataset: the data block plus the settings that shape its
graphs. Warming one builds the dataset, each graph level its tasks read, and
each policy market (data block x ``matching_policy``) they need, all through
public entry points: ``optimization_config_from_dict(...).make_dataset()``,
``Dataset.graph_for(level)``, ``Dataset.problem_for`` and
``optimization.data.mid.build_mid_market``. Warming is best effort: a failure
is reported, and the tasks then build what they need themselves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from benchmark.config import BenchmarkTask, stable_hash

#: Config fields that, with ``data``, decide a dataset and its graphs.
DATASET_FIELDS = ("data", "partition_area_weight", "weight_edges")


def dataset_key(config: Mapping[str, Any]) -> str:
    """The warm key of a task config: its data block and graph settings."""
    return stable_hash({name: config.get(name) for name in DATASET_FIELDS})


def market_key(config: Mapping[str, Any]) -> str:
    """The policy-market key of a task config: data block x matching_policy."""
    return stable_hash(
        {"data": config.get("data"), "matching_policy": config.get("matching_policy")}
    )


def task_levels(config: Mapping[str, Any]) -> list[str]:
    """Every graph level a task config reads, steps first, in order."""
    levels: list[str] = []
    for step in config.get("steps") or []:
        level = step.get("level") if isinstance(step, Mapping) else None
        if level and level not in levels:
            levels.append(str(level))
    for level in config.get("levels") or []:
        if level not in levels:
            levels.append(str(level))
    return levels


@dataclass
class WarmSpec:
    """One dataset to warm, with the levels and policies its tasks need."""

    key: str
    config: dict[str, Any]
    levels: list[str] = field(default_factory=list)
    policies: list[str | None] = field(default_factory=list)


def warm_specs(configs: Iterable[Mapping[str, Any]]) -> list[WarmSpec]:
    """Group task configs into one :class:`WarmSpec` per dataset key."""
    specs: dict[str, WarmSpec] = {}
    for config in configs:
        key = dataset_key(config)
        spec = specs.get(key)
        if spec is None:
            spec = specs[key] = WarmSpec(key=key, config=dict(config))
        for level in task_levels(config):
            if level not in spec.levels:
                spec.levels.append(level)
        policy = config.get("matching_policy")
        if policy not in spec.policies:
            spec.policies.append(policy)
    return list(specs.values())


def warm_specs_for_tasks(tasks: Sequence[BenchmarkTask]) -> list[WarmSpec]:
    return warm_specs(task.config for task in tasks)


def warm_dataset(spec: WarmSpec) -> dict[str, Any]:
    """Build ``spec``'s dataset, graphs, and policy markets; time each part.

    Runs in its own process on an allocation (memory is the optimization's).
    Never raises: a part that fails is listed in ``errors``.
    """
    from benchmark.config import _reads_assignment_market, optimization_config_from_dict

    started = time.perf_counter()
    summary: dict[str, Any] = {
        "key": spec.key,
        "levels": {},
        "markets": {},
        "errors": [],
    }

    def timed(label: str, bucket: dict | None, function):
        start = time.perf_counter()
        try:
            value = function()
        except Exception as exc:  # Best effort; the task rebuilds it.
            summary["errors"].append(f"{label}: {exc.__class__.__name__}: {exc}")
            return None
        if bucket is not None:
            bucket[label] = round(time.perf_counter() - start, 2)
        return value

    config = timed(
        "config", None, lambda: optimization_config_from_dict(spec.config)
    )
    dataset = None
    if config is not None:
        start = time.perf_counter()
        dataset = timed("dataset", None, config.make_dataset)
        summary["dataset_seconds"] = round(time.perf_counter() - start, 2)
    if dataset is not None:
        for level in spec.levels:
            timed(level, summary["levels"], lambda: dataset.graph_for(level))
        for policy in spec.policies:
            timed(
                str(policy),
                summary["markets"],
                lambda: _warm_market(
                    dataset, spec, policy, optimization_config_from_dict,
                    _reads_assignment_market,
                ),
            )
    summary["seconds"] = round(time.perf_counter() - started, 2)
    return summary


def _warm_market(dataset, spec, policy, from_dict, reads_market) -> bool:
    """Build the policy market the task's first solve step builds."""
    config = from_dict({**spec.config, "matching_policy": policy})
    if not reads_market(config):
        return False
    from optimization.data.mid import build_mid_market
    from optimization.pipeline import step_config

    effective, level = config, spec.levels[0] if spec.levels else config.levels[0]
    steps = list(getattr(config, "steps", None) or [])
    if steps:
        effective, level = step_config(config, steps[0], 0), steps[0].level
    problem = dataset.problem_for(level, config=effective)
    build_mid_market(problem, effective)
    return True


def format_warm_summary(summary: Mapping[str, Any]) -> str:
    """One allocation-log line for a warmed dataset."""
    levels = ",".join(f"{k}={v}s" for k, v in summary.get("levels", {}).items())
    markets = ",".join(f"{k}={v}s" for k, v in summary.get("markets", {}).items())
    errors = summary.get("errors") or []
    return (
        f"warm key={summary.get('key', '?')[:12]} "
        f"total={summary.get('seconds', 0)}s "
        f"dataset={summary.get('dataset_seconds', '-')}s "
        f"graphs=[{levels}] markets=[{markets}]"
        + (f" errors={len(errors)}: {'; '.join(errors)[:500]}" if errors else "")
    )
