#!/usr/bin/env python3
"""Score every zoning a benchmarked run saved, on one common set of yardsticks.

The benchmark only stores a MID welfare where a strategy happens to optimize
one, so ``iterative_choice`` runs have none at all and the MID numbers that do
exist are different quantities (continuum vs. finite-grid vs. solver-market).
This script puts every saved zoning on one MID yardstick by replaying it through
``finite_grid_oracle`` - the same call ``mid`` uses to report
``mid_finite_grid_welfare`` and ``short_bursts_choice`` uses as its burst score.

It replays *every stage*, not only the zoning the run ended on. The iterative
strategies save one zoning per iteration, and the one they hand back is chosen
by their own surrogate; scoring all of them lets the comparison pick each run's
best zoning on a common measure instead of trusting that selection. Each stage
is scored on all three measures the comparison figure shows, so the row that
wins on budget welfare also carries the choice welfare and cut edges of that
same zoning:

* ``mid_welfare`` - discrete budget-set welfare from ``finite_grid_oracle``;
* ``logsum_welfare`` - exact MNL logsum, the same call ``metrics.choice`` makes;
* ``cut_edges`` - crossings of the ``Block_0`` adjacency graph, the same count
  ``metrics.spatial`` reports.

The market depends on the problem, not the zoning, so it is built once per
distinct problem configuration and reused across that configuration's runs.
Scoring a single zoning then costs milliseconds.

Zonings are transferred by stable census area id, not by graph node index. Node
indices are positional and depend on the KaHIP coarsening, which differs between
the machine that ran the benchmark and the machine reading it; the run's saved
area assignment records the zoning at base-block resolution, which is portable.
Each local coarse node takes the majority zone of its constituent blocks, and
``transfer_purity`` reports the share of blocks whose own zone survived that
vote - read the numbers as the benchmark's geography re-expressed on the local
coarsening, exact only where purity is 1.0. Cut edges are always counted on
``Block_0``, where the saved zoning needs no transfer at all.

Writes ``mid_welfare.csv`` with one row per saved stage. Validate against the
stored values with ``--validate``: the row for each run's selected stage is
compared to that run's own reported MID welfare, choice utility, and cut edges.
Expect a small residual whenever purity is below 1.0 - an exact match requires
running on the machine whose graph the benchmark used.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from choice.models import build_mnl_choice_model
from loaders import load_scenario
from optimization import run_archive
from optimization.config import OptimizationConfig
from optimization.data.mid import MidMarket, build_mid_market, preprocess_mid_market
from optimization.levels import LevelSpec
from optimization.mid_oracle import finite_grid_oracle
from optimization.problem import ZoneProblem
from optimization.solvers.recom_policy import ReComSettings

DEFAULT_BENCHMARK_DIR = (
    "/soalnas/share/data/school_choice/local_runs/benchmark_choice_v4"
)

# Config keys that steer the solver or the outer loop but not the MID market or
# the underlying problem, so runs differing only in these share one market.
SOLVER_ONLY_KEYS = frozenset(
    {
        "strategy", "solver", "seed", "solve_time_limits", "gap_limits", "workers",
        "carry_over_compute", "budget_accounting", "hints", "feasible_hint_time_limit",
        "save_solver_logs", "save_solver_progress", "linearization_level",
        "cp_model_probing_level", "symmetry_level", "cp_sat_search_strategy",
        "enumerated_solutions", "recom_iterations",
        *(f"recom_{f.name}" for f in fields(ReComSettings)),
        # Runs from before the ReCom overhaul carry these instead.
        "short_bursts_length", "short_bursts_method", "adaptive_short_bursts_lr",
        "adaptive_short_bursts_temperature", "max_iterations", "tolerance",
        "mid_transport_bounds", "mid_complementary_slackness",
        "mid_complementary_slackness_slack", "saa_num_seeds",
        "saa_tie_breaking_method", "secondary_objective", "choice_utility_hints",
        "choice_model_method", "choice_utility_scale",
        "priced_access_cut_levels", "priced_access_price_source",
        "priced_access_price_scale",
    }
)

# Of those, the ones the *scoring* still depends on, so they stay in the key.
SCORING_KEYS = frozenset({"choice_model_method"})

CONFIG_FIELDS = {f.name for f in fields(OptimizationConfig)}


@dataclass(frozen=True)
class ScoringBundle:
    """Per-configuration machinery for scoring any zoning of that problem."""

    market: MidMarket
    problem: ZoneProblem
    choice_model: Any
    # Cut edges are counted here, so the number does not depend on the level the
    # run solved at or on the local coarsening.
    block0_graph: Any


def config_from_snapshot(snapshot: dict[str, Any]) -> OptimizationConfig:
    """Rebuild the run's config from the snapshot stored in ``result.json``.

    The snapshot adds a derived ``unit`` key that is not a dataclass field, and
    may carry keys from other schema versions; both are dropped.
    """
    kwargs = {k: v for k, v in snapshot.items() if k in CONFIG_FIELDS}
    return OptimizationConfig(**kwargs)


def bundle_key(snapshot: dict[str, Any]) -> str:
    """Stable key over the config keys that shape the market or the scoring."""
    relevant = {
        k: v for k, v in sorted(snapshot.items())
        if k in CONFIG_FIELDS and (k not in SOLVER_ONLY_KEYS or k in SCORING_KEYS)
    }
    return json.dumps(relevant, sort_keys=True, default=str)


def build_bundle(config: OptimizationConfig, target: LevelSpec) -> ScoringBundle:
    """Build the MID market exactly as the MID strategies do, plus the scorers."""
    dataset = config.make_dataset()
    problem = dataset.problem_for(target)
    problem.overage = -1.0
    problem.shortage = -1.0
    problem.boundary_prop = float(config.boundary_prop)
    return ScoringBundle(
        market=preprocess_mid_market(
            build_mid_market(problem, dataset.config), problem
        ),
        problem=problem,
        choice_model=build_mnl_choice_model(
            load_scenario(config.data), method=config.choice_model_method
        ),
        block0_graph=dataset.graph_for(target.base()),
    )


def transfer_zoning(
    graph, area_zones: dict[int, int]
) -> tuple[dict[int, int], float, int]:
    """Map a base-block zoning onto this machine's coarse nodes by majority vote.

    Returns the node zoning, the share of blocks whose own zone won its node's
    vote, and the number of nodes that were not unanimous.
    """
    zoning: dict[int, int] = {}
    agreed = 0
    total = 0
    split_nodes = 0
    for node in graph:
        attrs = graph.nodes[node]
        area_ids = (
            [int(attrs["area_id"])]
            if "area_id" in attrs
            else [int(area_id) for area_id in attrs["block_ids"]]
        )
        votes: dict[int, int] = {}
        for area_id in area_ids:
            zone = area_zones.get(area_id)
            if zone is not None:
                votes[zone] = votes.get(zone, 0) + 1
        if not votes:
            raise ValueError(f"No saved zone covers any block of node {node}.")
        # Ties break on the lower zone id so the transfer is deterministic.
        winner = max(sorted(votes), key=lambda zone: votes[zone])
        zoning[node] = winner
        agreed += votes[winner]
        total += sum(votes.values())
        if len(votes) > 1:
            split_nodes += 1
    return zoning, agreed / total, split_nodes


def saved_stages(
    run_dir: Path, result: dict[str, Any], level: LevelSpec
) -> list[dict[str, Any]]:
    """Every stage of one run that saved a portable zoning at ``level``, in run order.

    The run's own ``final_stage_index`` marks the stage it handed back, which is
    not always its last: ``iterative_choice`` and ``saa`` both select among the
    iterations they ran.
    """
    selected = result["run"].get("final_stage_index")
    records = [
        {
            "name": stage["name"],
            "index": stage["index"],
            "level": str(level),
            "path": f"stages/{stage['name']}",
        }
        for stage in result["run"]["stages"]
        if (run_dir / "stages" / stage["name"] / f"zone_dict_{level}.json").exists()
    ]
    archive = run_archive.load_run_zonings(run_dir, records)
    stages: list[dict[str, Any]] = []
    for saved in archive["stages"]:
        if saved["level"] != str(level):
            continue
        area_zones = run_archive.area_assignment(archive, saved)
        if not area_zones:
            # A master that returned no feasible solution still writes the stage,
            # with an empty zoning. Skipping it keeps one such run from aborting
            # the whole scoring pass.
            continue
        stages.append(
            {
                "stage_index": int(saved["index"]),
                "stage_name": saved["name"],
                "is_selected": saved["index"] == selected,
                "stage_objective": saved.get("objective"),
                "stage_wall_time": saved.get("wall_time"),
                "area_zones": area_zones,
            }
        )
    legacy_alias = run_dir / f"zone_dict_area_{level}.json"
    if not stages and legacy_alias.exists():
        # Older runs saved only the root alias; treat it as the one stage.
        with legacy_alias.open() as handle:
            area_zones = {int(a): int(z) for a, z in json.load(handle).items()}
        if area_zones:
            stages.append(
                {
                    "stage_index": selected if selected is not None else 0,
                    "stage_name": result["run"].get("final_stage") or "final",
                    "is_selected": True,
                    "stage_objective": result["run"].get("final_objective"),
                    "stage_wall_time": None,
                    "area_zones": area_zones,
                }
            )
    return stages


def score_zoning(
    bundle: ScoringBundle, area_zones: dict[int, int], lottery_scale: int
) -> dict[str, Any]:
    """Score one saved zoning on budget welfare, choice welfare, and cut edges."""
    zoning, purity, split_nodes = transfer_zoning(bundle.problem.G, area_zones)
    start = time.perf_counter()
    oracle = finite_grid_oracle(bundle.market, zoning, lottery_scale)
    oracle_seconds = time.perf_counter() - start

    block_zoning, _, _ = transfer_zoning(bundle.block0_graph, area_zones)
    cut_edges = sum(
        block_zoning[u] != block_zoning[v] for u, v in bundle.block0_graph.edges()
    )
    return {
        "mid_welfare": float(oracle.welfare),
        "mid_fixed_point_welfare": float(oracle.fixed_point_welfare),
        "logsum_welfare": float(
            bundle.choice_model.preassignment_utility(bundle.problem, zoning)
        ),
        "cut_edges": int(cut_edges),
        "transfer_purity": purity,
        "split_nodes": split_nodes,
        "oracle_seconds": oracle_seconds,
    }


def stored_mid_welfare(result: dict[str, Any]) -> float | None:
    """The finite-grid MID welfare the run itself recorded, when it recorded one."""
    for stage in reversed(result["run"]["stages"]):
        value = stage["metadata"].get("mid_finite_grid_welfare")
        if value is not None:
            return float(value)
    return None


def recompute(
    benchmark_dir: Path,
    validate: bool,
    level_mode: str = "base",
    keep_failed: bool = False,
) -> pd.DataFrame:
    bundles: dict[str, ScoringBundle] = {}
    rows: list[dict[str, Any]] = []

    paths = sorted(benchmark_dir.rglob("result.json"))
    for index, path in enumerate(paths, start=1):
        with path.open() as handle:
            result = json.load(handle)
        if "stages" not in result.get("run", {}):
            # The optimization phase writes a metrics-free payload; a run whose
            # evaluation phase has not run yet has nothing to score.
            print(f"  skip (not yet evaluated): {path.parent.name}")
            continue
        if result.get("status") == "ERROR" and not keep_failed:
            # ``plot_choice_benchmark`` drops these, so scoring them is work
            # spent on rows nothing reads. Kept in step with that script on
            # purpose: the two disagreeing about what counts as a run is how a
            # zoning ends up in a figure that the figure's own loader excluded.
            print(f"  skip (errored run): {path.parent.name}")
            continue
        snapshot = result["config"]
        config = config_from_snapshot(snapshot)
        run_level = LevelSpec.parse(config.levels[-1])
        target = run_level.base() if level_mode == "base" else run_level

        stages = saved_stages(path.parent, result, run_level)
        if not stages:
            print(f"  skip (no area zone dict): {path.parent.name}")
            continue

        key = bundle_key(snapshot)
        if key not in bundles:
            start = time.perf_counter()
            bundles[key] = build_bundle(config, target)
            print(
                f"  built MID market #{len(bundles)} for {snapshot['centroids_type']} "
                f"in {time.perf_counter() - start:.1f} s"
            )
        bundle = bundles[key]

        stored = stored_mid_welfare(result)
        run = result["run"]
        print(
            f"[{index}/{len(paths)}] {snapshot['strategy']:20s} "
            f"{snapshot['centroids_type']} seed{snapshot['seed']}  "
            f"{len(stages)} stage(s)"
        )
        scored: list[dict[str, Any]] = []
        for stage in stages:
            measures = score_zoning(
                bundle, stage["area_zones"], int(config.mid_lottery_scale)
            )
            scored.append({**stage, **measures})
            print(
                f"    {stage['stage_name']:22s}"
                f"{'*' if stage['is_selected'] else ' '} "
                f"budget {measures['mid_welfare']:10,.1f}  "
                f"choice {measures['logsum_welfare']:10,.1f}  "
                f"cuts {measures['cut_edges']:5d}  "
                f"purity {measures['transfer_purity']:.4f} "
                f"({measures['oracle_seconds'] * 1000:.0f} ms)",
                flush=True,
            )
            rows.append(
                {
                    "path": str(path.parent.relative_to(benchmark_dir)),
                    "strategy": snapshot["strategy"],
                    "solver": snapshot["solver"],
                    "centroid": snapshot["centroids_type"],
                    "level": str(target),
                    "seed": snapshot["seed"],
                    "complementary_slackness": snapshot.get(
                        "mid_complementary_slackness", False
                    ),
                    "stage_index": stage["stage_index"],
                    "stage_name": stage["stage_name"],
                    "is_selected": stage["is_selected"],
                    "stage_objective": stage["stage_objective"],
                    "stage_wall_time": stage["stage_wall_time"],
                    **measures,
                    "stored_mid_finite_grid_welfare": stored,
                    "stored_logsum_welfare": run["choice_preassignment_utility"][
                        "utility"
                    ],
                    "stored_final_cut_edges": run.get("final_cut_edges"),
                }
            )
        best = max(scored, key=lambda stage: stage["mid_welfare"])
        selected = next((stage for stage in scored if stage["is_selected"]), None)
        gain = (
            best["mid_welfare"] - selected["mid_welfare"] if selected else float("nan")
        )
        print(
            f"    best on budget welfare: {best['stage_name']} "
            f"({best['mid_welfare']:,.1f}), "
            f"{gain:+,.1f} over the stage the run selected"
        )

    table = pd.DataFrame(rows)
    if validate and not table.empty:
        report_validation(table)
    return table


def report_validation(table: pd.DataFrame) -> None:
    """Check each run's selected stage against what that run itself reported."""
    checks = (
        ("mid_welfare", "stored_mid_finite_grid_welfare", "MID welfare"),
        ("logsum_welfare", "stored_logsum_welfare", "choice utility"),
        ("cut_edges", "stored_final_cut_edges", "cut edges"),
    )
    selected = table[table["is_selected"]]
    for column, stored_column, label in checks:
        compared = selected.dropna(subset=[stored_column])
        if compared.empty:
            print(f"\nNo run stored a {label} to validate against.")
            continue
        delta = (compared[column] - compared[stored_column]).abs()
        print(
            f"\nValidation of {label} against the run's own value "
            f"({len(compared)} run(s)): max |delta| = {delta.max():.6g}"
        )
        for _, row in compared.assign(delta=delta).iterrows():
            flag = "ok " if row["delta"] < 1e-6 else "MISMATCH"
            print(
                f"  {flag} {row['strategy']:20s} {row['centroid']} "
                f"seed{row['seed']} cs={row['complementary_slackness']!s:5s} "
                f"recomputed {row[column]:,.4f} vs stored "
                f"{row[stored_column]:,.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore every zoning a benchmarked run saved on one yardstick."
    )
    parser.add_argument(
        "benchmark_dir", nargs="?", default=DEFAULT_BENCHMARK_DIR,
        help=f"Benchmark directory (default: {DEFAULT_BENCHMARK_DIR})",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Destination CSV (default: analysis/plots/output/mid_welfare.csv)",
    )
    parser.add_argument(
        "--level", choices=("base", "run"), default="base",
        help=(
            "Level to evaluate on. 'base' (default) scores the exact saved zoning at "
            "base-block resolution, so nothing is lost in transfer. 'run' scores on "
            "the run's own coarse level, which requires a lossy majority-vote "
            "transfer unless this machine reproduces the benchmark's coarsening."
        ),
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Compare each run's selected stage against the values it reported",
    )
    parser.add_argument(
        "--keep-failed", action="store_true",
        help=(
            "Also score runs the benchmark marked ERROR. Skipped by default to "
            "match plot_choice_benchmark, which excludes them from every figure"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "output"
        / f"mid_welfare_{args.level}.csv"
    )
    table = recompute(
        Path(args.benchmark_dir), validate=args.validate, level_mode=args.level,
        keep_failed=args.keep_failed,
    )
    if table.empty:
        raise SystemExit("No zonings were scored.")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(
        f"\nwrote {output}  ({len(table)} stage rows over "
        f"{table['path'].nunique()} run(s))"
    )


if __name__ == "__main__":
    main()
