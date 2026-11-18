#!/usr/bin/env python3
"""Combine benchmark_methods_2324 with its MIP and multilevel reruns.

benchmark_methods_2324 holds 2,100 runs. Two later sweeps reran parts of it
under the same task ids (their fixes were written to keep config hashes):

    benchmark_methods_2324_mipfix   the 600 MIP tasks, after the partial-start
                                    fix (267c70c)
    benchmark_methods_2324_release  the 600 multilevel MIP and CP tasks, after
                                    the refine releases (e10e8a0, which
                                    contains 267c70c)

The combined folder copies each run from its newest source: release, then
mipfix, then the base sweep. So single-level MIP comes from mipfix,
multilevel MIP and CP from release, and ReCom and single-level CP from the
base. ``provenance.json`` records each run's source, since the shared hashes
cannot. The sources are never modified.

    uv run python analysis/benchmark_methods_2324/combine_sweeps.py
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

from benchmark.results import aggregate_results, discover_run_dirs
from benchmark.runner import load_manifest

LOCAL_RUNS = Path("/soalnas/share/data/school_choice/local_runs")
BASE = "benchmark_methods_2324"
# Newest first: a task found in an earlier source wins.
OVERRIDES = ("benchmark_methods_2324_release", "benchmark_methods_2324_mipfix")
COMBINED = "benchmark_methods_2324_combined"


def _runs(root: Path) -> dict[str, Path]:
    """Map each run's config hash to its folder relative to ``root``."""
    runs = {}
    for run_dir in discover_run_dirs(str(root)):
        manifest = load_manifest(run_dir)
        runs[manifest["config_hash"]] = Path(run_dir).relative_to(root)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local-runs", type=Path, default=LOCAL_RUNS)
    parser.add_argument("--output", default=COMBINED)
    args = parser.parse_args()

    root = args.local_runs
    output = root / args.output
    if output.exists():
        raise SystemExit(f"{output} already exists; remove it first.")

    base = _runs(root / BASE)
    overrides = {name: _runs(root / name) for name in OVERRIDES}
    for name, runs in overrides.items():
        unknown = set(runs) - set(base)
        if unknown:
            raise SystemExit(f"{name} has {len(unknown)} tasks the base sweep lacks.")

    provenance = {}
    for config_hash, rel in sorted(base.items(), key=lambda item: str(item[1])):
        source = next((n for n in OVERRIDES if config_hash in overrides[n]), BASE)
        if source != BASE and overrides[source][config_hash] != rel:
            raise SystemExit(f"{config_hash}: {source} saved it at another path.")
        shutil.copytree(root / source / rel, output / rel, symlinks=True)
        provenance[str(rel)] = {"config_hash": config_hash, "source": source}

    counts = collections.Counter(entry["source"] for entry in provenance.values())
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "note": __doc__.split("\n\n")[1].replace("\n", " "),
                "counts": counts,
                "runs": provenance,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    runs, stages = aggregate_results(
        str(output), summary_csv="summary.csv", stages_csv="stages.csv"
    )
    print(dict(counts))
    print(f"{len(runs)} runs, {len(stages)} stage rows under {output}")


if __name__ == "__main__":
    main()
