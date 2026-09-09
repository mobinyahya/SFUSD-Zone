#!/usr/bin/env python3
"""Compare the SAA bound-tightening arms on what each one actually moved.

Four arms over {first_choice, transport} x {aggregated, disaggregated} cuts. The
numbers that matter are the reported upper bound (lower is tighter), the
incumbent welfare (higher is better, and must not regress), and the gap between
them. The a-priori constant is printed alongside because on these instances the
solver never improves on it, so the two are usually the same number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        if config["strategy"] != "saa":
            continue
        stages = result["run"]["stages"]
        final = stages[-1]["metadata"]
        # The master's own dual bound per iteration, before the loop takes the min.
        iteration_bounds = [
            stage["metadata"].get("choice_best_bound")
            for stage in stages
            if stage["metadata"].get("choice_best_bound") is not None
        ]
        declared = final.get("saa_welfare_upper_bound")
        rows.append(
            {
                "centroid": config["centroids_type"],
                "bound": config.get("saa_welfare_bound", "first_choice"),
                "cuts": (
                    "disaggregated"
                    if config.get("saa_disaggregate_cuts")
                    else "aggregated"
                ),
                "card": bool(config.get("choice_access_cardinality", False)),
                "tri": bool(config.get("choice_access_triangle", False)),
                # How many valid inequalities the solver actually found to add.
                "cliques_added": stages[0]["metadata"].get("choice_access_cardinality"),
                "triples_added": stages[0]["metadata"].get("choice_access_triangle"),
                "access_pairs": stages[0]["metadata"].get("choice_access_pairs"),
                "declared_constant": declared,
                "first_choice_constant": final.get("saa_first_choice_upper_bound"),
                "reported_upper_bound": final.get("saa_global_upper_bound"),
                "incumbent": final.get("saa_incumbent_welfare"),
                "gap": final.get("saa_absolute_gap"),
                "logsum_welfare": result["run"]["choice_preassignment_utility"][
                    "utility"
                ],
                "iterations": final.get("saa_iteration_count"),
                "termination": final.get("saa_termination_reason"),
                # Did the solver ever prove anything better than the constant?
                "best_iteration_bound": min(iteration_bounds) if iteration_bounds else None,
                "cuts_total": final.get("saa_cuts_total"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Experiment output directory.")
    parser.add_argument("-o", "--output", default=None, help="Optional CSV path.")
    args = parser.parse_args()

    table = load(Path(args.root))
    if table.empty:
        raise SystemExit("No SAA results found.")
    table["gap_pct"] = 100.0 * table["gap"] / table["incumbent"]
    table["bound_vs_first_choice"] = (
        table["declared_constant"] - table["first_choice_constant"]
    )
    table = table.sort_values(["centroid", "bound", "cuts", "card", "tri"])

    columns = [
        "centroid", "bound", "cuts", "card", "tri", "declared_constant",
        "reported_upper_bound", "incumbent", "gap", "gap_pct", "logsum_welfare",
        "cliques_added", "triples_added", "access_pairs", "termination",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[columns].to_string(index=False))

    print("\nDid any solver beat the constant it was handed?")
    for _, row in table.iterrows():
        slack = row["declared_constant"] - row["best_iteration_bound"]
        verdict = f"yes, by {slack:,.2f}" if slack > 1e-6 else "no"
        print(
            f"  {row['centroid']} {row['bound']:<13s} {row['cuts']:<14s} "
            f"card={str(row['card']):<5s} tri={str(row['tri']):<5s} "
            f"constant {row['declared_constant']:,.2f} -> best bound "
            f"{row['best_iteration_bound']:,.2f}   {verdict}"
        )

    # The baseline is whichever arm has every switch off; if the experiment
    # holds one switch on throughout, fall back to the plainest arm present.
    plain = table[(table["cuts"] == "aggregated") & ~table["card"] & ~table["tri"]]
    if (plain["bound"] == "first_choice").any():
        plain = plain[plain["bound"] == "first_choice"]
    baseline = plain.drop_duplicates("centroid").set_index("centroid")
    label = (
        baseline["bound"].iloc[0] if len(baseline) else "?"
    )
    print(f"\nChange against the {label} + aggregated, no-inequalities baseline:")
    for _, row in table.iterrows():
        if row["centroid"] not in baseline.index:
            continue
        base = baseline.loc[row["centroid"]]
        print(
            f"  {row['centroid']} {row['bound']:<13s} {row['cuts']:<14s} "
            f"card={str(row['card']):<5s} tri={str(row['tri']):<5s} "
            f"bound {row['reported_upper_bound'] - base['reported_upper_bound']:+9.2f}  "
            f"incumbent {row['incumbent'] - base['incumbent']:+9.2f}  "
            f"gap {row['gap'] - base['gap']:+9.2f}  "
            f"logsum {row['logsum_welfare'] - base['logsum_welfare']:+9.2f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
