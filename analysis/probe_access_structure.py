#!/usr/bin/env python3
"""How much conflict structure do the co-zoning indicators actually have?

The triangle and cardinality families can only bind where the access pairs
supply them with something to attach to: transitivity needs triples whose three
pairs are all present, and a cardinality cap needs neighbours that are pairwise
unable to share a zone. This measures both directly, plus how much a greedy
clique cover discards relative to constraining every conflicting pair, so a
weak result can be attributed to the geometry rather than to the encoding.

Needs one oracle call and no master solves.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from benchmark.slurm import create_plan
from optimization.access_inequalities import (
    cardinality_cliques,
    pair_key,
    transitivity_triples,
)
from optimization.data.initial_solutions import initial_solution
from optimization.data.saa import build_saa_market, sample_school_preferences
from optimization.levels import LevelSpec
from optimization.saa_oracle import SaaOracle
from optimization.strategies.saa import _configure_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--centroids", default="6-zone-3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = create_plan(args.config)
    config = plan.tasks[0].optimization_config()
    config.centroids_type = args.centroids
    options = config.make_strategy().options
    solver_options = dict(config.make_solver().options)
    target = LevelSpec.parse(config.levels[-1])
    dataset = config.make_dataset()
    problem = dataset.problem_for(target)
    _configure_problem(problem, options)

    zones = {node: set(problem.candidate_zones(node)) for node in problem.nodes}
    sizes = Counter(len(z) for z in zones.values())
    print(f"nodes {len(zones)}  zones {problem.Z}", flush=True)
    print(f"candidate-zone set sizes: {dict(sorted(sizes.items()))}", flush=True)

    market = build_saa_market(problem, dataset.config)
    sample = sample_school_preferences(
        market, 1, str(options.get("saa_tie_breaking_method", "MTB")).upper(),
        int(options.get("seed", 42)),
    )[0]
    oracle = SaaOracle(market, sample, 0, problem, workers=args.workers)
    hint = initial_solution(
        problem, options.get("hints", "voronoi"), solver_options=solver_options
    )
    cut = oracle.solve(hint.assignment).cut

    all_pairs = {pair_key(*pair) for pair, _ in cut.coefficients if pair[0] != pair[1]}
    # A pair is a variable only where the two nodes share at least one zone;
    # otherwise the solvers already fix the indicator to zero.
    variable_pairs = sorted(p for p in all_pairs if zones[p[0]] & zones[p[1]])
    constant_pairs = sorted(all_pairs - set(variable_pairs))

    neighbours: dict[int, set[int]] = {}
    for first, second in variable_pairs:
        neighbours.setdefault(first, set()).add(second)
        neighbours.setdefault(second, set()).add(first)

    # Every conflicting neighbour pair is one valid "a[u,x] + a[u,y] <= 1".
    conflicting = 0
    anchors_with_conflict = 0
    for anchor, others in neighbours.items():
        local = sum(
            1
            for x, y in combinations(sorted(others), 2)
            if not (zones[x] & zones[y])
        )
        conflicting += local
        anchors_with_conflict += local > 0

    cliques = cardinality_cliques(variable_pairs, problem.candidate_zones)
    covered_pairs = sum(len(members) * (len(members) - 1) // 2 for _, members in cliques)
    triples = list(transitivity_triples(variable_pairs))

    payload = {
        "centroids_type": args.centroids,
        "nodes": len(zones),
        "num_zones": problem.Z,
        "candidate_zone_set_sizes": {str(k): v for k, v in sorted(sizes.items())},
        "cut_pairs": len(all_pairs),
        "variable_pairs": len(variable_pairs),
        "constant_pairs": len(constant_pairs),
        "anchors": len(neighbours),
        "anchors_with_a_conflicting_neighbour_pair": anchors_with_conflict,
        "conflicting_neighbour_pairs": conflicting,
        "greedy_cliques": len(cliques),
        "pairs_covered_by_greedy_cliques": covered_pairs,
        "complete_triangles": len(triples),
    }
    for key, value in payload.items():
        print(f"  {key}: {value}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
