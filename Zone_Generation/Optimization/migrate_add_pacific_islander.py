"""
One-off migration: add Ethnicity_PacificIslander to existing graph pickles.

Existing BlockGroup_0/1/2.pickle files were built when AREA_ETHNICITIES had only
4 ethnicities (Black, Hispanic/Latinx, White, Asian). After adding
Ethnicity_PacificIslander to AREA_ETHNICITIES, the metric calculator expects
that attribute on every node and in G.graph['R']. This script patches existing
pickles in place so we don't have to re-solve any benchmarks.

Usage:
    uv run python -m Zone_Generation.Optimization.migrate_add_pacific_islander
"""
import os
import pickle
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# DesignZones loads centroids.yaml via the relative path "../Config/centroids.yaml",
# so we must run from Zone_Generation/Optimization.
os.chdir(os.path.join(REPO_ROOT, "Zone_Generation", "Optimization"))

from Zone_Generation.Config.Constants import AREA_ETHNICITIES, get_dropbox_path
from Zone_Generation.Optimization.design_zones import DesignZones


PI_KEY = "Ethnicity_PacificIslander"


def patch_block_group_0(G, dz):
    area_to_pi = dict(zip(
        dz.area_data["BlockGroup"].astype(int),
        dz.area_data[PI_KEY].astype(float),
    ))
    missing = []
    total_pi = 0.0
    total_students = 0.0
    for idx, attrs in G.nodes(data=True):
        area_id = attrs["area_id"]
        if area_id in area_to_pi:
            pi = float(area_to_pi[area_id])
        else:
            pi = 0.0
            missing.append(area_id)
        attrs[PI_KEY] = pi
        total_pi += pi
        total_students += attrs["ge_students"]
    G.graph["R"] = {eth: 0.0 for eth in AREA_ETHNICITIES}
    if total_students > 0:
        G.graph["R"][PI_KEY] = total_pi / total_students
        for eth in AREA_ETHNICITIES:
            if eth == PI_KEY:
                continue
            s = sum(G.nodes[n][eth] for n in G.nodes)
            G.graph["R"][eth] = s / total_students
    if missing:
        print(f"  WARNING: {len(missing)} BG_0 area_ids not found in source data (set PI=0)")
    print(f"  BG_0 district PI proportion: {G.graph['R'][PI_KEY]:.4f}")


def patch_aggregated(G, base_G):
    partition = G.graph.get("partition")
    if partition is None:
        raise RuntimeError("Aggregated graph missing G.graph['partition']")
    pi_per_part: dict = {}
    for base_node, part_id in partition.items():
        pi_per_part[part_id] = pi_per_part.get(part_id, 0.0) + float(
            base_G.nodes[base_node][PI_KEY]
        )
    total_pi = 0.0
    total_students = 0.0
    for part_id, attrs in G.nodes(data=True):
        attrs[PI_KEY] = pi_per_part.get(part_id, 0.0)
        total_pi += attrs[PI_KEY]
        total_students += attrs["ge_students"]
    G.graph["R"] = dict(base_G.graph["R"])
    print(f"  Aggregated district PI proportion: {G.graph['R'][PI_KEY]:.4f}")


def main():
    with open(os.path.join(REPO_ROOT, "Zone_Generation", "Config", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    config["level"] = "BlockGroup"
    is_local = config.get("is_local", False)

    print("Loading DesignZones to get Pacific Islander counts per BlockGroup...")
    dz = DesignZones(config=config)
    if PI_KEY not in dz.area_data.columns:
        raise RuntimeError(
            f"DesignZones.area_data missing column {PI_KEY}. "
            f"Check that AREA_ETHNICITIES includes it."
        )

    graphs_dir = f"{get_dropbox_path(is_local)}/Optimization/Zones/Graphs"

    # --- BlockGroup_0 ---
    bg0_path = f"{graphs_dir}/BlockGroup_0.pickle"
    print(f"\nPatching {bg0_path}")
    with open(bg0_path, "rb") as f:
        bg0 = pickle.load(f)
    patch_block_group_0(bg0, dz)
    with open(bg0_path, "wb") as f:
        pickle.dump(bg0, f)

    # --- BlockGroup_1 and BlockGroup_2 (both aggregated from BG_0) ---
    for level in (1, 2):
        path = f"{graphs_dir}/BlockGroup_{level}.pickle"
        print(f"\nPatching {path}")
        with open(path, "rb") as f:
            G = pickle.load(f)
        patch_aggregated(G, bg0)
        with open(path, "wb") as f:
            pickle.dump(G, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
