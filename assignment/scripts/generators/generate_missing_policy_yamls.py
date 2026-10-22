"""Generate missing policy YAMLs for the 2026-05-08 robustness expansion.

Creates:
  - <zone>+reserves_06frl.yaml for 9 zones currently in SUBCONFIGS
  - distance_05_3+reserves{,_05frl,_06frl}.yaml
  - distance_05_1_3{,+reserves,+reserves_05frl,+reserves_06frl}.yaml

Idempotent: skips files that already exist.
"""

from pathlib import Path

# Resolve relative to the repo root (scripts/generators/<this file>) so the
# script works regardless of the current working directory or checkout location.
REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = REPO_ROOT / "configs" / "policy_configs"
GEN_DIR = POLICY_DIR / "generated"

ZONES_06FRL = [
    "Zones_6_FRL_Dev_0.10_Objective_1430.0_6-zone-3",
    "Zones_10_FRL_Dev_0.15_Objective_2150.0_10-zone-6",
    "Zones_6_FRL_Dev_0.10_Objective_1500.0_6-zone",
    "Zones_10_FRL_Dev_0.15_Objective_2270.0_10-zone-3",
    "Zones_10_FRL_Dev_0.20_Objective_1960.0_10-zone",
    "Zones_18_FRL_Dev_0.30_Objective_2820.0_18-zone-1",
    "Zones_18_FRL_Dev_0.30_Objective_2830.0_18-zone-6",
    "Zones_13_FRL_Dev_0.25_Objective_2500.0_13-zone-2",
    "Zones_10_FRL_Dev_0.15_Objective_2250.0_10-zone-6",
]

ZONE_06FRL_TPL = """assignment-algorithm: DA
ctip-options:
- 1
restrict-zone: true
guard-rails: 1
non_designation_boost: 128
policies:
- {zone}
priority-weights:
  ctip: 8
  sibling: 16
  zone: 4
  language-programs:
    lp-sibling: 16
    lp: 8
    sibling: 4
    ctip: 2
sibling-access: true
zone-building-blocks: block_group
designate: true
ties-options:
- MTB
soft_reserve_boost: 64
reserve-settings:
  column: freelunch_prob
  thresholds: percentile:60
  lower_disadvantaged: true
  reserve_fraction:
  - 0.6
  - 0.4
"""

DIST_TPL = """# Policy config for {name}
assignment-algorithm: DA
ctip-options:
- 1
restrict-zone: false
guard-rails: {guard_rails}
{reserve_block}non_designation_boost: 128
{soft_boost}
policies:
- Con1
priority-weights:
  ctip: 8
  sibling: 16
  zone: 0
  distance: 4
  language-programs:
    lp-sibling: 16
    lp: 8
    sibling: 4
    ctip: 2
distance-priority:
  thresholds: {thresholds}
sibling-access: true
designate: true
zone-building-blocks: 'attendance_area'
ties-options:
- MTB
"""

# Reserves blocks
RESERVES = """reserve-settings: {{
  "column": "median_hh_income",
  "thresholds": [95292],
  "lower_disadvantaged": true,
  "citywide_only": false,
  "reserve_fraction": [0.57, 0.43]
}}
"""
RESERVES_05FRL = """reserve-settings: {{
  "column": "freelunch_prob",
  "thresholds": "percentile:50",
  "lower_disadvantaged": true,
  "reserve_fraction": [0.5, 0.5]
}}
"""
RESERVES_06FRL = """reserve-settings: {{
  "column": "freelunch_prob",
  "thresholds": "percentile:60",
  "lower_disadvantaged": true,
  "reserve_fraction": [0.6, 0.4]
}}
"""


def write_if_missing(p: Path, content: str) -> None:
    if p.exists():
        print(f"  SKIP (exists): {p.name}")
        return
    p.write_text(content)
    print(f"  WROTE: {p.name}")


def gen_zone_06frl() -> None:
    print("=== Zone +reserves_06frl ===")
    for zone in ZONES_06FRL:
        p = GEN_DIR / f"{zone}+reserves_06frl.yaml"
        write_if_missing(p, ZONE_06FRL_TPL.format(zone=zone))


def gen_distance(name: str, thresholds: list, reserve_kind: str) -> None:
    if reserve_kind == "none":
        block = ""
        soft = ""
        gr = "-1"
    elif reserve_kind == "reserves":
        block = RESERVES
        soft = "soft_reserve_boost: 64"
        gr = "1"
    elif reserve_kind == "reserves_05frl":
        block = RESERVES_05FRL
        soft = "soft_reserve_boost: 64"
        gr = "1"
    elif reserve_kind == "reserves_06frl":
        block = RESERVES_06FRL
        soft = "soft_reserve_boost: 64"
        gr = "1"
    else:
        raise ValueError(reserve_kind)
    p = POLICY_DIR / f"{name}.yaml"
    content = DIST_TPL.format(
        name=name,
        thresholds=str(thresholds),
        reserve_block=block,
        soft_boost=soft,
        guard_rails=gr,
    )
    write_if_missing(p, content)


def gen_distance_variants() -> None:
    print("=== distance_05_3 reserve variants ===")
    for kind in ("reserves", "reserves_05frl", "reserves_06frl"):
        gen_distance(f"distance_05_3+{kind}", [0.5, 3], kind)
    print("=== distance_05_1_3 (+ variants) ===")
    gen_distance("distance_05_1_3", [0.5, 1, 3], "none")
    for kind in ("reserves", "reserves_05frl", "reserves_06frl"):
        gen_distance(f"distance_05_1_3+{kind}", [0.5, 1, 3], kind)


if __name__ == "__main__":
    gen_zone_06frl()
    gen_distance_variants()
