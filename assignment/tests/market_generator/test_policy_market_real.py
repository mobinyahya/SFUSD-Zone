"""The shared policy market reproduces the simulator's own market exactly.

For each 11-18 policy, on today's attendance areas and on a generated zoning
(pairs of attendance areas joined at the Census-block level), the simulator
builds its preferences, priorities, zone eligibility, reserves and match
twice: through the shared ``PolicyMarket`` and through the generators' own
code. Everything must agree bit for bit.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from assignment.generated_zones import resolve_generated_zone_batch_config
from assignment.run_custom_config import load_custom_config
from assignment.student_assignment.configerator import Configerator
from assignment.student_assignment.da.guardrail_setup import GuardrailSetup
from assignment.student_assignment.market_generator import policy_market as pm
from assignment.student_assignment.market_generator.preference_generator import (
    PreferenceGenerator,
)
from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)

REPOSITORY = Path(__file__).resolve().parents[3]
CONFIG = REPOSITORY / "assignment/configs/attendance_areas/restrictive_aa_sib_2627.yaml"
POLICIES = [
    "restrictive_aa_sib_ctip",
    "restrictive_aa_sib_ctip_reserves",
    "restrictive_aa_sib_no_ctip",
    "restrictive_aa_sib_no_ctip_reserves",
]
DATA = {
    # The 11-18 sweep's market and the lottery-DA replay test's.
    "webster_enrolled_all": {
        "capacity_scenario": "ideal_webster_w2w_10_23_26",
        "student_population": "enrolled_all",
    },
    "nov13_applied_or_enrolled": {
        "capacity_scenario": "ideal_baseline_nov_13_2026",
        "student_population": "applied_or_enrolled",
    },
}


def _data(filters):
    return {
        "scenario": "attendance-areas-2627",
        "overrides": {"filters": {"optimization": filters, "assignment": filters}},
    }


def _attendance_area_configs(base):
    configurator = Configerator.from_config(base)
    configs = {}
    for name in POLICIES:
        configurator.load_subconfig_by_name(name)
        config = copy.deepcopy(configurator.config)
        config["subconfigs"] = []
        configs[name] = config
    return configs


def _paired_zone_file(market, path: Path) -> Path:
    """Join attendance areas two by two (in school order) into block zones."""
    schools = market.zones.aa_schools
    pair = {int(s): i // 2 for i, s in enumerate(sorted(schools["school_id"]))}
    zone_of = {}
    frame = market.students.student_data
    for block, area in zip(frame["census_block"], frame["idschoolattendance"]):
        if pd.notna(block) and pd.notna(area) and int(area) in pair:
            zone_of.setdefault(int(block), pair[int(area)])
    for school, block in zip(schools["school_id"], schools["Block"]):
        if pd.notna(block):
            zone_of[int(block)] = pair[int(school)]
    # Every student must live in some zone (GuardrailSetup maps them all).
    for block in frame["census_block"].dropna():
        zone_of.setdefault(int(block), 0)
    zones = {}
    for block, zone in zone_of.items():
        zones.setdefault(zone, []).append(block)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for zone in sorted(zones):
            writer.writerow(sorted(zones[zone]))
    return path


def _simulate(market, *, shared: bool):
    """One lottery of the active policy, from a clean generator state."""
    if shared:
        market.__dict__.pop("policy_market", None)
    else:
        market.policy_market = lambda: None  # the generators' own code
    market._reset_zones()
    market.priority_generator = PriorityGenerator(market)
    market.preference_generator = PreferenceGenerator(market)
    market._guardrail_setup_cache.clear()
    zone_policy = market.config["policies"][0]
    market.priority_generator.generate_base_priorities(zone_policy)
    prefs = market.preference_generator.initialize_real_preferences(
        designate=market.config["designate"]
    )
    pref_length = np.asarray(market.preference_generator.pref_length).copy()
    policy = market._policy_data_options(zone_policy)[0]
    base = market.priority_generator._set_policy_priorities(policy.ctip, policy.name)
    without = market.priority_generator.get_priorities_without_lottery(policy, prefs)
    np.random.seed(market.iteration_seed(market.config["random-seed"], 0))
    priorities = market.priority_generator.set_policy_specific_priorities(
        policy, prefs, iteration=0
    )
    result = {
        "prefs": prefs,
        "pref_length": pref_length,
        "base": base.copy(),
        "without_lottery": without,
        "priorities": priorities,
        "eligibility": np.asarray(market.zones.zone_eligibility_matrix).copy(),
    }
    if market.config["guard-rails"] != -1:
        match = market._generate_assignment_with_guardrails(prefs, priorities)
    else:
        match = market._generate_assignment(prefs, priorities)
    result["match"], result["rank"] = match[0], match[1]
    market.__dict__.pop("policy_market", None)
    return result


def _assert_identical(shared, own, label):
    for key, expected in own.items():
        actual = shared[key]
        assert actual.dtype == expected.dtype, (label, key)
        np.testing.assert_array_equal(actual, expected, err_msg=f"{label}: {key}")


def _assert_reserves_match(market, policy_market, prefs, label):
    guard = GuardrailSetup(market, prefs)
    guard._create_categories(market.config["reserve-settings"])
    guard._set_up_program_reserves(market.config["reserve-settings"])
    np.testing.assert_array_equal(
        guard.classOfStudent, policy_market.reserve_classes, err_msg=label
    )
    assert guard.num_classes == policy_market.num_classes
    for j, program_id in enumerate(policy_market.program_ids):
        if program_id.split("-")[1] != "GE":
            continue
        expected = guard.program_reserve_frac[j]
        np.testing.assert_array_equal(
            policy_market.reserve_fractions.get(program_id, np.zeros_like(expected)),
            expected,
            err_msg=f"{label}: {program_id}",
        )


@pytest.mark.real_data
@pytest.mark.parametrize("data_name", list(DATA))
def test_policy_market_reproduces_the_simulator(data_name, tmp_path):
    from assignment.student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    data = _data(DATA[data_name])
    try:
        base = load_custom_config(
            CONFIG, assignment_folder=tmp_path, absolute_assignment_folder=True
        )
        base["data"] = data
        configs = {"attendance_areas": _attendance_area_configs(base)}
        market = MarketGenerator(
            config=configs["attendance_areas"][POLICIES[0]],
            assignment_path=str(tmp_path),
            write_config=False,
            write_aggregate_metrics=False,
        )
    except FileNotFoundError as error:
        pytest.skip(f"real data not available: {error}")

    # The designation ordering's vectorised argsort is the row-by-row one.
    distances = market.students.distance_data
    lp_columns, lp_order, ge_columns, ge_order = pm.distance_orderings(distances)
    for columns, order in ((lp_columns, lp_order), (ge_columns, ge_order)):
        frame = distances[columns]
        expected = frame.apply(
            lambda row: pd.Series(frame.columns[np.argsort(row.values)]), axis=1
        )
        np.testing.assert_array_equal(
            np.asarray(columns, dtype=object)[order], expected.to_numpy()
        )

    zone_file = _paired_zone_file(market, tmp_path / "paired_zones.csv")
    target = {
        "id": "paired",
        "zone_file": str(zone_file),
        "zone_building_blocks": "block",
        "data": data,
    }
    _, resolved = resolve_generated_zone_batch_config(
        base, [target], assignment_folder=tmp_path
    )
    configs["paired_areas"] = {entry["policy"]: entry["config"] for entry in resolved}

    for zoning, by_policy in configs.items():
        for policy in POLICIES:
            label = f"{data_name}/{zoning}/{policy}"
            market.reconfigure(by_policy[policy], str(tmp_path), write_config=False)
            assert not pm.unmodelled_policy_settings(market.config, "simulator"), label
            own = _simulate(market, shared=False)
            shared = _simulate(market, shared=True)
            _assert_identical(shared, own, label)
            policy_market = market.policy_market()
            if market.config["guard-rails"] != -1:
                _assert_reserves_match(market, policy_market, shared["prefs"], label)
            # PolicyMarket.apply is the same thing in one call.
            applied = policy_market.apply(
                market.zones.zone_priority_matrix,
                market.config["ctip-options"][0],
                add_aa_schools=bool(market.config.get("add_aa_schools")),
                generated_zones=zoning != "attendance_areas",
                citywide_program_ids=getattr(market.zones, "citywide_program_ids", []),
            )
            for field, key in (
                ("preferences", "prefs"),
                ("pref_length", "pref_length"),
                ("policy_priorities", "base"),
                ("priorities", "without_lottery"),
                ("zone_eligibility", "eligibility"),
            ):
                np.testing.assert_array_equal(
                    getattr(applied, field), own[key], err_msg=f"{label}: {field}"
                )
