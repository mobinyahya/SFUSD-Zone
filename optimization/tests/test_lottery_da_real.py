"""The lottery DA reproduces the assignment simulator exactly on real data.

The simulator runs one lottery of a reserve policy on today's attendance areas;
its own lottery numbers are then replayed through the policy MID market, with
every student in the simulator's zone, and every student must land where the
simulator put them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmark.config import optimization_config_from_dict
from optimization.data.dataset import Dataset
from optimization.data.mid import APPENDED, build_mid_market
from optimization import lottery_da

POLICY = "restrictive_aa_sib_ctip_reserves"
FILTERS = {
    "capacity_scenario": "ideal_baseline_nov_13_2026",
    "student_population": "applied_or_enrolled",
}
DATA = {
    "scenario": "attendance-areas-2627",
    "overrides": {"filters": {"optimization": FILTERS, "assignment": FILTERS}},
}
REPOSITORY = Path(__file__).resolve().parents[2]


def _simulate(tmp_path, monkeypatch):
    from assignment.student_assignment.market_generator import (
        school_choice_market_generator as generator,
    )

    captured = {}
    original = generator.MarketGenerator._generate_assignment_with_guardrails

    def capture(self, prefs, priorities):
        result = original(self, prefs, priorities)
        captured.update(market=self, priorities=priorities.copy(), match=result[0])
        return result

    monkeypatch.setattr(
        generator.MarketGenerator, "_generate_assignment_with_guardrails", capture
    )
    config = yaml.safe_load(
        (
            REPOSITORY
            / "assignment/configs/attendance_areas/restrictive_aa_sib_2627.yaml"
        ).read_text()
    )
    config.update(
        data=DATA,
        subconfigs=[POLICY],
        iterations={"start": 0, "end": 1},
        reuse_assignments=False,
        **{"export-aggregate-metrics": False, "export-local-metrics": False},
    )
    config["paths"]["assignment-folder"] = str(tmp_path)
    config["output_dir"] = f"{tmp_path}/"
    generator.MarketGenerator(config=config, assignment_path=str(tmp_path)).simulate()
    return captured


@pytest.mark.real_data
def test_lottery_da_replays_the_simulator(tmp_path, monkeypatch):
    try:
        captured = _simulate(tmp_path, monkeypatch)
    except FileNotFoundError as error:
        pytest.skip(f"real data not available: {error}")
    sim = captured["market"]
    students, programs = sim.students, sim.programs
    sim_ids = {index: pid for pid, index in programs.indices.items()}
    lottery = captured["priorities"] - np.floor(captured["priorities"])

    config = optimization_config_from_dict(
        {
            "data": DATA,
            "centroids_type": "all",
            "levels": ["Block_0"],
            "mid_preference_source": "stated",
            "matching_policy": POLICY,
            "mid_lotteries": 1,
        }
    )
    problem = Dataset(config).problem_for("Block_0", config=config)
    market = build_mid_market(problem, config)
    prepared = lottery_da._prepared(market)

    # Today's map: one zone per attendance-area school, each student in the
    # zone of their district attendance area; a school nobody's area names
    # (Mission Bay) belongs to no zone and is open to everyone.
    rows = {str(sn): i for i, sn in students.idx2studentno.items()}
    zoning = {
        int(p.school_node): int(p.school_id)
        for p in market.programs
        if p.school_node is not None
    }
    base = prepared.max_node + 1
    for t, student_type in enumerate(market.types):
        studentno = students.idx2studentno[rows[student_type.student_id]]
        zoning[base + t] = int(students.attendance_area.get(studentno, -1) or -1)
    prepared.node = np.arange(base, base + len(market.types), dtype=np.int64)
    prepared.max_node = base + len(market.types)
    owned = {zoning[base + t] for t in range(len(market.types))}
    unowned = {
        i
        for i, p in enumerate(market.programs)
        if p.school_node is not None and int(p.school_id) not in owned
    }
    prepared.tier = prepared.tier.copy()
    for i in unowned:
        prepared.citywide[i] = True
    for position, program in enumerate(prepared.alternative):
        if program in unowned and prepared.kind[position] == APPENDED:
            prepared.tier[position] = -1.0

    draws = np.array(
        [
            [
                lottery[rows[t.student_id], programs.indices[pid] - 1]
                for t in market.types
                for pid in t.programs
            ]
        ]
    )
    placed, _, _ = lottery_da.lottery_placements(market, zoning, draws)
    model = {
        t.student_id: market.programs[placed[0, i]].program_id
        if placed[0, i] >= 0
        else None
        for i, t in enumerate(market.types)
    }
    simulated = {
        str(students.idx2studentno[r]): sim_ids[int(m)] if int(m) > 0 else None
        for r, m in enumerate(captured["match"])
    }
    assert model == simulated
