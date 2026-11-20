"""With the utility model, the lottery DA reproduces the simulator exactly.

The simulator runs a reserve policy with ``utility-model.enable: true`` for a
few iterations, drawing every student's list from the utility model each
time. The MID market of the same policy draws lottery ``l`` as the
simulator's iteration ``l`` (``mid_lottery_seed: 0``), lists and lottery
numbers alike, and nothing is replayed: every student must land where the
simulator put them, in every lottery, for a capped list (``list-length: 7``),
for the outside option, and for a short list with a smaller Gumbel scale and
language designation for everyone, on today's attendance areas and on a
generated zoning that pairs them.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from assignment.tests.market_generator.utility_estimate import (
    estimate_source,
    write_covering_estimate,
)

POLICY = "restrictive_aa_sib_ctip_reserves"
FILTERS = {
    "capacity_scenario": "ideal_baseline_nov_13_2026",
    "student_population": "applied_or_enrolled",
}
LOTTERIES = 3
REPOSITORY = Path(__file__).resolve().parents[2]
MATCHING_CONFIG = (
    REPOSITORY / "assignment/configs/attendance_areas/restrictive_aa_sib_2627.yaml"
)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from assignment.student_assignment.market_generator.school_choice_market import (
        load_market_data,
    )
    from loaders.config import load_scenario

    data = {
        "scenario": "attendance-areas-2627",
        "overrides": {"filters": {"optimization": FILTERS, "assignment": FILTERS}},
    }
    try:
        scenario = load_scenario(data)
        programs, _, students = load_market_data(scenario)
        estimate = write_covering_estimate(
            scenario,
            programs,
            students,
            tmp_path_factory.mktemp("estimate") / "estimate.csv",
        )
    except FileNotFoundError as error:
        pytest.skip(f"real data not available: {error}")
    data["overrides"]["sources"] = {"assignment.estimate": estimate_source(estimate)}
    return data


def _simulate(config, folder):
    """Run ``config``; return each iteration's market and match."""
    from assignment.student_assignment.market_generator import (
        school_choice_market_generator as generator,
    )

    captured = []
    original = generator.MarketGenerator._generate_assignment_with_guardrails

    def capture(self, prefs, priorities):
        result = original(self, prefs, priorities)
        captured.append(
            {
                "market": self,
                "shared": self.policy_market() is not None,
                "zone_priority": np.asarray(self.zones.zone_priority_matrix).copy(),
                "match": result[0].copy(),
            }
        )
        return result

    generator.MarketGenerator._generate_assignment_with_guardrails = capture
    try:
        generator.MarketGenerator(
            config=config,
            assignment_path=str(folder),
            write_config=False,
            write_aggregate_metrics=False,
        ).simulate()
    finally:
        generator.MarketGenerator._generate_assignment_with_guardrails = original
    assert len(captured) == LOTTERIES
    # The simulator ran the utility model on the shared policy market.
    assert all(run["shared"] for run in captured)
    return captured


def _matching_config(data, utility_model, folder):
    from assignment.run_custom_config import load_custom_config

    config = load_custom_config(
        MATCHING_CONFIG, assignment_folder=folder, absolute_assignment_folder=True
    )
    config.update(
        data=copy.deepcopy(data),
        subconfigs=[POLICY],
        iterations={"start": 0, "end": LOTTERIES},
        reuse_assignments=False,
        **{"export-aggregate-metrics": False, "export-local-metrics": False},
    )
    config["utility-model"] = utility_model
    return config


def _attendance_areas(config, folder):
    return _simulate(config, folder), None


def _paired_areas(config, folder):
    """Attendance areas joined two by two at the block level."""
    from assignment.generated_zones import resolve_generated_zone_batch_config
    from assignment.student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )
    from assignment.tests.market_generator.test_policy_market_real import (
        _paired_zone_file,
    )

    probe = MarketGenerator(
        config={**_policy_overlay(config), "subconfigs": []},
        assignment_path=str(folder),
        write_config=False,
        write_aggregate_metrics=False,
    )
    zone_file = _paired_zone_file(probe, folder / "paired_zones.csv")
    _, resolved = resolve_generated_zone_batch_config(
        config,
        [
            {
                "id": "paired",
                "zone_file": str(zone_file),
                "zone_building_blocks": "block",
                "data": config["data"],
            }
        ],
        assignment_folder=folder,
    )
    zone_of_block = {
        int(block): zone
        for zone, line in enumerate(zone_file.read_text().splitlines())
        for block in line.split(",")
    }
    return _simulate(resolved[0]["config"], folder), zone_of_block


def _policy_overlay(config):
    policy = yaml.safe_load(
        (
            REPOSITORY / f"assignment/configs/policy_configs/{POLICY}.yaml"
        ).read_text()
    )
    return {**config, **policy}


def _mid_market(data, utility_model, tmp_path, monkeypatch):
    from benchmark.config import optimization_config_from_dict
    from optimization.data import mid
    from optimization.data.dataset import Dataset

    # MID reads its utility model from the policy config, the simulator from
    # the matching config: give the policy the same block.
    policies = tmp_path / "policy_configs"
    policies.mkdir()
    policy = yaml.safe_load((mid.POLICY_CONFIG_DIR / f"{POLICY}.yaml").read_text())
    policy["utility-model"] = utility_model
    (policies / f"{POLICY}.yaml").write_text(yaml.safe_dump(policy))
    monkeypatch.setattr(mid, "POLICY_CONFIG_DIR", policies)
    config = optimization_config_from_dict(
        {
            "data": data,
            "centroids_type": "all",
            "levels": ["Block_0"],
            "mid_preference_source": "estimate",
            "matching_policy": POLICY,
            "mid_lotteries": LOTTERIES,
            "mid_lottery_seed": 0,
        }
    )
    problem = Dataset(config).problem_for("Block_0", config=config)
    return problem, mid.build_mid_market(problem, config)


def _model_zoning(market, problem, captured, zone_of_block):
    """The simulator's zoning as a MID zoning, and the programs it leaves in
    no zone (which the simulator opens to everyone)."""
    from optimization import lottery_da

    prepared = lottery_da._prepared(market)
    sim = captured[0]["market"]
    students = sim.students
    rows = {str(sn): i for i, sn in students.idx2studentno.items()}
    if zone_of_block is None:
        # One zone per attendance-area school, each student in the zone of
        # their district attendance area: give every student their own node.
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
    else:
        zoning = {
            int(node): zone_of_block.get(int(problem.G.nodes[node]["area_id"]), -1)
            for node in problem.G.nodes
        }
        unowned = {
            i
            for i, p in enumerate(market.programs)
            if p.school_node is not None and zoning[int(p.school_node)] < 0
        }
    for i in unowned:
        prepared.citywide[i] = True
    prepared.tier = np.where(
        np.isin(prepared.alternative, sorted(unowned))
        & (prepared.kind == 1),  # APPENDED: no zone owns the school
        -1.0,
        prepared.tier,
    )
    # Both sides agree on who is in whose zone at every zoned GE program.
    nodes = prepared.node
    zone_priority = captured[0]["zone_priority"]
    for i, p in enumerate(market.programs):
        if p.citywide or i in unowned or p.program_id.split("-")[1] != "GE":
            continue
        column = sim.programs.indices[p.program_id] - 1
        expected = np.array(
            [zone_priority[rows[t.student_id], column] > 0 for t in market.types]
        )
        in_zone = np.array([zoning[int(v)] for v in nodes]) == zoning[int(p.school_node)]
        np.testing.assert_array_equal(in_zone, expected, err_msg=p.program_id)
    return zoning


UTILITY_MODELS = {
    "seven": {"enable": True, "list-length": "7"},
    "outside_option": {"enable": True, "list-length": "outside_option"},
    "three_lp_for_all": {
        "enable": True,
        "list-length": "3",
        "gumbel-scale": 0.5,
        "designate-lp-for-all": True,
    },
}


@pytest.mark.real_data
@pytest.mark.parametrize("zoning_name", ["attendance_areas", "paired_areas"])
@pytest.mark.parametrize("model_name", list(UTILITY_MODELS))
def test_lottery_da_reproduces_the_utility_model(
    data, model_name, zoning_name, tmp_path, monkeypatch
):
    from optimization import lottery_da
    from optimization.data import mid

    utility_model = UTILITY_MODELS[model_name]
    folder = tmp_path / "simulation"
    config = _matching_config(data, utility_model, folder)
    simulate = _attendance_areas if zoning_name == "attendance_areas" else _paired_areas
    captured, zone_of_block = simulate(config, folder)

    problem, market = _mid_market(data, utility_model, tmp_path, monkeypatch)
    assert market.lottery.lists is not None
    assert market.lottery.random_seed == config["random-seed"]
    zoning = _model_zoning(market, problem, captured, zone_of_block)
    placed, kind, utility = lottery_da.lottery_placements(market, zoning)

    sim = captured[0]["market"]
    students, programs = sim.students, sim.programs
    sim_ids = {index: pid for pid, index in programs.indices.items()}
    rows = {str(sn): i for i, sn in students.idx2studentno.items()}
    for lottery in range(LOTTERIES):
        model = {
            t.student_id: market.programs[placed[lottery, i]].program_id
            if placed[lottery, i] >= 0
            else None
            for i, t in enumerate(market.types)
        }
        match = captured[lottery]["match"]
        simulated = {
            str(students.idx2studentno[r]): sim_ids[int(m)] if int(m) > 0 else None
            for r, m in enumerate(match)
        }
        assert model == simulated, f"lottery {lottery}"
        # Welfare reads the simulator's drawn utility of each placement.
        draw = sim.umodel if lottery == LOTTERIES - 1 else None
        if draw is not None:
            for i, t in enumerate(market.types):
                if placed[lottery, i] < 0:
                    continue
                r = rows[t.student_id]
                expected = draw.original_utilities[r, int(match[r]) - 1]
                assert utility[lottery, i] == (expected if np.isfinite(expected) else 0.0)
        assert int((kind[lottery] == mid.DESIGNATED).sum()) > 0


@pytest.mark.real_data
def test_preprocessing_and_the_recom_scorers_keep_the_utility_match(
    data, tmp_path, monkeypatch
):
    """``preprocess_mid_market`` (which the ReCom scorers run) drops only what
    no zoning can reach: every placement of every lottery is unchanged on a
    zoning the problem allows, and the scorers and the capacity reference
    read the same match."""
    from optimization import lottery_da
    from optimization.data.capacity_reference import build_capacity_reference
    from optimization.data.mid import preprocess_mid_market
    from optimization.solvers.burst_objectives import mid_burst_scorers

    problem, raw = _mid_market(data, UTILITY_MODELS["seven"], tmp_path, monkeypatch)
    market = preprocess_mid_market(raw, problem)
    assert len(market.lottery.lists.program) < len(raw.lottery.lists.program)
    zoning = {int(n): min(problem.candidate_zones(n)) for n in problem.nodes}

    def placements(m):
        program, kind, utility = lottery_da.lottery_placements(m, zoning)
        ids = np.array([p.program_id for p in m.programs] + [None], dtype=object)
        by_student = {t.student_id: i for i, t in enumerate(m.types)}
        order = [by_student[s] for s in sorted(by_student)]
        return ids[program][:, order], kind[:, order], utility[:, order]

    for before, after in zip(placements(raw), placements(market), strict=True):
        np.testing.assert_array_equal(before, after)
    expected = lottery_da.lottery_outcome(raw, zoning)
    outcome = lottery_da.lottery_outcome(market, zoning)
    for field in ("welfare", "unassigned", "designated", "fill_polarization"):
        assert getattr(outcome, field) == getattr(expected, field), field
    # Summed over another school order: equal to rounding.
    assert outcome.frl_dissimilarity == pytest.approx(
        expected.frl_dissimilarity, rel=0, abs=1e-9
    )

    scorers = mid_burst_scorers(problem, ("welfare", "frl_dissimilarity"), {"workers": 1})
    (welfare,) = scorers["welfare"]((zoning,), None)
    assert welfare == pytest.approx(expected.welfare, rel=0, abs=1e-9)
    reference = build_capacity_reference(problem, zoning, market, 20)
    assert reference["designated_students"] == pytest.approx(
        expected.designated, rel=0, abs=1e-9
    )
    assert reference["unassigned_students"] == pytest.approx(
        expected.unassigned, rel=0, abs=1e-9
    )
