from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from assignment.student_assignment.cli import cli
from assignment.student_assignment.market_generator.policy import Policy
from assignment.student_assignment.market_generator.preference_generator import (
    PreferenceGenerator,
)
from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)
from assignment.student_assignment.market_generator.school_choice_market import (
    SchoolChoiceMarket,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def _runtime_config(**overrides):
    config = {
        "assignment-algorithm": "DA",
        "save-assignment": True,
        "utility-model": {"enable": False},
        "random-seed": 7,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"save-assignment": False}, "save-assignment must be true"),
        ({"assignment-algorithm": "TTC"}, "only 'DA' is supported"),
    ],
)
def test_runtime_config_rejects_removed_execution_modes(override, message):
    config = _runtime_config(**override)

    with pytest.raises(ValueError, match=message):
        SchoolChoiceMarket._validate_config(config)


def test_simulate_executes_every_configured_subconfig():
    loaded_configs = [
        _runtime_config(**{"subconfig-name": "first"}),
        _runtime_config(**{"subconfig-name": "second"}),
    ]

    class FakeConfigurator:
        def __init__(self):
            self.config = {"subconfigs": ["first", "second"]}
            self.index = 0

        def load_next_subconfig(self):
            self.config = loaded_configs[self.index]
            self.index += 1
            return True

    market = MarketGenerator.__new__(MarketGenerator)
    market.configurator = FakeConfigurator()
    market.config = market.configurator.config
    market._materialize_config = Mock(
        side_effect=lambda config: setattr(market, "config", config)
    )
    market._reset_zones = Mock()
    market.create_iterations_generator = Mock(
        side_effect=[iter(["first"]), iter(["second"])]
    )

    market.simulate()

    assert market.configurator.index == 2
    assert market._materialize_config.call_count == 2
    assert market.create_iterations_generator.call_count == 2
    assert market._reset_zones.call_count == 2


def test_real_match_is_yielded_by_iterations_generator():
    expected = pd.DataFrame({"studentno": [1]})
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {"policies": ["real_match"]}
    market._read_real_match = Mock(return_value=expected)

    assignments = list(market.create_iterations_generator())

    assert len(assignments) == 1
    assert assignments[0] is expected
    market._read_real_match.assert_called_once_with()


def test_assignment_names_cover_all_policy_option_dimensions():
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {
        "grade": "KG",
        "guard-rails": 0,
        "reserve-settings": {"reserve_fraction": [0.57, 0.43]},
        "restrict-zone": True,
        "citywide-or-lp": [],
        "priority-weights": {"sibling": 1},
        "paths": {"zone-files": {"zones": "zones.csv"}},
    }
    baseline_policy = Policy("zones", 1, 0, "MTB")
    baseline = market._get_assignment_save_name(baseline_policy, 3)

    variants = []
    variants.append(market._get_assignment_save_name(Policy("zones", 1, 123, "MTB"), 3))
    variants.append(market._get_assignment_save_name(Policy("zones", 1, 0, "STB"), 3))

    market.config["guard-rails"] = 1
    variants.append(market._get_assignment_save_name(baseline_policy, 3))
    market.config["guard-rails"] = 0

    market.config["reserve-settings"] = {"reserve_fraction": [0.6, 0.4]}
    variants.append(market._get_assignment_save_name(baseline_policy, 3))
    market.config["reserve-settings"] = {"reserve_fraction": [0.57, 0.43]}

    market.config["citywide-or-lp"] = ["language"]
    variants.append(market._get_assignment_save_name(baseline_policy, 3))
    market.config["citywide-or-lp"] = []

    variants.append(market._get_assignment_save_name(baseline_policy, 4))

    assert baseline.endswith("_iteration3.csv")
    assert len({baseline, *variants}) == 1 + len(variants)


def test_preference_length_real_plus_three_and_ethnicity_means():
    student_data = pd.DataFrame(
        {
            "num_ranked": [2, 4, 9],
            "resolved_ethnicity": ["A", "A", "B"],
        }
    )
    market = SimpleNamespace(
        n=3,
        num_programs=12,
        students=SimpleNamespace(student_data=student_data),
        config={"utility-model": {"list-length": "real_length_+3"}},
    )
    generator = PreferenceGenerator(market)

    np.testing.assert_array_equal(generator.set_number_programs_ranked(), [5, 7, 12])

    market.config["utility-model"]["list-length"] = "length_by_ethn"
    np.testing.assert_array_equal(generator.set_number_programs_ranked(), [3, 3, 9])


def test_selective_high_school_offsets_and_missing_programs():
    market = SimpleNamespace(
        n=2,
        num_programs=2,
        config={"year": 20},
        programs=SimpleNamespace(indices={"815-GE-09": 1, "100-GE-09": 2}),
        students=SimpleNamespace(
            sota_eligible=np.array([0, 1]),
            lowell_eligible=np.array([0, 0]),
        ),
    )

    priorities = PriorityGenerator(market)._selective_hs_eligibility()

    np.testing.assert_array_equal(priorities, [[-500, 0], [0, 0]])


def test_non_designation_boost_comes_from_config():
    market = SimpleNamespace(
        n=1,
        num_programs=2,
        config={"non_designation_boost": 37, "restrict-zone": False},
        preference_generator=SimpleNamespace(pref_length=np.array([1])),
    )
    generator = PriorityGenerator(market)
    generator._set_rounds_merged = Mock(return_value=np.zeros((1, 2)))
    generator._set_policy_priorities = Mock(return_value=np.zeros((1, 2)))

    priorities = generator.get_priorities_without_lottery(
        Policy("zones", 0, 0, "MTB"), np.array([[1, 0]])
    )

    np.testing.assert_array_equal(priorities, [[37, 0]])


def test_round_merging_uses_ordinals_and_restricts_legacy_codes():
    students = SimpleNamespace(
        first_round=np.array([0, 1, 3]),
        rounds=4,
    )
    market = SimpleNamespace(n=3, num_programs=1, students=students)
    generator = PriorityGenerator(market)

    np.testing.assert_array_equal(
        generator._set_rounds_merged(0).ravel(), [3000, 2000, 0]
    )
    np.testing.assert_array_equal(
        generator._set_rounds_merged("all").ravel(), [0, 0, 0]
    )
    with pytest.raises(ValueError, match="supports at most three selected rounds"):
        generator._set_rounds_merged(123)


def test_unknown_tiebreakers_and_missing_lottery_iteration_are_fatal():
    market = SimpleNamespace(
        n=1,
        num_programs=1,
        config={"read-lotteries": False},
    )
    generator = PriorityGenerator(market)

    with pytest.raises(ValueError, match="Unknown tiebreaker"):
        generator._set_tiebreaker("unknown")

    market.config = {
        "read-lotteries": True,
        "paths": {"lotteries-path": "lottery_"},
    }
    with pytest.raises(ValueError, match="iteration is required"):
        generator._set_tiebreaker("MTB")


def test_policy_priorities_forward_the_lottery_iteration():
    market = SimpleNamespace(n=1, num_programs=1, config={})
    generator = PriorityGenerator(market)
    generator.get_priorities_without_lottery = Mock(
        return_value=np.zeros((1, 1))
    )
    generator._set_tiebreaker = Mock(return_value=np.zeros((1, 1)))

    generator.set_policy_specific_priorities(
        Policy("zones", 0, 0, "MTB"),
        np.array([[1]]),
        iteration=9,
    )

    generator._set_tiebreaker.assert_called_once_with("MTB", iteration=9)


def test_installed_cli_runs_subconfig_aware_simulation(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    assignment_path = tmp_path / "assignments"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"scenario": "legacy", "overrides": {}},
                "paths": {"assignment-folder": str(assignment_path)},
                "subconfigs": ["first", "second"],
            }
        )
    )
    calls = []

    class FakeMarketGenerator:
        def __init__(self, *, config, assignment_path):
            calls.append((config, Path(assignment_path)))

        def simulate(self):
            calls.append("simulate")

    import assignment.student_assignment.market_generator.school_choice_market_generator as market_module

    monkeypatch.setattr(market_module, "MarketGenerator", FakeMarketGenerator)

    result = CliRunner().invoke(cli, ["simulate", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert calls[-1] == "simulate"
    assert calls[0][0]["subconfigs"] == ["first", "second"]
    assert calls[0][1] == assignment_path.resolve()
