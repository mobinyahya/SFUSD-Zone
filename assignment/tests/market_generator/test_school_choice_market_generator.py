from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from assignment.student_assignment.definitions import CONFIGS_DIR
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


@pytest.mark.parametrize(
    ("save_path", "expected_calls"),
    [(None, []), ("utility_matrix.npy", ["utility_matrix.npy"])],
)
def test_utility_matrix_save_path_is_optional(save_path, expected_calls):
    market = MarketGenerator.__new__(MarketGenerator)
    utility_model_config = {"enable": True}
    if save_path is not None:
        utility_model_config["save-path"] = save_path

    market.config = {
        "iterations": {"start": 0},
        "utility-model": utility_model_config,
    }
    market.students = SimpleNamespace(only_keep_rows=None)
    market.programs = SimpleNamespace(only_keep_cols=None)
    market.umodel = Mock()
    market._simulate_policy = Mock(return_value=iter(()))

    list(market._run_single_iteration_of_policy(0, "status_quo"))

    assert [
        call.args[0] for call in market.umodel.save_utility_matrix.call_args_list
    ] == expected_calls


def test_real_preferences_honor_designate_policy_setting():
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {
        "utility-model": {"enable": False},
        "designate": False,
        "ctip-options": [],
        "rounds-merged-options": [],
        "ties-options": [],
    }
    market.priority_generator = Mock()
    market.preference_generator = Mock()

    list(market._simulate_policy("status_quo", 0))

    market.preference_generator.initialize_real_preferences.assert_called_once_with(
        designate=False
    )


def test_reconfigure_replaces_zone_dependent_state():
    market = MarketGenerator.__new__(MarketGenerator)
    market._guardrail_setup_cache = {"stale": object()}
    market._active_policy_cache_context = "stale"
    market._set_up_save_folder = Mock()
    market._initialize_market_data = Mock()
    market._initialize_utility_model = Mock()

    with open(f"{CONFIGS_DIR}base_config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    with open(f"{CONFIGS_DIR}local_path_config.yaml") as path_file:
        config.update(yaml.safe_load(path_file))
    config.update(
        {
            "assignment-algorithm": "DA",
            "subconfigs": [],
            "utility-model": {"enable": False, "list-length": "7"},
        }
    )
    config["data"]["overrides"] = {
        "sources": {
            "assignment.zones": {
                "policy": {"path": "zones.csv", "classification": "public"}
            }
        }
    }
    market.reconfigure(config, "assignments")

    assert market.external_config != config
    assert (
        config["data"]["overrides"]["sources"]["assignment.zones"]["policy"]["path"]
        == "zones.csv"
    )
    assert market.external_config["data"]["overrides"]["sources"]["assignment.zones"][
        "policy"
    ]["path"].endswith("zones.csv")
    assert market.config is not config
    assert market.config["paths"]["zone-files"]["policy"].endswith("zones.csv")
    assert market.configurator.config is not market.config
    market._initialize_market_data.assert_called_once_with()
    market._initialize_utility_model.assert_called_once_with()
    assert market.priority_generator.market is market
    assert market.preference_generator.market is market
    assert market._guardrail_setup_cache == {}
    assert market._active_policy_cache_context is None
    market._set_up_save_folder.assert_called_once_with("assignments", write_config=True)


def _enrollment_market(population, frame):
    market = MarketGenerator.__new__(MarketGenerator)
    scenario = Mock()
    scenario.filter.return_value = population
    market.students = SimpleNamespace(data_scenario=scenario, student_data=frame)
    return market


def test_enrollment_real_match_reads_where_students_enrolled():
    frame = pd.DataFrame(
        {
            # 1 carries its enrolled program; 2 enrolled where round 2 placed
            # it; 3 enrolled at a school it ranked; 4 at one it never saw.
            "enrolled_idschool": [420, 413, 500, 600],
            "enrolled_programcode": ["SE", np.nan, np.nan, np.nan],
            "r1_idschool": [420, 420, 420, 420],
            "r1_programcode": ["GE", "GE", "GE", "GE"],
            "r2_idschool": [np.nan, 413, np.nan, np.nan],
            "r2_programcode": [np.nan, "CE", np.nan, np.nan],
            "selected_ranked_idschool": [[420], [420], [420, 500], [420]],
            "selected_programs": [["GE"], ["GE"], ["GE", "SB"], ["GE"]],
        }
    )
    market = _enrollment_market("enrolled", frame)

    with pytest.warns(UserWarning, match="1 defaulted to GE"):
        match = market._enrolled_school_program(frame)

    assert match["final_school"].tolist() == [420, 413, 500, 600]
    assert match["final_program"].tolist() == ["SE", "CE", "SB", "GE"]
    # The recorded assignment is left alone.
    assert frame["r1_idschool"].tolist() == [420, 420, 420, 420]


def test_enrollment_real_match_requires_the_enrolled_population():
    frame = pd.DataFrame({"enrolled_idschool": [420]})
    market = _enrollment_market("applicant", frame)

    with pytest.raises(ValueError, match="requires the enrolled population"):
        market._enrolled_school_program(frame)
