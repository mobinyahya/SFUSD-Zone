from types import SimpleNamespace
from unittest.mock import Mock

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
    assert config["data"]["overrides"]["sources"]["assignment.zones"][
        "policy"
    ]["path"] == "zones.csv"
    assert market.external_config["data"]["overrides"]["sources"][
        "assignment.zones"
    ]["policy"]["path"].endswith("zones.csv")
    assert market.config is not config
    assert market.config["paths"]["zone-files"]["policy"].endswith("zones.csv")
    assert market.configurator.config is not market.config
    market._initialize_market_data.assert_called_once_with()
    market._initialize_utility_model.assert_called_once_with()
    assert market.priority_generator.market is market
    assert market.preference_generator.market is market
    assert market._guardrail_setup_cache == {}
    assert market._active_policy_cache_context is None
    market._set_up_save_folder.assert_called_once_with(
        "assignments", write_config=True
    )
