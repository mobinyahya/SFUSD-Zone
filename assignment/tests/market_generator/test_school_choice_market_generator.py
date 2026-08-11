from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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

    next(market._run_single_iteration_of_policy(0, "status_quo"))

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
