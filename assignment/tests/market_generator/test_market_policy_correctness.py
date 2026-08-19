from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from assignment.student_assignment.cli import cli
from assignment.student_assignment.evaluation.match_evaluator import MatchEvaluator
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


def test_zone_policy_reports_missing_file_path(tmp_path):
    zone_file = tmp_path / "missing-zones.csv"
    market = SimpleNamespace(
        config={"paths": {"zone-files": {"18zone_2": str(zone_file)}}}
    )

    with pytest.raises(FileNotFoundError) as error:
        PriorityGenerator(market).generate_base_priorities("18zone_2")

    assert "18zone_2" in str(error.value)
    assert str(zone_file.resolve()) in str(error.value)


def test_zone_policy_reports_available_aliases():
    market = SimpleNamespace(
        config={"paths": {"zone-files": {"18zone_2": "zones.csv"}}}
    )

    with pytest.raises(KeyError, match="not configured") as error:
        PriorityGenerator(market).generate_base_priorities("unknown")

    assert "18zone_2" in str(error.value)


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


def _assignment_saving_market(
    tmp_path, export_aggregate_metrics=None, export_local_metrics=None
):
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {
        "restrict-zone": False,
        "utility-model": {"enable": False},
        "subconfig-name": "status_quo",
    }
    if export_aggregate_metrics is not None:
        market.config["export-aggregate-metrics"] = export_aggregate_metrics
    if export_local_metrics is not None:
        market.config["export-local-metrics"] = export_local_metrics
    market.output_assignment_path = tmp_path / "assignments"
    market.students = SimpleNamespace(
        student_data=pd.DataFrame(index=pd.Index([1, 2], name="studentno")),
        distance_data=pd.DataFrame(
            {"101-GE-KG": [1.0, 2.0]},
            index=pd.Index([1, 2], name="studentno"),
        ),
        selected_preference_rank_matrix=Mock(
            return_value=np.array([[1.0], [1.0]])
        ),
    )
    market.programs = SimpleNamespace(
        codes={0: np.nan, 1: "101-GE-KG"},
        indices={"101-GE-KG": 1},
        program_df=pd.DataFrame({"capacity": [10]}),
    )
    market.preference_generator = SimpleNamespace(pref_length=1)
    market.data_scenario = object()
    market._write_aggregate_metrics = False
    market._reset_aggregate_metric_reports()
    market._get_assignment_save_name = Mock(
        side_effect=lambda _policy, iteration: f"policy/policy_iteration{iteration}.csv"
    )
    return market


def _aggregate_reports(config_name, value, sibling_value=None):
    program_ids = ["101-GE-KG"]
    program_types = ["GE"]
    program_values = [value]
    if sibling_value is not None:
        program_ids.append("101-GE-TK")
        program_types.append("GE")
        program_values.append(sibling_value)
    return {
        "program": pd.DataFrame(
            {
                "config_name": [config_name] * len(program_ids),
                "program_id": program_ids,
                "school_id": [101] * len(program_ids),
                "school_name": ["Alpha"] * len(program_ids),
                "school_category": ["Attendance"] * len(program_ids),
                "program_type": program_types,
                "metric": program_values,
            }
        ),
        "zip_code": pd.DataFrame(
            {"config_name": [config_name], "zip_code": [94110], "metric": [value]}
        ),
        "attendance_area": pd.DataFrame(
            {
                "config_name": [config_name],
                "attendance_area": [101],
                "metric": [value],
            }
        ),
        "citywide": pd.DataFrame(
            {"config_name": [config_name], "metric": [value]}
        ),
    }


def _configure_reuse_market(market):
    market.config.update(
        {
            "policies": ["zones"],
            "ctip-options": [0],
            "rounds-merged-options": [0],
            "ties-options": ["STB"],
            "guard-rails": -1,
            "reserve-settings": {},
            "restrict-zone": False,
            "citywide-or-lp": [],
            "iterations": {"start": 0, "end": 2},
            "random-seed": 7,
            "reuse_assignments": True,
        }
    )
    market._reset_zones = Mock()


def _write_reusable_assignment(market, iteration):
    return market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, 0, "STB"),
        iteration,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )


def test_saved_rank_uses_source_listed_position_not_filtered_mechanism_rank(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    market.students.selected_preference_rank_matrix.return_value = np.array(
        [[4.0], [1.0]]
    )

    assignment = market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )

    assert assignment.loc[0, "rank_basis"] == "listed"
    assert assignment.loc[0, "submitted_rank"] == 4
    assert assignment.loc[0, "rank"] == 4
    assert assignment.loc[0, "mechanism_rank"] == 1
    assert assignment.loc[1, ["rank", "mechanism_rank"]].isna().all()


def test_saved_utility_rank_is_independent_from_submitted_rank(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    market.config["utility-model"]["enable"] = True
    market.students.selected_preference_rank_matrix.return_value = np.array(
        [[4.0], [1.0]]
    )
    market.umodel = SimpleNamespace(
        original_preferences=np.array([[1], [1]]),
        original_utilities=np.array([[10.0], [5.0]]),
    )

    assignment = market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )

    assert assignment.loc[0, "rank_basis"] == "utility"
    assert assignment.loc[0, "submitted_rank"] == 4
    assert assignment.loc[0, "utility_rank"] == 1
    assert assignment.loc[0, "rank"] == 1


def test_saved_mechanism_rank_preserves_designation_position(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)

    assignment = market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([7, 2]),
        np.zeros(2),
    )

    assert assignment.loc[0, "mechanism_rank"] == 7
    assert assignment.loc[0, "designation"] == 1


def test_complete_assignment_run_is_reused_and_exported(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    for iteration in [0, 1]:
        _write_reusable_assignment(market, iteration)
    market.config["export-aggregate-metrics"] = True
    market._record_assignment_metric_reports = Mock()
    market._run_single_iteration_of_policy = Mock()

    assignments = list(market.create_iterations_generator())

    assert len(assignments) == 2
    market._run_single_iteration_of_policy.assert_not_called()
    assert market._record_assignment_metric_reports.call_count == 2


def test_incomplete_assignment_run_is_regenerated(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    _write_reusable_assignment(market, 0)
    market._run_single_iteration_of_policy = Mock(
        side_effect=lambda iteration, _policy: iter(
            [pd.DataFrame({"iteration": [iteration]})]
        )
    )

    assignments = list(market.create_iterations_generator())

    assert len(assignments) == 2
    assert market._run_single_iteration_of_policy.call_count == 2
    assert not market._assignment_save_path(
        Policy("zones", 0, 0, "STB"), 0
    ).exists()


def test_invalid_complete_assignment_run_is_rejected(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    for iteration in [0, 1]:
        _write_reusable_assignment(market, iteration)
    invalid_path = market._assignment_save_path(Policy("zones", 0, 0, "STB"), 1)
    pd.read_csv(invalid_path).iloc[[0]].to_csv(invalid_path, index=False)
    market._run_single_iteration_of_policy = Mock()

    with pytest.raises(ValueError, match="does not match the current students"):
        list(market.create_iterations_generator())

    market._run_single_iteration_of_policy.assert_not_called()


def test_assignment_reuse_can_be_disabled(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    for iteration in [0, 1]:
        _write_reusable_assignment(market, iteration)
    market.config["reuse_assignments"] = False
    market._run_single_iteration_of_policy = Mock(
        side_effect=lambda iteration, _policy: iter(
            [pd.DataFrame({"iteration": [iteration]})]
        )
    )

    list(market.create_iterations_generator())

    assert market._run_single_iteration_of_policy.call_count == 2
    assert not market._assignment_save_path(
        Policy("zones", 0, 0, "STB"), 0
    ).exists()


def test_assignment_reuse_rejects_rank_basis_from_different_policy_mode(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    policy = Policy("zones", 0, 0, "STB")
    _write_reusable_assignment(market, 0)
    market.config["utility-model"]["enable"] = True

    with pytest.raises(ValueError, match="does not match policy"):
        market._load_reusable_assignment(policy, 0)


def test_real_match_assignment_is_reused_before_reconstruction(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    market.config["reuse_assignments"] = True
    policy = Policy("real_match", None, None, None)
    market._save_assignment(
        np.array([[1], [0]]),
        policy,
        None,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )
    market.preference_generator.initialize_real_preferences = Mock()

    reused = market._read_real_match()

    assert len(reused) == 2
    market.preference_generator.initialize_real_preferences.assert_not_called()


def test_assignment_metrics_export_is_disabled_when_omitted(tmp_path, monkeypatch):
    market = _assignment_saving_market(tmp_path)
    from_scenario = Mock()
    monkeypatch.setattr(MatchEvaluator, "from_scenario", from_scenario)

    market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )

    assert (tmp_path / "assignments/status_quo/policy/policy_iteration0.csv").is_file()
    from_scenario.assert_not_called()
    assert not (tmp_path / "assignments/aggregate_metrics").exists()


def test_assignment_metrics_are_averaged_by_full_policy_variant(tmp_path, monkeypatch):
    market = _assignment_saving_market(
        tmp_path,
        export_aggregate_metrics=True,
        export_local_metrics=True,
    )
    evaluator = Mock()
    iteration_reports = iter(
        [
            _aggregate_reports("status_quo/policy", 1.0, 10.0),
            _aggregate_reports("status_quo/policy", 3.0, 20.0),
        ]
    )
    evaluator.eval_aggregate_metric_reports.side_effect = (
        lambda _config_name, **_kwargs: next(iteration_reports)
    )
    from_scenario = Mock(return_value=evaluator)
    monkeypatch.setattr(MatchEvaluator, "from_scenario", from_scenario)

    for iteration in [0, 1]:
        market._save_assignment(
            np.array([[1], [0]]),
            Policy("zones", 0, 0, "STB"),
            iteration,
            np.array([1, 0]),
            np.array([1, 2]),
            np.zeros(2),
        )

    reports = market._complete_aggregate_metric_reports()

    from_scenario.assert_called_once()
    assert all(
        call.args[0] is market.data_scenario for call in from_scenario.call_args_list
    )
    assert all(
        call.kwargs["program_data"] is market.programs.program_df
        for call in from_scenario.call_args_list
    )
    assert (
        from_scenario.call_args.kwargs["distance_cache"]
        is market.students.distance_data
    )
    evaluator.update_assignments.assert_called_once()
    assert evaluator.eval_aggregate_metric_reports.call_args_list == [
        (("status_quo/policy",), {"include_local_metrics": True}),
        (("status_quo/policy",), {"include_local_metrics": True}),
    ]
    assert reports["program"]["program_id"].tolist() == [
        "101-GE-KG",
        "101-GE-TK",
    ]
    assert reports["program"]["program_type"].tolist() == ["GE", "GE"]
    assert reports["program"]["metric"].tolist() == [2.0, 15.0]
    assert reports["program"]["school_id"].tolist() == [101, 101]
    assert reports["zip_code"].loc[0, "metric"] == 2
    assert reports["attendance_area"].loc[0, "metric"] == 2
    assert reports["citywide"].loc[0, "metric"] == 2
    assert (tmp_path / "assignments/status_quo/policy/policy_iteration0.csv").is_file()
    assert (tmp_path / "assignments/status_quo/policy/policy_iteration1.csv").is_file()
    assert not (tmp_path / "assignments/aggregate_metrics").exists()


def test_assignment_metrics_default_to_citywide_only(tmp_path, monkeypatch):
    market = _assignment_saving_market(
        tmp_path,
        export_aggregate_metrics=True,
        export_local_metrics=False,
    )
    evaluator = Mock()
    evaluator.eval_aggregate_metric_reports.return_value = {
        "citywide": pd.DataFrame(
            {"config_name": ["status_quo/policy"], "metric": [2.0]}
        )
    }
    monkeypatch.setattr(
        MatchEvaluator,
        "from_scenario",
        Mock(return_value=evaluator),
    )

    market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )
    reports = market._complete_aggregate_metric_reports()

    assert set(reports) == {"citywide"}
    evaluator.eval_aggregate_metric_reports.assert_called_once_with(
        "status_quo/policy", include_local_metrics=False
    )


def test_failed_metrics_export_removes_existing_assignment_marker(
    tmp_path, monkeypatch
):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=True)
    save_path = tmp_path / "assignments/status_quo/policy/policy_iteration0.csv"
    save_path.parent.mkdir(parents=True)
    save_path.write_text("stale assignment")
    evaluator = Mock()
    evaluator.eval_aggregate_metric_reports.side_effect = ValueError("report failed")
    monkeypatch.setattr(
        MatchEvaluator,
        "from_scenario",
        Mock(return_value=evaluator),
    )

    with pytest.raises(ValueError, match="report failed"):
        market._save_assignment(
            np.array([[1], [0]]),
            Policy("zones", 0, 0, "STB"),
            0,
            np.array([1, 0]),
            np.array([1, 2]),
            np.zeros(2),
        )

    assert not save_path.exists()


def test_combined_metric_writer_creates_only_four_run_level_csvs(tmp_path):
    reports = _aggregate_reports("status_quo/policy", 2.0, 5.0)
    legacy_dir = tmp_path / "aggregate_metrics/old-config/iteration0"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "obsolete.csv").write_text("stale")

    MarketGenerator.write_aggregate_metric_reports(tmp_path, reports)

    output_dir = tmp_path / "aggregate_metrics"
    assert {path.name for path in output_dir.iterdir()} == {
        "metrics_by_program.csv",
        "metrics_by_zip_code.csv",
        "metrics_by_attendance_area.csv",
        "metrics_citywide.csv",
    }
    program = pd.read_csv(output_dir / "metrics_by_program.csv")
    assert program["program_id"].tolist() == ["101-GE-KG", "101-GE-TK"]
    assert program["program_type"].tolist() == ["GE", "GE"]
    assert program["metric"].tolist() == [2.0, 5.0]
    assert program["school_id"].tolist() == [101, 101]
    citywide = pd.read_csv(output_dir / "metrics_citywide.csv")
    assert citywide.loc[0, "config_name"] == "status_quo/policy"
    assert citywide.loc[0, "metric"] == 2


def test_combined_metric_writer_can_create_only_citywide_csv(tmp_path):
    reports = _aggregate_reports("status_quo/policy", 2.0)

    MarketGenerator.write_aggregate_metric_reports(
        tmp_path, {"citywide": reports["citywide"]}
    )

    output_dir = tmp_path / "aggregate_metrics"
    assert {path.name for path in output_dir.iterdir()} == {
        "metrics_citywide.csv"
    }


def test_combined_metric_writer_expands_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    MarketGenerator.write_aggregate_metric_reports(
        "~/assignment-runs",
        _aggregate_reports("status_quo/policy", 2.0),
    )

    assert (
        tmp_path / "assignment-runs/aggregate_metrics/metrics_citywide.csv"
    ).is_file()


def test_empty_geography_report_keeps_metric_headers():
    empty_zip = pd.DataFrame(columns=["config_name", "zip_code", "metric"])
    averaged = MarketGenerator._average_aggregate_metric_frames(
        "zip_code", [empty_zip]
    )
    combined = MarketGenerator.combine_aggregate_metric_reports(
        [
            {
                "program": pd.DataFrame(
                    columns=[
                        "config_name",
                        "program_id",
                        "school_id",
                        "school_name",
                        "school_category",
                        "program_type",
                    ]
                ),
                "zip_code": averaged,
                "attendance_area": pd.DataFrame(),
                "citywide": pd.DataFrame(),
            }
        ]
    )

    assert combined["zip_code"].columns.tolist() == [
        "config_name",
        "zip_code",
        "metric",
    ]


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
