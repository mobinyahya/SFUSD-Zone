import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from loaders import ResolvedSource

from assignment.student_assignment.choice_ranks import ASSIGNMENT_SCHEMA_VERSION
from assignment.student_assignment.cli import cli
from assignment.student_assignment.lottery_files import (
    read_lotteries,
    render_lottery,
    saved_lotteries,
    write_lotteries,
)
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

from ..utils_for_tests import fake_scenario


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


def _zone_market(zones, **filters):
    market = SchoolChoiceMarket.__new__(SchoolChoiceMarket)
    market.config = {"zone-building-blocks": "block_group"}
    market.data_scenario = fake_scenario({"assignment.zones": zones}, **filters)
    return market


def test_zone_policy_reports_missing_file_path(tmp_path):
    zone_file = tmp_path / "missing-zones.csv"
    market = _zone_market({"18zone_2": zone_file})

    with pytest.raises(FileNotFoundError) as error:
        PriorityGenerator(market).generate_base_priorities("18zone_2")

    assert "18zone_2" in str(error.value)
    assert str(zone_file.resolve()) in str(error.value)


def test_zone_policy_reports_available_aliases():
    market = _zone_market({"18zone_2": "zones.csv"})

    with pytest.raises(KeyError, match="not configured") as error:
        PriorityGenerator(market).generate_base_priorities("unknown")

    assert "18zone_2" in str(error.value)


def test_zone_file_rejects_census_ids_of_another_vintage():
    zones = {"bg": ResolvedSource(Path("zones.csv"), geography_vintage="2020")}

    assert _zone_market(zones, geography_vintage="2020").zone_file("bg") == Path(
        "zones.csv"
    )
    with pytest.raises(ValueError, match="Census 2020 GEOIDs.*Census 2010"):
        _zone_market(zones).zone_file("bg")


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
    }
    market.data_scenario = fake_scenario({"assignment.zones": {"zones": "zones.csv"}})
    baseline_policy = Policy("zones", 1, "MTB")
    baseline = market._get_assignment_save_name(baseline_policy, 3)

    variants = []
    variants.append(market._get_assignment_save_name(Policy("zones", 1, "STB"), 3))

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
        selected_preference_rank_matrix=Mock(return_value=np.array([[1.0], [1.0]])),
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
        "citywide": pd.DataFrame({"config_name": [config_name], "metric": [value]}),
    }


def _frl_threshold_inputs(config_name):
    return pd.DataFrame(
        {
            "config_name": [config_name],
            "program_id": ["101-GE-KG"],
            "frl_assigned": [0.5],
            "frl_non_designated": [0.5],
            "district_frl": [0.5],
        }
    )


def _configure_reuse_market(market):
    market.config.update(
        {
            "policies": ["zones"],
            "ctip-options": [0],
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
    assignment = market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, "STB"),
        iteration,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )
    market._flush_pending_lotteries()
    return assignment


def _lottery_data_path(tmp_path):
    return tmp_path / "assignments/status_quo/policy/policy_lotteries.csv.gz"


def test_saved_rank_uses_source_listed_position_not_filtered_mechanism_rank(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    market.students.selected_preference_rank_matrix.return_value = np.array(
        [[4.0], [1.0]]
    )

    assignment = market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, "STB"),
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


def test_saved_rank_counts_an_honoured_promotion_as_a_first_choice(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    # Student 1 is a TK promote who filed no Main Round request, so the
    # conversion appended their feeder at position 4; student 2 is not.
    market.students.student_data = market.students.student_data.assign(
        promote=[["101-GE-KG"], []],
        mr_applicant=[0, 1],
    )
    market.students.selected_preference_rank_matrix.return_value = np.array(
        [[4.0], [1.0]]
    )

    assignment = market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )

    assert assignment.loc[0, "submitted_rank"] == 1
    assert assignment.loc[0, "rank"] == 1
    assert assignment.loc[0, "rank_excluding_promotion"] == 4
    assert (
        assignment.loc[1, ["submitted_rank", "rank_excluding_promotion"]].isna().all()
    )


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
        Policy("zones", 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )

    assert assignment.loc[0, "rank_basis"] == "utility"
    assert assignment.loc[0, "submitted_rank"] == 4
    assert assignment.loc[0, "utility_rank"] == 1
    assert assignment.loc[0, "rank"] == 1
    assert assignment["assigned_utility"].tolist() == [10.0, 0.0]


def test_saved_mechanism_rank_preserves_designation_position(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)

    assignment = market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, "STB"),
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
    assert not market._saved_assignment_exists(Policy("zones", 0, "STB"), 0)
    assert not _lottery_data_path(tmp_path).exists()


def test_invalid_complete_assignment_run_is_rejected(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    for iteration in [0, 1]:
        _write_reusable_assignment(market, iteration)
    data_path = _lottery_data_path(tmp_path)
    truncated = read_lotteries(data_path).frame(1).iloc[[0]]
    write_lotteries(data_path, {1: render_lottery(truncated)})
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
    assert not market._saved_assignment_exists(Policy("zones", 0, "STB"), 0)
    assert not _lottery_data_path(tmp_path).exists()


def test_assignment_reuse_rejects_rank_basis_from_different_policy_mode(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    policy = Policy("zones", 0, "STB")
    _write_reusable_assignment(market, 0)
    market.config["utility-model"]["enable"] = True

    with pytest.raises(ValueError, match="does not match policy"):
        market._load_reusable_assignment(policy, 0)


def test_real_match_assignment_is_reused_before_reconstruction(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    market.config["reuse_assignments"] = True
    policy = Policy("real_match", None, None)
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
        Policy("zones", 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )
    market._flush_pending_lotteries()

    assert saved_lotteries(_lottery_data_path(tmp_path)) == {0}
    assert not (
        tmp_path / "assignments/status_quo/policy/policy_iteration0.csv"
    ).exists()
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
    evaluator.eval_frl_threshold_inputs.return_value = _frl_threshold_inputs(
        "status_quo/policy"
    )
    from_scenario = Mock(return_value=evaluator)
    monkeypatch.setattr(MatchEvaluator, "from_scenario", from_scenario)

    for iteration in [0, 1]:
        market._save_assignment(
            np.array([[1], [0]]),
            Policy("zones", 0, "STB"),
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
        (
            ("status_quo/policy",),
            {"include_local_metrics": True, "include_area_metrics": True},
        ),
        (
            ("status_quo/policy",),
            {"include_local_metrics": True, "include_area_metrics": True},
        ),
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
    assert saved_lotteries(_lottery_data_path(tmp_path)) == {0, 1}
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
    evaluator.eval_frl_threshold_inputs.return_value = _frl_threshold_inputs(
        "status_quo/policy"
    )
    monkeypatch.setattr(
        MatchEvaluator,
        "from_scenario",
        Mock(return_value=evaluator),
    )

    market._save_assignment(
        np.array([[1], [0]]),
        Policy("zones", 0, "STB"),
        0,
        np.array([1, 0]),
        np.array([1, 2]),
        np.zeros(2),
    )
    reports = market._complete_aggregate_metric_reports()

    assert set(reports) == {"citywide"}
    evaluator.eval_aggregate_metric_reports.assert_called_once_with(
        "status_quo/policy", include_local_metrics=False, include_area_metrics=True
    )


def test_failed_lottery_run_leaves_no_partial_lottery_file(tmp_path, monkeypatch):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    _write_reusable_assignment(market, 0)
    before = _lottery_data_path(tmp_path).read_bytes()
    market.config["export-aggregate-metrics"] = True
    evaluator = Mock()
    evaluator.eval_aggregate_metric_reports.side_effect = ValueError("report failed")
    monkeypatch.setattr(
        MatchEvaluator,
        "from_scenario",
        Mock(return_value=evaluator),
    )

    def lottery(iteration):
        yield market._save_assignment(
            np.array([[1], [0]]),
            Policy("zones", 0, "STB"),
            iteration,
            np.array([1, 0]),
            np.array([1, 2]),
            np.zeros(2),
        )

    with pytest.raises(ValueError, match="report failed"):
        list(market._buffered_lotteries([lottery(1)]))

    # Lottery 1 never reaches disk, and the file holding lottery 0 is intact.
    assert market._pending_lottery_buffer() == {}
    assert saved_lotteries(_lottery_data_path(tmp_path)) == {0}
    assert _lottery_data_path(tmp_path).read_bytes() == before


def test_lottery_file_rebuilds_the_per_lottery_frames(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    fresh = [_write_reusable_assignment(market, iteration) for iteration in [0, 1]]
    saved = read_lotteries(_lottery_data_path(tmp_path))
    sidecar = saved.sidecar

    assert saved.lotteries == [0, 1]
    assert sidecar["schema_version"] == ASSIGNMENT_SCHEMA_VERSION
    assert sidecar["programcodes"] == {"0": "", "1": "101-GE-KG"}
    assert set(sidecar["derived_columns"]) == {
        "assignment_schema_version",
        "rank_basis",
        "utility_rank",
        "programcodes",
    }
    assert sidecar["columns"] == list(fresh[0].columns)
    for iteration, frame in enumerate(fresh):
        legacy = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        pd.testing.assert_frame_equal(saved.frame(iteration), legacy)


def test_legacy_per_lottery_assignment_files_are_reused(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    frames = [_write_reusable_assignment(market, iteration) for iteration in [0, 1]]
    _lottery_data_path(tmp_path).unlink()
    for iteration, frame in enumerate(frames):
        legacy_path = market._assignment_save_path(Policy("zones", 0, "STB"), iteration)
        frame.assign(assignment_schema_version=3).to_csv(legacy_path, index=False)
    market._run_single_iteration_of_policy = Mock()

    reused = list(market.create_iterations_generator())

    market._run_single_iteration_of_policy.assert_not_called()
    assert [frame["programno"].tolist() for frame in reused] == [[1, 0], [1, 0]]
    assert all(
        frame["assignment_schema_version"].eq(ASSIGNMENT_SCHEMA_VERSION).all()
        for frame in reused
    )


def test_target_iteration_keeps_sibling_lotteries(tmp_path):
    market = _assignment_saving_market(tmp_path, export_aggregate_metrics=False)
    _configure_reuse_market(market)
    _write_reusable_assignment(market, 0)

    def run(iteration, _policy):
        yield market._save_assignment(
            np.array([[1], [0]]),
            Policy("zones", 0, "STB"),
            iteration,
            np.array([1, 0]),
            np.array([1, 2]),
            np.zeros(2),
        )

    market._run_single_iteration_of_policy = Mock(side_effect=run)
    list(market.create_target_iteration_generator(1, include_real_match=False))

    assert saved_lotteries(_lottery_data_path(tmp_path)) == {0, 1}


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
    assert {path.name for path in output_dir.iterdir()} == {"metrics_citywide.csv"}


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
    averaged = MarketGenerator._average_aggregate_metric_frames("zip_code", [empty_zip])
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


def test_alternative_frl_thresholds_compare_after_averaging_iterations():
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {"export-aggregate-metrics": True}
    market._reset_aggregate_metric_reports()
    metric = "#GE programs above +15% district FRL"
    market._aggregate_metric_batches["citywide"] = [
        pd.DataFrame({"config_name": ["config-a"], metric: [1]}),
        pd.DataFrame({"config_name": ["config-a"], metric: [1]}),
    ]
    market._frl_threshold_input_batches = [
        pd.DataFrame(
            {
                "config_name": ["config-a", "config-a"],
                "program_id": ["A", "B"],
                "frl_assigned": [0.8, 0.2],
                "frl_non_designated": [0.92, 0.2],
                "district_frl": [0.5, 0.5],
            }
        ),
        pd.DataFrame(
            {
                "config_name": ["config-a", "config-a"],
                "program_id": ["A", "B"],
                "frl_assigned": [0.4, 0.6],
                "frl_non_designated": [0.42, 0.6],
                "district_frl": [0.5, 0.5],
            }
        ),
    ]

    market._finalize_aggregate_metric_batch()

    citywide = market._aggregate_metric_results["citywide"][0].iloc[0]
    assert citywide[metric] == 1
    assert citywide["Alternative # of GE programs above +10% district FRL"] == 1
    assert citywide["Alternative # of GE programs above +15% district FRL"] == 0
    assert (
        citywide[
            "Alternative # of GE programs above +15% district FRL (Non-Designated)"
        ]
        == 1
    )
    assert citywide["Alternative # of GE programs below -10% district FRL"] == 1
    assert citywide["Alternative # of GE programs below -15% district FRL"] == 0


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


def test_outside_option_list_length_ranks_only_programs_above_it():
    market = SimpleNamespace(
        n=2,
        num_programs=3,
        config={"utility-model": {"list-length": "outside_option"}},
        umodel=SimpleNamespace(
            original_utilities=np.array([[2.0, -1.0, 0.5], [-3.0, -2.0, -1.0]]),
            outside_utilities=np.array([0.2, -1.5]),
        ),
    )
    generator = PreferenceGenerator(market)
    eligible = np.array([[1, 1, 0], [1, 1, 1]])

    # Student 0's 0.5 is above the outside option but ineligible.
    np.testing.assert_array_equal(
        generator._number_above_outside_option(eligible), [1, 1]
    )

    market.umodel.outside_utilities = None
    with pytest.raises(ValueError, match="outside-option draw"):
        generator._number_above_outside_option(eligible)


def test_outside_option_draw_leaves_program_draws_unchanged():
    from assignment.student_assignment.market_generator.utility_model import (
        UtilityModel,
    )

    model = UtilityModel("unused.npy", programs=None, students=None)
    model._load_base_utility_matrix = lambda *args: np.zeros((3, 4))
    np.random.seed(7)
    model.draw_utility_model_randomness()
    without = model.original_utilities.copy()
    assert model.outside_utilities is None
    np.random.seed(7)
    model.draw_utility_model_randomness(outside_option=True)
    np.testing.assert_array_equal(model.original_utilities, without)
    assert model.outside_utilities.shape == (3,)

    model.draw_utility_model_randomness(gumbel_scale=0, outside_option=True)
    np.testing.assert_array_equal(model.outside_utilities, np.zeros(3))


def test_selective_high_school_offsets_and_missing_programs():
    market = SimpleNamespace(
        n=2,
        num_programs=2,
        config={},
        data_scenario=fake_scenario(year="2021"),
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
    generator._round_priorities = Mock(return_value=np.zeros((1, 2)))
    generator._set_policy_priorities = Mock(return_value=np.zeros((1, 2)))

    priorities = generator.get_priorities_without_lottery(
        Policy("zones", 0, "MTB"), np.array([[1, 0]])
    )

    np.testing.assert_array_equal(priorities, [[37, 0]])


def test_earlier_rounds_outrank_later_ones_by_ordinal():
    students = SimpleNamespace(
        first_round=np.array([0, 1, 3]),
        rounds=4,
    )
    market = SimpleNamespace(n=3, num_programs=2, students=students)
    generator = PriorityGenerator(market)

    np.testing.assert_array_equal(
        generator._round_priorities(), [[3000, 3000], [2000, 2000], [0, 0]]
    )


def test_unknown_tiebreakers_and_missing_lottery_iteration_are_fatal():
    market = SimpleNamespace(
        n=1,
        num_programs=1,
        config={"read-lotteries": False},
    )
    generator = PriorityGenerator(market)

    with pytest.raises(ValueError, match="Unknown tiebreaker"):
        generator._set_tiebreaker("unknown")

    market.config = {"read-lotteries": True}
    with pytest.raises(ValueError, match="iteration is required"):
        generator._set_tiebreaker("MTB")


def test_policy_priorities_forward_the_lottery_iteration():
    market = SimpleNamespace(n=1, num_programs=1, config={})
    generator = PriorityGenerator(market)
    generator.get_priorities_without_lottery = Mock(return_value=np.zeros((1, 1)))
    generator._set_tiebreaker = Mock(return_value=np.zeros((1, 1)))

    generator.set_policy_specific_priorities(
        Policy("zones", 0, "MTB"),
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


def _reference_metric_csv(path, report):
    """The pre-optimization writer's text and dtypes, for comparison."""
    report.to_csv(path, index=False)
    dtypes = {}
    for column in report:
        inferred = pd.api.types.infer_dtype(report[column].dropna(), skipna=True)
        if inferred == "integer":
            dtypes[column] = "Int64"
        elif inferred == "boolean":
            dtypes[column] = "boolean"
        elif inferred in {"string", "unicode"}:
            dtypes[column] = "string"
    return path.read_bytes(), dtypes


def test_metric_csv_writer_matches_reference_writer(tmp_path):
    report = pd.DataFrame(
        {
            "config_name": ["a/x", "a/y", "a/z"],
            "program_id": ["1-GE-KG", "2-GE-KG", None],
            "count": [1, 2, 3],
            "share": [0.1 + 0.2, np.nan, 1 / 3],
            "flag": [True, False, True],
            "nullable": pd.array([1, None, 3], dtype="Int64"),
            "mixed": [1.0, 2.0, np.nan],
        }
    )
    metadata = MarketGenerator._write_metric_csv(
        tmp_path / "new.csv", "citywide", report
    )
    expected_bytes, expected_dtypes = _reference_metric_csv(
        tmp_path / "old.csv", report
    )

    assert (tmp_path / "new.csv").read_bytes() == expected_bytes
    assert metadata["dtypes"] == expected_dtypes
    assert metadata["sha256"] == MarketGenerator._file_sha256(tmp_path / "new.csv")


def test_metric_csv_writer_rejects_values_that_do_not_round_trip(tmp_path):
    # "NA" is read back as missing, so the file would not mean the report.
    report = pd.DataFrame({"config_name": ["a/x"], "label": ["NA"]})

    with pytest.raises(ValueError, match="did not round-trip"):
        MarketGenerator._write_metric_csv(tmp_path / "bad.csv", "citywide", report)
