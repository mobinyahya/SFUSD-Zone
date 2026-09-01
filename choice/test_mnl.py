"""Tests for shared MNL zoning utility helpers."""

import warnings

import numpy as np
import pandas as pd
import pytest

from choice import mnl
from choice.models import MNLChoiceModel, get_configured_choice_model
from loaders import load_scenario
from optimization.tests.synthetic import make_grid_problem


ASSIGNMENT_FILTERS = {
    "year": "2324",
    "grades": ["KG"],
    "student_population": "applicant",
    "rounds": [1],
    "special_programs": "include",
    "capacity_profile": "status_quo",
    "include_mission_bay": True,
}


def _scenario(utility_path, student_path):
    return load_scenario(
        {
            "scenario": "legacy",
            "overrides": {
                "sources": {
                    "choice.estimate": {"path": str(utility_path)},
                    "assignment.students": {"path": str(student_path)},
                },
                "filters": {"assignment": ASSIGNMENT_FILTERS},
            },
        },
        environ={},
    )


def test_mnl_choice_model_evaluates_and_builds_cuts(tmp_path):
    utility_path = tmp_path / "utility.csv"
    student_path = tmp_path / "students.csv"
    pd.DataFrame(
        {
            "studentno": ["2324-1", "2324-2"],
            "100-GE-KG": [2.0, 1.0],
            "200-GE-KG": [0.5, 4.0],
        }
    ).to_csv(utility_path, index=False)
    pd.DataFrame(
        {
            "studentno": [1, 2],
            "census_blockgroup": [1001, 1002],
            "grade": ["KG", "KG"],
            "r1_ranked_idschool": ["[100]", "[200]"],
            "r1_programs": ["['GE']", "['GE']"],
        }
    ).to_csv(student_path, index=False)

    problem = make_grid_problem(2, 2)
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    model = MNLChoiceModel(_scenario(utility_path, student_path), method="max")

    evaluated = model.evaluate_with_cuts(problem, assignment)

    assert evaluated.utility == pytest.approx(6.0)
    assert model.preassignment_utility(problem, assignment) == pytest.approx(
        evaluated.utility
    )
    assert evaluated.cuts
    assert {cut.node for cut in evaluated.cuts} == set(problem.nodes)


def test_mnl_choice_utility_hint_cuts_use_nearest_average_school_count(tmp_path):
    utility_path = tmp_path / "utility.csv"
    student_path = tmp_path / "students.csv"
    pd.DataFrame(
        {
            "studentno": [1],
            "100-GE-KG": [1.0],
            "200-GE-KG": [2.0],
            "300-GE-KG": [10.0],
        }
    ).to_csv(utility_path, index=False)
    pd.DataFrame(
        {
            "studentno": [1],
            "census_blockgroup": [1001],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[100]"],
            "r1_programs": ["['GE']"],
        }
    ).to_csv(student_path, index=False)

    problem = make_grid_problem(2, 2)
    problem.G.nodes[1]["school_ids"] = [300]
    problem.G.nodes[1]["num_schools"] = 1
    problem.G.graph["school_data"][300] = {}
    evaluator = mnl.MNLZoningUtility(
        _scenario(utility_path, student_path), method="max"
    )

    cuts = evaluator.choice_utility_hint_cuts(problem)

    cut = next(cut for cut in cuts if cut.node == 1 and cut.zone == 0)
    coeffs = {term.node: term.coefficient for term in cut.terms}
    assert cut.constant == pytest.approx(2.0)
    assert set(coeffs) == {1}
    assert coeffs[1] == pytest.approx(8.0)


@pytest.mark.parametrize("method", ["max", "logsum"])
def test_mnl_block_impacts_match_direct_add_remove_deltas(tmp_path, method):
    utility_path = tmp_path / "utility.csv"
    student_path = tmp_path / "students.csv"
    pd.DataFrame(
        {
            "studentno": ["2324-1"],
            "100-GE-KG": [2.0],
            "200-GE-KG": [4.0],
            "300-GE-KG": [1.0],
        }
    ).to_csv(utility_path, index=False)
    pd.DataFrame(
        {
            "studentno": [1],
            "census_blockgroup": [1001],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[100]"],
            "r1_programs": ["['GE']"],
        }
    ).to_csv(student_path, index=False)

    problem = make_grid_problem(2, 2)
    problem.G.nodes[1]["school_ids"] = [300]
    problem.G.nodes[1]["num_schools"] = 1
    problem.G.graph["school_data"][300] = {}
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    evaluator = mnl.MNLZoningUtility(
        _scenario(utility_path, student_path), method=method
    )

    prepared = evaluator._prepare(problem, assignment)
    impacts = evaluator._block_impacts(
        problem,
        assignment,
        prepared.merged,
        prepared.zone_to_cols,
        prepared.student_area_col,
    )

    def utility(cols):
        return float(evaluator._utilities_for_cols(prepared.merged, cols)[0])

    zone_cols = prepared.zone_to_cols[0]
    school_100_cols = evaluator.school_to_cols["100"]
    school_200_cols = evaluator.school_to_cols["200"]
    baseline = utility(zone_cols)
    added_cols = zone_cols + [col for col in school_200_cols if col not in zone_cols]
    remaining_cols = [col for col in zone_cols if col not in school_100_cols]

    assert impacts["1001"]["200"]["add"] == pytest.approx(
        utility(added_cols) - baseline
    )
    assert impacts["1001"]["100"]["remove"] == pytest.approx(
        baseline - utility(remaining_cols)
    )
    assert impacts["1001"]["100"]["remove"] >= 0.0


@pytest.mark.parametrize("method", ["max", "logsum"])
def test_mnl_choice_cuts_upper_bound_substitute_school(tmp_path, method):
    utility_path = tmp_path / "utility.csv"
    student_path = tmp_path / "students.csv"
    pd.DataFrame(
        {
            "studentno": ["2324-1"],
            "100-GE-KG": [10.0],
            "200-GE-KG": [9.0],
        }
    ).to_csv(utility_path, index=False)
    pd.DataFrame(
        {
            "studentno": [1],
            "census_blockgroup": [1001],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[100]"],
            "r1_programs": ["['GE']"],
        }
    ).to_csv(student_path, index=False)

    problem = make_grid_problem(2, 2)
    incumbent = {0: 0, 1: 0, 2: 1, 3: 1}
    alternative = {0: 0, 1: 1, 2: 1, 3: 1}
    evaluator = mnl.MNLZoningUtility(
        _scenario(utility_path, student_path), method=method
    )

    evaluated = evaluator.evaluate_with_cuts(problem, incumbent)
    cut = next(cut for cut in evaluated.cuts if cut.node == 1 and cut.zone == 1)
    rhs = cut.constant + sum(
        term.coefficient
        for term in cut.terms
        if alternative.get(term.node) == term.zone
    )
    true_utility = _node_utility(evaluator, problem, alternative, node=1)

    assert rhs >= true_utility - 1e-9
    assert rhs == pytest.approx(true_utility)


def test_configured_choice_model_defaults_to_mnl(tmp_path):
    utility_path = tmp_path / "utility.csv"
    student_path = tmp_path / "students.csv"
    utility_path.write_text("studentno\n", encoding="utf-8")
    student_path.write_text("studentno\n", encoding="utf-8")

    assert isinstance(
        get_configured_choice_model({}, _scenario(utility_path, student_path)),
        MNLChoiceModel,
    )


def test_mnl_uses_central_assignment_student_selection(tmp_path):
    utility_path = tmp_path / "utility.csv"
    student_path = tmp_path / "students.csv"
    pd.DataFrame(
        {
            "studentno": [1, 2],
            "100-GE-KG": [3.0, 100.0],
        }
    ).to_csv(utility_path, index=False)
    pd.DataFrame(
        {
            "studentno": [1, 2],
            "census_blockgroup": [1001, 1001],
            "grade": ["KG", "01"],
            "r1_ranked_idschool": ["[100]", "[100]"],
            "r1_programs": ["['GE']", "['GE']"],
        }
    ).to_csv(student_path, index=False)
    problem = make_grid_problem(2, 2)

    evaluator = mnl.MNLZoningUtility(
        _scenario(utility_path, student_path), method="max"
    )

    assert evaluator.preassignment_utility(
        problem, {0: 0, 1: 0, 2: 1, 3: 1}
    ) == pytest.approx(3.0)
    assert evaluator.student_df["studentno"].tolist() == [1]


def test_log_helpers_handle_extreme_values_without_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        logsum = mnl._logsumexp(
            np.array([[-np.inf, -np.inf], [1000.0, 999.0]]),
            axis=1,
        )
        softplus = mnl._log1pexp(np.array([1000.0, -1000.0, 0.0]))

    assert np.isneginf(logsum[0])
    assert logsum[1] == pytest.approx(1000.0 + np.log1p(np.exp(-1.0)))
    assert softplus[0] == pytest.approx(1000.0)
    assert softplus[1] == pytest.approx(0.0)
    assert softplus[2] == pytest.approx(np.log(2.0))


def _node_utility(evaluator, problem, assignment, node: int) -> float:
    utility = evaluator._preassignment_utility(problem, assignment)
    return sum(
        float(utility.block_utilities.loc[block_id])
        for block_id in mnl._node_area_ids(problem.G.nodes[node])
        if block_id in utility.block_utilities.index
    )
