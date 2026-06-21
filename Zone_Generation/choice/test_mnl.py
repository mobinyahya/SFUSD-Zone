"""Tests for shared MNL zoning utility helpers."""

import warnings

import numpy as np
import pandas as pd
import pytest

from Zone_Generation.choice import mnl
from Zone_Generation.choice.models import MNLChoiceModel
from Zone_Generation.optimization.tests.synthetic import make_grid_problem


def test_mnl_choice_model_evaluates_and_builds_cuts(tmp_path, monkeypatch):
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
        }
    ).to_csv(student_path, index=False)

    problem = make_grid_problem(2, 2)
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    monkeypatch.setattr(mnl, "DEFAULT_UTILITY_PATH", str(utility_path))
    monkeypatch.setattr(mnl, "DEFAULT_STUDENT_PATH", str(student_path))
    model = MNLChoiceModel(method="max")

    evaluated = model.evaluate_with_cuts(problem, assignment)

    assert evaluated.utility == pytest.approx(6.0)
    assert model.preassignment_utility(problem, assignment) == pytest.approx(
        evaluated.utility
    )
    assert evaluated.cuts
    assert {cut.node for cut in evaluated.cuts} == set(problem.nodes)


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
