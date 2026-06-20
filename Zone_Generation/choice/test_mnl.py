"""Tests for shared MNL zoning utility helpers."""

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
    assert evaluated.cuts
    assert {cut.node for cut in evaluated.cuts} == set(problem.nodes)
