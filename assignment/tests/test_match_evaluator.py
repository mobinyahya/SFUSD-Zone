"""
Pytests for evaluation/match_evaluator.py to run the sanity tests on the metrics.
Another testing function test_run_all_metrics running all of the wrapper
functions for metrics is skipped as it is not compatible with empty data.

Example Usage: python -m pytest tests/test_match_evaluator.py -k <test_name> -s
Run all the tests in this file: python -m pytest tests/test_match_evaluator.py

Last modified: November 13th, 2023
"""

import pandas as pd
import pytest

from assignment.student_assignment.data_interfaces import Programs, Schools
from assignment.student_assignment.evaluation.match_evaluator import MatchEvaluator
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)

# Currently, we only test match evaluator without real data. If we want to test
# it with real data (e.g. with the test test_run_all_metrics) we need to change
# the data folder path and related data filenames below.
TEST_DATA_FOLDER = "path/to/real/data/folder/"
PROGRAM_FILE = TEST_DATA_FOLDER + "programs_cols.csv"
STUDENT_FILE = TEST_DATA_FOLDER + "drop_optout_cols.csv"
PROGRAM_CODES_FILE = TEST_DATA_FOLDER + "program_codes_cols.csv"
SCHOOL_FILE = TEST_DATA_FOLDER + "schools_rehauled_cols.csv"
SIMULATED_ASSIGNMENT_FILE = TEST_DATA_FOLDER + "assignments_cols.csv"


@pytest.fixture(scope="module")
def programs():
    config = {"grade": "KG", "generator": {"year": 18}}
    return Programs(PROGRAM_FILE, PROGRAM_CODES_FILE, config)


@pytest.fixture(scope="module")
def schools(programs):
    return Schools(SCHOOL_FILE, programs)


@pytest.fixture(scope="module")
def market():
    market = MarketGenerator()
    return market


# This is the match evaluator if we want to test the functions with the real
# data. The market generator it use will use the data in the paths in cofigs.
@pytest.fixture(scope="module")
def me_real(market):
    assignments = pd.read_csv(
        SIMULATED_ASSIGNMENT_FILE,
        index_col=0,
    )
    return MatchEvaluator(market.students, assignments, market.students.get_distances())


# Match Evaluator with empty initialization to test function with generated
# student data instead of real data.
class MatchEvaluatorEmpty(MatchEvaluator):
    def __init__(self):
        pass


# Run all the evaluation functions to ensure that they are compatible with the
# current environment. This test is not compatible with the empty MatchEvaluator
# so we skip it. Comment out the next line to run the test if needed.
@pytest.mark.skip(reason="Skip this test as it requires real data.")
def test_run_all_metrics(programs, schools, market, me_real):
    me_real.eval_assignment_basic()


# Test the distance metrics for average distance and fractions of distances
# above or below the given threshold with manually created distance data.
def test_metric_dist():
    me = MatchEvaluatorEmpty()
    dists = [x / 100 for x in range(1, 101)]
    dists_df = pd.DataFrame(data={"assignment_dist": dists})
    assert me.metric_dist_av(dists_df) == 0.505
    assert me.metric_dist_threshold(dists_df, 0.5, True) == 0.5
    assert me.metric_dist_threshold(dists_df, 0.51, False) == 0.5


# Test the diversity metrics with manually created students data.
def test_metric_diversity():
    me = MatchEvaluatorEmpty()
    # Test 1000 students with frl scores of 0 to 0.999 with 0.001 increments
    # divided into 100 schools evenly in increasing order of frl scores.
    frls = [x / 1000 for x in range(0, 1000)]
    assigned_schools = [str(x // 10) for x in range(0, 1000)]
    SES_categorys = [1 for _ in range(0, 1000)]
    students_df = pd.DataFrame(
        data={
            "frl": frls,
            "assigned school": assigned_schools,
            "SES_category": SES_categorys,  # used for counting in the evaluator
        }
    )

    assert me.metric_school_frl_above_district(0.1, students_df) == 0.4
    assert me.school_frl_range_district(0.1, students_df, above=False) == 0.2
    assert me.school_frl_range_district(0.1, students_df, above=True) == 0.4

    group_students = students_df[
        students_df["frl"].isin([x / 10 for x in range(0, 10)])
    ]
    assert me.metric_FRL_concentration(students_df, group_students, 0.1) == 0.4
    assert me.poverty_concentration(students_df, group_students, 0.1) == 0.4

    enrollments = students_df.groupby("assigned school").size()
    assert me.metric_dissimilarity(group_students, enrollments) == pytest.approx(0.9)

    group_students = students_df[
        students_df["frl"].isin(
            [x / 10 for x in range(0, 100)] + [x / 10 + 0.001 for x in range(0, 100)]
        )
    ]
    assert me.metric_isolation(group_students, 2) == 0
    assert me.metric_isolation(group_students, 3) == 10


def test_metric_racial_majority_schools():
    me = MatchEvaluatorEmpty()
    students = pd.DataFrame(
        {
            "assigned school": ["101"] * 3 + ["202"] * 4 + ["303"] * 3,
            "resolved_ethnicity": [
                "Asian",
                "Asian",
                "White",
                "Black",
                "Black",
                "White",
                "White",
                "Hispanic",
                None,
                None,
            ],
        }
    )

    assert me.metric_racial_majority_schools(students) == 1
    assert me.metric_racial_majority_schools(students.iloc[0:0]) == 0


# Test the school choice metrics with manually created students data.
def test_metric_choice():
    me = MatchEvaluatorEmpty()
    distributions = [0 for _ in range(50)] + [1 for _ in range(50)]
    ranks = [x // 20 + 1 for x in range(100)]
    dists = [x / 100 for x in range(1, 101)]
    students_df = pd.DataFrame(
        data={
            "programno": distributions,
            "designation": distributions,
            "rank": ranks,
            "In-Zone Rank": ranks,
            "assignment_dist": dists,
        }
    )
    assert me.metric_unassigned(students_df) == 0.5
    assert me.metric_designated(students_df) == 0.5
    assert me.metric_top_choice(students_df, 3) == 0.6
    assert me.metric_top_in_zone_choice(students_df, 3) == 0.6
    correct_output = [False] * 60 + [True] * 40
    assert (
        me.metric_dist_and_rank(students_df, 0.2, 4).values.tolist() == correct_output
    )
    correct_output = [False] * 39 + [True] * 61
    assert (
        me.metric_dist_and_rank(students_df, 0.4, 1).values.tolist() == correct_output
    )


# Test the BG cohension metric with manually created data.
def test_metric_BG_cohesion():
    me = MatchEvaluatorEmpty()
    census_blockgroup = [x // 10 for x in range(0, 1000)]
    assigned_schools = [str(x // 10) for x in range(0, 1000)]
    students_df = pd.DataFrame(
        data={
            "assigned school": assigned_schools,
            "census_blockgroup": census_blockgroup,
        }
    )

    assert me.metric_BG_cohesion(students_df, 0) == 1
    assert me.metric_BG_cohesion(students_df, 10) == 1
    assert me.metric_BG_cohesion(students_df, 11) == 0
