import numpy as np

from student_assignment.da.da import DeferredAcceptance


def test_deferred_acceptance_rejects_lower_priority_students():
    da = DeferredAcceptance(
        school_caps=np.array([1, 1]),
        student_priorities=np.array(
            [
                [10.0, 5.0],
                [20.0, 1.0],
                [1.0, 30.0],
            ]
        ),
        student_prefs=np.array(
            [
                [1, 2],
                [1, 2],
                [2, 1],
            ]
        ),
    )

    match, cutoffs, rank = da.run()

    np.testing.assert_array_equal(match, np.array([0, 1, 2]))
    np.testing.assert_array_equal(cutoffs, np.array([20.0, 30.0]))
    np.testing.assert_array_equal(rank, np.array([2, 1, 1]))


def test_deferred_acceptance_ties_replace_earlier_match():
    da = DeferredAcceptance(
        school_caps=np.array([1, 0]),
        student_priorities=np.array(
            [
                [10.0, 0.0],
                [10.0, 0.0],
            ]
        ),
        student_prefs=np.array(
            [
                [1],
                [1],
            ]
        ),
    )

    match, cutoffs, rank = da.run()

    np.testing.assert_array_equal(match, np.array([0, 1]))
    np.testing.assert_array_equal(cutoffs, np.array([10.0, 0.0]))
    np.testing.assert_array_equal(rank, np.array([1, 1]))


def test_deferred_acceptance_skips_infeasible_priorities():
    da = DeferredAcceptance(
        school_caps=np.array([1, 1]),
        student_priorities=np.array(
            [
                [-1.0, 5.0],
                [3.0, 4.0],
            ]
        ),
        student_prefs=np.array(
            [
                [1, 2],
                [1, 2],
            ]
        ),
    )

    match, cutoffs, rank = da.run()

    np.testing.assert_array_equal(match, np.array([2, 1]))
    np.testing.assert_array_equal(cutoffs, np.array([3.0, 5.0]))
    np.testing.assert_array_equal(rank, np.array([2, 1]))
