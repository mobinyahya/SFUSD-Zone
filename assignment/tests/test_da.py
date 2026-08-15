import numpy as np

from assignment.student_assignment.da.da import DeferredAcceptance
from assignment.student_assignment.da.da_with_guardrails import DAwithGuards


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


def test_strict_guardrails_leave_unreserved_programs_open():
    da = DAwithGuards(
        SchoolCaps=np.array([1, 0]),
        StudentPrts=np.array([[10.0, -1.0]]),
        StudPrefs=np.array([[1, 0]]),
        classOfStudent=np.array([0]),
        strictGuards=1,
    )
    da.setguards(
        program_reserve_frac=np.array([[0.0, 0.0], [0.0, 0.0]]),
        numOfClasses=2,
    )

    match, rank = da.run()

    np.testing.assert_array_equal(match, np.array([1]))
    np.testing.assert_array_equal(rank, np.array([1]))


def test_deferred_acceptance_zero_sentinel_does_not_take_last_program_seat():
    da = DeferredAcceptance(
        school_caps=np.array([0, 1]),
        student_priorities=np.array(
            [
                [0.0, 20.0],
                [0.0, 10.0],
            ]
        ),
        student_prefs=np.array(
            [
                [0, 0],
                [2, 0],
            ]
        ),
    )

    match, cutoffs, _ = da.run()

    np.testing.assert_array_equal(match, np.array([0, 2]))
    np.testing.assert_array_equal(cutoffs, np.array([0.0, 10.0]))
    assert da.schools[-1].matches == {1}


def test_guardrail_da_zero_sentinel_does_not_take_last_program_seat():
    da = DAwithGuards(
        SchoolCaps=np.array([0, 1]),
        StudentPrts=np.array(
            [
                [0.0, 20.0],
                [0.0, 10.0],
            ]
        ),
        StudPrefs=np.array(
            [
                [0, 0],
                [2, 0],
            ]
        ),
        classOfStudent=np.array([0, 0]),
    )
    da.setguards(
        program_reserve_frac=np.array([[0.0], [0.0]]),
        numOfClasses=1,
    )

    match, _ = da.run()

    np.testing.assert_array_equal(match, np.array([0, 2]))


def test_deferred_acceptance_one_program_exhaustion_terminates():
    da = DeferredAcceptance(
        school_caps=np.array([1]),
        student_priorities=np.array([[20.0], [10.0]]),
        student_prefs=np.array([[1], [1]]),
    )

    match, cutoffs, rank = da.run()

    np.testing.assert_array_equal(match, np.array([1, 0]))
    np.testing.assert_array_equal(cutoffs, np.array([20.0]))
    np.testing.assert_array_equal(rank, np.array([1, 1]))


def test_guardrail_da_one_program_exhaustion_terminates():
    da = DAwithGuards(
        SchoolCaps=np.array([1]),
        StudentPrts=np.array([[20.0], [10.0]]),
        StudPrefs=np.array([[1], [1]]),
        classOfStudent=np.array([0, 0]),
    )
    da.setguards(
        program_reserve_frac=np.array([[0.0]]),
        numOfClasses=1,
    )

    match, rank = da.run()

    np.testing.assert_array_equal(match, np.array([1, 0]))
    np.testing.assert_array_equal(rank, np.array([1, 1]))
