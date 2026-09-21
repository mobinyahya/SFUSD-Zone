"""Tests for the congestion-priced access surrogate and its cuts.

The whole point of this surrogate is that its cuts are valid upper bounds for
*every* zoning, not just the one they were built at -- that is what makes the
master's objective a certificate rather than an estimate. So validity off the
anchor is the property most of these tests check.
"""

from __future__ import annotations

import itertools

import pytest

from choice.priced_access import PricedAccessUtility
from optimization.data.mid import MidProgram, MidStudent
from optimization.data.saa import SaaMarket
from optimization.tests.synthetic import make_grid_problem


def _market(students, programs) -> SaaMarket:
    return SaaMarket(
        programs=tuple(programs),
        students=tuple(students),
        utility_student_count=len(students),
        utility_handling="omit_nonpositive",
    )


def _simple() -> tuple[SaaMarket, object]:
    """Two schools at nodes 0 and 8 of a 3x3 grid, three students in between."""
    problem = make_grid_problem(3, 3)
    programs = [
        MidProgram("A", 100, 10, False, 0),
        MidProgram("B", 200, 10, False, 8),
    ]
    students = [
        MidStudent(4, ("A", "B"), (0, 0), (5.0, 3.0), (500, 300)),
        MidStudent(4, ("B", "A"), (0, 0), (4.0, 1.0), (400, 100)),
        MidStudent(1, ("A",), (0,), (2.0,), (200,)),
    ]
    return _market(students, programs), problem


def _cut_value(cut, zoning: dict[int, int]) -> float:
    value = cut.constant
    own = zoning.get(cut.node)
    for term in cut.terms:
        if own is not None and zoning.get(term.node) == own:
            value += term.coefficient
    return value


def _node_truth(
    utility: PricedAccessUtility, zoning: dict[int, int]
) -> dict[int, float]:
    truth: dict[int, float] = {}
    for index in range(len(utility._student_options)):
        node = utility._student_nodes[index]
        accessible = utility._accessible(index, zoning)
        truth[node] = truth.get(node, 0.0) + (
            accessible[0].value if accessible else 0.0
        )
    return truth


def _all_zonings(problem, zones=2):
    """Every assignment of the grid's nodes to ``zones`` zones."""
    nodes = list(problem.nodes)
    for combo in itertools.product(range(zones), repeat=len(nodes)):
        yield dict(zip(nodes, combo))


def test_price_constant_is_capacity_weighted():
    market, problem = _simple()
    utility = PricedAccessUtility(market, problem, {"A": 0.5, "B": 0.25})
    assert utility.price_constant == pytest.approx(10 * 0.5 + 10 * 0.25)


def test_prices_are_subtracted_from_utilities():
    market, problem = _simple()
    zoning = {node: 0 for node in problem.nodes}
    unpriced = PricedAccessUtility(market, problem, {}).evaluate(problem, zoning)
    priced = PricedAccessUtility(market, problem, {"A": 1.0}).evaluate(problem, zoning)
    # Everyone reachable, so both A-preferrers lose exactly the price of A; the
    # student who prefers B is untouched.
    assert unpriced == pytest.approx(5.0 + 4.0 + 2.0)
    assert priced == pytest.approx(4.0 + 4.0 + 1.0)


def test_level_zero_cut_is_tight_at_the_anchor():
    market, problem = _simple()
    utility = PricedAccessUtility(market, problem, {}, cut_levels=1)
    zoning = {node: (0 if node < 5 else 1) for node in problem.nodes}
    evaluation = utility.evaluate_with_cuts(problem, zoning)
    truth = _node_truth(utility, zoning)
    for cut in evaluation.cuts:
        assert _cut_value(cut, zoning) == pytest.approx(truth.get(cut.node, 0.0))
    assert evaluation.utility == pytest.approx(sum(truth.values()))


@pytest.mark.parametrize("cut_levels", [1, 2, 3])
def test_cuts_are_valid_at_every_other_zoning(cut_levels):
    market, problem = _simple()
    utility = PricedAccessUtility(market, problem, {"A": 0.5}, cut_levels=cut_levels)
    anchor = {node: (0 if node < 5 else 1) for node in problem.nodes}
    cuts = utility.evaluate_with_cuts(problem, anchor).cuts
    for zoning in _all_zonings(problem):
        truth = _node_truth(utility, zoning)
        for cut in cuts:
            assert _cut_value(cut, zoning) >= truth.get(cut.node, 0.0) - 1e-9


def test_initial_cuts_are_valid_everywhere_and_cover_every_node():
    market, problem = _simple()
    utility = PricedAccessUtility(market, problem, {})
    cuts = utility.initial_cuts(problem)
    # One per node: an uncut node's utility variable is bounded only by the
    # loosest node's bound, which would wreck the first master solve.
    assert {cut.node for cut in cuts} == set(problem.nodes)
    for zoning in _all_zonings(problem):
        truth = _node_truth(utility, zoning)
        for cut in cuts:
            assert _cut_value(cut, zoning) >= truth.get(cut.node, 0.0) - 1e-9


def test_student_less_nodes_are_pinned_to_zero():
    market, problem = _simple()
    utility = PricedAccessUtility(market, problem, {})
    empty = {cut.node: cut for cut in utility.initial_cuts(problem)}[7]
    assert empty.constant == pytest.approx(0.0)
    assert empty.terms == ()


def test_threshold_never_drops_below_an_unrevokable_option():
    """A citywide option is a floor: no threshold may price it away.

    Without the clamp, a deep level would set the threshold below the citywide
    value and the cut would fall under the welfare the student keeps no matter
    how the zoning is drawn.
    """
    problem = make_grid_problem(3, 3)
    programs = [
        MidProgram("CITY", 100, 10, True, None),
        MidProgram("NEAR", 200, 10, False, 0),
    ]
    students = [MidStudent(4, ("NEAR", "CITY"), (0, 0), (6.0, 3.0), (600, 300))]
    utility = PricedAccessUtility(
        _market(students, programs), problem, {}, cut_levels=4
    )
    anchor = {node: 0 for node in problem.nodes}
    cuts = utility.evaluate_with_cuts(problem, anchor).cuts
    # Cut off from NEAR the student still holds CITY, worth 3.0.
    isolated = {node: (0 if node == 4 else 1) for node in problem.nodes}
    for cut in cuts:
        if cut.node == 4:
            assert _cut_value(cut, isolated) >= 3.0 - 1e-9


def test_options_worse_than_the_outside_option_are_dropped():
    problem = make_grid_problem(3, 3)
    programs = [MidProgram("A", 100, 10, False, 0)]
    students = [MidStudent(4, ("A",), (0,), (1.0,), (100,))]
    # A price above the utility makes the option worthless, so welfare is the
    # outside option everywhere and no cut carries a term.
    utility = PricedAccessUtility(_market(students, programs), problem, {"A": 2.0})
    zoning = {node: 0 for node in problem.nodes}
    evaluation = utility.evaluate_with_cuts(problem, zoning)
    assert evaluation.utility == pytest.approx(0.0)
    assert all(cut.terms == () for cut in evaluation.cuts)


def test_node_utility_bounds_dominate_every_node():
    market, problem = _simple()
    utility = PricedAccessUtility(market, problem, {})
    lower, upper = utility.node_utility_bounds(problem)
    assert lower == 0.0
    for zoning in _all_zonings(problem):
        for value in _node_truth(utility, zoning).values():
            assert lower - 1e-9 <= value <= upper + 1e-9


def test_cut_levels_must_be_positive():
    market, problem = _simple()
    with pytest.raises(ValueError, match="cut_levels"):
        PricedAccessUtility(market, problem, {}, cut_levels=0)


def test_student_less_nodes_are_exactly_zero_not_merely_bounded():
    """A node with no students must contribute precisely 0 to the objective.

    The solver builds one utility variable per node with ``total == sum(...)``,
    so the pin has to come from both sides: the cut caps the variable at 0 and
    the lower bound floors it at 0.
    """
    market, problem = _simple()
    utility = PricedAccessUtility(market, problem, {})
    populated = set(utility._student_nodes)
    empty = [node for node in problem.nodes if node not in populated]
    assert empty, "fixture must leave some node student-less"

    lower, _ = utility.node_utility_bounds(problem)
    by_node = {cut.node: cut for cut in utility.initial_cuts(problem)}
    for node in empty:
        cut = by_node[node]
        assert cut.constant == 0.0
        assert cut.terms == ()
        # cut caps at 0, bound floors at 0 => the variable can only be 0.
        assert lower == 0.0
        for zoning in _all_zonings(problem):
            assert _cut_value(cut, zoning) == 0.0

    # And every evaluation keeps them at zero too.
    zoning = {node: 0 for node in problem.nodes}
    for cut in utility.evaluate_with_cuts(problem, zoning).cuts:
        if cut.node in empty:
            assert _cut_value(cut, zoning) == 0.0


def test_student_on_an_unmodelled_node_is_an_error():
    """Dropping such a student would desync the utility from the cuts."""
    problem = make_grid_problem(3, 3)
    programs = [MidProgram("A", 100, 10, False, 0)]
    # Node 99 is not in the 3x3 grid.
    students = [MidStudent(99, ("A",), (0,), (5.0,), (500,))]
    with pytest.raises(ValueError, match="absent from the zoning problem"):
        PricedAccessUtility(_market(students, programs), problem, {})

    # The guard also holds if a *different* problem is handed to the methods
    # later, which is how the strategy actually calls them.
    smaller = make_grid_problem(2, 2)
    market, big = _simple()
    utility = PricedAccessUtility(market, big, {})
    with pytest.raises(ValueError, match="absent from the zoning problem"):
        utility.initial_cuts(smaller)
    with pytest.raises(ValueError, match="absent from the zoning problem"):
        utility.evaluate_with_cuts(smaller, {node: 0 for node in smaller.nodes})
