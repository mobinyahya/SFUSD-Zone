"""The zoned transportation relaxation: validity, where it bites, and caching."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from Config.Constants import AREA_ETHNICITIES
from optimization.data import contiguity
from optimization.data.mid import MidProgram, MidStudent
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solvers import get_solver
from optimization.tests.synthetic import _attach_closer_neighbors, make_grid_problem
from optimization.welfare_bounds import (
    WELFARE_BOUNDS,
    transport_upper_bound,
    welfare_upper_bound,
)
from optimization.zoned_transport import (
    ZONED_TRANSPORT_MODELS,
    market_fingerprint,
    zoned_transport_bound,
)


def _path_problem(**overrides) -> ZoneProblem:
    """Five-node path anchored at both ends, with a third school in the middle.

    School 300 at node 2 is the contested one: it can belong to only one of the
    two anchored zones, which is exactly the confinement this bound prices.
    """
    G = nx.path_graph(5)
    schools = {0: [100], 2: [300], 4: [500]}
    for node in G.nodes:
        ids = schools.get(node, [])
        G.nodes[node].update(
            {
                "area_id": 3000 + node,
                "ge_students": 1.0,
                "ge_capacity": 1.0,
                "all_prog_students": 1.0,
                "all_prog_capacity": 1.0,
                "num_schools": len(ids),
                "FRL": 0.5,
                "school_ids": ids,
                "lat": 0.0,
                "lon": float(node),
                **{ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES},
            }
        )
    G.graph["distance_dict"] = {
        source: {target: float(abs(source - target)) for target in G.nodes}
        for source in G.nodes
    }
    G.graph["F"] = 0.5
    G.graph["R"] = {ethnicity: 0.2 for ethnicity in AREA_ETHNICITIES}
    G.graph["school_data"] = {100: {}, 300: {}, 500: {}}
    _attach_closer_neighbors(G)

    params = {
        "frl_dev": 1.0,
        "racial_dev": 1.0,
        # SAA disables the capacity tolerances, and this bound is built for it.
        "overage": -1.0,
        "shortage": -1.0,
        "max_distance": float("inf"),
    }
    params.update(overrides)
    return ZoneProblem(
        G=G,
        level=LevelSpec("BlockGroup", 0),
        centroids=[0, 4],
        centroid_school_ids=[100, 500],
        **params,
    )


def _contested_market():
    """Two applicants on opposite sides, both wanting the middle school."""
    programs = (MidProgram("P", 300, 2, False, 2),)
    students = (
        MidStudent(1, ("P",), (0,), (1.0,), (100,)),
        MidStudent(3, ("P",), (0,), (1.0,), (100,)),
    )
    return programs, students


@pytest.mark.parametrize("contiguity_model", ZONED_TRANSPORT_MODELS)
def test_candidate_sets_price_the_confinement(contiguity_model):
    """The middle school has two seats but can only be zoned with one applicant.

    ``max_distance = 2`` pins node 1 to zone 0 and node 3 to zone 1, so node 2's
    membership is the only degree of freedom left and the two access indicators
    have to share one unit of mass between them. The zone-blind bound cannot see
    that and hands out both seats.
    """
    problem = _path_problem(max_distance=2.0)
    programs, students = _contested_market()

    assert problem.candidate_zones(1) == {0}
    assert problem.candidate_zones(3) == {1}
    assert problem.candidate_zones(2) == {0, 1}

    bound = zoned_transport_bound(
        programs, students, problem, contiguity_model=contiguity_model
    )

    assert transport_upper_bound(programs, students) == pytest.approx(2.0)
    assert bound.objective == pytest.approx(1.0)
    assert bound.metadata["zoned_transport_contiguity_model"] == contiguity_model
    # One access pair per applicant; both are genuinely variable here.
    assert bound.metadata["zoned_transport_access_pairs"] == 2


@pytest.mark.parametrize("contiguity_model", ZONED_TRANSPORT_MODELS)
def test_the_uniform_point_makes_the_access_relaxation_vacuous(contiguity_model):
    """With nothing pinning membership, the bound collapses to the zone-blind one.

    Splitting every unanchored vertex evenly across the zones satisfies every
    ratio constraint exactly -- each zone receives a ``1/k`` share of every
    quantity -- and drives each co-zoning indicator to 1, because the same pair
    can be co-zoned a little bit in every zone at once. This is the honest limit
    of the bound and the reason its usefulness depends on ``max_distance``, the
    anchors and the candidate sets rather than on the contiguity rows.
    """
    problem = _path_problem(max_distance=float("inf"))
    programs, students = _contested_market()

    bound = zoned_transport_bound(
        programs, students, problem, contiguity_model=contiguity_model
    )

    assert bound.objective == pytest.approx(transport_upper_bound(programs, students))


@pytest.mark.parametrize("contiguity_model", ZONED_TRANSPORT_MODELS)
@pytest.mark.parametrize("max_distance", [2.0, float("inf")])
def test_the_bound_never_exceeds_the_zone_blind_transport_bound(
    contiguity_model, max_distance
):
    """Deleting the access rows recovers the transportation polytope exactly."""
    problem = _path_problem(max_distance=max_distance)
    programs, students = _contested_market()

    bound = zoned_transport_bound(
        programs, students, problem, contiguity_model=contiguity_model
    )

    assert bound.objective <= transport_upper_bound(programs, students) + 1e-9


def test_capacity_duals_are_non_negative_admissible_prices():
    """Proposition 5 holds for any non-negative price vector, including these."""
    problem = _path_problem(max_distance=2.0)
    programs, students = _contested_market()

    bound = zoned_transport_bound(programs, students, problem)

    assert set(bound.prices) == {"P"}
    assert all(price >= 0.0 for price in bound.prices.values())


@pytest.mark.parametrize("kind", ["zoned_transport_neighbors", "zoned_transport_flow"])
def test_welfare_upper_bound_dispatches_the_zoned_kinds(kind):
    problem = _path_problem(max_distance=2.0)
    programs, students = _contested_market()

    assert kind in WELFARE_BOUNDS
    assert welfare_upper_bound(
        kind, programs, students, problem=problem
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["zoned_transport_neighbors", "zoned_transport_flow"])
def test_zoned_kinds_refuse_to_run_without_the_zoning_problem(kind):
    programs, students = _contested_market()

    with pytest.raises(ValueError, match="needs the zoning problem"):
        welfare_upper_bound(kind, programs, students)


def test_unknown_contiguity_model_is_rejected():
    problem = _path_problem()
    programs, students = _contested_market()

    with pytest.raises(ValueError, match="contiguity_model"):
        zoned_transport_bound(programs, students, problem, contiguity_model="wishful")


def test_market_fingerprint_reads_only_what_the_relaxation_reads():
    """Stability is discarded, so priorities cannot move the value or the key."""
    programs, students = _contested_market()
    reprioritized = tuple(
        MidStudent(
            student.node,
            student.programs,
            tuple(priority + 5 for priority in student.priorities),
            student.utilities,
            tuple(value + 1 for value in student.scaled_utilities),
        )
        for student in students
    )
    richer = tuple(
        MidStudent(
            student.node,
            student.programs,
            student.priorities,
            tuple(utility + 1.0 for utility in student.utilities),
            student.scaled_utilities,
        )
        for student in students
    )

    assert market_fingerprint(programs, reprioritized) == market_fingerprint(
        programs, students
    )
    assert market_fingerprint(programs, richer) != market_fingerprint(
        programs, students
    )


def test_the_solve_is_cached_and_the_worker_count_is_not_in_the_key(scenario_factory):
    """A second call at a different thread count reuses the first solve."""
    scenario = scenario_factory()
    problem = _path_problem(max_distance=2.0)
    problem.optimization_config = SimpleNamespace(data_scenario=scenario)
    programs, students = _contested_market()

    first = zoned_transport_bound(programs, students, problem, workers=1)
    second = zoned_transport_bound(programs, students, problem, workers=4)

    assert first.metadata["zoned_transport_cache"] == "miss"
    assert second.metadata["zoned_transport_cache"] == "hit"
    assert (
        second.metadata["zoned_transport_cache_key"]
        == first.metadata["zoned_transport_cache_key"]
    )
    assert second.objective == pytest.approx(first.objective)
    assert second.prices == pytest.approx(first.prices)


def test_the_contiguity_variant_is_part_of_the_cache_key(scenario_factory):
    """The two variants are different relaxations and must not share a payload."""
    scenario = scenario_factory()
    problem = _path_problem(max_distance=2.0)
    problem.optimization_config = SimpleNamespace(data_scenario=scenario)
    programs, students = _contested_market()

    neighbors = zoned_transport_bound(
        programs, students, problem, contiguity_model="neighbors"
    )
    flow = zoned_transport_bound(programs, students, problem, contiguity_model="flow")

    assert flow.metadata["zoned_transport_cache"] == "miss"
    assert (
        flow.metadata["zoned_transport_cache_key"]
        != neighbors.metadata["zoned_transport_cache_key"]
    )


def test_the_hint_is_not_part_of_the_cache_key(scenario_factory):
    """An LP solved to optimality does not care where it was warm started."""
    scenario = scenario_factory()
    programs, students = _contested_market()

    plain = _path_problem(max_distance=2.0)
    plain.optimization_config = SimpleNamespace(data_scenario=scenario)
    hinted = _path_problem(max_distance=2.0, hint={0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
    hinted.optimization_config = SimpleNamespace(data_scenario=scenario)

    first = zoned_transport_bound(programs, students, plain)
    second = zoned_transport_bound(programs, students, hinted)

    assert first.metadata["zoned_transport_cache"] == "miss"
    assert second.metadata["zoned_transport_cache"] == "hit"
    assert (
        second.metadata["zoned_transport_cache_key"]
        == first.metadata["zoned_transport_cache_key"]
    )


def test_flow_contiguity_returns_a_contiguous_zoning():
    """The replacement description is exact at binary ``x``, not merely valid."""
    problem = make_grid_problem(3, 3)
    solver = get_solver("mip", solve_time_limit=30, workers=1, contiguity_model="flow")

    solution = solver.solve(problem)

    assert solution.feasible
    assert contiguity.is_contiguous(problem.G, solution.assignment, problem.centroids)
    for zone, centroid in enumerate(problem.centroids):
        assert solution.assignment[centroid] == zone


def test_priced_access_takes_its_prices_and_a_bound_from_the_zoned_relaxation(
    monkeypatch,
):
    """The zone-aware price sources buy two things from one cached LP.

    The capacity duals are an admissible price vector by Proposition 5, and the
    LP value is a welfare bound valid at every zoning, which the strategy
    reports beside its own.
    """
    from choice import models as choice_models
    from optimization.data import saa as saa_data
    from optimization.data.saa import SaaMarket

    problem = _path_problem(max_distance=2.0)
    programs, students = _contested_market()
    market = SaaMarket(
        programs=programs,
        students=students,
        utility_student_count=len(students),
        utility_handling="omit_nonpositive",
    )
    monkeypatch.setattr(saa_data, "build_saa_market", lambda *args: market)

    model = choice_models.build_priced_access_choice_model(
        problem,
        SimpleNamespace(),
        price_source="zoned_transport_neighbors",
    )

    assert model.welfare_bound == pytest.approx(1.0)
    assert model.bound_metadata["zoned_transport_contiguity_model"] == "neighbors"
    assert all(price >= 0.0 for price in model.evaluator.prices.values())


def test_zone_blind_price_sources_report_no_zoned_bound(monkeypatch):
    from choice import models as choice_models
    from optimization.data import saa as saa_data
    from optimization.data.saa import SaaMarket

    problem = _path_problem(max_distance=2.0)
    programs, students = _contested_market()
    monkeypatch.setattr(
        saa_data,
        "build_saa_market",
        lambda *args: SaaMarket(
            programs=programs,
            students=students,
            utility_student_count=len(students),
            utility_handling="omit_nonpositive",
        ),
    )

    model = choice_models.build_priced_access_choice_model(
        problem, SimpleNamespace(), price_source="none"
    )

    assert model.welfare_bound is None
    assert model.bound_metadata == {}


def test_the_reported_certificate_is_the_smaller_of_the_two_bounds():
    """Both bounds are valid at once, so nothing is lost by taking the min."""
    from optimization.strategies.priced_access import _tightest_bound

    assert _tightest_bound(15_000.0, 14_500.0) == pytest.approx(14_500.0)
    assert _tightest_bound(14_200.0, 14_500.0) == pytest.approx(14_200.0)
    assert _tightest_bound(14_200.0, None) == pytest.approx(14_200.0)
    assert _tightest_bound(None, 14_500.0) == pytest.approx(14_500.0)
    assert _tightest_bound(None, None) is None
