"""Synthetic checks for the outside-option MNL estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from choice.estimation.exp8 import Design, build_design, ranked_program_ids
from choice.estimation.likelihood import OUTSIDE, ChoiceData, OutsideOptionMNL


def _synthetic(seed: int = 0, n: int = 3000):
    rng = np.random.default_rng(seed)
    schools = np.repeat(np.arange(8), 2)
    types = np.tile(["GE", "SE"], 8)
    J = len(schools)
    features = rng.normal(size=(n, J, 2))
    eligible = np.ones((n, J), dtype=bool)
    eligible[: n // 3, 1::2] = False
    design = Design(
        student_ids=np.arange(n),
        program_ids=np.array([f"{s}-{t}-KG" for s, t in zip(schools, types)]),
        school_ids=schools,
        program_types=types,
        eligible=eligible,
        features=features,
        feature_names=("x0", "x1"),
        low_income=np.zeros(n, dtype=bool),
        located=np.ones(n, dtype=bool),
    )
    z = np.column_stack([rng.random(n) < 0.4, rng.random(n) < 0.3]).astype(float)
    school_fe = np.linspace(-0.5, 1.5, 8)
    truth = {
        "school": school_fe,
        "SE": -0.7,
        "beta": np.array([1.0, -0.5]),
        "gamma": np.array([0.8, -1.0]),
    }
    V = (
        school_fe[schools][None]
        + np.where(types == "SE", truth["SE"], 0.0)[None]
        + features @ truth["beta"]
    )
    V = np.where(eligible, V, -np.inf)
    V0 = z @ truth["gamma"]

    rankings, offers, enrolled = [], [], []
    for i in range(n):
        u = V[i] + rng.gumbel(size=J)
        order = [j for j in np.argsort(-u) if eligible[i, j]]
        rankings.append(order[: rng.integers(1, 6)])
        seat = int(rng.choice(np.nonzero(eligible[i])[0]))
        # Fresh draws: the estimator treats enrolment as independent of the ranking.
        take = V[i, seat] + rng.gumbel() > V0[i] + rng.gumbel()
        offers.append([seat])
        enrolled.append(seat if take else OUTSIDE)
    data = ChoiceData(design, rankings, offers, enrolled, z, ("outside_a", "outside_b"))
    return data, truth


def test_gradient_matches_finite_differences():
    data, _ = _synthetic(n=200)
    model = OutsideOptionMNL(data)
    theta = np.random.default_rng(1).normal(scale=0.3, size=len(model.names))
    _, grad = model.objective(theta)
    step = 1e-6
    for p in range(len(theta)):
        e = np.zeros_like(theta)
        e[p] = step
        numeric = (model.objective(theta + e)[0] - model.objective(theta - e)[0]) / (
            2 * step
        )
        assert abs(numeric - grad[p]) < 1e-4 * max(1.0, abs(numeric))


def test_recovers_the_outside_option_level():
    data, truth = _synthetic()
    fit = OutsideOptionMNL(data, ridge=1e-6).fit()
    assert fit.converged
    estimate = dict(zip(fit.names, fit.theta))
    se = dict(zip(fit.names, fit.std_error))
    for s, value in enumerate(truth["school"]):
        assert abs(estimate[f"school_{s}"] - value) < 4 * se[f"school_{s}"]
    assert abs(estimate["program_type_SE"] - truth["SE"]) < 4 * se["program_type_SE"]
    for k, value in enumerate(truth["beta"]):
        assert abs(estimate[f"x{k}"] - value) < 4 * se[f"x{k}"]
    for name, value in zip(("outside_a", "outside_b"), truth["gamma"]):
        assert abs(estimate[name] - value) < 4 * se[name]


def test_without_enrolment_the_level_is_unidentified():
    data, _ = _synthetic(n=400)
    blind = ChoiceData(
        data.design,
        data.rankings,
        data.offers,
        [None] * len(data.enrolled),
        data.outside_features,
        data.outside_names,
    )
    model = OutsideOptionMNL(blind, ridge=0.0)
    theta = np.random.default_rng(2).normal(size=len(model.names))
    shifted = theta.copy()
    shifted[:8] += 3.0  # every school FE: a pure level shift
    assert np.isclose(model.objective(theta)[0], model.objective(shifted)[0])


def test_design_flags_feeder_and_language_eligibility():
    students = pd.DataFrame(
        {
            "studentno": [1, 2],
            "homelang": ["English", "Spanish"],
            "resolved_ethnicity": ["Hispanic", "White"],
            "median_hh_income": [50_000.0, np.nan],
            "latitude": [37.75, np.nan],
            "longitude": [-122.45, np.nan],
            "sibling": ["[]", "[2]"],
            "r1_ranked_idschool": ["[1]", "[2]"],
            "r1_programs": ["['SE']", "['SB']"],
        }
    )
    programs = pd.DataFrame(
        {
            "program_id": ["1-SE-KG", "1-SN-KG", "2-SB-KG"],
            "school_id": [1, 1, 2],
            "program_type": ["SE", "SN", "SB"],
        }
    )
    schools = pd.DataFrame(
        {"school_id": [1, 2], "lat": [37.75, 37.76], "lon": [-122.45, -122.46]}
    )
    design = build_design(
        students,
        programs,
        schools,
        revealed=ranked_program_ids(students),
        feeders={1: "1-SE-KG"},
    )
    f = dict(zip(design.feature_names, np.moveaxis(design.features, -1, 0)))
    # English speaker ranked only SE: no Spanish-family eligibility, SE always open.
    assert design.eligible[0].tolist() == [True, False, False]
    # Spanish home language opens the family without ranking it.
    assert design.eligible[1].tolist() == [True, True, True]
    assert f["prog_SE_x_English_x_ethn_match1"][0, 0] == 1
    assert f["lang_match_SB"][1, 2] == 1 and f["lang_match_not_XB"][1, 1] == 1
    assert f["sibling"][1].tolist() == [0, 0, 1]
    assert f["promote_feeder"][0].tolist() == [1, 0, 0]
    # Missing income is not low income; missing coordinates zero every distance term.
    assert design.low_income.tolist() == [True, False]
    assert f["distance_indicator_lt_0_5"][0, 0] == 1
    assert (
        not f["distance_x_low_income0"][1].any()
        and not f["distance_indicator_lt_0_5"][1].any()
    )
