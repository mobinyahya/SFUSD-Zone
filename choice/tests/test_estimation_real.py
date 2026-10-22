"""The rebuilt exp8 design against the published 2023-24 utilities and the 2025-26 transfer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from choice.estimation.exp8 import build_design, published_utilities, ranked_program_ids
from loaders import load_scenario

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def root():
    path = load_scenario({"scenario": "legacy", "overrides": {}}).roots["data"]
    published = path / "simulation-files/choice-model/ChoiceModel_20240514"
    if not (published / "estimates_2324.csv").exists():
        pytest.skip("shared choice-model data is not available")
    return path


def test_design_reproduces_published_2324_utilities(root):
    published = root / "simulation-files/choice-model/ChoiceModel_20240514"
    cleaned = root / "Data/Cleaned"
    weights = (
        pd.read_csv(published / "weights.csv")
        .set_index("feature")["coefficient"]
        .to_dict()
    )
    estimates = pd.read_csv(published / "estimates_2324.csv")
    estimates["studentno"] = (
        estimates["studentno"].str.removeprefix("2324-").astype(int)
    )
    estimates = estimates.set_index("studentno")
    students = pd.read_csv(
        cleaned / "student_without_specialprogs_2324.csv", low_memory=False
    )
    students = (
        students[students["grade"] == "KG"]
        .set_index("studentno")
        .loc[estimates.index]
        .reset_index()
    )
    schools = pd.read_csv(cleaned / "schools_rehauled_withMissionBay_2324.csv")
    parts = estimates.columns.str.split("-")
    programs = pd.DataFrame(
        {
            "program_id": estimates.columns,
            "school_id": parts.str[0].astype(int),
            "program_type": parts.str[1],
        }
    )

    design = build_design(
        students, programs, schools, revealed=ranked_program_ids(students)
    )
    ours = published_utilities(design, weights)
    theirs = estimates.to_numpy()

    both = np.isfinite(ours) & np.isfinite(theirs)
    gap = np.abs(ours[both] - theirs[both])
    # The original distance table differs from haversine by ~0.01 mi. That is
    # all the noise there is, apart from a handful of pairs it moves across the
    # 0.5-mile indicator, which miss by exactly that coefficient.
    assert np.quantile(gap, 0.999) < 0.02
    boundary = gap > 0.05
    assert boundary.sum() < 50
    np.testing.assert_allclose(
        gap[boundary], weights["distance_indicator_lt_0_5"], atol=0.02
    )
    agree = np.isfinite(ours) == np.isfinite(theirs)
    assert agree.mean() > 0.998
    assert agree.all(axis=1).mean() > 0.985


def test_2526_choice_data_counts(root):
    from choice.estimation.fit_transfer_year import build_choice_data

    data, students, counts = build_choice_data("2526")
    assert counts["market_students"] == 3980
    assert len(students) == 3980 - counts["students_dropped_special_seat"]
    decisions = [c for c in data.enrolled if c is not None]
    not_enrolled = sum(c == -1 for c in decisions) / len(decisions)
    assert 0.17 < not_enrolled < 0.19
    # Every promotion-eligible student with a ranking ranks their feeder.
    feeder = data.design.features[
        ..., data.design.feature_names.index("promote_feeder")
    ]
    for i in np.nonzero(feeder.any(axis=1))[0]:
        if data.rankings[i]:
            assert feeder[i, data.rankings[i]].any()
