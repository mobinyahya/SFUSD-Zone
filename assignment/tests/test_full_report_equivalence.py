"""The array-based metric reports equal the DataFrame reports they replaced.

``legacy_match_evaluator.py`` is the evaluator as of origin/main cd5be7f,
before the full report moved to ``evaluation/full_report.py``. Every report a
matching run writes per lottery (citywide, program, ZIP code, attendance area,
zone, FRL-threshold inputs, heatmap) is built by both and compared cell by
cell: same columns in the same order, same dtypes, same NaNs, same strings,
integers exactly and floats to 1e-9 relative (with no absolute slack, so a
zero must stay a zero).

The synthetic markets always run and are built to reach the edge cases: areas
whose students are all unassigned, one-student areas, empty demographic
groups, missing incomes and distances, designated students without a listed
rank, utilities and outside options, overage seats, and both ``Float64`` and
``float64`` FRL columns. The real-data test replays two lotteries of the
2026-27 attendance-area simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from assignment.student_assignment.choice_ranks import ASSIGNMENT_SCHEMA_VERSION
from assignment.student_assignment.evaluation.match_evaluator import (
    MatchEvaluator as NewEvaluator,
)
from assignment.tests.legacy_match_evaluator import MatchEvaluator as OldEvaluator

REPOSITORY = Path(__file__).resolve().parents[2]

RAW_ETHNICITIES = [
    "Chinese",
    "Filipino",
    "Asian Indian",
    "Hispanic/Latino",
    "Hispanic/Latinx",
    "White",
    "Middle Eastern/Arabic",
    "Black or African American",
    "Samoan",
    "Other Pacific Islander",
    "Two or More",
    "Decline to State",
    "Not Specified",
    "American Indian",
]
PROGRAM_TYPES = ["GE", "GE", "CE", "SN", "ME", "CB", "FB", "SA"]


def assert_frames_equivalent(old: pd.DataFrame, new: pd.DataFrame, label: str):
    assert list(old.columns) == list(new.columns), label
    pd.testing.assert_frame_equal(
        old,
        new,
        check_exact=False,
        rtol=1e-9,
        atol=0.0,
        obj=label,
    )


def assert_reports_equivalent(old: dict, new: dict, label: str):
    assert list(old) == list(new), label
    for name in old:
        assert_frames_equivalent(old[name], new[name], f"{label}/{name}")


def _reports(evaluator, config_name, zone_file=None, building_block=None):
    reports = evaluator.eval_aggregate_metric_reports(config_name)
    reports["frl_threshold_inputs"] = evaluator.eval_frl_threshold_inputs(config_name)
    if zone_file is not None:
        reports["zone"] = evaluator.eval_assignment_metrics_by_zone(
            config_name, zone_file, building_block
        )
        reports["heatmap"] = evaluator.eval_heatmap_data(config_name, building_block)
    return reports


def _synthetic_market(seed, *, n_students=400, frl_dtype="Float64", utilities=False):
    rng = np.random.default_rng(seed)
    school_ids = [101, 202, 303, 404, 505, 606, 1731]
    schools = pd.DataFrame(
        {
            "school_id": school_ids,
            "school_name": [f"School {s}" for s in school_ids],
            "category": ["Attendance"] * 5 + ["Citywide", "Attendance"],
            "lat": 37.7 + rng.random(len(school_ids)) / 10,
            "lon": -122.5 + rng.random(len(school_ids)) / 10,
            "Block": [11, 12, 13, 14, 15, 16, 17],
        }
    )
    rows = []
    for school in school_ids:
        types = rng.choice(PROGRAM_TYPES, size=3, replace=False)
        if school != 606:
            types = ["GE", *[t for t in types if t != "GE"]]
        for program_type in dict.fromkeys(types):
            rows.append((f"{school}-{program_type}-KG", school, program_type))
    programs = pd.DataFrame(rows, columns=["program_id", "school_id", "program_type"])
    programs["programno"] = np.arange(1, len(programs) + 1)
    programs["capacity"] = rng.integers(0, 40, len(programs)).astype(float)
    programs.loc[programs.index[0], "capacity"] = 0.0

    n = n_students
    frl = rng.choice([0.0, 1 / 3, 0.5, 2 / 3, 1.0, 1 / 6, 0.123456789], n)
    income = rng.choice([40_000.0, 95_292.0, 110_850.0, 150_000.0, np.nan], n)
    zipcode = rng.choice([94110.0, 94112.0, 94116.0, 94118.0, 94122.0], n)
    attendance = rng.choice(school_ids[:5], n).astype(float)
    # One-student, all-unassigned and missing-geography areas.
    zipcode[:3] = [94199.0, 94198.0, 94198.0]
    attendance[:3] = [999.0, 998.0, 998.0]
    zipcode[3] = np.nan
    attendance[4] = np.nan
    students = pd.DataFrame(
        {
            "studentno": np.arange(1, n + 1) * 7 + 1000,
            "census_block": rng.choice([11, 12, 13, 14, 15, 16, 17], n),
            "latitude": 37.7 + rng.random(n) / 10,
            "longitude": -122.5 + rng.random(n) / 10,
            "freelunch_prob": pd.array(frl, dtype=frl_dtype),
            "reducedlunch_prob": np.where(rng.random(n) < 0.05, np.nan, 0.0),
            "resolved_ethnicity": rng.choice(RAW_ETHNICITIES, n),
            "median_hh_income": income,
            "ctip1": rng.choice([0.0, 1.0, np.nan], n),
            "et2": rng.choice([0, 1], n),
            "zipcode": zipcode,
            "idschoolattendance": attendance,
        }
    )
    distances = pd.DataFrame(
        rng.random((n, len(programs))) * 6,
        index=pd.Index(students["studentno"], name="studentno"),
        columns=programs["program_id"],
    )
    distances = distances.mask(rng.random(distances.shape) < 0.03)

    lotteries = []
    for _ in range(3):
        assigned = rng.random(n) < 0.85
        assigned[1:3] = False
        program = rng.integers(0, len(programs), n)
        designation = (assigned & (rng.random(n) < 0.25)).astype(int)
        rank = np.where(
            assigned & ~((designation == 1) & (rng.random(n) < 0.6)),
            rng.integers(1, 12, n),
            np.nan,
        ).astype(float)
        mechanism = np.where(assigned, rng.integers(1, 9, n), np.nan).astype(float)
        assignments = pd.DataFrame(
            {
                "assignment_schema_version": ASSIGNMENT_SCHEMA_VERSION,
                "studentno": students["studentno"],
                "programno": np.where(
                    assigned, programs["programno"].to_numpy()[program], 0
                ),
                "programcodes": np.where(
                    assigned, programs["program_id"].to_numpy()[program], ""
                ),
                "rank_basis": "listed",
                "submitted_rank": rank,
                "rank_excluding_promotion": rank,
                "utility_rank": np.nan,
                "rank": rank,
                "mechanism_rank": mechanism,
                "designation": designation,
                "In-Zone Rank": mechanism,
            }
        )
        if utilities:
            assignments["assigned_utility"] = np.where(
                assigned, rng.normal(size=n), np.nan
            )
            assignments.loc[assignments.index[5], "assigned_utility"] = -np.inf
            assignments["outside_utility"] = rng.normal(size=n)
        lotteries.append(assignments.sample(frac=1, random_state=seed))
    return students, lotteries, programs, schools, distances


def _zone_file(tmp_path):
    path = tmp_path / "zones.csv"
    path.write_text("11,12,13\n14,15\n16,17\n")
    return path


@pytest.mark.parametrize(
    ("seed", "frl_dtype", "utilities", "overscribe_aa"),
    [
        (0, "Float64", False, False),
        (1, "float64", True, False),
        (2, "Float64", True, True),
        (3, "float64", False, True),
    ],
)
def test_synthetic_reports_match_legacy(
    tmp_path, seed, frl_dtype, utilities, overscribe_aa
):
    students, lotteries, programs, schools, distances = _synthetic_market(
        seed, frl_dtype=frl_dtype, utilities=utilities
    )
    kwargs = dict(
        program_data=programs,
        schools_data=schools,
        distance_cache=distances,
        overscribe_aa=overscribe_aa,
    )
    old = OldEvaluator(students, lotteries[0], **kwargs)
    new = NewEvaluator(students, lotteries[0], **kwargs)
    zone_file = _zone_file(tmp_path)
    for index, assignments in enumerate(lotteries):
        if index:
            old.update_assignments(assignments)
            new.update_assignments(assignments)
        pd.testing.assert_frame_equal(old.student_data, new.student_data)
        label = f"seed{seed}/lottery{index}"
        assert_reports_equivalent(
            _reports(old, label, zone_file, "block"),
            _reports(new, label, zone_file, "block"),
            label,
        )
        pd.testing.assert_series_equal(
            old.eval_assignment_full(), new.eval_assignment_full(), rtol=1e-9, atol=0
        )
        subset = old.student_data.iloc[::3]
        pd.testing.assert_series_equal(
            old.eval_assignment_full(subset),
            new.eval_assignment_full(subset),
            rtol=1e-9,
            atol=0,
        )


def test_synthetic_reports_with_overage_seat_column_match_legacy():
    students, lotteries, programs, schools, distances = _synthetic_market(4)
    kwargs = dict(
        program_data=programs,
        schools_data=schools,
        distance_cache=distances,
        overscribe_aa=True,
    )
    assignments = lotteries[0]
    ge = assignments["programcodes"].isin(
        programs.loc[programs["program_type"].eq("GE"), "program_id"]
    )
    assignments["overage_seat"] = ge & (np.arange(len(assignments)) % 4 == 0)
    old = OldEvaluator(students, assignments, **kwargs)
    new = NewEvaluator(students, assignments, **kwargs)
    assert_reports_equivalent(_reports(old, "c"), _reports(new, "c"), "overage")


def test_parallel_area_reports_match_legacy():
    students, lotteries, programs, schools, distances = _synthetic_market(5)
    kwargs = dict(program_data=programs, schools_data=schools, distance_cache=distances)
    old = OldEvaluator(students, lotteries[0], **kwargs)
    new = NewEvaluator(students, lotteries[0], **kwargs)
    new.metric_workers = 2
    assert_reports_equivalent(_reports(old, "c"), _reports(new, "c"), "parallel")


def test_reports_without_area_metrics_match_legacy():
    students, lotteries, programs, schools, distances = _synthetic_market(6)
    kwargs = dict(program_data=programs, schools_data=schools, distance_cache=distances)
    old = OldEvaluator(students, lotteries[0], **kwargs)
    new = NewEvaluator(students, lotteries[0], **kwargs)
    for options in (
        dict(include_area_metrics=False),
        dict(include_local_metrics=False),
    ):
        assert_reports_equivalent(
            old.eval_aggregate_metric_reports("c", **options),
            new.eval_aggregate_metric_reports("c", **options),
            str(options),
        )


POLICY = "restrictive_aa_sib_ctip_reserves"
REAL_FILTERS = {
    "capacity_scenario": "ideal_webster_w2w_10_23_26",
    "student_population": "enrolled_all",
}
REAL_DATA = {
    "scenario": "attendance-areas-2627",
    "overrides": {
        "filters": {"optimization": REAL_FILTERS, "assignment": REAL_FILTERS}
    },
}


@pytest.mark.real_data
def test_real_lottery_reports_match_legacy(tmp_path, monkeypatch):
    from assignment.student_assignment.market_generator import (
        school_choice_market_generator as generator,
    )

    captured = []

    def capture(self, assignment_df, save_name, iteration, policy):
        captured.append(
            dict(
                scenario=self.data_scenario,
                assignments=assignment_df.copy(),
                programs=self.programs.program_df.copy(),
                distances=self.students.distance_data,
                zone_file=self.zone_file(policy),
                building_block=self.config["zone-building-blocks"],
            )
        )

    monkeypatch.setattr(
        generator.MarketGenerator, "_record_assignment_metric_reports", capture
    )
    config = yaml.safe_load(
        (
            REPOSITORY
            / "assignment/configs/attendance_areas/restrictive_aa_sib_2627.yaml"
        ).read_text()
    )
    config.update(
        data=REAL_DATA,
        subconfigs=[POLICY],
        iterations={"start": 0, "end": 2},
        reuse_assignments=False,
    )
    config["paths"]["assignment-folder"] = str(tmp_path)
    config["output_dir"] = f"{tmp_path}/"
    try:
        generator.MarketGenerator(
            config=config, assignment_path=str(tmp_path)
        ).simulate()
    except FileNotFoundError as error:
        pytest.skip(f"real data not available: {error}")
    assert len(captured) == 2

    first = captured[0]
    for overscribe_aa in (False, True):
        evaluators = [
            cls.from_scenario(
                first["scenario"],
                first["assignments"],
                program_data=first["programs"],
                distance_cache=first["distances"],
                overscribe_aa=overscribe_aa,
            )
            for cls in (OldEvaluator, NewEvaluator)
        ]
        for index, lottery in enumerate(captured):
            reports = []
            for evaluator in evaluators:
                if index:
                    evaluator.update_assignments(lottery["assignments"])
                reports.append(
                    _reports(
                        evaluator,
                        f"real/{index}",
                        lottery["zone_file"],
                        lottery["building_block"],
                    )
                )
            assert len(reports[0]["attendance_area"]) > 10
            assert_reports_equivalent(
                *reports, f"real/overscribe={overscribe_aa}/lottery{index}"
            )
