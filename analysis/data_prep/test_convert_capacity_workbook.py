from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.data_prep.convert_capacity_workbook import convert, scenario_name

HEADER = [
    "Scenario",
    "School code",
    "School",
    "Program",
    "Grade",
    "Ideal seats",
    "Seat type",
    "School type (this scenario)",
    "Change vs. Baseline",
    "Note",
]
AA, CW = "Attendance Area", "Citywide"


def _tab(rows: list[tuple]) -> pd.DataFrame:
    """A scenario tab: free text, a blank row, then the header and rows."""
    preamble = [["Baseline: kindergarten seats"], [], HEADER]
    return pd.DataFrame(preamble + [list(row) for row in rows])


def _baseline_rows(label: str) -> list[tuple]:
    return [
        (label, 413, "Alamo ES", "GE", "K", 66, AA, AA, "-"),
        (label, 420, "Alvarado ES", "GE", "K", 44, AA, AA, "-"),
        (label, 420, "Alvarado ES", "SP DLI", "K", 44, CW, AA, "-"),
        (label, 449, "Carmichael (Bessie) K-8", "GE", "K", 66, CW, CW, "-"),
        (label, 731, "Mission Bay ES", "GE", "K", 66, AA, AA, "-"),
        (label, "NEW", "Mandarin Immersion TK-8", "MAN DLI", "K", 66, CW, CW, "-"),
    ]


def _write(path: Path, tabs: dict[str, pd.DataFrame], summary: list[str]) -> Path:
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Date Added": None, "Scenario": summary}).to_excel(
            writer, sheet_name="Summary", index=False
        )
        for title, frame in tabs.items():
            frame.to_excel(writer, sheet_name=title, index=False, header=False)
    return path


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    w2w = [row for row in _baseline_rows("Alvarado W2W") if row[1] != 420] + [
        ("Alvarado W2W", 420, "Alvarado ES", "GE", "K", 0, CW, CW, -44),
        ("Alvarado W2W", 420, "Alvarado ES", "SP DLI", "K", 88, CW, CW, 44),
    ]
    return _write(
        tmp_path / "capacities.xlsx",
        {
            "Ideal Capacity Baseline (Aug 20": _tab(_baseline_rows("Baseline")),
            "Ideal Capacity + Alvarado W2W (": _tab(w2w),
            "Ideal Capacity + MPD Full Propo": _tab(_baseline_rows("MPD")),
            "REF Building Capacity": pd.DataFrame([["School code", "School"]]),
        },
        [
            "Ideal Capacity Baseline (Aug 2026)",
            "Ideal Capacity + Alvarado W2W (10.23.26)",
            "Ideal Capacity + MPD Full Proposal (10.23.26)",
        ],
    )


def test_scenario_names_drop_dates_and_the_word_capacity():
    assert scenario_name("Ideal Capacity Baseline (Aug 2026)", "2627") == (
        "ideal_baseline_2627"
    )
    assert scenario_name("Ideal Capacity + WestPortal W2W (10.23.26)", None) == (
        "ideal_westportal_w2w"
    )


def test_every_scenario_tab_becomes_capacity_and_category_columns(
    workbook,
):
    frame = convert(workbook, suffix="2627").set_index(["SchNum", "PathwayCode"])

    names = ["ideal_baseline_2627", "ideal_alvarado_w2w_2627"]
    assert [column for column in frame.columns if column.endswith("_Capacity")] == [
        f"Scenario_{name}_Capacity" for name in names
    ]
    base, w2w = (f"Scenario_{name}" for name in names)

    assert frame.loc[(420, "SE+SN"), f"{base}_Capacity"] == 44
    assert frame.loc[(420, "SE+SN"), f"{w2w}_Capacity"] == 88
    assert frame.loc[(420, "GE"), f"{w2w}_Capacity"] == 0
    assert frame.loc[(420, "SE+SN"), f"{w2w}_AttendanceAreaCapacity"] == 0
    assert frame.loc[(420, "GE"), f"{base}_AttendanceAreaCapacity"] == 44
    assert frame.loc[(449, "GE"), f"{base}_AttendanceAreaCapacity"] == 0
    # Alvarado holds attendance-area GE seats in the baseline, so it is an
    # attendance-area school there even though its DLI seats are citywide.
    assert frame.loc[(420, "SE+SN"), f"{base}_Category"] == "Attendance"
    assert frame.loc[(420, "SE+SN"), f"{w2w}_Category"] == "Citywide"
    assert frame.loc[(449, "GE"), f"{base}_Category"] == "Citywide"


def test_school_codes_are_corrected_and_new_schools_get_their_placeholder(
    workbook,
):
    frame = convert(workbook, suffix="2627").set_index(["SchNum", "PathwayCode"])

    assert frame.loc[(1731, "GE"), "SheetSchoolCode"] == 731
    assert frame.loc[(400, "ME+MN"), "SchoolName"] == "Mandarin Immersion TK-8"
    assert frame.loc[(400, "ME+MN"), "Scenario_ideal_baseline_2627_Capacity"] == 66
    assert frame.index.get_level_values("SchNum").tolist() == sorted(
        frame.index.get_level_values("SchNum")
    )


def test_unknown_labels_and_uncoded_schools_stop_the_conversion(tmp_path):
    summary = ["Ideal Capacity Baseline (Aug 2026)"]
    title = "Ideal Capacity Baseline (Aug 20"
    unknown_program = [("Baseline", 413, "Alamo ES", "ARB DLI", "K", 22, CW, AA, "-")]
    with pytest.raises(ValueError, match=r"unmapped program labels \['ARB DLI'\]"):
        convert(_write(tmp_path / "a.xlsx", {title: _tab(unknown_program)}, summary))

    uncoded = [("Baseline", "NEW", "Arabic Academy", "GE", "K", 22, CW, CW, "-")]
    with pytest.raises(ValueError, match="add them to NEW_SCHOOL_CODES"):
        convert(_write(tmp_path / "b.xlsx", {title: _tab(uncoded)}, summary))


def test_a_mislabelled_school_type_is_reported_and_the_seats_win(tmp_path, capsys):
    rows = [("Baseline", 413, "Alamo ES", "GE", "K", 66, AA, CW, "-")]
    path = _write(
        tmp_path / "c.xlsx",
        {"Ideal Capacity Baseline (Aug 20": _tab(rows)},
        ["Ideal Capacity Baseline (Aug 2026)"],
    )

    frame = convert(path)

    assert frame["Scenario_ideal_baseline_Category"].tolist() == ["Attendance"]
    assert "Alamo ES is labelled 'Citywide'" in capsys.readouterr().err


def test_a_program_split_between_seat_types_keeps_its_attendance_area_share(
    tmp_path,
):
    rows = [
        ("Baseline", 413, "Alamo ES", "GE", "K", 44, AA, AA, "-"),
        ("Baseline", 413, "Alamo ES", "GE", "K", 22, CW, AA, "-"),
    ]
    path = _write(
        tmp_path / "d.xlsx",
        {"Ideal Capacity Baseline (Aug 20": _tab(rows)},
        ["Ideal Capacity Baseline (Aug 2026)"],
    )

    (row,) = convert(path).to_dict("records")

    assert row["Scenario_ideal_baseline_Capacity"] == 66
    assert row["Scenario_ideal_baseline_AttendanceAreaCapacity"] == 44
    assert row["Scenario_ideal_baseline_Category"] == "Attendance"

    with pytest.raises(ValueError, match="repeats rows"):
        convert(
            _write(
                tmp_path / "e.xlsx",
                {"Ideal Capacity Baseline (Aug 20": _tab([rows[0], rows[0]])},
                ["Ideal Capacity Baseline (Aug 2026)"],
            )
        )
