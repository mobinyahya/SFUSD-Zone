from __future__ import annotations

import pandas as pd

from analysis.data_prep.convert_sfusd_transfer import Report
from analysis.data_prep.fill_from_full_student_file import (
    fill_table,
    homelang_names,
)


def _full() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ethnicity": ["African-American", "Multiple", "-"],
            "homelang": ["SP", "SP", "IT"],
            "zipcode": ["94110", "94112", ""],
            "englprof": ["E", "L", "-"],
            "sped": ["Y", "0", ""],
            "lat": ["37.7", "37.8", ""],
            "lon": ["-122.4", "-122.5", ""],
        },
        index=pd.Index([1, 2, 3], name="studentno"),
    )
    return frame


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "studentno": ["1", "2", "3"],
            "resolved_ethnicity": ["", "Chinese", ""],
            "homelang": ["", "Spanish", ""],
            "zipcode": ["", "94117.0", ""],
            "englprof": ["", "", ""],
            "sped": ["", "0", ""],
            "latitude": ["37.1", "37.2", ""],
            "longitude": ["-122.1", "-122.2", ""],
        }
    )


def test_only_blanks_are_filled_and_existing_values_are_kept():
    table = _table()
    filled, counts = fill_table(
        table,
        _full(),
        homelang={"SP": "Spanish"},
        block_indices=pd.DataFrame(),
        report=Report(year="2627", transfer="test"),
    )

    assert filled["resolved_ethnicity"].tolist() == [
        "Black or African American",
        "Chinese",  # kept, although the file says Multiple
        "",  # the file's "-" is no information
    ]
    assert filled["homelang"].tolist() == ["Spanish", "Spanish", "IT"]
    assert filled["zipcode"].tolist() == ["94110.0", "94117.0", ""]
    assert filled["sped"].tolist() == ["1", "0", ""]
    # Every student here already has coordinates or the file has none.
    assert filled["latitude"].tolist() == ["37.1", "37.2", ""]
    assert counts["latitude/longitude"] == 0
    assert counts["resolved_ethnicity"] == 1
    pd.testing.assert_frame_equal(table, _table())


def test_only_en_means_english_in_the_language_crosswalk():
    full = pd.DataFrame(
        {"homelang": ["SP"] * 12 + ["IT"] * 12},
        index=pd.Index(range(24), name="studentno"),
    )
    names = ["Spanish"] * 10 + ["English"] * 2 + ["English"] * 12
    table = pd.DataFrame({"studentno": [str(i) for i in range(24)], "homelang": names})

    assert homelang_names({"student": table}, full) == {"SP": "Spanish"}
