"""Tests for constructing the cutoff assignment market."""

from types import SimpleNamespace

import pandas as pd

from optimization.data.cutoffs import _zone_restricted_schools


def test_zone_restricted_schools_are_attendance_schools_with_ge():
    market = SimpleNamespace(
        schools=SimpleNamespace(
            school_df=pd.DataFrame(
                {
                    "category": ["Attendance", "Attendance", "Citywide"],
                },
                index=[100, 101, 200],
            )
        ),
        programs=SimpleNamespace(
            program_df=pd.DataFrame(
                {
                    "school_id": [100, 101, 200],
                    "program_type": ["GE", "CB", "GE"],
                }
            )
        ),
    )

    restricted = _zone_restricted_schools(market, [100, 101, 200])

    assert restricted == frozenset({100})
