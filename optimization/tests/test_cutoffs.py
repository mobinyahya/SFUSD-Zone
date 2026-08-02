"""Tests for constructing the cutoff assignment market."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from optimization.data.cutoffs import (
    _exclude_citywide_school_columns,
    _zone_restricted_schools,
)


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


def test_remove_citywide_excludes_columns_and_restricts_every_remaining_school():
    market = SimpleNamespace(
        schools=SimpleNamespace(
            school_df=pd.DataFrame(
                {"category": ["Attendance", "Attendance", "Citywide"]},
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
    priorities = np.array([[1, 2, 3], [4, 5, 6]])
    utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    school_ids, priorities, utilities, excluded = _exclude_citywide_school_columns(
        frozenset({200}),
        [100, 101, 200],
        priorities,
        utilities,
    )
    restricted = _zone_restricted_schools(market, school_ids, restrict_all=True)

    assert school_ids == [100, 101]
    assert excluded == [200]
    np.testing.assert_array_equal(priorities, [[1, 2], [4, 5]])
    np.testing.assert_array_equal(utilities, [[0.1, 0.2], [0.4, 0.5]])
    assert restricted == frozenset({100, 101})
