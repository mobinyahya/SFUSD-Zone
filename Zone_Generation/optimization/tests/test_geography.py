import math

import pytest

from Zone_Generation.optimization.data.geography import (
    EARTH_RADIUS_MILES,
    great_circle_miles,
)


def test_great_circle_miles_is_zero_for_identical_points():
    assert great_circle_miles(37.7749, -122.4194, 37.7749, -122.4194) == 0.0


def test_great_circle_miles_for_quarter_circumference():
    distance = great_circle_miles(0.0, 0.0, 0.0, 90.0)

    assert distance == pytest.approx(EARTH_RADIUS_MILES * math.pi / 2)
