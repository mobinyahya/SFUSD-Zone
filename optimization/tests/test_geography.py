import math

import pytest

from optimization.data.geography import (
    EARTH_RADIUS_MILES,
    great_circle_miles,
)


def test_great_circle_miles_is_zero_for_identical_points():
    assert great_circle_miles(37.7749, -122.4194, 37.7749, -122.4194) == 0.0


def test_great_circle_miles_for_quarter_circumference():
    distance = great_circle_miles(0.0, 0.0, 0.0, 90.0)

    assert distance == pytest.approx(EARTH_RADIUS_MILES * math.pi / 2)


def test_great_circle_matrix_is_bit_identical_to_scalar():
    import random

    from optimization.data.geography import great_circle_miles_matrix

    rng = random.Random(0)
    lat = [rng.uniform(37.70, 37.82) for _ in range(60)] + [0.0, 0.0]
    lon = [rng.uniform(-122.52, -122.35) for _ in range(60)] + [0.0, 90.0]
    # Small blocks exercise the row chunking.
    matrix = great_circle_miles_matrix(lat, lon, max_cells=100)
    for i in range(len(lat)):
        for j in range(len(lat)):
            assert matrix[i, j] == great_circle_miles(lat[i], lon[i], lat[j], lon[j])


def test_great_circle_distance_dict_keeps_node_order_and_zero_diagonal():
    from optimization.data.geography import great_circle_distance_dict

    nodes = [5, 2, 9]
    lat, lon = [37.75, 37.76, 37.70], [-122.45, -122.40, -122.50]
    distances = great_circle_distance_dict(nodes, lat, lon)
    assert list(distances) == nodes
    for a, i in enumerate(nodes):
        assert list(distances[i]) == nodes
        for b, j in enumerate(nodes):
            expected = 0.0 if i == j else great_circle_miles(lat[a], lon[a], lat[b], lon[b])
            assert distances[i][j] == expected
            assert type(distances[i][j]) is float
