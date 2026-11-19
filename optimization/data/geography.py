"""Low-level geographic calculations shared by data ingestion and graphs."""

from __future__ import annotations

import math

import numpy as np


EARTH_RADIUS_MILES = 6371.01 * 0.621371

# ``math.radians`` multiplies by exactly this constant.
_DEGREES_TO_RADIANS = math.pi / 180.0


def great_circle_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in miles between two WGS84 points."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon_delta_rad = math.radians(lon1 - lon2)
    cosine = math.sin(lat1_rad) * math.sin(lat2_rad) + math.cos(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(lon_delta_rad)
    return EARTH_RADIUS_MILES * math.acos(max(-1.0, min(1.0, cosine)))


def _libm(function, values: np.ndarray) -> np.ndarray:
    """Apply a ``math`` function elementwise, keeping libm's exact results.

    NumPy's own SIMD transcendentals can differ from libm in the last ulp, and
    ``acos`` near 1 turns one ulp into a relative error of ~1e-7 at block
    spacing. Mapping the ``math`` builtin keeps every value bit-identical to
    :func:`great_circle_miles` on every machine.
    """
    flat = values.ravel()
    return np.fromiter(map(function, flat.tolist()), float, flat.size).reshape(
        values.shape
    )


def great_circle_miles_matrix(
    lat1, lon1, lat2=None, lon2=None, *, max_cells: int = 1 << 20
) -> np.ndarray:
    """Pairwise :func:`great_circle_miles` from points 1 (rows) to points 2.

    Bit-identical to calling the scalar function for every pair: the same
    operations in the same order, IEEE arithmetic vectorised, and libm's
    ``sin``/``cos``/``acos``. Points 2 default to points 1. Rows are computed
    in blocks of about ``max_cells`` entries to bound temporary memory.
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = lat1 if lat2 is None else np.asarray(lat2, dtype=float)
    lon2 = lon1 if lon2 is None else np.asarray(lon2, dtype=float)
    lat1_rad = lat1 * _DEGREES_TO_RADIANS
    lat2_rad = lat2 * _DEGREES_TO_RADIANS
    sin1, cos1 = _libm(math.sin, lat1_rad), _libm(math.cos, lat1_rad)
    sin2, cos2 = _libm(math.sin, lat2_rad), _libm(math.cos, lat2_rad)
    out = np.empty((lat1.size, lat2.size), dtype=float)
    step = max(1, max_cells // max(1, lat2.size))
    for start in range(0, lat1.size, step):
        rows = slice(start, start + step)
        cos_delta = _libm(
            math.cos, (lon1[rows, None] - lon2[None, :]) * _DEGREES_TO_RADIANS
        )
        cosine = (
            sin1[rows, None] * sin2[None, :]
            + cos1[rows, None] * cos2[None, :] * cos_delta
        )
        np.minimum(cosine, 1.0, out=cosine)
        np.maximum(cosine, -1.0, out=cosine)
        out[rows] = EARTH_RADIUS_MILES * _libm(math.acos, cosine)
    return out


def great_circle_distance_dict(nodes, lat, lon) -> dict:
    """``{i: {j: miles}}`` over ``nodes`` (zero on the diagonal), in node order."""
    nodes = list(nodes)
    matrix = great_circle_miles_matrix(lat, lon)
    distance_dict = {}
    for index, node in enumerate(nodes):
        row = dict(zip(nodes, matrix[index].tolist()))
        row[node] = 0.0
        distance_dict[node] = row
    return distance_dict
