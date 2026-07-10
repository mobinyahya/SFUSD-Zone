"""Low-level geographic calculations shared by data ingestion and graphs."""

from __future__ import annotations

import math


EARTH_RADIUS_MILES = 6371.01 * 0.621371


def great_circle_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in miles between two WGS84 points."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon_delta_rad = math.radians(lon1 - lon2)
    cosine = math.sin(lat1_rad) * math.sin(lat2_rad) + math.cos(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(lon_delta_rad)
    return EARTH_RADIUS_MILES * math.acos(max(-1.0, min(1.0, cosine)))
