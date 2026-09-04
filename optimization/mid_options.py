"""Shared parsing for MID-specific optimization options."""

from __future__ import annotations

import math


def normalize_complementary_slackness_slack(value: object) -> float | str:
    """Return a canonical non-negative seat, percentage, or ``auto`` slack."""

    message = (
        "mid_complementary_slackness_slack must be 'auto', a non-negative "
        "number, or a percentage string."
    )
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() == "auto":
            return "auto"
        if normalized.endswith("%"):
            try:
                percentage = float(normalized[:-1])
            except ValueError as exc:
                raise ValueError(message) from exc
            if not math.isfinite(percentage) or percentage < 0:
                raise ValueError(message)
            return f"{percentage:g}%"
        try:
            number = float(normalized)
        except ValueError as exc:
            raise ValueError(message) from exc
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise ValueError(message)

    if not math.isfinite(number) or number < 0:
        raise ValueError(message)
    return number
