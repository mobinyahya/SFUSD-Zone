"""Per-participant session activity logger.

Appends one JSON object per line to logs/{participant_id}.jsonl. Used by the
website backend to record every user action (solution loaded, cluster selected,
chat message) along with the current filter/constraint state at that step.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent / "logs"
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_\-]")


def _safe_participant_id(participant_id: str) -> str:
    return _SAFE_ID_RE.sub("_", participant_id)[:64] or "anonymous"


def serialize_filter_state(filter_state: Any) -> dict[str, dict[str, Optional[float]]]:
    """Convert a FilterState dataclass into a plain {metric_name: {min, max}} dict.

    Only includes metrics with at least one active bound, to keep the log compact.
    """
    if filter_state is None or not hasattr(filter_state, "bounds"):
        return {}
    out: dict[str, dict[str, Optional[float]]] = {}
    for name, bounds in filter_state.bounds.items():
        mn = getattr(bounds, "min_bound", None)
        mx = getattr(bounds, "max_bound", None)
        if mn is None and mx is None:
            continue
        out[name] = {"min": mn, "max": mx}
    return out


def log_event(
    participant_id: Optional[str],
    session_id: Optional[str],
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Append a JSONL entry to the participant's log file. No-op if no participant_id."""
    if not participant_id:
        return
    try:
        LOG_DIR.mkdir(exist_ok=True)
        safe_id = _safe_participant_id(participant_id)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "participant_id": participant_id,
            "session_id": session_id,
            "event": event_type,
            **payload,
        }
        line = json.dumps(entry, default=str)
        with (LOG_DIR / f"{safe_id}.jsonl").open("a") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.error(f"Failed to write activity log for {participant_id}: {e}")
