"""State management for versioned zoning proposals."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .filters import FilterState

if TYPE_CHECKING:
    from .zoning_agent import ZoningAgent


def _direction_text(metric) -> str:
    """Return a human-readable direction hint for a metric."""
    if metric.direction == "minimize":
        return "(lower better)"
    elif metric.direction == "maximize":
        return "(higher better)"
    return "(informational)"


# ============================================================================
# TOOL RESULT: STRUCTURED RETURN FROM TOOL EXECUTION
# ============================================================================

@dataclass
class ToolResult:
    """Structured result from a tool execution.

    text: The string sent to the LLM as tool output
    solution_path: If the tool produced/changed a solution, the path to it
    """
    text: str
    solution_path: Optional[str] = None


# ============================================================================
# STATE MANAGEMENT FOR VERSIONED ZONING PROPOSALS
# ============================================================================

@dataclass
class ProposalVersion:
    """A versioned snapshot of a zoning proposal state."""
    version_id: int
    timestamp: str
    filter_state: FilterState
    solution_path: Optional[str] = None
    solution_count: int = 0
    description: str = ""


@dataclass
class AgentState:
    """Complete state for the zoning agent session."""
    # Version history
    versions: list[ProposalVersion] = field(default_factory=list)
    current_version: int = 0

    # Clustering state
    cluster_labels: Optional[list] = None
    cluster_centers: Optional[list] = None
    cluster_directions: Optional[dict] = None
    clustered_solutions: Optional[object] = None
    clustered_vectors: Optional[object] = None
    cluster_columns: Optional[list[str]] = None

    # Interaction state
    awaiting_confirmation: bool = False
    pending_action: Optional[dict] = None
    last_action: str = ""

    def save_version(self, filter_state: FilterState, solution_path: str = None,
                     solution_count: int = 0, description: str = "") -> int:
        """Save a new version and return version ID."""
        version_id = len(self.versions)
        version = ProposalVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            filter_state=copy.deepcopy(filter_state),
            solution_path=solution_path,
            solution_count=solution_count,
            description=description
        )
        self.versions.append(version)
        self.current_version = version_id
        return version_id

    def undo(self, steps: int = 1) -> Optional[ProposalVersion]:
        """Undo to a previous version. Returns the version or None if not possible."""
        target_version = self.current_version - steps
        if target_version < 0 or target_version >= len(self.versions):
            return None
        self.current_version = target_version
        return self.versions[target_version]

    def get_current_version(self) -> Optional[ProposalVersion]:
        """Get the current version."""
        if 0 <= self.current_version < len(self.versions):
            return self.versions[self.current_version]
        return None


# ============================================================================
# DEFERRED VERSION SAVE HELPER
# ============================================================================

def save_or_defer_version(agent: ZoningAgent, description: str,
                          solution_path: Optional[str],
                          solution_count: int) -> None:
    """Save a version immediately, or defer if batching is active.

    When ``agent._defer_version_save`` is True (inside a single chat turn that
    may invoke multiple filter tools), the description is accumulated and a
    single version is flushed at the end of the turn.  Otherwise the version is
    saved immediately.
    """
    if agent._defer_version_save:
        agent._pending_descriptions.append(description)
        if solution_path:
            agent._pending_solution_path = solution_path
    else:
        agent.state.save_version(
            agent.filter_state,
            solution_path=solution_path,
            solution_count=solution_count,
            description=description,
        )
