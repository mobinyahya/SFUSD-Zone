"""Student-assignment matching integration for benchmark outputs."""

from Zone_Generation.benchmark.matching.runner import (
    MatchingBatchResult,
    MatchingResult,
    merge_matching_result,
    preserve_matching_payload,
    run_matching_for_existing_runs,
    run_matching_for_solution,
    write_matching_zone_csv,
)

__all__ = [
    "MatchingBatchResult",
    "MatchingResult",
    "merge_matching_result",
    "preserve_matching_payload",
    "run_matching_for_existing_runs",
    "run_matching_for_solution",
    "write_matching_zone_csv",
]
