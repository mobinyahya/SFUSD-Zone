"""Build and validate the assignment Slurm job graph."""

from __future__ import annotations

from collections.abc import Iterable
import re


MAX_CPUS_PER_NODE = 40
MAX_SLURM_JOBS = 12
SUPPORTED_DEPENDENCIES = {"afterok"}
_JOB_ID = re.compile(r"^[a-z][a-z0-9-]*$")


def _task_batches(task_count: int, batch_count: int) -> list[list[int]]:
    if task_count < 0 or batch_count < 1:
        raise ValueError("Task and batch counts must be positive.")
    batch_count = min(task_count, batch_count)
    quotient, remainder = divmod(task_count, batch_count)
    batches = []
    start = 0
    for index in range(batch_count):
        size = quotient + (index < remainder)
        batches.append(list(range(start, start + size)))
        start += size
    return batches


def build_job_graph(
    assignment_count: int,
    metrics_count: int,
    *,
    metrics_work_counts: list[int] | None = None,
) -> list[dict]:
    """Allocate work across at most twelve Slurm jobs."""
    if assignment_count < 1:
        raise ValueError("A Slurm plan requires at least one assignment task.")
    if metrics_count < 0:
        raise ValueError("Metrics task count cannot be negative.")

    work_counts = metrics_work_counts or [1] * metrics_count
    if len(work_counts) != metrics_count or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in work_counts
    ):
        raise ValueError("Metrics work counts must cover every metrics task.")

    assignment_job_limit = MAX_SLURM_JOBS - int(bool(metrics_count))
    assignment_batches = _task_batches(
        assignment_count, min(assignment_count, assignment_job_limit)
    )
    jobs = [
        {
            "id": f"assignment-{index}",
            "kind": "assignment",
            "task_indices": task_indices,
            "cpus": min(MAX_CPUS_PER_NODE, len(task_indices)),
            "dependencies": {},
        }
        for index, task_indices in enumerate(assignment_batches)
    ]

    if metrics_count:
        jobs.append(
            {
                "id": "metrics-finalize",
                "kind": "metrics-finalize",
                "task_indices": list(range(metrics_count)),
                "cpus": min(MAX_CPUS_PER_NODE, sum(work_counts)),
                "dependencies": {"afterok": [job["id"] for job in jobs]},
            }
        )
    validate_job_graph(
        jobs,
        assignment_count=assignment_count,
        metrics_count=metrics_count,
    )
    return jobs


def dependency_ids(job: dict) -> list[str]:
    return [
        dependency_id
        for dependency_ids in job["dependencies"].values()
        for dependency_id in dependency_ids
    ]


def topological_jobs(jobs: Iterable[dict]) -> list[dict]:
    """Return jobs in dependency order and reject unknown edges or cycles."""
    ordered_input = list(jobs)
    jobs_by_id = {job["id"]: job for job in ordered_input}
    if len(jobs_by_id) != len(ordered_input):
        raise ValueError("Slurm job IDs must be unique.")

    remaining = {job["id"]: set(dependency_ids(job)) for job in ordered_input}
    unknown = set().union(*remaining.values(), set()) - jobs_by_id.keys()
    if unknown:
        raise ValueError(f"Slurm jobs reference unknown dependencies: {sorted(unknown)}.")

    result = []
    completed = set()
    while remaining:
        ready = [
            job
            for job in ordered_input
            if job["id"] in remaining and remaining[job["id"]] <= completed
        ]
        if not ready:
            raise ValueError("Slurm job graph contains a dependency cycle.")
        for job in ready:
            result.append(job)
            completed.add(job["id"])
            del remaining[job["id"]]
    return result


def validate_job_graph(
    jobs: list[dict], *, assignment_count: int, metrics_count: int
) -> None:
    if not jobs or len(jobs) > MAX_SLURM_JOBS:
        raise ValueError(
            f"Slurm plans require between 1 and {MAX_SLURM_JOBS} jobs."
        )

    assignment_indices = []
    metrics_indices = []
    assignment_ids = []
    finalizers = []
    for job in jobs:
        job_id = job.get("id")
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise ValueError(f"Invalid Slurm job ID: {job_id!r}.")
        cpus = job.get("cpus")
        if (
            isinstance(cpus, bool)
            or not isinstance(cpus, int)
            or cpus < 1
            or cpus > MAX_CPUS_PER_NODE
        ):
            raise ValueError(f"Invalid CPU count for Slurm job {job_id!r}: {cpus!r}.")
        task_indices = job.get("task_indices")
        if (
            not isinstance(task_indices, list)
            or not task_indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in task_indices
            )
            or len(task_indices) != len(set(task_indices))
        ):
            raise ValueError(f"Invalid task indices for Slurm job {job_id!r}.")
        dependencies = job.get("dependencies")
        if not isinstance(dependencies, dict):
            raise ValueError(f"Invalid dependencies for Slurm job {job_id!r}.")
        unknown_conditions = set(dependencies) - SUPPORTED_DEPENDENCIES
        if unknown_conditions:
            raise ValueError(
                f"Unsupported Slurm dependency conditions: {sorted(unknown_conditions)}."
            )
        for condition, dependency_job_ids in dependencies.items():
            if (
                not isinstance(dependency_job_ids, list)
                or any(not isinstance(value, str) for value in dependency_job_ids)
                or len(dependency_job_ids) != len(set(dependency_job_ids))
            ):
                raise ValueError(
                    f"Invalid {condition} dependencies for Slurm job {job_id!r}."
                )
            if job_id in dependency_job_ids:
                raise ValueError(f"Slurm job {job_id!r} cannot depend on itself.")

        kind = job.get("kind")
        if kind == "assignment":
            if dependencies:
                raise ValueError("Assignment jobs cannot have dependencies.")
            assignment_ids.append(job_id)
            assignment_indices.extend(task_indices)
        elif kind == "metrics-finalize":
            finalizers.append(job)
            metrics_indices.extend(task_indices)
        else:
            raise ValueError(f"Unknown Slurm job kind: {kind!r}.")

    topological_jobs(jobs)
    if sorted(assignment_indices) != list(range(assignment_count)):
        raise ValueError("Slurm jobs must cover every assignment task exactly once.")
    if sorted(metrics_indices) != list(range(metrics_count)):
        raise ValueError("Slurm jobs must cover every metrics task exactly once.")
    if len(finalizers) != int(bool(metrics_count)):
        raise ValueError("Metrics plans require exactly one finalizer job.")
    if finalizers and finalizers[0]["dependencies"] != {"afterok": assignment_ids}:
        raise ValueError("The metrics finalizer must depend on every assignment job.")
