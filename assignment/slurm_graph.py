"""Build and validate the assignment Slurm job graph."""

from __future__ import annotations

from collections.abc import Iterable
import re


MAX_CPUS_PER_NODE = 40
MAX_RUNNING_SLURM_JOBS = 12
MAX_ASSIGNMENT_JOBS = 12
MAX_METRICS_JOBS = 8
MAX_SLURM_JOBS = MAX_ASSIGNMENT_JOBS + MAX_METRICS_JOBS
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
    metric_assignment_dependencies: list[list[int]] | None = None,
) -> list[dict]:
    """Allocate assignment and dependent metrics work across Slurm jobs."""
    if assignment_count < 1:
        raise ValueError("A Slurm plan requires at least one assignment task.")
    if metrics_count < 0:
        raise ValueError("Metrics task count cannot be negative.")

    work_counts = (
        [1] * metrics_count
        if metrics_work_counts is None
        else metrics_work_counts
    )
    if len(work_counts) != metrics_count or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in work_counts
    ):
        raise ValueError("Metrics work counts must cover every metrics task.")
    _validate_metric_dependencies(
        metric_assignment_dependencies,
        assignment_count=assignment_count,
        metrics_count=metrics_count,
    )

    assignment_batches = _task_batches(
        assignment_count, min(assignment_count, MAX_ASSIGNMENT_JOBS)
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

    assignment_jobs = list(jobs)
    if metrics_count:
        metric_job_count = min(metrics_count, MAX_METRICS_JOBS)
        metric_batches = _task_batches(metrics_count, metric_job_count)
        for index, task_indices in enumerate(metric_batches):
            required_assignment_tasks = {
                assignment_index
                for task_index in task_indices
                for assignment_index in metric_assignment_dependencies[task_index]
            }
            assignment_dependencies = [
                job["id"]
                for job in assignment_jobs
                if required_assignment_tasks.intersection(job["task_indices"])
            ]
            is_finalizer = index == metric_job_count - 1
            dependencies = assignment_dependencies
            if is_finalizer:
                dependencies = [
                    *dependencies,
                    *(job["id"] for job in jobs if job["kind"] == "metrics"),
                ]
            jobs.append(
                {
                    "id": "metrics-finalize" if is_finalizer else f"metrics-{index}",
                    "kind": "metrics-finalize" if is_finalizer else "metrics",
                    "task_indices": task_indices,
                    "cpus": min(
                        MAX_CPUS_PER_NODE,
                        sum(work_counts[task_index] for task_index in task_indices),
                    ),
                    "dependencies": {"afterok": dependencies},
                }
            )
    validate_job_graph(
        jobs,
        assignment_count=assignment_count,
        metrics_count=metrics_count,
        metric_assignment_dependencies=metric_assignment_dependencies,
    )
    return jobs


def _validate_metric_dependencies(
    dependencies: list[list[int]] | None,
    *,
    assignment_count: int,
    metrics_count: int,
) -> None:
    if metrics_count == 0:
        if dependencies not in (None, []):
            raise ValueError("A plan without metrics cannot have metric dependencies.")
        return
    if not isinstance(dependencies, list) or len(dependencies) != metrics_count:
        raise ValueError("Metric dependencies must cover every metrics task.")
    for task_dependencies in dependencies:
        if (
            not isinstance(task_dependencies, list)
            or not task_dependencies
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= assignment_count
                for index in task_dependencies
            )
            or len(task_dependencies) != len(set(task_dependencies))
        ):
            raise ValueError("Metric tasks have invalid assignment dependencies.")


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
    jobs: list[dict],
    *,
    assignment_count: int,
    metrics_count: int,
    metric_assignment_dependencies: list[list[int]] | None = None,
) -> None:
    if not jobs or len(jobs) > MAX_SLURM_JOBS:
        raise ValueError(
            f"Slurm plans require between 1 and {MAX_SLURM_JOBS} jobs."
        )

    assignment_indices = []
    metrics_indices = []
    assignment_jobs = []
    metrics_jobs = []
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
            assignment_jobs.append(job)
            assignment_indices.extend(task_indices)
        elif kind == "metrics":
            metrics_jobs.append(job)
            metrics_indices.extend(task_indices)
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
    if len(assignment_jobs) > MAX_ASSIGNMENT_JOBS:
        raise ValueError("Slurm plan exceeds the assignment job limit.")
    if len(metrics_jobs) + len(finalizers) > MAX_METRICS_JOBS:
        raise ValueError("Slurm plan exceeds the metrics job limit.")
    if len(finalizers) != int(bool(metrics_count)):
        raise ValueError("Metrics plans require exactly one finalizer job.")
    _validate_metric_dependencies(
        metric_assignment_dependencies,
        assignment_count=assignment_count,
        metrics_count=metrics_count,
    )
    for job in [*metrics_jobs, *finalizers]:
        required_assignment_tasks = {
            assignment_index
            for task_index in job["task_indices"]
            for assignment_index in metric_assignment_dependencies[task_index]
        }
        expected_dependencies = [
            assignment_job["id"]
            for assignment_job in assignment_jobs
            if required_assignment_tasks.intersection(
                assignment_job["task_indices"]
            )
        ]
        if job["kind"] == "metrics-finalize":
            expected_dependencies.extend(metric_job["id"] for metric_job in metrics_jobs)
        if job["dependencies"] != {"afterok": expected_dependencies}:
            raise ValueError(
                f"Slurm job {job['id']!r} does not have its exact required dependencies."
            )
