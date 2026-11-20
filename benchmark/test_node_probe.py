"""A job leaves a node too slow to load the code, resubmitting itself without it."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from benchmark.slurm import (
    EXCLUDE_ENV,
    _allocation_script_path,
    _sbatch_options,
    write_plan,
    write_submission_script,
)
from benchmark.test_slurm import _plan


def _fake_tools(tmp_path: Path, *, python_seconds: float, host: str = "soal-4"):
    """Fake python/sbatch/uv/hostname on PATH; return (bin dir, sbatch log, uv log)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch_log = tmp_path / "sbatch.log"
    uv_log = tmp_path / "uv.log"
    scripts = {
        "sbatch": f'#!/bin/sh\necho "$@" >> "{sbatch_log}"\necho 4242\n',
        "uv": f'#!/bin/sh\necho "$@" >> "{uv_log}"\n',
        "hostname": f"#!/bin/sh\necho {host}\n",
    }
    for name, text in scripts.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    venv = tmp_path / "project" / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text(f"#!/bin/sh\nsleep {python_seconds}\n")
    (venv / "python").chmod(0o755)
    return bin_dir, sbatch_log, uv_log


def _job_command(tmp_path):
    plan = _plan(tmp_path)
    plan_path = write_plan(plan)
    write_submission_script(plan, plan_path)
    options = _sbatch_options(plan, plan_path, 0)
    return plan, options[options.index("--wrap") + 1]


@pytest.mark.parametrize("previous", ["", "soal-10"])
def test_a_slow_node_resubmits_the_allocation_without_itself(tmp_path, previous):
    plan, command = _job_command(tmp_path)
    bin_dir, sbatch_log, uv_log = _fake_tools(tmp_path, python_seconds=5)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SFUSD_NODE_PROBE_SECONDS": "1",
        EXCLUDE_ENV: previous,
    }

    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path / "project",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not uv_log.exists()  # the worker never ran here
    expected = "soal-10,soal-4" if previous else "soal-4"
    submitted = sbatch_log.read_text()
    assert f"--exclude={expected}" in submitted
    assert "--job-name=benchmark-0" in submitted
    assert "node probe: soal-4" in result.stderr
    assert _allocation_script_path(plan, 0).is_file()


def test_a_healthy_node_runs_the_worker(tmp_path):
    _plan_, command = _job_command(tmp_path)
    bin_dir, sbatch_log, uv_log = _fake_tools(tmp_path, python_seconds=0)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path / "project",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "worker-allocation" in uv_log.read_text()
    assert not sbatch_log.exists()


def test_hopping_stops_after_too_many_slow_nodes(tmp_path):
    _plan_, command = _job_command(tmp_path)
    bin_dir, sbatch_log, uv_log = _fake_tools(tmp_path, python_seconds=5)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SFUSD_NODE_PROBE_SECONDS": "1",
        EXCLUDE_ENV: ",".join(f"soal-{i}" for i in (1, 2, 3, 5, 6, 7)),
    }

    result = subprocess.run(
        ["bash", "-c", command],
        cwd=tmp_path / "project",
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "too many slow nodes" in result.stderr
    assert not sbatch_log.exists()
    assert not uv_log.exists()
