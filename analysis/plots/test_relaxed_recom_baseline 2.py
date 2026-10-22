"""Tests for the relaxed-ReCom baseline's resource detection."""

from __future__ import annotations

import os

from analysis.plots.relaxed_recom_baseline import available_cpus


def test_available_cpus_respects_a_slurm_allocation(monkeypatch):
    """A one-CPU Slurm job must not start one worker per core on the node."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert available_cpus() == 1

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert available_cpus() == 4


def test_available_cpus_never_exceeds_the_real_core_count(monkeypatch):
    """A nonsense allocation is still capped by what the machine has."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "100000")
    assert available_cpus() <= (os.cpu_count() or 1)


def test_available_cpus_ignores_unparseable_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "")
    assert available_cpus() >= 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    assert available_cpus() >= 1


def test_available_cpus_is_at_least_one(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert available_cpus() >= 1
