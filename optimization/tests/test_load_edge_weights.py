"""load_edge_weights: a weighted graph whose problems still cost cut edges."""

from __future__ import annotations

import pytest

from optimization.config import OptimizationConfig


def test_it_is_off_by_default_and_loads_weights_without_costing_them() -> None:
    plain = OptimizationConfig()
    assert not plain.load_edge_weights and not plain.graph_weight_edges

    loaded = OptimizationConfig(load_edge_weights=True)
    assert loaded.graph_weight_edges
    assert not loaded.uses_weighted_edges


def test_it_keys_the_same_graph_as_a_weighted_run() -> None:
    weighted = OptimizationConfig(weight_edges=True)
    loaded = OptimizationConfig(load_edge_weights=True)
    assert weighted.graph_weight_edges == loaded.graph_weight_edges
    assert weighted.uses_weighted_edges and not loaded.uses_weighted_edges


def test_it_must_be_a_boolean() -> None:
    with pytest.raises(ValueError, match="load_edge_weights"):
        OptimizationConfig(load_edge_weights="yes")
