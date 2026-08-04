"""Integer Lagrangian certificate assembly for pattern pricing bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PricingMultipliers:
    """Quantized raw-unit multipliers; node signs are unrestricted."""

    node: dict[int, int]
    boundary: int

    def __post_init__(self) -> None:
        node = {
            _integer("node", graph_node): _integer("node multiplier", value)
            for graph_node, value in self.node.items()
        }
        boundary = _integer("boundary multiplier", self.boundary)
        if boundary < 0:
            raise ValueError("Boundary multiplier must be nonnegative.")
        object.__setattr__(self, "node", node)
        object.__setattr__(self, "boundary", boundary)


@dataclass(frozen=True, slots=True)
class LagrangianCertificate:
    """Auditable integer weak-duality bound for one branch-price node."""

    labels: tuple[int, ...]
    coverage_nodes: frozenset[int]
    zone_perimeter_cap: int
    multipliers: PricingMultipliers
    pricing_upper_bounds: dict[int, int]
    upper_bound: int


def quantize_multipliers(
    node_multipliers: Mapping[int, Real], boundary_multiplier: Real
) -> PricingMultipliers:
    """Round finite guidance duals to the integer raw-objective grid."""
    nodes = {
        _integer("node", node): _nearest_integer(value, "node multiplier")
        for node, value in node_multipliers.items()
    }
    boundary = max(0, _nearest_integer(boundary_multiplier, "boundary multiplier"))
    return PricingMultipliers(node=nodes, boundary=boundary)


def assemble_lagrangian_certificate(
    *,
    labels: Sequence[int],
    coverage_nodes: Sequence[int] | set[int] | frozenset[int],
    zone_perimeter_cap: int,
    multipliers: PricingMultipliers,
    pricing_upper_bounds: Mapping[int, int],
) -> LagrangianCertificate:
    """Assemble ``sum(pi) + cap*mu + sum(U_z)`` entirely in Python integers."""
    label_tuple = tuple(_integer("label", label) for label in labels)
    if len(set(label_tuple)) != len(label_tuple):
        raise ValueError("Certificate labels must be distinct.")
    coverage = frozenset(_integer("coverage node", node) for node in coverage_nodes)
    unknown_nodes = set(multipliers.node) - coverage
    if unknown_nodes:
        raise ValueError(
            f"Multipliers include nodes without master coverage rows: {sorted(unknown_nodes)}."
        )
    cap = _integer("zone_perimeter_cap", zone_perimeter_cap)
    if cap < 0:
        raise ValueError("zone_perimeter_cap must be nonnegative.")
    bounds = {
        _integer("pricing label", label): _integer("pricing upper bound", value)
        for label, value in pricing_upper_bounds.items()
    }
    if set(bounds) != set(label_tuple):
        raise ValueError("A safe pricing upper bound is required for every label.")
    upper_bound = (
        sum(multipliers.node.get(node, 0) for node in coverage)
        + cap * multipliers.boundary
        + sum(bounds[label] for label in label_tuple)
    )
    return LagrangianCertificate(
        labels=label_tuple,
        coverage_nodes=coverage,
        zone_perimeter_cap=cap,
        multipliers=multipliers,
        pricing_upper_bounds=bounds,
        upper_bound=upper_bound,
    )


def _nearest_integer(value: Real, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    return int(round(value))


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)
