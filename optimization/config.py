"""Optimization configuration and factories.

``OptimizationConfig`` is the single typed description of a run: which levels, which
solver, which strategy, and all the data/optimization parameters. Its factory
methods build the concrete :class:`Dataset`, :class:`Solver` and
:class:`Strategy`, wiring the three layers together from a string config.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from loaders import DataScenario, anchor_data_config, load_scenario
from optimization.dw_options import DW_MASTER_METHODS, DW_OBJECTIVES
from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec
from optimization.mid_options import (
    MID_PREFERENCE_SOURCES,
    normalize_complementary_slackness_slack,
)
from optimization.strategies.budget import BUDGET_ACCOUNTING_MODES
from choice.models import PRICE_SOURCES
from optimization.welfare_bounds import WELFARE_BOUNDS


# Strategies that read max_iterations, and those that read tolerance.
_MAX_ITERATION_STRATEGIES = {"iterative_choice", "priced_access", "mid_decomp", "saa"}
_TOLERANCE_STRATEGIES = _MAX_ITERATION_STRATEGIES | {"dantzig_wolfe"}

# Field validation tables. ``strategy``, ``solver`` and
# ``adaptive_short_bursts_objective`` are checked against their registries.
_CHOICES = {
    "dw_objective": DW_OBJECTIVES,
    "dw_master_method": tuple(sorted(DW_MASTER_METHODS)),
    "mid_utility_handling": ("exponentiate", "omit_nonpositive"),
    "mid_preference_source": MID_PREFERENCE_SOURCES,
    "saa_welfare_bound": WELFARE_BOUNDS,
    "priced_access_price_source": PRICE_SOURCES,
    "budget_accounting": BUDGET_ACCOUNTING_MODES,
    "hints": ("feasible", "voronoi", "none"),
    "short_bursts_method": ("recom", "relaxed_recom"),
    "adaptive_short_bursts_pair_selector": ("uniform", "lagrangian_softmax"),
}
_BOOLEANS = (
    "weight_edges",
    "mid_transport_bounds",
    "mid_complementary_slackness",
    "saa_relative_gap",
    "saa_multicut",
    "stable_cutoff_non_wastefulness",
    "stable_cutoff_aggregate_stability",
    "choice_access_triangle",
    "choice_access_cardinality",
    "choice_access_complete",
    "dw_pricing_parallel",
    "dw_redraw",
)
# Integer fields and their minimum (``None``: any integer).
_INTEGERS = {
    "enumerated_solutions": None,
    "recom_iterations": None,
    "centroid_neighbor_radius": 0,
    "choice_access_triangle_limit": 0,
    "choice_access_completion_limit": 0,
    "dw_recom_samples": 0,
    "short_bursts_length": 1,
    "mid_lottery_scale": 1,
    "saa_num_seeds": 1,
    "zoned_transport_workers": 1,
    "stable_cutoff_num_seeds": 1,
    "priced_access_cut_levels": 1,
    "dw_recom_chains": 1,
    "dw_pricing_scale": 1,
    "dw_pricing_columns_per_call": 1,
    "dw_redraw_splits_per_call": 1,
}
_INTEGER_KINDS = {
    None: "an integer",
    0: "a non-negative integer",
    1: "a positive integer",
}
# Numeric fields, coerced to float: the error message and the bounds.
_NUMBERS = {
    "boundary_prop": ("must be at most 1; negative disables it", {"maximum": 1.0}),
    "feasible_hint_time_limit": (
        "must be positive",
        {"minimum": 0.0, "open_minimum": True, "finite": True},
    ),
    "priced_access_price_scale": ("must be non-negative", {"minimum": 0.0}),
    "adaptive_short_bursts_lr": (
        "must be positive",
        {"minimum": 0.0, "open_minimum": True},
    ),
    "adaptive_short_bursts_temperature": (
        "must be positive",
        {"minimum": 0.0, "open_minimum": True},
    ),
    "adaptive_short_bursts_unassigned_weight": (
        "must be finite and non-negative",
        {"minimum": 0.0, "finite": True},
    ),
    "adaptive_short_bursts_designated_weight": (
        "must be finite and non-negative",
        {"minimum": 0.0, "finite": True},
    ),
    "dw_recom_time_limit": (
        "must be finite and nonnegative",
        {"minimum": 0.0, "finite": True},
    ),
    "dw_dual_smoothing": (
        "must lie in (0, 1]",
        {"minimum": 0.0, "maximum": 1.0, "open_minimum": True},
    ),
    "dw_pricing_time_limit": (
        "must be positive or infinity",
        {"minimum": 0.0, "open_minimum": True},
    ),
    "dw_overlap_prop": (
        "must lie in [0, 1]",
        {"minimum": 0.0, "maximum": 1.0},
    ),
    "dw_redraw_time_limit": ("must be nonnegative or infinity", {"minimum": 0.0}),
}


def _is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _legacy_data_config() -> dict[str, Any]:
    return {"scenario": "legacy", "overrides": {}}


@dataclass
class OptimizationConfig:
    # --- what to solve ------------------------------------------------- #
    centroids_type: str = "5-zone-AF"
    levels: list[str] = field(default_factory=lambda: ["BlockGroup_0"])
    solver: str = "cp_int"
    strategy: str = "single"

    # --- optimization parameters -------------------------------------- #
    frl_dev: float = 0.3
    racial_dev: float = 0.3
    overage: float = 0.8
    shortage: float = 0.2
    # Zones hold the average school count +/- this many schools; 0 = exactly one
    # school per zone when every school is a centroid.
    school_count_tolerance: float = 1.0
    looseness: float = 1.0
    max_distance: float | str = float("inf")
    centroid_neighbor_radius: int = 0
    solve_time_limits: list[float] = field(default_factory=lambda: [60.0])
    carry_over_compute: bool = False
    budget_accounting: str = "wall_clock"
    gap_limits: list[float] = field(default_factory=lambda: [0.0])
    hints: str = "voronoi"
    # Path to a saved ``{area_id: zone}`` zoning (a benchmark
    # ``zone_dict_area_<level>.json``). When set it is projected onto every
    # level by area id and becomes the problem's hint, ahead of ``hints``.
    hint_zoning: str | None = None
    feasible_hint_time_limit: float = 60.0
    save_solver_logs: bool = False
    save_solver_progress: bool = False
    secondary_objective: bool = False
    weight_edges: bool = False
    # CP/MIP objective without a choice model: "cut_edges" (boundary cost) or
    # "school_distance" (weighted squared distance from each node to its
    # zone's school; ReCom solvers use adaptive_short_bursts_objective instead).
    compactness_objective: str = "cut_edges"
    # school_distance node weight = area share * this + student share * that;
    # both in [0, 1] and summing to one.
    school_distance_area_weight: float = 0.75
    school_distance_student_weight: float = 0.25
    seed: int = 42
    workers: int = 8
    linearization_level: int | None = None
    cp_model_probing_level: int | None = None
    symmetry_level: int | None = None
    cp_sat_search_strategy: str | None = None
    enumerated_solutions: int = -1
    recom_iterations: int = -1
    short_bursts_length: int = 25
    short_bursts_method: str = "recom"
    adaptive_short_bursts_lr: float = 0.1
    adaptive_short_bursts_temperature: float = 1.0
    adaptive_short_bursts_pair_selector: str = "lagrangian_softmax"
    # What adaptive short bursts rank feasible partitions by: ``cut_edges``
    # (boundary cost), ``welfare`` (discrete MID welfare, maximized), or
    # ``capacity_match`` (weighted unassigned plus designated students,
    # minimized). The two choice objectives build a MID market and need
    # program_population='All'. Constraints, including boundary_prop's
    # cut-edge cap, are penalized in the Lagrangian under every objective.
    adaptive_short_bursts_objective: str = "cut_edges"
    adaptive_short_bursts_unassigned_weight: float = 1.0
    adaptive_short_bursts_designated_weight: float = 1.0
    # --- strategy-specific -------------------------------------------- #
    boundary_radius: int = 1
    boundary_prop: float = -1.0
    max_iterations: int = 5
    choice_model_method: str = "logsum"
    choice_utility_scale: float = 100.0
    choice_utility_hints: bool = False
    tolerance: float = 1e-6
    mid_lottery_scale: int = 20
    mid_utility_handling: str = "omit_nonpositive"
    # Where MID market preferences come from: "estimate" (choice.estimate, a
    # per-student fit for one cohort) or "stated" (each applicant's submitted
    # list, ordinal only). Use "stated" for cohorts the estimate does not cover.
    mid_preference_source: str = "estimate"
    mid_transport_bounds: bool = True
    mid_complementary_slackness: bool = False
    mid_complementary_slackness_slack: float | str = "auto"
    saa_num_seeds: int = 5
    saa_tie_breaking_method: str = "MTB"
    # Which a-priori constant bounds the master's objective variable. The
    # reported bound is min(constant, what the cuts prove), so a smaller
    # constant can only help. Ordered loosest first: "first_choice" ignores
    # capacity; "transport" respects it; the two "zoned_transport_*" kinds also
    # price the zone confinement, maximizing the zone-restricted transport
    # value over the LP relaxation of the zoning model itself, with the suffix
    # naming the contiguity description.
    #
    # "zoned_transport_neighbors" is the default because it is both the
    # tightest and the cheapest of the four: measured on Block_2 it comes in
    # ~1,200 below "transport" and ~100 below "zoned_transport_flow", for one
    # cached LP of ~2-3s. Relaxing the closer-neighbor rows the master itself
    # carries, rather than swapping in the flow description, is also the
    # closer companion to the master. See optimization/welfare_bounds.py,
    # optimization/zoned_transport.py, and Propositions 4, 13 and 14 in
    # paper/paper.tex.
    saa_welfare_bound: str = "zoned_transport_neighbors"
    # Gurobi threads for the zoned-transport LP. One, deliberately: Gurobi runs
    # concurrent LP, and on a model this size the racing algorithms mostly just
    # contend. Across 1/2/4/6/8 threads on two Block_2 instances, no count
    # above 1 was reliably faster and several were clearly slower -- flow went
    # 2.4s -> 4.4s from 1 to 8 threads, neighbors 1.8s -> 2.5s on 6-zone-3.
    # (One cell bucked it: neighbors on 6-zone-9 at 4 threads, 2.5s against
    # 2.8s single-threaded, on a single observation.) Since the spread is a
    # couple of seconds either way against a 300s master solve, 1 is the right
    # default mostly because it leaves the cores alone. Separate knob from
    # `workers` for that reason, and excluded from the bound's cache key
    # because it cannot change the value.
    zoned_transport_workers: int = 1
    # Compare the outer-loop gap against tolerance * |incumbent| rather than
    # against tolerance outright, which on a welfare of ~15,000 asks for ten
    # significant digits and can never fire.
    saa_relative_gap: bool = False
    # Multicut: give each sampled scenario its own epigraph variable eta_psi
    # and maximize their sum. Summing the per-scenario cuts of one iteration
    # reproduces the averaged cut, so the multicut master is never looser at
    # the same cut pool -- it is the formulation of record (Proposition 12 in
    # paper/paper.tex). False recovers the single-cut master, which bounds one
    # expected-welfare variable with one averaged hyperplane per iteration and
    # trades that tightness for S times fewer rows; that arm measured *better*
    # on the cluster on 2026-09-05, so a regression is worth checking here
    # first. Formerly `saa_disaggregate_cuts`, still accepted when reading
    # saved configs.
    saa_multicut: bool = True
    # --- stable_cutoff ------------------------------------------------- #
    # How many strict priority orders the exact matching model averages over.
    # Each one costs a full matching block -- 2|Gamma| binaries and O(|Gamma|)
    # rows -- so this is the single knob that decides whether the model fits.
    stable_cutoff_num_seeds: int = 3
    # Non-wastefulness (S1). Unnecessary for exactness: the integral optimum is
    # the same with and without it. Kept on because it is worth ~100x in time
    # through presolve. See optimization/solvers/stable_cutoff.py.
    stable_cutoff_non_wastefulness: bool = True
    # The aggregate stability row (S2), which implies the per-pair no-blocking
    # row and tightens the relaxation from +0.91% to +0.73% of realised DA
    # welfare at a fixed zoning, at one extra prefix variable per pair.
    stable_cutoff_aggregate_stability: bool = True
    # Single tie-breaking is the SFUSD lottery's own design, and with a handful
    # of samples it is also the lower-variance choice.
    stable_cutoff_tie_breaking_method: str = "STB"
    # --- priced_access ------------------------------------------------- #
    # Thresholds each student contributes per evaluation. Level 0 is tight at
    # the incumbent; deeper levels price what the student loses when their best
    # accessible options are taken away.
    priced_access_cut_levels: int = 3
    # Where the congestion prices come from, and a multiplier on them. Any
    # non-negative price vector keeps the surrogate a valid upper bound, so the
    # scale is free to tune: 0 reduces it to best-school-in-zone.
    priced_access_price_source: str = "transport"
    priced_access_price_scale: float = 1.0
    # Valid inequalities on the co-zoning indicators. The LP relaxation of the
    # access linearization can push many indicators to 1 at once by spreading
    # the assignment variables fractionally, which makes every cut slack and
    # leaves the a-priori constant as the only bound. Both families below cut
    # that off. See optimization/solvers/cpsat.py.
    choice_access_triangle: bool = False
    choice_access_cardinality: bool = False
    choice_access_triangle_limit: int = 200_000
    # Create the missing (school, school) side of each near-triangle. The cut
    # pairs are near-bipartite, and a bipartite graph has no triangles, so
    # without this the transitivity family has almost nothing to bind on.
    choice_access_complete: bool = False
    choice_access_completion_limit: int = 20_000
    dw_objective: str = "mid"
    dw_recom_samples: int = 500
    dw_recom_chains: int = 4
    dw_recom_time_limit: float = 60.0
    # Barrier without crossover, so the pricing problem is aimed at an
    # interior dual point rather than a degenerate set-partitioning vertex.
    dw_master_method: str = "barrier"
    # Wentges smoothing weight on this round's duals; 1.0 is no smoothing.
    dw_dual_smoothing: float = 1.0
    # The budgeted-elastic master. K, as a proportion of the district's total
    # student mass (floored at one unit per node), that the Phase-II cover rows
    # may carry as coverage mismatch: `sum(lambda) + d_v - e_v = 1` with
    # `sum_v w_v (d_v + e_v) <= K`. It exists because the measured obstruction
    # is that the master LP's feasible set is a single point, so a priced zone
    # colliding with the incumbent's other zones enters with a zero step
    # length. A budget rather than a penalty because the row's dual is then the
    # penalty M, chosen by the LP every round instead of guessed in
    # welfare-per-node units. Relaxation at every value, so the bound stays
    # valid; 0.0 is the exact master and the default until K is swept.
    dw_overlap_prop: float = 0.0
    # Integer units the CP-SAT pricing objective is measured in.
    dw_pricing_scale: int = 1000
    dw_pricing_columns_per_call: int = 8
    dw_pricing_parallel: bool = True
    # Seconds one column-generation round may spend pricing, doubled whenever a
    # round neither adds a column nor proves a bound. Uncapped, the first round
    # on a real instance eats the whole budget and nothing about convergence is
    # observable; `.inf` restores that.
    dw_pricing_time_limit: float = 30.0
    # Two-zone redraws: re-partition a pair of the incumbent's zones optimally,
    # leaving the other Z - 2 alone. Every solution of that model is a complete
    # tiling, so this is the only column source here that grows the number of
    # ways the master can cover V -- which is the measured obstruction. It
    # contributes columns and incumbents only, never a bound.
    dw_redraw: bool = True
    dw_redraw_time_limit: float = 60.0
    dw_redraw_splits_per_call: int = 8

    # --- data ingestion ----------------------------------------------- #
    data: dict[str, Any] = field(default_factory=_legacy_data_config)
    _resolved_data_scenario: InitVar[DataScenario | None] = None

    def __post_init__(self, _resolved_data_scenario: DataScenario | None):
        if not isinstance(self.data, Mapping):
            raise ValueError("data must be a {scenario, overrides} map.")
        self.data = deepcopy(dict(self.data))
        self._data_scenario = _resolved_data_scenario or load_scenario(self.data)

        self.unit
        specs = [LevelSpec.parse(level) for level in self.levels]
        unsupported = [
            level.name
            for level in specs
            if not level.is_base
            and level.depth not in LEVEL_NODE_TARGETS.get(level.unit, {})
        ]
        if unsupported:
            raise ValueError(
                f"No predefined graph size for levels: {', '.join(unsupported)}."
            )
        import optimization.solvers  # noqa: F401  (registers the solvers)
        from optimization.solvers.base import available_solvers
        from optimization.solvers.recom import ADAPTIVE_OBJECTIVES
        from optimization.strategies.base import available_strategies

        for name, choices in {
            "strategy": available_strategies(),
            "solver": available_solvers(),
            "adaptive_short_bursts_objective": ADAPTIVE_OBJECTIVES,
            **_CHOICES,
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of: {', '.join(choices)}.")
        for name in _BOOLEANS:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a Boolean.")
        for name, minimum in _INTEGERS.items():
            value = getattr(self, name)
            if not _is_integer(value) or (minimum is not None and value < minimum):
                raise ValueError(f"{name} must be {_INTEGER_KINDS[minimum]}.")
        for name, (message, bounds) in _NUMBERS.items():
            self._check_number(name, message, **bounds)
        for name in ("saa_tie_breaking_method", "stable_cutoff_tie_breaking_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or value.upper() not in {"MTB", "STB"}:
                raise ValueError(f"{name} must be one of: MTB, STB.")
            setattr(self, name, value.upper())
        if self.strategy in _MAX_ITERATION_STRATEGIES and (
            not _is_integer(self.max_iterations) or self.max_iterations <= 0
        ):
            raise ValueError(
                f"max_iterations must be a positive integer for {self.strategy}."
            )
        if self.strategy in _TOLERANCE_STRATEGIES:
            self._check_number(
                "tolerance",
                f"must be finite and non-negative for {self.strategy}",
                minimum=0.0,
                finite=True,
            )

        if self.strategy == "recursive" and self.looseness < 1.0:
            raise ValueError("looseness must be >= 1.0 for recursive runs.")
        if self.solver == "cp_single_zone" and self.strategy != "single":
            raise ValueError("cp_single_zone requires strategy='single'.")
        if self.enumerated_solutions > 0:
            if self.solver not in {"cp_bool", "cp_int"}:
                raise ValueError(
                    "enumerated_solutions requires solver='cp_bool' or 'cp_int'."
                )
            if self.strategy != "single":
                raise ValueError("enumerated_solutions requires strategy='single'.")
        self._validate_school_distance()
        if isinstance(self.max_distance, str):
            if self.max_distance.strip().lower() == "auto":
                self.max_distance = "auto"
            else:
                try:
                    val = float(self.max_distance)
                    if math.isnan(val) or val < 0:
                        raise ValueError
                    self.max_distance = val
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "max_distance must be a non-negative float, 'inf', or 'auto'."
                    ) from exc
        elif isinstance(self.max_distance, bool) or not isinstance(
            self.max_distance, (int, float)
        ):
            raise ValueError(
                "max_distance must be a non-negative float, 'inf', or 'auto'."
            )
        else:
            self.max_distance = float(self.max_distance)
            if math.isnan(self.max_distance) or self.max_distance < 0:
                raise ValueError(
                    "max_distance must be a non-negative float, 'inf', or 'auto'."
                )
        # Resolve the strict scenario-backed selectors eagerly.
        self.years
        self.grades
        self.student_population
        self.rounds
        self.special_programs
        self.program_population
        self.capacity_scenario
        self.include_k8
        self.include_citywide_zoning
        self.include_citywide_choice_opt
        self.include_mission_bay
        self.frl_estimate
        self.outside_district_students
        if self.strategy == "dantzig_wolfe":
            if self.solver != "cp_bool":
                raise ValueError("dantzig_wolfe requires solver='cp_bool'.")
            if self.include_citywide_choice_opt:
                raise ValueError(
                    "dantzig_wolfe requires include_citywide_choice_opt=false."
                )
            if self.budget_accounting != "wall_clock":
                raise ValueError(
                    "dantzig_wolfe requires budget_accounting='wall_clock'."
                )
            if self.dw_objective != "boundary":
                if self.program_population != "All":
                    raise ValueError(
                        "DW welfare objectives require program_population='All'."
                    )
                if self._data_scenario.filter(
                    "optimization", "geography_vintage"
                ) != self._data_scenario.filter("assignment", "geography_vintage"):
                    raise ValueError(
                        "DW welfare objectives require matching geography_vintage "
                        "values."
                    )
        if self.strategy in {"mid", "mid_decomp", "saa"}:
            if self.strategy == "saa" and self.solver not in {"cp_bool", "mip"}:
                raise ValueError("saa requires solver='cp_bool' or solver='mip'.")
            if self.strategy != "saa" and self.solver != "cp_bool":
                raise ValueError(f"{self.strategy} requires solver='cp_bool'.")
            if self.program_population != "All":
                raise ValueError(f"{self.strategy} requires program_population='All'.")
            optimization_vintage = self._data_scenario.filter(
                "optimization", "geography_vintage"
            )
            assignment_vintage = self._data_scenario.filter(
                "assignment", "geography_vintage"
            )
            if optimization_vintage != assignment_vintage:
                raise ValueError(
                    f"{self.strategy} requires matching optimization and assignment "
                    "geography_vintage values."
                )
        self.mid_complementary_slackness_slack = (
            normalize_complementary_slackness_slack(
                self.mid_complementary_slackness_slack
            )
        )
        if self.hint_zoning is not None:
            if not isinstance(self.hint_zoning, (str, Path)) or not str(
                self.hint_zoning
            ):
                raise ValueError("hint_zoning must be a file path or null.")
            self.hint_zoning = str(Path(self.hint_zoning).expanduser())
        if self.recom_iterations < 0 and not self.solve_time_limits:
            raise ValueError(
                "solve_time_limits must include a value when recom_iterations is negative."
            )
        if (
            self.adaptive_short_bursts_objective == "capacity_match"
            and self.adaptive_short_bursts_unassigned_weight == 0
            and self.adaptive_short_bursts_designated_weight == 0
        ):
            raise ValueError(
                "capacity_match needs a positive unassigned or designated weight."
            )

    def _check_number(
        self,
        name: str,
        message: str,
        *,
        minimum: float = -math.inf,
        maximum: float = math.inf,
        open_minimum: bool = False,
        finite: bool = False,
    ) -> None:
        """Coerce ``name`` to a float within its bounds, or raise ``message``."""

        value = getattr(self, name)
        try:
            if isinstance(value, bool):
                raise TypeError
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} {message}.") from exc
        if (
            math.isnan(value)
            or (finite and not math.isfinite(value))
            or not minimum <= value <= maximum
            or (open_minimum and value == minimum)
        ):
            raise ValueError(f"{name} {message}.")
        setattr(self, name, value)

    def _validate_school_distance(self) -> None:
        if self.compactness_objective not in {"cut_edges", "school_distance"}:
            raise ValueError(
                "compactness_objective must be one of: cut_edges, school_distance."
            )
        if self.compactness_objective != "cut_edges" and self.solver in {
            "recom",
            "relaxed_recom",
            "short_bursts",
            "adaptive_short_bursts",
        }:
            raise ValueError(
                "compactness_objective applies to CP/MIP solvers; ReCom solvers "
                "take adaptive_short_bursts_objective: school_distance instead."
            )
        weights = []
        for name in ("school_distance_area_weight", "school_distance_student_weight"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a number in [0, 1].")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number in [0, 1].") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a number in [0, 1].")
            setattr(self, name, value)
            weights.append(value)
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError(
                "school_distance_area_weight and school_distance_student_weight "
                f"must sum to 1, got {sum(weights):g}."
            )

    @property
    def unit(self) -> str:
        """The one Census unit every level shares (the base graph is per unit)."""
        units = {LevelSpec.parse(level).unit for level in self.levels}
        if len(units) != 1:
            raise ValueError(f"All levels must share one unit; got {sorted(units)}.")
        return units.pop()

    # ------------------------------------------------------------------ #
    # scenario-backed data settings
    # ------------------------------------------------------------------ #
    @property
    def data_scenario(self) -> DataScenario:
        """The immutable scenario loaded from the serializable ``data`` field."""
        return self._data_scenario

    @property
    def years(self) -> tuple[str, ...]:
        return tuple(self._data_scenario.filter("optimization", "years"))

    @property
    def grades(self) -> tuple[str, ...]:
        return tuple(self._data_scenario.filter("optimization", "grades"))

    @property
    def student_population(self) -> str:
        return self._data_scenario.filter("optimization", "student_population")

    @property
    def rounds(self) -> str | tuple[int, ...]:
        value = self._data_scenario.filter("optimization", "rounds")
        return value if value == "all" else tuple(value)

    @property
    def special_programs(self) -> str:
        return self._data_scenario.filter("optimization", "special_programs")

    @property
    def program_population(self) -> str:
        return self._data_scenario.filter("optimization", "program_population")

    @property
    def capacity_scenario(self) -> str:
        return self._data_scenario.filter("optimization", "capacity_scenario")

    @property
    def include_k8(self) -> bool:
        return self._data_scenario.filter("optimization", "include_k8")

    @property
    def include_citywide_zoning(self) -> bool:
        """Whether citywide schools exist for the base zoning problem.

        Governs the school table, so it decides whether a citywide school gets
        a graph node and whether its seats count toward the capacity
        (``overage``/``shortage``) and school-count balance of the zone that
        contains it. It is a property of the geography the optimizer partitions
        and says nothing about welfare.
        """
        return self._data_scenario.filter("optimization", "include_citywide_zoning")

    @property
    def include_citywide_choice_opt(self) -> bool:
        """Whether citywide programs are alternatives in the welfare markets.

        Governs the MID and SAA markets and the MNL zoning utility -- every
        method that optimizes or reports choice welfare. Independent of
        :attr:`include_citywide_zoning`: a citywide school with no graph node
        is still an option every student holds under every zoning, which is
        exactly what these markets model with ``school_node=None``.
        """
        return self._data_scenario.filter("optimization", "include_citywide_choice_opt")

    @property
    def include_mission_bay(self) -> bool:
        return self._data_scenario.filter("optimization", "include_mission_bay")

    @property
    def outside_district_students(self) -> str:
        return self._data_scenario.filter("optimization", "outside_district_students")

    @property
    def frl_estimate(self) -> str | None:
        return self._data_scenario.filter("optimization", "frl_estimate")

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str) -> "OptimizationConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Optimization config YAML must contain a map.")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}.")
        if "data" in raw:
            raw["data"] = anchor_data_config(raw["data"], config_path.parent)
        return cls(**raw)

    # ------------------------------------------------------------------ #
    # factories
    # ------------------------------------------------------------------ #
    def make_dataset(self):
        from optimization.data.dataset import Dataset

        return Dataset(self)

    def make_solver(self, output_dir: str | None = None):
        from optimization.solvers import get_solver

        options = {
            "solve_time_limit": self.solve_time_limits[0],
            "relative_gap_limit": self.gap_limits[0],
            "seed": self.seed,
            "workers": self.workers,
            "linearization_level": self.linearization_level,
            "cp_model_probing_level": self.cp_model_probing_level,
            "symmetry_level": self.symmetry_level,
            "cp_sat_search_strategy": self.cp_sat_search_strategy,
            "hints": self.hints,
            "feasible_hint_time_limit": self.feasible_hint_time_limit,
            "save_solver_logs": self.save_solver_logs,
            "save_solver_progress": self.save_solver_progress,
            "secondary_objective": self.secondary_objective,
            "choice_access_triangle": self.choice_access_triangle,
            "choice_access_cardinality": self.choice_access_cardinality,
            "choice_access_triangle_limit": self.choice_access_triangle_limit,
            "choice_access_complete": self.choice_access_complete,
            "choice_access_completion_limit": self.choice_access_completion_limit,
            "centroid_neighbor_radius": self.centroid_neighbor_radius,
            "recom_iterations": self.recom_iterations,
            "short_bursts_length": self.short_bursts_length,
            "short_bursts_method": self.short_bursts_method,
            "adaptive_short_bursts_lr": self.adaptive_short_bursts_lr,
            "adaptive_short_bursts_temperature": self.adaptive_short_bursts_temperature,
            "adaptive_short_bursts_pair_selector": (
                self.adaptive_short_bursts_pair_selector
            ),
            "adaptive_short_bursts_objective": self.adaptive_short_bursts_objective,
            "adaptive_short_bursts_unassigned_weight": (
                self.adaptive_short_bursts_unassigned_weight
            ),
            "adaptive_short_bursts_designated_weight": (
                self.adaptive_short_bursts_designated_weight
            ),
            # The adaptive short bursts' welfare scorer reads it here; the
            # MID strategies read their own copy.
            "mid_lottery_scale": self.mid_lottery_scale,
        }
        if output_dir is not None:
            options["output_dir"] = output_dir
        return get_solver(self.solver, **options)

    def make_strategy(self):
        from optimization.strategies import get_strategy

        return get_strategy(
            self.strategy,
            levels=self.levels,
            solve_time_limits=self.solve_time_limits,
            carry_over_compute=self.carry_over_compute,
            budget_accounting=self.budget_accounting,
            gap_limits=self.gap_limits,
            enumerated_solutions=self.enumerated_solutions,
            seed=self.seed,
            hints=self.hints,
            feasible_hint_time_limit=self.feasible_hint_time_limit,
            looseness=self.looseness,
            boundary_radius=self.boundary_radius,
            boundary_prop=self.boundary_prop,
            max_iterations=self.max_iterations,
            choice_model_method=self.choice_model_method,
            choice_utility_scale=self.choice_utility_scale,
            choice_utility_hints=self.choice_utility_hints,
            tolerance=self.tolerance,
            mid_lottery_scale=self.mid_lottery_scale,
            mid_transport_bounds=self.mid_transport_bounds,
            mid_complementary_slackness=self.mid_complementary_slackness,
            mid_complementary_slackness_slack=self.mid_complementary_slackness_slack,
            saa_num_seeds=self.saa_num_seeds,
            saa_tie_breaking_method=self.saa_tie_breaking_method,
            saa_welfare_bound=self.saa_welfare_bound,
            zoned_transport_workers=self.zoned_transport_workers,
            saa_relative_gap=self.saa_relative_gap,
            saa_multicut=self.saa_multicut,
            stable_cutoff_num_seeds=self.stable_cutoff_num_seeds,
            stable_cutoff_non_wastefulness=self.stable_cutoff_non_wastefulness,
            stable_cutoff_aggregate_stability=self.stable_cutoff_aggregate_stability,
            stable_cutoff_tie_breaking_method=self.stable_cutoff_tie_breaking_method,
            priced_access_cut_levels=self.priced_access_cut_levels,
            priced_access_price_source=self.priced_access_price_source,
            priced_access_price_scale=self.priced_access_price_scale,
            dw_objective=self.dw_objective,
            dw_recom_samples=self.dw_recom_samples,
            dw_recom_chains=self.dw_recom_chains,
            dw_recom_time_limit=self.dw_recom_time_limit,
            dw_master_method=self.dw_master_method,
            dw_dual_smoothing=self.dw_dual_smoothing,
            dw_overlap_prop=self.dw_overlap_prop,
            dw_pricing_scale=self.dw_pricing_scale,
            dw_pricing_columns_per_call=self.dw_pricing_columns_per_call,
            dw_pricing_parallel=self.dw_pricing_parallel,
            dw_pricing_time_limit=self.dw_pricing_time_limit,
            dw_redraw=self.dw_redraw,
            dw_redraw_time_limit=self.dw_redraw_time_limit,
            dw_redraw_splits_per_call=self.dw_redraw_splits_per_call,
        )
