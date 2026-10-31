"""Zoning-level MNL utility evaluation and linearized choice cuts."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import pandas as pd
from loaders import DataScenario, load_student_records
from scipy.special import logsumexp

from choice.estimate import (
    area_key,
    identity_text,
    load_estimate,
    node_area_ids,
    program_school,
    school_to_node,
    student_area_column,
)
from choice.objective import ChoiceCut, ChoiceEvaluation, ChoiceModel, ChoiceTerm
from optimization.data import loaders as optimization_loaders
from optimization.problem import ZoneProblem

# Loaded (students, estimate, school columns) by scenario source fingerprint,
# which covers source contents and filters. Metrics build a fresh model per run,
# so a sweep's metric pass otherwise re-read both tables (~5 s) for every run.
# The frames are read-only once loaded: ``_merged`` joins into a new frame.
_LOADED_TABLES: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]]]] = {}


@dataclass(frozen=True)
class _PreparedZoningData:
    merged: pd.DataFrame
    zone_to_cols: dict[int, list[str]]
    student_area_col: str


@dataclass(frozen=True)
class _PreassignmentUtility:
    prepared: _PreparedZoningData
    block_utilities: pd.Series
    total: float


class MNLChoiceModel(ChoiceModel):
    """Evaluate MNL welfare for a zoning and build Benders-style cuts.

    The utility matrix is student-by-program. A zone offers every program whose
    school is located in that zone; a student's zoning utility is either the max
    utility or log-sum-exp utility across those offered programs.

    ``citywide_schools`` are offered by *every* zone instead. Availability here
    is read off the graph -- a zone offers the schools sitting on its nodes --
    and a citywide school has no node whenever ``include_citywide_zoning`` is
    off, so without this set it belongs to no zone and drops out of the logsum
    entirely rather than being available everywhere. Pass the set when
    ``include_citywide_choice_opt`` is on; the empty default reproduces the
    zone-located-only behaviour.

    Both utilities are submodular in the offered set, so the cuts are the
    Nemhauser-Wolsey upper bounds around a reference set. Their ground set is
    the schools some zoning can offer -- node-hosted ones plus the citywide
    ones -- and a citywide school, offered whatever the zoning, is in every
    reference set and carries no co-zoning gradient even when it sits on a node.
    """

    def __init__(
        self,
        data: DataScenario,
        *,
        method: str = "logsum",
        area_column: str | None = None,
        empty_utility: float = -1e10,
        citywide_schools: Iterable[object] = (),
    ):
        if method not in {"max", "logsum"}:
            raise ValueError("MNL utility method must be 'max' or 'logsum'.")
        if not isinstance(data, DataScenario):
            raise TypeError("MNL zoning utility data must be a DataScenario.")
        self.data = data
        self.method = method
        self.area_column = area_column
        self.empty_utility = float(empty_utility)
        self.citywide_schools = frozenset(
            int(school_id) for school_id in citywide_schools
        )

        self.utility_df: pd.DataFrame | None = None
        self.student_df: pd.DataFrame | None = None
        self.school_to_cols: dict[int, list[str]] = {}

    def evaluate(self, problem: ZoneProblem, assignment: dict[int, int]) -> float:
        return self.preassignment_utility(problem, assignment)

    def preassignment_utility(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> float:
        """Total student utility from the schools available in each zone."""

        return self._preassignment_utility(problem, assignment).total

    def evaluate_with_cuts(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> ChoiceEvaluation:
        utility = self._preassignment_utility(problem, assignment)
        prepared = utility.prepared
        block_impacts = self._block_impacts(
            problem,
            assignment,
            prepared.merged,
            prepared.zone_to_cols,
            prepared.student_area_col,
        )
        cuts = self._build_cuts(
            problem,
            assignment,
            utility.block_utilities,
            block_impacts,
        )
        return ChoiceEvaluation(utility=utility.total, cuts=tuple(cuts))

    def choice_utility_hint_cuts(self, problem: ZoneProblem) -> tuple[ChoiceCut, ...]:
        """Initial cuts around each area's nearest average-zone-size school set."""

        if problem.Z <= 0:
            return ()
        schools_per_zone = int(
            sum(problem.num_schools(node) for node in problem.nodes) / problem.Z
        )
        if schools_per_zone <= 0:
            return ()

        self._ensure_loaded()
        student_area_col = self._student_area_col(problem)
        merged = self._merged()
        merged["_area_key"] = merged[student_area_col].map(area_key)

        school_items = sorted(self._zoned_schools(problem).items())
        if not school_items:
            return ()
        citywide_cols = self._cols(self.citywide_schools)
        all_cols = self._cols(sid for sid, _ in school_items) + citywide_cols

        def summed_utility(frame: pd.DataFrame, cols: list[str]) -> float:
            if frame.empty:
                return 0.0
            return finite_or(float(np.sum(self._utilities_for_cols(frame, cols))))

        distance_dict = problem.G.graph.get("distance_dict", {})
        cuts: list[ChoiceCut] = []
        empty_frame = merged.iloc[0:0]
        for node in problem.nodes:
            block_ids = set(node_area_ids(problem.G.nodes[node]))
            group = (
                merged[merged["_area_key"].isin(block_ids)]
                if block_ids and not merged.empty
                else empty_frame
            )
            nearest_schools = {
                sid
                for sid, _school_node in sorted(
                    school_items,
                    key=lambda item: (
                        distance_dict.get(node, {}).get(item[1], float("inf")),
                        item[1],
                        item[0],
                    ),
                )[:schools_per_zone]
            }
            baseline_cols = self._cols(nearest_schools) + citywide_cols
            baseline_value = summed_utility(group, baseline_cols)
            full_value = summed_utility(group, all_cols)
            constant = baseline_value
            coeffs: dict[int, float] = {}

            for sid, school_node in school_items:
                sid_cols = set(self._cols([sid]))
                if sid in nearest_schools:
                    remaining_cols = [col for col in all_cols if col not in sid_cols]
                    grad = full_value - summed_utility(group, remaining_cols)
                    constant -= grad
                else:
                    grad = summed_utility(group, baseline_cols + sorted(sid_cols))
                    grad -= baseline_value
                grad = finite_or(grad)
                if grad:
                    coeffs[school_node] = coeffs.get(school_node, 0.0) + grad

            constant = finite_or(constant)
            terms = tuple(
                ChoiceTerm(coefficient=coef, node=school_node)
                for school_node, coef in sorted(coeffs.items())
                if coef
                and bool(
                    problem.candidate_zones(node) & problem.candidate_zones(school_node)
                )
            )
            cuts.append(ChoiceCut(node=node, constant=constant, terms=terms))
        return tuple(cuts)

    def _preassignment_utility(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> _PreassignmentUtility:
        prepared = self._prepare(problem, assignment)
        utilities = self._student_utilities(prepared.merged, prepared.zone_to_cols)
        block_utilities = self._block_utilities(
            prepared.merged, utilities, prepared.student_area_col
        )
        total = float(block_utilities.sum()) if not block_utilities.empty else 0.0
        return _PreassignmentUtility(
            prepared=prepared,
            block_utilities=block_utilities,
            total=total,
        )

    def _ensure_loaded(self) -> None:
        if self.utility_df is not None and self.student_df is not None:
            return

        key = self.data.source_fingerprint
        if key not in _LOADED_TABLES:
            _LOADED_TABLES[key] = self._load_tables()
        self.student_df, self.utility_df, self.school_to_cols = _LOADED_TABLES[key]
        self._warn_on_estimate_coverage()

    def _load_tables(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]]]:
        student_df = load_student_records(
            self.data,
            "assignment.students",
            filter_group="assignment",
            low_memory=False,
        )
        if "studentno" not in student_df.columns:
            paths = [
                str(source.path) for source in self.data.sources("assignment.students")
            ]
            raise ValueError(f"MNL student files {paths} lack studentno column.")
        student_df["studentno"] = student_df["studentno"].map(identity_text)
        utility_df = load_estimate(self.data, student_df["studentno"])

        school_to_cols: dict[int, list[str]] = {}
        for col in utility_df.columns:
            school_to_cols.setdefault(program_school(col), []).append(col)
        return student_df, utility_df, school_to_cols

    _COVERAGE_WARNING_THRESHOLD = 0.5

    def _warn_on_estimate_coverage(self) -> None:
        """Warn when the estimate covers few of the students being scored.

        The estimate is a per-student utility matrix, and ``_prepare`` inner-
        joins it to the student table on ``studentno``. A student the estimate
        does not name therefore contributes nothing, and if it names none of
        them the total is ``0.0`` -- a number indistinguishable from "every
        student got their worst option". That is silent by construction, so it
        is worth saying out loud rather than leaving in a metrics column.

        This only warns. Partial coverage is legitimate for some analyses, and
        raising here would break callers that already accept it.
        """
        assert self.utility_df is not None
        assert self.student_df is not None
        students = len(self.student_df)
        if not students:
            return
        covered = len(self.utility_df)
        fraction = covered / students
        if fraction >= self._COVERAGE_WARNING_THRESHOLD:
            return
        detail = (
            f"MNL utility estimate {self.data.source('choice.estimate').path} "
            f"covers {covered} of {students} selected students ({fraction:.1%}). "
            "Utilities for the rest are absent, not zero-valued, so every MNL and "
            "choice metric derived from this estimate understates welfare"
        )
        if covered == 0:
            detail += (
                ". With no student in common the total is exactly 0.0, which is "
                "not a welfare measurement. Fit an estimate for this cohort, or "
                "read the choice metrics as unavailable rather than as zero"
            )
        warnings.warn(detail + ".", stacklevel=2)

    def _merged(self) -> pd.DataFrame:
        """Selected students joined to their estimate rows (inner join)."""
        assert self.utility_df is not None
        assert self.student_df is not None
        return self.student_df.join(self.utility_df, on="studentno", how="inner")

    def _student_area_col(self, problem: ZoneProblem) -> str:
        assert self.student_df is not None
        column = self.area_column or student_area_column(problem)
        if column not in self.student_df.columns:
            raise ValueError(f"MNL student file lacks {column!r}.")
        return column

    def _cols(self, schools: Iterable[int]) -> list[str]:
        return [
            col for sid in sorted(schools) for col in self.school_to_cols.get(sid, [])
        ]

    def _zoned_schools(self, problem: ZoneProblem) -> dict[int, int]:
        """Schools a zoning can give or take away: on a node, estimated, not citywide."""
        return {
            sid: node
            for sid, node in school_to_node(problem).items()
            if sid in self.school_to_cols and sid not in self.citywide_schools
        }

    def _prepare(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> _PreparedZoningData:
        self._ensure_loaded()
        student_area_col = self._student_area_col(problem)

        zone_to_schools: dict[int, set[int]] = {z: set() for z in range(problem.Z)}
        area_to_zone: dict[str, int] = {}
        for node, zone in assignment.items():
            attrs = problem.G.nodes[node]
            zone_to_schools.setdefault(zone, set()).update(
                int(sid) for sid in attrs.get("school_ids", [])
            )
            for area_id in node_area_ids(attrs):
                area_to_zone[area_id] = zone
        # Held by every zone, so no zoning can take them away. Added after the
        # per-node pass because a citywide school may sit on a node too, when
        # ``include_citywide_zoning`` is on -- a set union keeps that idempotent.
        for schools in zone_to_schools.values():
            schools.update(self.citywide_schools)

        merged = self._merged()
        merged["_area_key"] = merged[student_area_col].map(area_key)
        merged["assigned_zone"] = merged["_area_key"].map(area_to_zone)
        merged = merged.dropna(subset=["assigned_zone"]).copy()
        if merged.empty:
            return _PreparedZoningData(merged, {}, student_area_col)
        merged["assigned_zone"] = merged["assigned_zone"].astype(int)
        merged.reset_index(drop=True, inplace=True)

        zone_to_cols = {
            zone: self._cols(schools) for zone, schools in zone_to_schools.items()
        }
        return _PreparedZoningData(merged, zone_to_cols, student_area_col)

    def _student_utilities(
        self, merged: pd.DataFrame, zone_to_cols: dict[int, list[str]]
    ) -> np.ndarray:
        utilities = np.full(len(merged), self.empty_utility, dtype=float)
        if merged.empty:
            return utilities
        for zone, group in merged.groupby("assigned_zone"):
            cols = zone_to_cols.get(int(zone), [])
            utilities[group.index.to_numpy()] = self._utilities_for_cols(group, cols)
        return _finite_array(utilities, self.empty_utility)

    def _utilities_for_cols(self, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.full(len(frame), self.empty_utility, dtype=float)
        data = frame[cols].to_numpy(dtype=float)
        if self.method == "max":
            utilities = np.max(data, axis=1)
        else:
            utilities = logsumexp(data, axis=1)
        return _finite_array(utilities, self.empty_utility)

    def _block_utilities(
        self, merged: pd.DataFrame, utilities: np.ndarray, student_area_col: str
    ) -> pd.Series:
        if merged.empty:
            return pd.Series(dtype=float)
        block_keys = merged[student_area_col].map(area_key)
        return pd.Series(utilities, index=block_keys).groupby(level=0).sum()

    def _block_impacts(
        self,
        problem: ZoneProblem,
        assignment: dict[int, int],
        merged: pd.DataFrame,
        zone_to_cols: dict[int, list[str]],
        student_area_col: str,
    ) -> dict[str, dict[int, dict[str, float]]]:
        zoned = self._zoned_schools(problem)
        school_to_current_zone = {
            sid: assignment[node] for sid, node in zoned.items() if node in assignment
        }

        impacts: dict[str, dict[int, dict[str, float]]] = {}
        if merged.empty:
            return impacts
        full_cols = self._cols(zoned) + self._cols(self.citywide_schools)

        for zone, group in merged.groupby("assigned_zone"):
            zone = int(zone)
            cols = zone_to_cols.get(zone, [])
            baseline = self._utilities_for_cols(group, cols)
            full_baseline = self._utilities_for_cols(group, full_cols)
            block_keys = group[student_area_col].map(area_key).to_numpy()

            for sid in sorted(zoned):
                sid_cols = self.school_to_cols[sid]
                sid_col_set = set(sid_cols)
                full_remaining_cols = [
                    col for col in full_cols if col not in sid_col_set
                ]
                full_removed_utils = self._utilities_for_cols(
                    group, full_remaining_cols
                )
                full_remove_impacts = full_baseline - full_removed_utils
                is_in_zone = school_to_current_zone.get(sid) == zone
                if is_in_zone:
                    remaining_cols = [col for col in cols if col not in sid_col_set]
                    new_utils = self._utilities_for_cols(group, remaining_cols)
                    student_impacts = baseline - new_utils
                    impact_type = "remove"
                else:
                    added_cols = cols + [col for col in sid_cols if col not in cols]
                    new_utils = self._utilities_for_cols(group, added_cols)
                    student_impacts = new_utils - baseline
                    impact_type = "add"

                student_impacts = _finite_array(student_impacts, 0.0)
                full_remove_impacts = _finite_array(full_remove_impacts, 0.0)
                for block_id, impact, full_remove in zip(
                    block_keys, student_impacts, full_remove_impacts
                ):
                    if block_id is None:
                        continue
                    school_map = impacts.setdefault(block_id, {})
                    type_map = school_map.setdefault(
                        sid, {"add": 0.0, "remove": 0.0, "full_remove": 0.0}
                    )
                    type_map[impact_type] += float(impact)
                    type_map["full_remove"] += float(full_remove)
        return impacts

    def _build_cuts(
        self,
        problem: ZoneProblem,
        assignment: dict[int, int],
        block_utilities: pd.Series,
        block_impacts: dict[str, dict[int, dict[str, float]]],
    ) -> list[ChoiceCut]:
        zoned = self._zoned_schools(problem)
        zone_current_schools: dict[int, set[int]] = {z: set() for z in range(problem.Z)}
        for sid, node in zoned.items():
            if node in assignment:
                zone_current_schools.setdefault(assignment[node], set()).add(sid)

        cuts: list[ChoiceCut] = []
        for node in problem.nodes:
            block_ids = node_area_ids(problem.G.nodes[node])
            value = 0.0
            total_impacts: dict[int, dict[str, float]] = {}
            has_data = False

            for block_id in block_ids:
                if block_id in block_utilities.index:
                    value += float(block_utilities.loc[block_id])
                    has_data = True
                for sid, by_type in block_impacts.get(block_id, {}).items():
                    target = total_impacts.setdefault(
                        sid, {"add": 0.0, "remove": 0.0, "full_remove": 0.0}
                    )
                    for key in target:
                        target[key] += float(by_type.get(key, 0.0))

            assigned_zone = assignment.get(node)
            current_schools = zone_current_schools.get(assigned_zone, set())
            constant = value if has_data or total_impacts else 0.0
            coeffs: dict[int, float] = {}

            for sid, by_type in total_impacts.items():
                school_node = zoned[sid]
                if sid in current_schools:
                    grad = by_type["full_remove"]
                    constant -= grad
                else:
                    grad = by_type["add"]
                if grad:
                    coeffs[school_node] = coeffs.get(school_node, 0.0) + grad

            terms = tuple(
                ChoiceTerm(coefficient=coef, node=school_node)
                for school_node, coef in sorted(coeffs.items())
                if coef
                and bool(
                    problem.candidate_zones(node) & problem.candidate_zones(school_node)
                )
            )
            anchor_access = tuple(
                (
                    (node, term.node),
                    int(assignment.get(node) == assignment.get(term.node)),
                )
                for term in terms
            )
            cuts.append(
                ChoiceCut(
                    node=node,
                    constant=constant,
                    terms=terms,
                    anchor_access=anchor_access,
                )
            )
        return cuts


def build_mnl_choice_model(
    data: DataScenario,
    *,
    method: str = "logsum",
) -> MNLChoiceModel:
    """Build the sole supported zoning choice model.

    Resolves ``include_citywide_choice_opt`` from the scenario rather than
    taking it as an argument, so every caller -- the strategy, the metrics, the
    offline replays -- gets the same answer for the same scenario.
    """

    citywide: frozenset[int] = frozenset()
    if data.filter("optimization", "include_citywide_choice_opt", False):
        citywide = optimization_loaders.citywide_school_ids(data)
    return MNLChoiceModel(data=data, method=method, citywide_schools=citywide)


def _finite_array(values: np.ndarray, replacement: float) -> np.ndarray:
    return np.nan_to_num(
        values,
        nan=replacement,
        posinf=replacement if replacement > 0 else abs(replacement),
        neginf=replacement,
    )


def finite_or(value: float, replacement: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else float(replacement)
