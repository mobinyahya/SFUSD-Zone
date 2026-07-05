"""Zoning-level MNL utility evaluation and linearized choice cuts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Zone_Generation.choice.objective import ChoiceCut, ChoiceEvaluation, ChoiceTerm
from Zone_Generation.optimization.problem import ZoneProblem


DEFAULT_UTILITY_PATH = "/share/data/school_choice/simulation-files/choice-model/estimates_2324_exp8_0514.csv"
DEFAULT_STUDENT_PATH = "/share/data/school_choice/Data/Cleaned/r1_filter_student_without_specialprogs_2324.csv"


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


class MNLZoningUtility:
    """Evaluate MNL welfare for a zoning and build Benders-style cuts.

    The utility matrix is student-by-program. A zone offers every program whose
    school is located in that zone; a student's zoning utility is either the max
    utility or log-sum-exp utility across those offered programs.
    """

    def __init__(
        self,
        *,
        method: str = "logsum",
        area_column: str | None = None,
        empty_utility: float = -1e10,
    ):
        if method not in {"max", "logsum"}:
            raise ValueError("MNL utility method must be 'max' or 'logsum'.")
        self.method = method
        self.area_column = area_column
        self.empty_utility = float(empty_utility)

        self.utility_df: pd.DataFrame | None = None
        self.student_df: pd.DataFrame | None = None
        self.school_to_cols: dict[str, list[str]] = {}

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

        utility_path = Path(DEFAULT_UTILITY_PATH).expanduser()
        student_path = Path(DEFAULT_STUDENT_PATH).expanduser()
        if not utility_path.exists():
            raise FileNotFoundError(f"MNL utility file not found: {utility_path}")
        if not student_path.exists():
            raise FileNotFoundError(f"MNL student file not found: {student_path}")

        utility_df = pd.read_csv(utility_path)
        if "studentno" not in utility_df.columns:
            raise ValueError(f"MNL utility file {utility_path} lacks studentno column.")
        utility_df["studentno"] = _normalize_studentno(utility_df["studentno"])

        school_to_cols: dict[str, list[str]] = {}
        for col in utility_df.columns:
            if col == "studentno":
                continue
            school_id = str(col).split("-", 1)[0]
            school_to_cols.setdefault(school_id, []).append(str(col))

        student_df = pd.read_csv(student_path, low_memory=False)
        if "studentno" not in student_df.columns:
            raise ValueError(f"MNL student file {student_path} lacks studentno column.")
        student_df["studentno"] = _normalize_studentno(student_df["studentno"])

        self.utility_df = utility_df.dropna(subset=["studentno"])
        self.student_df = student_df.dropna(subset=["studentno"])
        self.school_to_cols = school_to_cols

    def _prepare(
        self, problem: ZoneProblem, assignment: dict[int, int]
    ) -> _PreparedZoningData:
        self._ensure_loaded()
        assert self.utility_df is not None
        assert self.student_df is not None

        student_area_col = self.area_column or _student_area_column(problem)
        if student_area_col not in self.student_df.columns:
            raise ValueError(f"MNL student file lacks {student_area_col!r}.")

        zone_to_schools: dict[int, set[str]] = {z: set() for z in range(problem.Z)}
        area_to_zone: dict[str, int] = {}
        for node, zone in assignment.items():
            attrs = problem.G.nodes[node]
            zone_to_schools.setdefault(zone, set()).update(
                _school_key(sid) for sid in attrs.get("school_ids", [])
            )
            for area_id in _node_area_ids(attrs):
                area_to_zone[area_id] = zone

        merged = self.student_df.merge(self.utility_df, on="studentno").copy()
        merged["_area_key"] = merged[student_area_col].map(_area_key)
        merged["assigned_zone"] = merged["_area_key"].map(area_to_zone)
        merged = merged.dropna(subset=["assigned_zone"]).copy()
        if merged.empty:
            return _PreparedZoningData(merged, {}, student_area_col)
        merged["assigned_zone"] = merged["assigned_zone"].astype(int)
        merged.reset_index(drop=True, inplace=True)

        zone_to_cols: dict[int, list[str]] = {}
        for zone, schools in zone_to_schools.items():
            cols: list[str] = []
            for sid in schools:
                cols.extend(self.school_to_cols.get(sid, []))
            zone_to_cols[zone] = [col for col in cols if col in merged.columns]
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
        data = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        data = np.nan_to_num(data, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        if self.method == "max":
            utilities = np.max(data, axis=1)
        else:
            utilities = _logsumexp(data, axis=1)
        return _finite_array(utilities, self.empty_utility)

    def _block_utilities(
        self, merged: pd.DataFrame, utilities: np.ndarray, student_area_col: str
    ) -> pd.Series:
        if merged.empty:
            return pd.Series(dtype=float)
        block_keys = merged[student_area_col].map(_area_key)
        return pd.Series(utilities, index=block_keys).groupby(level=0).sum()

    def _block_impacts(
        self,
        problem: ZoneProblem,
        assignment: dict[int, int],
        merged: pd.DataFrame,
        zone_to_cols: dict[int, list[str]],
        student_area_col: str,
    ) -> dict[str, dict[str, dict[str, float]]]:
        school_to_current_zone: dict[str, int] = {}
        for node, zone in assignment.items():
            for sid in problem.G.nodes[node].get("school_ids", []):
                school_to_current_zone[_school_key(sid)] = zone

        impacts: dict[str, dict[str, dict[str, float]]] = {}
        if merged.empty:
            return impacts
        all_schools = sorted(self.school_to_cols)

        for zone, group in merged.groupby("assigned_zone"):
            zone = int(zone)
            cols = zone_to_cols.get(zone, [])
            baseline = self._utilities_for_cols(group, cols)
            block_keys = group[student_area_col].map(_area_key).to_numpy()

            for sid in all_schools:
                sid_cols = [
                    col for col in self.school_to_cols[sid] if col in group.columns
                ]
                if not sid_cols:
                    continue
                sid_utils = self._utilities_for_cols(group, sid_cols)
                is_in_zone = school_to_current_zone.get(sid) == zone
                if self.method == "logsum":
                    diff = sid_utils - baseline
                    if is_in_zone:
                        safe_diff = np.minimum(diff, -1e-15)
                        student_impacts = np.log1p(-np.exp(safe_diff))
                        impact_type = "remove"
                    else:
                        student_impacts = _log1pexp(diff)
                        impact_type = "add"
                else:
                    if is_in_zone:
                        remaining_cols = [col for col in cols if col not in sid_cols]
                        new_utils = self._utilities_for_cols(group, remaining_cols)
                        student_impacts = new_utils - baseline
                        impact_type = "remove"
                    else:
                        student_impacts = np.maximum(baseline, sid_utils) - baseline
                        impact_type = "add"

                student_impacts = _finite_array(student_impacts, 0.0)
                for block_id, impact in zip(block_keys, student_impacts):
                    if block_id is None:
                        continue
                    school_map = impacts.setdefault(block_id, {})
                    type_map = school_map.setdefault(sid, {"add": 0.0, "remove": 0.0})
                    type_map[impact_type] = type_map.get(impact_type, 0.0) + float(
                        impact
                    )
        return impacts

    def _build_cuts(
        self,
        problem: ZoneProblem,
        assignment: dict[int, int],
        block_utilities: pd.Series,
        block_impacts: dict[str, dict[str, dict[str, float]]],
    ) -> list[ChoiceCut]:
        school_to_node = _school_to_node(problem)
        zone_current_schools: dict[int, set[str]] = {z: set() for z in range(problem.Z)}
        for node, zone in assignment.items():
            zone_current_schools.setdefault(zone, set()).update(
                _school_key(sid) for sid in problem.G.nodes[node].get("school_ids", [])
            )

        cuts: list[ChoiceCut] = []
        for node in problem.nodes:
            block_ids = _node_area_ids(problem.G.nodes[node])
            value = 0.0
            total_impacts: dict[str, dict[str, float]] = {}
            has_data = False

            for block_id in block_ids:
                if block_id in block_utilities.index:
                    value += float(block_utilities.loc[block_id])
                    has_data = True
                for sid, by_type in block_impacts.get(block_id, {}).items():
                    target = total_impacts.setdefault(sid, {"add": 0.0, "remove": 0.0})
                    target["add"] += float(by_type.get("add", 0.0))
                    target["remove"] += float(by_type.get("remove", 0.0))

            assigned_zone = assignment.get(node)
            current_schools = zone_current_schools.get(assigned_zone, set())
            constant = value if has_data or total_impacts else 0.0
            coeffs: dict[int, float] = {}

            for sid, by_type in total_impacts.items():
                school_node = school_to_node.get(sid)
                if school_node is None:
                    continue
                if sid in current_schools:
                    grad = float(by_type.get("remove", 0.0))
                    constant -= grad
                else:
                    grad = float(by_type.get("add", 0.0))
                if grad:
                    coeffs[school_node] = coeffs.get(school_node, 0.0) + grad

            for zone in problem.candidate_zones(node):
                terms = tuple(
                    ChoiceTerm(coef, zone, school_node)
                    for school_node, coef in coeffs.items()
                    if zone in problem.candidate_zones(school_node)
                )
                cuts.append(
                    ChoiceCut(node=node, zone=zone, constant=constant, terms=terms)
                )
        return cuts


def _student_area_column(problem: ZoneProblem) -> str:
    unit = problem.level.unit
    if unit == "BlockGroup":
        return "census_blockgroup"
    if unit == "Block":
        return "census_block"
    if unit == "attendance_area":
        return "attendance_area"
    return f"census_{unit.lower()}"


def _normalize_studentno(values: pd.Series) -> pd.Series:
    as_text = values.astype(str).str.split("-").str[-1]
    return pd.to_numeric(as_text, errors="coerce").astype("Int64")


def _area_key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _school_key(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _node_area_ids(attrs: dict[str, Any]) -> list[str]:
    if "block_ids" in attrs:
        return [key for key in (_area_key(v) for v in attrs["block_ids"]) if key]
    if "area_id" in attrs:
        key = _area_key(attrs["area_id"])
        return [key] if key else []
    return []


def _school_to_node(problem: ZoneProblem) -> dict[str, int]:
    out: dict[str, int] = {}
    for node, attrs in problem.G.nodes(data=True):
        for sid in attrs.get("school_ids", []):
            out[_school_key(sid)] = node
    return out


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis)
    finite_max = np.isfinite(max_values)
    safe_max = np.where(finite_max, max_values, 0.0)
    expanded = np.expand_dims(safe_max, axis=axis)
    summed = np.sum(np.exp(values - expanded), axis=axis)
    safe_summed = np.where(finite_max & (summed > 0), summed, 1.0)
    return np.where(finite_max, safe_max + np.log(safe_summed), -np.inf)


def _log1pexp(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    positive = values > 0
    out[positive] = values[positive] + np.log1p(np.exp(-values[positive]))
    out[~positive] = np.log1p(np.exp(values[~positive]))
    return out


def _finite_array(values: np.ndarray, replacement: float) -> np.ndarray:
    return np.nan_to_num(
        values,
        nan=replacement,
        posinf=replacement if replacement > 0 else abs(replacement),
        neginf=replacement,
    )


def finite_or(value: float, replacement: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else float(replacement)
