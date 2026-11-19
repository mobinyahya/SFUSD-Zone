"""The extended assignment report (``MatchEvaluator.eval_assignment_full``) on arrays.

The report holds ~2,200 metrics and the matching pipeline evaluates it once
citywide and once per ZIP code and attendance area for every lottery. Built
from DataFrame filters it spent nearly all its time in pandas overhead on
groups of ~50 students, so it is computed here from integer-coded and float
arrays built once per student table (``ReportColumns``) and sliced per group
by position.

Every metric is the value the DataFrame formulation returns. Where a float
reduction feeds a threshold comparison (a school's mean FRL against the
district mean plus 10 points, say), it is computed by the same algorithm
pandas uses, so ties fall the same way:

* ``Series.mean`` of float64 is ``nanops.nanmean`` (NaNs to zero, numpy
  pairwise sum, divided by the count); of a masked ``Float64`` column it is
  ``np.mean(values, where=~mask)``;
* ``groupby(...).mean()`` is pandas' Kahan-compensated ``group_mean``, summed
  in row order (``_kahan_group_means``);
* ``Series.var`` is ``nanops.nanvar``'s two-pass formula.

Sums that Python's ``sum`` folds left to right in the DataFrame formulation
(segregation and dissimilarity) are folded the same way here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

AALPI = ["Black", "Hispanic", "Pacific Islander"]
DIAGNOSTIC_ETHNICITIES = [
    "Asian",
    "Black",
    "Decline to State",
    "Hispanic",
    "Other",
    "Pacific Islander",
    "Two or More Races",
    "White",
]
DIAGNOSTIC_PROGRAM_TYPES = [
    "CB",
    "CE",
    "CN",
    "CT",
    "FB",
    "GE",
    "JE",
    "JN",
    "KE",
    "KN",
    "ME",
    "MN",
    "NC",
    "NS",
    "SB",
    "SE",
    "SN",
]
FULL_REPORT_GROUPS = [
    "All Assigned",
    "Black",
    "Asian",
    "Hispanic",
    "White",
    "Two or More Races",
    "Decline to State",
    "High FRL",
    "Low FRL",
    "CTIP",
    "non-CTIP",
    "ET (2024)",
    "non-ET (2024)",
    "AALPI",
]
_GE_AAPI = ("Black", "Pacific Islander")
_PROXIMITY_ETHNICITIES = ("Black", "Asian", "Hispanic", "White")
_FULL_REPORT_ETHNIC_GROUPS = (
    "Black",
    "Asian",
    "Hispanic",
    "White",
    "Two or More Races",
    "Decline to State",
)
_INCOME_POINTS = (10, 15, -10, -15)
_NAN = float("nan")


def _bool_values(values) -> np.ndarray:
    """A pandas boolean result as a numpy mask, NA counting as False."""
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=bool, na_value=False)
    return np.asarray(values, dtype=bool)


def _float_values(values: pd.Series) -> np.ndarray:
    """``choice_ranks._float_values``: numeric values as float, NaN for NA."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=float, na_value=np.nan)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _codes(values: pd.Series, *, sort: bool = True):
    codes, uniques = pd.factorize(values, sort=sort, use_na_sentinel=True)
    return codes.astype(np.int64, copy=False), list(uniques)


def _code_of(uniques, value) -> int:
    try:
        return uniques.index(value)
    except ValueError:
        return -2


_add = np.add.reduce


def _mean(values: np.ndarray):
    """``Series.mean`` of a float64 (or int) array (``nanops.nanmean``)."""
    size = values.size
    if size == 0:
        return _NAN
    if values.dtype.kind != "f":
        return values.sum(dtype=np.float64) / np.float64(size)
    total = _add(values)
    if total == total:
        return total / np.float64(size)
    mask = np.isnan(values)
    count = size - int(np.count_nonzero(mask))
    if count == 0:
        return _NAN
    return _add(np.where(mask, 0.0, values)) / np.float64(count)


def _masked_mean(values: np.ndarray, mask: np.ndarray | None):
    """``Series.mean`` of a masked (``Float64``) column; ``pd.NA`` when empty."""
    if mask is None:
        return _mean(values)
    if not values.size or mask.all():
        return pd.NA
    with np.errstate(all="ignore"):
        return np.mean(values, where=~mask)


def _median(values: np.ndarray):
    """``Series.median``: the middle value, or the mean of the middle two."""
    values = np.sort(values)
    size = values.size
    if size and values[-1] != values[-1]:
        values = values[~np.isnan(values)]
        size = values.size
    if not size:
        return _NAN
    middle = size // 2
    if size % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _var(values: np.ndarray):
    """``Series.var`` (ddof 1) of a float64 array (``nanops.nanvar``)."""
    size = values.size
    if size == 0:
        return _NAN
    total = _add(values)
    if total == total:
        if size <= 1:
            return _NAN
        avg = total / np.float64(size)
        return _add((avg - values) ** 2) / np.float64(size - 1)
    mask = np.isnan(values)
    count = size - int(mask.sum())
    if count <= 1:
        return _NAN
    values = np.where(mask, 0.0, values)
    avg = _add(values) / np.float64(count)
    sqr = (avg - values) ** 2
    sqr[mask] = 0
    return _add(sqr) / np.float64(count - 1)


def _bool_mean(mask: np.ndarray):
    """``Series.mean`` of a boolean Series (and ``_array_mean``)."""
    return mask.sum() / mask.size if mask.size else _NAN


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else _NAN


def _kahan_group_means(labels: np.ndarray, values: np.ndarray, groups: int):
    """pandas' ``group_mean``: Kahan-compensated sums in row order."""
    sums = [0.0] * groups
    compensation = [0.0] * groups
    counts = [0] * groups
    for label, value in zip(labels.tolist(), values.tolist()):
        if value != value:
            continue
        counts[label] += 1
        y = value - compensation[label]
        t = sums[label] + y
        c = t - sums[label] - y
        compensation[label] = 0.0 if c != c else c
        sums[label] = t
    return np.array(
        [s / n if n else _NAN for s, n in zip(sums, counts)], dtype=np.float64
    )


@dataclass
class _Groups:
    """Dense codes of one categorical column within one student subset."""

    keys: np.ndarray  # sorted table-level codes present in the subset
    local: np.ndarray  # each subset row's position in ``keys``
    sizes: np.ndarray

    @classmethod
    def of(cls, codes: np.ndarray):
        keys, local, sizes = np.unique(codes, return_inverse=True, return_counts=True)
        return cls(keys, local.reshape(-1), sizes)

    def count(self, mask: np.ndarray) -> np.ndarray:
        return np.bincount(self.local[mask], minlength=self.keys.size)


class ReportColumns:
    """One student table as the arrays the full report reads.

    Predicates are evaluated with pandas over the whole table once, with the
    DataFrame formulation's own comparisons, and then sliced by position.
    """

    def __init__(self, evaluator, student_data: pd.DataFrame):
        sd = student_data
        self.size = len(sd)
        medium_income = evaluator.medium_income
        low_income = evaluator.low_income

        programno = sd["programno"]
        self.assigned = _bool_values(programno > 0)
        self.unassigned = _bool_values(programno == 0)
        # cumulative_choice_rates reads these columns through float arrays.
        programno_values = _float_values(programno)
        designation_values = _float_values(sd["designation"])
        self.eligible = (programno_values > 0) & (designation_values == 0)
        self.rank = _float_values(sd["rank"])
        self.mechanism_rank = _float_values(sd["mechanism_rank"])

        designation = sd["designation"]
        self.designation_sum_values = designation.to_numpy()
        self.designation_ne0 = _bool_values(designation != 0)
        self.designation_eq0 = _bool_values(designation == 0)
        self.designation_eq1 = _bool_values(designation == 1)
        raw_designation = designation.to_numpy()
        self.raw_designation_eq1 = np.asarray(raw_designation == 1, dtype=bool)
        self.raw_designation_eq0 = np.asarray(raw_designation == 0, dtype=bool)

        self.ethnicity, ethnicities = _codes(sd["ethnicity"])
        self.ethnicity_count = len(ethnicities)
        self.ethnicity_code = {name: _code_of(ethnicities, name) for name in AALPI}
        for name in (*_FULL_REPORT_ETHNIC_GROUPS, *DIAGNOSTIC_ETHNICITIES):
            self.ethnicity_code.setdefault(name, _code_of(ethnicities, name))
        self.aalpi = np.isin(
            self.ethnicity,
            [self.ethnicity_code[name] for name in AALPI],
        )
        race = sd["ethnicity"]
        self.valid_race = _bool_values(
            race.notna() & race.astype("string").str.strip().ne("")
        )

        frl = sd["frl"]
        # A masked (``Float64``) column reduces through its own mean.
        self.frl_mask = (
            frl.isna().to_numpy()
            if isinstance(frl.array, pd.arrays.FloatingArray | pd.arrays.IntegerArray)
            else None
        )
        self.frl = frl.to_numpy(dtype=np.float64, na_value=np.nan)
        self.frl_high = _bool_values(frl > 0.5)
        self.frl_low = _bool_values(frl <= 0.5)
        dissimilarity_frl = pd.to_numeric(frl, errors="coerce").fillna(0.0)
        self.dissimilarity_frl = dissimilarity_frl.to_numpy(
            dtype=np.float64, na_value=0.0
        )
        self.is_frl = sd["is_frl"].to_numpy(dtype=np.float64)

        self.ctip = _bool_values(sd["ctip1"] == 1)
        self.non_ctip = _bool_values(sd["ctip1"] == 0)
        self.et = _bool_values(sd["et2"] == 1)
        self.non_et = _bool_values(sd["et2"] == 0)

        income = pd.to_numeric(sd["median_hh_income"], errors="coerce")
        self.income_valid = _bool_values(income.notna())
        self.income_high = _bool_values(income >= medium_income)
        self.income_low = _bool_values(income <= low_income)
        self.income_below_medium = _bool_values(income <= medium_income)
        self.income_above_medium = _bool_values(income > medium_income)

        self.school, _ = _codes(sd["assigned school"])
        self.assignment, _ = _codes(sd["assignment"])
        self.ge = _bool_values(sd["programtype"] == "GE")
        self.programtype, programtypes = _codes(sd["programtype"])
        self.program_group_labels = tuple(
            label for label, _ in evaluator._full_report_program_groups()
        )
        # Per program-type group: is each student's program of that kind.
        self.program_kinds = [
            np.isin(
                self.programtype,
                [i for i, name in enumerate(programtypes) if name in kinds],
            )
            for _, kinds in evaluator._full_report_program_groups()
        ]
        self.distance = _float_values(sd["assignment_dist"])
        self.overage_seat = sd["overage_seat"].to_numpy()

        program_codes = sd["programcodes"].to_numpy()
        ge_position = {pid: i for i, pid in enumerate(evaluator._ge_program_ids)}
        self.ge_program = np.fromiter(
            (ge_position.get(code, -1) for code in program_codes),
            dtype=np.int64,
            count=self.size,
        )
        program_ids = evaluator.programs["program_id"].astype(str).to_numpy()
        program_position = {pid: i for i, pid in enumerate(program_ids)}
        self.program = np.fromiter(
            (program_position.get(code, -1) for code in program_codes),
            dtype=np.int64,
            count=self.size,
        )

        self.utility = None
        if "assigned_utility" in sd:
            self.utility = pd.to_numeric(
                sd["assigned_utility"], errors="coerce"
            ).to_numpy(dtype=np.float64, na_value=np.nan)
            self.utility_assigned = _bool_values(
                pd.to_numeric(sd["programno"], errors="coerce").fillna(0) > 0
            )
        self.outside_utility = None
        if "outside_utility" in sd and "assigned_utility" in sd:
            self.outside_utility = pd.to_numeric(
                sd["outside_utility"], errors="coerce"
            ).to_numpy(dtype=np.float64, na_value=np.nan)
            self.utility_designated = _bool_values(
                pd.to_numeric(sd["designation"], errors="coerce").eq(1)
            )


class _ProgramTable:
    """Lottery-invariant program arrays the diagnostic metrics read."""

    def __init__(self, evaluator):
        programs = evaluator.programs
        self.size = len(programs)
        self.ge_capacities = evaluator._ge_program_capacities
        self.ge_program_count = int((programs["program_type"] == "GE").sum())
        self.has_capacity = "capacity" in programs.columns and not programs.empty
        if self.has_capacity:
            self.capacities = programs["capacity"].to_numpy(dtype=float)
            self.positive_capacity = self.capacities > 0
            self.capacity_total = self.capacities.sum()
            type_col = "program_type" if "program_type" in programs.columns else "type"
            self.type_masks = []
            if type_col in programs.columns:
                types = programs[type_col].to_numpy()
                for program_type in np.unique(types):
                    mask = types == program_type
                    self.type_masks.append(
                        (program_type, mask, self.capacities[mask].sum())
                    )


def _record(metrics, name, numerator, denominator):
    metrics[name] = numerator / denominator if denominator else _NAN
    metrics[f"{name} numerator"] = numerator
    metrics[f"{name} denominator"] = denominator


def full_report_metrics(evaluator, columns: ReportColumns, rows: np.ndarray) -> dict:
    """The full report of the students at positions ``rows`` of ``columns``.

    ``rows`` must be ascending: the DataFrame formulation reduces each group
    in table order.
    """
    c = columns
    program_table = evaluator._full_report_program_table()
    low_income = evaluator.low_income
    medium_income = evaluator.medium_income
    eth_code = c.ethnicity_code
    metrics = {}

    S = rows
    n_students = S.size
    A = S[c.assigned[S]]
    n_assigned = A.size
    a_designation_ne0 = c.designation_ne0[A]
    D = A[a_designation_ne0]
    ND = A[c.designation_eq0[A]]

    eth_S = c.ethnicity[S]
    eth_A = c.ethnicity[A]
    aalpi_A = c.aalpi[A]

    # Schools of the assigned students (``school_stats``).
    schools = _Groups.of(c.school[A])
    enrollment = schools.sizes
    frl_A = c.frl[A]
    school_frl = _kahan_group_means(schools.local, frl_A, schools.keys.size)
    nd_schools = _Groups.of(c.school[ND])
    nd_school_frl = _kahan_group_means(
        nd_schools.local, c.frl[ND], nd_schools.keys.size
    )
    district_frl = _masked_mean(frl_A, None if c.frl_mask is None else c.frl_mask[A])
    student_school = schools.local  # each assigned student's school position

    # Student groups, as masks over A.
    groups = {
        "All Assigned": np.ones(n_assigned, dtype=bool),
        "Black": eth_A == eth_code["Black"],
        "Asian": eth_A == eth_code["Asian"],
        "Hispanic": eth_A == eth_code["Hispanic"],
        "White": eth_A == eth_code["White"],
        "Two or More Races": eth_A == eth_code["Two or More Races"],
        "Decline to State": eth_A == eth_code["Decline to State"],
        "High FRL": c.frl_high[A],
        "Low FRL": c.frl_low[A],
        "CTIP": c.ctip[A],
        "non-CTIP": c.non_ctip[A],
        "ET (2024)": c.et[A],
        "non-ET (2024)": c.non_et[A],
        "AALPI": aalpi_A,
        f"Income below {medium_income}": c.income_below_medium[A],
        f"Income above {medium_income}": c.income_above_medium[A],
    }

    metrics["Total AALPI students"] = int(c.aalpi[S].sum())
    metrics["Total AALPI assigned students"] = int(aalpi_A.sum())

    # "# Racial majority schools"
    if n_assigned == 0:
        metrics["# Racial majority schools"] = 0
    else:
        valid = c.valid_race[A]
        if not valid.any():
            metrics["# Racial majority schools"] = 0
        else:
            k = max(c.ethnicity_count + 1, 1)
            pairs = student_school[valid] * k + (eth_A[valid] + 1)
            pair_keys, pair_counts = np.unique(pairs, return_counts=True)
            largest = np.zeros(schools.keys.size, dtype=np.int64)
            np.maximum.at(largest, pair_keys // k, pair_counts)
            present = np.zeros(schools.keys.size, dtype=bool)
            present[pair_keys // k] = True
            share = largest[present] / enrollment[present]
            metrics["# Racial majority schools"] = int((share > 0.5).sum())

    # "Dissimilarity (FRL)"
    if n_assigned == 0:
        metrics["Dissimilarity (FRL)"] = _NAN
    else:
        frl_sum = _kahan_group_sums(
            student_school, c.dissimilarity_frl[A], schools.keys.size
        )
        total_frl = frl_sum.sum()
        other = enrollment.sum() - total_frl
        if total_frl <= 0 or other <= 0:
            metrics["Dissimilarity (FRL)"] = 0.0
        else:
            metrics["Dissimilarity (FRL)"] = float(
                50.0
                * np.abs(frl_sum / total_frl - (enrollment - frl_sum) / other).sum()
            )

    # _full_proximity_metrics
    designated_S = c.designation_eq1[S]
    non_designated_S = c.designation_eq0[S]
    distance_A = c.distance[A]
    metrics["Total Designated"] = int(designated_S.sum())
    metrics["Not designated"] = int(non_designated_S.sum())
    metrics["Tot Nb Students (Round 1)"] = n_students
    metrics["Tot Nb Assigned (Round 1)"] = n_assigned
    metrics["Tot Nb Designated (Round 1)"] = int(c.designation_ne0[S].sum())
    metrics["Distance Av (All Assigned)"] = _mean(distance_A)
    metrics["Distance Median (All Assigned)"] = _median(distance_A)
    metrics["Distance < 0.5 (All Assigned)"] = _bool_mean(distance_A < 0.5)
    metrics["Distance < 1 (All Assigned)"] = _bool_mean(distance_A < 1)
    metrics["Distance > 3 (All Assigned)"] = _bool_mean(distance_A > 3)
    for group in _PROXIMITY_ETHNICITIES:
        is_group = eth_S == eth_code[group]
        metrics[f" Non-designated {group} students"] = int(
            (is_group & non_designated_S).sum()
        )
        metrics[f" Designated {group} students"] = int((is_group & designated_S).sum())

    # GE programs' Black and Pacific Islander counts.
    aapi_codes = [eth_code[name] for name in _GE_AAPI]
    aapi = c.assigned[S] & np.isin(eth_S, aapi_codes)
    ge_positions = c.ge_program[S][aapi]
    ge_aapi_counts = np.bincount(
        ge_positions[ge_positions >= 0], minlength=len(evaluator._ge_program_ids)
    )
    ge_range = evaluator._ge_aapi_range_from_counts
    capacities = program_table.ge_capacities
    metrics[
        "#GE programs that have more than 0% and less than 10% of their capacity as African American or Pacific Islander students"
    ] = ge_range(ge_aapi_counts, capacities, 0.1, 0.0001, True, True)
    metrics[
        "#GE programs that have exactly 0 African American or Pacific Islander students"
    ] = ge_range(ge_aapi_counts, capacities, 1, 0, True, False)
    metrics[
        "#GE programs that have 1-4 African American or Pacific Islander students"
    ] = ge_range(ge_aapi_counts, capacities, 4, 1, False, False)

    nd_student_school = nd_schools.local

    def frl_range(pct, means, student_positions):
        if means.size == 0:
            return 0, _NAN, 0
        if pct >= 0:
            matching = means >= district_frl + pct
        else:
            matching = means <= district_frl + pct
        return (
            int(matching.sum()),
            matching.sum() / matching.size,
            int(matching[student_positions].sum()),
        )

    for x in (10, 15, -10, -15):
        word = "above" if x >= 0 else "below"
        (
            metrics[f"#Schools {word} {x}% district FRL"],
            metrics[f"Schools {word} {x}% district FRL"],
            metrics[f"#Students in schools {word} {x}% district FRL"],
        ) = frl_range(x / 100.0, school_frl, student_school)
        (
            metrics[f"#Schools {word} {x}% district FRL (Non-Designated)"],
            metrics[f"Schools {word} {x}% district FRL (Non-Designated)"],
            metrics[f"#Students in schools {word} {x}% district FRL  (Non-Designated)"],
        ) = frl_range(x / 100.0, nd_school_frl, nd_student_school)

    aalpi_school_frl = school_frl[student_school[aalpi_A]]
    n_aalpi = aalpi_school_frl.size
    for name, threshold in (
        ("AALPI in school with +10% FRL", 0.1),
        ("AALPI in school with +15% FRL", 0.15),
        ("AALPI in school with -10% FRL", -0.1),
        ("AALPI in school with -15% FRL", -0.15),
    ):
        if not n_aalpi:
            metrics[name] = _NAN
            continue
        if threshold >= 0:
            count = int((aalpi_school_frl > district_frl + threshold).sum())
        else:
            count = int((aalpi_school_frl < district_frl + threshold).sum())
        metrics[name] = count / n_aalpi

    # GE programs.
    ge_in_A = c.ge[A]
    GE = A[ge_in_A]
    ge_programs = _Groups.of(c.assignment[GE])
    ge_frl = _kahan_group_means(ge_programs.local, c.frl[GE], ge_programs.keys.size)
    ge_nd_in_GE = c.designation_eq0[GE]
    GE_ND = GE[ge_nd_in_GE]
    ge_nd_programs = _Groups.of(c.assignment[GE_ND])
    ge_nd_frl = _kahan_group_means(
        ge_nd_programs.local, c.frl[GE_ND], ge_nd_programs.keys.size
    )

    def ge_frl_range(n_ge, means, threshold):
        if n_ge == 0:
            return 0, _NAN
        if threshold >= 0:
            matching = means >= district_frl + threshold
        else:
            matching = means <= district_frl + threshold
        return int(matching.sum()), _bool_mean(matching)

    (
        metrics["#GE programs above +10% district FRL"],
        metrics["Proportion of GE programs above +10% district FRL"],
    ) = ge_frl_range(GE.size, ge_frl, 0.1)
    (
        metrics["#GE programs above +15% district FRL"],
        metrics["Proportion of GE programs above +15% district FRL"],
    ) = ge_frl_range(GE.size, ge_frl, 0.15)
    metrics["#GE programs above +15% district FRL (Non-Designated)"] = ge_frl_range(
        GE_ND.size, ge_nd_frl, 0.15
    )[0]
    (
        metrics["#GE programs below -10% district FRL"],
        metrics["Proportion of GE programs below -10% district FRL"],
    ) = ge_frl_range(GE.size, ge_frl, -0.1)
    (
        metrics["#GE programs below -15% district FRL"],
        metrics["Proportion of GE programs below -15% district FRL"],
    ) = ge_frl_range(GE.size, ge_frl, -0.15)

    ge_aalpi = c.aalpi[GE]
    ge_aalpi_program = ge_programs.local[ge_aalpi]
    ge_aalpi_frl = ge_frl[ge_aalpi_program]
    n_ge_aalpi = ge_aalpi_program.size
    for name, threshold in (
        ("AALPI in GE programs with +10% FRL", 0.1),
        ("AALPI in GE programs with +15% FRL", 0.15),
        ("AALPI in GE programs with -10% FRL", -0.1),
        ("AALPI in GE programs with -15% FRL", -0.15),
    ):
        if not n_ge_aalpi:
            metrics[name] = _NAN
        elif threshold >= 0:
            metrics[name] = _bool_mean(ge_aalpi_frl > district_frl + threshold)
        else:
            metrics[name] = _bool_mean(ge_aalpi_frl < district_frl + threshold)

    n_designated = D.size
    if n_designated == 0:
        metrics["AALPI in designated"] = _NAN
    else:
        metrics["AALPI in designated"] = int(c.aalpi[D].sum()) / n_designated

    # Income composition of schools (``school_stats`` income columns).
    income_valid_A = c.income_valid[A]
    valid_by_school = schools.count(income_valid_A)
    high_by_school = schools.count(c.income_high[A])
    low_by_school = schools.count(c.income_low[A])
    income_schools = valid_by_school > 0
    valid_income_count = valid_by_school[income_schools].sum()
    high_shares = high_by_school[income_schools] / valid_by_school[income_schools]
    low_shares = low_by_school[income_schools] / valid_by_school[income_schools]
    district_high_share = _safe_ratio(
        high_by_school[income_schools].sum(), valid_income_count
    )
    district_low_share = _safe_ratio(
        low_by_school[income_schools].sum(), valid_income_count
    )
    income_school_positions = np.flatnonzero(income_schools)

    def outside_range(shares, district_share, prop_threshold, positions):
        """Positions (in ``positions``' frame) of the selected entries."""
        if shares.size == 0 or pd.isna(district_share):
            return positions[:0]
        cutoff = district_share + prop_threshold
        matches = shares >= cutoff if prop_threshold > 0 else shares < cutoff
        return positions[matches]

    def selected_mask(selected, size):
        mask = np.zeros(size, dtype=bool)
        mask[selected] = True
        return mask

    ge_income_valid = c.income_valid[GE]
    ge_valid_count = ge_programs.count(ge_income_valid)
    ge_high_count = ge_programs.count(ge_income_valid & c.income_high[GE])
    ge_income_programs = ge_valid_count > 0
    ge_high_shares = (
        ge_high_count[ge_income_programs] / ge_valid_count[ge_income_programs]
    )
    ge_income_positions = np.flatnonzero(ge_income_programs)
    ge_program_count = program_table.ge_program_count

    high_school_sets = {}
    for points in _INCOME_POINTS:
        direction = "above" if points > 0 else "below"
        threshold_name = (
            f"{direction} {points:+d}% district High Income ({medium_income})"
        )
        selected = outside_range(
            high_shares, district_high_share, points / 100, income_school_positions
        )
        school_mask = selected_mask(selected, schools.keys.size)
        high_school_sets[points] = school_mask
        metrics[f"#Schools {threshold_name}"] = selected.size
        metrics[f"#Students in schools {threshold_name}"] = int(
            school_mask[student_school].sum()
        )
        selected_programs = outside_range(
            ge_high_shares, district_high_share, points / 100, ge_income_positions
        )
        metrics[f"#GE programs {threshold_name}"] = selected_programs.size
        metrics[f"Proportion of GE programs {threshold_name}"] = _safe_ratio(
            selected_programs.size, ge_program_count
        )
        program_mask = selected_mask(selected_programs, ge_programs.keys.size)
        metrics[
            f"AALPI in GE programs with {points:+d}% High Income ({medium_income})"
        ] = _safe_ratio(int(program_mask[ge_aalpi_program].sum()), n_ge_aalpi)

    income_report_groups = dict(groups)
    for ethnicity in DIAGNOSTIC_ETHNICITIES:
        income_report_groups.setdefault(ethnicity, eth_A == eth_code[ethnicity])
    for group, mask in income_report_groups.items():
        group_schools = student_school[mask]
        group_size = group_schools.size
        for points, school_mask in high_school_sets.items():
            direction = "above" if points > 0 else "below"
            label = f"{direction} {points:+d}% district High Income ({medium_income})"
            count = int(school_mask[group_schools].sum())
            metrics[f"#Students in schools {label} ({group})"] = count
            metrics[f"Prop students in schools {label} ({group})"] = _safe_ratio(
                count, group_size
            )

    for points in (10, 15, -10, -15):
        selected = outside_range(
            high_shares, district_high_share, points / 100, income_school_positions
        )
        label = f"{points:+d}% High Income ({medium_income})"
        metrics[f"#Schools with {label}"] = selected.size
        metrics[f"Prop schools with {label}"] = _safe_ratio(
            selected.size, high_shares.size
        )

    aalpi_by_school = schools.count(aalpi_A)

    def selected_students(composition_shares, district_share, prop, positions):
        selected = outside_range(composition_shares, district_share, prop, positions)
        return selected, selected_mask(selected, schools.keys.size)

    for prop, label in ((0.1, "+10%"), (0.15, "+15%"), (-0.1, "-10%"), (-0.15, "-15%")):
        _, school_mask = selected_students(
            high_shares, district_high_share, prop, income_school_positions
        )
        in_selected = school_mask[student_school]
        metrics[f"Prop AALPI in {label} High Income Schools ({medium_income})"] = (
            _safe_ratio(int(aalpi_A[in_selected].sum()), int(in_selected.sum()))
        )
    for prop, label in ((0.1, "+10%"), (0.15, "+15%"), (-0.1, "-10%"), (-0.15, "-15%")):
        selected, school_mask = selected_students(
            high_shares, district_high_share, prop, income_school_positions
        )
        if not school_mask[student_school].any():
            value = _NAN
        else:
            value = _mean(aalpi_by_school[selected] / enrollment[selected])
        metrics[f"Avg Prop AALPI in {label} High Income Schools ({medium_income})"] = (
            value
        )
    for prop, label in ((-0.10, "-10%"), (-0.15, "-15%")):
        _, school_mask = selected_students(
            low_shares, district_low_share, prop, income_school_positions
        )
        in_selected = school_mask[student_school]
        metrics[f"Prop AALPI in {label} Low Income Schools ({low_income})"] = (
            _safe_ratio(int(aalpi_A[in_selected].sum()), int(in_selected.sum()))
        )
    is_frl_A = c.is_frl[A]
    for prop, label in ((0.1, "+10%"), (0.15, "+15%")):
        selected, school_mask = selected_students(
            high_shares, district_high_share, prop, income_school_positions
        )
        in_selected = school_mask[student_school]
        if not in_selected.any():
            value = _NAN
        else:
            means = _kahan_group_means(
                student_school[in_selected],
                is_frl_A[in_selected],
                schools.keys.size,
            )[selected]
            value = _mean(means)
        metrics[f"Avg Prop FRL in {label} High Income Schools ({medium_income})"] = (
            value
        )

    # CHOICE METRICS
    metrics.update(_utility_metrics(c, S))
    n_unassigned = int(c.unassigned[S].sum())
    metrics["#Unassigned"] = n_unassigned
    metrics["Unassigned"] = _safe_ratio(n_unassigned, n_students)
    designation_A = c.designation_sum_values[A]
    metrics["#Designated"] = designation_A.sum()
    metrics["Designated"] = _mean(designation_A)
    overage_count = c.overage_seat[A].sum()
    metrics["num_overage_seats"] = int(overage_count)
    metrics["overage"] = (
        _safe_ratio(overage_count, n_students) if evaluator.overscribe_aa else 0.0
    )

    eligible = c.eligible
    rank = c.rank
    mechanism_rank = c.mechanism_rank
    for student_type, positions in (("All Assigned", A), ("Non-Designated", ND)):
        size = positions.size
        group_eligible = eligible[positions]
        group_rank = rank[positions]
        group_mechanism = mechanism_rank[positions]
        for top in range(1, 4):
            _record(
                metrics,
                f"Prop Top {top} choice ({student_type})",
                int(np.count_nonzero(group_eligible & (group_rank <= top))),
                size,
            )
        metrics[f"Mean Choice ({student_type})"] = _mean(group_rank)
        metrics[f"Median Choice ({student_type})"] = _median(group_rank)
        for top in range(1, 4):
            _record(
                metrics,
                f"Top {top} in-zone choice ({student_type})",
                int(np.count_nonzero(group_eligible & (group_mechanism <= top))),
                size,
            )
        group_distance = c.distance[positions]
        metrics[f"Distance Av ({student_type})"] = _mean(group_distance)
        metrics[f"Distance Median ({student_type})"] = _median(group_distance)

    proportion_counts = {
        "AALPI": aalpi_by_school,
        "High FRL": schools.count(groups["High FRL"]),
        "Low FRL": schools.count(groups["Low FRL"]),
    }
    white = groups["White"]
    for ethnicity in ("Black", "Hispanic"):
        ethnic = groups[ethnicity]
        for label, proportion in (
            ("AALPI", "AALPI"),
            ("poverty", "High FRL"),
            ("low FRL", "Low FRL"),
        ):
            metrics[f"{ethnicity}/White exposure to {label}"] = _segregation(
                schools, ethnic, white, proportion_counts[proportion], enrollment
            )

    exposures = {
        name: counts / enrollment for name, counts in proportion_counts.items()
    }
    exposure_groups = ("AALPI", "White", "Black", "Hispanic", "High FRL", "Low FRL")
    for group_name in exposure_groups:
        group_schools = student_school[groups[group_name]]
        for label, proportion in (
            ("AALPI", "AALPI"),
            ("high FRL", "High FRL"),
            ("low FRL", "Low FRL"),
        ):
            metrics[f"{group_name} exposure to {label}"] = (
                np.mean(exposures[proportion][group_schools])
                if group_schools.size
                else _NAN
            )

    # Schools above the district FRL mean.
    if school_frl.size:
        school_flags = (
            school_frl > district_frl + 0.1,
            school_frl > district_frl + 0.15,
            school_frl > district_frl,
        )
    else:
        school_flags = (school_frl, school_frl, school_frl)
    eligible_S = eligible[S]
    rank_S = rank[S]
    for top in range(1, 4):
        _record(
            metrics,
            f"Prop Top {top} choice (All Students)",
            int(np.count_nonzero(eligible_S & (rank_S <= top))),
            n_students,
        )

    report_groups = {
        group: groups[group]
        for group in FULL_REPORT_GROUPS
        + [f"Income below {medium_income}", f"Income above {medium_income}"]
    }
    metrics.update(
        _group_metrics(
            c,
            A,
            report_groups,
            student_school,
            school_frl,
            district_frl,
            school_flags,
            exposures["High FRL"],
            schools,
            enrollment,
            distance_A,
            _report_names(tuple(report_groups), c.program_group_labels),
        )
    )
    metrics["Theil H"] = _theil(schools, eth_A, enrollment)
    metrics.update(
        _diagnostic_metrics(
            c, S, eth_S, program_table, DIAGNOSTIC_ETHNICITIES, DIAGNOSTIC_PROGRAM_TYPES
        )
    )
    return metrics


def _kahan_group_sums(labels, values, groups):
    sums = [0.0] * groups
    compensation = [0.0] * groups
    for label, value in zip(labels.tolist(), values.tolist()):
        if value != value:
            continue
        y = value - compensation[label]
        t = sums[label] + y
        c = t - sums[label] - y
        compensation[label] = 0.0 if c != c else c
        sums[label] = t
    return np.array(sums, dtype=np.float64)


def _utility_metrics(c: ReportColumns, S):
    name = "Designated below outside option"
    if c.outside_utility is None:
        below = {
            name: _NAN,
            f"{name} numerator": _NAN,
            f"{name} denominator": _NAN,
        }
    else:
        utility = c.utility[S]
        below_mask = (
            c.utility_designated[S]
            & c.utility_assigned[S]
            & ~(utility >= c.outside_utility[S])
        )
        count = int(below_mask.sum())
        total = S.size
        below = {
            name: count / total if total else _NAN,
            f"{name} numerator": count,
            f"{name} denominator": total,
        }
    if c.utility is None:
        return {"Total Utility": _NAN, "Average Utility": _NAN, **below}
    utilities = np.where(c.utility_assigned[S], c.utility[S], 0.0)
    total_utility = utilities.sum()
    average = total_utility / S.size if S.size else _NAN
    return {"Total Utility": total_utility, "Average Utility": average, **below}


def _segregation(schools: _Groups, group_a, group_b, proportion, enrollment):
    a_counts = schools.count(group_a)
    b_counts = schools.count(group_b)
    a_sum = int(group_a.sum())
    b_sum = int(group_b.sum())
    if a_sum == 0 or b_sum == 0:
        return _NAN
    terms = (a_counts / a_sum - b_counts / b_sum) * (proportion / enrollment)
    # Python floats: the DataFrame formulation sums Python floats, which
    # sum() adds with compensation (Python 3.12+).
    return sum(terms.tolist())


def _dissimilarity(schools: _Groups, group_schools, size, enrollment):
    total = int(enrollment.sum())
    if size == 0 or total == 0:
        return _NAN
    ratio = size / total
    counts = np.bincount(group_schools, minlength=schools.keys.size)
    terms = np.abs(counts - enrollment * ratio) / 2
    return sum(terms) / size


def _program_assignment_counts(
    metrics, group, programtypes, designated, non_designated, program_groups
):
    size = int(programtypes.max()) + 1 if programtypes.size else 0
    assigned = np.bincount(programtypes[programtypes >= 0], minlength=size)
    designated_counts = np.bincount(
        programtypes[designated & (programtypes >= 0)], minlength=size
    )
    non_designated_counts = np.bincount(
        programtypes[non_designated & (programtypes >= 0)], minlength=size
    )
    for label, codes in program_groups:
        codes = codes[codes < size]
        metrics[f"Nb assigned students ({group}) to {label}"] = int(
            assigned[codes].sum()
        )
        metrics[f"Nb designated students ({group}) to {label}"] = int(
            designated_counts[codes].sum()
        )
        if label != "LP program":
            metrics[f"Nb non-designated students ({group}) to {label}"] = int(
                non_designated_counts[codes].sum()
            )


def _theil(schools: _Groups, eth_A, enrollment):
    if eth_A.size == 0:
        return 0.0
    eth_keys, eth_local, ethnic_total = np.unique(
        eth_A, return_inverse=True, return_counts=True
    )
    eth_local = eth_local.reshape(-1)
    ethnic_matrix = np.zeros((schools.keys.size, eth_keys.size), dtype=int)
    np.add.at(ethnic_matrix, (schools.local, eth_local), 1)
    ethnic_total_norm = ethnic_total / ethnic_total.sum()
    district_entropy = np.sum(ethnic_total_norm * np.log(1.0 / ethnic_total_norm))
    if district_entropy == 0:
        return 0.0
    school_totals = ethnic_matrix.sum(axis=1)
    school_props = ethnic_matrix / school_totals[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(
            school_props > 0,
            school_props * np.log(1.0 / school_props),
            0,
        )
    school_entropy = entropy_terms.sum(axis=1)
    theil_sum = np.sum(school_totals * (district_entropy - school_entropy))
    return theil_sum / (district_entropy * ethnic_total.sum())


def _diagnostic_metrics(c, S, eth_S, program_table, ethnicities, program_types):
    metrics = {}
    assigned = c.assigned[S]
    n_students = S.size
    assigned_count = int(assigned.sum())
    counts = np.bincount(eth_S[eth_S >= 0], minlength=c.ethnicity_count)
    enrolled = np.bincount(eth_S[assigned & (eth_S >= 0)], minlength=c.ethnicity_count)
    for group in ethnicities:
        code = c.ethnicity_code[group]
        total = int(counts[code]) if code >= 0 else 0
        enrolled_count = int(enrolled[code]) if code >= 0 else 0
        metrics[f"count_students_{group}"] = total
        metrics[f"enrollment_count_{group}"] = enrolled_count
        metrics[f"enrollment_rate_{group}"] = _safe_ratio(enrolled_count, total)
    metrics["count_students_Total"] = n_students
    metrics["enrollment_count_Total"] = assigned_count
    metrics["enrollment_rate_Total"] = _safe_ratio(assigned_count, n_students)

    metrics.update({f"utilization_{kind}": _NAN for kind in program_types})
    metrics["utilization_rate_avg"] = _NAN
    if program_table.size and program_table.has_capacity:
        programs = c.program[S][assigned]
        enrollment = np.bincount(
            programs[programs >= 0], minlength=program_table.size
        ).astype(float)
        if program_table.capacity_total > 0:
            positive = program_table.positive_capacity
            metrics["utilization_rate_avg"] = np.mean(
                enrollment[positive] / program_table.capacities[positive]
            )
        for program_type, mask, capacity in program_table.type_masks:
            if capacity > 0:
                metrics[f"utilization_{program_type}"] = (
                    enrollment[mask].sum() / capacity
                )
    return metrics


_ND_TOP = 3
_TOPS = 10


@lru_cache(maxsize=16)
def _group_metric_names(group, program_labels, all_assigned):
    """The per-group metric names, in the order the report inserts them."""
    names = [
        f"Number of assigned students ({group})",
        f"{group} exposure to high FRL",
        f"{group} exposure to FRL prob",
        f"{group} in school with +10% FRL",
        f"{group} in school with +15% FRL",
        f"{group} in school with > avg FRL",
    ]

    def record(name):
        names.extend([name, f"{name} numerator", f"{name} denominator"])

    for top in range(1, 4):
        record(f"Prop Top {top} choice ({group})")
    names += [
        f"Distance Av ({group})",
        f"Distance Median ({group})",
        f"Dissimilarity ({group})",
    ]
    for label in program_labels:
        names.append(f"Nb assigned students ({group}) to {label}")
        names.append(f"Nb designated students ({group}) to {label}")
        if label != "LP program":
            names.append(f"Nb non-designated students ({group}) to {label}")
    names += [
        f"#Students in schools above +15% district FRL ({group})",
        f"Prop students in schools above +15% district FRL ({group})",
    ]
    if not all_assigned:
        names.append(f"Number of designated students ({group})")
        record(f"Prop Top 1 choice Non-Designated ({group})")
        record(f"Prop Top 2 choice Non-Designated ({group})")
        for top in range(1, _TOPS + 1):
            record(f"Proportion of students in top {top} ({group})")
        record(f"Prop Top 3 choice Non-Designated ({group})")
        for top in range(1, 4):
            record(f"Top {top} in-zone choice Non-Designated ({group})")
        names += [
            f"Mean Choice Non-Designated ({group})",
            f"Median Choice Non-Designated ({group})",
            f"Distance Av Non-Designated ({group})",
            f"Distance Median Non-Designated ({group})",
            f"Distance < 0.5 Non-Designated ({group})",
            f"Distance > 3 Non-Designated ({group})",
        ]
    names += [
        f"Prop Distance > 3 and designated ({group})",
        f"Prop Distance > 3 and Top 3 choice, non-designated ({group})",
        f"Prop Distance > 3 and non-designated ({group})",
        f"Prop Distance > 3 and Rank>=5 ({group})",
        f"Prop Distance > 3 and in-zone Rank>=5 ({group})",
        f"Prop Distance > 3 and Rank>=5 Non-Designated ({group})",
        f"Prop Distance > 3 and in-zone Rank>=5 Non-Designated ({group})",
        f"Prop Distance > 3 and Rank>=4 ({group})",
        f"Prop Distance > 3 and Rank>=4 Non-Designated ({group})",
        f"Prop Distance > 3 and (Rank>=4 or designated) ({group})",
        f"Prop Distance > 3 and (Rank>=5 or designated) ({group})",
        f"Prop Distance > 3 and (in-zone Rank>=5 or designated) ({group})",
        f"Variance of rank ({group})",
        f"Variance of in-zone rank ({group})",
        f"Variance of distance ({group})",
    ]
    return tuple(names)


def _report_names(group_names, program_labels):
    return [
        _group_metric_names(group, program_labels, group == "All Assigned")
        for group in group_names
    ]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else _NAN


def _group_metrics(
    c,
    A,
    groups,
    student_school,
    school_frl,
    district_frl,
    school_flags,
    high_frl_exposure,
    schools,
    enrollment,
    distance_A,
    names,
):
    """The per-group block of the report for every report group at once.

    Every count is one row of ``membership @ predicates``; the float
    reductions run per group on that group's values in table order.
    """
    above_10, above_15, above_avg = school_flags
    n_assigned = A.size
    rank_A = c.rank[A]
    mechanism_A = c.mechanism_rank[A]
    eligible_A = c.eligible[A]
    designated = c.raw_designation_eq1[A]
    non_designated = c.raw_designation_eq0[A]
    has_schools = school_frl.size > 0
    if has_schools:
        at_least_15 = school_frl >= district_frl + 0.15
        flags = np.column_stack((above_10, above_15, above_avg, at_least_15))[
            student_school
        ]
    else:
        flags = np.zeros((n_assigned, 4), dtype=bool)

    over_3 = distance_A > 3
    rank_ge_5 = rank_A >= 5
    rank_ge_4 = rank_A >= 4
    mechanism_ge_5 = mechanism_A >= 5
    columns = [flags]
    columns.append(
        np.column_stack([eligible_A & (rank_A <= top) for top in range(1, _TOPS + 1)])
        if n_assigned
        else np.zeros((0, _TOPS), dtype=bool)
    )
    nd_eligible = eligible_A & non_designated
    predicates = [
        *(nd_eligible & (rank_A <= top) for top in range(1, _ND_TOP + 1)),
        *(nd_eligible & (mechanism_A <= top) for top in range(1, 4)),
        non_designated & (distance_A < 0.5),
        non_designated & over_3,
        over_3 & designated,
        over_3 & (rank_A <= 3) & non_designated,
        over_3 & non_designated,
        over_3 & rank_ge_5,
        over_3 & mechanism_ge_5,
        over_3 & rank_ge_5 & non_designated,
        over_3 & mechanism_ge_5 & non_designated,
        over_3 & rank_ge_4,
        over_3 & rank_ge_4 & non_designated,
        over_3 & (rank_ge_4 | designated),
        over_3 & (rank_ge_5 | designated),
        over_3 & (mechanism_ge_5 | designated),
        designated,
        non_designated,
    ]
    for kind in c.program_kinds:
        of_kind = kind[A]
        predicates += [of_kind, of_kind & designated, of_kind & non_designated]
    columns.append(
        np.column_stack(predicates)
        if n_assigned
        else np.zeros((0, len(predicates)), dtype=bool)
    )
    predicate_matrix = np.hstack(columns).astype(np.int64)
    membership = np.vstack(list(groups.values())).astype(np.int64)
    counts = (membership @ predicate_matrix).tolist()
    sizes = membership.sum(axis=1).tolist()
    program_offset = 4 + _TOPS + 22

    frl_masked = c.frl_mask is not None
    values = {}
    for (group, mask), group_names, row, size in zip(
        groups.items(), names, counts, sizes, strict=True
    ):
        out = [size]
        if size > 0:
            g_school = student_school[mask]
            g_school_frl = school_frl[g_school]
            out.append(_add(high_frl_exposure[g_school]) / size)
            out.append(
                np.mean(g_school_frl, where=np.ones(size, dtype=bool))
                if frl_masked
                else _mean(g_school_frl)
            )
            out += [row[0] / size, row[1] / size, row[2] / size]
        else:
            g_school = student_school[:0]
            out += [_NAN] * 5
        tops = row[4 : 4 + _TOPS]
        for top in range(3):
            out += [_ratio(tops[top], size), tops[top], size]
        g_distance = distance_A[mask]
        if size > 0:
            out += [_mean(g_distance), _median(g_distance)]
        else:
            out += [_NAN, _NAN]
        out.append(_dissimilarity(schools, g_school, size, enrollment))
        for index, label in enumerate(c.program_group_labels):
            base = program_offset + 3 * index
            out += [row[base], row[base + 1]]
            if label != "LP program":
                out.append(row[base + 2])
        above_15_count = row[3] if has_schools else 0
        out += [above_15_count, _safe_ratio(above_15_count, size)]

        p = 4 + _TOPS
        nd_size = row[p + 21]
        g_rank = rank_A[mask]
        g_mechanism = mechanism_A[mask]
        if group != "All Assigned":
            g_non_designated = non_designated[mask]
            nd_rank = g_rank[g_non_designated]
            nd_distance = g_distance[g_non_designated]
            out.append(row[p + 20])
            nd_tops = row[p : p + 3]
            for top in (0, 1):
                out += [_ratio(nd_tops[top], nd_size), nd_tops[top], nd_size]
            for top in range(_TOPS):
                out += [_ratio(tops[top], size), tops[top], size]
            out += [_ratio(nd_tops[2], nd_size), nd_tops[2], nd_size]
            for top in range(3):
                count = row[p + 3 + top]
                out += [_ratio(count, nd_size), count, nd_size]
            out += [
                _mean(nd_rank),
                _median(nd_rank),
                _mean(nd_distance),
                _median(nd_distance),
                _ratio(row[p + 6], nd_size),
                _ratio(row[p + 7], nd_size),
            ]
        out += [
            _ratio(row[p + 8], size),
            _ratio(row[p + 9], size),
            _ratio(row[p + 10], size),
            _ratio(row[p + 11], size),
            _ratio(row[p + 12], size),
            _ratio(row[p + 13], nd_size),
            _ratio(row[p + 14], nd_size),
            _ratio(row[p + 15], size),
            _ratio(row[p + 16], nd_size),
            _ratio(row[p + 17], size),
            _ratio(row[p + 18], size),
            _ratio(row[p + 19], size),
        ]
        if size > 0:
            out += [_var(g_rank), _var(g_mechanism), _var(g_distance)]
        else:
            out += [_NAN, _NAN, _NAN]
        values.update(zip(group_names, out, strict=True))
    return values
