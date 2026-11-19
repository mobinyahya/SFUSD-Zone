#!/usr/bin/env python
"""Convert a district capacity-scenario workbook into a capacity overlay.

The district sends kindergarten capacity proposals as one Excel workbook with
a ``Summary`` sheet and one tab per scenario, and sometimes a single newer tab
on its own, exported from Google Sheets as ``<workbook> - <tab name>.csv``.
Every scenario tab holds a header row such as::

    Scenario | School code | School | Program | Grade | Ideal seats |
    Seat type | School type (this scenario) | Change vs. Baseline | Note

below a few rows of free-text description. This script reads every such tab
of every input (other than the excluded ones) and writes one CSV in the
overlay format ``loaders.tables.apply_capacity_scenario`` reads, with three
columns per scenario:

* ``Scenario_<name>_Capacity`` -- the program's seats (0 where the scenario
  closes or never lists it);
* ``Scenario_<name>_AttendanceAreaCapacity`` -- how many of those seats the
  workbook labels ``Attendance Area`` (the rest are ``Citywide``). A program
  listed once per seat type has its two rows summed;
* ``Scenario_<name>_Category`` -- the school's category in that scenario,
  ``Attendance`` or ``Citywide``, which the loaders apply to the school tables
  when the scenario is selected.

The base ``capacity_scenarios`` registry binds each ``<name>`` to the CSV, so a
run selects one with ``capacity_scenario: <name>`` in either filter group.
Run this again on a newer workbook and register the new names; nothing else
needs to change unless the script stops on something it does not recognise.

Usage
-----

.. code-block:: bash

    RAW=/soalnas/share/data/school_choice/Data/capacity_management/raw
    uv run python analysis/data_prep/convert_capacity_workbook.py \\
        "$RAW/Capacities for Stanford (2026-27).xlsx" \\
        "$RAW/Capacities for Stanford (2026-27) - Ideal Capacity Baseline (Nov 13, 2026).csv" \\
        /soalnas/share/data/school_choice/Data/capacity_management/sfusd_capacities_2627.csv

It prints the scenario names it wrote; add any new ones to
``loaders/configs/base.yaml``.

How the workbook maps onto the program tables
---------------------------------------------

* **Scenario names.** Tab titles are cut at Excel's 31 characters, so each
  workbook tab is matched to the full name in the ``Summary`` sheet's
  ``Scenario`` column; a CSV's name is its file name after `` - ``. The name
  keeps the sheet's date, drops the word "Capacity", and becomes snake case:
  "Ideal Capacity + Bessie GE (10.23.26)" becomes ``ideal_bessie_ge_10_23_26``
  and "Ideal Capacity Baseline (Nov 13, 2026)" ``ideal_baseline_nov_13_2026``.
  The date is what tells two baselines apart. ``--suffix`` appends
  ``_<suffix>``.
* **Grades and seat types.** Only the ``--grade`` rows (default ``K``) are
  read; newer tabs list TK-8. Seats outside the lottery's programs -- ``SDC
  (IEP placement)``, ``Newcomer``, ``TK Feeder``, ``EES`` -- are dropped, and
  the seats dropped are printed.
* **Attendance-area seats.** The ``Seat type`` column, not the program or the
  school, decides which seats are attendance-area: a GE program at an
  attendance-area school may hold citywide seats too. The loaders expose the
  count as ``attendance_area_capacity``, which ``aa_capacity_match`` scores.
* **School category.** A school is an attendance-area school in a scenario
  when any of its programs holds attendance-area seats there, however many of
  its other seats are citywide; otherwise it is citywide. The workbook's own
  ``School type (this scenario)`` column is checked against that rule and any
  disagreement is printed, but the rule wins.
* **Program labels.** A workbook label is one district program; the program
  tables split a language program into its lottery pools (native / non-native
  speakers). ``SP DLI`` becomes ``SE+SN``, and the overlay spreads a
  ``+``-joined row over the pools in proportion to their current seats. Newer
  tabs list such a program once per pool, with nothing naming the pool, so
  rows of a multi-pool label are summed; a single-pool label listed twice per
  seat type stops the conversion. An unknown label stops it too: add it to
  ``PROGRAM_CODES``.
* **School codes.** Codes are the district's. Mission Bay ES is 1731, the
  repository's canonical ID; a workbook that still carries its old typo 731 is
  corrected. A school without a code (``NEW``) needs an entry in
  ``NEW_SCHOOL_CODES``, which assigns it an otherwise unused placeholder;
  without one the conversion stops. Such a school is in no program or school
  table; its overlay seats apply only when a ``planned_schools`` entry in
  ``loaders/configs/base.yaml`` adds it (``include_mandarin_immersion``).
* **Excluded tabs.** Tabs whose title contains any ``--exclude`` text are
  skipped (default ``MPD``: the MPD proposals are not modelled).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from loaders.tables import MISSION_BAY_SCHOOL_ID  # noqa: E402

#: Workbook program label -> the program-table codes it covers. A label with
#: several codes is one program split into lottery pools.
PROGRAM_CODES = {
    "GE": ("GE",),
    "SP BIL": ("SB",),
    "CAN BIL": ("CB",),
    "FIL FLES": ("FB",),
    "SP DLI": ("SE", "SN"),
    "CAN DLI": ("CE", "CN", "CT"),
    "MAN DLI": ("ME", "MN"),
    "JPN BIL": ("JE", "JN"),
    "KOR DLI": ("KE", "KN"),
}

#: District school codes the workbook got wrong -> the right ones.
SCHOOL_CODE_CORRECTIONS = {731: MISSION_BAY_SCHOOL_ID}

#: Schools the district has not issued a code yet -> a placeholder no school,
#: program, or transfer file uses. Replace it once the district issues one.
NEW_SCHOOL_CODES = {"Mandarin Immersion TK-8": 400}

#: Workbook header -> output column. Every one is required.
HEADER = {
    "School code": "SheetSchoolCode",
    "School": "SchoolName",
    "Program": "PathwayName",
    "Grade": "Grade",
    "Ideal seats": "capacity",
    "Seat type": "seat_type",
    "School type": "school_type",
}
#: Headers whose parenthetical varies by tab -> the HEADER name they stand for:
#: "School type (this scenario)" in the workbook, "(baseline)" in a baseline.
HEADER_PREFIXES = ("School type",)
#: The first header cell of a scenario tab.
HEADER_MARKER = "Scenario"

SEAT_TYPES = {"Attendance Area": "Attendance", "Citywide": "Citywide"}
#: Seat types outside the lottery's programs, dropped before conversion.
IGNORED_SEAT_TYPES = ("SDC (IEP placement)", "Newcomer", "TK Feeder", "EES")

KEYS = ["SchNum", "PathwayCode", "Grade"]


def scenario_name(full_name: str, suffix: str | None) -> str:
    """Return the registry name for one workbook scenario."""
    text = re.sub(r"\bcapacity\b", " ", full_name, flags=re.IGNORECASE)
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words:
        raise ValueError(f"Scenario {full_name!r} yields an empty name.")
    return "_".join([*words, suffix] if suffix else words)


def _full_names(path: Path) -> list[str]:
    summary = pd.read_excel(path, sheet_name="Summary", dtype=object)
    if HEADER_MARKER not in summary.columns:
        raise ValueError("The Summary sheet has no 'Scenario' column.")
    return [str(name).strip() for name in summary[HEADER_MARKER].dropna()]


def _match_full_name(title: str, full_names: list[str]) -> str:
    matches = [name for name in full_names if name.startswith(title.strip())]
    if len(matches) != 1:
        raise ValueError(
            f"Tab {title!r} matches {len(matches)} Summary scenarios: {matches}."
        )
    return matches[0]


def _scenario_rows(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Return a tab's rows below its header row, or None if it has none."""
    markers = raw.index[raw[0].astype(str).str.strip().eq(HEADER_MARKER)]
    if markers.empty:
        return None
    header = [str(value).strip() for value in raw.loc[markers[0]]]
    header = [
        next((p for p in HEADER_PREFIXES if name.startswith(f"{p} (")), name)
        for name in header
    ]
    missing = sorted(set(HEADER) - set(header))
    if missing:
        raise ValueError(f"Scenario header is missing columns {missing}.")
    rows = raw.loc[markers[0] + 1 :].set_axis(header, axis=1)
    rows = rows.loc[rows["School"].notna()]
    return rows[list(HEADER)].rename(columns=HEADER)


def _school_codes(rows: pd.DataFrame, title: str) -> pd.Series:
    numeric = pd.to_numeric(rows["SheetSchoolCode"], errors="coerce")
    uncoded = numeric.isna()
    unknown = sorted(set(rows.loc[uncoded, "SchoolName"]) - set(NEW_SCHOOL_CODES))
    if unknown:
        raise ValueError(
            f"{title!r}: schools without a numeric code {unknown}; add them to "
            "NEW_SCHOOL_CODES."
        )
    numeric.loc[uncoded] = rows.loc[uncoded, "SchoolName"].map(NEW_SCHOOL_CODES)
    if not numeric.eq(numeric.round()).all():
        raise ValueError(f"{title!r}: school codes must be whole numbers.")
    return numeric.astype(int).replace(SCHOOL_CODE_CORRECTIONS)


def _selected_rows(
    rows: pd.DataFrame, title: str, grades: tuple[str, ...]
) -> pd.DataFrame:
    """Keep the selected grades' lottery seats, printing the seats dropped."""
    rows = rows.loc[rows["Grade"].astype(str).str.strip().isin(grades)]
    ignored = rows["seat_type"].isin(IGNORED_SEAT_TYPES)
    if ignored.any():
        seats = pd.to_numeric(rows.loc[ignored, "capacity"].replace("-", 0))
        dropped = seats.groupby(rows.loc[ignored, "seat_type"]).sum()
        print(
            f"{title!r}: dropped "
            + ", ".join(f"{int(n)} {kind} seats" for kind, n in dropped.items()),
            file=sys.stderr,
        )
    return rows.loc[~ignored]


def read_scenario(
    rows: pd.DataFrame, title: str, grades: tuple[str, ...] = ("K",)
) -> pd.DataFrame:
    """Normalise one scenario tab to one row per school, program, and grade."""
    rows = _selected_rows(rows, title, grades).copy()
    if rows.empty:
        raise ValueError(f"{title!r} has no rows for grades {list(grades)}.")
    unknown = sorted(set(rows["PathwayName"].astype(str)) - set(PROGRAM_CODES))
    if unknown:
        raise ValueError(f"{title!r}: unmapped program labels {unknown}.")
    bad_seat_types = sorted(set(rows["seat_type"].astype(str)) - set(SEAT_TYPES))
    if bad_seat_types:
        raise ValueError(f"{title!r}: unknown seat types {bad_seat_types}.")
    rows["SchNum"] = _school_codes(rows, title)
    rows["PathwayCode"] = rows["PathwayName"].map(
        lambda label: "+".join(PROGRAM_CODES[label])
    )
    rows["Grade"] = rows["Grade"].astype(str).str.strip()
    rows["capacity"] = pd.to_numeric(rows["capacity"].replace("-", 0), errors="raise")
    if rows["capacity"].isna().any() or rows["capacity"].lt(0).any():
        raise ValueError(f"{title!r}: seats must be present and non-negative.")
    # A program may be listed twice, once per seat type, when some of its
    # seats are attendance-area and the rest citywide. A multi-pool program
    # may also be listed once per pool; a single-pool one never twice per type.
    single_pool = rows["PathwayName"].map(lambda label: len(PROGRAM_CODES[label]) == 1)
    repeated = rows[[*KEYS, "seat_type"]].duplicated(keep=False) & single_pool
    if repeated.any():
        raise ValueError(
            f"{title!r} repeats rows: {rows.loc[repeated, KEYS].to_dict('records')}."
        )
    rows["attendance_area_capacity"] = rows["capacity"].where(
        rows["seat_type"].eq("Attendance Area"), 0
    )
    attendance = rows.groupby("SchNum")["attendance_area_capacity"].sum().gt(0)
    rows["category"] = (
        rows["SchNum"].map(attendance).map({True: "Attendance", False: "Citywide"})
    )
    stated = rows["school_type"].map(SEAT_TYPES)
    disagree = rows.loc[
        stated.ne(rows["category"]), ["SchoolName", "school_type", "category"]
    ]
    for school, stated_type, category in disagree.drop_duplicates().itertuples(
        index=False
    ):
        print(
            f"{title!r}: {school} is labelled {stated_type!r}, but its seats make "
            f"it {category}; using the seats.",
            file=sys.stderr,
        )
    return rows.groupby(KEYS, as_index=False, sort=False).agg(
        SheetSchoolCode=("SheetSchoolCode", "first"),
        SchoolName=("SchoolName", "first"),
        PathwayName=("PathwayName", "first"),
        capacity=("capacity", "sum"),
        attendance_area_capacity=("attendance_area_capacity", "sum"),
        category=("category", "first"),
    )


def _tabs(path: Path) -> list[tuple[str, pd.DataFrame, list[str] | None]]:
    """Return ``(title, raw rows, Summary names)`` for every tab of one input.

    A ``.csv`` is one tab exported from Google Sheets, named
    ``<workbook> - <tab name>.csv``, whose title is its full name (no Summary
    names); anything else is a workbook whose tab titles are matched to its
    ``Summary`` sheet.
    """
    if path.suffix.lower() == ".csv":
        if " - " not in path.stem:
            raise ValueError(
                f"{path.name}: a CSV tab must be named '<workbook> - <tab name>.csv'."
            )
        title = path.stem.split(" - ", 1)[1].strip()
        raw = pd.read_csv(path, header=None, dtype=object, keep_default_na=False)
        return [(title, raw.replace("", None), None)]
    full_names = _full_names(path)
    book = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    return [
        (title, raw, full_names) for title, raw in book.items() if title != "Summary"
    ]


def convert(
    paths: Path | list[Path],
    *,
    suffix: str | None = None,
    exclude: tuple[str, ...] = ("MPD",),
    grades: tuple[str, ...] = ("K",),
) -> pd.DataFrame:
    """Return the overlay table for every scenario tab of the inputs."""
    paths = [paths] if isinstance(paths, Path) else list(paths)
    scenarios: dict[str, pd.DataFrame] = {}
    for path in paths:
        for title, raw, full_names in _tabs(path):
            if any(text in title for text in exclude):
                continue
            rows = _scenario_rows(raw)
            if rows is None:
                continue
            full_name = (
                title if full_names is None else _match_full_name(title, full_names)
            )
            name = scenario_name(full_name, suffix)
            if name in scenarios:
                raise ValueError(f"Two tabs produce the scenario name {name!r}.")
            scenarios[name] = read_scenario(rows, title, grades)
    if not scenarios:
        raise ValueError(f"{[str(path) for path in paths]} have no scenario tabs.")

    descriptive = ["SheetSchoolCode", "SchoolName", "PathwayName"]
    frame = pd.concat(scenarios.values())[[*KEYS, *descriptive]]
    frame = frame.drop_duplicates(KEYS).set_index(KEYS)
    for name, rows in scenarios.items():
        rows = rows.set_index(KEYS)
        frame[f"Scenario_{name}_Capacity"] = rows["capacity"]
        frame[f"Scenario_{name}_AttendanceAreaCapacity"] = rows[
            "attendance_area_capacity"
        ]
        category = rows["category"].groupby(level="SchNum").first()
        frame[f"Scenario_{name}_Category"] = frame.index.get_level_values("SchNum").map(
            category
        )
        # A program a scenario does not list has no seats in it; a school it
        # does not list at all keeps no category from it.
        for kind in ("Capacity", "AttendanceAreaCapacity"):
            column = f"Scenario_{name}_{kind}"
            frame[column] = frame[column].fillna(0).astype(int)
    frame = frame.reset_index()
    order = frame["SchNum"].astype(int), frame["PathwayCode"]
    return (
        frame.assign(_school=order[0])
        .sort_values(["_school", "PathwayCode"], kind="stable")
        .drop(columns="_school")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="workbooks (.xlsx) and single-tab exports (.csv)",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--suffix",
        help="appended to every scenario name, e.g. the school year 2627",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="skip tabs whose title contains this text (default: MPD)",
    )
    parser.add_argument(
        "--grade",
        action="append",
        help="grade rows to read, as the sheet writes them (default: K)",
    )
    args = parser.parse_args()
    frame = convert(
        args.inputs,
        suffix=args.suffix,
        exclude=tuple(args.exclude or ("MPD",)),
        grades=tuple(args.grade or ("K",)),
    )
    frame.to_csv(args.output, index=False)
    names = [
        column.removeprefix("Scenario_").removesuffix("_Capacity")
        for column in frame.columns
        if column.endswith("_Capacity")
    ]
    print(f"Wrote {len(frame)} rows for scenarios: {', '.join(names)}")


if __name__ == "__main__":
    main()
