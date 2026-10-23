#!/usr/bin/env python
"""Convert the district's ``v.10.22.26 Capacities.xlsx`` into a capacity overlay.

The workbook holds two kindergarten capacity proposals, one per sheet:

* ``Ideal Capacity`` -- "Max seats if all classrooms used (ideal)".
* ``MPD Proposal``   -- "After Change Scenario (Manual Edits)".

Each row names a school by the district's school code, a program by a
district label such as ``SP DLI``, and a grade. The output is a CSV in the
overlay format ``loaders.tables.apply_capacity_scenario`` reads (``SchNum``,
``PathwayCode``, ``Grade``, ``Scenario_<name>_Capacity``), with one capacity
column per sheet. The base ``capacity_scenarios`` registry binds the names
``v102226_ideal`` and ``v102226_mpd`` to it.

Usage
-----

.. code-block:: bash

    uv run python analysis/data_prep/convert_capacities_v102226.py \
        "/soalnas/share/data/school_choice/Data/capacity_management/raw/v.10.22.26 Capacities.xlsx" \
        /soalnas/share/data/school_choice/Data/capacity_management/sfusd_capacities_v10.22.26.csv

How the workbook maps onto the program tables
---------------------------------------------

* **Program labels.** The workbook's label is one district program; the
  program tables split a language program into its lottery pools (native /
  non-native speakers). ``SP DLI`` becomes ``SE+SN``, and the overlay spreads a
  ``+``-joined row over the pools in proportion to their current seats.
* **School codes.** The workbook gives Mission Bay its district code 731; the
  program tables carry it as 909, aliased to 999 when Mission Bay is
  included. Its rows are written as 999, keeping 731 in ``SheetSchoolCode``.
  The ``NEW`` Mandarin Immersion TK-8 has no school code, location, or program
  in any table, so it is dropped.
* **Missing values.** ``-`` in ``MPD Proposal`` means the proposal removes the
  program and is written as 0. A blank ``Ideal Capacity`` cell marks a program
  that only ``MPD Proposal`` creates and is written as 0 as well.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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

#: District school code -> the program tables' school ID.
SCHOOL_IDS = {731: 999}

#: Sheet -> overlay scenario name.
SHEETS = {"Ideal Capacity": "v102226_ideal", "MPD Proposal": "v102226_mpd"}

KEYS = ["SheetSchoolCode", "SchoolName", "PathwayName", "Grade"]


def _read_sheet(path: Path, sheet: str, name: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, dtype=object)
    if len(frame.columns) != 5:
        raise ValueError(f"{sheet!r} has {len(frame.columns)} columns, expected 5.")
    frame.columns = [*KEYS, f"Scenario_{name}_Capacity"]
    if frame[KEYS].duplicated().any():
        raise ValueError(f"{sheet!r} repeats a school/program/grade row.")
    return frame


def convert(path: Path) -> pd.DataFrame:
    sheets = [_read_sheet(path, sheet, name) for sheet, name in SHEETS.items()]
    frame = sheets[0]
    for other in sheets[1:]:
        frame = frame.merge(other, on=KEYS, how="outer")

    unknown = sorted(set(frame["PathwayName"]) - set(PROGRAM_CODES))
    if unknown:
        raise ValueError(f"Unmapped workbook program labels: {unknown}.")
    code = pd.to_numeric(frame["SheetSchoolCode"], errors="coerce")
    unmatched = frame.loc[code.isna(), "SheetSchoolCode"].unique().tolist()
    if unmatched != ["NEW"]:
        raise ValueError(f"Unexpected non-numeric school codes: {unmatched}.")
    frame = frame.loc[code.notna()].copy()
    frame["SheetSchoolCode"] = code.loc[frame.index].astype(int)

    frame["SchNum"] = frame["SheetSchoolCode"].map(
        lambda value: SCHOOL_IDS.get(value, value)
    )
    frame["PathwayCode"] = frame["PathwayName"].map(
        lambda label: "+".join(PROGRAM_CODES[label])
    )
    for name in SHEETS.values():
        column = f"Scenario_{name}_Capacity"
        values = frame[column].replace("-", 0)
        frame[column] = pd.to_numeric(values, errors="raise").fillna(0).astype(int)
    columns = [
        "Grade",
        "PathwayCode",
        "PathwayName",
        "SchNum",
        "SchoolName",
        "SheetSchoolCode",
        *(f"Scenario_{name}_Capacity" for name in SHEETS.values()),
    ]
    return frame[columns].sort_values(["SchNum", "PathwayCode"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.workbook).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
