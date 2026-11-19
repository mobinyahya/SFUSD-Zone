#!/usr/bin/env python
"""Fill blanks in a year's cleaned student tables from the full student file.

The district's "full student file" (one row per student, fall snapshot) holds
demographics and a home location for students whose Main Round or CBEDS
record left them blank. This script reads the converted tables of one year
(``convert_sfusd_transfer.py`` output) and writes a copy of each with those
blanks filled, as ``<table>_with_full_file.csv`` beside the original. It never
changes a value the table already has, so a student's application address,
ethnicity, or home language stays what the converter wrote; and it never
rewrites the original tables.

Filled, where the table is blank and the file has a value:

* ``resolved_ethnicity`` -- the file's category, spelled the way the
  normalisers (``Config.Constants.ETHNICITY_DICT`` and the match evaluator)
  read it; see ``ETHNICITY``. ``-`` and blank leave it blank.
* ``latitude`` / ``longitude`` -- only when both are blank and the file has
  both. Those students then get their 2010 Census Block, BlockGroup and Tract
  and the Block equity indices (``freelunch_prob``, ``FRL Score``, ...)
  exactly as the converter derives them, so a student located this way is in
  the market under ``outside_district_students: ignore`` like any other.
* ``homelang`` -- the table's own name for the file's code, learned from the
  students who carry both (``HOMELANG_MIN_AGREEMENT``); a code with no such
  name is written as the code, which the language-program eligibility reads
  too.
* ``zipcode``, ``englprof``, and ``sped`` (``Y`` -> ``1``).

Usage
-----

.. code-block:: bash

    uv run python analysis/data_prep/fill_from_full_student_file.py \\
        "/soalnas/share/data/school_choice/Data/raw_SFUSD_data_downloads/Nov 18 2026 data transfer/SY26-27/out_updated_SY26-27 full student file.csv" \\
        --year 2627

It also writes ``full_file_fill_report_<year>.md`` / ``.json``: what was filled,
per table and column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.data_prep.convert_sfusd_transfer import (  # noqa: E402
    Report,
    _default_cleaned_dir,
    attach_block_indices,
    attach_geography,
    load_block_indices,
)
from analysis.data_prep.sfusd_transfer_schema import BLOCK_INDEX_COLUMNS  # noqa: E402

TABLES = ("student", "enrolled", "enrolled_all", "applied_or_enrolled")
SUFFIX = "_with_full_file"

#: Full-file ethnicity -> the value written. Each is one the optimization
#: normaliser and the match evaluator both map to its group; ``None`` is no
#: information. An unlisted category stops the fill.
ETHNICITY: dict[str, str | None] = {
    "African-American": "Black or African American",
    "Black or African American": "Black or African American",
    "Asian": "Asian",
    "Filipino": "Filipino",
    "Hispanic/Latino": "Hispanic/Latino",
    "Hispanic": "Hispanic/Latino",
    "White": "White",
    "Pacific Islander": "Pacific Islander",
    "American Indian": "American Indian or Alaska Native",
    "Multiple": "Two or More",
    "Two or More": "Two or More",
    # The converted tables record a declined answer as "Not Specified".
    "Decline to state": "Not Specified",
    "Not Specified": "Not Specified",
    "-": None,
    "0": None,
    "": None,
    # A few rows carry the converted tables' own detailed spelling; kept.
    **{
        name: name
        for name in (
            "American Indian or Alaska Native",
            "Asian Indian",
            "Cambodian",
            "Chinese",
            "Japanese",
            "Korean",
            "Middle Eastern/Arab",
            "Other Asian",
            "Other Pacific Islander",
            "Samoan",
            "Vietnamese",
        )
    },
}
#: A full-file language code takes the table's name for it when a majority of
#: at least this share of the students carrying both agree on that name. The
#: two records disagree for some students (an English name beside the code SP),
#: so only EN may mean English and the bar is a clear majority of the rest.
HOMELANG_MIN_AGREEMENT = 0.6
#: ... out of at least this many students; a rarer code is written as the code.
HOMELANG_MIN_STUDENTS = 10
NO_VALUE = {"", "-"}


def _blank(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().isin(NO_VALUE)


def read_full_file(path: Path) -> pd.DataFrame:
    """The full student file, one row per ``studentno``, every value a string."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [column.strip().lower() for column in frame.columns]
    ids = frame["scrambledstudentno"].str.strip()
    if not ids.str.fullmatch(r"S\d+").all():
        raise ValueError(f"{path.name}: scrambledstudentno outside 'S<digits>'.")
    frame.index = ids.str.slice(1).astype("int64").rename("studentno")
    if frame.index.duplicated().any():
        raise ValueError(f"{path.name} repeats students.")
    unknown = sorted(set(frame["ethnicity"].str.strip()) - set(ETHNICITY))
    if unknown:
        raise ValueError(f"{path.name}: unmapped ethnicity categories {unknown}.")
    return frame.apply(lambda column: column.str.strip())


def homelang_names(tables: dict[str, pd.DataFrame], full: pd.DataFrame) -> dict:
    """Full-file language code -> the converted tables' name for it."""
    pairs = pd.concat(
        [table[["studentno", "homelang"]] for table in tables.values()]
    ).drop_duplicates("studentno")
    pairs = pairs.loc[~_blank(pairs["homelang"])]
    ids = pairs["studentno"].astype("int64")
    pairs = pairs.assign(code=full["homelang"].reindex(ids).to_numpy())
    pairs = pairs.loc[~_blank(pairs["code"])]
    names = {}
    for code, group in pairs.groupby("code")["homelang"]:
        # The commonest disagreement is an English name beside another
        # language's code, so only EN may mean English.
        if code != "EN":
            group = group.loc[group.ne("English")]
        counts = group.value_counts()
        if (
            counts.sum() >= HOMELANG_MIN_STUDENTS
            and counts.iloc[0] / counts.sum() >= HOMELANG_MIN_AGREEMENT
        ):
            names[code] = counts.index[0]
    return names


def _zip_text(value: str) -> str:
    return repr(float(value)) if value.isdigit() else ""


def _numeric_text(values: pd.Series) -> pd.Series:
    return values.map(lambda value: "" if pd.isna(value) else repr(float(value)))


def fill_table(
    table: pd.DataFrame,
    full: pd.DataFrame,
    *,
    homelang: dict[str, str],
    block_indices: pd.DataFrame,
    report: Report,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return ``table`` with its blanks filled, and the count filled per column."""
    result = table.copy()
    ids = result["studentno"].astype("int64")
    source = full.reindex(ids.to_numpy())
    source.index = result.index
    counts: dict[str, int] = {}

    def fill(column: str, values: pd.Series) -> None:
        target = _blank(result[column]) & ~_blank(values)
        result.loc[target, column] = values[target]
        counts[column] = int(target.sum())

    fill("resolved_ethnicity", source["ethnicity"].map(ETHNICITY).fillna(""))
    fill("homelang", source["homelang"].map(lambda code: homelang.get(code, code)))
    # The converted tables store zip codes as floats ("94117.0").
    fill("zipcode", source["zipcode"].map(_zip_text))
    fill("englprof", source["englprof"])
    fill("sped", source["sped"].replace({"Y": "1"}))

    located = (
        _blank(result["latitude"])
        & _blank(result["longitude"])
        & ~_blank(source["lat"])
        & ~_blank(source["lon"])
    )
    counts["latitude/longitude"] = int(located.sum())
    if located.any():
        moved = pd.DataFrame(
            {
                "latitude": pd.to_numeric(source.loc[located, "lat"]),
                "longitude": pd.to_numeric(source.loc[located, "lon"]),
            },
            index=result.index[located],
        )
        moved = attach_geography(moved, report, report_label="full_file_geography")
        moved = attach_block_indices(
            moved, block_indices, report, report_label="full_file_block_indices"
        )
        result.loc[located, "latitude"] = source.loc[located, "lat"]
        result.loc[located, "longitude"] = source.loc[located, "lon"]
        for column in ("census_block", "census_blockgroup", "census_tract"):
            result.loc[located, column] = moved[column].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        for column in BLOCK_INDEX_COLUMNS:
            if column in result.columns and column in moved.columns:
                result.loc[located, column] = _numeric_text(moved[column])
        counts["located_inside_district"] = int(moved["census_block"].notna().sum())
    else:
        counts["located_inside_district"] = 0
    return result, counts


def fill_year(full_file: Path, year: str, *, cleaned_dir: Path, out_dir: Path) -> dict:
    full = read_full_file(full_file)
    tables = {
        name: pd.read_csv(
            cleaned_dir / f"{name}_{year}.csv", dtype=str, keep_default_na=False
        )
        for name in TABLES
    }
    report = Report(year=year, transfer=str(full_file))
    block_indices = load_block_indices(cleaned_dir, report)
    homelang = homelang_names(tables, full)
    summary: dict = {
        "full_file": str(full_file),
        "year": year,
        "homelang_names": homelang,
        "tables": {},
    }
    outputs = {}
    for name, table in tables.items():
        filled, counts = fill_table(
            table, full, homelang=homelang, block_indices=block_indices, report=report
        )
        missing = int((~table["studentno"].astype("int64").isin(full.index)).sum())
        summary["tables"][name] = {
            "rows": len(table),
            "students_absent_from_full_file": missing,
            "filled": counts,
        }
        outputs[f"{name}_{year}{SUFFIX}.csv"] = filled
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_name, frame in outputs.items():
        frame.to_csv(out_dir / file_name, index=False)
    (out_dir / f"full_file_fill_report_{year}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (out_dir / f"full_file_fill_report_{year}.md").write_text(to_markdown(summary))
    return summary


def to_markdown(summary: dict) -> str:
    columns = list(next(iter(summary["tables"].values()))["filled"])
    lines = [
        f"# Full student file fill, {summary['year']}",
        "",
        f"Source: `{summary['full_file']}`. Only blanks are filled; every value "
        "the converted table had is kept.",
        "",
        "| table | rows | absent from file | " + " | ".join(columns) + " |",
        "|---|---|---|" + "---|" * len(columns),
    ]
    for name, entry in summary["tables"].items():
        cells = [str(entry["filled"][column]) for column in columns]
        lines.append(
            f"| {name} | {entry['rows']} | "
            f"{entry['students_absent_from_full_file']} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "`located_inside_district` counts the newly located students whose "
        "coordinates fall in a 2010 Census Block; the rest stay outside the "
        "market under `outside_district_students: ignore`.",
        "",
        "Home-language codes written as the tables' names: "
        + ", ".join(
            f"{code} -> {name}" for code, name in summary["homelang_names"].items()
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("full_file", type=Path)
    parser.add_argument("--year", required=True)
    parser.add_argument("--cleaned-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cleaned_dir = args.cleaned_dir or _default_cleaned_dir()
    summary = fill_year(
        args.full_file,
        args.year,
        cleaned_dir=cleaned_dir,
        out_dir=args.out or cleaned_dir,
    )
    print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
