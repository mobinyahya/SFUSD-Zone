"""Fit the exp8 MNL with an outside option on one transfer year's kindergarten.

Uses the same features as ``utility.2324.exp8`` (see :mod:`choice.estimation.exp8`),
plus TK-to-K auto-promotion, and adds an outside option identified from who
enrolled:

* **Rankings** are the converted ``r1`` lists in ``student_<year>.csv``, which
  already carry the transfer's promotion handling: a promotion-eligible student's
  feeder is appended to their list unless they rank it. With the feeder
  guaranteed, ranking costs a promote nothing, so they should list every program
  they prefer to it, and ``list + [feeder]`` is a revealed ranking. A promote
  with no list of their own (``feeder_only``) ranks the feeder alone. The
  attendance-area program the converter appends for those students, and
  ``aa_only`` students' whole list, are invented, so they are not used.
* **Enrolment** is the year's ``enrolled_<year>.csv``, which the converter builds
  from the demographics extract: a seated student with a fall record at
  kindergarten at a real school is in it, at that school, and school 899, no
  record, or a record at another grade leaves them out. A student who enrolled at
  their post-run seat chose it over the outside option. One who enrolled
  somewhere else chose that program over the seat and the outside option. One
  who never enrolled chose the outside option over the seat.

Autopromotion status enters twice: ``promote_feeder`` marks the program a
student holds a promotion claim on, and ``outside_promote_eligible`` shifts
their outside option. ``outside_low_income`` lets the outside option differ by
the same income split exp8 uses for distance.

Like exp8, students seated in a special-education program are dropped and those
programs are removed from everyone's choice set. Closed programs (no seats) are
removed too.

    uv run python -m choice.estimation.fit_transfer_year --year 2526
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from choice.estimation.exp8 import (
    SPECIAL_PROGRAM_TYPES,
    build_design,
    ranked_program_ids,
)
from choice.estimation.likelihood import (
    OUTSIDE,
    ChoiceData,
    FitResult,
    OutsideOptionMNL,
)
from loaders import DataScenario, load_scenario, read_csv_source
from loaders.tables import NOT_ENROLLED_SCHOOL_ID

PUBLISHED_WEIGHTS = "simulation-files/choice-model/ChoiceModel_20240514/weights.csv"
GRADE = "KG"
OUTSIDE_NAMES = ("outside_low_income", "outside_promote_eligible")
#: Applicants this far from every school are moving in; their distances are not used.
MAX_LOCATED_MILES = 10.0
#: Mission Bay opened in August 2026; earlier years' Mission Bay tables are
#: counterfactual, so a year before this one is read without it.
MISSION_BAY_OPENED = "2627"


def _data_root() -> Path:
    return Path(load_scenario({"scenario": "legacy", "overrides": {}}).roots["data"])


def transfer_scenario(year: str, population: str = "applicant") -> DataScenario:
    """The ``sfusd-<year>`` kindergarten market, with Mission Bay where it existed."""
    filters = {
        "student_population": population,
        "include_mission_bay": year >= MISSION_BAY_OPENED,
    }
    return load_scenario(
        {"scenario": f"sfusd-{year}", "overrides": {"filters": {"assignment": filters}}}
    )


def load_market(year: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """``year``'s kindergarten students, programs and schools, and their sources.

    Read whole from the registry-selected tables, without the assignment
    group's filters: the fit and the prediction select rows themselves.
    """
    data = transfer_scenario(year)
    roles = ("assignment.students", "assignment.programs", "assignment.schools")
    sources = {role: data.source(role) for role in roles}
    students, programs, schools = (
        read_csv_source(sources[role], low_memory=False) for role in roles
    )
    students = students[students["grade"] == GRADE].reset_index(drop=True)
    paths = {role.split(".")[1]: str(source.path) for role, source in sources.items()}
    return students, programs, schools, paths


def load_enrolment(year: str) -> pd.DataFrame:
    """Where each enrolled kindergartner enrolled, indexed by ``studentno``.

    A student absent from the table did not enroll. Rows at school 899
    ("Central Enrollment") or with no school, which a pre-2024-25 table can
    carry, are dropped as not enrolled too.
    """
    enrolled = read_csv_source(
        transfer_scenario(year, "enrolled").source("assignment.students"),
        low_memory=False,
    )
    enrolled = enrolled[enrolled["grade"] == GRADE]
    school = pd.to_numeric(enrolled["enrolled_idschool"], errors="coerce")
    enrolled = enrolled[school.notna() & school.ne(NOT_ENROLLED_SCHOOL_ID)]
    return pd.DataFrame(
        {
            "school": enrolled["enrolled_idschool"].astype(int).to_numpy(),
            "program": enrolled["enrolled_programcode"].to_numpy(),
        },
        index=enrolled["studentno"].astype(int).to_numpy(),
    )


def _program_id(school, program) -> str | None:
    if pd.isna(school) or pd.isna(program):
        return None
    return f"{int(school)}-{program}-{GRADE}"


def build_choice_data(year: str, *, top_k: int = 5):
    """Assemble the design and observed choices for ``year``'s kindergarten."""
    students, programs, schools, _ = load_market(year)
    counts: Counter[str] = Counter(market_students=len(students))

    special = programs["program_type"].isin(SPECIAL_PROGRAM_TYPES)
    closed = programs["capacity"] <= 0
    counts["programs_special"] = int(special.sum())
    counts["programs_closed"] = int((closed & ~special).sum())
    programs = programs[~special & ~closed].reset_index(drop=True)
    choice_set = set(programs["program_id"])

    seated_special = students["r1_programcode"].isin(SPECIAL_PROGRAM_TYPES)
    counts["students_dropped_special_seat"] = int(seated_special.sum())
    students = students[~seated_special].reset_index(drop=True)

    record = load_enrolment(year).reindex(students["studentno"].astype(int).to_numpy())

    lists = ranked_program_ids(students, GRADE)
    feeders = {
        int(row.studentno): _program_id(row.feeder_school, row.feeder_program)
        for row in students.itertuples()
        if row.promote_eligible == 1
    }
    rankings_ids, offers_ids, enrolled_ids = [], [], []
    for i, row in enumerate(students.itertuples()):
        source = row.pref_source
        listed = lists[i]
        if source == "feeder_only":
            listed = [feeders[int(row.studentno)]]
            if lists[i][:1] != listed:
                raise ValueError(
                    f"Student {row.studentno}: feeder_only list does not start at the feeder."
                )
        elif source == "aa_only":
            listed = []
        kept = [p for p in listed if p in choice_set]
        counts["ranked_entries_outside_choice_set"] += len(listed) - len(kept)
        rankings_ids.append(kept)
        counts[f"rank_{source}" if kept else "rank_none"] += 1

        seat = _program_id(row.r1_idschool, row.r1_programcode)
        seat = seat if seat in choice_set else None
        school = record["school"].iloc[i]
        if pd.isna(school):
            offer, choice = ([seat], OUTSIDE) if seat else ([], None)
            counts["enrol_not_enrolled" if seat else "enrol_skipped_no_seat"] += 1
        else:
            took = _program_id(school, record["program"].iloc[i])
            if took not in choice_set:
                offer, choice = [], None
                counts["enrol_skipped_enrolled_outside_choice_set"] += 1
            elif took == seat:
                offer, choice = [seat], seat
                counts["enrol_at_seat"] += 1
            else:
                offer, choice = [p for p in (seat, took) if p], took
                counts["enrol_elsewhere"] += 1
        offers_ids.append(offer)
        enrolled_ids.append(choice)

    revealed = [set(r) | set(o) for r, o in zip(rankings_ids, offers_ids)]
    design = build_design(
        students,
        programs,
        schools,
        revealed=revealed,
        feeders=feeders,
        max_located_miles=MAX_LOCATED_MILES,
    )
    counts["students_unlocated"] = int((~design.located).sum())
    column = {p: j for j, p in enumerate(design.program_ids)}
    promote = students["promote_eligible"].fillna(0).to_numpy(dtype=float)
    data = ChoiceData(
        design=design,
        rankings=[[column[p] for p in r] for r in rankings_ids],
        offers=[[column[p] for p in o] for o in offers_ids],
        enrolled=[c if c in (None, OUTSIDE) else column[c] for c in enrolled_ids],
        outside_features=np.column_stack([design.low_income.astype(float), promote]),
        outside_names=OUTSIDE_NAMES,
        top_k=top_k,
    )
    return data, students, dict(counts)


def diagnostics(
    model: OutsideOptionMNL, fit: FitResult, students: pd.DataFrame
) -> dict:
    """Predicted against observed non-enrolment, and acceptable-set sizes."""
    V, V0 = model.utilities(fit.theta)
    a = model.arrays
    rows = a.offer_rows
    present = a.offer_cols >= 0
    inside = np.where(present, V[rows[:, None], np.maximum(a.offer_cols, 0)], -np.inf)
    options = np.concatenate([V0[rows, None], inside], axis=1)
    p_out = np.exp(options[:, 0] - np.logaddexp.reduce(options, axis=1))
    observed = (a.chosen_slot == 0).astype(float)
    frame = pd.DataFrame(
        {
            "predicted": p_out,
            "observed": observed,
            "low_income": model.data.design.low_income[rows],
            "promote_eligible": students["promote_eligible"].to_numpy()[rows] == 1,
            "pref_source": students["pref_source"].to_numpy()[rows],
        }
    )
    groups = {"all": frame.assign(all="all").groupby("all")}
    groups |= {
        c: frame.groupby(c) for c in ("low_income", "promote_eligible", "pref_source")
    }
    not_enrolled = {
        name: {
            str(k): {
                "n": int(len(g)),
                "observed": float(g.observed.mean()),
                "predicted": float(g.predicted.mean()),
            }
            for k, g in grouped
        }
        for name, grouped in groups.items()
    }
    net = V - V0[:, None]
    acceptable = (net > 0).sum(axis=1)
    return {
        "not_enrolled_share": not_enrolled,
        "acceptable_programs": {
            "share_with_none": float((acceptable == 0).mean()),
            "median": float(np.median(acceptable)),
            "share_with_none_low_income": float(
                (acceptable[model.data.design.low_income] == 0).mean()
            ),
        },
    }


def compare_published(fit: FitResult, root: Path, year: str) -> dict:
    """Correlate the shared coefficients with the published 2023-24 exp8 fit."""
    published = pd.read_csv(root / PUBLISHED_WEIGHTS).set_index("feature")[
        "coefficient"
    ]
    ours = pd.Series(fit.theta, index=fit.names)
    shared = [
        n for n in ours.index if n in published.index and not n.startswith("school_")
    ]
    schools = [
        n for n in ours.index if n.startswith("school_") and n in published.index
    ]
    demeaned = ours[schools] - ours[schools].mean()
    return {
        "non_fe_coefficients": {
            n: {year: float(ours[n]), "2324_published": float(published[n])}
            for n in shared
        },
        "school_fe_correlation": float(
            np.corrcoef(demeaned, published[schools] - published[schools].mean())[0, 1]
        ),
        "school_fe_count": len(schools),
    }


def write_outputs(
    out: Path, year: str, model: OutsideOptionMNL, fit: FitResult, report: dict
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    se = fit.std_error
    z = np.divide(fit.theta, se, out=np.full_like(se, np.nan), where=se > 0)
    weights = pd.DataFrame(
        {
            "feature": fit.names,
            "coefficient": fit.theta,
            "std_error": se,
            "z_stats": z,
            "p_value": 2 * norm.sf(np.abs(z)),
        }
    )
    reference = pd.DataFrame(
        [
            {
                "feature": f"program_type_{model.data.reference_type}",
                "coefficient": 0.0,
                "std_error": 0.0,
            }
        ]
    )
    weights = pd.concat([weights, reference], ignore_index=True)
    weights["log_likelihood"] = fit.log_likelihood
    weights.to_csv(out / "weights.csv", index=False)

    V, V0 = model.utilities(fit.theta)
    design = model.data.design
    labels = [f"{year}-{s}" for s in design.student_ids]
    estimates = pd.DataFrame(V - V0[:, None], columns=design.program_ids)
    estimates.insert(0, "studentno", labels)
    estimates.to_csv(out / f"estimates_{year}.csv", index=False)
    pd.DataFrame({"studentno": labels, "outside_utility": V0}).to_csv(
        out / f"outside_option_{year}.csv", index=False
    )
    (out / "fit_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", default="2526")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args(argv)

    root = _data_root()
    out = (
        args.out
        or root
        / "simulation-files"
        / "choice-model"
        / f"ChoiceModel_{args.year}_outside"
    )
    data, students, counts = build_choice_data(args.year, top_k=args.top_k)
    model = OutsideOptionMNL(data, ridge=args.ridge)
    fit = model.fit()
    report = {
        "year": args.year,
        "grade": GRADE,
        "top_k": args.top_k,
        "ridge": args.ridge,
        "counts": counts,
        "students": len(data.design.student_ids),
        "programs": len(data.design.program_ids),
        "parameters": len(fit.names),
        "log_likelihood": fit.log_likelihood,
        "rank_log_likelihood": fit.rank_log_likelihood,
        "enrolment_log_likelihood": fit.enrolment_log_likelihood,
        "converged": fit.converged,
        "message": fit.message,
        "iterations": fit.iterations,
        "max_abs_gradient": fit.max_abs_gradient,
        "diagnostics": diagnostics(model, fit, students),
        "vs_2324_exp8": compare_published(fit, root, args.year),
        "utility_convention": (
            f"estimates_{args.year}.csv holds V_ij - V_i0: utility relative to the student's own "
            "outside option, -inf where ineligible. A program is preferred to not enrolling iff "
            "its entry is > 0."
        ),
    }
    write_outputs(out, args.year, model, fit, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("counts", "log_likelihood", "converged", "max_abs_gradient")
            },
            indent=2,
        )
    )
    print(json.dumps(report["diagnostics"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
