"""Apply a fitted outside-option MNL to another year's kindergarten applicants.

The fit in ``ChoiceModel_<fit year>_outside/weights.csv`` is evaluated on the
target year's own students and programs: their homes, lists, siblings and
promotion claims, through the same exp8 design the fit used. Like the fit's own
``estimates_<year>.csv``, the output is utility net of each student's outside
option, ``-inf`` where ineligible, so it can drive ``list-length:
outside_option`` directly.

A school the fit never saw has no fixed effect. It gets the mean fitted school
effect, which says nothing about it beyond what the other features do. A program
type the fit never saw gets the GE reference effect (0). Mission Bay (999),
which opened in August 2026, and the one-seat ``476-NC`` and ``724-NS``
programs, closed in 2025-26, are the cases this exists for. The report lists
every program either rule touched.

    uv run python -m choice.estimation.predict --fit-year 2526 --year 2627
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from choice.estimation.exp8 import (
    SPECIAL_PROGRAM_TYPES,
    build_design,
    published_utilities,
    ranked_program_ids,
)
from choice.estimation.fit_transfer_year import (
    GRADE,
    MAX_LOCATED_MILES,
    OUTSIDE_NAMES,
    _data_root,
    _program_id,
    load_market,
)


def predict_year(fit_dir: Path, year: str) -> tuple[pd.DataFrame, dict]:
    weights = pd.read_csv(fit_dir / "weights.csv").set_index("feature")["coefficient"]
    students, programs, schools, sources = load_market(year)
    programs = programs[~programs["program_type"].isin(SPECIAL_PROGRAM_TYPES)]
    programs = programs.reset_index(drop=True)

    feeders = {
        int(row.studentno): _program_id(row.feeder_school, row.feeder_program)
        for row in students.itertuples()
        if row.promote_eligible == 1
    }
    lists = ranked_program_ids(students, GRADE)
    revealed = [
        set(listed) | ({feeders[int(s)]} if int(s) in feeders else set())
        for listed, s in zip(lists, students["studentno"])
    ]
    design = build_design(
        students,
        programs,
        schools,
        revealed=revealed,
        feeders=feeders,
        max_located_miles=MAX_LOCATED_MILES,
    )

    school_effects = weights[weights.index.str.startswith("school_")]
    imputed = sorted(
        {int(s) for s in design.school_ids if f"school_{s}" not in school_effects}
    )
    missing_types = sorted(
        {t for t in design.program_types if f"program_type_{t}" not in weights}
    )
    inside = published_utilities(design, weights)

    promote = students["promote_eligible"].fillna(0).to_numpy(dtype=float)
    outside_features = np.column_stack([design.low_income.astype(float), promote])
    outside = outside_features @ np.array([weights[name] for name in OUTSIDE_NAMES])
    net = inside - outside[:, None]

    estimates = pd.DataFrame(net, columns=design.program_ids)
    estimates.insert(0, "studentno", [f"{year}-{s}" for s in design.student_ids])
    acceptable = (net > 0).sum(axis=1)
    report = {
        "fit": str(fit_dir),
        "year": year,
        "students": len(students),
        "programs": len(programs),
        "programs_source": sources["programs"],
        "schools_source": sources["schools"],
        "schools_with_imputed_effect": imputed,
        "imputed_school_effect": float(school_effects.mean()),
        "program_types_with_reference_effect": missing_types,
        "programs_with_imputed_effects": [
            p
            for p, sch, t in zip(
                design.program_ids, design.school_ids, design.program_types
            )
            if int(sch) in imputed or t in missing_types
        ],
        "students_unlocated": int((~design.located).sum()),
        "acceptable_programs": {
            "share_with_none": float((acceptable == 0).mean()),
            "median": float(np.median(acceptable)),
        },
        "utility_convention": (
            "V_ij - V_i0 under the fit's coefficients; > 0 means preferred to not "
            "enrolling, -inf where ineligible."
        ),
    }
    return estimates, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit-year", default="2526")
    parser.add_argument("--year", default="2627")
    parser.add_argument("--fit-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    root = _data_root()
    fit_dir = args.fit_dir or (
        root / "simulation-files/choice-model" / f"ChoiceModel_{args.fit_year}_outside"
    )
    estimates, report = predict_year(fit_dir, args.year)
    estimates.to_csv(fit_dir / f"estimates_{args.year}.csv", index=False)
    (fit_dir / f"prediction_report_{args.year}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
