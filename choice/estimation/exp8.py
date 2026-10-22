"""The exp8 MNL design behind ``utility.2324.exp8``, rebuilt from the data.

The code that fitted ``ChoiceModel_20240514`` is not in this repository; its
config lists the features by name only. Every rule below was decoded from the
published utilities instead: applying ``ChoiceModel_20240514/weights.csv`` to
this design reproduces ``estimates_2324.csv`` on every finite entry, to the
~0.01-util noise that comes from the original distance table, and reproduces
its eligibility mask for ~99% of students per language family. The mismatches
are students whose later-round lists made them eligible, which the transfer
years do not carry. ``choice/tests/test_estimation_real.py`` pins both.

Decoded rules that are not obvious from the feature names:

* ``ethn_match`` is Hispanic for Spanish and an East/Southeast Asian group for
  Cantonese, Mandarin, Japanese and Korean (Asian Indian is not in it).
* ``lang_match_not_XB`` fires only on the native-speaker tracks ``SN``, ``NS``,
  ``CN``, ``CT`` and ``MN``; it never fires on ``JN``, ``KN``, ``NC`` or any
  English-track (``XE``) program in the published utilities.
* ``prog_XN_x_English`` covers ``SN``, ``CN``, ``MN``, ``JN`` and ``KN``.
* A missing median income counts as not low-income, and a student with no
  coordinates gets zero on every distance feature.
* A language family is eligible when the student's home language is its
  language (Spanish, Cantonese, Mandarin, Korean only) or the student ranked any
  program in it. English-track and GE programs are always eligible.

``promote_feeder`` is not part of exp8. It is the TK-to-K auto-promotion
indicator: 1 at the program a promotion-eligible student holds a claim on.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

SPECIAL_PROGRAM_TYPES = frozenset({"AF", "AO", "ED", "MM", "MS", "SA", "TC"})

#: ``low_income`` is ``median_hh_income < LOW_INCOME_CUTOFF``. The published
#: design puts the cutoff in (83,125, 83,233]; no 2023-24 student falls between.
LOW_INCOME_CUTOFF = 83_233.0
NEAR_MILES = 0.5
EARTH_RADIUS_MILES = 3958.8

#: Program types whose eligibility is restricted, grouped by language family.
RESTRICTED_FAMILIES: dict[str, frozenset[str]] = {
    "Spanish": frozenset({"SB", "SN", "NS"}),
    "Cantonese": frozenset({"CB", "CN", "CT", "NC"}),
    "Mandarin": frozenset({"MN"}),
    "Japanese": frozenset({"JN"}),
    "Korean": frozenset({"KN"}),
    "Filipino": frozenset({"FB"}),
}
#: Home languages that make a family eligible without ranking it.
AUTO_ELIGIBLE_LANGUAGES = frozenset({"Spanish", "Cantonese", "Mandarin", "Korean"})

HOME_LANGUAGES = {
    "English": "English",
    "Spanish": "Spanish",
    "Cantonese": "Cantonese",
    "Mandarin (Putonghua)": "Mandarin",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Filipino (Pilipino or Tagalog)": "Filipino",
    "Tagalog": "Filipino",
}

ENGLISH_TRACKS = {
    "CE": "Cantonese",
    "SE": "Spanish",
    "JE": "Japanese",
    "ME": "Mandarin",
    "KE": "Korean",
}
NATIVE_TRACKS = frozenset({"SN", "CN", "MN", "JN", "KN"})
BILITERACY = {"CB": "Cantonese", "FB": "Filipino", "SB": "Spanish"}
LANG_MATCH_NOT_XB = {
    "SN": "Spanish",
    "NS": "Spanish",
    "CN": "Cantonese",
    "CT": "Cantonese",
    "MN": "Mandarin",
}

HISPANIC = frozenset({"Hispanic/Latino", "Hispanic"})
#: Cambodian is the only Southeast Asian label 2023-24 shows; Hmong and Laotian
#: are assumed to belong with it.
ASIAN = frozenset(
    {
        "Chinese",
        "Filipino",
        "Japanese",
        "Korean",
        "Vietnamese",
        "Other Asian",
        "Cambodian",
        "Hmong",
        "Laotian",
    }
)

EXP8_FEATURES: tuple[str, ...] = (
    "sibling",
    *(f"prog_{t}_x_English_x_ethn_match{m}" for t in ENGLISH_TRACKS for m in (0, 1)),
    "prog_XN_x_English",
    *(f"lang_match_{t}" for t in BILITERACY),
    "lang_match_not_XB",
    "distance_indicator_lt_0_5",
    "distance_x_low_income0",
    "distance_x_low_income1",
)
PROMOTE_FEATURE = "promote_feeder"


@dataclass(frozen=True)
class Design:
    """Student-by-program design for an exp8-style MNL.

    ``features[i, j]`` holds the non-fixed-effect regressors; the school and
    program-type fixed effects are carried as ``school_ids``/``program_types``
    so the likelihood can index them instead of materialising ~100 dummies.
    """

    student_ids: np.ndarray
    program_ids: np.ndarray
    school_ids: np.ndarray
    program_types: np.ndarray
    eligible: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    low_income: np.ndarray
    located: np.ndarray


def parse_list(value) -> list:
    """Parse a stringified list column (``"[413, 420]"``); blank is empty."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text in {"[]", "nan"}:
        return []
    return list(ast.literal_eval(text))


def ranked_program_ids(students: pd.DataFrame, grade: str = "KG") -> list[list[str]]:
    """Each student's ``r1`` list as program ids, in rank order, deduplicated."""
    lists = []
    for schools, types in zip(students["r1_ranked_idschool"], students["r1_programs"]):
        schools, types = parse_list(schools), parse_list(types)
        if len(schools) != len(types):
            raise ValueError("r1_ranked_idschool and r1_programs differ in length.")
        ordered = [f"{int(s)}-{t}-{grade}" for s, t in zip(schools, types)]
        lists.append(list(dict.fromkeys(ordered)))
    return lists


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def build_design(
    students: pd.DataFrame,
    programs: pd.DataFrame,
    schools: pd.DataFrame,
    *,
    revealed: Sequence[Iterable[str]],
    feeders: Mapping[int, str] | None = None,
    max_located_miles: float | None = None,
) -> Design:
    """Build the exp8 design for ``students`` over ``programs``.

    ``revealed[i]`` is every program id student ``i`` is known to have had in
    play -- ranked, assigned or enrolled at. It decides restricted-family
    eligibility, so a program a student was seated at is never masked out.
    ``feeders`` maps a promotion-eligible student to their feeder program id and
    adds the ``promote_feeder`` column; leave it ``None`` for plain exp8.
    A student farther than ``max_located_miles`` from every school is treated
    like one with no coordinates: an applicant moving in from out of state has
    no meaningful distance to any school. ``None`` keeps exp8's behaviour.
    """
    if len(revealed) != len(students):
        raise ValueError("revealed must have one entry per student.")
    program_ids = programs["program_id"].astype(str).to_numpy()
    school_ids = programs["school_id"].astype(int).to_numpy()
    types = programs["program_type"].astype(str).to_numpy()
    n, J = len(students), len(program_ids)

    language = students["homelang"].map(HOME_LANGUAGES).to_numpy(dtype=object)
    english = students["homelang"].eq("English").to_numpy()
    ethnicity = students["resolved_ethnicity"].to_numpy(dtype=object)
    hispanic = np.isin(ethnicity, list(HISPANIC))
    asian = np.isin(ethnicity, list(ASIAN))
    income = pd.to_numeric(students["median_hh_income"], errors="coerce").to_numpy()
    low_income = np.nan_to_num(income, nan=np.inf) < LOW_INCOME_CUTOFF

    location = schools.set_index("school_id").loc[school_ids]
    lat = pd.to_numeric(students["latitude"], errors="coerce").to_numpy()[:, None]
    lon = pd.to_numeric(students["longitude"], errors="coerce").to_numpy()[:, None]
    distance = haversine_miles(
        lat, lon, location["lat"].to_numpy()[None], location["lon"].to_numpy()[None]
    )
    located = np.isfinite(distance)
    if max_located_miles is not None:
        located &= (
            np.nanmin(distance, axis=1, initial=np.inf, where=located)
            <= max_located_miles
        )[:, None]
    distance = np.where(located, distance, 0.0)

    eligible = np.ones((n, J), dtype=bool)
    type_of = dict(zip(program_ids, types))
    for family, family_types in RESTRICTED_FAMILIES.items():
        columns = np.isin(types, list(family_types))
        ranked = np.array(
            [any(type_of.get(p) in family_types for p in seen) for seen in revealed]
        )
        home = (
            language == family
            if family in AUTO_ELIGIBLE_LANGUAGES
            else np.zeros(n, bool)
        )
        eligible[:, columns] = (ranked | home)[:, None]

    sibling_sets = [set(int(s) for s in parse_list(v)) for v in students["sibling"]]
    columns: dict[str, np.ndarray] = {}
    columns["sibling"] = np.array(
        [[s in sibs for s in school_ids] for sibs in sibling_sets], dtype=float
    )
    for track, track_language in ENGLISH_TRACKS.items():
        at = (types == track)[None]
        match = hispanic if track_language == "Spanish" else asian
        for flag in (0, 1):
            chosen = match if flag else ~match
            columns[f"prog_{track}_x_English_x_ethn_match{flag}"] = (
                at & (english & chosen)[:, None]
            ).astype(float)
    columns["prog_XN_x_English"] = (
        np.isin(types, list(NATIVE_TRACKS))[None] & english[:, None]
    ).astype(float)
    for track, track_language in BILITERACY.items():
        columns[f"lang_match_{track}"] = (
            (types == track)[None] & (language == track_language)[:, None]
        ).astype(float)
    program_language = np.array([LANG_MATCH_NOT_XB.get(t) for t in types], dtype=object)
    columns["lang_match_not_XB"] = (
        (program_language[None] == language[:, None]) & (program_language != None)[None]  # noqa: E711
    ).astype(float)
    columns["distance_indicator_lt_0_5"] = (located & (distance < NEAR_MILES)).astype(
        float
    )
    columns["distance_x_low_income0"] = distance * (~low_income)[:, None]
    columns["distance_x_low_income1"] = distance * low_income[:, None]
    names = list(EXP8_FEATURES)
    if feeders is not None:
        column_of = {p: j for j, p in enumerate(program_ids)}
        feeder = np.zeros((n, J))
        for i, studentno in enumerate(students["studentno"].astype(int)):
            program = feeders.get(int(studentno))
            if program in column_of:
                feeder[i, column_of[program]] = 1.0
        columns[PROMOTE_FEATURE] = feeder
        names.append(PROMOTE_FEATURE)

    features = np.stack([columns[name] for name in names], axis=-1)
    return Design(
        student_ids=students["studentno"].astype(int).to_numpy(),
        program_ids=program_ids,
        school_ids=school_ids,
        program_types=types,
        eligible=eligible,
        features=features,
        feature_names=tuple(names),
        low_income=low_income,
        located=located[:, 0],
    )


def inside_utilities(design: Design, fixed: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """``fixed_j + X_ij . beta``, ``-inf`` where ineligible.

    ``fixed`` is each program's school plus program-type effect, one per design
    column, and ``beta`` is aligned to ``design.feature_names``.
    """
    utility = fixed[None] + design.features @ beta
    return np.where(design.eligible, utility, -np.inf)


def published_utilities(design: Design, weights: Mapping[str, float]) -> np.ndarray:
    """Evaluate a ``weights.csv``-style coefficient table on ``design``.

    A school the table never saw gets its mean school effect, which says nothing
    about it beyond what the other features do; a program type it never saw gets
    the reference effect, 0.
    """
    school_effects = {k: v for k, v in weights.items() if k.startswith("school_")}
    mean_school = float(np.mean(list(school_effects.values())))
    fixed = np.array(
        [
            school_effects.get(f"school_{s}", mean_school)
            + weights.get(f"program_type_{t}", 0.0)
            for s, t in zip(design.school_ids, design.program_types)
        ]
    )
    beta = np.array([weights[name] for name in design.feature_names])
    return inside_utilities(design, fixed, beta)
