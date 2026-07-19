"""Generate a small fully-fake dataset for pipeline tests.

Produces year-consistent (2223) CSVs with the exact schemas the simulator
and evaluator expect:

  <out>/student_2223_filtered.csv
  <out>/programs_without_specialprogs_2223.csv
  <out>/Cleaned/schools_rehauled_2223.csv
  <out>/models/selectedfake_2223_k1_prog_gesplit/estimates_2223.csv
  <out>/zones/concept1zones.csv

No real student records are used — every row is synthetic and the
generation is deterministic (fixed seed). The committed copy lives in
tests/fixtures/fake_2223/ and is exercised by tests/test_full_pipeline.py.

Usage:
    python scripts/generators/generate_fake_dataset.py \
        --out-dir tests/fixtures/fake_2223 --num-students 200
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FAKE_YEAR_PREFIX = "2223"
FAKE_MODEL_NAME = "selectedfake_2223_k1_prog_gesplit"
GRADE = "KG"

# 15 attendance-area schools, SF-plausible coordinates (lat, lon).
SCHOOL_IDS = [
    401,
    405,
    413,
    420,
    435,
    449,
    456,
    478,
    485,
    497,
    509,
    521,
    537,
    549,
    562,
]
# Language programs at a few schools (program_type -> school ids).
LANGUAGE_PROGRAMS = {
    "SN": [405, 449, 521],
    "CB": [413, 497, 562],
}
ETHNICITIES = [
    "Chinese",
    "Hispanic/Latino",
    "White",
    "Black or African American",
    "Filipino",
    "Pacific Islander",
    "Two or More Races",
    "Decline to State",
]
HOMELANGS = ["English", "SP-Spanish", "CC-Chinese Cantonese"]

# Exact column order of the real student_<year>_filtered.csv files.
STUDENT_COLUMNS = [
    "studentno",
    "r1_ranked_idschool",
    "r1_listed_ranks",
    "r1_programs",
    "grade",
    "r1_randomnumber",
    "r1_cohortstring",
    "bayview_to_all_ms",
    "brown_ms_to_hs",
    "bayview_to_brown_ms",
    "r1_designation_randomnumber",
    "requestprogramdesignation",
    "latitude",
    "longitude",
    "previous_pathway",
    "msf",
    "r2_ranked_idschool",
    "r2_listed_ranks",
    "r2_programs",
    "r2_randomnumber",
    "r2_cohortstring",
    "r2_designation_randomnumber",
    "r1_idschool",
    "r1_programcode",
    "r1_rank",
    "r1_isdesignation",
    "r1_distance",
    "ctip1",
    "idschoolattendance",
    "r2_idschool",
    "r2_programcode",
    "r2_rank",
    "r2_isdesignation",
    "r2_distance",
    "enrolled_idschool",
    "englprof",
    "sped",
    "resolved_ethnicity",
    "homelang",
    "math_scalescore",
    "ela_scalescore",
    "enrolled_pathway",
    "final_school",
    "num_ranked",
    "census_block",
    "freelunch_prob",
    "reducedlunch_prob",
    "census_blockgroup",
    "census_tract",
    "FRL Score",
    "N'hood SES Score",
    "Academic Score",
    "AALPI Score",
    "HOCidx1",
    "sibling",
    "currentlpsibling",
    "currentlp",
    "aaprek",
    "prek",
    "aa",
    "zipcode",
    "median_hh_income",
    "lowell_ranked",
    "sota_ranked",
]


def _make_schools(rng: np.random.Generator) -> pd.DataFrame:
    """Build the schools_rehauled-style schools table.

    Args:
        rng (np.random.Generator): Seeded random generator.

    Returns:
        pd.DataFrame: One row per school with the real schema; school_id is
            the first column (the loader reads it with index_col=0).
    """
    num_schools = len(SCHOOL_IDS)
    latitudes = rng.uniform(37.715, 37.795, num_schools).round(6)
    longitudes = rng.uniform(-122.505, -122.395, num_schools).round(6)
    blocks = [60750100000000 + 1000 * i + 1 for i in range(num_schools)]
    block_groups = [60750100000 + 100 * i + 1 for i in range(num_schools)]
    tracts = [6075010000 + 10 * i + 1 for i in range(num_schools)]

    return pd.DataFrame(
        {
            "school_id": SCHOOL_IDS,
            "school_name": [f"Fake School {sid}" for sid in SCHOOL_IDS],
            "school_name_long": [
                f"Fake Elementary School {sid}" for sid in SCHOOL_IDS
            ],
            "lat": latitudes,
            "lon": longitudes,
            "zip": rng.integers(94102, 94135, num_schools),
            "category": "Attendance",
            "grades": "KG-5",
            "greatschools_rating": rng.integers(1, 11, num_schools),
            "ela_color": "Green",
            "math_color": "Green",
            "chronic_color": "Green",
            "suspension_color": "Green",
            "index": range(num_schools),
            "Block": blocks,
            "BlockGroup": block_groups,
            "Tract": tracts,
        }
    )


def _make_programs(rng: np.random.Generator) -> pd.DataFrame:
    """Build the programs_without_specialprogs-style programs table.

    Every school gets a GE program; a few get SN / CB language programs.

    Args:
        rng (np.random.Generator): Seeded random generator.

    Returns:
        pd.DataFrame: One row per program with the real schema and
            contiguous 1-indexed programno.
    """
    rows: list[dict] = []
    for school_id in SCHOOL_IDS:
        rows.append({"school_id": school_id, "program_type": "GE"})
    for program_type, school_list in LANGUAGE_PROGRAMS.items():
        for school_id in school_list:
            rows.append({"school_id": school_id, "program_type": program_type})

    program_df = pd.DataFrame(rows)
    program_df["program_id"] = (
        program_df["school_id"].astype(str)
        + "-"
        + program_df["program_type"]
        + "-"
        + GRADE
    )
    program_df["capacity"] = rng.integers(8, 19, len(program_df))
    program_df["programno"] = range(1, len(program_df) + 1)
    program_df["r2_capacity"] = program_df["capacity"]
    program_df["r1_assigned"] = rng.integers(4, 15, len(program_df))
    program_df["r1_noenroll"] = rng.integers(0, 4, len(program_df))
    program_df["r1_first_choice"] = rng.integers(0, 25, len(program_df))
    program_df.insert(0, "Unnamed: 0", range(len(program_df)))
    return program_df[
        [
            "Unnamed: 0",
            "program_id",
            "school_id",
            "program_type",
            "capacity",
            "programno",
            "r2_capacity",
            "r1_assigned",
            "r1_noenroll",
            "r1_first_choice",
        ]
    ]


def _rank_programs_for_student(
    rng: np.random.Generator,
    program_df: pd.DataFrame,
    homelang: str,
) -> tuple[list[int], list[str]]:
    """Sample a ranked list of programs for one student.

    Students rank 3-8 programs. Language programs are mostly ranked by
    students whose home language matches.

    Args:
        rng (np.random.Generator): Seeded random generator.
        program_df (pd.DataFrame): The fake programs table.
        homelang (str): The student's home language.

    Returns:
        Tuple[List[int], List[str]]: Ranked school ids and program types.
    """
    ge_programs = program_df[program_df["program_type"] == "GE"]
    lang_type = {"SP-Spanish": "SN", "CC-Chinese Cantonese": "CB"}.get(homelang)

    list_length = int(rng.integers(3, 9))
    chosen = ge_programs.sample(
        n=list_length, replace=False, random_state=int(rng.integers(0, 2**31))
    )
    schools = chosen["school_id"].tolist()
    types = chosen["program_type"].tolist()

    if lang_type is not None and rng.random() < 0.8:
        lang_programs = program_df[program_df["program_type"] == lang_type]
        lang_pick = lang_programs.sample(
            n=1, random_state=int(rng.integers(0, 2**31))
        )
        # Put the language program first; drop the last GE to keep length.
        schools = lang_pick["school_id"].tolist() + schools[:-1]
        types = lang_pick["program_type"].tolist() + types[:-1]

    return schools, types


def _make_students(
    rng: np.random.Generator,
    schools_df: pd.DataFrame,
    program_df: pd.DataFrame,
    num_students: int,
) -> pd.DataFrame:
    """Build the student_<year>_filtered-style students table.

    Args:
        rng (np.random.Generator): Seeded random generator.
        schools_df (pd.DataFrame): The fake schools table.
        program_df (pd.DataFrame): The fake programs table.
        num_students (int): Number of fake students to generate.

    Returns:
        pd.DataFrame: One row per student with the full real column set.
    """
    school_by_id = schools_df.set_index("school_id")
    rows: list[dict] = []
    for i in range(num_students):
        studentno = 100000 + i
        home_school = int(rng.choice(SCHOOL_IDS))
        home = school_by_id.loc[home_school]
        homelang = str(rng.choice(HOMELANGS, p=[0.7, 0.15, 0.15]))
        ranked_schools, ranked_types = _rank_programs_for_student(
            rng, program_df, homelang
        )
        num_ranked = len(ranked_schools)
        random_numbers = rng.uniform(0, 1, num_ranked).round(8).tolist()

        median_income = int(rng.integers(40_000, 200_000))
        sibling = (
            f"[{int(rng.choice(SCHOOL_IDS))}]" if rng.random() < 0.1 else "[]"
        )

        rows.append(
            {
                "studentno": studentno,
                "r1_ranked_idschool": str(ranked_schools),
                "r1_listed_ranks": str(list(range(1, num_ranked + 1))),
                "r1_programs": str(ranked_types),
                "grade": GRADE,
                "r1_randomnumber": str(random_numbers),
                "r1_cohortstring": "[]",
                "bayview_to_all_ms": 0,
                "brown_ms_to_hs": 0,
                "bayview_to_brown_ms": 0,
                "r1_designation_randomnumber": float(rng.uniform(0, 1)),
                "requestprogramdesignation": int(rng.random() < 0.1),
                "latitude": float(home["lat"])
                + float(rng.uniform(-0.008, 0.008)),
                "longitude": float(home["lon"])
                + float(rng.uniform(-0.008, 0.008)),
                "previous_pathway": "",
                "msf": np.nan,
                "r2_ranked_idschool": "[]",
                "r2_listed_ranks": "[]",
                "r2_programs": "[]",
                "r2_randomnumber": "[]",
                "r2_cohortstring": "[]",
                "r2_designation_randomnumber": np.nan,
                "r1_idschool": ranked_schools[0],
                "r1_programcode": ranked_types[0],
                "r1_rank": 1,
                "r1_isdesignation": 0,
                "r1_distance": np.nan,
                "ctip1": int(rng.random() < 0.15),
                "idschoolattendance": home_school,
                "r2_idschool": np.nan,
                "r2_programcode": np.nan,
                "r2_rank": np.nan,
                "r2_isdesignation": np.nan,
                "r2_distance": np.nan,
                "enrolled_idschool": ranked_schools[0],
                "englprof": str(rng.choice(["EO", "EL", "RFEP"])),
                "sped": 0,
                "resolved_ethnicity": str(rng.choice(ETHNICITIES)),
                "homelang": homelang,
                "math_scalescore": np.nan,
                "ela_scalescore": np.nan,
                "enrolled_pathway": "",
                "final_school": ranked_schools[0],
                "num_ranked": num_ranked,
                "census_block": int(home["Block"]),
                "freelunch_prob": round(float(rng.uniform(0, 1)), 4),
                "reducedlunch_prob": round(float(rng.uniform(0, 0.3)), 4),
                "census_blockgroup": int(home["BlockGroup"]),
                "census_tract": int(home["Tract"]),
                "FRL Score": round(float(rng.uniform(0, 1)), 4),
                "N'hood SES Score": round(float(rng.uniform(0, 1)), 4),
                "Academic Score": round(float(rng.uniform(0, 1)), 4),
                "AALPI Score": round(float(rng.uniform(0, 1)), 4),
                "HOCidx1": round(float(rng.uniform(0, 1)), 4),
                "sibling": sibling,
                "currentlpsibling": "[]",
                "currentlp": "",
                "aaprek": "[]",
                "prek": "[]",
                "aa": home_school,
                "zipcode": int(rng.integers(94102, 94135)),
                "median_hh_income": median_income,
                "lowell_ranked": 0,
                "sota_ranked": 0,
            }
        )
    return pd.DataFrame(rows)[STUDENT_COLUMNS]


def _make_estimates(
    rng: np.random.Generator,
    students_df: pd.DataFrame,
    program_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a fake estimates matrix (utilities per student x program).

    Args:
        rng (np.random.Generator): Seeded random generator.
        students_df (pd.DataFrame): The fake students table.
        program_df (pd.DataFrame): The fake programs table.

    Returns:
        pd.DataFrame: studentno (as "<year>-<no>") x program_id utilities,
            with ~5% -inf entries (ineligible programs).
    """
    program_ids = program_df["program_id"].tolist()
    utilities = rng.normal(0.0, 1.5, (len(students_df), len(program_ids)))
    ineligible_mask = rng.random(utilities.shape) < 0.05
    utilities[ineligible_mask] = -np.inf

    estimates_df = pd.DataFrame(utilities.round(6), columns=program_ids)
    estimates_df.insert(
        0,
        "studentno",
        [
            f"{FAKE_YEAR_PREFIX}-{studentno}"
            for studentno in students_df["studentno"]
        ],
    )
    return estimates_df


def _make_zone_rows(num_zones: int = 3) -> list[list[int]]:
    """Partition the fake schools into zones (Con1-style zone file rows).

    Args:
        num_zones (int): Number of zones to create.

    Returns:
        List[List[int]]: One list of attendance-area school ids per zone.
    """
    return [SCHOOL_IDS[i::num_zones] for i in range(num_zones)]


def generate_dataset(
    out_dir: Path, num_students: int, seed: int = 20260609
) -> None:
    """Generate and save the full fake dataset.

    Args:
        out_dir (Path): Output directory root.
        num_students (int): Number of fake students.
        seed (int): Random seed for deterministic output.
    """
    rng = np.random.default_rng(seed)

    schools_df = _make_schools(rng)
    program_df = _make_programs(rng)
    students_df = _make_students(rng, schools_df, program_df, num_students)
    estimates_df = _make_estimates(rng, students_df, program_df)
    zone_rows = _make_zone_rows()

    cleaned_dir = out_dir / "Cleaned"
    model_dir = out_dir / "models" / FAKE_MODEL_NAME
    zones_dir = out_dir / "zones"
    for directory in (out_dir, cleaned_dir, model_dir, zones_dir):
        directory.mkdir(parents=True, exist_ok=True)

    students_path = out_dir / f"student_{FAKE_YEAR_PREFIX}_filtered.csv"
    programs_path = (
        out_dir / f"programs_without_specialprogs_{FAKE_YEAR_PREFIX}.csv"
    )
    schools_path = cleaned_dir / f"schools_rehauled_{FAKE_YEAR_PREFIX}.csv"
    estimates_path = model_dir / f"estimates_{FAKE_YEAR_PREFIX}.csv"
    zones_path = zones_dir / "concept1zones.csv"

    students_df.to_csv(students_path, index=False)
    program_df.to_csv(programs_path, index=False)
    schools_df.to_csv(schools_path, index=False)
    estimates_df.to_csv(estimates_path, index=False)
    with open(zones_path, "w") as zone_file:
        for zone in zone_rows:
            zone_file.write(",".join(str(x) for x in zone) + "\n")

    logger.info("Wrote %d students to %s", len(students_df), students_path)
    logger.info("Wrote %d programs to %s", len(program_df), programs_path)
    logger.info("Wrote %d schools to %s", len(schools_df), schools_path)
    logger.info("Wrote estimates to %s", estimates_path)
    logger.info("Wrote %d zones to %s", len(zone_rows), zones_path)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Generate a fully-fake test dataset (year 2223)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures/fake_2223"),
        help="Output directory (default: tests/fixtures/fake_2223)",
    )
    parser.add_argument(
        "--num-students", type=int, default=200, help="Number of fake students"
    )
    parser.add_argument(
        "--seed", type=int, default=20260609, help="Random seed"
    )
    args = parser.parse_args()
    generate_dataset(args.out_dir, args.num_students, args.seed)


if __name__ == "__main__":
    main()
