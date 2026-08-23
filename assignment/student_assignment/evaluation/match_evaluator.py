"""Assignment outcome metrics for benchmarks and standalone trend analysis.

Created 12/11/2022
@author vravoson

``eval_assignment_basic`` is the compact metric set consumed by the benchmark
suite. ``eval_assignment_full`` is the extended standalone report used by
``scripts/analysis/analyze_trends.py``. The latter intentionally remains a
config-and-saved-CSV workflow because not every assignment policy uses zones.
"""

import ast
import csv
import logging
from collections import Counter
from dataclasses import dataclass
from math import asin, cos, isnan, radians, sin, sqrt

import numpy as np
import pandas as pd
from loaders import (
    DataScenario,
    SPECIAL_PROGRAMS,
    load_program_records,
    load_scenario,
    load_school_records,
    load_student_records,
)

from ..choice_ranks import (
    ChoiceRate,
    cumulative_choice_rates,
    listed_preference_rank_matrix,
    normalize_assignment_ranks,
    ranks_for_matches,
)
from ..data_interfaces.programs import Programs
from ..definitions.constants import ZONE_COLORS

logger = logging.getLogger(__name__)

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
DIAGNOSTIC_ETHNICITY_SLUGS = {
    "Asian": "asian",
    "Black": "black",
    "Decline to State": "decline_to_state",
    "Hispanic": "hispanic",
    "Other": "other",
    "Pacific Islander": "pacific_islander",
    "Two or More Races": "two_or_more_races",
    "White": "white",
}
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


@dataclass
class _FullReportAggregates:
    student_data: pd.DataFrame
    assigned_students: pd.DataFrame
    designated_students: pd.DataFrame
    non_designated_students: pd.DataFrame
    student_groups: dict[str, pd.DataFrame]
    school_stats: pd.DataFrame
    non_designated_school_stats: pd.DataFrame
    program_report_stats: pd.DataFrame | None
    non_designated_program_report_stats: pd.DataFrame | None
    designated_program_report_stats: pd.DataFrame | None
    school_frl_context: tuple[float, pd.Series]
    non_designated_school_frl_context: tuple[float, pd.Series]
    school_exposures: dict[str, dict]
    high_income_composition: tuple[pd.Series, float]
    low_income_composition: tuple[pd.Series, float]
    ge_students: pd.DataFrame
    ge_aalpi_students: pd.DataFrame
    ge_frl_context: tuple[pd.Series, float]
    ge_non_designated_students: pd.DataFrame
    ge_non_designated_program_frl_context: tuple[pd.Series, float]
    ge_aapi_counts: np.ndarray
    all_student_top_choice_outcomes: dict[int, ChoiceRate]
    overage: float


class MatchEvaluator:
    def __init__(
        self,
        student_data,
        assignments,
        distances=None,
        *,
        first_round=False,
        dropout=False,
        low_income=95292,
        medium_income=95292,
        high_income=110850,
        year=None,
        grade=None,
        no_special_program=False,
        program_file=None,
        schools_latlon_path=None,
        new_ctip_path=None,
        program_data=None,
        schools_data=None,
        distance_cache=None,
        overscribe_aa=False,
    ):
        """Build an evaluator from benchmark objects or standalone data frames.

        The benchmark passes a ``Students``-like object, an assignment frame
        indexed by ``studentno``, and a wide student-by-program distance frame.
        Standalone analysis passes raw student and assignment data frames plus
        program and school CSV paths.
        """
        self._set_program_groups()
        self.overscribe_aa = overscribe_aa
        if not isinstance(student_data, pd.DataFrame):
            self._init_basic(student_data, assignments, distances)
            return

        self._mode = "full"
        self.student_data = student_data.copy()
        self._validate_student_ids(self.student_data)
        self._distance_cache = distance_cache
        if distance_cache is not None:
            self._distance_cache_values = distance_cache.to_numpy(
                dtype=float, copy=False
            )
            self._distance_cache_rows = {
                self._distance_identity(value): position
                for position, value in enumerate(distance_cache.index)
            }
            self._distance_cache_columns = {
                str(value): position
                for position, value in enumerate(distance_cache.columns)
            }
        self.low_income = low_income
        self.medium_income = medium_income
        self.high_income = high_income
        self.grade = grade
        self.year = year

        if schools_data is not None:
            self.schools_latlon = schools_data.copy()
        elif schools_latlon_path is not None:
            self.schools_latlon = pd.read_csv(schools_latlon_path)
        else:
            raise ValueError(
                "schools_data or schools_latlon_path is required for full evaluation"
            )

        new_ctip_list = np.load(new_ctip_path) if new_ctip_path else np.array([])

        if program_data is not None:
            self.programs = program_data.copy()
        elif program_file is not None:
            self.programs = pd.read_csv(program_file)
        else:
            raise ValueError(
                "program_data or program_file is required for full evaluation"
            )

        if first_round:
            self._parse_student_list_column("r1_ranked_idschool")
            self._parse_student_list_column("r1_programs")
            participating = self.student_data["r1_ranked_idschool"].map(bool)
            self.student_data = self.student_data[participating].copy()
            if self.student_data.empty:
                raise ValueError(
                    "first_round=True requires at least one student with a "
                    "nonempty r1_ranked_idschool list"
                )
        elif no_special_program:
            self._parse_student_list_column("r1_programs")

        if no_special_program:
            if "program_type" not in self.programs:
                raise ValueError(
                    "Program table is missing required column: program_type"
                )
            self.programs = self.programs[
                ~self.programs["program_type"].isin(SPECIAL_PROGRAMS)
            ].copy()
            self.student_data = self.student_data[
                ~self.student_data["r1_programs"].apply(self._has_special_program)
            ].copy()
            if first_round and self.student_data.empty:
                raise ValueError(
                    "No first-round students remain after removing special programs"
                )

        self._prepare_reference_data()
        equity_blocks = set(new_ctip_list.tolist())
        self.student_data["et2"] = (
            self.student_data["census_block"].isin(equity_blocks).astype(int)
        )
        self.student_data["frl"] = pd.to_numeric(
            self.student_data["freelunch_prob"], errors="coerce"
        ).fillna(0) + pd.to_numeric(
            self.student_data["reducedlunch_prob"], errors="coerce"
        ).fillna(0)
        self.student_data["is_frl"] = (self.student_data["frl"] >= 0.5).astype(int)
        self.map_ethnicity()
        self._base_student_data = self.student_data.copy()
        self.update_assignments(assignments)

    def update_assignments(self, assignments):
        """Replace assignment-dependent data without reloading source tables."""
        if self._mode != "full":
            raise ValueError("update_assignments requires a full MatchEvaluator instance")

        self.student_data = self._base_student_data.copy()
        self.assignments = assignments.copy()
        if "designation" not in self.assignments:
            self.assignments["designation"] = 0
        self._validate_full_assignments()
        self.student_data = self.student_data.merge(
            self.assignments,
            on="studentno",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        self.student_data["assignment"] = self.student_data["programcodes"].mask(
            self.student_data["programcodes"].str.strip() == "", "000-ZZ"
        )
        self.student_data["assigned_school"] = (
            self.student_data["programcodes"]
            .map(self._program_school_by_id)
            .fillna(0)
            .astype(int)
        )
        self.student_data["programtype"] = self.student_data["programcodes"].map(
            self._program_type_by_id
        )
        self.student_data["assigned school"] = self.student_data["assigned_school"]
        self._mark_overage_seats()
        if self._distance_cache is None:
            self.eval_distance()
        else:
            self.student_data["assignment_dist"] = (
                self._assignment_distances_from_cache()
            )
        return self

    def _mark_overage_seats(self):
        self.student_data["overage_seat"] = False
        if not self.overscribe_aa:
            return
        if "capacity" not in self.programs:
            raise ValueError("Overage metrics require program column: capacity")

        assigned = self._assigned_mask(self.student_data) & self.student_data[
            "programtype"
        ].eq("GE")
        if "overage_seat" in self.assignments:
            overage_seats = pd.to_numeric(
                self.assignments["overage_seat"], errors="coerce"
            )
            if overage_seats.isna().any() or not overage_seats.isin([0, 1]).all():
                raise ValueError("Assignments contain invalid overage_seat values")
            overage_by_student = pd.Series(
                overage_seats.astype(bool).to_numpy(),
                index=self.assignments["studentno"],
            )
            marked = self.student_data["studentno"].map(overage_by_student)
            if (marked & ~assigned).any():
                raise ValueError(
                    "Only students assigned to GE programs can occupy overage seats"
                )
            self.student_data["overage_seat"] = marked.to_numpy(dtype=bool)
            return

        assigned_students = self.student_data.loc[assigned]
        seat_numbers = (
            assigned_students.groupby("assignment", sort=False).cumcount() + 1
        )
        capacities = assigned_students["assignment"].map(
            self.programs.set_index("program_id")["capacity"]
        )
        self.student_data.loc[assigned, "overage_seat"] = (
            seat_numbers.to_numpy() > capacities.to_numpy()
        )

    @classmethod
    def from_scenario(
        cls,
        data: DataScenario | dict,
        assignments: pd.DataFrame,
        *,
        program_data: pd.DataFrame | None = None,
        **kwargs,
    ):
        """Build a full evaluator from scenario-normalized shared tables."""
        scenario = data if isinstance(data, DataScenario) else load_scenario(data)
        conflicting_options = [
            option
            for option in ("first_round", "no_special_program")
            if kwargs.pop(option, False)
        ]
        if conflicting_options:
            raise ValueError(
                "Scenario evaluation population is controlled by assignment "
                "filters, not evaluator options: "
                f"{conflicting_options}."
            )
        students = load_student_records(
            scenario,
            "assignment.students",
            filter_group="assignment",
            low_memory=False,
        )
        required_student_columns = {
            "studentno",
            "census_block",
            "latitude",
            "longitude",
            "freelunch_prob",
            "reducedlunch_prob",
            "resolved_ethnicity",
        }
        missing_student_columns = required_student_columns - set(students.columns)
        if missing_student_columns:
            raise ValueError(
                "Full assignment evaluation requires student columns: "
                f"{sorted(missing_student_columns)}"
            )
        programs = (
            program_data.copy()
            if program_data is not None
            else load_program_records(
                scenario,
                "assignment.programs",
                filter_group="assignment",
            )
        )
        school_metadata = load_school_records(
            scenario,
            "assignment.schools",
            filter_group="assignment",
        )
        school_locations = load_school_records(
            scenario,
            "assignment.school_coordinates",
            filter_group="assignment",
        )
        schools = school_metadata.drop(columns=["lat", "lon"], errors="ignore").merge(
            school_locations[["school_id", "lat", "lon"]].drop_duplicates(
                "school_id"
            ),
            on="school_id",
            how="left",
        )
        year = scenario.filter("assignment", "year")
        grade = scenario.filter("assignment", "grades")[0]
        kwargs.setdefault("year", int(year[:2]))
        kwargs.setdefault("grade", grade)
        if kwargs.get("new_ctip_path") is None:
            try:
                new_ctip_path = scenario.source("assignment.new_ctip").path
            except KeyError:
                pass
            else:
                if new_ctip_path.is_file():
                    kwargs["new_ctip_path"] = new_ctip_path
        return cls(
            students,
            assignments,
            program_data=programs,
            schools_data=schools,
            **kwargs,
        )

    def _set_program_groups(self):
        self.immersion_programs = [
            "CE",
            "CN",
            "CT",
            "JE",
            "JN",
            "KE",
            "KN",
            "ME",
            "MN",
            "SE",
            "SN",
        ]
        self.non_immersion_programs = ["CB", "FB", "NC", "NS", "SB"]
        self.ge = ["GE"]
        self.ce_se = ["CN", "CE", "CB", "SN", "SB", "SE"]
        self.ge_immer_notimmer = (
            self.immersion_programs + self.non_immersion_programs + self.ge
        )

    @staticmethod
    def _validate_student_ids(student_data):
        if "studentno" not in student_data:
            raise ValueError("Student data is missing required column: studentno")
        if student_data["studentno"].isna().any():
            raise ValueError("Student data contains a missing studentno")
        duplicates = student_data.loc[
            student_data["studentno"].duplicated(keep=False), "studentno"
        ].unique()
        if len(duplicates):
            raise ValueError(
                "Student data contains duplicate studentno values: "
                f"{duplicates.tolist()}"
            )

    @staticmethod
    def _parse_serialized_list(value, column):
        if isinstance(value, list):
            return value.copy()
        if isinstance(value, (tuple, np.ndarray)):
            return list(value)
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            return []
        if not isinstance(value, str):
            raise ValueError(f"{column} must contain lists, got {value!r}")
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"{column} contains an invalid serialized list: {value!r}"
            ) from exc
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"{column} must contain lists, got {value!r}")
        return list(parsed)

    def _parse_student_list_column(self, column):
        if column not in self.student_data:
            raise ValueError(f"Student data is missing required column: {column}")

        parsed = []
        for studentno, value in zip(
            self.student_data["studentno"], self.student_data[column]
        ):
            try:
                parsed.append(self._parse_serialized_list(value, column))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {column} for student {studentno}: {exc}"
                ) from exc
        self.student_data[column] = parsed

    def _prepare_reference_data(self):
        required_program_columns = {
            "program_id",
            "programno",
            "school_id",
            "program_type",
        }
        missing_program_columns = required_program_columns - set(self.programs.columns)
        if missing_program_columns:
            raise ValueError(
                "Program table is missing required columns: "
                f"{sorted(missing_program_columns)}"
            )

        program_ids = self.programs["program_id"].astype("string").str.strip()
        if (program_ids.isna() | program_ids.eq("").fillna(False)).any():
            raise ValueError("Program table contains a missing program_id")
        if program_ids.duplicated().any():
            duplicates = program_ids[program_ids.duplicated(keep=False)].unique()
            raise ValueError(
                f"Program table contains duplicate program_id values: {duplicates.tolist()}"
            )

        program_numbers = Programs.normalized_programnos(self.programs, "Program table")

        program_types = self.programs["program_type"].astype("string").str.strip()
        if (program_types.isna() | program_types.eq("").fillna(False)).any():
            raise ValueError("Program table contains a missing program_type")

        program_schools = pd.to_numeric(self.programs["school_id"], errors="coerce")
        invalid_program_schools = (
            program_schools.isna()
            | ~np.isfinite(program_schools)
            | (program_schools <= 0)
            | (program_schools % 1 != 0)
        )
        if invalid_program_schools.any():
            raise ValueError("Program table contains invalid school_id values")

        required_school_columns = {"school_id", "lat", "lon"}
        missing_school_columns = required_school_columns - set(
            self.schools_latlon.columns
        )
        if missing_school_columns:
            raise ValueError(
                "School table is missing required columns: "
                f"{sorted(missing_school_columns)}"
            )
        school_ids = pd.to_numeric(self.schools_latlon["school_id"], errors="coerce")
        invalid_school_ids = (
            school_ids.isna()
            | ~np.isfinite(school_ids)
            | (school_ids <= 0)
            | (school_ids % 1 != 0)
        )
        if invalid_school_ids.any():
            raise ValueError("School table contains invalid school_id values")

        self.programs["program_id"] = program_ids.astype(str)
        self.programs["programno"] = program_numbers.astype(int)
        self.programs["school_id"] = program_schools.astype(int)
        self.programs["program_type"] = program_types.astype(str)
        if "capacity" in self.programs:
            capacities = pd.to_numeric(self.programs["capacity"], errors="coerce")
            invalid_capacities = (
                capacities.isna() | ~np.isfinite(capacities) | (capacities < 0)
            )
            if invalid_capacities.any():
                raise ValueError("Program table contains invalid capacity values")
            self.programs["capacity"] = capacities
        self.schools_latlon["school_id"] = school_ids.astype(int)
        self.schools_latlon["lat"] = pd.to_numeric(
            self.schools_latlon["lat"], errors="coerce"
        )
        self.schools_latlon["lon"] = pd.to_numeric(
            self.schools_latlon["lon"], errors="coerce"
        )

        self._program_number_by_id = self.programs.set_index("program_id")[
            "programno"
        ].to_dict()
        self._program_school_by_id = self.programs.set_index("program_id")[
            "school_id"
        ].to_dict()
        self._program_type_by_id = self.programs.set_index("program_id")[
            "program_type"
        ].to_dict()
        if "capacity" in self.programs:
            ge_programs = self.programs[self.programs["program_type"] == "GE"]
            self._ge_program_ids = ge_programs["program_id"].astype(str).to_numpy()
            self._ge_program_capacities = ge_programs["capacity"].to_numpy(dtype=float)
        else:
            self._ge_program_ids = np.array([], dtype=str)
            self._ge_program_capacities = np.array([], dtype=float)
        valid_locations = (
            self.schools_latlon[["lat", "lon"]].notna().all(axis=1)
            & np.isfinite(self.schools_latlon["lat"])
            & np.isfinite(self.schools_latlon["lon"])
        )
        self._schools_with_locations = set(
            self.schools_latlon.loc[valid_locations, "school_id"].tolist()
        )

    def _validate_full_assignments(self):
        required_columns = {"studentno", "programno", "programcodes"}
        missing_columns = required_columns - set(self.assignments.columns)
        if missing_columns:
            raise ValueError(
                f"Assignments are missing required columns: {sorted(missing_columns)}"
            )
        if self.assignments["studentno"].isna().any():
            raise ValueError("Assignments contain a missing studentno")
        duplicate_students = self.assignments.loc[
            self.assignments["studentno"].duplicated(keep=False), "studentno"
        ].unique()
        if len(duplicate_students):
            raise ValueError(
                "Assignments contain duplicate studentno values: "
                f"{duplicate_students.tolist()}"
            )

        expected_ids = pd.Index(self.student_data["studentno"])
        assignment_ids = pd.Index(self.assignments["studentno"])
        missing_ids = expected_ids[~expected_ids.isin(assignment_ids)].tolist()
        extra_ids = assignment_ids[~assignment_ids.isin(expected_ids)].tolist()
        if missing_ids or extra_ids:
            raise ValueError(
                "Assignments must contain exactly one row per retained student; "
                f"missing studentno values: {missing_ids}; "
                f"extra studentno values: {extra_ids}"
            )

        program_numbers = pd.to_numeric(self.assignments["programno"], errors="coerce")
        invalid_program_numbers = (
            program_numbers.isna()
            | ~np.isfinite(program_numbers)
            | (program_numbers < 0)
            | (program_numbers % 1 != 0)
        )
        if invalid_program_numbers.any():
            students = self.assignments.loc[
                invalid_program_numbers, "studentno"
            ].tolist()
            raise ValueError(
                f"Assignments contain invalid programno values for students: {students}"
            )
        self.assignments["programno"] = program_numbers.astype(int)

        program_codes = self.assignments["programcodes"].astype("string").str.strip()
        positive = self.assignments["programno"] > 0
        blank_codes = program_codes.isna() | program_codes.eq("").fillna(False)
        if (positive & blank_codes).any():
            students = self.assignments.loc[
                positive & blank_codes, "studentno"
            ].tolist()
            raise ValueError(
                "Positive programno assignments require programcodes for students: "
                f"{students}"
            )
        if ((~positive) & ~blank_codes).any():
            students = self.assignments.loc[
                (~positive) & ~blank_codes, "studentno"
            ].tolist()
            raise ValueError(
                f"Unassigned students must have blank programcodes: {students}"
            )

        assigned_codes = program_codes[positive]
        known_programs = assigned_codes.isin(self._program_number_by_id)
        if not known_programs.all():
            unknown = assigned_codes[~known_programs].unique().tolist()
            raise ValueError(f"Assignments reference unknown program IDs: {unknown}")

        expected_program_numbers = assigned_codes.map(self._program_number_by_id)
        mismatched_numbers = expected_program_numbers.ne(
            self.assignments.loc[positive, "programno"]
        )
        if mismatched_numbers.any():
            students = self.assignments.loc[
                expected_program_numbers.index[mismatched_numbers], "studentno"
            ].tolist()
            raise ValueError(
                f"programno does not match programcodes for students: {students}"
            )

        expected_schools = assigned_codes.map(self._program_school_by_id)
        missing_locations = sorted(
            set(expected_schools.astype(int)) - self._schools_with_locations
        )
        if missing_locations:
            raise ValueError(
                "Assigned programs reference schools without known locations: "
                f"{missing_locations}"
            )

        self.assignments = normalize_assignment_ranks(
            self.assignments,
            listed_ranks=self._full_source_submitted_ranks(),
        )

        raw_designation = self.assignments["designation"]
        designation = pd.to_numeric(raw_designation, errors="coerce")
        supplied_designation = raw_designation.notna() & raw_designation.astype(
            "string"
        ).str.strip().ne("")
        invalid_designation = supplied_designation & (
            designation.isna() | ~np.isfinite(designation) | ~designation.isin([0, 1])
        )
        if invalid_designation.any():
            students = self.assignments.loc[invalid_designation, "studentno"].tolist()
            raise ValueError(
                f"Assignments contain invalid designation values for students: {students}"
            )
        designation = designation.fillna(0).astype(int)
        unassigned_designated = self.assignments["programno"].eq(0) & designation.eq(1)
        if unassigned_designated.any():
            students = self.assignments.loc[unassigned_designated, "studentno"].tolist()
            raise ValueError(f"Unassigned students cannot be designated: {students}")
        self.assignments["designation"] = designation
        self.assignments["programcodes"] = program_codes.fillna("").astype(str)

    def _full_source_submitted_ranks(self) -> pd.Series | None:
        columns = set(self.student_data.columns)
        has_selected = {
            "selected_ranked_idschool",
            "selected_programs",
        } <= columns
        has_first_round = {"r1_ranked_idschool", "r1_programs"} <= columns
        if not has_selected and not has_first_round:
            return None

        rank_matrix = listed_preference_rank_matrix(
            self.student_data,
            self._program_number_by_id,
        )
        matches = (
            self.assignments.set_index("studentno")["programno"]
            .reindex(self.student_data["studentno"])
            .to_numpy()
        )
        ranks = ranks_for_matches(rank_matrix, matches)
        rank_by_student = pd.Series(ranks, index=self.student_data["studentno"])
        return self.assignments["studentno"].map(rank_by_student).set_axis(
            self.assignments.index
        )

    @staticmethod
    def _has_special_program(value):
        if not isinstance(value, (list, tuple, np.ndarray)):
            programs = MatchEvaluator._parse_serialized_list(value, "r1_programs")
        else:
            programs = value
        return bool(set(programs).intersection(SPECIAL_PROGRAMS))

    def _init_basic(self, students, assignments, distances):
        if not hasattr(students, "student_data"):
            raise TypeError(
                "student_data must be a DataFrame or a Students-like object"
            )
        self._mode = "basic"
        self.students = students
        self.distance_data = distances
        self.assignments = assignments.copy()
        if "designation" not in self.assignments:
            self.assignments["designation"] = 0
        self._validate_basic_assignment_values()
        listed_ranks = None
        if hasattr(students, "selected_preference_rank_matrix"):
            rank_matrix = students.selected_preference_rank_matrix()
            student_ids = students.student_data.index
            matches = (
                self.assignments["programno"].reindex(student_ids).fillna(0).to_numpy()
            )
            rank_by_student = pd.Series(
                ranks_for_matches(rank_matrix, matches),
                index=student_ids,
            )
            listed_ranks = pd.Series(
                self.assignments.index.map(rank_by_student),
                index=self.assignments.index,
                dtype=float,
            )
        self.assignments = normalize_assignment_ranks(
            self.assignments,
            listed_ranks=listed_ranks,
        )
        if "assigned_utility" in self.assignments:
            self.assignments["assigned_utility"] = self.assignments[
                "assigned_utility"
            ].replace(-np.inf, np.nan)

        base = students.student_data.copy()
        if base.index.name != "studentno":
            base.index.name = "studentno"
        self.student_data = base.join(self.assignments, how="left")
        self.student_data.rename(columns={"programcodes": "assignment"}, inplace=True)
        for column, default in {
            "assignment": pd.NA,
            "programno": 0,
            "designation": 0,
            "freelunch_prob": 0.0,
            "reducedlunch_prob": 0.0,
            "resolved_ethnicity": "",
            "SES_category": np.nan,
            "census_blockgroup": np.nan,
        }.items():
            if column not in self.student_data:
                self.student_data[column] = default

        self.student_data["programtype"] = self.student_data["assignment"].str[4:6]
        self.student_data["assigned school"] = self.student_data["assignment"].str[:3]
        self.student_data["frl"] = pd.to_numeric(
            self.student_data["freelunch_prob"], errors="coerce"
        ).fillna(0) + pd.to_numeric(
            self.student_data["reducedlunch_prob"], errors="coerce"
        ).fillna(0)
        if self.student_data["SES_category"].isna().all():
            self.student_data["SES_category"] = self._derive_ses_category(
                self.student_data
            )
        self.student_data["assignment_dist"] = self._assigned_distances(distances)
        self.match_ranks = self.assignments["rank"]
        self.num_students = len(self.student_data)
        self.num_schools = self.student_data["assigned school"].nunique()

    def _validate_basic_assignment_values(self):
        if "programno" not in self.assignments:
            raise ValueError("Assignments are missing required column: programno")
        programno = pd.to_numeric(self.assignments["programno"], errors="coerce")
        invalid_programno = (
            programno.isna()
            | ~np.isfinite(programno)
            | programno.lt(0)
            | programno.mod(1).ne(0)
        )
        if invalid_programno.any():
            raise ValueError("Assignments contain invalid programno values.")
        self.assignments["programno"] = programno.astype(int)

        designation = pd.to_numeric(
            self.assignments["designation"], errors="coerce"
        )
        invalid_designation = (
            designation.isna()
            | ~np.isfinite(designation)
            | ~designation.isin([0, 1])
        )
        if invalid_designation.any():
            raise ValueError("Assignments contain invalid designation values.")
        assigned = self.assignments["programno"].gt(0)
        if ((~assigned) & designation.eq(1)).any():
            raise ValueError("Unassigned students cannot be designated.")
        self.assignments["designation"] = designation.astype(int)

    def _assigned_distances(self, distances):
        if distances is None or distances.empty:
            return pd.Series(np.nan, index=self.student_data.index, dtype=float)

        def lookup(row):
            program = row["assignment"]
            if pd.isna(program) or program not in distances.columns:
                return np.nan
            studentno = row.name
            if studentno not in distances.index:
                return np.nan
            return distances.at[studentno, program]

        return self.student_data.apply(lookup, axis=1)

    @staticmethod
    def _derive_ses_category(student_data):
        if {
            "N'hood SES Score",
            "FRL Score",
        } <= set(student_data.columns):
            score = 0.25 * pd.to_numeric(
                student_data["N'hood SES Score"], errors="coerce"
            ) + 0.25 * pd.to_numeric(student_data["FRL Score"], errors="coerce")
        else:
            score = pd.to_numeric(student_data["frl"], errors="coerce")
        non_null = score.dropna()
        if non_null.empty:
            return pd.Series(1, index=student_data.index, dtype=int)
        lower, upper = np.percentile(non_null, [33, 66])
        return score.apply(
            lambda value: 1 if value < lower else (2 if value < upper else 3)
        )

    def eval_assignment_basic(self):
        """Return the compact assignment metrics consumed by benchmarks.

        These names and formulas intentionally preserve the existing benchmark
        contract. In particular, ``Programs with 1-4 AA`` remains the legacy
        count of schools with one to four Black students; the full evaluator
        has the correctly scoped Black-or-Pacific-Islander GE-program metric.
        """
        metrics = {}
        student_data = self.student_data
        assigned_students = student_data[self._assigned_mask(student_data)].copy()
        enrollment = assigned_students.groupby("assigned school").size()

        metrics.update(self._basic_distance_metrics(assigned_students))
        metrics["Schools above 10% district FRL"] = (
            self.metric_school_frl_above_district(0.1)
        )
        metrics["Schools above 15% district FRL"] = (
            self.metric_school_frl_above_district(0.15)
        )

        aalpi = [
            "Black or African American",
            "Hispanic/Latino",
            "Hispanic/Latinx",
            "Pacific Islander",
        ]
        aalpi_students = assigned_students[
            assigned_students["resolved_ethnicity"].isin(aalpi)
        ]
        metrics["AALPI in school with +10% FRL"] = self.metric_FRL_concentration(
            assigned_students, aalpi_students, 0.1
        )
        metrics["AALPI in school with +15% FRL"] = self.metric_FRL_concentration(
            assigned_students, aalpi_students, 0.15
        )
        metrics["Dissimilarity AALPI"] = self._basic_dissimilarity(
            aalpi_students, enrollment
        )
        ses3_students = assigned_students[assigned_students["SES_category"] == 3]
        metrics["Dissimilarity SES3"] = self._basic_dissimilarity(
            ses3_students, enrollment
        )
        black_students = assigned_students[
            assigned_students["resolved_ethnicity"] == "Black or African American"
        ]
        metrics["Programs with 1-4 AA"] = self.metric_isolation(black_students, 5)
        metrics["# Racial majority schools"] = self.metric_racial_majority_schools(
            assigned_students
        )

        metrics.update(self._basic_choice_metrics(student_data, assigned_students))
        metrics["BG Cohesion (3)"] = self.metric_BG_cohesion(assigned_students, 3)

        high_frl_assigned = assigned_students[assigned_students["frl"] > 0.5]
        low_frl_assigned = assigned_students[assigned_students["frl"] <= 0.5]
        groups = {
            "Black or African American": assigned_students[
                assigned_students["resolved_ethnicity"] == "Black or African American"
            ],
            "Asian": assigned_students[
                assigned_students["resolved_ethnicity"] == "Asian"
            ],
            "Hispanic/Latino": assigned_students[
                assigned_students["resolved_ethnicity"].isin(
                    ["Hispanic/Latino", "Hispanic/Latinx"]
                )
            ],
            "Pacific Islander": assigned_students[
                assigned_students["resolved_ethnicity"] == "Pacific Islander"
            ],
            "White": assigned_students[
                assigned_students["resolved_ethnicity"] == "White"
            ],
            "High FRL": high_frl_assigned,
            "Low FRL": low_frl_assigned,
        }
        for group, students in groups.items():
            self._record_choice_outcome(
                metrics,
                f"Top 3 choice {group}",
                self._top_choice_outcomes(students, "rank", [3])[3],
            )
            metrics[f"Distance Av {group}"] = self.metric_dist_av(students)
            metrics[f"{group} in school with +15% FRL"] = self.metric_FRL_concentration(
                assigned_students, students, 0.15
            )
            metrics[f"{group} Dist >= 3, Rank >= 5"] = self.metric_dist_and_rank(
                students, 3, 5
            ).mean()

        return pd.Series(metrics)

    def _basic_distance_metrics(self, assigned_students):
        return {
            "Distance Av": self.metric_dist_av(assigned_students),
            "Distance < 0.5": self.metric_dist_threshold(
                assigned_students, 0.5, above=False
            ),
            "Distance > 3": self.metric_dist_threshold(
                assigned_students, 3, above=True
            ),
        }

    def _basic_choice_metrics(self, student_data, assigned_students):
        metrics = {
            "Unassigned": self.metric_unassigned(student_data),
            "Designated": self.metric_designated(assigned_students),
            "Dist >= 3, Rank >= 5": self.metric_dist_and_rank(
                assigned_students, 3, 5
            ).mean(),
            **self._utility_metrics(student_data),
        }
        choice_outcomes = self._top_choice_outcomes(
            assigned_students, "rank", [1, 3]
        )
        mechanism_outcomes = self._top_choice_outcomes(
            assigned_students, "mechanism_rank", [1, 3]
        )
        for threshold in (1, 3):
            self._record_choice_outcome(
                metrics,
                f"Top {threshold} choice",
                choice_outcomes[threshold],
            )
            self._record_choice_outcome(
                metrics,
                f"Top {threshold} in-zone choice",
                mechanism_outcomes[threshold],
            )
        return metrics

    @classmethod
    def _utility_metrics(cls, student_data):
        if "assigned_utility" not in student_data:
            return {"Total Utility": np.nan, "Average Utility": np.nan}

        utilities = pd.to_numeric(student_data["assigned_utility"], errors="coerce")
        utilities = utilities.where(cls._assigned_mask(student_data), 0.0)
        total_utility = utilities.sum(skipna=False)
        average_utility = (
            total_utility / len(student_data) if len(student_data) else np.nan
        )
        return {
            "Total Utility": total_utility,
            "Average Utility": average_utility,
        }

    @staticmethod
    def _assigned_mask(student_data):
        return pd.to_numeric(student_data["programno"], errors="coerce").fillna(0) > 0

    def metric_dist_av(self, assigned_students):
        return assigned_students["assignment_dist"].mean()

    def metric_dist_threshold(self, assigned_students, threshold, above):
        distances = assigned_students["assignment_dist"]
        return (
            (distances > threshold).mean() if above else (distances < threshold).mean()
        )

    def metric_school_frl_above_district(self, threshold, student_data=None):
        if student_data is None:
            student_data = self.student_data
        district_avg = student_data["frl"].mean()
        school_frl = student_data.groupby("assigned school")["frl"].mean()
        return (school_frl >= district_avg + threshold).mean()

    @staticmethod
    def metric_racial_majority_schools(assigned_students):
        """Count schools where one racial group comprises over half of students."""
        if assigned_students.empty:
            return 0

        race_column = (
            "ethnicity" if "ethnicity" in assigned_students else "resolved_ethnicity"
        )
        race = assigned_students[race_column]
        valid_race = race.notna() & race.astype("string").str.strip().ne("")
        race_counts = (
            assigned_students[valid_race]
            .groupby(["assigned school", race_column])
            .size()
        )
        if race_counts.empty:
            return 0

        school_enrollment = assigned_students.groupby("assigned school").size()
        largest_racial_group = race_counts.groupby(level="assigned school").max()
        racial_share = largest_racial_group / school_enrollment.reindex(
            largest_racial_group.index
        )
        return int((racial_share > 0.5).sum())

    def school_frl_range_district(self, threshold, student_data=None, above=False):
        """Return the share of schools beyond a district FRL threshold."""
        if student_data is None:
            student_data = self.student_data
        district_avg = student_data["frl"].mean()
        school_frl = student_data.groupby("assigned school")["frl"].mean()
        if above:
            return (school_frl >= district_avg + threshold).mean()
        return ((school_frl - district_avg).abs() <= threshold).mean()

    def metric_FRL_concentration(self, all_students, group_students, threshold):
        if group_students.empty:
            return np.nan
        school_frl = all_students.groupby("assigned school")["frl"].mean()
        district_avg = all_students["frl"].mean()
        return (
            group_students["assigned school"].map(school_frl) > district_avg + threshold
        ).mean()

    @staticmethod
    def _basic_dissimilarity(group_students, total_enrollment):
        """Preserve the benchmark's historical one-sided dissimilarity formula."""
        n = len(group_students)
        total_n = pd.to_numeric(total_enrollment, errors="coerce").sum()
        if n == 0 or total_n == 0:
            return np.nan
        ratio = n / total_n
        enrollment = group_students.groupby("assigned school").size()
        total = 0.0
        for index, count in enumerate(enrollment):
            total += abs(count - total_enrollment.iloc[index] * ratio) / 2
        return total / n

    @staticmethod
    def metric_isolation(group_students, threshold):
        enrollment = group_students.groupby("assigned school").size()
        return int(((enrollment >= 1) & (enrollment < threshold)).sum())

    @staticmethod
    def metric_unassigned(students):
        if students.empty:
            return np.nan
        return (
            pd.to_numeric(students["programno"], errors="coerce").fillna(0) == 0
        ).mean()

    @staticmethod
    def metric_designated(assigned_students):
        return pd.to_numeric(assigned_students["designation"], errors="coerce").mean()

    @classmethod
    def metric_top_choice(cls, assigned_students, threshold):
        return cls._top_choice_outcomes(
            assigned_students, "rank", [threshold]
        )[threshold].value

    @staticmethod
    def _top_choice_outcomes(students, rank_column, thresholds):
        return cumulative_choice_rates(students, rank_column, thresholds)

    @staticmethod
    def _record_choice_outcome(metrics, name, outcome):
        metrics[name] = outcome.value
        metrics[f"{name} numerator"] = outcome.numerator
        metrics[f"{name} denominator"] = outcome.denominator

    @classmethod
    def metric_top_in_zone_choice(cls, assigned_students, threshold):
        return cls._top_choice_outcomes(
            assigned_students, "mechanism_rank", [threshold]
        )[threshold].value

    @staticmethod
    def metric_dist_and_rank(assigned_students, distance, rank):
        return (assigned_students["assignment_dist"] >= distance) & (
            assigned_students["rank"] >= rank
        )

    def metric_BG_cohesion(self, assigned_students, count):
        if assigned_students.empty:
            return np.nan
        cohesive = sum(
            self._bgcohesion(group, count)
            for _, group in assigned_students.groupby("census_blockgroup")
        )
        return cohesive / len(assigned_students)

    @staticmethod
    def _bgcohesion(group, count):
        school_counts = group["assigned school"].value_counts()
        return school_counts[school_counts >= count].sum()

    @staticmethod
    def _safe_ratio(numerator, denominator):
        return numerator / denominator if denominator else np.nan

    @staticmethod
    def _array_mean(values):
        return values.mean() if values.size else np.nan

    def _full_proximity_metrics(self, student_data, assigned_students):
        designated = student_data[student_data["designation"] == 1]
        non_designated = student_data[student_data["designation"] == 0]
        metrics = {
            "Total Designated": len(designated),
            "Not designated": len(non_designated),
            "Tot Nb Students (Round 1)": len(student_data),
            "Tot Nb Assigned (Round 1)": len(assigned_students),
            "Tot Nb Designated (Round 1)": int(
                (student_data["designation"] != 0).sum()
            ),
            "Distance Av (All Assigned)": assigned_students["assignment_dist"].mean(),
            "Distance Median (All Assigned)": assigned_students[
                "assignment_dist"
            ].median(),
            "Distance < 0.5 (All Assigned)": (
                assigned_students["assignment_dist"] < 0.5
            ).mean(),
            "Distance < 1 (All Assigned)": (
                assigned_students["assignment_dist"] < 1
            ).mean(),
            "Distance > 3 (All Assigned)": (
                assigned_students["assignment_dist"] > 3
            ).mean(),
        }
        for group in ["Black", "Asian", "Hispanic", "White"]:
            metrics[f" Non-designated {group} students"] = len(
                non_designated[non_designated["ethnicity"] == group]
            )
            metrics[f" Designated {group} students"] = len(
                designated[designated["ethnicity"] == group]
            )
        return metrics

    def _add_program_assignment_counts(self, metrics, group, students):
        program_groups = {
            "LP program": self.immersion_programs + self.non_immersion_programs,
            "immersion program": self.immersion_programs,
            "non-immersion program": self.non_immersion_programs,
            "GE program": self.ge,
        }
        program_groups.update({code: [code] for code in self.ce_se})
        program_types = students["programtype"].to_numpy()
        designations = students["designation"].to_numpy()
        assigned_counts = Counter(program_types)
        designated_counts = Counter(
            program_type
            for program_type, designation in zip(
                program_types, designations, strict=True
            )
            if designation == 1
        )
        non_designated_counts = Counter(
            program_type
            for program_type, designation in zip(
                program_types, designations, strict=True
            )
            if designation == 0
        )
        for label, program_types in program_groups.items():
            metrics[f"Nb assigned students ({group}) to {label}"] = int(
                sum(assigned_counts.get(code, 0) for code in program_types)
            )
            metrics[f"Nb designated students ({group}) to {label}"] = int(
                sum(designated_counts.get(code, 0) for code in program_types)
            )
            if label != "LP program":
                metrics[f"Nb non-designated students ({group}) to {label}"] = int(
                    sum(non_designated_counts.get(code, 0) for code in program_types)
                )

    def map_ethnicity(self):
        def ethn(x):
            if x in [
                "Asian",
                "Asian Indian",
                "Chinese",
                "Vietnamese",
                "Filipino",
                "Japanese",
                "Korean",
                "Hmong",
                "Other Asian",
                "Cambodian",
                "Laotian",
            ]:
                return "Asian"
            elif x in ["Hispanic/Latino", "Hispanic/Latinx", "Hispanic"]:
                return "Hispanic"
            elif x in ["White", "Middle Eastern/Arabic"]:
                return "White"
            elif x in ["Black or African American", "Black/African American"]:
                return "Black"
            elif x in [
                "Other Pacific Islander",
                "Pacific Islander",
                "Samoan",
                "Hawaiian Native",
            ]:
                return "Pacific Islander"
            elif x in ["Two or More Races", "Multi-Racial", "Two or More"]:
                return "Two or More Races"
            elif x in ["Decline to State", "Decline To State"]:
                return "Decline to State"
            return "Other"

        self.student_data["ethnicity"] = self.student_data["resolved_ethnicity"].apply(
            ethn
        )

    @staticmethod
    def _distance_identity(value):
        if pd.isna(value):
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value).strip()
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return str(value).strip()

    def _assignment_distances_from_cache(self):
        student_keys = self.student_data["studentno"].map(self._distance_identity)
        program_keys = self.student_data["programcodes"].astype(str).str.strip()
        row_positions = np.array(
            [self._distance_cache_rows.get(key, -1) for key in student_keys],
            dtype=int,
        )
        column_positions = np.array(
            [self._distance_cache_columns.get(key, -1) for key in program_keys],
            dtype=int,
        )
        valid = (row_positions >= 0) & (column_positions >= 0)
        values = np.full(len(self.student_data), np.nan, dtype=float)
        values[valid] = self._distance_cache_values[
            row_positions[valid], column_positions[valid]
        ]
        return pd.Series(values, index=self.student_data.index)

    def eval_distance(self):
        """Calculate Haversine distance to each assigned school in miles."""

        school_locations = (
            self.schools_latlon.drop_duplicates("school_id")
            .set_index("school_id")[["lat", "lon"]]
            .to_dict("index")
        )

        def haversine_dist(lat1: float, lon1: float, school):
            if pd.isna(lat1) or pd.isna(lon1) or pd.isna(school) or school <= 0:
                return np.nan
            location = school_locations.get(school)
            if location is None or pd.isna(location["lat"]) or pd.isna(location["lon"]):
                return np.nan
            lat2 = float(location["lat"])
            lon2 = float(location["lon"])

            lat1, lat2, lon1, lon2 = [radians(_) for _ in [lat1, lat2, lon1, lon2]]
            c = 2 * asin(
                sqrt(
                    sin((lat2 - lat1) / 2) ** 2
                    + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
                )
            )  # angle
            r = 3958.8  # earth radius (in miles)

            return r * c

        self.student_data["assignment_dist"] = self.student_data.apply(
            lambda x: haversine_dist(
                x["latitude"], x["longitude"], x["assigned_school"]
            ),
            axis=1,
        )

    def is_designated(self, student_data=None):
        """Output: Series where index = student code, value = True (if student was assigned) or False( if not)."""
        if student_data is None:
            student_data = self.student_data
        return student_data["designation"].astype(bool)

    def is_assigned(self, student_data=None):
        """Output: Series where index = student code, value = True (if student was assigned) or False (if not)."""
        if student_data is None:
            student_data = self.student_data
        return self._assigned_mask(student_data)

    @staticmethod
    def _income_composition(all_students, income_threshold, high_income):
        """Return school and district shares on one side of an income cutoff."""
        income = pd.to_numeric(all_students["median_hh_income"], errors="coerce")
        valid = all_students[income.notna()].copy()
        if valid.empty:
            return pd.Series(dtype=float), np.nan
        qualifies = income[income.notna()] >= income_threshold
        if not high_income:
            qualifies = income[income.notna()] <= income_threshold
        totals = valid.groupby("assigned school").size()
        counts = valid[qualifies].groupby("assigned school").size()
        return counts.reindex(totals.index, fill_value=0) / totals, qualifies.mean()

    @staticmethod
    def _schools_outside_composition_range(
        school_shares, district_share, prop_threshold
    ):
        if school_shares.empty or pd.isna(district_share):
            return pd.Index([])
        cutoff = district_share + prop_threshold
        matches = (
            school_shares >= cutoff if prop_threshold > 0 else school_shares < cutoff
        )
        return school_shares.index[matches]

    def school_income_range(
        self, all_students, income_threshold, prop_threshold, composition=None
    ):
        school_shares, district_share = composition or self._income_composition(
            all_students, income_threshold, high_income=False
        )
        schools = self._schools_outside_composition_range(
            school_shares, district_share, prop_threshold
        )
        return self._safe_ratio(len(schools), len(school_shares))

    def high_income_school_range(
        self, all_students, income_threshold, prop_threshold, composition=None
    ):
        """Count schools by high-income-student share relative to the district."""
        school_shares, district_share = composition or self._income_composition(
            all_students, income_threshold, high_income=True
        )
        schools = self._schools_outside_composition_range(
            school_shares, district_share, prop_threshold
        )
        return len(schools), self._safe_ratio(len(schools), len(school_shares))

    def aalpi_in_high_income_schools(
        self,
        all_students,
        income_threshold,
        prop_threshold,
        composition=None,
    ):
        """Computes the proportion of AALPI (African American, Latino, Pacific Islander) students
        in schools with average income above a specified threshold.
        """
        shares, district_share = composition or self._income_composition(
            all_students, income_threshold, high_income=True
        )
        high_income_schools = self._schools_outside_composition_range(
            shares, district_share, prop_threshold
        )
        student_data_hi = all_students[
            all_students["assigned school"].isin(high_income_schools)
        ]
        return self._safe_ratio(
            student_data_hi["ethnicity"].isin(AALPI).sum(), len(student_data_hi)
        )

    def aalpi_in_low_income_schools(
        self,
        all_students,
        income_threshold,
        prop_threshold,
        composition=None,
    ):
        """Computes the proportion of AALPI (African American, Latino, Pacific Islander) students
        in schools with average income below a specified threshold.
        """
        shares, district_share = composition or self._income_composition(
            all_students, income_threshold, high_income=False
        )
        low_income_schools = self._schools_outside_composition_range(
            shares, district_share, prop_threshold
        )
        student_data_li = all_students[
            all_students["assigned school"].isin(low_income_schools)
        ]
        return self._safe_ratio(
            student_data_li["ethnicity"].isin(AALPI).sum(), len(student_data_li)
        )

    def avg_aalpi_in_high_income_schools(
        self,
        all_students,
        income_threshold,
        prop_threshold,
        composition=None,
    ):
        """Calculates the average proportion of AALPI (African American, Latino, Pacific Islander)
        students across high income schools with average income above a threshold.
        """
        shares, district_share = composition or self._income_composition(
            all_students, income_threshold, high_income=True
        )
        high_income_schools = self._schools_outside_composition_range(
            shares, district_share, prop_threshold
        )
        student_data_hi = all_students[
            all_students["assigned school"].isin(high_income_schools)
        ]
        if student_data_hi.empty:
            return np.nan
        return (
            student_data_hi.assign(is_aalpi=student_data_hi["ethnicity"].isin(AALPI))
            .groupby("assigned school")["is_aalpi"]
            .mean()
            .mean()
        )

    def avg_frl_in_high_income_schools(
        self,
        all_students,
        income_threshold,
        prop_threshold,
        composition=None,
    ):
        """Calculates the average proportion of FRL students across high
        income schools with average income above a threshold.
        """
        shares, district_share = composition or self._income_composition(
            all_students, income_threshold, high_income=True
        )
        high_income_schools = self._schools_outside_composition_range(
            shares, district_share, prop_threshold
        )
        student_data_hi = all_students[
            all_students["assigned school"].isin(high_income_schools)
        ]
        if student_data_hi.empty:
            return np.nan
        return student_data_hi.groupby("assigned school")["is_frl"].mean().mean()

    def poverty_concentration(
        self,
        all_students,
        students,
        threshold,
        return_count=False,
        school_frl_context=None,
    ):
        """Proportion of students in schools where the percentage of FRL students
        exceeds or falls below the district average by a certain threshold.
        """
        if school_frl_context is None:
            schools_frl = all_students.groupby("assigned school")["frl"].mean()
            district_avg = all_students["frl"].mean()
        else:
            district_avg, schools_frl = school_frl_context
        num_students = len(students)
        group_school_frl = students["assigned school"].map(schools_frl)
        if threshold >= 0:
            count = int((group_school_frl > district_avg + threshold).sum())
        else:
            count = int((group_school_frl < district_avg + threshold).sum())
        if return_count:
            return count
        return count / num_students if num_students else np.nan

    def ge_frl_range(self, ge_students, ge_groups, threshold, frl_context=None):
        if ge_students.empty:
            return 0, np.nan
        if frl_context is None:
            ges_frl = ge_groups["frl"].mean()
            district_avg = ge_students["frl"].mean()
        else:
            ges_frl, district_avg = frl_context
        if threshold >= 0:
            ges_matching = ges_frl >= district_avg + threshold
            return np.sum(ges_matching), np.mean(ges_matching)
        else:
            ges_matching = ges_frl <= district_avg + threshold
            return np.sum(ges_matching), np.mean(ges_matching)

    def metric_ge_FRL_concentration(
        self,
        ge_students,
        group_students,
        ge_groups,
        threshold,
        frl_context=None,
    ):
        """Proportion of students in GE programs where % of FRL students
        exceeds or falls below the district average by a certain threshold.
        """
        if frl_context is None:
            ges_frl = ge_groups["frl"].mean()
            district_avg = ge_students["frl"].mean()
        else:
            ges_frl, district_avg = frl_context
        num_students = len(group_students)
        if not num_students:
            return np.nan
        group_ge_frl = group_students["assignment"].map(ges_frl)
        if threshold >= 0:
            return (group_ge_frl > district_avg + threshold).mean()
        return (group_ge_frl < district_avg + threshold).mean()

    def ge_AAPI_range(
        self,
        higher_threshold,
        lower_threshold=0,
        smaller_strict=True,
        percentage=True,
        student_data=None,
    ):
        """Compute the number of GE programs that have less than threshold of their
        capacity or a certain number African American or Pacific Islander students.
        """
        if student_data is None:
            student_data = self.student_data
        counts = self._ge_aapi_counts(student_data)
        return self._ge_aapi_range_from_counts(
            counts,
            self._ge_program_capacities,
            higher_threshold,
            lower_threshold,
            smaller_strict,
            percentage,
        )

    def _ge_aapi_counts(self, student_data):
        assigned_mask = self._assigned_mask(student_data).to_numpy()
        aapi_mask = assigned_mask & np.isin(
            student_data["ethnicity"].to_numpy(),
            ["Black", "Pacific Islander"],
        )
        program_aapi_counts = Counter(
            student_data["programcodes"].to_numpy()[aapi_mask]
        )
        return np.fromiter(
            (program_aapi_counts.get(program_id, 0) for program_id in self._ge_program_ids),
            dtype=int,
            count=len(self._ge_program_ids),
        )

    @staticmethod
    def _ge_aapi_range_from_counts(
        counts,
        capacities,
        higher_threshold,
        lower_threshold=0,
        smaller_strict=True,
        percentage=True,
    ):
        if not percentage:
            if smaller_strict:
                return int(
                    ((counts >= lower_threshold) & (counts < higher_threshold)).sum()
                )
            return int(
                ((counts >= lower_threshold) & (counts <= higher_threshold)).sum()
            )

        lower_bound = lower_threshold * capacities
        upper_bound = higher_threshold * capacities
        if smaller_strict:
            in_range = (counts < upper_bound) & (counts >= lower_bound)
        else:
            in_range = (counts <= upper_bound) & (counts >= lower_bound)
        return int(in_range.sum())

    def load_zone_file_dict(self, zone_file):
        """Load a dictionary for zone_file mapping zone building block to zone id.

        Input:
            zone_file: path to the zone file
        Returns:
            the dictionary mapping zone building block to zone id
        """
        if zone_file is None:
            logger.warning("Zone file is None; map metrics may be incomplete.")
            return {}
        with open(zone_file) as f:
            reader = csv.reader(f)
            zones = list(reader)

        zone_dict = {}
        for idx, schools in enumerate(zones):
            zone_dict = {
                **zone_dict,
                **{int(float(s)): idx for s in schools if s != ""},
            }
        return zone_dict

    def eval_assignment_metrics_by_student_area(
        self,
        program_data,  # not used here.
        building_block="idschoolattendance",
        zone_file=None,
        school_order=None,
        zone_order=None,
    ):
        """Generate the dataframe for map-level metrics for 2024 summer SFUSD
        dashboard based on student area.

        TODO: similar to the function eval_assignment_metrics_by_assigned_area
        with only differences on grouping students. Refactor if have time.

        Input:
            program_data: program df used for in capacity and empty seats
            building_block: one of "idschoolattendance", "Block", "BlockGroup", or
                "Tract"
            zone_file: zone file as used in configs. Can only be none if using "aa" as
                zone building_block.
            school_order: a list of ordered schools to order the columns in the outputs
            zone_order: a list of ordered zones to order the columns in the outputs
        Returns:
            the dataframe where rows are metrics, and columns are the zones or AA.
        """
        student_data = self.student_data.copy()
        school_id_to_name = {
            x: y
            for [x, y] in self.schools_latlon[["school_id", "school_name"]].to_numpy()
        }
        # Map student to their map area by their building block.
        if building_block == "idschoolattendance":
            student_data["map_area"] = student_data["idschoolattendance"]
            school_to_zone_dict = {
                x: x for x in self.schools_latlon["school_id"].to_numpy()
            }
        elif building_block in ["Block", "BlockGroup", "Tract"]:
            zone_dict = self.load_zone_file_dict(zone_file)
            building_block_col = "census_" + building_block.lower()
            student_data["map_area"] = student_data[building_block_col].apply(
                lambda x: zone_dict[x] if not isnan(x) else np.nan
            )

            # Count attendance school to zone, and citywide school as itself.
            school_to_zone_dict = self.schools_latlon[
                ["school_id", building_block, "category"]
            ].apply(
                lambda x: [
                    x[0],
                    zone_dict[x[1]] if x[2] == "Attendance" else x[0],
                ],
                axis=1,
            )
            school_to_zone_dict = {x: y for [x, y] in school_to_zone_dict.to_numpy()}
        else:
            logger.warning(
                "Expected building_block to be one of idschoolattendance, "
                "Block, BlockGroup, or Tract; returning None."
            )
            return None

        # Build up program data matched to compute program capacity.
        program_data = program_data.copy()
        program_data["program_area"] = program_data["school_id"].apply(
            lambda x: school_to_zone_dict[x]
        )
        # Gather the area list for building blocks.
        if building_block == "BlockGroup":
            # For formatting, have 18 columns no matter how many zones we have.
            area_list = [x for x in range(18)]
        elif building_block == "Block":
            # Adding missing zones
            area_list = [x for x in range(59)]
        elif building_block == "Tract":
            area_list = sorted(set(zone_dict.values()))
        elif building_block == "idschoolattendance":
            area_list = [
                x for x in self.schools_latlon["school_id"].to_numpy() if x != 909
            ]

        list_metrics = []
        for area in area_list:
            cur_students = student_data[student_data["map_area"] == area]
            # cur_programs = program_data[program_data["program_area"] == area]
            metrics = self.generate_metrics_by_aera(cur_students, None)
            list_metrics.append(pd.Series(metrics))

        # Format to return
        area_list = [self.format_cur_col_name(x, school_id_to_name) for x in area_list]
        col_names = dict(zip(range(len(area_list)), area_list))
        all_metrics_df = pd.concat(list_metrics, axis=1).rename(columns=col_names)
        all_metrics_df = self.reorder_schools_zones(
            all_metrics_df, school_order=school_order, zone_order=zone_order
        )
        return all_metrics_df

    def eval_assignment_metrics_by_assigned_area(
        self,
        program_data,
        building_block="idschoolattendance",
        zone_file=None,
        school_order=None,
        zone_order=None,
    ):
        """Generate the dataframe for map-level metrics for 2024 summer SFUSD
        dashboard based on assigned school area.

        TODO: the current codes are very messy, we need to clean it up and/or optimize
            it when we have time.

        Input:
            program_data: program df used for in capacity and empty seats
            building_block: one of "idschoolattendance", "Block", "BlockGroup", or
                "Tract"
            zone_file: zone file as used in configs. Can only be none if using "aa" as
                zone building_block.
            school_order: a list of ordered schools to order the columns in the outputs
            zone_order: a list of ordered zones to order the columns in the outputs
        Returns:
            the dataframe where rows are metrics, and columns are the zones or AA.
        """
        student_data = self.student_data.copy()
        school_id_to_name = {
            x: y
            for [x, y] in self.schools_latlon[["school_id", "school_name"]].to_numpy()
        }
        # Map student to their assigned area by assigned_school.
        if building_block == "idschoolattendance":
            school_to_zone_dict = {
                x: x for x in self.schools_latlon["school_id"].to_numpy()
            }
        elif building_block in ["Block", "BlockGroup", "Tract"]:
            zone_dict = self.load_zone_file_dict(zone_file)
            # Count attendance school to zone, and citywide school as itself.
            school_to_zone_dict = self.schools_latlon[
                ["school_id", building_block, "category"]
            ].apply(
                lambda x: [
                    x[0],
                    zone_dict[x[1]] if x[2] == "Attendance" else x[0],
                ],
                axis=1,
            )
            school_to_zone_dict = {x: y for [x, y] in school_to_zone_dict.to_numpy()}
        else:
            logger.warning(
                "Expected building_block to be one of idschoolattendance, "
                "Block, BlockGroup, or Tract; returning None."
            )
            return None
        # Count unassigned student to their located zone based on attandance area.
        student_data["assigned_area"] = student_data[
            ["assigned_school", "idschoolattendance"]
        ].apply(
            lambda x: (
                school_to_zone_dict[x[0]]
                if not isnan(x[0])
                else school_to_zone_dict[x[1]]
            ),
            axis=1,
        )
        # Build up program data matched to compute program capacity.
        program_data = program_data.copy()
        program_data["program_area"] = program_data["school_id"].apply(
            lambda x: school_to_zone_dict[x]
        )
        # area_list = ["All"] + list(np.unique(student_data["assigned_area"].to_numpy()))
        area_list = ["All"]
        aa_schools = self.schools_latlon[
            self.schools_latlon["category"] == "Attendance"
        ]["school_id"].to_numpy()
        aa_schools = [x for x in aa_schools if x != 909]
        all_schools = [
            x for x in self.schools_latlon["school_id"].to_numpy() if x != 909
        ]
        citywide_schools = list(set(all_schools) - set(aa_schools))
        # Fill in what metrics need to be generated for each config.
        # Zone ids + citywide schools for block or blockgroup and all schools for AA.
        if building_block == "BlockGroup":
            # For formatting, have 18 columns no matter how many zones we have.
            area_list += [x for x in range(18) if x not in area_list]
            area_list += citywide_schools
        elif building_block == "Block":
            # Adding missing zones
            area_list += [x for x in range(59) if x not in area_list]
            area_list += citywide_schools
        elif building_block == "Tract":
            area_list += [
                zone
                for zone in sorted(set(zone_dict.values()))
                if zone not in area_list
            ]
            area_list += citywide_schools
        elif building_block == "idschoolattendance":
            area_list += all_schools

        list_metrics = []
        for area in area_list:
            if area == "All":
                cur_students = student_data
                cur_programs = program_data
            else:
                cur_students = student_data[student_data["assigned_area"] == area]
                cur_programs = program_data[program_data["program_area"] == area]

            metrics = self.generate_metrics_by_aera(cur_students, cur_programs)
            list_metrics.append(pd.Series(metrics))

        # Add metrics at attendance school level.
        # TODO: currently code is messy and redundant. We should clean this up
        # and/or optimize when we have time. Ideally, we should not seperate
        # Attendance area school from the citywide schools.
        if building_block in ["Block", "BlockGroup", "Tract"]:
            aa_schools = self.schools_latlon[
                self.schools_latlon["category"] == "Attendance"
            ]["school_id"].to_numpy()
            # 909 is duplicate of 999 and not used in assignment.
            aa_schools = [x for x in aa_schools if x != 909]
            area_list += list(aa_schools)
            for aa_school in aa_schools:
                cur_students = student_data[
                    student_data["assigned_school"] == aa_school
                ]
                cur_programs = program_data[program_data["school_id"] == aa_school]
                metrics = self.generate_metrics_by_aera(cur_students, cur_programs)
                list_metrics.append(pd.Series(metrics))

        area_list = [self.format_cur_col_name(x, school_id_to_name) for x in area_list]
        col_names = dict(zip(range(len(area_list)), area_list))
        all_metrics_df = pd.concat(list_metrics, axis=1).rename(columns=col_names)
        all_metrics_df = self.reorder_schools_zones(
            all_metrics_df, school_order=school_order, zone_order=zone_order
        )
        return all_metrics_df

    def reorder_schools_zones(self, all_metrics_df, school_order=None, zone_order=None):
        """Reorder columns based on input school and/or zone orders. Assume that
        the columns of all_metrics_df contains "All", and a list of zones with format
        "Zone <zone id>" (e.g. "Zone 1") and/or a list of schools in the format
        "<School id> <school name>".
        TODO: optimize if have time.
        """
        columns = all_metrics_df.columns
        new_columns = ["All"] if "All" in columns else []
        # Separate list of zone and schools.
        list_zones = [x for x in columns if "Zone" in x]
        list_schools = [x for x in columns if "Zone" not in x and x != "All"]
        # TODO: zone order if we have.
        new_columns += list_zones
        # School orders, need to rename as the school name are different in our
        # records v.s. the ones given by SFUSD.
        if school_order is None:
            new_columns += list_schools
        else:
            # Exclude schools not in results from school order
            existing_schools = set([x[:3] for x in list_schools])
            school_order = [x for x in school_order if x[:3] in existing_schools]
            # Rename columns to the new school names.
            new_schoolid2school = {x[:3]: x for x in school_order}
            school_old2new = {x: new_schoolid2school[x[:3]] for x in list_schools}
            all_metrics_df = all_metrics_df.rename(columns=school_old2new)
            new_columns += school_order
        return all_metrics_df[new_columns]

    def format_cur_col_name(self, area, school_id_to_name):
        """Helper function for eval_assignment_metrics_by_area to decide the column
        name for the map-level metrics.
        """
        if area == "All":
            return area
        # With time limit, assume no zone has id > 100
        if area > 100:
            return f"{area} {school_id_to_name[area]}"
        else:
            return f"Zone {area + 1} {ZONE_COLORS[area]}"

    def generate_metrics_by_aera(
        self,
        cur_students,
        cur_programs,
        all_students=None,
        ge_over_assigned=True,
    ):
        if all_students is None:
            all_students = self.student_data.copy()
        ignore_program_related = cur_programs is None
        assigned_students = all_students[all_students["programno"] > 0]

        metrics = {}
        cur_assigned_students = cur_students[cur_students["programno"] > 0]
        cur_aalpi_students = cur_assigned_students[
            cur_assigned_students["ethnicity"].apply(lambda x: x in AALPI)
        ]
        # Counts of assigned, capacity, designated, unassigned.
        if not ignore_program_related:
            metrics["Capacity"] = cur_programs["capacity"].sum()
        metrics["Students"] = len(cur_students)
        if not ignore_program_related:
            metrics["# schools"] = len(np.unique(cur_programs["school_id"].to_numpy()))
        metrics["Assigned students"] = len(cur_assigned_students)
        metrics["Assigned to language program"] = len(
            cur_students[
                cur_students["programtype"].isin(
                    self.immersion_programs + self.non_immersion_programs
                )
            ]
        )
        metrics["Not assigned"] = len(cur_students[cur_students["programno"] == 0])
        metrics["Designated"] = len(cur_students[(cur_students["designation"] != 0)])
        if not ignore_program_related:
            metrics["Empty Seats"] = metrics["Capacity"] - metrics["Assigned students"]
            if ge_over_assigned:
                # Given time limit on implementation, assume over-assignment is only for GE.
                ge_capacity = cur_programs[cur_programs["program_type"] == "GE"][
                    ["capacity", "school_id"]
                ]
                ge_students = (
                    cur_students[cur_students["programtype"] == "GE"][
                        ["assigned_school", "studentno"]
                    ]
                    .groupby("assigned_school")
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            "assigned_school": "school_id",
                            "studentno": "student_count",
                        }
                    )
                )
                ge_empty_seats = ge_capacity.merge(
                    ge_students, how="outer", on="school_id"
                )
                ge_empty_seats["empty_seats"] = (
                    ge_empty_seats["capacity"] - ge_empty_seats["student_count"]
                )
                metrics["# GE Empty Seats"] = np.sum(
                    ge_empty_seats[ge_empty_seats["empty_seats"] >= 0][
                        "empty_seats"
                    ].to_numpy()
                )
                metrics["# GE Over Assigned Seats"] = -np.sum(
                    ge_empty_seats[ge_empty_seats["empty_seats"] < 0][
                        "empty_seats"
                    ].to_numpy()
                )
        # School choice.
        choice_outcomes = self._top_choice_outcomes(
            cur_assigned_students, "rank", [1, 3]
        )
        self._record_choice_outcome(
            metrics,
            "Assigned to 1st choice",
            choice_outcomes[1],
        )
        self._record_choice_outcome(
            metrics,
            "Assigned top-3 choice",
            choice_outcomes[3],
        )
        # Distances
        metrics["Avg. Distance"] = cur_assigned_students["assignment_dist"].mean()
        metrics["Median Distance"] = cur_assigned_students["assignment_dist"].median()
        metrics["% assigned within 0.5mi"] = (
            cur_assigned_students["assignment_dist"] < 0.5
        ).mean()
        metrics["% assigned beyond 3mi"] = (
            cur_assigned_students["assignment_dist"] > 3
        ).mean()
        # Schools
        (
            metrics["# high-poverty schools"],
            _,
            metrics["# students in high-poverty schools"],
        ) = self.school_frl_range(
            0.15,
            student_data=cur_assigned_students,
            all_student_data=assigned_students,
        )
        metrics["# AALPI students in high-poverty schools"] = (
            self.poverty_concentration(
                assigned_students, cur_aalpi_students, 0.15, return_count=True
            )
        )
        metrics["# of students from high-poverty area"] = len(
            cur_assigned_students[cur_assigned_students["frl"] > 0.5]
        )
        # Count of students ethnicity groups.
        groups = ["Black", "Asian", "Hispanic", "White"]
        for group in groups:
            group_name = group if group != "Black" else "African American"
            metrics[f"# {group_name} students"] = len(
                cur_students[(cur_students["ethnicity"] == group)]
            )
        return metrics

    def eval_ge_utilization_by_school(self, config_name):
        """Return GE capacity and assignments grouped by school for heatmaps."""
        if self._mode != "full":
            raise ValueError("GE utilization requires a full MatchEvaluator instance")
        required = {"program_id", "school_id", "program_type", "capacity"}
        missing = sorted(required - set(self.programs.columns))
        if missing:
            raise ValueError(f"GE utilization requires program columns: {missing}")

        ge_programs = self.programs.loc[
            self.programs["program_type"].eq("GE"),
            ["program_id", "school_id", "capacity"],
        ].copy()
        assigned = self.student_data.loc[
            self.student_data["programno"] > 0, "assignment"
        ].value_counts()
        ge_programs["assigned"] = ge_programs["program_id"].map(assigned).fillna(0)
        result = (
            ge_programs.groupby("school_id", as_index=False, sort=True)[
                ["capacity", "assigned"]
            ]
            .sum(min_count=1)
            .reset_index(drop=True)
        )
        result.insert(0, "config_name", config_name)
        return result

    def eval_assignment_metrics_by_program(self, aggregates=None):
        """Return simulated assignment outcomes with one row per program."""
        if self._mode != "full":
            raise ValueError(
                "Program-level metrics require a full MatchEvaluator instance"
            )

        required_school_columns = {"school_id", "school_name", "category"}
        missing_school_columns = required_school_columns - set(
            self.schools_latlon.columns
        )
        if missing_school_columns:
            raise ValueError(
                "Program-level metrics require school columns: "
                f"{sorted(missing_school_columns)}"
            )
        if "capacity" not in self.programs:
            raise ValueError("Program-level metrics require program column: capacity")

        school_metadata = self.schools_latlon[
            ["school_id", "school_name", "category"]
        ].copy()
        duplicate_school_ids = school_metadata.loc[
            school_metadata["school_id"].duplicated(keep=False), "school_id"
        ].unique()
        if len(duplicate_school_ids):
            raise ValueError(
                "School metadata contains duplicate school_id values: "
                f"{duplicate_school_ids.tolist()}"
            )
        for column in ("school_name", "category"):
            values = school_metadata[column].astype("string").str.strip()
            if (values.isna() | values.eq("").fillna(False)).any():
                raise ValueError(
                    f"School metadata contains missing or blank {column} values"
                )
            school_metadata[column] = values.astype(str)
        missing_school_ids = sorted(
            set(self.programs["school_id"]) - set(school_metadata["school_id"])
        )
        if missing_school_ids:
            raise ValueError(
                "Program school_id values absent from school metadata: "
                f"{missing_school_ids}"
            )
        school_metadata = school_metadata.rename(
            columns={"category": "school_category"}
        )
        programs = self.programs[
            ["program_id", "school_id", "program_type", "capacity"]
        ].merge(school_metadata, on="school_id", how="left", validate="many_to_one")
        programs = programs[
            [
                "program_id",
                "school_id",
                "school_name",
                "school_category",
                "program_type",
                "capacity",
            ]
        ].copy()
        if aggregates is None:
            aggregates = self._prepare_full_report_aggregates(
                self.student_data, include_program_report_stats=True
            )
        assigned_stats = aggregates.program_report_stats
        non_designated_stats = aggregates.non_designated_program_report_stats
        designated_stats = aggregates.designated_program_report_stats
        program_ids = programs["program_id"]
        assigned = program_ids.map(assigned_stats["count"]).fillna(0).astype(int)
        designated = program_ids.map(designated_stats["count"]).fillna(0).astype(int)
        capacity = programs["capacity"]

        programs["assigned"] = assigned
        programs["designated"] = designated
        programs["mean_travel_dist_assigned"] = program_ids.map(
            assigned_stats["assignment_dist"]
        )
        programs["mean_travel_dist_designated"] = program_ids.map(
            designated_stats["assignment_dist"]
        )
        programs["percent_designated"] = [
            self._safe_ratio(value, total)
            for value, total in zip(designated, assigned, strict=True)
        ]
        programs["frl_assigned"] = program_ids.map(assigned_stats["frl"])
        programs["frl_designated"] = program_ids.map(designated_stats["frl"])
        programs["frl_non_designated"] = program_ids.map(non_designated_stats["frl"])
        programs["program_utilization"] = [
            self._safe_ratio(total, seats)
            for total, seats in zip(assigned, capacity, strict=True)
        ]
        programs["overage"] = [
            self._safe_ratio(max(total - seats, 0), seats)
            for total, seats in zip(assigned, capacity, strict=True)
        ]
        programs["underage"] = [
            self._safe_ratio(max(seats - total, 0), seats)
            for total, seats in zip(assigned, capacity, strict=True)
        ]
        for rank in range(1, 4):
            numerators = program_ids.map(non_designated_stats[f"top_{rank}"]).fillna(0)
            programs[f"prop_top_{rank}"] = [
                self._safe_ratio(numerator, total)
                for numerator, total in zip(numerators, assigned, strict=True)
            ]

        demographic_columns = []
        for ethnicity in DIAGNOSTIC_ETHNICITIES:
            slug = DIAGNOSTIC_ETHNICITY_SLUGS[ethnicity]
            non_designated_column = f"non_designated_{slug}_students"
            designated_column = f"designated_{slug}_students"
            programs[non_designated_column] = (
                program_ids.map(non_designated_stats[ethnicity]).fillna(0).astype(int)
            )
            programs[designated_column] = (
                program_ids.map(designated_stats[ethnicity]).fillna(0).astype(int)
            )
            demographic_columns.extend([non_designated_column, designated_column])

        return programs[
            [
                "program_id",
                "school_id",
                "school_name",
                "school_category",
                "program_type",
                "capacity",
                "assigned",
                "designated",
                "mean_travel_dist_assigned",
                "mean_travel_dist_designated",
                "percent_designated",
                "frl_assigned",
                "frl_designated",
                "frl_non_designated",
                "program_utilization",
                "overage",
                "underage",
                "prop_top_1",
                "prop_top_2",
                "prop_top_3",
                *demographic_columns,
            ]
        ]

    def _eval_assignment_full_by_group(
        self, source_column, output_column, metric_columns=None
    ):
        """Evaluate the complete full report for each residential group."""
        if self._mode != "full":
            raise ValueError(
                "Geography-level metrics require a full MatchEvaluator instance"
            )
        if source_column not in self.student_data:
            raise ValueError(
                f"Geography-level metrics require student column: {source_column}"
            )

        geography = pd.to_numeric(self.student_data[source_column], errors="coerce")
        supplied = self.student_data[source_column].notna()
        invalid = supplied & (
            geography.isna() | ~np.isfinite(geography) | (geography % 1 != 0)
        )
        if invalid.any():
            values = self.student_data.loc[invalid, source_column].unique().tolist()
            raise ValueError(
                f"Student column {source_column} contains invalid values: {values}"
            )

        rows = []
        students_with_geography = self.student_data[supplied].assign(
            _aggregate_geography=geography[supplied].astype("int64")
        )
        for area, students in students_with_geography.groupby(
            "_aggregate_geography", sort=True
        ):
            students = students.drop(columns="_aggregate_geography")
            aggregates = self._prepare_full_report_aggregates(students)
            rows.append(
                {
                    output_column: area,
                    **self._eval_assignment_full_from_aggregates(
                        aggregates
                    ).to_dict(),
                }
            )
        if rows:
            return pd.DataFrame(rows)
        if metric_columns is None:
            metric_columns = self.eval_assignment_full().index.tolist()
        return pd.DataFrame(
            columns=[output_column, *metric_columns]
        )

    def eval_assignment_metrics_by_zip_code(self):
        """Return the complete full report for each student ZIP code."""
        return self._eval_assignment_full_by_group("zipcode", "zip_code")

    def eval_assignment_metrics_by_attendance_area(self):
        """Return the complete full report for each student's attendance area."""
        return self._eval_assignment_full_by_group(
            "idschoolattendance", "attendance_area"
        )

    def eval_frl_threshold_inputs(self, config_name):
        """Return per-program inputs for average-first FRL threshold metrics."""
        assigned_students = self.student_data[self._assigned_mask(self.student_data)]
        ge_students = assigned_students[assigned_students["programtype"] == "GE"]
        non_designated = ge_students[ge_students["designation"] == 0]
        program_ids = self.programs.loc[
            self.programs["program_type"] == "GE", "program_id"
        ]
        return pd.DataFrame(
            {
                "config_name": config_name,
                "program_id": program_ids,
                "frl_assigned": program_ids.map(
                    ge_students.groupby("assignment")["frl"].mean()
                ),
                "frl_non_designated": program_ids.map(
                    non_designated.groupby("assignment")["frl"].mean()
                ),
                "district_frl": assigned_students["frl"].mean(),
            }
        ).reset_index(drop=True)

    def eval_aggregate_metric_reports(
        self, config_name, *, include_local_metrics=True
    ):
        """Return citywide and, when requested, local reports for one assignment."""
        if self._distance_cache is None:
            raise ValueError(
                "Aggregate assignment metrics require the cached "
                "student-program distance matrix."
            )
        required_student_columns = {
            "ctip1",
            "median_hh_income",
        }
        if include_local_metrics:
            required_student_columns.update({"idschoolattendance", "zipcode"})
        missing_student_columns = required_student_columns - set(
            self.student_data.columns
        )
        if missing_student_columns:
            raise ValueError(
                "Aggregate assignment metrics require student columns: "
                f"{sorted(missing_student_columns)}"
            )
        district_aggregates = self._prepare_full_report_aggregates(
            self.student_data,
            include_program_report_stats=include_local_metrics,
        )
        citywide_metrics = self._eval_assignment_full_from_aggregates(
            district_aggregates
        )
        reports = {"citywide": pd.DataFrame([citywide_metrics.to_dict()])}
        if include_local_metrics:
            metric_columns = citywide_metrics.index.tolist()
            reports.update(
                {
                    "program": self.eval_assignment_metrics_by_program(
                        district_aggregates
                    ),
                    "zip_code": self._eval_assignment_full_by_group(
                        "zipcode", "zip_code", metric_columns
                    ),
                    "attendance_area": self._eval_assignment_full_by_group(
                        "idschoolattendance", "attendance_area", metric_columns
                    ),
                }
            )
        for report in reports.values():
            report.insert(0, "config_name", config_name)
        return reports

    def school_frl_range(
        self,
        pct,
        student_data=None,
        all_student_data=None,
        non_desig_only=False,
    ):
        """(#of schools, proportion of schools, #of students in those schools) above/below pct
        the average proportion of FRL students in the district.
        """
        if all_student_data is None:
            all_student_data = self.student_data
        all_student_data = all_student_data[
            self._assigned_mask(all_student_data)
        ].copy()
        if student_data is None:
            student_data = all_student_data
        else:
            student_data = student_data[self._assigned_mask(student_data)].copy()
        if non_desig_only:
            student_data = student_data[student_data["designation"] == 0]
        context = self._school_frl_context(all_student_data, non_desig_only)
        return self._school_frl_range_from_context(pct, student_data, context)

    @staticmethod
    def _school_frl_context(all_student_data, non_desig_only=False):
        district_avg = all_student_data["frl"].mean()
        composition_students = (
            all_student_data[all_student_data["designation"] == 0]
            if non_desig_only
            else all_student_data
        )
        school_frl = composition_students.groupby("assigned school")["frl"].mean()
        return district_avg, school_frl

    @staticmethod
    def _school_frl_range_from_context(pct, student_data, context):
        district_avg, school_frl = context
        if school_frl.empty:
            return 0, np.nan, 0

        if pct >= 0:
            schools_matching = school_frl.to_numpy() >= district_avg + pct
        else:
            schools_matching = school_frl.to_numpy() <= district_avg + pct
        school_ids = school_frl.index.to_numpy()[schools_matching]
        return (
            np.sum(schools_matching),
            np.mean(schools_matching),
            np.isin(student_data["assigned school"].to_numpy(), school_ids).sum(),
        )

    def metric_dissimilarity(self, students, total_enrollment):
        """Historical one-sided group dissimilarity, aligned by school."""
        n = len(students)
        total_n = pd.to_numeric(total_enrollment, errors="coerce").sum()
        if n == 0 or total_n == 0:
            return np.nan
        ratio = n / total_n
        enrollment = Counter(students["assigned school"].to_numpy())
        dissimilarity_total = sum(
            abs(enrollment.get(school, 0) - total_students * ratio) / 2
            for school, total_students in total_enrollment.items()
        )
        return dissimilarity_total / n

    def metrics_segregation(self, group_a, group_b, proportion, total_enrollment):
        en_a = Counter(group_a["assigned school"].to_numpy())
        en_a_sum = len(group_a)
        en_b = Counter(group_b["assigned school"].to_numpy())
        en_b_sum = len(group_b)
        if en_a_sum == 0 or en_b_sum == 0:
            return np.nan
        prop = Counter(proportion["assigned school"].to_numpy())
        return sum(
            ((en_a.get(school, 0) / en_a_sum) - (en_b.get(school, 0) / en_b_sum))
            * (prop.get(school, 0) / total_students)
            for school, total_students in total_enrollment.items()
        )

    @staticmethod
    def _school_exposure(proportion, total_enrollment):
        counts = Counter(proportion["assigned school"].to_numpy())
        return {
            school: counts.get(school, 0) / total_students
            for school, total_students in total_enrollment.items()
        }

    def metrics_exposure(
        self,
        group,
        proportion,
        total_enrollment,
        school_exposure=None,
    ):
        """Compute the absolute exposure of a group to a proportion.

        This calculates the average proportion of 'proportion' students in the
        schools attended by students in 'group'. Unlike metrics_segregation,
        this does not compute a difference between two groups.

        Args:
            group: DataFrame of students in the group of interest.
            proportion: DataFrame of students representing the proportion
                (e.g., high FRL students, AALPI students).
            total_enrollment: Series with school as index and total enrollment
                as value.

        Returns:
            float: The weighted average exposure.
        """
        if group.empty:
            return np.nan
        if school_exposure is None:
            school_exposure = self._school_exposure(proportion, total_enrollment)
        return np.mean(
            [school_exposure[school] for school in group["assigned school"].to_numpy()]
        )

    def metric_theil(self, student_data=None):
        """Compute Theil's H (entropy-based) segregation index.

        Measures how evenly ethnic groups are distributed across schools
        relative to the district-wide composition.

        Returns:
            float: Theil H index (0 = perfect integration, 1 = complete segregation)
        """
        if student_data is None:
            student_data = self.student_data
        assigned = student_data[student_data["programno"] > 0].copy()
        if len(assigned) == 0:
            return 0.0

        _, school_codes = np.unique(
            assigned["assigned school"].to_numpy(),
            return_inverse=True,
        )
        _, ethnicity_codes, ethnic_total = np.unique(
            assigned["ethnicity"].to_numpy(),
            return_inverse=True,
            return_counts=True,
        )
        ethnic_matrix = np.zeros(
            (school_codes.max() + 1, ethnicity_codes.max() + 1),
            dtype=int,
        )
        np.add.at(ethnic_matrix, (school_codes, ethnicity_codes), 1)
        ethnic_total_norm = ethnic_total / ethnic_total.sum()

        # District entropy
        district_entropy = np.sum(
            ethnic_total_norm * np.log(1.0 / ethnic_total_norm)
        )
        if district_entropy == 0:
            return 0.0

        # School-level entropy weighted by school size
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

    def _prepare_full_report_aggregates(
        self,
        student_data,
        *,
        include_program_report_stats=False,
    ):
        assigned_students = student_data[student_data["programno"] > 0]
        designated_students = assigned_students[
            assigned_students["designation"] != 0
        ]
        non_designated_students = assigned_students[
            assigned_students["designation"] == 0
        ]
        ethnicity = assigned_students["ethnicity"]
        income = pd.to_numeric(
            assigned_students["median_hh_income"], errors="coerce"
        )

        student_groups = {
            "All Assigned": assigned_students,
            "Black": assigned_students[ethnicity == "Black"],
            "Asian": assigned_students[ethnicity == "Asian"],
            "Hispanic": assigned_students[ethnicity == "Hispanic"],
            "White": assigned_students[ethnicity == "White"],
            "Two or More Races": assigned_students[ethnicity == "Two or More Races"],
            "Decline to State": assigned_students[ethnicity == "Decline to State"],
            "High FRL": assigned_students[assigned_students["frl"] > 0.5],
            "Low FRL": assigned_students[assigned_students["frl"] <= 0.5],
            "CTIP": assigned_students[assigned_students["ctip1"] == 1],
            "non-CTIP": assigned_students[assigned_students["ctip1"] == 0],
            "ET (2024)": assigned_students[assigned_students["et2"] == 1],
            "non-ET (2024)": assigned_students[assigned_students["et2"] == 0],
            "AALPI": assigned_students[ethnicity.isin(AALPI)],
            f"Income below {self.medium_income}": assigned_students[
                income <= self.medium_income
            ],
            f"Income above {self.medium_income}": assigned_students[
                income > self.medium_income
            ],
        }

        stats_source = assigned_students[
            ["assigned school", "studentno", "frl", "assignment_dist", "rank"]
        ].copy()
        stats_source["_aalpi"] = ethnicity.isin(AALPI).to_numpy()
        stats_source["_high_frl"] = (assigned_students["frl"] > 0.5).to_numpy()
        stats_source["_low_frl"] = (assigned_students["frl"] <= 0.5).to_numpy()
        stats_source["_income_valid"] = income.notna().to_numpy()
        stats_source["_high_income"] = (income >= self.medium_income).to_numpy()
        stats_source["_low_income"] = (income <= self.low_income).to_numpy()
        school_stats = stats_source.groupby("assigned school").agg(
            enrollment=("studentno", "size"),
            frl=("frl", "mean"),
            assignment_dist=("assignment_dist", "mean"),
            aalpi=("_aalpi", "sum"),
            high_frl=("_high_frl", "sum"),
            low_frl=("_low_frl", "sum"),
            income_valid=("_income_valid", "sum"),
            high_income=("_high_income", "sum"),
            low_income=("_low_income", "sum"),
        )

        def assignment_school_stats(students):
            return students.groupby("assigned school").agg(
                enrollment=("studentno", "size"),
                frl=("frl", "mean"),
                assignment_dist=("assignment_dist", "mean"),
                rank=("rank", "mean"),
            )

        non_designated_school_stats = assignment_school_stats(
            non_designated_students
        )

        def program_report_stats(students):
            source = students[
                [
                    "assignment",
                    "studentno",
                    "assignment_dist",
                    "rank",
                    "ethnicity",
                    "frl",
                ]
            ].copy()
            for rank in range(1, 4):
                source[f"top_{rank}"] = source["rank"] <= rank
            for ethnicity in DIAGNOSTIC_ETHNICITIES:
                source[ethnicity] = source["ethnicity"] == ethnicity
            return source.groupby("assignment").agg(
                count=("studentno", "size"),
                assignment_dist=("assignment_dist", "mean"),
                frl=("frl", "mean"),
                **{f"top_{rank}": (f"top_{rank}", "sum") for rank in range(1, 4)},
                **{
                    ethnicity: (ethnicity, "sum")
                    for ethnicity in DIAGNOSTIC_ETHNICITIES
                },
            )

        if include_program_report_stats:
            assignment_program_report_stats = program_report_stats(assigned_students)
            non_designated_program_report_stats = program_report_stats(
                non_designated_students
            )
            designated_program_report_stats = program_report_stats(designated_students)
        else:
            assignment_program_report_stats = None
            non_designated_program_report_stats = None
            designated_program_report_stats = None
        district_frl = assigned_students["frl"].mean()
        school_frl_context = (district_frl, school_stats["frl"])
        non_designated_school_frl_context = (
            district_frl,
            non_designated_school_stats["frl"],
        )
        enrollment = school_stats["enrollment"]
        school_exposures = {
            name: (school_stats[column] / enrollment).to_dict()
            for name, column in {
                "AALPI": "aalpi",
                "High FRL": "high_frl",
                "Low FRL": "low_frl",
            }.items()
        }

        income_stats = school_stats[school_stats["income_valid"] > 0]
        valid_income_count = income_stats["income_valid"].sum()
        high_income_composition = (
            income_stats["high_income"] / income_stats["income_valid"],
            self._safe_ratio(income_stats["high_income"].sum(), valid_income_count),
        )
        low_income_composition = (
            income_stats["low_income"] / income_stats["income_valid"],
            self._safe_ratio(income_stats["low_income"].sum(), valid_income_count),
        )

        ge_students = assigned_students[assigned_students["programtype"] == "GE"]
        ge_aalpi_students = ge_students[ge_students["ethnicity"].isin(AALPI)]
        ge_frl_context = (
            ge_students.groupby("assignment")["frl"].mean(),
            district_frl,
        )
        ge_non_designated_students = ge_students[ge_students["designation"] == 0]
        ge_non_designated_program_frl_context = (
            ge_non_designated_students.groupby("assignment")["frl"].mean(),
            district_frl,
        )
        if not self.overscribe_aa:
            overage = 0.0
        else:
            overage = self._safe_ratio(
                assigned_students["overage_seat"].sum(), len(student_data)
            )
        return _FullReportAggregates(
            student_data=student_data,
            assigned_students=assigned_students,
            designated_students=designated_students,
            non_designated_students=non_designated_students,
            student_groups=student_groups,
            school_stats=school_stats,
            non_designated_school_stats=non_designated_school_stats,
            program_report_stats=assignment_program_report_stats,
            non_designated_program_report_stats=(
                non_designated_program_report_stats
            ),
            designated_program_report_stats=designated_program_report_stats,
            school_frl_context=school_frl_context,
            non_designated_school_frl_context=non_designated_school_frl_context,
            school_exposures=school_exposures,
            high_income_composition=high_income_composition,
            low_income_composition=low_income_composition,
            ge_students=ge_students,
            ge_aalpi_students=ge_aalpi_students,
            ge_frl_context=ge_frl_context,
            ge_non_designated_students=ge_non_designated_students,
            ge_non_designated_program_frl_context=(
                ge_non_designated_program_frl_context
            ),
            ge_aapi_counts=self._ge_aapi_counts(student_data),
            all_student_top_choice_outcomes=self._top_choice_outcomes(
                student_data, "rank", range(1, 4)
            ),
            overage=overage,
        )

    def eval_assignment_full(self, student_data=None):
        """Return the extended report, optionally scoped to selected students."""
        if self._mode != "full":
            raise ValueError(
                "eval_assignment_full requires raw student/assignment data and "
                "the program_file and schools_latlon_path resources"
            )
        if student_data is None:
            student_data = self.student_data
        return self._eval_assignment_full_from_aggregates(
            self._prepare_full_report_aggregates(student_data)
        )

    def _eval_assignment_full_from_aggregates(self, aggregates):
        low_income = self.low_income
        medium_income = self.medium_income
        metrics = {}
        student_data = aggregates.student_data
        assigned_students = aggregates.assigned_students
        designated_students = aggregates.designated_students
        non_designated_students = aggregates.non_designated_students

        # Add total AALPI counts for both all students and assigned students
        metrics["Total AALPI students"] = len(
            student_data[student_data["ethnicity"].isin(AALPI)]
        )
        metrics["Total AALPI assigned students"] = len(
            assigned_students[assigned_students["ethnicity"].isin(AALPI)]
        )

        enrollment = aggregates.school_stats["enrollment"]
        metrics["# Racial majority schools"] = self.metric_racial_majority_schools(
            assigned_students
        )

        class_below_medium_income = aggregates.student_groups[
            f"Income below {medium_income}"
        ]
        class_above_medium_income = aggregates.student_groups[
            f"Income above {medium_income}"
        ]

        metrics.update(self._full_proximity_metrics(student_data, assigned_students))
        school_frl_context = aggregates.school_frl_context
        non_designated_school_frl_context = (
            aggregates.non_designated_school_frl_context
        )

        # DIVERSITY METRICS
        metrics[
            "#GE programs that have more than 0% and less than 10% of their capacity as African American or Pacific Islander students"
        ] = self._ge_aapi_range_from_counts(
            aggregates.ge_aapi_counts,
            self._ge_program_capacities,
            0.1,
            lower_threshold=0.0001,
            smaller_strict=True,
            percentage=True,
        )
        metrics[
            "#GE programs that have exactly 0 African American or Pacific Islander students"
        ] = self._ge_aapi_range_from_counts(
            aggregates.ge_aapi_counts,
            self._ge_program_capacities,
            1,
            lower_threshold=0,
            smaller_strict=True,
            percentage=False,
        )
        metrics[
            "#GE programs that have 1-4 African American or Pacific Islander students"
        ] = self._ge_aapi_range_from_counts(
            aggregates.ge_aapi_counts,
            self._ge_program_capacities,
            4,
            lower_threshold=1,
            smaller_strict=False,
            percentage=False,
        )

        for x in [10, 15, -10, -15]:
            (col1, col2, col3) = (
                f"#Schools {'above' if x >= 0 else 'below'} {x}% district FRL",
                f"Schools {'above' if x >= 0 else 'below'} {x}% district FRL",
                f"#Students in schools {'above' if x >= 0 else 'below'} {x}% district FRL",
            )
            (metrics[col1], metrics[col2], metrics[col3]) = (
                self._school_frl_range_from_context(
                    x / 100.0,
                    assigned_students,
                    school_frl_context,
                )
            )
            (col1, col2, col3) = (
                f"#Schools {'above' if x >= 0 else 'below'} {x}% district FRL (Non-Designated)",
                f"Schools {'above' if x >= 0 else 'below'} {x}% district FRL (Non-Designated)",
                f"#Students in schools {'above' if x >= 0 else 'below'} {x}% district FRL  (Non-Designated)",
            )
            (metrics[col1], metrics[col2], metrics[col3]) = (
                self._school_frl_range_from_context(
                    x / 100.0,
                    non_designated_students,
                    non_designated_school_frl_context,
                )
            )

        aalpi_students = aggregates.student_groups["AALPI"]
        metrics["AALPI in school with +10% FRL"] = self.poverty_concentration(
            assigned_students,
            aalpi_students,
            0.1,
            school_frl_context=school_frl_context,
        )
        metrics["AALPI in school with +15% FRL"] = self.poverty_concentration(
            assigned_students,
            aalpi_students,
            0.15,
            school_frl_context=school_frl_context,
        )
        metrics["AALPI in school with -10% FRL"] = self.poverty_concentration(
            assigned_students,
            aalpi_students,
            -0.1,
            school_frl_context=school_frl_context,
        )
        metrics["AALPI in school with -15% FRL"] = self.poverty_concentration(
            assigned_students,
            aalpi_students,
            -0.15,
            school_frl_context=school_frl_context,
        )

        ge_students = aggregates.ge_students
        ge_aalpi_students = aggregates.ge_aalpi_students
        ge_groups = ge_students.groupby("assignment")

        (
            metrics["#GE programs above +10% district FRL"],
            metrics["Proportion of GE programs above +10% district FRL"],
        ) = self.ge_frl_range(
            ge_students, ge_groups, 0.1, frl_context=aggregates.ge_frl_context
        )
        (
            metrics["#GE programs above +15% district FRL"],
            metrics["Proportion of GE programs above +15% district FRL"],
        ) = self.ge_frl_range(
            ge_students, ge_groups, 0.15, frl_context=aggregates.ge_frl_context
        )
        (
            metrics["#GE programs above +15% district FRL (Non-Designated)"],
            _,
        ) = self.ge_frl_range(
            aggregates.ge_non_designated_students,
            aggregates.ge_non_designated_students.groupby("assignment"),
            0.15,
            frl_context=aggregates.ge_non_designated_program_frl_context,
        )
        (
            metrics["#GE programs below -10% district FRL"],
            metrics["Proportion of GE programs below -10% district FRL"],
        ) = self.ge_frl_range(
            ge_students, ge_groups, -0.1, frl_context=aggregates.ge_frl_context
        )
        (
            metrics["#GE programs below -15% district FRL"],
            metrics["Proportion of GE programs below -15% district FRL"],
        ) = self.ge_frl_range(
            ge_students, ge_groups, -0.15, frl_context=aggregates.ge_frl_context
        )

        metrics["AALPI in GE programs with +10% FRL"] = (
            self.metric_ge_FRL_concentration(
                ge_students,
                ge_aalpi_students,
                ge_groups,
                0.1,
                frl_context=aggregates.ge_frl_context,
            )
        )
        metrics["AALPI in GE programs with +15% FRL"] = (
            self.metric_ge_FRL_concentration(
                ge_students,
                ge_aalpi_students,
                ge_groups,
                0.15,
                frl_context=aggregates.ge_frl_context,
            )
        )
        metrics["AALPI in GE programs with -10% FRL"] = (
            self.metric_ge_FRL_concentration(
                ge_students,
                ge_aalpi_students,
                ge_groups,
                -0.1,
                frl_context=aggregates.ge_frl_context,
            )
        )

        metrics["AALPI in GE programs with -15% FRL"] = (
            self.metric_ge_FRL_concentration(
                ge_students,
                ge_aalpi_students,
                ge_groups,
                -0.15,
                frl_context=aggregates.ge_frl_context,
            )
        )

        designated_aalpi_students = designated_students[
            designated_students["ethnicity"].isin(AALPI)
        ]
        if len(designated_students) == 0:
            metrics["AALPI in designated"] = np.nan
        else:
            metrics["AALPI in designated"] = len(designated_aalpi_students) / len(
                designated_students
            )

        high_income_composition = aggregates.high_income_composition
        low_income_composition = aggregates.low_income_composition

        (
            metrics[f"#Schools with +10% High Income ({medium_income})"],
            metrics[f"Prop schools with +10% High Income ({medium_income})"],
        ) = self.high_income_school_range(
            assigned_students,
            medium_income,
            0.1,
            composition=high_income_composition,
        )
        (
            metrics[f"#Schools with +15% High Income ({medium_income})"],
            metrics[f"Prop schools with +15% High Income ({medium_income})"],
        ) = self.high_income_school_range(
            assigned_students,
            medium_income,
            0.15,
            composition=high_income_composition,
        )
        (
            metrics[f"#Schools with -10% High Income ({medium_income})"],
            metrics[f"Prop schools with -10% High Income ({medium_income})"],
        ) = self.high_income_school_range(
            assigned_students,
            medium_income,
            -0.1,
            composition=high_income_composition,
        )
        (
            metrics[f"#Schools with -15% High Income ({medium_income})"],
            metrics[f"Prop schools with -15% High Income ({medium_income})"],
        ) = self.high_income_school_range(
            assigned_students,
            medium_income,
            -0.15,
            composition=high_income_composition,
        )

        metrics[f"Prop AALPI in +10% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                0.1,
                composition=high_income_composition,
            )
        )
        metrics[f"Prop AALPI in +15% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                0.15,
                composition=high_income_composition,
            )
        )
        metrics[f"Prop AALPI in -10% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                -0.1,
                composition=high_income_composition,
            )
        )
        metrics[f"Prop AALPI in -15% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                -0.15,
                composition=high_income_composition,
            )
        )

        metrics[f"Avg Prop AALPI in +10% High Income Schools ({medium_income})"] = (
            self.avg_aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                0.1,
                composition=high_income_composition,
            )
        )
        metrics[f"Avg Prop AALPI in +15% High Income Schools ({medium_income})"] = (
            self.avg_aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                0.15,
                composition=high_income_composition,
            )
        )
        metrics[f"Avg Prop AALPI in -10% High Income Schools ({medium_income})"] = (
            self.avg_aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                -0.1,
                composition=high_income_composition,
            )
        )
        metrics[f"Avg Prop AALPI in -15% High Income Schools ({medium_income})"] = (
            self.avg_aalpi_in_high_income_schools(
                assigned_students,
                medium_income,
                -0.15,
                composition=high_income_composition,
            )
        )

        metrics[f"Prop AALPI in -10% Low Income Schools ({low_income})"] = (
            self.aalpi_in_low_income_schools(
                assigned_students,
                low_income,
                -0.10,
                composition=low_income_composition,
            )
        )

        metrics[f"Prop AALPI in -15% Low Income Schools ({low_income})"] = (
            self.aalpi_in_low_income_schools(
                assigned_students,
                low_income,
                -0.15,
                composition=low_income_composition,
            )
        )

        metrics[f"Avg Prop FRL in +10% High Income Schools ({medium_income})"] = (
            self.avg_frl_in_high_income_schools(
                assigned_students,
                medium_income,
                0.1,
                composition=high_income_composition,
            )
        )
        metrics[f"Avg Prop FRL in +15% High Income Schools ({medium_income})"] = (
            self.avg_frl_in_high_income_schools(
                assigned_students,
                medium_income,
                0.15,
                composition=high_income_composition,
            )
        )

        # CHOICE METRICS

        metrics.update(self._utility_metrics(student_data))
        metrics["#Unassigned"] = student_data[student_data["programno"] == 0].shape[0]
        metrics["Unassigned"] = self._safe_ratio(
            student_data[student_data["programno"] == 0].shape[0],
            student_data.shape[0],
        )
        all_designated = assigned_students["designation"].mean()
        metrics["#Designated"] = assigned_students["designation"].sum()
        metrics["Designated"] = all_designated
        metrics["num_overage_seats"] = int(assigned_students["overage_seat"].sum())
        metrics["overage"] = aggregates.overage

        dico_student_type = {
            "All Assigned": assigned_students,
            "Non-Designated": non_designated_students,
            #  "Designated": designated_students,
        }

        for student_type, data_student in dico_student_type.items():
            top_choice_outcomes = self._top_choice_outcomes(
                data_student, "rank", range(1, 4)
            )
            mechanism_choice_outcomes = self._top_choice_outcomes(
                data_student, "mechanism_rank", range(1, 4)
            )
            for rank in range(1, 4):
                self._record_choice_outcome(
                    metrics,
                    f"Prop Top {rank} choice ({student_type})",
                    top_choice_outcomes[rank],
                )
            metrics[f"Mean Choice ({student_type})"] = data_student["rank"].mean()
            metrics[f"Median Choice ({student_type})"] = data_student["rank"].median()
            for rank in range(1, 4):
                self._record_choice_outcome(
                    metrics,
                    f"Top {rank} in-zone choice ({student_type})",
                    mechanism_choice_outcomes[rank],
                )

            metrics[f"Distance Av ({student_type})"] = data_student[
                "assignment_dist"
            ].mean()
            metrics[f"Distance Median ({student_type})"] = data_student[
                "assignment_dist"
            ].median()

        high_frl_assigned = aggregates.student_groups["High FRL"]
        low_frl_assigned = aggregates.student_groups["Low FRL"]

        dico_income = {
            f"Income below {medium_income}": class_below_medium_income,
            f"Income above {medium_income}": class_above_medium_income,
        }

        white_students = aggregates.student_groups["White"]
        for ethnicity in ["Black", "Hispanic"]:
            ethnic_students = aggregates.student_groups[ethnicity]
            metrics[f"{ethnicity}/White exposure to AALPI"] = self.metrics_segregation(
                ethnic_students, white_students, aalpi_students, enrollment
            )
            metrics[f"{ethnicity}/White exposure to poverty"] = (
                self.metrics_segregation(
                    ethnic_students,
                    white_students,
                    high_frl_assigned,
                    enrollment,
                )
            )
            # New: exposure difference to low FRL
            metrics[f"{ethnicity}/White exposure to low FRL"] = (
                self.metrics_segregation(
                    ethnic_students,
                    white_students,
                    low_frl_assigned,
                    enrollment,
                )
            )

        # Absolute exposure metrics (no difference between groups)
        # Exposure to high FRL and low FRL for key groups
        exposure_groups = {
            "AALPI": aalpi_students,
            "White": white_students,
            "Black": aggregates.student_groups["Black"],
            "Hispanic": aggregates.student_groups["Hispanic"],
            "High FRL": high_frl_assigned,
            "Low FRL": low_frl_assigned,
        }
        school_exposures = aggregates.school_exposures

        for group_name, group_students in exposure_groups.items():
            # Exposure to AALPI
            metrics[f"{group_name} exposure to AALPI"] = self.metrics_exposure(
                group_students,
                aalpi_students,
                enrollment,
                school_exposure=school_exposures["AALPI"],
            )
            # Exposure to high FRL (poverty)
            metrics[f"{group_name} exposure to high FRL"] = self.metrics_exposure(
                group_students,
                high_frl_assigned,
                enrollment,
                school_exposure=school_exposures["High FRL"],
            )
            # Exposure to low FRL
            metrics[f"{group_name} exposure to low FRL"] = self.metrics_exposure(
                group_students,
                low_frl_assigned,
                enrollment,
                school_exposure=school_exposures["Low FRL"],
            )

        # Precompute school FRL probabilities for efficiency
        # Average probability of FRL students in each school
        district_frl_prob, school_frl_probs = school_frl_context

        # Precalculate schools meeting relative thresholds
        # Schools with FRL > District Avg + 10%
        schools_high_frl_rel = school_frl_probs[
            school_frl_probs > district_frl_prob + 0.1
        ].index
        # Schools with FRL > District Avg + 15%
        schools_high_frl_rel_15 = school_frl_probs[
            school_frl_probs > district_frl_prob + 0.15
        ].index
        # Schools with FRL > District Avg
        schools_avg_frl_rel = school_frl_probs[
            school_frl_probs > district_frl_prob
        ].index
        all_student_top_choice_outcomes = (
            aggregates.all_student_top_choice_outcomes
        )
        for rank, outcome in all_student_top_choice_outcomes.items():
            self._record_choice_outcome(
                metrics,
                f"Prop Top {rank} choice (All Students)",
                outcome,
            )

        for group in FULL_REPORT_GROUPS + list(dico_income.keys()):
            students = aggregates.student_groups[group]
            distance_values = students["assignment_dist"].to_numpy()
            rank_values = students["rank"].to_numpy()
            in_zone_rank_values = students["mechanism_rank"].to_numpy()
            designation_values = students["designation"].to_numpy()
            designated_mask = designation_values == 1
            non_designated_mask = designation_values == 0
            non_designated_group = students[non_designated_mask]
            top_choice_outcomes = self._top_choice_outcomes(
                students,
                "rank",
                range(1, 11),
            )

            metrics[f"Number of assigned students ({group})"] = len(students)

            # --- New FRL Metrics ---
            if len(students) > 0:
                # 1. Exposure to High FRL (threshold 0.5)
                # Note: This is exposure to PEERS who are High FRL
                metrics[f"{group} exposure to high FRL"] = self.metrics_exposure(
                    students,
                    high_frl_assigned,
                    enrollment,
                    school_exposure=school_exposures["High FRL"],
                )

                # 2. Exposure to FRL (prob)
                # Average school FRL probability for students in this group
                # Maps each student's school to its avg FRL prob, then takes mean over group
                metrics[f"{group} exposure to FRL prob"] = (
                    students["assigned school"].map(school_frl_probs).mean()
                )

                # 3. Relative High FRL (>+10% dist FRL)
                # Proportion of group in schools with > Dist + 10% FRL
                metrics[f"{group} in school with +10% FRL"] = (
                    students["assigned school"].isin(schools_high_frl_rel).mean()
                )

                metrics[f"{group} in school with +15% FRL"] = (
                    students["assigned school"].isin(schools_high_frl_rel_15).mean()
                )

                # 4. FRL district average (Relative FRL > Avg)
                # Proportion of group in schools with > Dist Avg FRL
                metrics[f"{group} in school with > avg FRL"] = (
                    students["assigned school"].isin(schools_avg_frl_rel).mean()
                )
            else:
                metrics[f"{group} exposure to high FRL"] = np.nan
                metrics[f"{group} exposure to FRL prob"] = np.nan
                metrics[f"{group} in school with +10% FRL"] = np.nan
                metrics[f"{group} in school with +15% FRL"] = np.nan
                metrics[f"{group} in school with > avg FRL"] = np.nan
            # -----------------------

            # --- Added Choice Metrics for Subgroups ---
            for rank in range(1, 4):
                self._record_choice_outcome(
                    metrics,
                    f"Prop Top {rank} choice ({group})",
                    top_choice_outcomes[rank],
                )
            if len(students) > 0:
                metrics[f"Distance Av ({group})"] = students["assignment_dist"].mean()
                metrics[f"Distance Median ({group})"] = students[
                    "assignment_dist"
                ].median()
            else:
                metrics[f"Distance Av ({group})"] = np.nan
                metrics[f"Distance Median ({group})"] = np.nan
            # ------------------------------------------

            metrics[f"Dissimilarity ({group})"] = self.metric_dissimilarity(
                students, enrollment
            )
            self._add_program_assignment_counts(metrics, group, students)

            (
                _,
                _,
                metrics[f"#Students in schools above +15% district FRL ({group})"],
            ) = self._school_frl_range_from_context(
                0.15,
                students,
                school_frl_context,
            )
            metrics[f"Prop students in schools above +15% district FRL ({group})"] = (
                self._safe_ratio(
                    metrics[f"#Students in schools above +15% district FRL ({group})"],
                    len(students),
                )
            )

            if group != "All Assigned":
                metrics[f"Number of designated students ({group})"] = int(
                    designated_mask.sum()
                )
                non_designated_choice_outcomes = self._top_choice_outcomes(
                    non_designated_group,
                    "rank",
                    range(1, 4),
                )
                self._record_choice_outcome(
                    metrics,
                    f"Prop Top 1 choice Non-Designated ({group})",
                    non_designated_choice_outcomes[1],
                )

                self._record_choice_outcome(
                    metrics,
                    f"Prop Top 2 choice Non-Designated ({group})",
                    non_designated_choice_outcomes[2],
                )

                for i in range(1, 11):
                    self._record_choice_outcome(
                        metrics,
                        f"Proportion of students in top {i} ({group})",
                        top_choice_outcomes[i],
                    )
                self._record_choice_outcome(
                    metrics,
                    f"Prop Top 3 choice Non-Designated ({group})",
                    non_designated_choice_outcomes[3],
                )
                non_designated_mechanism_outcomes = self._top_choice_outcomes(
                    non_designated_group,
                    "mechanism_rank",
                    range(1, 4),
                )
                for rank in range(1, 4):
                    self._record_choice_outcome(
                        metrics,
                        f"Top {rank} in-zone choice Non-Designated ({group})",
                        non_designated_mechanism_outcomes[rank],
                    )
                metrics[f"Mean Choice Non-Designated ({group})"] = (
                    non_designated_group["rank"].mean()
                )
                metrics[f"Median Choice Non-Designated ({group})"] = (
                    non_designated_group["rank"].median()
                )
                metrics[f"Distance Av Non-Designated ({group})"] = (
                    non_designated_group["assignment_dist"].mean()
                )
                metrics[f"Distance Median Non-Designated ({group})"] = (
                    non_designated_group["assignment_dist"].median()
                )
                metrics[f"Distance < 0.5 Non-Designated ({group})"] = (
                    self._array_mean(
                        distance_values[non_designated_mask] < 0.5
                    )
                )
                metrics[f"Distance > 3 Non-Designated ({group})"] = (
                    self._array_mean(distance_values[non_designated_mask] > 3)
                )

            # Evaluate for all groups, append together as distances.
            distance_over_3 = distance_values > 3
            metrics[f"Prop Distance > 3 and designated ({group})"] = (
                self._array_mean(distance_over_3 & designated_mask)
            )
            metrics[f"Prop Distance > 3 and Top 3 choice, non-designated ({group})"] = (
                self._array_mean(
                    distance_over_3 & (rank_values <= 3) & non_designated_mask
                )
            )
            metrics[f"Prop Distance > 3 and non-designated ({group})"] = (
                self._array_mean(distance_over_3 & non_designated_mask)
            )

            metrics[f"Prop Distance > 3 and Rank>=5 ({group})"] = (
                self._array_mean(distance_over_3 & (rank_values >= 5))
            )
            metrics[f"Prop Distance > 3 and in-zone Rank>=5 ({group})"] = (
                self._array_mean(distance_over_3 & (in_zone_rank_values >= 5))
            )

            metrics[f"Prop Distance > 3 and Rank>=5 Non-Designated ({group})"] = (
                self._array_mean(
                    distance_over_3[non_designated_mask]
                    & (rank_values[non_designated_mask] >= 5)
                )
            )
            metrics[
                f"Prop Distance > 3 and in-zone Rank>=5 Non-Designated ({group})"
            ] = (
                self._array_mean(
                    distance_over_3[non_designated_mask]
                    & (in_zone_rank_values[non_designated_mask] >= 5)
                )
            )

            # --- Prop Distance > 3 and Rank>=4 ---
            metrics[f"Prop Distance > 3 and Rank>=4 ({group})"] = (
                self._array_mean(distance_over_3 & (rank_values >= 4))
            )
            metrics[f"Prop Distance > 3 and Rank>=4 Non-Designated ({group})"] = (
                self._array_mean(
                    distance_over_3[non_designated_mask]
                    & (rank_values[non_designated_mask] >= 4)
                )
            )
            # Prop Distance > 3 and (Rank>=4 or designated)
            metrics[f"Prop Distance > 3 and (Rank>=4 or designated) ({group})"] = (
                self._array_mean(
                    distance_over_3 & ((rank_values >= 4) | designated_mask)
                )
            )
            # Prop Distance > 3 and (Rank>=5 or designated)
            metrics[f"Prop Distance > 3 and (Rank>=5 or designated) ({group})"] = (
                self._array_mean(
                    distance_over_3 & ((rank_values >= 5) | designated_mask)
                )
            )
            # Prop Distance > 3 and (in-zone Rank>=5 or designated)
            metrics[
                f"Prop Distance > 3 and (in-zone Rank>=5 or designated) ({group})"
            ] = (
                self._array_mean(
                    distance_over_3
                    & ((in_zone_rank_values >= 5) | designated_mask)
                )
            )

            # --- Variance metrics ---
            if len(students) > 0:
                metrics[f"Variance of rank ({group})"] = students["rank"].var()
                metrics[f"Variance of in-zone rank ({group})"] = students[
                    "mechanism_rank"
                ].var()
                metrics[f"Variance of distance ({group})"] = students[
                    "assignment_dist"
                ].var()
            else:
                metrics[f"Variance of rank ({group})"] = np.nan
                metrics[f"Variance of in-zone rank ({group})"] = np.nan
                metrics[f"Variance of distance ({group})"] = np.nan

        # --- Theil's H segregation index ---
        metrics["Theil H"] = self.metric_theil(student_data)

        # Add diagnostic metrics
        diagnostic_metrics = self.eval_diagnostic_metrics(student_data)
        metrics.update(diagnostic_metrics)

        return pd.Series(metrics)

    def eval_assignment_paper_metrics(self):
        """Compatibility alias; new callers should choose basic or full."""
        if self._mode == "basic":
            return self.eval_assignment_basic()
        return self.eval_assignment_full()

    def eval_diagnostic_metrics(self, student_data=None):
        """Compute diagnostic metrics for trend analysis.

        Returns:
            Dict[str, float]: Dictionary of diagnostic metrics including:
                - count_students_{ethnicity}: Count per ethnic group
                - count_students_Total: Total student count
                - enrollment_count_{ethnicity}: Assigned students per group
                - enrollment_rate_{ethnicity}: Assignment rate per group
                - utilization_{program_type}: Capacity utilization per type
        """
        metrics = {}
        if student_data is None:
            student_data = self.student_data
        programs = self.programs
        assigned_mask = student_data["programno"].to_numpy() > 0

        # ===== Demographic Metrics =====
        if "ethnicity" in student_data.columns:
            ethnicity = student_data["ethnicity"].to_numpy()
            counts = Counter(ethnicity)
            enrolled_counts = Counter(ethnicity[assigned_mask])
            for group in DIAGNOSTIC_ETHNICITIES:
                total = int(counts.get(group, 0))
                enrolled = int(enrolled_counts.get(group, 0))
                metrics[f"count_students_{group}"] = total
                metrics[f"enrollment_count_{group}"] = enrolled
                metrics[f"enrollment_rate_{group}"] = self._safe_ratio(enrolled, total)
            metrics["count_students_Total"] = len(student_data)
            metrics["enrollment_count_Total"] = int(assigned_mask.sum())
            metrics["enrollment_rate_Total"] = self._safe_ratio(
                assigned_mask.sum(), len(student_data)
            )

        # ===== Capacity Utilization Metrics =====
        metrics.update(
            {
                f"utilization_{program_type}": np.nan
                for program_type in DIAGNOSTIC_PROGRAM_TYPES
            }
        )
        metrics["utilization_rate_avg"] = np.nan
        if "programno" in student_data.columns and not programs.empty:
            if (
                "programcodes" in student_data.columns
                and "program_id" in programs.columns
            ):
                enrollment_counts = Counter(
                    student_data["programcodes"].to_numpy()[assigned_mask]
                )
                program_keys = programs["program_id"].astype(str).to_numpy()
            elif "programno" in programs.columns:
                enrollment_counts = Counter(
                    student_data["programno"].to_numpy()[assigned_mask]
                )
                program_keys = programs["programno"].to_numpy()
            else:
                enrollment_counts = None

            if enrollment_counts is not None and "capacity" in programs.columns:
                enrollment = np.fromiter(
                    (enrollment_counts.get(key, 0) for key in program_keys),
                    dtype=float,
                    count=len(program_keys),
                )
                capacities = programs["capacity"].to_numpy(dtype=float)
                positive_capacity = capacities > 0
                if capacities.sum() > 0:
                    metrics["utilization_rate_avg"] = np.mean(
                        enrollment[positive_capacity] / capacities[positive_capacity]
                    )

                type_col = (
                    "program_type"
                    if "program_type" in programs.columns
                    else "type"
                )
                if type_col in programs.columns:
                    program_types = programs[type_col].to_numpy()
                    for program_type in np.unique(program_types):
                        type_mask = program_types == program_type
                        cap = capacities[type_mask].sum()
                        if cap > 0:
                            metrics[f"utilization_{program_type}"] = (
                                enrollment[type_mask].sum() / cap
                            )

        return metrics
