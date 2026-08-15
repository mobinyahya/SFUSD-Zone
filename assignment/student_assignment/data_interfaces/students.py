"""Data interface for students."""

import ast
import csv
import os
import pathlib
import re
import warnings

import numpy as np
import pandas as pd

from ..definitions.constants import (
    LANGUAGE_PATHWAY_PRIORITIES,
    SPECIAL_PROGRAMS,
)


class Students:
    def __init__(
        self,
        student_data_file: pathlib.Path,
        programs,
        school_data_file: pathlib.Path,
        block_data_file: pathlib.Path,
        config,
    ):
        self.programs = programs
        self.program_df = programs.program_df
        self.school_data_file = school_data_file
        self.student_data_file = student_data_file
        self.block_data_file = block_data_file
        self.config = config

        self.output_path = pathlib.Path(
            self.config["paths"]["student-save"]
        ).expanduser()

        self.year = self.config["year"]
        self.grade = self._normalize_grade(self.config["grade"])
        self.num_programs = programs.num_programs
        self._create_output_path()
        self.qualified_program_dict = None

        self.student_data = self._build_student_data()
        if self.config.get("remove-special-lps", False):
            self._remove_students_with_special_lps()
        else:
            # Record the line to keep for students to remove rows in utility files.
            # if some rows are removed. Record as None to indicate no rows are removed.
            self.only_keep_rows = None
        if self.student_data.empty:
            raise ValueError(
                f"Student data contains no students for grade {self.grade}."
            )
        self._validate_ranked_programs()
        self.n = len(self.student_data.index)  # number of student
        self._round_participation = self._calc_round_participation()
        self.distance_data = self.get_distances()

        # subset students (note: student data needs to be indexed by studentno)
        self.student_data.set_index("studentno", inplace=True)

        # create idx2studentno and studentno2idx functions (note: student data needs to be indexed by student no)
        self.student_data.reset_index(inplace=True)
        self.idx2studentno = dict(
            zip(self.student_data.index, self.student_data["studentno"])
        )
        self.studentno2idx = dict(
            zip(self.student_data["studentno"], self.student_data.index)
        )
        self.student_data.set_index("studentno", inplace=True)
        self.get_diversity_categories()

        self._prefs = {}
        self._sibling = None
        self._prek = None

    def _create_output_path(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_grade(value) -> str:
        """Normalize numeric grades so values such as 6 and '06' compare."""
        if pd.isna(value):
            return ""
        text = str(value).strip().upper()
        try:
            number = float(text)
        except ValueError:
            return text
        if np.isfinite(number) and number.is_integer():
            return str(int(number)).zfill(2)
        return text

    @staticmethod
    def _validate_student_identities(df: pd.DataFrame) -> None:
        if "studentno" not in df.columns:
            raise ValueError("Student data is missing required column 'studentno'.")
        values = df["studentno"]
        missing = values.isna() | values.astype("string").str.strip().eq("")
        if missing.fillna(True).any():
            raise ValueError("Student data contains a missing studentno identity.")
        duplicates = values[values.duplicated(keep=False)]
        if not duplicates.empty:
            duplicate_values = duplicates.astype(str).unique().tolist()
            raise ValueError(
                "Student data contains duplicate studentno identities: "
                f"{duplicate_values[:10]}"
            )

    def _build_student_data(
        self,
    ):
        """Load student data table and student location data."""
        st_df = pd.read_csv(
            self.student_data_file, low_memory=False
        )  # , index_col=0)
        if "grade" not in st_df.columns:
            raise ValueError("Student data is missing required column 'grade'.")
        normalized_grades = st_df["grade"].map(self._normalize_grade)
        st_df = st_df.loc[normalized_grades == self.grade].copy()
        st_df.reset_index(inplace=True, drop=True)
        self._validate_student_identities(st_df)

        school_rounds = {
            int(match.group(1))
            for column in st_df.columns
            if (match := re.fullmatch(r"r(\d+)_ranked_idschool", column))
        }
        program_rounds = {
            int(match.group(1))
            for column in st_df.columns
            if (match := re.fullmatch(r"r(\d+)_programs", column))
        }
        if school_rounds != program_rounds:
            raise ValueError(
                "Ranked school/program columns must occur in pairs; "
                f"missing school rounds={sorted(program_rounds - school_rounds)}, "
                f"missing program rounds={sorted(school_rounds - program_rounds)}."
            )
        self.rounds = max(school_rounds, default=0)

        st_df = self._make_cols_lists(st_df)
        return st_df

    def _str_to_list(self, value):
        """Parse a submitted ranked-school list, ignoring only empty tokens."""
        if isinstance(value, (list, tuple, np.ndarray)):
            items = list(value)
        elif pd.isna(value) or not str(value).strip():
            return []
        else:
            text = str(value).strip()
            if not (text.startswith("[") and text.endswith("]")):
                raise ValueError(f"Expected a bracketed list, got {value!r}.")
            items = text[1:-1].split(",")

        schools = []
        for item in items:
            if isinstance(item, str):
                item = item.strip().strip("'\"")
            if pd.isna(item) or item == "":
                continue
            try:
                number = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ranked school ID {item!r}.") from exc
            if not np.isfinite(number) or not number.is_integer():
                raise ValueError(f"Invalid ranked school ID {item!r}.")
            schools.append(int(number))
        return schools

    @staticmethod
    def _programs_to_list(value):
        """Safely parse a submitted ranked-program list."""
        if isinstance(value, (list, tuple, np.ndarray)):
            items = list(value)
        elif pd.isna(value) or not str(value).strip():
            return []
        else:
            try:
                items = ast.literal_eval(str(value))
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Expected a bracketed program list, got {value!r}."
                ) from exc
            if not isinstance(items, (list, tuple)):
                raise ValueError(f"Expected a bracketed program list, got {value!r}.")

        programs = []
        for item in items:
            if pd.isna(item):
                raise ValueError("Ranked program lists cannot contain null values.")
            program = str(item).strip()
            if program:
                programs.append(program)
        return programs

    def _make_cols_lists(self, df):
        """Helper function used to format the ranking lists as read in from the
        student data file.
        """
        for round in range(1, self.rounds + 1):
            # format column name
            col1 = f"r{round}_ranked_idschool"
            col2 = f"r{round}_programs"
            if col1 not in df.columns:
                continue
            parsed_schools = []
            parsed_programs = []
            for idx in df.index:
                studentno = df.at[idx, "studentno"]
                try:
                    schools = self._str_to_list(df.at[idx, col1])
                    programs = self._programs_to_list(df.at[idx, col2])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid round {round} preferences for student "
                        f"{studentno}: {exc}"
                    ) from exc
                if len(schools) != len(programs):
                    raise ValueError(
                        f"Student {studentno} round {round} has "
                        f"{len(schools)} ranked schools but {len(programs)} "
                        "ranked programs."
                    )
                parsed_schools.append(schools)
                parsed_programs.append(programs)
            df[col1] = pd.Series(parsed_schools, index=df.index, dtype=object)
            df[col2] = pd.Series(parsed_programs, index=df.index, dtype=object)

        return df

    def _validate_ranked_programs(self) -> None:
        known_programs = set(self.programs.indices)
        for round in range(1, self.rounds + 1):
            school_column = f"r{round}_ranked_idschool"
            program_column = f"r{round}_programs"
            if school_column not in self.student_data.columns:
                continue
            for _, row in self.student_data.iterrows():
                ranked = [
                    f"{school}-{program}-{self.grade}"
                    for school, program in zip(row[school_column], row[program_column])
                ]
                unknown = [
                    program for program in ranked if program not in known_programs
                ]
                if unknown:
                    raise ValueError(
                        f"Student {row['studentno']} round {round} ranked "
                        f"unknown program IDs: "
                        f"{list(dict.fromkeys(unknown))[:10]}"
                    )

    def _remove_students_with_special_lps(self):
        """Remove students who rank special programs in round 1."""
        student_data = self.student_data
        student_data["is_special"] = student_data["r1_programs"].apply(
            lambda x: (
                False
                if str(x) == "nan"
                else len(set(x).intersection(SPECIAL_PROGRAMS)) > 0
            )
        )
        # Record the rows that we keep to filter utility models later.
        self.only_keep_rows = student_data.index[
            student_data["is_special"] == 0
        ].to_numpy()
        student_data = student_data[student_data["is_special"] == 0].drop(
            columns=["is_special"]
        )
        student_data.reset_index(inplace=True, drop=True)
        self.student_data = student_data

    def _make_student_preferences(self, round, code2idx):
        """Compute student preferences for a specified round in matrix form."""
        st_df = self.student_data
        prefs = np.zeros((self.n, self.num_programs), dtype=int)
        col1 = f"r{round}_ranked_idschool"
        col2 = f"r{round}_programs"
        if col1 not in st_df.columns or col2 not in st_df.columns:
            raise ValueError(
                f"Student data has no preference columns for round {round}."
            )
        for indexcounter, (_, row) in enumerate(st_df.iterrows()):
            codes = [
                f"{school}-{program}-{self.grade}"
                for school, program in zip(row[col1], row[col2])
            ]
            prog_idxs = code2idx(codes)
            if len(prog_idxs) != len(codes):
                raise ValueError(
                    f"Student {row.name} round {round} contains an unknown program ID."
                )
            prefs[indexcounter, 0 : len(prog_idxs)] = prog_idxs
        return prefs

    def student_preferences(self, round, code2idx):
        """Get student preference list of lists for a given round. To get the
        student with index i's 2nd choice, use prefs[i][1] (zero indexed)
        Preferences are cached only in memory because an unlabeled matrix cannot
        be safely matched to a different student or program ordering.
        """
        if round in self._prefs.keys():
            return self._prefs[round]
        pr = self._make_student_preferences(round, code2idx)
        self._prefs[round] = pr
        return pr

    def _make_distance_ranking(self, filename):
        """Create (number of students) by (number of programs) array indicating
        the pairwise distance between each student and school. Use program code
        file to translate between program code and index, and self.studentno2idx
        dictionary to go from student number to index.
        """
        # codes = pd.read_csv(self.program_codes_file)
        # codes["school_id"] = codes["code"].apply(lambda x: int(x[:3]))
        codes = self.program_df[["program_id", "school_id"]]

        # get school latitude and longitude DataFrame
        sch_latlong = os.path.expanduser(self.school_data_file)
        sc_ll = pd.read_csv(sch_latlong)

        codes = codes.merge(sc_ll, how="left", on="school_id")
        codes.loc[:, "key"] = 0

        tmp = self.student_data[["studentno", "latitude", "longitude"]].copy()
        # if student lat lon is 0, 0, treat that as a missing value
        tmp["latitude"] = tmp["latitude"].replace({0: np.nan})
        tmp["longitude"] = tmp["longitude"].replace({0: np.nan})
        tmp.loc[:, "key"] = 0

        tmp = tmp.merge(codes, how="outer", on="key")

        def get_distance(row):
            return (
                6371.01
                * np.arccos(
                    np.sin(row["lat"] * np.pi / 180)
                    * np.sin(row["latitude"] * np.pi / 180)
                    + np.cos(row["lat"] * np.pi / 180)
                    * np.cos(row["latitude"] * np.pi / 180)
                    * np.cos((row["lon"] - row["longitude"]) * np.pi / 180)
                )
                * 0.621371
            )

        tmp["distance"] = tmp.apply(get_distance, axis=1)

        table = pd.pivot_table(
            tmp,
            values="distance",
            index=["studentno"],
            columns=["program_id"],
            aggfunc="sum",
        )

        # Manually zero out distances of students with any distance larger than
        # 10, indicating that the students' addresses are outside of SFUSD.
        table.loc[table.ge(10).any(axis=1), :] = 0

        # don't save over computed distances if subset of students
        table.to_csv(filename)
        return table

    @staticmethod
    def _identity_key(value) -> str:
        if pd.isna(value):
            raise ValueError("Identity values cannot be null.")
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        key = str(value).strip()
        if not key:
            raise ValueError("Identity values cannot be empty.")
        return key

    @staticmethod
    def _read_distance_cache(filename: pathlib.Path) -> pd.DataFrame:
        with open(filename, newline="", encoding="utf-8-sig") as cache_file:
            try:
                header = next(csv.reader(cache_file))
            except StopIteration as exc:
                raise ValueError("Distance cache is empty.") from exc
        duplicate_columns = list(
            dict.fromkeys(column for column in header if header.count(column) > 1)
        )
        if duplicate_columns:
            raise ValueError(
                f"Distance cache has duplicate columns: {duplicate_columns[:10]}"
            )
        if header.count("studentno") != 1:
            raise ValueError(
                "Distance cache must contain exactly one studentno column."
            )
        return pd.read_csv(filename, index_col="studentno")

    def _align_distances(self, dist: pd.DataFrame) -> pd.DataFrame:
        required_students = self.student_data["studentno"].tolist()
        required_student_keys = [
            self._identity_key(studentno) for studentno in required_students
        ]
        actual_student_keys = [
            self._identity_key(studentno) for studentno in dist.index
        ]
        if len(actual_student_keys) != len(set(actual_student_keys)):
            raise ValueError("Distance cache has duplicate studentno rows.")

        required_programs = self.program_df["program_id"].astype(str).tolist()
        actual_programs = [str(program) for program in dist.columns]
        if len(actual_programs) != len(set(actual_programs)):
            raise ValueError("Distance cache has duplicate program columns.")

        missing_students = sorted(set(required_student_keys) - set(actual_student_keys))
        missing_programs = sorted(set(required_programs) - set(actual_programs))
        if missing_students or missing_programs:
            raise ValueError(
                "Distance cache is missing required identities; "
                f"students={missing_students[:10]}, programs={missing_programs[:10]}."
            )

        aligned = dist.copy()
        aligned.index = actual_student_keys
        aligned.columns = actual_programs
        aligned = aligned.loc[required_student_keys, required_programs]
        aligned = aligned.apply(pd.to_numeric, errors="coerce")
        values = aligned.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                "Distance cache contains non-numeric or non-finite values."
            )
        aligned.index = pd.Index(required_students, name="studentno")
        return aligned

    def get_distances(self):
        """Load student distances to each program if already calculated,
        otherwise compute them. Description of array in _make_distance_ranking
        docstring.
        """
        if pathlib.Path(self.student_data_file).name[:-9] == "drop_optout":
            filename = (
                self.output_path
                / "student_program_distances_dropoptout_{}_{}{}.csv".format(
                    self.grade, self.year, self.year + 1
                )
            )
        else:
            filename = (
                self.output_path
                / "student_program_distances_{}_{}{}.csv".format(
                    self.grade, self.year, self.year + 1
                )
            )

        if filename.is_file():
            try:
                return self._align_distances(self._read_distance_cache(filename))
            except (
                OSError,
                UnicodeDecodeError,
                csv.Error,
                pd.errors.ParserError,
                ValueError,
            ) as exc:
                warnings.warn(
                    f"Ignoring invalid distance cache {filename}: {exc}. "
                    "Recomputing it.",
                    stacklevel=2,
                )

        dist = self._make_distance_ranking(filename)
        try:
            return self._align_distances(dist)
        except ValueError as exc:
            raise ValueError(f"Recomputed distance data is invalid: {exc}") from exc

    def _calc_round_participation(self):
        """Array of size 3 for each student indicating in which rounds the
        student participated.
        """
        participated = np.zeros((self.n, self.rounds), dtype=int)

        def find_null(x):
            return 1 if len(x) > 0 else 0

        for r in range(self.rounds):
            name = f"r{r + 1}_ranked_idschool"  # column names 1 indexed
            if name not in self.student_data.columns:
                participated[:, r] = 0
            else:
                participated[:, r] = self.student_data[name].apply(find_null)
        return participated

    @property
    def round_participation(self):
        """Array of size 3 for each student indicating in which rounds the
        student participated.
        """
        return self._round_participation

    @property
    def first_round(self):
        """Array of size 3 for each student indicating in which rounds the
        student participated.
        """
        return np.argmax(self._round_participation, axis=1)

    @property
    def frl(self):
        """Return a pd.Series of length n containing the free and reduced lunch
        percentage of that student's block (0 for missing data).
        """
        free = self.student_data["freelunch_prob"].fillna(value=0)
        reduced = self.student_data["reducedlunch_prob"].fillna(value=0)
        return free + reduced

    @property
    def ctip(self):
        """Return an array of length n containing 1 when the student has CTIP1
        priority, 0 otherwise (0 for missing data as well).
        """
        vec = (
            self.student_data["ctip1"]
            .fillna(value=0)
            .astype("int64")
            .to_numpy()
        )
        return np.ones((self.n, self.num_programs)) * vec[:, np.newaxis]

    @property
    def new_ctip(self):
        """Return an array of length n containing 1 when the student has new_CTIP1
        priority, 0 otherwise (0 for missing data as well).
        """
        # Configure via paths.new-ctip-path; legacy cluster path as default.
        new_ctip_path = self.config["paths"].get(
            "new-ctip-path",
            "/share/data/school_choice/Data/Tie-breakers/ETB_2024.npy",
        )
        new_ctip = np.load(new_ctip_path)
        self.student_data["new_ctip1"] = self.student_data[
            "census_block"
        ].apply(lambda x: 1 if x in new_ctip else 0)
        vec = (
            self.student_data["new_ctip1"]
            .fillna(value=0)
            .astype("int64")
            .to_numpy()
        )
        return np.ones((self.n, self.num_programs)) * vec[:, np.newaxis]

    @property
    def new_ctip_blockgroup(self):
        """Return an array of length n containing 1 when the student has new_CTIP1 (Blockgroup solution)
        priority, 0 otherwise (0 for missing data as well).
        """
        # Configure via paths.new-ctip-blockgroup-path; legacy default.
        new_ctip_bg_path = self.config["paths"].get(
            "new-ctip-blockgroup-path",
            "/share/data/school_choice/Data/Tie-breakers/ETB_2024_BlockGroup.npy",
        )
        new_ctip_bg = np.load(new_ctip_bg_path)
        self.student_data["new_ctip_blockgroup1"] = self.student_data[
            "census_blockgroup"
        ].apply(lambda x: 1 if x in new_ctip_bg else 0)
        vec = (
            self.student_data["new_ctip_blockgroup1"]
            .fillna(value=0)
            .astype("int64")
            .to_numpy()
        )
        return np.ones((self.n, self.num_programs)) * vec[:, np.newaxis]

    @property
    def language_designation(self):
        """Return an pd.Series of length n containing 1 when the student requested
        language program designation priority, 0 otherwise (0 for missing data
        as well), index is the student number.
        """
        return (
            self.student_data["requestprogramdesignation"]
            .fillna(value=0)
            .astype("int64")
        )

    @property
    def attendance_area(self):
        """Return a pd.Series of length n containing the attendance area of each
        student (0 if missing), index is the student number.
        """
        return (
            self.student_data["idschoolattendance"]
            .fillna(value=0)
            .astype("int64")
        )

    @property
    def enrolled(self):
        """Return a pd.Series of enrolled or not."""
        df = (
            self.student_data["enrolled_idschool"]
            .fillna(value=0)
            .astype("int64")
        )
        # df = df[df > 0] = 1
        return df

    @property
    def ethnicity(self):
        """Return an pd.Series of length n containing the student's resolved
        ethnicity code (epmpty string if missing), index is the student number.
        """
        return self.student_data["resolved_ethnicity"].fillna(value="")

    @property
    def bayview_to_all_ms(self):
        return self.student_data.bayview_to_all_ms.to_numpy()

    @property
    def bayview_to_brown_ms(self):
        return self.student_data.bayview_to_brown_ms.to_numpy()

    @property
    def brown_ms_to_hs(self):
        return self.student_data.brown_ms_to_hs.to_numpy()

    @property
    def zip_94124(self):
        return np.where(self.student_data.zipcode == 94124, 1, 0)

    @property
    def lowell_eligible(self):
        return self.student_data.lowell_ranked.to_numpy()

    @property
    def sota_eligible(self):
        return self.student_data.sota_ranked.to_numpy()

    def sibling(self, programs):
        """Return a (number of students) by (number of programs) array with 1s
        indicating that student has sibling priority at that program. Sibling
        priority is given to all programs at that school.
        """
        if self._sibling is not None:
            return self._sibling
        sibling = np.zeros((self.n, self.num_programs), dtype=int)
        df = self.student_data.dropna(subset=["sibling"])
        for idx, row in df.iterrows():
            sib_schools = eval(row["sibling"])
            st_idx = self.studentno2idx[idx]
            for school in sib_schools:
                if int(school) in programs.school_to_indices:
                    program_list = [
                        x - 1 for x in programs.school_to_indices[int(school)]
                    ]
                    sibling[st_idx, program_list] = 1
        self._sibling = sibling
        return sibling

    def prek(self):
        if self._prek is not None:
            return self._prek
        prek = np.zeros((self.n, self.num_programs), dtype=int)
        df = self.student_data.dropna(subset=["sibling"])
        for studentno, row in df.iterrows():
            prek_id = eval(row["aaprek"]) + eval(row["prek"])
            if prek_id:
                program_idx = (
                    self.programs.index(
                        f"{prek_id[0]}-GE-{self.config['grade']}"
                    )
                    - 1
                )
                st_idx = self.studentno2idx[studentno]
                prek[st_idx, program_idx] = 1
        self._prek = prek
        return prek

    def language_pathway_priority(self, program_type2indexes):
        language_pathway = np.zeros((self.n, self.num_programs), dtype=int)
        for i, pw in enumerate(self.student_data.previous_pathway):
            program_types = LANGUAGE_PATHWAY_PRIORITIES.get(pw, [])
            indices = [
                y - 1
                for x in program_types
                for y in program_type2indexes.get(x, [])
            ]
            language_pathway[i, indices] = 1
        return language_pathway

    def language_pathway_priority_kg(self, program_id2index):
        language_pathway = np.zeros((self.n, self.num_programs), dtype=int)
        for studentno, row in self.student_data.iterrows():
            for r in [1, 2, 3]:
                cohort = f"r{r}_cohortstring"
                if cohort not in row or pd.isna(row[cohort]):
                    continue
                for idx in [
                    i for i, x in enumerate(eval(row[cohort])) if "CL;" in x
                ]:
                    school = row[f"r{r}_ranked_idschool"][idx]
                    program = row[f"r{r}_programs"][idx]
                    program_id = f"{school}-{program}-{self.config['grade']}"
                    if program_id in program_id2index:
                        language_pathway[
                            self.studentno2idx[studentno],
                            program_id2index[program_id] - 1,
                        ] = 1
        return language_pathway

    def language_pathway_sibling(self, program_id2index):
        language_sibling = np.zeros((self.n, self.num_programs), dtype=int)
        for i, x in enumerate(self.student_data.currentlpsibling):
            indices = [
                program_id2index[i] - 1
                for i in eval(x)
                if i in program_id2index
            ]
            language_sibling[i, indices] = 1
        return language_sibling

    def msf(self, school2indices):
        msf_indicator = np.zeros((self.n, self.num_programs), dtype=int)
        for i, ms in enumerate(self.student_data.msf):
            if np.isnan(ms):
                continue
            indices = [x - 1 for x in school2indices[int(ms)]]
            msf_indicator[i, indices] = 1
        return msf_indicator

    def _make_program_type_lists(self, df):
        """Create column with each type of program applied to."""
        for round in range(1, self.rounds + 1):
            # format column name
            col = f"r{round}_programs"
            # format round rankings
            if col not in self.student_data.columns:
                continue
            df[col] = df[col].fillna("")
            # df[col] = df[col].apply(lambda x: [l[1:-1] for l in x if "" not in x])

            if round == 1:
                df["program_types"] = df[col]
            else:
                df["program_types"] = df["program_types"] + df[col]
        df["program_types"] = df["program_types"].apply(lambda x: np.unique(x))
        return df

    def get_qualified_programs_dict(self) -> dict:
        """Build a dictionary mapping each student ID to the type of language programs they are eligible for.

        Eligibility is based on home language and what types of programs they already applied for.

        Returns:
            dict: A dictionary mapping student ID to a list of program types they are eligible for.
        """
        # if already computed, return it
        if self.qualified_program_dict is not None:
            return self.qualified_program_dict
        if "program_types" not in self.student_data.columns:
            self.student_data = self._make_program_type_lists(self.student_data)

        homelang = "homelang" if self.year >= 21 else "homelang_desc"
        homelang2prog = {
            "CC-Chinese Cantonese": ["CN", "CB", "CT", "NC"],
            "CM-Chinese Mandarin": ["MN"],
            "SP-Spanish": ["SN", "SB", "NS"],
            "KO-Korean": ["KN"],
            "CC": ["CN", "CB", "CT", "NC"],
            "CM": ["MN"],
            "SP": ["SN", "SB", "NS"],
            "KO": ["KN"],
            "Cantonese": ["CN", "CB", "CT", "NC"],
            "Mandarin (Putonghua)": ["MN"],
            "Spanish": ["SN", "SB", "NS"],
            "Korean": ["KN"],
        }

        all_eligible = {"GE", "SE", "CE", "JE", "KE", "ME"}
        # all_eligible = {"GE", "SE", "CE", "JE", "KE", "ME",'SA', 'MS', 'AF', 'MM', 'TC', 'ED', 'AO'}
        # all_eligible = {"GE"}

        def combine(row):
            if any(
                program in SPECIAL_PROGRAMS for program in row["r1_programs"]
            ):
                return list(SPECIAL_PROGRAMS)
            else:
                if row[homelang] in homelang2prog:  # add home language programs
                    both = set(
                        homelang2prog[row[homelang]]
                        + list(row["program_types"])
                    ).union(all_eligible)
                else:  # otherwise just add program types they ranked and programs everyone is eligible for
                    both = set(row["program_types"]).union(all_eligible)
                for v in homelang2prog.values():  # if ranked a language program, can rank any language program of that type
                    if both.intersection(set(v)):
                        both.update(v)
                return list(both - {""})

        self.student_data["qualified"] = self.student_data.apply(
            combine, axis=1
        )

        self.qualified_program_dict = dict(
            zip(self.student_data.index.values, self.student_data["qualified"])
        )

        return self.qualified_program_dict

    def get_5_ctip_types(self) -> None:
        CTIPtypes = np.ones(
            [self.n]
        )  # CTIPtypes[i] is CTIP status of student i, from 1 to 5, 1 being highest priority
        block_data = pd.read_excel(
            self.block_data_file, sheet_name="block database"
        )
        block_data_list = list(block_data["Block"])
        for i in range(self.n):
            if (
                self.student_data["census_block"].iloc[i] > -1
            ):  # eliminates NaN values
                census_block = int(self.student_data["census_block"].iloc[i])
                if census_block in block_data_list:
                    row = block_data_list.index(census_block)
                    CTIP = block_data["CTIP_2013 assignment"].iloc[
                        row
                    ]  # String in the form 'CTIPX', where X is a number
                    CTIPnum = int(CTIP[4])
                    CTIPtypes[i] = CTIPnum
        self.student_data["CTIPtype"] = CTIPtypes

    def get_diversity_categories(self):
        """Create Diversity Categories from 2020 policy based on HOCidx1."""
        HOCidx1 = self.student_data.HOCidx1
        HOCidx1 = np.array(HOCidx1)
        HOCidx1 = np.nan_to_num(HOCidx1, nan=0.4)
        percentages3 = 100 * np.array(range(1, 3)) / 3
        percentages5 = 100 * np.array(range(1, 5)) / 5
        quantiles3 = np.percentile(HOCidx1, percentages3)
        quantiles5 = np.percentile(HOCidx1, percentages5)
        classes3 = np.searchsorted(quantiles3, HOCidx1)
        classes5 = np.searchsorted(quantiles5, HOCidx1)
        self.student_data["Diversity_Category3"] = classes3
        self.student_data["Diversity_Category5"] = classes5

    def get_ses_score(self):
        self.student_data["SES_score"] = (
            0.25 * self.student_data["N'hood SES Score"]
            + 0.25 * self.student_data["FRL Score"]
        )
        thresh33, thresh66 = np.percentile(
            self.student_data["SES_score"].dropna(), [33, 66]
        )
        self.student_data["SES_category"] = self.student_data[
            "SES_score"
        ].apply(lambda x: 1 if x < thresh33 else (2 if x < thresh66 else 3))
