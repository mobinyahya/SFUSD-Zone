"""Created 7/27/20.

@author Max Allman

Class containing utility model for market simulator
"""

import csv
import pathlib
import re

import numpy as np
import pandas as pd


class UtilityModel:
    def __init__(self, estimate_path, programs, students):
        self.programs = programs
        self.students = students
        self.estimate_path = estimate_path
        self._base_utility_cache_key = None
        self._base_utility_matrix = None

    @staticmethod
    def _index_cache_key(values):
        if values is None:
            return None
        return tuple(np.asarray(values).tolist())

    @staticmethod
    def _identity_text(value) -> str:
        if pd.isna(value):
            raise ValueError("Utility identities cannot be null.")
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        identity = str(value).strip()
        if not identity:
            raise ValueError("Utility identities cannot be empty.")
        return identity

    @classmethod
    def _normalize_labeled_identity(cls, value, required_identities: set[str]) -> str:
        identity = cls._identity_text(value)
        if identity in required_identities:
            return identity
        year_prefixed = re.fullmatch(r"(?:\d{2}|\d{4})-(.+)", identity)
        if year_prefixed:
            return year_prefixed.group(1)
        return identity

    @staticmethod
    def _validate_csv_header(path: pathlib.Path) -> None:
        with open(path, newline="", encoding="utf-8-sig") as utility_file:
            try:
                header = next(csv.reader(utility_file))
            except StopIteration as exc:
                raise ValueError("Utility CSV is empty.") from exc
        duplicate_columns = list(
            dict.fromkeys(column for column in header if header.count(column) > 1)
        )
        if duplicate_columns:
            raise ValueError(
                f"Utility CSV has duplicate columns: {duplicate_columns[:10]}"
            )
        if header.count("studentno") != 1:
            raise ValueError("Utility CSV must contain exactly one studentno column.")

    def _load_utility_from_csv(self):
        """Load the utility matrix from a csv dataframe. The csv dataframe should have
        indices with studentno and columns with program ids.

        Returns:
            np.ndarray: the loaded utility matrix ordered by student and program indexing.
        """
        path = pathlib.Path(self.estimate_path).expanduser()
        self._validate_csv_header(path)
        estimates_df = pd.read_csv(path, dtype={"studentno": "string"})

        required_student_values = self.students.student_data.index.tolist()
        required_students = [
            self._identity_text(studentno) for studentno in required_student_values
        ]
        if len(required_students) != len(set(required_students)):
            raise ValueError("Required student identities are not unique.")
        required_student_set = set(required_students)

        student_labels = [
            self._normalize_labeled_identity(studentno, required_student_set)
            for studentno in estimates_df.pop("studentno")
        ]
        duplicate_students = pd.Index(student_labels)[
            pd.Index(student_labels).duplicated(keep=False)
        ].unique()
        if len(duplicate_students):
            raise ValueError(
                "Utility CSV has duplicate student rows: "
                f"{duplicate_students.tolist()[:10]}"
            )
        estimates_df.index = pd.Index(student_labels, name="studentno")

        required_programs = [
            self._identity_text(program_id)
            for program_id in self.programs.program_df["program_id"]
        ]
        if len(required_programs) != len(set(required_programs)):
            raise ValueError("Required program identities are not unique.")
        required_program_set = set(required_programs)
        program_labels = [
            self._normalize_labeled_identity(program_id, required_program_set)
            for program_id in estimates_df.columns
        ]
        duplicate_programs = pd.Index(program_labels)[
            pd.Index(program_labels).duplicated(keep=False)
        ].unique()
        if len(duplicate_programs):
            raise ValueError(
                "Utility CSV has duplicate program columns: "
                f"{duplicate_programs.tolist()[:10]}"
            )
        estimates_df.columns = program_labels

        available_students = set(estimates_df.index)
        missing_students = sorted(required_student_set - available_students)
        if missing_students:
            raise ValueError(
                f"Utility CSV is missing required student rows: {missing_students[:10]}"
            )
        available_programs = set(estimates_df.columns)
        missing_programs = sorted(required_program_set - available_programs)
        if missing_programs:
            raise ValueError(
                "Utility CSV is missing required program columns: "
                f"{missing_programs[:10]}"
            )

        aligned = estimates_df.loc[required_students, required_programs]
        numeric = aligned.apply(pd.to_numeric, errors="coerce")
        numeric_values = numeric.to_numpy(dtype=float)
        invalid = np.isnan(numeric_values) | np.isposinf(numeric_values)
        if invalid.any():
            row, column = np.argwhere(invalid)[0]
            raise ValueError(
                "Utility CSV contains a non-numeric or NaN utility, or positive "
                "infinity, at "
                f"student {required_students[row]}, program "
                f"{required_programs[column]}."
            )
        return numeric_values

    def _load_utility_from_npy(self, rows_to_keep=None, cols_to_keep=None):
        path = pathlib.Path(self.estimate_path).expanduser()
        mus = np.load(path, allow_pickle=False)
        if mus.ndim != 2:
            raise ValueError(
                f"Utility NPY must be two-dimensional; found shape {mus.shape}."
            )
        try:
            # Saved matrices are already reduced to the active students and
            # programs. Apply source-data selectors only to unreduced matrices.
            if rows_to_keep is not None and mus.shape[0] != self.students.n:
                mus = mus[np.asarray(rows_to_keep)]
            if cols_to_keep is not None and mus.shape[1] != self.programs.num_programs:
                mus = mus[:, np.asarray(cols_to_keep)]
        except (IndexError, TypeError) as exc:
            raise ValueError("Utility NPY subset indices are invalid.") from exc

        expected_shape = (self.students.n, self.programs.num_programs)
        if mus.shape != expected_shape:
            raise ValueError(
                f"Utility NPY shape {mus.shape} does not match required shape "
                f"{expected_shape}."
            )
        if np.iscomplexobj(mus):
            raise ValueError("Utility NPY must contain real numeric utilities.")
        try:
            mus = np.asarray(mus, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Utility NPY contains non-numeric utilities.") from exc
        if np.isnan(mus).any() or np.isposinf(mus).any():
            raise ValueError("Utility NPY contains NaN or positive-infinite utilities.")
        return mus

    def _load_base_utility_matrix(self, rows_to_keep=None, cols_to_keep=None):
        cache_key = (
            self.estimate_path,
            self._index_cache_key(rows_to_keep),
            self._index_cache_key(cols_to_keep),
        )
        if cache_key == self._base_utility_cache_key:
            return self._base_utility_matrix

        if pathlib.Path(self.estimate_path).suffix.lower() == ".csv":
            mus = self._load_utility_from_csv()
        else:
            mus = self._load_utility_from_npy(rows_to_keep, cols_to_keep)

        expected_shape = (self.students.n, self.programs.num_programs)
        if mus.shape != expected_shape:
            raise ValueError(
                f"Utility matrix shape {mus.shape} does not match required "
                f"shape {expected_shape}."
            )

        self._base_utility_cache_key = cache_key
        self._base_utility_matrix = mus
        return mus

    def draw_utility_model_randomness(
        self,
        iteration=None,
        rows_to_keep=None,
        cols_to_keep=None,
        gumbel_scale=1.0,
    ):
        mus = self._load_base_utility_matrix(rows_to_keep, cols_to_keep)

        if not np.isfinite(gumbel_scale) or gumbel_scale < 0:
            raise ValueError("gumbel_scale must be a finite non-negative number.")
        n, p = mus.shape
        if gumbel_scale == 0:
            utilities = mus.copy()
        else:
            utilities = np.array(mus + np.random.gumbel(0, gumbel_scale, (n, p)))

        self.original_utilities = utilities
        self.original_preferences = np.argsort(-utilities, axis=1) + 1

    def save_utility_matrix(self, save_path):
        """Save the computed utility matrix to a file."""
        save_path = pathlib.Path(save_path).expanduser()
        if save_path.suffix.lower() == ".csv":
            # Get student numbers (names/IDs)
            student_names = [
                self.students.idx2studentno[i] for i in range(self.students.n)
            ]

            # Get program IDs/names
            program_names = self.programs.program_df["program_id"].tolist()

            # Convert to DataFrame with proper indices and columns
            df = pd.DataFrame(
                self.original_utilities,
                index=student_names,
                columns=program_names,
            )
            df.index.name = "studentno"
            df.to_csv(save_path)
        else:
            # Save as numpy array
            np.save(save_path, self.original_utilities)
