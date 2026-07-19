"""Created 7/27/20.

@author Max Allman

Class containing utility model for market simulator
"""

import os
import pathlib

import numpy as np
import pandas as pd


class UtilityModel:
    def __init__(
        self,
        estimate_path,
        programs,
        students,
        read_prefs=False,
        codex_paths=None,
    ):
        self.programs = programs
        self.students = students
        self.estimate_path = estimate_path
        self.read_prefs = read_prefs
        self._base_utility_cache_key = None
        self._base_utility_matrix = None
        if codex_paths is not None:
            self.student_codex_path = codex_paths[0]
            self.program_codex_path = codex_paths[1]

    @staticmethod
    def _index_cache_key(values):
        if values is None:
            return None
        return tuple(np.asarray(values).tolist())

    def _load_utility_from_csv(self):
        """Load the utility matrix from a csv dataframe. The csv dataframe should have
        indices with studentno and columns with program ids.

        Returns:
            np.ndarray: the loaded utility matrix ordered by student and program indexing.
        """
        estimates_df = pd.read_csv(self.estimate_path)
        # Remove students and programs that are not used, and order the dataframe
        # by student and program index. The studentno in the input csv dataframe
        # is in the format <year>-<studentno>.
        year = estimates_df.studentno.apply(lambda x: int(x.split("-")[0]))[0]
        estimates_df.studentno = estimates_df.studentno.apply(
            lambda x: int(x.split("-")[1])
        )
        print(f"Loaded utility for students in year {year}.")
        estimates_df = estimates_df.set_index("studentno")
        required_students = self.students.student_data.index.to_numpy()
        available_students = set(estimates_df.index)
        missing_students = [
            s for s in required_students if s not in available_students
        ]
        if missing_students:
            print(
                f"  Warning: {len(missing_students)} students missing "
                f"from estimates — filling with -inf."
            )
            missing_rows = pd.DataFrame(
                -np.inf,
                index=missing_students,
                columns=estimates_df.columns,
            )
            missing_rows.index.name = "studentno"
            estimates_df = pd.concat([estimates_df, missing_rows])
        estimates_df = estimates_df.loc[required_students]
        estimates_df = estimates_df.iloc[
            estimates_df.index.map(self.students.studentno2idx).argsort()
        ]
        required_programs = self.programs.program_df.program_id.to_numpy()
        available_programs = set(estimates_df.columns)
        missing_programs = [
            p for p in required_programs if p not in available_programs
        ]
        if missing_programs:
            print(
                f"  Warning: {len(missing_programs)} programs missing "
                f"from estimates — filling with -inf: "
                f"{missing_programs[:10]}{'...' if len(missing_programs) > 10 else ''}"
            )
            for prog in missing_programs:
                estimates_df[prog] = -np.inf
        estimates_df = estimates_df[required_programs]
        estimates_df.rename(columns=self.programs.indices, inplace=True)
        sorted_columns = sorted(estimates_df.columns, key=int)
        estimates_df = estimates_df[sorted_columns]
        return estimates_df.to_numpy()

    def _load_base_utility_matrix(self, rows_to_keep=None, cols_to_keep=None):
        cache_key = (
            self.estimate_path,
            self._index_cache_key(rows_to_keep),
            self._index_cache_key(cols_to_keep),
        )
        if cache_key == self._base_utility_cache_key:
            return self._base_utility_matrix

        if ".csv" in str(self.estimate_path):
            mus = self._load_utility_from_csv()
        else:
            mus = np.load(pathlib.Path(self.estimate_path).expanduser())
            # Keep only the rows or columns if specified. This would be used for when
            # we only kept a subset of students and/or programs corresponding to a
            # subset of all the rows and/or columns in the utility matrix.
            if rows_to_keep is not None:
                mus = mus[rows_to_keep]
            if cols_to_keep is not None:
                mus = mus[:, cols_to_keep]

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
        if self.read_prefs:
            if ".csv" in str(self.estimate_path):
                mus = self._load_utility_from_csv()
            else:
                mus = np.load(pathlib.Path(self.estimate_path).expanduser())
                if rows_to_keep is not None:
                    mus = mus[rows_to_keep]
                if cols_to_keep is not None:
                    mus = mus[:, cols_to_keep]
        else:
            mus = self._load_base_utility_matrix(rows_to_keep, cols_to_keep)

        n, p = mus.shape
        print(f"Loaded utility matrix of shape {mus.shape}.")
        print(
            f"Number of students: {self.students.n}, number of programs: {self.programs.num_programs}."
        )
        # Verify that the utility matrix has the same size as student and programs.
        error_msg = "the utility matrix length does not match the number of {}."
        assert p == self.programs.num_programs, error_msg.format("programs")
        assert n == self.students.n, error_msg.format("students")

        if not self.read_prefs:
            # The contribution of the random part of the preference model.
            # gumbel_scale=0 → deterministic argmax (no noise); >0 → MNL draw.
            if gumbel_scale == 0:
                utilities = mus.copy()
            else:
                utilities = np.array(
                    mus + np.random.gumbel(0, gumbel_scale, (n, p))
                )
            # num_eligible = np.where(utilities > -100, 1, 0).sum(axis=1)

            self.original_utilities = utilities
            self.original_preferences = np.argsort(
                -utilities, axis=1
            ) + np.ones(utilities.shape)
            # for i, num in enumerate(num_eligible):
            #     self.original_preferences[i, num:] = 0
            # print(self.original_preferences)

        else:
            student_codex = np.load(os.path.expanduser(self.student_codex_path))
            program_codex = np.load(os.path.expanduser(self.program_codex_path))
            student_nos = list(self.student_data.index)

            choice = np.load(self.estimate_path)
            choice = choice[iteration, :, :]
            n2, p2 = choice.shape
            self.original_preferences = np.zeros([n, p])
            for i in range(n2):
                student_no = student_codex[i]
                if student_no in student_nos:
                    student_idx = student_nos.index(student_no)
                    for j in range(p2):
                        program_code = program_codex[int(choice[i, j])]
                        program_idx = int(
                            self.programs.indices[program_code]
                        )  # Ranges from 1 to 159
                        self.original_preferences[student_idx, j] = program_idx
            utilities = np.zeros([n, p])
            for i in range(n):
                for j in range(p):
                    program_idx = int(self.original_preferences[i, j] - 1)
                    if program_idx >= 0:
                        if utilities[i, program_idx] == 0:
                            utilities[i, program_idx] = 1000 - j

            count = 0
            for i in range(n):
                if self.original_preferences[i, 60] == 0:
                    count += 1
            print("COUNT:", count)

            self.original_utilities = utilities

    def save_utility_matrix(self, save_path):
        """Save the computed utility matrix to a file."""
        if save_path.endswith(".csv"):
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
