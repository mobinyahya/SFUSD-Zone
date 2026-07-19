"""Data interface for programs."""

import numpy as np
import pandas as pd

from student_assignment.definitions import (
    K8S,
    LANGUAGE_PATHWAYS,
    SPECIAL_PROGRAMS,
)


class Programs:
    def __init__(
        self, program_data_file: str, program_codes_file: str, config: dict
    ):
        self._config = config
        print("Loading program data from:", program_data_file)
        self.program_df = pd.read_csv(program_data_file)
        if self._config.get("remove-special-lps", False):
            self._remove_special_lps()
        else:
            # Record the line to keep for programs to remove columns in utility files.
            # if some columns are removed. Record as None to indicate no columns are removed.
            self.only_keep_cols = None
        self._set_up_programno(program_codes_file)
        self.num_programs = len(self.program_df)
        self._program_type2indices = None
        self._school2indices = None
        if config["grade"] == "06":
            self.fix_k8_capacities()
        elif config["grade"] == "09":
            self._selective_hs_capacities()

    def _set_up_programno(self, program_codes_file: str):
        """Set up program indices and codes as a column in the dataframe.

        Args:
            program_codes_file (str): path to file mapping program codes to indices
        """
        if "programno" in self.program_df.columns:
            numeric_programno = pd.to_numeric(
                self.program_df["programno"], errors="coerce"
            )
            self.program_df["programno"] = numeric_programno
            self.program_df = self.program_df.dropna(
                subset=["programno"]
            ).copy()
            self.program_df["programno"] = self.program_df["programno"].astype(
                int
            )

            unique_programnos = sorted(
                self.program_df["programno"].unique().tolist()
            )
            contiguous_programnos = list(range(1, len(self.program_df) + 1))
            if unique_programnos != contiguous_programnos:
                self.program_df = self.program_df.reset_index(drop=True)
                self.program_df["programno"] = self.program_df.index + 1

            self.indices = dict(
                zip(self.program_df["program_id"], self.program_df["programno"])
            )
            self.codes = dict(
                zip(self.program_df["programno"], self.program_df["program_id"])
            )
        else:
            program_codes = pd.read_csv(program_codes_file)
            self.indices = dict(
                zip(program_codes["code"], program_codes["index"])
            )
            self.codes = dict(
                zip(program_codes["index"], program_codes["code"])
            )
            self.program_df["programno"] = self.program_df.program_id.replace(
                self.indices
            )

    def _remove_special_lps(self):
        """Remove special programs."""
        program_data = self.program_df
        self.only_keep_cols = program_data.index[
            ~program_data["program_type"].isin(SPECIAL_PROGRAMS)
        ].to_numpy()
        self.program_df = program_data[
            ~program_data["program_type"].isin(SPECIAL_PROGRAMS)
        ].reset_index(drop=True)
        # Reset programno as needed
        self.program_df["programno"] = self.program_df.index + 1

    def index(self, program: str, quiet: bool = False) -> int:
        """Get program index for a given program code.

        Args:
            program (str): program code
            quiet (bool, optional): If True, do not print error message if program code is not found. Defaults to False.

        Returns:
            int: program index
        """
        if program not in self.indices:
            if not quiet:
                print("Programs.index: no such program: ", program)
            return -1
        return self.indices[program]

    def index_list(self, programs: list) -> list:
        """Get program indices for a list of program codes.

        Args:
            programs (list): list of program codes

        Returns:
            list: list of program indices
        """
        try:
            idxs = [self.indices[x] for x in programs]
        except KeyError:
            return [self.indices[x] for x in programs if x in self.indices]
        return idxs

    def fix_k8_capacities(self):
        """Fix K8 capacities to be equal to round 1 assignments.

        For 6th grade, the capacity files give the full school capacity at K8s, but most of those seats are already occupied by students continuing from 5th grade. We use round 1 assignments as a proxy for the number of seats that are actually free for 6th grade assignment.
        """
        self.program_df.loc[self.program_df.school_id.isin(K8S), "capacity"] = (
            self.program_df.loc[
                self.program_df.school_id.isin(K8S), "r1_assigned"
            ].fillna(0)
        )
        self.program_df.loc[
            self.program_df.school_id.isin(K8S), "r2_capacity"
        ] = self.program_df.loc[
            self.program_df.school_id.isin(K8S), "r1_assigned"
        ].fillna(0)

    def _selective_hs_capacities(self):
        """Adjust capacities at selective HS, who accept all eligible students who propose."""
        self.program_df.loc[
            self.program_df.program_id == "815-GE-09", "capacity"
        ] = 10000  # SOTA
        # lowell not selective in 2021-22 or 2022-23
        if self._config["year"] not in [21, 22]:
            self.program_df.loc[
                self.program_df.program_id == "697-GE-09", "capacity"
            ] = 10000  # Lowell

    def set_program_capacities(self, capacities: np.ndarray | pd.Series):
        """Manually set program capacities.

        Args:
            capacities (Union[np.ndarray, pd.Series]): array or series with capacities for each program
        """
        self.program_df["capacity"] = capacities

    @property
    def school(self) -> pd.Series:
        """Return series mapping from each program to the school it is in."""
        return self.program_df["school_id"]

    @property
    def capacity(self) -> pd.Series:
        """Return series mapping from each program to the EPC capacities."""
        return self.program_df["capacity"].fillna(0)

    @property
    def capacityRnd2(self) -> pd.Series:
        """Return series mapping from each program to the EPC capacities."""
        return self.program_df["r2_capacity"]

    @property
    def capacityFinal(self) -> pd.Series:
        """Return series mapping from each program to the EPC capacities."""
        try:
            return self.program_df["r3_capacity"]
        except KeyError:
            return self.program_df["capacity"]

    @property
    def program_type(self) -> pd.Series:
        """Return series mapping from each program to program type."""
        return self.program_df["program_type"]

    @property
    def program_type_to_indices(self) -> dict:
        """Get dictionary of program types to list of all indices of programs of that type.

        Returns:
            dict: dictionary of program types to list of program indices
        """
        if self._program_type2indices is None:
            self._program_type2indices = dict(
                self.program_df.groupby("program_type").programno.agg(list)
            )
        return self._program_type2indices

    @property
    def school_to_indices(self) -> dict:
        """Get dictionary of school ids to list of all indices of programs in that school.

        Returns:
            dict: dictionary of school ids to list of program indices
        """
        if self._school2indices is None:
            self._school2indices = dict(
                self.program_df.groupby("school_id").programno.agg(list)
            )
        return self._school2indices

    def language_program_indices(self) -> list:
        """Get list of indices of all language pathways programs.

        Returns:
            list: list of program indices
        """
        lps = LANGUAGE_PATHWAYS.intersection(
            set(self.program_type_to_indices.keys())
        )
        return [x for y in lps for x in self.program_type_to_indices[y]]

    def citywide_program_indices(self, citywide_schools: list) -> list:
        """Get list of indices of all citywide programs.

        Args:
            citywide_schools (list): list of school ids that are citywide

        Returns:
            list: list of program indices at those schools
        """
        citywide_schools = [
            x
            for x in citywide_schools
            if x in self.program_df["school_id"].unique().tolist()
        ]
        citywide_programs = {
            y for x in citywide_schools for y in self.school_to_indices[x]
        }
        return list(citywide_programs)

    def citywide_language_program_indices(self, citywide_schools: list) -> list:
        """Get list of indices of all citywide language pathways programs.

        Args:
            citywide_schools (list): list of school ids that are citywide

        Returns:
            list: list of program indices at those schools
        """
        citywide_programs = set(self.citywide_program_indices(citywide_schools))
        lps = set(self.language_program_indices())
        return list(citywide_programs.intersection(lps))
