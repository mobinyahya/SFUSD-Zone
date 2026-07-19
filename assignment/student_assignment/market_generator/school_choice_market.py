import pathlib

import numpy as np

from ..configerator import Configerator
from ..data_interfaces import Programs, Schools, Students, Zones
from ..definitions import (
    BLOCK_DATA_FILE,
    PROGRAM_CODES_FILE,
    PROGRAM_DATA_FILE,
    SCHOOL_DATA_FILE,
    STUDENT_DATA_FILE,
    Path,
)
from .utility_model import UtilityModel


class SchoolChoiceMarket:
    def __init__(self, estimate_path: str = None, config: dict = None):
        """Initialize a school choice market.

        Args:
            estimate_path (str, optional): Path to the estimated preferences. Defaults to None, which
                means we use the path specified in the config.
            config (dict, optional): Configerator config. Defaults to None, which means we read the
                config from the default location.
        """
        self.yaml = None
        if config is None:
            self.configurator = Configerator()
            self.config = self.configurator.config
            self.yaml = self.configurator._original_config
        else:
            self.config = config

        np.random.seed(
            self.config["random-seed"]
        )  # set again for each subconfig
        # Project files are located relative to paths["sfusd"]
        self.input_path_generator = Path(self.config["paths"]["sfusd"])

        self._initialize_market_data()
        self._initialize_utility_model(estimate_path)

    def _initialize_market_data(self):
        """Initialize data interface objects."""
        self._initialize_schools_and_programs()
        self._initialize_students()
        aa_schools = self.schools.school_df.loc[
            self.schools.school_df.category == "Attendance"
        ]
        self.zones = Zones(
            self.config,
            attendance_area_schools=aa_schools,
            programs=self.programs,
            students=self.students,
        )

    def _initialize_schools_and_programs(self):
        """Initialize schools and programs data interfaces."""
        gr = f"{self.config['grade']}_" if self.config["grade"] != "KG" else ""
        school_data_file = self.config["paths"].get(
            "school-data",
            SCHOOL_DATA_FILE.format(
                gr,
                self.config["year"],
                self.config["year"] + 1,
            ),
        )
        school_data_file = self.input_path_generator.absolute_path(
            school_data_file
        )

        program_data_file = self.config["paths"].get(
            "program-data",
            PROGRAM_DATA_FILE.format(
                gr,
                self.config["year"],
                self.config["year"] + 1,
            ),
        )
        program_data_file = self.input_path_generator.absolute_path(
            program_data_file
        )

        program_codes_file = self.input_path_generator.absolute_path(
            PROGRAM_CODES_FILE
        )

        self.programs = Programs(
            program_data_file, program_codes_file, self.config
        )
        self.schools = Schools(school_data_file, self.programs)

    def _initialize_students(self):
        """Initialize students data interface."""
        student_data_file = self.config["paths"].get(
            "student-data",
            STUDENT_DATA_FILE.format(
                self.config["year"], self.config["year"] + 1
            ),
        )
        student_data_file = self.input_path_generator.absolute_path(
            student_data_file
        )
        school_location_file = self.config["paths"].get(
            "school-data",
            SCHOOL_DATA_FILE.format(
                f"{self.config['grade']}_"
                if self.config["grade"] != "KG"
                else "",
                self.config["year"],
                self.config["year"] + 1,
            ),
        )
        school_location_file = self.input_path_generator.absolute_path(
            school_location_file
        )

        block_data_file = self.input_path_generator.absolute_path(
            BLOCK_DATA_FILE
        )
        self.students = Students(
            student_data_file=student_data_file,
            programs=self.programs,
            school_data_file=school_location_file,
            block_data_file=block_data_file,
            config=self.config,
        )
        self.n = self.students.n
        self.num_programs = self.students.num_programs

    def _initialize_utility_model(self, estimate_path: str):
        """Initialize utility model.

        Args:
            estimate_path (str): Path to the estimated preferences. If None, use the path
                specified in the config.
        """
        if self.config["utility-model"]["enable"] or self.config[
            "utility-model"
        ].get("read-precomuted-umodel-prefs", False):
            input_estimate_path = (
                estimate_path
                if estimate_path is not None
                else self.config["paths"]["estimate-path"]
            )
            precomputed_prefs = self.config["utility-model"].get(
                "read-precomuted-umodel-prefs", False
            )
            if precomputed_prefs:
                codex_paths = [
                    self.config["paths"]["student-codex"],
                    self.config["paths"]["program-codex"],
                ]
            else:
                codex_paths = None
            self.umodel = UtilityModel(
                input_estimate_path,
                self.programs,
                self.students,
                read_prefs=precomputed_prefs,
                codex_paths=codex_paths,
            )
            self.umodel.student_data = self.students.student_data

    def get_supplemental_zone_path_list(self) -> list:
        """Get paths for language program, special education, or other citywide zones.

        Returns:
            list: list of paths for supplemental zones
        """
        paths = []

        for zone_name in self.config.get("citywide-or-lp", []):
            if zone_name not in self.config["paths"]["citywide-or-lp-zones"]:
                raise ValueError(
                    f"Citywide zone {zone_name} not found in local paths config. Please add it under "
                    f"'paths' --> 'citywide-or-lp-zones'."
                )
            paths.append(
                pathlib.Path(
                    self.config["paths"]["citywide-or-lp-zones"][zone_name]
                ).expanduser()
            )

        return paths
