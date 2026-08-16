import copy
import hashlib
import json
import pathlib
from collections.abc import Mapping

import numpy as np
from loaders import (
    ResolvedSource,
    load_program_records,
    load_scenario,
    load_school_records,
    load_student_records,
    read_csv_source,
)

from ..configerator import Configerator
from ..data_interfaces import Programs, Schools, Students, Zones
from .utility_model import UtilityModel


ASSIGNMENT_IMMUTABLE_SOURCE_ROLES = (
    "assignment.students",
    "assignment.programs",
    "assignment.schools",
    "assignment.school_coordinates",
    "assignment.program_codes",
    "assignment.estimate",
    "assignment.block_data",
    "assignment.new_ctip",
    "assignment.new_ctip_blockgroup",
    "assignment.geography.blocks",
    "assignment.geography.crosswalk",
)
ASSIGNMENT_OPTIONAL_IMMUTABLE_SOURCE_ROLES = (
    "assignment.programs.catalog",
    "assignment.geography.blockgroups",
    "assignment.geography.tracts",
)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _resolved_paths(value):
    if isinstance(value, ResolvedSource):
        return str(value.path)
    if isinstance(value, Mapping):
        return {str(key): _resolved_paths(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_resolved_paths(item) for item in value]
    raise TypeError(f"Unsupported resolved assignment source {value!r}.")


def assignment_source_identity(data_scenario) -> str:
    """Fingerprint immutable assignment inputs without loading their tables."""
    roles = list(ASSIGNMENT_IMMUTABLE_SOURCE_ROLES)
    if data_scenario.filter("assignment", "capacity_scenario") != "programs":
        roles.append("assignment.capacity")
    for role in ASSIGNMENT_OPTIONAL_IMMUTABLE_SOURCE_ROLES:
        try:
            data_scenario.resolved(role)
        except KeyError:
            continue
        roles.append(role)
    payload = {
        "assignment_filters": _plain(data_scenario.filters["assignment"]),
        "sources": data_scenario.source_manifest(roles),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SchoolChoiceMarket:
    def __init__(
        self,
        estimate_path: str = None,
        config: dict = None,
        configurator=None,
    ):
        """Initialize a school choice market.

        Args:
            estimate_path (str, optional): Path to the estimated preferences. Defaults to None, which
                means we use the path specified in the config.
            config (dict, optional): Configerator config. Defaults to None, which means we read the
                config from the default location.
            configurator (optional): Config source used by policy simulations. Cannot be combined
                with config.
        """
        if config is not None and configurator is not None:
            raise ValueError("Pass either config or configurator, not both.")
        if estimate_path is not None:
            raise ValueError(
                "Configure utility estimates with "
                "data.overrides.sources.assignment.estimate."
            )

        self.yaml = None
        if configurator is not None:
            self.configurator = configurator
        elif config is None:
            self.configurator = Configerator()
            self.yaml = self.configurator.original_config
        else:
            self.configurator = Configerator.from_config(config)
        self._materialize_config(self.configurator.config)
        self._validate_config(self.config)

        np.random.seed(
            self.config["random-seed"]
        )  # set again for each subconfig
        self._initialize_market_data()
        self._initialize_utility_model()

    def _materialize_config(self, external_config: dict) -> None:
        """Resolve strict external data config into internal policy conveniences."""
        validated = Configerator.from_config(external_config)
        self.external_config = copy.deepcopy(validated.config)
        self.data_scenario = load_scenario(self.external_config["data"])
        self.source_identity = assignment_source_identity(self.data_scenario)
        assignment_filters = self.data_scenario.filters["assignment"]

        config = copy.deepcopy(self.external_config)
        config["year"] = int(assignment_filters["year"][:2])
        config["grade"] = assignment_filters["grades"][0]
        config["special_programs"] = assignment_filters["special_programs"]

        paths = copy.deepcopy(config.get("paths", {}))
        scalar_roles = {
            "assignment.students": "student-data",
            "assignment.programs": "program-data",
            "assignment.schools": "school-data",
            "assignment.school_coordinates": "school-coordinate-data",
            "assignment.program_codes": "program-codes",
            "assignment.estimate": "estimate-path",
            "assignment.block_data": "block-data",
            "assignment.new_ctip": "new-ctip-path",
            "assignment.new_ctip_blockgroup": "new-ctip-blockgroup-path",
        }
        for role, path_key in scalar_roles.items():
            paths[path_key] = str(self.data_scenario.source(role).path)

        paths["zone-files"] = _resolved_paths(
            self.data_scenario.source_map("assignment.zones")
        )
        paths["citywide-or-lp-zones"] = _resolved_paths(
            self.data_scenario.source_map("assignment.citywide_zones")
        )
        try:
            paths["lotteries-path"] = str(
                self.data_scenario.source("assignment.lotteries").path
            )
        except KeyError:
            pass
        try:
            paths["peng-boosts"] = _resolved_paths(
                self.data_scenario.source_map("assignment.peng_boosts")
            )
        except KeyError:
            pass
        config["paths"] = paths

        provenance = {
            "scenario": self.data_scenario.id,
            "schema_version": self.data_scenario.schema_version,
            "roots": _plain(self.data_scenario.roots),
            "filters": _plain(self.data_scenario.filters),
            "sources": self.data_scenario.source_manifest()["sources"],
            "semantic_fingerprint": self.data_scenario.semantic_fingerprint,
            "source_fingerprint": self.data_scenario.source_fingerprint,
        }
        config["data-provenance"] = provenance
        self.config = config
        self.resolved_config = copy.deepcopy(self.external_config)

    @staticmethod
    def _validate_config(config: dict) -> None:
        """Reject unsupported execution modes before loading market data."""
        if config.get("save-assignment") is not True:
            raise ValueError("save-assignment must be true.")

        algorithm = config.get("assignment-algorithm")
        if algorithm is not None and algorithm != "DA":
            raise ValueError(
                f"Assignment algorithm '{algorithm}' not recognized; only 'DA' is supported."
            )

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
        program_data = load_program_records(
            self.data_scenario,
            "assignment.programs",
            filter_group="assignment",
        )
        program_codes = None
        if "programno" not in program_data.columns:
            program_codes = read_csv_source(
                self.data_scenario.source("assignment.program_codes")
            )
        school_data = load_school_records(
            self.data_scenario,
            "assignment.schools",
            filter_group="assignment",
        )
        self.programs = Programs(program_data, program_codes, self.config)
        self.schools = Schools(school_data, self.programs)

    def _initialize_students(self):
        """Initialize students data interface."""
        student_data = load_student_records(
            self.data_scenario,
            "assignment.students",
            filter_group="assignment",
            low_memory=False,
        )
        school_locations = load_school_records(
            self.data_scenario,
            "assignment.school_coordinates",
            filter_group="assignment",
        )
        self.students = Students(
            student_data_file=student_data,
            programs=self.programs,
            school_data_file=school_locations,
            block_data_file=self.data_scenario.source(
                "assignment.block_data"
            ).path,
            config=self.config,
            data_scenario=self.data_scenario,
        )
        self.n = self.students.n
        self.num_programs = self.students.num_programs

    def _initialize_utility_model(self):
        """Initialize the utility model from its materialized scenario role."""
        if self.config["utility-model"]["enable"]:
            self.umodel = UtilityModel(
                self.config["paths"]["estimate-path"],
                self.programs,
                self.students,
            )
            self.umodel.student_data = self.students.student_data
        elif hasattr(self, "umodel"):
            del self.umodel

    def _reuse_market_data(self) -> None:
        """Rebind immutable interfaces to a new policy/output configuration."""
        self.programs._config = self.config
        self.students.reconfigure_context(self.config, self.data_scenario)
        self.n = self.students.n
        self.num_programs = self.students.num_programs
        aa_schools = self.schools.school_df.loc[
            self.schools.school_df.category == "Attendance"
        ]
        self.zones = Zones(
            self.config,
            attendance_area_schools=aa_schools,
            programs=self.programs,
            students=self.students,
        )

    def _reuse_utility_model(self) -> None:
        if not self.config["utility-model"]["enable"]:
            if hasattr(self, "umodel"):
                del self.umodel
            return
        if not hasattr(self, "umodel"):
            self._initialize_utility_model()
            return
        self.umodel.programs = self.programs
        self.umodel.students = self.students
        self.umodel.student_data = self.students.student_data

    def get_supplemental_zone_path_list(self) -> list:
        """Get paths for language program, special education, or other citywide zones.

        Returns:
            list: list of paths for supplemental zones
        """
        paths = []
        supplemental_zones = self.config["paths"]["citywide-or-lp-zones"]

        for zone_name in self.config.get("citywide-or-lp", []):
            if zone_name not in supplemental_zones:
                raise ValueError(
                    f"Citywide zone {zone_name} is not configured under "
                    "data.overrides.sources.assignment.citywide_zones."
                )
            paths.append(
                pathlib.Path(
                    supplemental_zones[zone_name]
                ).expanduser()
            )

        return paths
