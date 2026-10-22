import copy
import hashlib
import json
import pathlib
import re

import numpy as np
from loaders import (
    DataScenario,
    ResolvedSource,
    load_program_records,
    load_scenario,
    load_school_records,
    load_student_records,
    read_csv_source,
    require_geography_vintage,
    selected_geography_vintage,
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
    "assignment.ctip",
    "assignment.geography.blocks",
    "assignment.geography.crosswalk",
)
ASSIGNMENT_OPTIONAL_IMMUTABLE_SOURCE_ROLES = (
    "assignment.programs.catalog",
    "assignment.geography.blockgroups",
    "assignment.geography.tracts",
)
#: Zone building blocks whose zone files list Census GEOIDs, compared against
#: the students' IDs in the selected vintage.
_CENSUS_ZONE_BUILDING_BLOCKS = {"block", "block_group", "tract"}


def vintage_geography_roles(data_scenario) -> list[str]:
    """Other vintages' Blocks roles, to which the equity tie-breaker is matched."""
    return [
        role
        for role in data_scenario.roles
        if re.fullmatch(r"assignment\.geography\.\d{4}\.(blocks|crosswalk)", role)
    ]


def assignment_source_identity(data_scenario) -> str:
    """Fingerprint immutable assignment inputs without loading their tables."""
    roles = list(ASSIGNMENT_IMMUTABLE_SOURCE_ROLES)
    if data_scenario.filter("assignment", "capacity_scenario") != "programs":
        roles.append("assignment.capacity")
    if data_scenario.filter("assignment", "frl_estimate") is not None:
        roles.append("assignment.frl_estimate")
    roles.extend(
        role
        for role in data_scenario.roles
        if role in ASSIGNMENT_OPTIONAL_IMMUTABLE_SOURCE_ROLES
    )
    roles.extend(vintage_geography_roles(data_scenario))
    # The manifest carries every filter, the assignment selectors included.
    encoded = json.dumps(
        data_scenario.source_manifest(roles), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_market_data(
    data_scenario: DataScenario,
) -> tuple[Programs, Schools, Students]:
    """Load one assignment market's programs, schools, and students."""
    program_data = load_program_records(
        data_scenario, "assignment.programs", filter_group="assignment"
    )
    program_codes = (
        None
        if "programno" in program_data.columns
        else read_csv_source(data_scenario.source("assignment.program_codes"))
    )
    programs = Programs(program_data, program_codes)
    schools = Schools(
        load_school_records(
            data_scenario, "assignment.schools", filter_group="assignment"
        ),
        programs,
    )
    students = Students(
        load_student_records(
            data_scenario,
            "assignment.students",
            filter_group="assignment",
            low_memory=False,
        ),
        programs,
        load_school_records(
            data_scenario,
            "assignment.school_coordinates",
            filter_group="assignment",
        ),
        data_scenario.filter("assignment", "grade"),
        data_scenario,
    )
    return programs, schools, students


def _named_source(
    data_scenario: DataScenario, role: str, name: str
) -> ResolvedSource:
    sources = data_scenario.source_map(role)
    if name not in sources:
        available = ", ".join(sorted(sources))
        raise KeyError(
            f"{name!r} is not configured under data.overrides.sources.{role}; "
            f"available: {available or '(none)'}."
        )
    source = sources[name]
    if not isinstance(source, ResolvedSource):
        raise ValueError(f"Source {role}.{name} must be one file.")
    return source


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

        np.random.seed(self.config["random-seed"])  # set again for each subconfig
        self._initialize_market_data()
        self._initialize_utility_model()

    def _materialize_config(self, external_config: dict) -> None:
        """Adopt a validated config, loading its data scenario only when it changed."""
        self.external_config = copy.deepcopy(external_config)
        data = self.external_config["data"]
        if data != getattr(self, "_data_config", None):
            self.data_scenario = load_scenario(data)
            self._data_config = copy.deepcopy(data)
        self.source_identity = assignment_source_identity(self.data_scenario)

        config = copy.deepcopy(self.external_config)
        config["data-provenance"] = self.data_scenario.provenance()
        self.config = config
        self.resolved_config = copy.deepcopy(self.external_config)

    @property
    def grade(self) -> str:
        """The market's canonical grade, the suffix of its program IDs."""
        return self.data_scenario.filter("assignment", "grade")

    def zone_file(self, policy: str) -> pathlib.Path:
        """Return the zone file configured for a zone policy.

        A zone file of Census GEOIDs is compared against the students' IDs in
        the selected vintage, so one declaring another vintage is rejected.
        Untagged zone files are not checked.
        """
        source = _named_source(self.data_scenario, "assignment.zones", policy)
        if (
            self.config.get("zone-building-blocks") in _CENSUS_ZONE_BUILDING_BLOCKS
            and source.geography_vintage is not None
        ):
            require_geography_vintage(
                source,
                selected_geography_vintage(self.data_scenario, "assignment"),
                f"Zone policy {policy!r}",
            )
        return source.path

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

    def _reset_zones(self) -> None:
        aa_schools = self.schools.school_df.loc[
            self.schools.school_df.category == "Attendance"
        ]
        self.zones = Zones(
            self.config,
            attendance_area_schools=aa_schools,
            programs=self.programs,
            students=self.students,
            grade=self.grade,
        )

    def _initialize_market_data(self):
        """Load the market's programs, schools, and students."""
        self.programs, self.schools, self.students = load_market_data(
            self.data_scenario
        )
        self.n = self.students.n
        self.num_programs = self.students.num_programs
        self._reset_zones()

    def _initialize_utility_model(self):
        """Initialize the utility model from its scenario role."""
        if self.config["utility-model"]["enable"]:
            self.umodel = UtilityModel(
                self.data_scenario.source("assignment.estimate").path,
                self.programs,
                self.students,
            )
            self.umodel.student_data = self.students.student_data
        elif hasattr(self, "umodel"):
            del self.umodel

    def _reuse_market_data(self) -> None:
        """Rebind immutable interfaces to a new policy/output configuration."""
        self.students.reconfigure_context(self.data_scenario)
        self.n = self.students.n
        self.num_programs = self.students.num_programs
        self._reset_zones()

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
        return [
            _named_source(self.data_scenario, "assignment.citywide_zones", name).path
            for name in self.config.get("citywide-or-lp", [])
        ]
