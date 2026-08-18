# TODO: Ensure the ability for a config to patch from default if it needs.
# TODO: Verify default configs compatability when patching.
# TODO: Update default config in terms of values that should be chosen.
"""Author: @hguru.

This class serves as an interface for a Configerator. It,
1. Constructs a Configerator() object, from which a config can be extracted.
2. Constructs a config for every user, if one doesn't exist already.
3. Validates that a configeration meets the schema at path CONFIG_SCHEMA_NAME.
4. Is a Singleton class to ensure consistence across different places.
"""

import copy
import getpass
import os
import warnings
from pathlib import Path

import yamale
import yaml
from loaders import anchor_data_config, load_scenario

from ..definitions import (
    BASE_CONFIG_NAME,
    CONFIG_SCHEMA_NAME,
    CONFIGS_DIR,
    LOCAL_PATH_CONFIG_NAME,
    SUBCONFIGS_DIR,
    USER_CONFIG_SUFFIX,
)

_LEGACY_TOP_LEVEL_DATA_KEYS = {"grade", "remove-special-lps", "year"}
_LEGACY_INPUT_PATH_KEYS = {
    "citywide-or-lp-zones",
    "estimate-path",
    "lotteries-path",
    "new-ctip-blockgroup-path",
    "new-ctip-path",
    "program-data",
    "school-data",
    "sfusd",
    "student-data",
    "student-save",
    "zone-files",
}


class Configerator:
    instance = None

    class __Singleton_Configerator:
        def __init__(self, config=None, *, declaring_path=None):
            self._config = None
            self._original_config = None
            self._path = None

            if config is None:
                # Load main config
                self._load_config()
                # self._validate_rules(ruleset) # KLM commented out because moved sibling access to policy config
            else:
                self._config = self._anchor_config(config, declaring_path)
                self._original_config = copy.deepcopy(self._config)

            self._validate_execution_config(self._config)
            self.subconfigs = iter(self.config.get("subconfigs", []))

        @staticmethod
        def _anchor_config(config, declaring_path=None):
            """Anchor strict data paths without changing other public inputs.

            File-backed configurations resolve relative data paths from the
            declaring YAML directory. In-memory mappings have no such context,
            so their relative data paths intentionally resolve from cwd.
            """
            if not isinstance(config, dict):
                return copy.deepcopy(config)
            anchored = copy.deepcopy(config)
            if isinstance(anchored.get("data"), dict):
                base_dir = (
                    Path.cwd()
                    if declaring_path is None
                    else Path(declaring_path).expanduser().resolve().parent
                )
                anchored["data"] = anchor_data_config(anchored["data"], base_dir)
            return anchored

        def _validate_execution_config(self, config):
            """Apply the same strict external validation to every config source."""
            if not isinstance(config, dict):
                raise ValueError("Assignment configuration must be a map.")

            legacy_keys = sorted(_LEGACY_TOP_LEVEL_DATA_KEYS.intersection(config))
            if legacy_keys:
                raise ValueError(
                    "Assignment data filters must be configured under "
                    "data.overrides.filters.assignment; forbidden top-level keys: "
                    f"{legacy_keys}."
                )

            paths = config.get("paths", {})
            if not isinstance(paths, dict):
                raise ValueError("paths must be a map.")
            legacy_paths = sorted(_LEGACY_INPUT_PATH_KEYS.intersection(paths))
            if legacy_paths:
                raise ValueError(
                    "Assignment input sources must be configured under "
                    "data.overrides.sources; forbidden paths keys: "
                    f"{legacy_paths}."
                )

            data = config.get("data")
            if not isinstance(data, dict):
                raise ValueError("Assignment configuration must define a valid data map.")

            subconfigs = config.get("subconfigs", [])
            if isinstance(subconfigs, list):
                duplicate_subconfigs = sorted(
                    {
                        name
                        for name in subconfigs
                        if subconfigs.count(name) > 1
                    }
                )
                if duplicate_subconfigs:
                    raise ValueError(
                        "Assignment configuration contains duplicate subconfigs: "
                        f"{duplicate_subconfigs}."
                    )

            self._validate_schema(
                yamale.make_data(content=yaml.safe_dump(config)),
                f"{CONFIGS_DIR}{CONFIG_SCHEMA_NAME}",
                strict=False,
            )
            if config.get("export-local-metrics", False) and not config.get(
                "export-aggregate-metrics", False
            ):
                raise ValueError(
                    "export-local-metrics requires export-aggregate-metrics to be true."
                )
            iterations = config["iterations"]
            if iterations["start"] < 0 or iterations["end"] <= iterations["start"]:
                raise ValueError(
                    "Assignment iterations must satisfy 0 <= start < end."
                )
            scenario = load_scenario(data)
            year = scenario.filter("assignment", "year")
            if not isinstance(year, str) or len(year) != 4 or not year.isdigit():
                raise ValueError(
                    "data assignment year must be a canonical four-digit string."
                )
            grades = scenario.filter("assignment", "grades")
            if len(grades) != 1:
                raise ValueError(
                    "Assignment execution requires exactly one grade; "
                    f"selected grades are {list(grades)}."
                )

        def _load_config(self):
            """Find config file located at ../../configs/{$USER}.config.yaml
            If file doesn't exist, create it as a copy of default,
            and assign to $USER.
            If $USER doesn't exist, just use DEFAULT.
            """
            user = os.environ.get("SFUSD_ASSIGNMENT_CONFIG_USER")
            if not user:
                try:
                    user = getpass.getuser()
                except Exception:  # TODO: Remove this after testing
                    warnings.warn(
                        "getpass.getuser() failed; using the default user.",
                        stacklevel=2,
                    )
                    user = os.environ.get("USER", "default")

            self._path = f"{CONFIGS_DIR}{user}{USER_CONFIG_SUFFIX}"

            if not os.path.isfile(self._path):
                # Load base config
                base_config = self._load_yaml(f"{CONFIGS_DIR}{BASE_CONFIG_NAME}")
                # Local and cluster path files now contain output paths only.
                path_config = self._load_yaml(
                    f"{CONFIGS_DIR}{LOCAL_PATH_CONFIG_NAME}"
                )
                base_config.update(path_config)
                # Write atomically: parallel simulations (e.g. the pipeline
                # script launches every run at once) all auto-create the same
                # user config on first use; a plain open(..., "w") truncates
                # the file, so a concurrent reader could load a half-written,
                # schema-invalid config. Dump to a unique temp file, then
                # os.replace (atomic rename on POSIX).
                tmp_path = f"{self._path}.{os.getpid()}.tmp"
                with open(tmp_path, "w") as file:
                    yaml.safe_dump(base_config, file, default_flow_style=False)
                os.replace(tmp_path, self._path)

            self._config = self._anchor_config(
                self._load_yaml(self._path), self._path
            )
            self._original_config = copy.deepcopy(self._config)

        def _load_subconfig(self, name):
            path = f"{SUBCONFIGS_DIR}{name}.yaml"
            schema_path = f"{SUBCONFIGS_DIR}policy.schema.yaml"
            self._validate_schema(yamale.make_data(path), schema_path)
            subconfig = self._load_yaml(path)
            # Use original config (same from _load_config) to clear optional configs.
            self._config = {**self._original_config, **subconfig}
            self._config["data"] = copy.deepcopy(self._original_config["data"])
            self._config["subconfig-name"] = name
            self._validate_execution_config(self._config)

        def _load_yaml(self, path):
            """Given a path, create the config object and load into self.config in
            dict form.
            """
            with open(path) as yf:
                return yaml.safe_load(yf)

        def _validate_schema(self, data, schema_path, *, strict=True):
            schema = yamale.make_schema(schema_path)
            yamale.validate(schema, data, strict=strict)

        def _validate_rules(self, rules):
            rule_outcomes = map(lambda f: f(self.config), rules)
            failed_rules = [i for i, b in enumerate(rule_outcomes) if not b]
            assert not failed_rules, (
                f"Your configuration does not match rules: {failed_rules}."
            )

        def dynamic_config(self, func):
            """Update the config values, given a function which can derive values
            from the default config.

            Params:
                @func: A function which takes a dictionary of the current config,
                and returns a new dictionary of derived values. These values
                update the current config.
            """
            derived_values = func(self._config)
            candidate = {**self._config, **derived_values}
            self._validate_execution_config(candidate)
            self._config = candidate

        def load_all_subconfigs(self):
            """Load all the subconfigs."""
            for subconfig in self.subconfigs:
                self._load_subconfig(subconfig)

        def load_next_subconfig(self):
            """Load only the next subconfig (returns True iff a subconfig was loaded)."""
            subconfig = next(self.subconfigs, 0)
            if subconfig != 0:
                self._load_subconfig(subconfig)
                return True
            return False

        def load_subconfig_by_name(self, subconfig_name):
            """Load a specific subconfig by name, primarily for tests."""
            self._load_subconfig(subconfig_name)

        @property
        def config(self):
            """Getter method for config."""
            return self._config

        @property
        def original_config(self):
            """Return the base configuration before policy overlays."""
            return self._original_config

        def clear(self):
            Configerator.instance = None

    def __new__(cls):
        if not Configerator.instance:
            Configerator.instance = Configerator.__Singleton_Configerator()
        return Configerator.instance

    @classmethod
    def from_config(cls, config, *, declaring_path=None):
        """Create an isolated configurator from a strict public mapping.

        ``declaring_path`` should be the YAML file that declared the mapping.
        If omitted, relative data paths use the current working directory.
        """
        return cls.__Singleton_Configerator(
            config, declaring_path=declaring_path
        )

    @classmethod
    def from_path(cls, path):
        """Load a strict public config and retain its declaring directory."""
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        return cls.from_config(config, declaring_path=config_path)

    def __getattr__(self, name):
        return getattr(self.instance, name)
