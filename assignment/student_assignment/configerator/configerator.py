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

import getpass
import os

import yamale
import yaml

from ..definitions import (
    BASE_CONFIG_NAME,
    CLUSTER_PATH_CONFIG_NAME,
    CONFIG_SCHEMA_NAME,
    CONFIGS_DIR,
    LOCAL_PATH_CONFIG_NAME,
    SUBCONFIGS_DIR,
    USER_CONFIG_SUFFIX,
)


class Configerator:
    instance = None

    class __Singleton_Configerator:
        def __init__(self):
            self._config = None
            self._original_config = None
            self._path = None

            # Load main config
            self._load_config()
            self._validate_schema(
                yamale.make_data(self._path),
                f"{CONFIGS_DIR}{CONFIG_SCHEMA_NAME}",
            )
            # self._validate_rules(ruleset) # KLM commented out because moved sibling access to policy config

            self.subconfigs = iter(self.config["subconfigs"])

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
                    print("getpass.getuser() failed, using default user")
                    user = os.environ.get("USER", "default")

            self._path = f"{CONFIGS_DIR}{user}{USER_CONFIG_SUFFIX}"

            if not os.path.isfile(self._path):
                # Load base config
                base_config = self._load_yaml(
                    f"{CONFIGS_DIR}{BASE_CONFIG_NAME}"
                )
                # Load environment specific paths
                path_config = self._load_yaml(self._get_path_config())
                # Update base config with path config
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

            self._config = self._load_yaml(self._path)
            self._original_config = self._load_yaml(self._path)

        def _load_subconfig(self, name):
            path = f"{SUBCONFIGS_DIR}{name}.yaml"
            schema_path = f"{SUBCONFIGS_DIR}policy.schema.yaml"
            self._validate_schema(yamale.make_data(path), schema_path)
            subconfig = self._load_yaml(path)
            # Use original config (same from _load_config) to clear optional configs.
            self._config = {**self._original_config, **subconfig}
            self._config["subconfig-name"] = name

        def _get_path_config(self):
            """Get the path to path_config depending on environment.

            Returns:
                str: The path to path_config file.
            """
            if self._is_on_cluster():
                return f"{CONFIGS_DIR}{CLUSTER_PATH_CONFIG_NAME}"
            return f"{CONFIGS_DIR}{LOCAL_PATH_CONFIG_NAME}"

        def _load_yaml(self, path):
            """Given a path, create the config object and load into self.config in
            dict form.
            """
            with open(path) as yf:
                return yaml.full_load(yf)

        def _is_on_cluster(self):
            """Check if the code is currently running on cluster.

            Returns:
                boolean: true if the code is running on cluster, false otherwise.
            """
            return "soal" in os.popen("hostname").read()

        def _validate_schema(self, data, schema_path):
            schema = yamale.make_schema(schema_path)
            yamale.validate(schema, data)

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
            self._config = {**self._config, **derived_values}

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
        def config(self, patch=True):
            """Getter method for config."""
            return self._config

        def clear(self):
            Configerator.instance = None

    def __new__(cls):
        if not Configerator.instance:
            Configerator.instance = Configerator.__Singleton_Configerator()
        return Configerator.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)
