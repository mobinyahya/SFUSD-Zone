import hashlib
import json
import pathlib
import re
import shutil
import tempfile
import warnings
from collections.abc import Generator
from itertools import product

import numpy as np
import pandas as pd
import yaml

from ..configerator import Configerator
from ..da.da import DeferredAcceptance
from ..da.guardrail_setup import GuardrailSetup
from ..data_interfaces import Zones
from .policy import Policy
from .preference_generator import PreferenceGenerator
from .priority_generator import PriorityGenerator
from .school_choice_market import SchoolChoiceMarket


class MarketGenerator(SchoolChoiceMarket):
    AGGREGATE_METRIC_FILES = {
        "school": "metrics_by_school.csv",
        "zip_code": "metrics_by_zip_code.csv",
        "attendance_area": "metrics_by_attendance_area.csv",
        "citywide": "metrics_citywide.csv",
    }
    AGGREGATE_METRIC_GROUPS = {
        "school": [
            "config_name",
            "school_id",
            "school_name",
            "school_category",
        ],
        "zip_code": ["config_name", "zip_code"],
        "attendance_area": ["config_name", "attendance_area"],
        "citywide": ["config_name"],
    }

    def __init__(
        self,
        estimate_path: str = None,
        assignment_path: str = None,
        config: dict | None = None,
        configurator=None,
        write_config: bool = True,
        write_aggregate_metrics: bool = True,
    ):
        """Initialize market generator.

        Args:
            estimate_path (str, optional): path to folder with estimated utility model parameters.
                Defaults to None, meaning we use the estimate path specified in the config.
            assignment_path (str, optional): path to folder to save assignments. Defaults to None.
            config (dict, optional): in-memory configuration. Defaults to loading the
                configuration through Configerator.
            configurator (optional): Config source used by policy simulations.
            write_config (bool, optional): Whether to write the initial config to the
                assignment directory. Defaults to True.
            write_aggregate_metrics (bool, optional): Whether this process owns the
                run-level combined metric files. Parallel workers set this to False.
        """
        super().__init__(
            estimate_path,
            config=config,
            configurator=configurator,
        )
        self._set_up_save_folder(assignment_path, write_config=write_config)
        self.priority_generator = PriorityGenerator(self)
        self.preference_generator = PreferenceGenerator(self)
        self._guardrail_setup_cache = {}
        self._active_policy_cache_context = None
        self._write_aggregate_metrics = write_aggregate_metrics
        self._aggregate_metric_evaluator = None
        self._reset_aggregate_metric_reports()

    def _set_up_save_folder(
        self, assignment_path: str, *, write_config: bool = True
    ):
        """Create folder for saving assignments.

        Args:
            assignment_path (str): path to folder to save assignments
        """
        output_assignment_path = (
            self.config["paths"]["assignment-folder"]
            if assignment_path is None
            else assignment_path
        )
        self.output_assignment_path = pathlib.Path(
            output_assignment_path
        ).expanduser()
        self.output_assignment_path.mkdir(parents=True, exist_ok=True)

        if write_config:
            if self.yaml is None:
                config_save_path = self.output_assignment_path / "config.json"
                with open(config_save_path, "w") as config_file:
                    json.dump(self.resolved_config, config_file, indent=4)
            else:
                config_save_path = self.output_assignment_path / "config.yaml"
                with open(config_save_path, "w") as config_file:
                    yaml.safe_dump(
                        self.resolved_config,
                        config_file,
                        default_flow_style=False,
                    )

    def reconfigure(
        self,
        config: dict,
        assignment_path: str = None,
        *,
        write_config: bool = True,
    ) -> None:
        """Replace a run config, reusing immutable data for the same sources."""
        previous_source_identity = getattr(self, "source_identity", None)
        configurator = Configerator.from_config(config)
        self.configurator = configurator
        self._materialize_config(self.configurator.config)
        self._validate_config(self.config)
        self.yaml = None
        np.random.seed(self.config["random-seed"])
        if previous_source_identity != self.source_identity:
            self._initialize_market_data()
            self._initialize_utility_model()
        else:
            self._reuse_market_data()
            self._reuse_utility_model()
        self._set_up_save_folder(assignment_path, write_config=write_config)
        self.priority_generator = PriorityGenerator(self)
        self.preference_generator = PreferenceGenerator(self)
        self._guardrail_setup_cache.clear()
        self._active_policy_cache_context = None
        if previous_source_identity != self.source_identity:
            self._aggregate_metric_evaluator = None

    def _reset_zones(self):
        aa_schools = self.schools.school_df.loc[
            self.schools.school_df.category == "Attendance"
        ]
        self.zones = Zones(
            self.config,
            attendance_area_schools=aa_schools,
            programs=self.programs,
            students=self.students,
        )

    def simulate(self) -> dict[str, pd.DataFrame] | None:
        """Load and execute every configured policy subconfig."""
        if getattr(self, "_write_aggregate_metrics", True) and hasattr(
            self, "output_assignment_path"
        ):
            self.clear_aggregate_metric_reports(self.output_assignment_path)
        self._reset_aggregate_metric_reports()
        subconfigs = list(self.config.get("subconfigs", []))
        if not subconfigs:
            self._validate_config(self.config)
            self.execute_generator(self.create_iterations_generator())
            return self._complete_aggregate_metric_reports()

        for subconfig in subconfigs:
            if not self.configurator.load_next_subconfig():
                raise RuntimeError(f"Failed to load policy subconfig '{subconfig}'.")
            previous_source_identity = getattr(self, "source_identity", None)
            self._materialize_config(self.configurator.config)
            self._validate_config(self.config)
            np.random.seed(
                self.config["random-seed"]
            )  # ensure subconfig order does not affect lottery draws
            current_source_identity = getattr(self, "source_identity", None)
            if previous_source_identity is None or current_source_identity is None:
                self._reset_zones()
            elif previous_source_identity != current_source_identity:
                self._initialize_market_data()
                self._initialize_utility_model()
                self._aggregate_metric_evaluator = None
            else:
                self._reuse_market_data()
                self._reuse_utility_model()
            self.execute_generator(self.create_iterations_generator())
            self._finalize_aggregate_metric_batch()
        return self._complete_aggregate_metric_reports()

    def _reset_aggregate_metric_reports(self):
        self._aggregate_metric_batches = {
            report: [] for report in self.AGGREGATE_METRIC_FILES
        }
        self._aggregate_metric_results = {
            report: [] for report in self.AGGREGATE_METRIC_FILES
        }

    @classmethod
    def _average_aggregate_metric_frames(cls, report_name, frames):
        if not frames:
            return pd.DataFrame(columns=cls.AGGREGATE_METRIC_GROUPS[report_name])
        combined = pd.concat(frames, ignore_index=True, sort=False)
        group_columns = cls.AGGREGATE_METRIC_GROUPS[report_name]
        if combined.empty:
            return combined.reset_index(drop=True)
        numeric_columns = [
            column
            for column in combined.columns
            if column not in group_columns
            and pd.api.types.is_numeric_dtype(combined[column])
        ]
        return (
            combined.groupby(group_columns, as_index=False, dropna=False, sort=True)[
                numeric_columns
            ]
            .mean()
            .reset_index(drop=True)
        )

    def _finalize_aggregate_metric_batch(self):
        if not self.config.get("export-aggregate-metrics", False):
            return
        for report_name, frames in self._aggregate_metric_batches.items():
            if frames:
                self._aggregate_metric_results[report_name].append(
                    self._average_aggregate_metric_frames(report_name, frames)
                )
            frames.clear()

    def _record_aggregate_metric_reports(self, reports):
        for report_name, report in reports.items():
            self._aggregate_metric_batches[report_name].append(report)

    def _complete_aggregate_metric_reports(self):
        if not self.config.get("export-aggregate-metrics", False):
            return None
        self._finalize_aggregate_metric_batch()
        report_names = (
            tuple(self.AGGREGATE_METRIC_FILES)
            if self.config.get("export-local-metrics", False)
            else ("citywide",)
        )
        reports = self.combine_aggregate_metric_reports(
            [
                {
                    name: pd.concat(
                        self._aggregate_metric_results[name],
                        ignore_index=True,
                        sort=False,
                    )
                    if self._aggregate_metric_results[name]
                    else pd.DataFrame(columns=self.AGGREGATE_METRIC_GROUPS[name])
                    for name in report_names
                }
            ]
        )
        if self._write_aggregate_metrics:
            self.write_aggregate_metric_reports(self.output_assignment_path, reports)
        return reports

    @classmethod
    def combine_aggregate_metric_reports(cls, report_sets):
        report_sets = list(report_sets)
        combined = {}
        report_names = [
            report_name
            for report_name in cls.AGGREGATE_METRIC_FILES
            if any(
                reports is not None and report_name in reports
                for reports in report_sets
            )
        ]
        for report_name in report_names:
            frames = [
                reports[report_name]
                for reports in report_sets
                if reports is not None and report_name in reports
            ]
            if frames:
                frame = pd.concat(frames, ignore_index=True, sort=False)
                if not frame.empty:
                    frame = frame.sort_values(
                        cls.AGGREGATE_METRIC_GROUPS[report_name],
                        kind="stable",
                    )
                frame = frame.reset_index(drop=True)
            else:
                frame = pd.DataFrame(
                    columns=cls.AGGREGATE_METRIC_GROUPS[report_name]
                )
            combined[report_name] = frame
        return combined

    @classmethod
    def write_aggregate_metric_reports(cls, assignment_path, reports):
        assignment_path = pathlib.Path(assignment_path).expanduser()
        assignment_path.mkdir(parents=True, exist_ok=True)
        output_dir = assignment_path / "aggregate_metrics"
        staging_dir = pathlib.Path(
            tempfile.mkdtemp(prefix="aggregate_metrics.tmp.", dir=assignment_path)
        )
        try:
            for report_name, filename in cls.AGGREGATE_METRIC_FILES.items():
                if report_name in reports:
                    reports[report_name].to_csv(staging_dir / filename, index=False)
            cls.clear_aggregate_metric_reports(assignment_path)
            staging_dir.replace(output_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    @staticmethod
    def clear_aggregate_metric_reports(assignment_path):
        output_dir = pathlib.Path(assignment_path).expanduser() / "aggregate_metrics"
        if output_dir.exists():
            shutil.rmtree(output_dir)

    @staticmethod
    def execute_generator(iterations_generator: Generator):
        """Exhaust an assignment generator so its outputs are written."""
        for _ in iterations_generator:
            pass

    def create_iterations_generator(
        self,
    ) -> Generator[pd.DataFrame, None, None]:
        """Create generator for iterating over each iteration of a given policy.

        Returns:
            Generator: generator for iterating over each iteration of a given policy
        """
        for policy in self.config["policies"]:
            if policy == "real_match":
                yield self._read_real_match()
                continue

            # Loop for reserve and restrict settings.
            reserve_options = self._get_reserve_options()
            restrict_options = self._get_restrict_options()

            for reserve_option, restrict_option in product(
                reserve_options, restrict_options
            ):
                self.config["guard-rails"] = reserve_option["guard-rails"]
                self.config["reserve-settings"] = reserve_option[
                    "reserve-settings"
                ]
                self.config["restrict-zone"] = restrict_option["restrict-zone"]
                self.config["citywide-or-lp"] = restrict_option[
                    "citywide-or-lp"
                ]

                # Reset zone and seed so that the order of policy does not matter.
                np.random.seed(self.config["random-seed"])
                self._reset_zones()
                reusable_assignments = self._load_reusable_policy_run(policy)
                if reusable_assignments is not None:
                    yield from reusable_assignments
                    continue
                for iteration in range(
                    self.config["iterations"]["start"],
                    self.config["iterations"]["end"],
                ):
                    yield from self._run_single_iteration_of_policy(
                        iteration, policy
                    )

    def _get_reserve_options(self):
        """Get a list of guard rails and reserve setting options from policy configs.

        Returns:
        - A list of dictionary, each contains key and value for "guard-rails" and
            "reserve-settings".
        """
        # Must contain one of guard-rails-reserve-options or guard-rails
        reserve_options = self.config.get("guard-rails-reserve-options", {})
        if not len(reserve_options):
            if "guard-rails" not in self.config:
                raise ValueError(
                    "Error: must provide at least one of guard-rails or "
                    + "guard-rails-reserve-options in policy configs."
                )
            reserve_setting = self.config.get("reserve-settings", {})
            reserve_options = [
                {
                    "guard-rails": self.config["guard-rails"],
                    "reserve-settings": reserve_setting,
                }
            ]
        return reserve_options

    def _get_restrict_options(self):
        """Get a list of restrict and citywide-or-lp setting options from policy configs.

        Returns:
        - A list of dictionary, each contains key and value for "restrict-zone" and
            "citywide-or-lp".
        """
        # Must contain one of restrict-zone-options or restrict-zone
        restrict_options = self.config.get("restrict-zone-options", {})
        if not len(restrict_options):
            if "restrict-zone" not in self.config:
                raise ValueError(
                    "Error: must provide at least one of restrict-zone or "
                    + "restrict-zone-options in policy configs."
                )
            citywide_or_lp = self.config.get("citywide-or-lp", [])
            restrict_options = [
                {
                    "restrict-zone": self.config["restrict-zone"],
                    "citywide-or-lp": citywide_or_lp,
                }
            ]
        return restrict_options

    def _run_single_iteration_of_policy(
        self, iteration: int, policy: str
    ) -> Generator[pd.DataFrame, None, None]:
        """Run a single iteration of a given policy with all possible sets of .

        Args:
            iteration (int): iteration number
            policy (str): policy name

        Returns:
            Generator[pd.DataFrame]: saved assignments for each priority subsetting
        """
        if self.config["utility-model"]["enable"]:
            self.umodel.draw_utility_model_randomness(
                iteration,
                rows_to_keep=self.students.only_keep_rows,
                cols_to_keep=self.programs.only_keep_cols,
                gumbel_scale=self.config["utility-model"].get(
                    "gumbel-scale", 1.0
                ),
            )  # re-draw preferences

            save_path = self.config["utility-model"].get("save-path")
            if iteration == self.config["iterations"]["start"] and save_path:
                self.umodel.save_utility_matrix(save_path)

        yield from self._simulate_policy(policy, iteration)

    def _read_real_match(self) -> pd.DataFrame:
        """Read the historical assignment from the student data file (DA is not run).

        Returns:
            pd.DataFrame: saved historical student assignments
        """
        policy_data = Policy(
            name="real_match", ctip=None, rounds_merged=None, tiebreaker=None
        )
        reusable = self._load_reusable_assignment(policy_data, None)
        if reusable is not None:
            return reusable
        if not self.config.get("reuse_assignments", True):
            self._assignment_save_path(policy_data, None).unlink(missing_ok=True)

        self._active_policy_cache_context = None
        if self.config["utility-model"]["enable"]:
            self.umodel.draw_utility_model_randomness(
                iteration=None,
                rows_to_keep=self.students.only_keep_rows,
                cols_to_keep=self.programs.only_keep_cols,
                gumbel_scale=self.config["utility-model"].get(
                    "gumbel-scale", 1.0
                ),
            )  # re-draw preferences

        prefs = self.preference_generator.initialize_real_preferences(
            designate=False
        )
        match, in_zone_rank = self._get_real_match(prefs)
        cutoffs = np.zeros([self.num_programs])
        return self._save_assignment(
            prefs,
            policy_data,
            None,
            match,
            in_zone_rank,
            cutoffs,
        )

    def _simulate_policy(
        self, policy: str, iteration: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Simulate a single iteration of a single policy for each priority subsetting.

        Args:
            policy (str): policy name
            iteration (int): iteration number

        Returns:
            Generator[pd.DataFrame]: saved assignment dataframes
        """
        self.priority_generator.generate_base_priorities(policy)

        if self.config["utility-model"]["enable"]:
            prefs = self.preference_generator.get_utility_model_preferences_after_truncation()

        else:
            prefs = self.preference_generator.initialize_real_preferences(
                designate=self.config["designate"]
            )

        for policy_data in self._policy_data_options(policy):
            priorities = self.priority_generator.set_policy_specific_priorities(
                policy_data, prefs, iteration=iteration
            )

            if self.config["guard-rails"] != -1:
                (
                    match,
                    in_zone_rank,
                    cutoffs,
                ) = self._generate_assignment_with_guardrails(prefs, priorities)
            else:
                match, in_zone_rank, cutoffs = self._generate_assignment(
                    prefs, priorities
                )

            yield self._save_assignment(
                prefs, policy_data, iteration, match, in_zone_rank, cutoffs
            )

    def _policy_data_options(self, policy):
        return [
            Policy(
                name=policy,
                ctip=ctip,
                rounds_merged=rounds_merged,
                tiebreaker=ties,
            )
            for ctip, rounds_merged, ties in product(
                self.config["ctip-options"],
                self.config["rounds-merged-options"],
                self.config["ties-options"],
            )
        ]

    def _load_reusable_policy_run(self, policy):
        expected = [
            (policy_data, iteration)
            for iteration in range(
                self.config["iterations"]["start"],
                self.config["iterations"]["end"],
            )
            for policy_data in self._policy_data_options(policy)
        ]
        assignment_paths = [
            self._assignment_save_path(policy_data, iteration)
            for policy_data, iteration in expected
        ]
        if not self.config.get("reuse_assignments", True) or not all(
            path.is_file() for path in assignment_paths
        ):
            for path in assignment_paths:
                path.unlink(missing_ok=True)
            return None
        return [
            self._load_reusable_assignment(policy_data, iteration)
            for policy_data, iteration in expected
        ]

    def _overscribe_attendance_area(
        self,
        prefs: np.ndarray,
        match: np.ndarray,
        in_zone_rank: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place otherwise-unassigned students at their attendance-area school."""
        if not self.config.get("overscribe_aa", False):
            return match, in_zone_rank

        match = match.copy()
        in_zone_rank = in_zone_rank.copy()
        attendance_areas = self.students.attendance_area
        grade = self.config["grade"]

        for student_idx in np.flatnonzero(match == 0):
            studentno = self.students.idx2studentno[student_idx]
            attendance_area = attendance_areas.get(studentno, 0)
            program_idx = self.programs.indices.get(
                f"{attendance_area}-GE-{grade}"
            )
            if program_idx is None:
                continue

            match[student_idx] = program_idx
            preference_ranks = np.flatnonzero(
                prefs[student_idx] == program_idx
            )
            in_zone_rank[student_idx] = (
                preference_ranks[0] + 1
                if preference_ranks.size
                else self.preference_generator.pref_length[student_idx] + 1
            )

        return match, in_zone_rank

    def _get_final_program(self, df: pd.DataFrame):
        """Modify dataframe in place to add final program column.

        Args:
            df (pd.DataFrame): student data
        """
        if self.config["year"] == 18:

            def fn(x):
                return (
                    x["r2_programs"][
                        x["r2_ranked_idschool"].index(x["enrolled_idschool"])
                    ]
                    if not np.any(pd.isna(x["r2_ranked_idschool"]))
                    and x["enrolled_idschool"] in x["r2_ranked_idschool"]
                    else np.nan
                )

            df["r2_programcode"] = df.apply(fn, axis=1)

        df["final_program"] = np.nan
        f_s_copy = df["final_school"].copy()
        last_iteration = 6

        if self.config["r1-only"]:
            last_iteration = 2
            # Update 'final_school' based on the condition
            # Corrected Code with Nullable Integer Type

            df["final_school"] = np.where(
                df["r1_idschool"].notna(),  # Check if 'r1_idschool' is not NaN
                df["r1_idschool"]
                .fillna(0)
                .astype(
                    int
                ),  # Fill NaN with 0 temporarily, then convert valid entries to int
                f_s_copy,  # Otherwise, retain the original 'final_school' value
            )

        # TODO final_school is not correct for non R1 case.
        for rnd in reversed(range(1, last_iteration)):
            rnd_name = f"r{rnd}_programcode"
            if rnd_name in df.columns:
                df["final_program"] = df["final_program"].fillna(df[rnd_name])

    def _get_real_match(
        self, preferences: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the real match from the student data file (DA is not run).

        Args:
            preferences (np.ndarray): student historical preferences

        Returns:
            Tuple[np.ndarray, np.ndarray]: matched program array and rank of assigned program array
        """
        self._get_final_program(self.students.student_data)
        assignment = self.students.student_data[
            ["final_school", "final_program"]
        ].copy()
        assignment["programcodes"] = assignment.apply(
            lambda x: (
                f"{x.final_school}-{x.final_program}-{self.config['grade']}"
                if not pd.isna(x.final_school)
                else np.nan
            ),
            axis=1,
        )
        assignment["programno"] = assignment.programcodes.apply(
            lambda x: (
                self.programs.index(x) if x in self.programs.indices else 0
            )
        )
        match = assignment.programno.to_numpy()

        # calculate rank according to preferences in first round of participation
        ranks = np.zeros(len(match))
        for i in np.where(match > 0)[0]:
            assigned = np.where(preferences[i, :] == match[i])[0]
            if len(assigned) > 0:  # assigned to ranked program
                ranks[i] = np.where(preferences[i, :] == match[i])[0][0] + 1
            else:  # designated
                ranks[i] = self.preference_generator.pref_length[i] + 1
        ranks[ranks == 0] = (
            self.preference_generator.pref_length[ranks == 0] + 1
        )
        ranks = np.clip(
            ranks, a_min=None, a_max=self.preference_generator.pref_length + 1
        )
        # ranks = self._calculate_rank(self, preferences, "real_match", match, in_zone_ranks)

        # TODO: check for designation, unassigned, and programs not ranked in first round of participation
        return match, ranks

    def _get_match_utilities(self, match: np.ndarray) -> np.ndarray:
        """Get utility of assigned program from the utility model.

        Args:
            match (np.ndarray): matched program array

        Returns:
            np.ndarray: utility of assigned program array
        """
        match_idxs = np.array(np.expand_dims(match, axis=1) - 1, dtype=int)
        match_utilities = np.take_along_axis(
            self.umodel.original_utilities, match_idxs, axis=1
        ).flatten()
        match_utilities[match == 0] = np.nan
        return match_utilities

    def _calculate_rank(
        self,
        prefs: np.ndarray,
        policy: str,
        match: np.ndarray,
        in_zone_ranks: np.ndarray,
    ) -> np.ndarray:
        """Calculate rank of assigned program for each student.

        Designated students and unassigned students are given a rank of one greater than
        their last ranked program.

        Args:
            prefs (np.ndarray): student preferences
            policy (str): policy name
            match (np.ndarray): matched program array
            in_zone_ranks (np.ndarray): rank of assigned program array

        Returns:
            np.ndarray: rank of assigned program array
        """
        rank = np.zeros(self.n)
        # TODO: Decide what rank makes sense for unassigned students
        unassigned_idxs = np.where(match == 0)[0]
        rank[unassigned_idxs] = (
            self.preference_generator.pref_length[unassigned_idxs] + 1
        )

        # Designated students: rank = pref_length + 1 (per docstring contract).
        # Without this, the rank reflects the position of the matched program in
        # the full preference list including appended designation programs,
        # which can be arbitrarily large (e.g. 106 for a real list of length 1).
        designated_idxs = np.where(
            np.logical_and(
                in_zone_ranks > self.preference_generator.pref_length, match > 0
            )
        )[0]
        rank[designated_idxs] = (
            self.preference_generator.pref_length[designated_idxs] + 1
        )

        assigned_not_designated_idxs = (
            set(range(self.n)) - set(unassigned_idxs) - set(designated_idxs)
        )
        using_umodel = (
            self.config["utility-model"]["enable"] and policy != "real_match"
        )
        full_prefs = self.umodel.original_preferences if using_umodel else prefs
        missing_matches = 0
        for idx in assigned_not_designated_idxs:
            try:
                rank[idx] = np.where(full_prefs[idx, :] == match[idx])[0][0] + 1
            except IndexError:
                missing_matches += 1
        if missing_matches:
            warnings.warn(
                f"{missing_matches} assigned programs were absent from student "
                "preference lists; their ranks remain unset.",
                stacklevel=2,
            )
        return rank

    def _generate_assignment(
        self, prefs: np.ndarray, priorities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign students to schools with the desired mechanisms, calculating rank and cutoffs.

        Args:
            prefs (np.ndarray): student preferences
            priorities (np.ndarray): student priorities

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: matched program array, rank of assigned
                program array, and cutoffs (the lowest priority students assigned, if program
                is at capacity)
        """
        if self.config["assignment-algorithm"] != "DA":
            raise ValueError(
                f"Assignment algorithm '{self.config['assignment-algorithm']}' "
                "not recognized; only 'DA' is supported."
            )
        da = DeferredAcceptance(
            self.programs.capacity,
            priorities,
            prefs,
            self.students.idx2studentno,
            self.students.studentno2idx,
            self.programs.indices,
        )
        (
            match,
            cutoffs,
            rank,
        ) = da.run()
        rank = np.clip(
            rank, a_min=None, a_max=self.preference_generator.pref_length + 1
        )
        match, rank = self._overscribe_attendance_area(prefs, match, rank)
        return match, rank, cutoffs

    def _generate_assignment_with_guardrails(
        self, preferences: np.ndarray, priorities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate an assignment using reserves (guardrails).

        Args:
            preferences (np.ndarray): student preferences
            priorities (np.ndarray): student priorities

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: matched program array, rank of assigned
                program array, and cutoffs
        """
        guardrail_cache_key = (
            getattr(self, "_active_policy_cache_context", None),
            id(preferences),
            repr(self.config["reserve-settings"]),
            self.config["guard-rails"],
        )
        guard_assignment = self._guardrail_setup_cache.get(guardrail_cache_key)
        if guard_assignment is None:
            guard_assignment = GuardrailSetup(self, preferences)
            self._guardrail_setup_cache[guardrail_cache_key] = guard_assignment
        match, rank = guard_assignment.run(
            priorities,
            reserve_settings=self.config["reserve-settings"],
            strictGuards=self.config["guard-rails"],
        )
        rank = np.clip(
            rank, a_min=None, a_max=self.preference_generator.pref_length + 1
        )
        cutoffs = np.zeros(
            [len(match)]
        )  # TODO: calculate cutoffs in reserve setting
        match, rank = self._overscribe_attendance_area(
            preferences, match, rank
        )
        return match, rank, cutoffs

    def _assignment_save_path(self, policy_data, iteration):
        save_name = self._get_assignment_save_name(policy_data, iteration)
        return (
            self.output_assignment_path
            / self.config.get("subconfig-name", "default")
            / save_name
        )

    def _validate_reusable_assignment(self, assignment_df, assignment_path):
        required_columns = {
            "studentno",
            "programno",
            "programcodes",
            "rank",
            "designation",
            "In-Zone Rank",
        }
        missing_columns = required_columns - set(assignment_df.columns)
        if missing_columns:
            raise ValueError(
                f"Reusable assignment {assignment_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )
        if assignment_df["studentno"].isna().any() or assignment_df[
            "studentno"
        ].duplicated().any():
            raise ValueError(
                f"Reusable assignment {assignment_path} has missing or duplicate "
                "studentno values"
            )

        expected_ids = pd.Index(self.students.student_data.index)
        assignment_ids = pd.Index(assignment_df["studentno"])
        missing_ids = expected_ids[~expected_ids.isin(assignment_ids)].tolist()
        extra_ids = assignment_ids[~assignment_ids.isin(expected_ids)].tolist()
        if missing_ids or extra_ids:
            raise ValueError(
                f"Reusable assignment {assignment_path} does not match the current "
                f"students; missing: {missing_ids}; extra: {extra_ids}"
            )

        program_numbers = pd.to_numeric(
            assignment_df["programno"], errors="coerce"
        )
        valid_program_numbers = {0, *self.programs.codes.keys()}
        invalid_programs = (
            program_numbers.isna()
            | ~np.isfinite(program_numbers)
            | (program_numbers % 1 != 0)
            | ~program_numbers.isin(valid_program_numbers)
        )
        if invalid_programs.any():
            raise ValueError(
                f"Reusable assignment {assignment_path} has invalid programno values"
            )
        assignment_df["programno"] = program_numbers.astype(int)

        program_codes = assignment_df["programcodes"].astype("string").str.strip()
        assigned = assignment_df["programno"] > 0
        expected_codes = assignment_df.loc[assigned, "programno"].map(
            self.programs.codes
        )
        actual_codes = program_codes[assigned]
        invalid_codes = actual_codes.isna() | actual_codes.ne(expected_codes)
        unassigned_with_code = (~assigned) & program_codes.notna() & program_codes.ne("")
        if invalid_codes.any() or unassigned_with_code.any():
            raise ValueError(
                f"Reusable assignment {assignment_path} has programcodes that do "
                "not match programno"
            )
        assignment_df["programcodes"] = program_codes.fillna("").astype(str)

        for column in ("rank", "In-Zone Rank"):
            values = pd.to_numeric(assignment_df[column], errors="coerce")
            invalid = (
                values.isna()
                | ~np.isfinite(values)
                | (values <= 0)
                | (values % 1 != 0)
            )
            if invalid.any():
                raise ValueError(
                    f"Reusable assignment {assignment_path} has invalid {column} values"
                )
            assignment_df[column] = values.astype(int)

        designation = pd.to_numeric(
            assignment_df["designation"], errors="coerce"
        )
        if (
            designation.isna()
            | ~np.isfinite(designation)
            | ~designation.isin([0, 1])
        ).any() or ((~assigned) & designation.eq(1)).any():
            raise ValueError(
                f"Reusable assignment {assignment_path} has invalid designation values"
            )
        assignment_df["designation"] = designation.astype(int)
        return assignment_df

    def _record_assignment_metric_reports(self, assignment_df, save_name, iteration):
        if not self.config.get("export-aggregate-metrics", False):
            return
        from ..evaluation.match_evaluator import MatchEvaluator

        evaluator = getattr(self, "_aggregate_metric_evaluator", None)
        if evaluator is None:
            evaluator = MatchEvaluator.from_scenario(
                self.data_scenario,
                assignment_df,
                program_data=self.programs.program_df,
                distance_cache=self.students.distance_data,
            )
            self._aggregate_metric_evaluator = evaluator
        else:
            evaluator.update_assignments(assignment_df)
        variant_name = pathlib.Path(save_name).stem
        iteration_suffix = f"_iteration{iteration}" if iteration is not None else None
        if iteration_suffix and variant_name.endswith(iteration_suffix):
            variant_name = variant_name[: -len(iteration_suffix)]
        config_name = f"{self.config.get('subconfig-name', 'default')}/{variant_name}"
        self._record_aggregate_metric_reports(
            evaluator.eval_aggregate_metric_reports(
                config_name,
                include_local_metrics=self.config.get(
                    "export-local-metrics", False
                ),
            )
        )

    def _load_reusable_assignment(self, policy_data, iteration):
        if not self.config.get("reuse_assignments", True):
            return None
        assignment_path = self._assignment_save_path(policy_data, iteration)
        if not assignment_path.is_file():
            return None
        assignment_df = self._validate_reusable_assignment(
            pd.read_csv(assignment_path),
            assignment_path,
        )
        self._record_assignment_metric_reports(
            assignment_df,
            self._get_assignment_save_name(policy_data, iteration),
            iteration,
        )
        return assignment_df

    def _save_assignment(
        self,
        prefs: np.ndarray,
        policy_data: Policy,
        iteration: int | None,
        match: np.ndarray,
        in_zone_rank: np.ndarray,
        cutoffs: np.ndarray,
    ) -> pd.DataFrame:
        """Create and save a dataframe with student assignments.

        Args:
            prefs (np.ndarray): student preferences
            policy_data (Policy): policy data
            iteration (int): iteration number
            match (np.ndarray): matched program array
            in_zone_rank (np.ndarray): rank of assigned program array
            cutoffs (np.ndarray): program cutoffs from DA

        Returns:
            pd.DataFrame: saved student assignments
        """
        assignment_df = pd.DataFrame()
        assignment_df["studentno"] = self.students.student_data.index
        assignment_df["programno"] = match
        assignment_df["programcodes"] = [
            self.programs.codes.get(x, np.nan) for x in match
        ]
        if self.config["restrict-zone"]:
            assignment_df["rank"] = self._calculate_rank(
                prefs, policy_data.name, match, in_zone_rank
            )
        else:
            assignment_df["rank"] = in_zone_rank
        assignment_df["designation"] = np.where(
            np.logical_and(
                in_zone_rank > self.preference_generator.pref_length, match > 0
            ),
            1,
            0,
        )
        if self.config["utility-model"]["enable"]:
            assignment_df["assigned_utility"] = self._get_match_utilities(match)
        assignment_df["In-Zone Rank"] = in_zone_rank
        # TODO: add cutoffs to assignment_df?

        save_name = self._get_assignment_save_name(policy_data, iteration)
        save_path = self._assignment_save_path(policy_data, iteration)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.get("export-aggregate-metrics", False):
            save_path.unlink(missing_ok=True)
            self._record_assignment_metric_reports(
                assignment_df,
                save_name,
                iteration,
            )
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=save_path.parent,
                prefix=f".{save_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = pathlib.Path(temporary_file.name)
                assignment_df.to_csv(temporary_file, index=False)
            temporary_path.replace(save_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return assignment_df

    def _get_assignment_save_name(
        self, policy_data: Policy, iteration: int | None
    ) -> str:
        """Create unique folder and name for assignment file.
        Example folder: 18zone_fuzzyRestrict_yesReserve_ctip1_yesSibling.

        Args:
            policy_data (Policy): policy data
            iteration (int): iteration number

        Returns:
            str: unique name for assignment file
        """
        if policy_data.name == "real_match":
            gr = (
                ""
                if self.config["grade"] == "KG"
                else f"{self.config['grade']}_"
            )
            return f"Assignment_{gr}real_match.csv"

        restrict = self.config["restrict-zone"]
        if restrict and self.config.get("citywide-or-lp", []):
            restrict_label = "fuzzy"
        elif restrict:
            restrict_label = "yes"
        else:
            restrict_label = "no"

        guard_rails = self.config["guard-rails"]
        reserve_label = {-1: "no", 0: "soft", 1: "strict"}.get(
            guard_rails, f"guard{guard_rails}"
        )
        zone_policy = policy_data.name if policy_data.name != "Con1" else "aa"
        ctip = policy_data.ctip
        ctip = (
            "noCtip"
            if ctip == 0
            else ("newETB" if ctip == "new_ctip" else ctip)
        )
        ctip = "ctip1" if ctip == 1 else ctip
        sibling = "sibling" in self.config["priority-weights"]
        sibling = "yes" if sibling else "no"

        option_state = {
            "policy": policy_data.name,
            "zone_file": self.config.get("paths", {})
            .get("zone-files", {})
            .get(policy_data.name),
            "ctip": policy_data.ctip,
            "rounds_merged": policy_data.rounds_merged,
            "tiebreaker": policy_data.tiebreaker,
            "guard_rails": guard_rails,
            "reserve_settings": self.config.get("reserve-settings", {}),
            "soft_reserve_boost": self.config.get("soft_reserve_boost"),
            "citywide_separate_reserves": self.config.get(
                "citywide-separate-reserves", True
            ),
            "citywide_reserve_ratios": self.config.get(
                "citywide-reserve-ratios", [0.57, 0.43]
            ),
            "restrict_zone": restrict,
            "citywide_or_lp": self.config.get("citywide-or-lp", []),
        }
        signature = hashlib.sha256(
            json.dumps(
                option_state,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]

        def safe_token(value) -> str:
            return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value)).strip("-") or "none"

        code_folder = (
            f"{safe_token(zone_policy)}_{restrict_label}Restrict_"
            f"{reserve_label}Reserve_{safe_token(ctip)}_{sibling}Sibling_"
            f"rounds{safe_token(policy_data.rounds_merged)}_"
            f"ties{safe_token(policy_data.tiebreaker)}_{signature}"
        )
        return f"{code_folder}/{code_folder}_iteration{iteration}.csv"
