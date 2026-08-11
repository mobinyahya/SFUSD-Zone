import json
import pathlib
import warnings
from collections.abc import Generator
from itertools import product

import numpy as np
import pandas as pd
import yaml

from ..da.da import DeferredAcceptance
from ..da.guardrail_setup import GuardrailSetup
from ..da.ttc import TTC
from ..data_interfaces import Zones
from .policy import Policy
from .preference_generator import PreferenceGenerator
from .priority_generator import PriorityGenerator
from .school_choice_market import SchoolChoiceMarket


class MarketGenerator(SchoolChoiceMarket):
    def __init__(
        self,
        estimate_path: str = None,
        assignment_path: str = None,
        config: dict | None = None,
    ):
        """Initialize market generator.

        Args:
            estimate_path (str, optional): path to folder with estimated utility model parameters.
                Defaults to None, meaning we use the estimate path specified in the config.
            assignment_path (str, optional): path to folder to save assignments. Defaults to None.
            config (dict, optional): in-memory configuration. Defaults to loading the
                configuration through Configerator.
        """
        super().__init__(estimate_path, config=config)
        self._set_up_save_folder(assignment_path)
        self.priority_generator = PriorityGenerator(self)
        self.preference_generator = PreferenceGenerator(self)
        self._guardrail_setup_cache = {}

    def _set_up_save_folder(self, assignment_path: str):
        """Create folder for saving assignments.

        Args:
            assignment_path (str): path to folder to save assignments
        """
        if self.config["save-assignment"]:
            output_assignment_path = (
                self.config["paths"]["assignment-folder"]
                if assignment_path is None
                else assignment_path
            )
            self.output_assignment_path = pathlib.Path(
                output_assignment_path
            ).expanduser()
            pathlib.Path(self.output_assignment_path).mkdir(
                parents=True, exist_ok=True
            )

            if self.yaml is None:
                config_save_path = self.output_assignment_path / "config.json"
                with open(config_save_path, "w") as config_file:
                    json.dump(self.config, config_file, indent=4)
            else:
                config_save_path = self.output_assignment_path / "config.yaml"
                with open(config_save_path, "w") as config_file:
                    yaml.dump(
                        self.config, config_file, default_flow_style=False
                    )

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

    def simulate(self) -> Generator | None:
        """Simulate every policy specified in the config.

        Returns:
            Optional[Generator]: generator for iterating over each policy (only not None if
                save_assignment is False)
        """
        for _ in range(len(self.config["subconfigs"])):
            np.random.seed(
                self.config["random-seed"]
            )  # set again here to ensure order of subconfigs doesn't matter
            self.configurator.load_next_subconfig()
            self.config = self.configurator.config
            self._reset_zones()
            iterations_generator = self.create_iterations_generator()
            if self.config["save-assignment"]:
                self.execute_generator(iterations_generator)
            else:
                return iterations_generator

    @staticmethod
    def execute_generator(iterations_generator: Generator):
        """Execute generator to create assignments.

        Generator structure is used for dynamic assignment options (e.g., running zone
        optimization variants based on assignment outcomes), and this function is used
        in the settings where we don't need the generator and instead want to save assignments.
        """
        # for iterations_generator in policy_generator:
        for policy_suboptions_generator in iterations_generator:
            for priority_suboptions_generator in policy_suboptions_generator:
                for assignment in priority_suboptions_generator:
                    pass

    def create_iterations_generator(self) -> Generator[Generator, None, None]:
        """Create generator for iterating over each iteration of a given policy.

        Returns:
            Generator: generator for iterating over each iteration of a given policy
        """
        if self.config["policies"] == ["real_match"]:
            # assignment = self._run_single_iteration_of_policy(0)
            # return assignment
            return self._read_real_match()

        for policy in self.config["policies"]:
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
                for iteration in range(
                    self.config["iterations"]["start"],
                    self.config["iterations"]["end"],
                ):
                    policy_suboptions_generator = (
                        self._run_single_iteration_of_policy(iteration, policy)
                    )
                    yield policy_suboptions_generator

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
    ) -> Generator[Generator, None, None]:
        """Run a single iteration of a given policy with all possible sets of .

        Args:
            iteration (int): iteration number
            policy (str): policy name

        Returns:
            Generator[Generator]: generator for iterating over each policy subsettings
        """
        if self.config["utility-model"]["enable"] or self.config[
            "utility-model"
        ].get("read-precomuted-umodel-prefs", False):
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

        policy_suboptions_generator = self._simulate_policy(policy, iteration)
        yield policy_suboptions_generator

    def _read_real_match(self) -> pd.DataFrame | None:
        """Read the historical assignment from the student data file (DA is not run).

        Returns:
            Optional[pd.DataFrame]: dataframe with student assignments (only not None if
                save_assignment is False)
        """
        if self.config["utility-model"]["enable"] or self.config[
            "utility-model"
        ].get("read-precomuted-umodel-prefs", False):
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
        policy_data = Policy(
            name="real_match", ctip=None, rounds_merged=None, tiebreaker=None
        )
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
    ) -> Generator[pd.DataFrame | None, None, None]:
        """Simulate a single iteration of a single policy for each priority subsetting.

        Args:
            policy (str): policy name
            iteration (int): iteration number

        Returns:
            Generator[Optional[pd.DataFrame]]: dataframe with student assignments (only not None if
                save_assignment is False)
        """
        self.priority_generator.generate_base_priorities(policy)

        if self.config["utility-model"]["enable"]:
            prefs = self.preference_generator.get_utility_model_preferences_after_truncation()

        else:
            prefs = self.preference_generator.initialize_real_preferences(
                designate=self.config["designate"]
            )

        for ctip, rounds_merged, ties in product(
            self.config["ctip-options"],
            self.config["rounds-merged-options"],
            self.config["ties-options"],
        ):
            policy_data = Policy(
                name=policy,
                ctip=ctip,
                rounds_merged=rounds_merged,
                tiebreaker=ties,
            )

            priorities = self.priority_generator.set_policy_specific_priorities(
                policy_data, prefs
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
            self.config["utility-model"]["enable"]
            or self.config["utility-model"].get(
                "read-precomuted-umodel-prefs", False
            )
        ) and policy != "real_match"
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
        if self.config["assignment-algorithm"] == "DA":
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
        elif self.config["assignment-algorithm"] == "TTC":
            (
                match,
                rank,
            ) = TTC(
                self.programs.capacity.copy(), priorities.copy(), prefs.copy()
            )
            cutoffs = np.zeros([len(match)])
        else:
            raise ValueError(
                f"Assignment algorithm '{self.config['assignment-algorithm']}' not recognized."
            )
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

    def _save_assignment(
        self,
        prefs: np.ndarray,
        policy_data: Policy,
        iteration: int,
        match: np.ndarray,
        in_zone_rank: np.ndarray,
        cutoffs: np.ndarray,
    ) -> pd.DataFrame | None:
        """Create dataframe with student assignments and return it or save to csv.

        Args:
            prefs (np.ndarray): student preferences
            policy_data (Policy): policy data
            iteration (int): iteration number
            match (np.ndarray): matched program array
            in_zone_rank (np.ndarray): rank of assigned program array
            cutoffs (np.ndarray): program cutoffs from DA

        Returns:
            Optional[pd.DataFrame]: dataframe with student assignments
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

        if self.config["save-assignment"]:
            save_name = self._get_assignment_save_name(policy_data, iteration)
            save_path = (
                self.output_assignment_path
                / self.config["subconfig-name"]
                / save_name
            )

            # Check if the parent directory exists; if not, create it
            save_path.parent.mkdir(parents=True, exist_ok=True)

            assignment_df.to_csv(save_path, index=False)
        else:
            return assignment_df

    def _get_assignment_save_name(
        self, policy_data: Policy, iteration: int
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
        if restrict and len(self.config.get("citywide-or-lp", [])) > 0:
            restrict = "fuzzy"
        else:
            restrict = "" if restrict else "no"
        reserve = self.config["guard-rails"] == 0
        reserve = "yes" if reserve else "no"
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

        code_folder = f"{zone_policy}_{restrict}Restrict_{reserve}Reserve_{ctip}_{sibling}Sibling/"
        # code_folder = f"{zone_policy}_{sibling}Sibling_{reserve}Reserve_{ctip}/"

        pathlib.Path(
            self.output_assignment_path
            / self.config["subconfig-name"]
            / code_folder
        ).mkdir(parents=True, exist_ok=True)

        return f"{code_folder}/{code_folder[:-1]}_iteration{iteration}.csv"
