"""Build the priority component of student utilities.

Encodes the SFUSD priority structure (sibling, attendance area, CTIP/diversity,
language-pathway, and policy-specific tie-breakers). For background on the
SFUSD priority rules, see the research group's internal priority-structure
documentation (available to collaborators on request).
"""

import pathlib
import warnings

import numpy as np
import pandas as pd

from .policy import Policy


class PriorityGenerator:
    def __init__(self, market):
        self.market = market
        self._zone_context_key = None
        self._zone_setup_cache_key = None
        self._round_priorities_cache = {}
        self._policy_priorities_cache = {}

    def _make_zone_context_key(self, policy: str, zone_file: pathlib.Path):
        return (
            id(self.market.zones),
            policy,
            str(zone_file),
            self.market.config.get("restrict-zone"),
            self.market.config.get("sibling-access"),
            tuple(self.market.config.get("citywide-or-lp", [])),
            self.market.config.get("zone-building-blocks"),
        )

    def _make_policy_priorities_cache_key(self, ctip: dict | str | int, policy: str):
        return (
            self._zone_context_key,
            repr(ctip),
            policy,
            self.market.config.get("subconfig-name"),
            self.market.config.get("grade"),
            self.market.config.get("year"),
            repr(self.market.config.get("priority-weights")),
            repr(self.market.config.get("distance-priority")),
            repr(self.market.config.get("distance-boost")),
            self.market.config.get("aa_boost", 0),
        )

    def generate_base_priorities(self, policy: str):
        """Generate the base priorities for each student in the market (the zone priorities).

        Args:
            policy (str): The policy name for the current simulation and zones.
        """
        zone_files = self.market.config["paths"]["zone-files"]
        if policy not in zone_files:
            available = ", ".join(sorted(zone_files))
            raise KeyError(
                f"Zone policy {policy!r} is not configured. Available zone "
                f"policies: {available or '(none)'}"
            )
        zone_file = pathlib.Path(zone_files[policy]).expanduser().resolve()
        if not zone_file.is_file():
            raise FileNotFoundError(
                f"Zone file for policy {policy!r} does not exist: {zone_file}"
            )
        self.zone_file = zone_file
        zone_context_key = self._make_zone_context_key(policy, zone_file)
        self._zone_context_key = zone_context_key
        self.market._active_policy_cache_context = zone_context_key
        # lp_same_as_ge = policy == "Con1"
        # if lp_same_as_ge:
        #     print("WARNING: Using GE zones as LP zones as well.")
        if zone_context_key != self._zone_setup_cache_key:
            self.market.zones._zone_priority_matrix = None
            self.market.zones._zone_eligibility_matrix = None
            self.market.zones.set_zone(zone_file)
            zone_priority_for_citywide = self.market.config["restrict-zone"]
            self.a2p = self.market.zones.set_area_id2prog_list_dict(
                lp_zone_path_list=self.market.get_supplemental_zone_path_list(),
                remaining_programs_citywide=zone_priority_for_citywide,
                # lp_same_as_ge=lp_same_as_ge,
            )
            self._zone_setup_cache_key = zone_context_key
        # self.zone_priority = self.get_zone_priority_component()
        # Reset stb and mtb random lottery.
        self.reset_stb_mtb_lottery()

    def get_base_policy_priorities(
        self,
        policy: str,
        ctip: dict | str | int,
        *,
        zone_priority_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return policy priorities without rounds, designation, or lottery."""
        if zone_priority_matrix is None:
            self.generate_base_priorities(policy)
        else:
            expected = (self.market.n, self.market.num_programs)
            zone_priority_matrix = np.asarray(zone_priority_matrix)
            if zone_priority_matrix.shape != expected:
                raise ValueError(
                    "Zone priority matrix shape "
                    f"{zone_priority_matrix.shape} does not match {expected}."
                )
            self.market.zones._zone_priority_matrix = zone_priority_matrix.copy()
            self._zone_context_key = ("provided", id(self.market.zones))
            self._policy_priorities_cache.clear()
        return self._set_policy_priorities(ctip, policy).copy()

    def _set_rounds_merged(self, rounds_merged: int | str) -> np.ndarray:
        """Use priorities to determine if rounds of applicants are considered together or separately.

        Students receive higher priority for joining in an earlier round if rounds are not merged. This way, students
        who apply in earlier rounds are considered first. If rounds are merged, students are assigned in a single go.

        Args:
            rounds_merged: `0` or `all`, with legacy codes `123`, `23`, and `12`
                available for at most three selected rounds.
        """
        cache_key = (rounds_merged, id(self.market.students))
        if cache_key in self._round_priorities_cache:
            return self._round_priorities_cache[cache_key]

        first_round_of_participation = self.market.students.first_round
        if rounds_merged in {12, 23, 123} and self.market.students.rounds > 3:
            raise ValueError(
                f"Legacy rounds-merged code {rounds_merged} supports at most "
                "three selected rounds; use 0 or 'all' for additional rounds."
            )
        generators = {
            123: self._merge_all_rounds,
            "all": self._merge_all_rounds,
            0: self._merge_no_rounds,
            23: self._merge_rounds_2_and_3,
            12: self._merge_rounds_1_and_2,
        }
        try:
            generator = generators[rounds_merged]
        except KeyError as exc:
            raise ValueError(
                f"Unknown rounds-merged option {rounds_merged!r}; expected "
                "0, 12, 23, 123, or 'all'."
            ) from exc
        round_priorities = generator(first_round_of_participation)
        result = np.reshape(round_priorities, (self.market.n, 1)) * np.ones(
            [1, self.market.num_programs]
        )
        self._round_priorities_cache[cache_key] = result
        return result

    def _merge_rounds_1_and_2(
        self, first_round_of_participation: np.ndarray
    ) -> np.ndarray:
        """Create round priorities for merging rounds 1 and 2.

        Args:
            first_round_of_participation (np.ndarray): 1 if student participated in first round, 2 if
                student participated in second round, etc.

        Returns:
            np.ndarray: round based priority boost for each student
        """
        return 1000 * np.where(first_round_of_participation <= 1, 1, 0)

    def _merge_rounds_2_and_3(
        self, first_round_of_participation: np.ndarray
    ) -> np.ndarray:
        """Create round priorities for merging rounds 2 and 3.

        Args:
            first_round_of_participation (np.ndarray): 1 if student participated in first round, 2 if
                student participated in second round, etc.

        Returns:
            np.ndarray: round based priority boost for each student
        """
        return 1000 * np.where(first_round_of_participation == 0, 1, 0)

    def _merge_no_rounds(self, first_round_of_participation: np.ndarray) -> np.ndarray:
        """Create round priorities for merging no rounds.

        Args:
            first_round_of_participation (np.ndarray): 1 if student participated in first round, 2 if
                student participated in second round, etc.

        Returns:
            np.ndarray: round based priority boost for each student
        """
        return 1000 * (
            first_round_of_participation.max() - first_round_of_participation
        )

    def _merge_all_rounds(self, _) -> np.ndarray:
        """Create round priorities for merging all rounds.

        Returns:
            np.ndarray: round based priority boost for each student
        """
        return np.zeros([self.market.n])

    def _set_policy_priorities(self, ctip: dict | str, policy: str) -> np.ndarray:
        """Set numerical values for students' priorities depending on priority_weights
        Parameters: priorityweights : dictionary mapping each of {'ctip','sibling','zone'} to a numeric weight.
        """
        cache_key = self._make_policy_priorities_cache_key(ctip, policy)
        if cache_key in self._policy_priorities_cache:
            return self._policy_priorities_cache[cache_key]

        if self.market.config["grade"] == "06":
            priorities = self._sixth_grade_priorities()
            priorities += self._get_attendance_area_priorities()
            self._policy_priorities_cache[cache_key] = priorities
            return priorities
        if self.market.config["grade"] == "09":
            priorities = self._ninth_grade_priorities()
            priorities += self._get_attendance_area_priorities()
            self._policy_priorities_cache[cache_key] = priorities
            return priorities

        priorities = np.zeros((self.market.n, self.market.num_programs))
        ctip_indicator = self.set_ctip_priority(ctip)
        sibling = self.market.students.sibling(self.market.programs)
        weights = self.market.config["priority-weights"]

        for k, v in weights.items():
            if k == "ctip":
                priorities += v * ctip_indicator
            elif k == "prek":
                priorities += v * self.market.students.prek()
            elif k == "sibling":
                priorities += v * sibling
            elif k == "zone":
                priorities += v * self.market.zones.zone_priority_matrix
            elif k == "peng":
                try:
                    path = self.market.config["paths"]["peng-boosts"][policy]
                except KeyError as exc:
                    raise ValueError(
                        "PENG priority requires "
                        "data.overrides.sources.assignment.peng_boosts."
                    ) from exc
                boost_matrix = np.load(str(path))
                priorities += boost_matrix
            elif k == "distance":
                # Check if distance boost settings are configured
                distance_boost = self.market.config.get("distance-boost", None)

                # Get sorted program list to ensure consistent ordering
                prog_list = [
                    x[0]
                    for x in sorted(
                        self.market.programs.indices.items(), key=lambda x: x[1]
                    )
                ]

                if distance_boost:
                    low_income_distances = self.market.students.distance_data.copy()
                    high_income_distances = self.market.students.distance_data.copy()

                    # Get income threshold from config, default to 95292
                    income_threshold = distance_boost.get("income_threshold", 95292)
                    low_income_boost = distance_boost.get("low_income_boost", 0.2)

                    # Get index of students above and below the threshold
                    low_income_index = self.market.students.student_data[
                        self.market.students.student_data["median_hh_income"]
                        < income_threshold
                    ].index

                    high_income_index = self.market.students.student_data[
                        self.market.students.student_data["median_hh_income"]
                        >= income_threshold
                    ].index

                    all_student_index = self.market.students.student_data.index
                    low_income_pos = all_student_index.get_indexer(low_income_index)
                    high_income_pos = all_student_index.get_indexer(high_income_index)

                    # Sort distances based on students and programs order
                    low_income_distances = low_income_distances.loc[
                        low_income_index.to_numpy()
                    ]
                    high_income_distances = high_income_distances.loc[
                        high_income_index.to_numpy()
                    ]

                    # Ensure program ordering is consistent
                    low_income_distances = low_income_distances[prog_list].to_numpy()
                    high_income_distances = high_income_distances[prog_list].to_numpy()

                    low_income_priority = self._get_distance_priority(
                        low_income_distances, boost=low_income_boost
                    )
                    high_income_priority = self._get_distance_priority(
                        high_income_distances
                    )

                    full_priority_matrix = np.zeros_like(priorities)
                    full_priority_matrix[low_income_pos, :] = low_income_priority
                    full_priority_matrix[high_income_pos, :] = high_income_priority

                    priorities += v * full_priority_matrix
                else:
                    distances = self.market.students.distance_data.copy()
                    distances = distances.loc[
                        self.market.students.student_data.index.to_numpy()
                    ]

                    distances = distances[prog_list].to_numpy()
                    priorities += v * self._get_distance_priority(distances)

        # mask for language programs must happen last
        # TODO: make this compatible with zone restriction
        if "language-programs" in weights:
            priorities = self._get_kg_language_program_priorities(
                self.market.config["priority-weights"]["language-programs"],
                priorities,
                ctip_indicator,
                sibling,
            )

        priorities += self._get_attendance_area_priorities()
        self._policy_priorities_cache[cache_key] = priorities
        return priorities

    def _get_attendance_area_priorities(self) -> np.ndarray:
        """Return the configured priority boost at each student's AA GE program."""
        priorities = np.zeros((self.market.n, self.market.num_programs))
        boost = self.market.config.get("aa_boost", 0)
        if not boost:
            return priorities

        attendance_areas = self.market.students.attendance_area
        grade = self.market.config["grade"]
        for student_idx, studentno in self.market.students.idx2studentno.items():
            attendance_area = attendance_areas.get(studentno, 0)
            program_idx = self.market.programs.indices.get(
                f"{attendance_area}-GE-{grade}"
            )
            if program_idx is not None:
                priorities[student_idx, program_idx - 1] = boost
        return priorities

    def _get_distance_priority(self, distances, boost=0):
        """This function computes the priority based on student-program distances.
        Inputs:
        - distances: distance matrix of shape (# of student, # of program).

        Returns:
        - the priorities based on distances. The max value is set to 1 to be
            consistent with other priority weight settings.

        If ``distance-priority.weights`` is provided in the config (a list
        of floats with one entry per threshold, ordered closest-to-farthest),
        those values are used directly instead of the default linear
        ``(N - band) / N`` formula.  Example::

            distance-priority:
              thresholds: [0.5, 3]
              weights: [0.8, 0.3]   # <=0.5mi -> 0.8, 0.5-3mi -> 0.3, >3mi -> 0
        """
        priors = self.market.config.get("distance-priority", {"step-size": 10})
        custom_weights = priors.get("weights", None)

        # Only one *mode* key (thresholds / step-size / continuous) is
        # allowed; the optional ``weights`` key does not count.
        mode_keys = {k: v for k, v in priors.items() if k != "weights"}
        if len(mode_keys) > 1:
            warnings.warn(
                "Only one distance priority setting is supported; using zero priorities.",
                stacklevel=2,
            )
            return np.zeros(distances.shape)

        f, vals = list(mode_keys.items())[0]
        if isinstance(vals, list):
            vals = [val + boost for val in vals]

        # We manually set invalid distances to 0 (e.g. for students without
        # location or with location outsides of SFUSD where distance >=10)
        # so we need to exclude priority for those students here by setting
        # their distance to the largest possible value 10.
        max_dist = 10
        distances = np.where(distances == 0, max_dist, distances)

        if f == "continuous":
            if vals == "1_over_x_sqaure":
                distances = 1 / np.square(distances)
                return distances / np.max(distances)
            if vals == "x":
                return 1 - distances / np.max(distances)

        thresholds = []
        if f == "step-size":
            thresholds = np.arange(vals, max_dist, vals)
        elif f == "thresholds":
            thresholds = vals
        if len(thresholds) == 0:
            warnings.warn(
                "Invalid or unsupported distance priority setting. "
                "Using zero priorities.",
                stacklevel=2,
            )
            return np.zeros(distances.shape)
        if not np.all(np.diff(thresholds) > 0):
            warnings.warn(
                "Received non-strictly-increasing distance-priority "
                f"thresholds: {thresholds}.",
                stacklevel=2,
            )

        # Assign each cell a band index: 0 (closest), 1, …, N (beyond all).
        step_vals = [i + 1 for i in range(len(thresholds))]
        # np piecewise gives the last condition priority if multiple are true.
        band_indices = np.piecewise(
            distances, [distances > i for i in thresholds], step_vals
        )

        if custom_weights is not None:
            if len(custom_weights) != len(thresholds):
                warnings.warn(
                    f"Distance-priority weights have {len(custom_weights)} entries "
                    f"but there are {len(thresholds)} thresholds. "
                    "Falling back to default weights.",
                    stacklevel=2,
                )
                return (len(thresholds) - band_indices) / len(thresholds)
            # Map band index → custom weight (beyond-all band → 0).
            weight_map = np.array(list(custom_weights) + [0.0], dtype=float)
            return weight_map[band_indices.astype(int)]

        return (len(thresholds) - band_indices) / len(thresholds)

    def _get_kg_language_program_priorities(
        self,
        weights: dict,
        ge_priorities: np.ndarray,
        ctip: np.ndarray,
        sibling: np.ndarray,
    ) -> np.ndarray:
        """Get kindergarten citywide language program priorities.

        Args:
            weights (dict): weights for each priority category specifically at language programs
            ge_priorities (np.ndarray): general-education priority matrix to
                build the language-program priorities on top of
            sibling (np.ndarray): sibling indicator matrix
            ctip (np.ndarray): ctip indicator matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]: (num students) by (num programs) array with language
                program priorities and (num progams) length mask identifying which programs are
                handled by this priority class
        """
        lp_mask = np.zeros(self.market.num_programs)
        citywide_lp = self.market.programs.citywide_language_program_indices(
            self.market.schools.citywide_schools
        )
        lp_indices = [x - 1 for x in citywide_lp]
        lp_mask[lp_indices] = 1
        sib_weights = weights["sibling"] if "sibling" in weights else 0
        lp_priorities = sib_weights * sibling + weights["ctip"] * ctip
        lp_sibling = self.market.students.language_pathway_sibling(
            self.market.programs.indices
        )
        sib_weights = weights["lp-sibling"] if "lp-sibling" in weights else 0
        lp_priorities += sib_weights * lp_sibling
        lp = self.market.students.language_pathway_priority_kg(
            self.market.programs.indices
        )
        lp_priorities += weights["lp"] * lp
        priorities = np.multiply(lp_priorities, lp_mask) + np.multiply(
            ge_priorities, 1 - lp_mask
        )
        return priorities

    def set_ctip_priority(self, ctip: dict | str | int) -> np.ndarray:
        """Add ctip priority component to priorities matrix.

        Args:
            ctip (Union[dict, str, int]): ctip priority type
            priorities (np.ndarray): current priorities matrix

        Returns:
            np.ndarray: updated priorities matrix
        """
        if isinstance(ctip, dict):
            classes_matrix = self._set_custom_equity_tiebreaker(ctip)
        elif ctip == 1:
            classes_matrix = self.market.students.ctip
        elif ctip == "new_ctip":
            classes_matrix = self.market.students.new_ctip
        elif ctip == "new_ctip_blockgroup":
            classes_matrix = self.market.students.new_ctip_blockgroup
        elif "D" in str(ctip):
            classes_matrix = self._set_diversity_category_priority(ctip)
        elif ctip == 5:
            CTIPtypes = np.array(self.market.students.student_data["CTIPtype"]).reshape(
                (self.market.n, 1)
            )
            classes_matrix = 5 * np.ones(
                (self.market.n, self.market.num_programs)
            ) - np.matmul(CTIPtypes, np.ones([1, self.market.num_programs]))
        elif ctip == 0:
            classes_matrix = np.zeros((self.market.n, self.market.num_programs))
        else:
            raise ValueError(f"Unknown ctip type '{ctip}'.")
        return classes_matrix

    def _set_diversity_category_priority(self, ctip: str) -> np.ndarray:
        """Construct HOCidx1 categories for each student.

        Args:
            ctip (str): ctip priority type, expecting in the form "xD" where x is the number
                of categories

        Returns:
            np.ndarray: matrix of ctip priority categories for each student
        """
        num_categories = int(ctip[0])
        hoc_idx_1 = self.market.students.student_data.HOCidx1
        hoc_idx_1 = np.array(hoc_idx_1)
        hoc_idx_1 = np.nan_to_num(hoc_idx_1, nan=0.4)
        percentages = 100 * np.array(range(1, num_categories)) / num_categories
        quantiles = np.percentile(hoc_idx_1, percentages)
        classes = np.searchsorted(quantiles, hoc_idx_1)
        classes_matrix = np.matmul(
            classes.reshape((self.market.n, 1)),
            np.ones([1, self.market.num_programs]),
        )

        return classes_matrix

    def _set_custom_equity_tiebreaker(self, ctip: dict) -> np.ndarray:
        """Construct custom equity tiebreaker categories for each student.

        Args:
            ctip (dict): ctip priority type. Expecting either "num_categories" or "thresholds",
                as well as "column" and "lower_disadvantaged" as entries in dictionary.

        Returns:
            np.ndarray: matrix of ctip priority categories for each student
        """
        index_col = self.market.students.student_data[ctip["column"]].to_numpy()
        index_col = np.nan_to_num(index_col, nan=np.nanmean(index_col))
        if "num_categories" in ctip:
            percentages = (
                100
                * np.array(range(1, ctip["num_categories"]))
                / ctip["num_categories"]
            )
            thresholds = np.percentile(index_col, percentages)
        elif "thresholds" in ctip:
            thresholds = ctip["thresholds"]  # expect lower index => more disadvantaged
        else:
            raise ValueError(
                "Neither 'num_categories' nor 'thresholds' defined for diversity tiebreaker."
            )
        classes = np.searchsorted(thresholds, index_col)
        if ctip["lower_disadvantaged"]:
            classes = (
                len(thresholds) - classes
            )  # switch to higher value => greater priority
        classes_matrix = np.matmul(
            classes.reshape((self.market.n, 1)),
            np.ones([1, self.market.num_programs]),
        )

        return classes_matrix

    def _get_brown_ms_priorities(
        self, weights: dict, sibling: np.ndarray, ctip: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get brown middle school priorities.

        Args:
            weights (dict): weights for each priority category specifically at Brown MS
            sibling (np.ndarray): sibling indicator matrix
            ctip (np.ndarray): ctip indicator matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]: (num students) by (num programs) array with brown middle
                school priorities and (num progams) length mask identifying which programs are
                handled by this priority class
        """
        brown_ms_index = self.market.programs.index("858-GE-06") - 1
        brown_ms_mask = np.zeros(self.market.num_programs)
        brown_ms_mask[brown_ms_index] = 1
        priorities = np.multiply(
            weights["sibling"] * sibling + weights["ctip"] * ctip, brown_ms_mask
        )
        priorities[:, brown_ms_index] += (
            weights["bayview-to-brown"] * self.market.students.bayview_to_brown_ms
        )
        priorities[:, brown_ms_index] += (
            weights["zip-94124"] * self.market.students.zip_94124
        )
        return priorities, brown_ms_mask

    def _get_sixth_grade_language_program_priorities(
        self,
        weights: dict,
        sibling: np.ndarray,
        msf: np.ndarray,
        ctip: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get 6th grade language program priorities.

        Args:
            weights (dict): weights for each priority category specifically at language programs
            sibling (np.ndarray): sibling indicator matrix
            msf (np.ndarray): msf indicator matrix
            ctip (np.ndarray): ctip indicator matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]: (num students) by (num programs) array with language
                program priorities and (num progams) length mask identifying which programs are
                handled by this priority class
        """
        lp_mask = np.zeros(self.market.num_programs)
        lp_indices = [x - 1 for x in self.market.programs.language_program_indices()]
        lp_mask[lp_indices] = 1
        priorities = np.multiply(
            weights["sibling"] * sibling
            + weights["msf"] * msf
            + weights["ctip"] * ctip,
            lp_mask,
        )
        # TODO: Determine ordering of bayview-to-all priority for language programs
        lp_sibling = self.market.students.language_pathway_sibling(
            self.market.programs.indices
        )
        priorities += weights["lp-sibling"] * lp_sibling
        lp = self.market.students.language_pathway_priority(
            self.market.programs.program_type_to_indices
        )
        priorities += weights["lp"] * lp
        return priorities, lp_mask  # , np.zeros(self.market.n)

    def _get_bayview_student_priorities(
        self, weights: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get 6th grade priorities for bayview students.

        Args:
            weights (dict): weights for each priority category specifically for bayview students

        Returns:
            Tuple[np.ndarray, np.ndarray]: (num students) by (num programs) array with bayview
                student priorities and (num progams) length mask identifying which programs are
                handled by this priority class
        """
        student_mask = self.market.students.bayview_to_all_ms
        priorities = weights["bayview-to-all"] * np.outer(
            student_mask, np.ones(self.market.num_programs)
        )
        return priorities, np.zeros(self.market.num_programs)

    def _get_remaining_ms_priorities(
        self,
        weights: dict,
        sibling: np.ndarray,
        msf: np.ndarray,
        ctip: np.ndarray,
        program_mask: np.ndarray,
    ) -> np.ndarray:
        """Get 6th grade priorities for programs not handled by language programs or Brown MS.

        Args:
            weights (dict): weights for each priority category specifically for remaining programs
            sibling (np.ndarray): sibling indicator matrix
            msf (np.ndarray): msf indicator matrix
            ctip (np.ndarray): ctip indicator matrix
            program_mask (np.ndarray): (num programs) length mask identifying which programs are
                handled by other priority classes

        Returns:
            np.ndarray: (num students) by (num programs) array with remaining program priorities
        """
        remaining_programs = np.ones(self.market.num_programs) - program_mask
        unmasked = (
            weights["sibling"] * sibling + weights["msf"] * msf + weights["ctip"] * ctip
        )
        priorities = np.multiply(unmasked, remaining_programs)
        return priorities

    def _sixth_grade_priorities(self) -> np.ndarray:
        """Get 6th grade priorities.

        Returns:
            np.ndarray: (num students) by (num programs) array with 6th grade priorities
        """
        priorities = np.zeros((self.market.n, self.market.num_programs))
        program_mask = np.zeros(self.market.num_programs)

        sibling = self.market.students.sibling(self.market.programs)
        msf = self.market.students.msf(self.market.programs.school_to_indices)
        ctip = self.market.students.ctip

        for category, weights in self.market.config["priority-weights"].items():
            if category == "brown-ms":
                (
                    partial_priorities,
                    partial_program_mask,
                ) = self._get_brown_ms_priorities(weights, sibling, ctip)
            elif category == "bayview-students":
                (
                    partial_priorities,
                    partial_program_mask,
                ) = self._get_bayview_student_priorities(weights)
            elif category == "language-programs":
                (
                    partial_priorities,
                    partial_program_mask,
                ) = self._get_sixth_grade_language_program_priorities(
                    weights, sibling, msf, ctip
                )
            elif category == "remaining":
                continue
            else:
                raise ValueError(f"Unknown priority category '{category}'.")
            priorities += partial_priorities
            program_mask += partial_program_mask

        weights = self.market.config["priority-weights"]["remaining"]
        priorities += self._get_remaining_ms_priorities(
            weights, sibling, msf, ctip, program_mask
        )
        return priorities

    def _get_brown_ms_to_hs_priorities(self) -> np.ndarray:
        """Get 9th grade priority for students from brown middle school.

        Returns:
            np.ndarray: (num students) by (num programs) array with brown middle school to
                all high schools priority
        """
        brown_to_hs = self.market.students.brown_ms_to_hs
        return np.outer(brown_to_hs, np.ones(self.market.num_programs))

    def _selective_hs_eligibility(self):
        """Make students ineligible for selective high schools unless they qualify.

        Students are eligibile for selective high schools if they ranked Lowell or SOTA in
        their historical preferences.

        Returns:
            np.ndarray: (num students) by (num programs) array where 1 indicates selective high
                school eligibility, 0 otherwise
        """
        priority = np.zeros((self.market.n, self.market.num_programs))
        sota = self.market.programs.indices.get("815-GE-09")
        if sota is not None:
            priority[:, sota - 1] = (1 - self.market.students.sota_eligible) * -500

        # lowell not selective in 2021-22 or 2022-23
        if self.market.config["year"] not in [21, 22]:
            lowell = self.market.programs.indices.get("697-GE-09")
            if lowell is not None:
                priority[:, lowell - 1] = (
                    1 - self.market.students.lowell_eligible
                ) * -500
        return priority

    def _ninth_grade_priorities(self) -> np.ndarray:
        """Get 9th grade priorities.

        Returns:
            np.ndarray: (num students) by (num programs) array with 9th grade priorities
        """
        weights = self.market.config["priority-weights"]
        priorities = np.zeros((self.market.n, self.market.num_programs))

        sibling = self.market.students.sibling(self.market.programs)
        priorities += sibling * weights["sibling"]
        brown_to_hs = self._get_brown_ms_to_hs_priorities()
        priorities += brown_to_hs * weights["brown-ms-to-hs"]
        ctip = self.market.students.ctip
        priorities += ctip * weights["ctip"]

        lp_sibling = self.market.students.language_pathway_sibling(
            self.market.programs.indices
        )
        priorities += lp_sibling * weights["lp-sibling"]
        lp = self.market.students.language_pathway_priority(
            self.market.programs.program_type_to_indices
        )
        priorities += lp * weights["lp"]

        priorities += self._selective_hs_eligibility()

        return priorities

    def _set_tiebreaker(
        self, tiebreaker: str, iteration: int | None = None
    ) -> np.ndarray:
        """Set random lottery tiebreaker.

        Args:
            tiebreaker (str): tiebreaker type
            iteration (int, optional): iteration number used when reading saved lotteries.

        Returns:
            np.ndarray: (num students) by (num programs) array with random lottery tiebreaker
        """
        lottery_generators = {
            "STB": self._stb,
            "STB_REAL": self._stb_real,
            "MTB": self._mtb,
            "MTB_REAL": self._mtb_real,
            "HTB": self._htb,
            "STBcoordinated": self._stb_coordinated,
        }
        if tiebreaker not in lottery_generators:
            raise ValueError(f"Unknown tiebreaker '{tiebreaker}'.")

        if self.market.config.get("read-lotteries", False):
            if iteration is None:
                raise ValueError(
                    "An iteration is required when read-lotteries is enabled."
                )
            path = (
                self.market.config["paths"]["lotteries-path"]
                + tiebreaker
                + str(iteration)
            )
            A = np.array(pd.read_csv(path))
            lottery = A[:, 1:]
            return lottery

        return lottery_generators[tiebreaker]()

    def _stb(self) -> np.ndarray:
        """Use single tiebreaking from random numbers.

        Single tiebreaking means that each student has the same random number for each program.

        Returns:
            np.ndarray: (num students) by (num programs) array with single tiebreaking lottery numbers
        """
        if len(self._stb_matrix) == 0:
            self._stb_matrix = np.random.rand(self.market.n, 1) * np.ones(
                [1, self.market.num_programs]
            )
        return self._stb_matrix

    def _mtb(self) -> np.ndarray:
        """Use multiple tiebreaking from random numbers.

        Multiple tiebreaking means that each student has different random numbers for each
        program.

        Returns:
            np.ndarray: (num students) by (num programs) array with multiple tiebreaking lottery numbers
        """
        if len(self._mtb_matrix) == 0:
            self._mtb_matrix = np.random.rand(self.market.n, self.market.num_programs)
        return self._mtb_matrix

    def reset_stb_mtb_lottery(self):
        """Reset randomed mtb and stb matrix to empty to re-generate next time."""
        self._mtb_matrix = np.array([])
        self._stb_matrix = np.array([])

    def _stb_coordinated(self) -> np.ndarray:
        """Use an single tiebreaking that is the same for all students in the same census blockgroup.

        Returns:
            np.ndarray: (num students) by (num programs) array with coordinated random lottery
        """
        lottery = np.zeros([self.market.n, self.market.num_programs])
        student_data = self.market.students.student_data
        block_group_list = student_data["census_blockgroup"].unique()
        for block in block_group_list:
            idxs = np.where(student_data["census_blockgroup"] == block)
            lottery[idxs, :] += np.random.rand(len(idxs), 1) * np.ones(
                [1, self.market.num_programs]
            )
        return lottery

    def _htb(self):
        """Original code description: Tie-breaking experiment: tiebreaker can also equal 'HTB', where
        popular popular programs use a common lottery and otherwise seperate lotteries.
        """
        # if prefs is None:
        #     raise NotImplementedError(
        #             "Need to set preferences/popularity of programs for HTB implementation."
        #         )
        # self.market.programs.set_popular(prefs)
        # popular = self.market.programs.popular  # 1 by p array
        # A = np.diag(popular)
        # B = np.identity(self.market.num_programs) - A
        # lottery = np.matmul(
        #         np.random.rand(self.market.n, 1)
        #         * np.ones([1, self.market.num_programs]),
        #         A,
        #     )
        # lottery += np.matmul(
        #         np.random.rand(self.market.n, self.market.num_programs), B
        #     )
        raise NotImplementedError("HTB code broken, please see source")

    def _mtb_real(self) -> np.ndarray:
        """Use multiple tiebreaking from selected historical preferences.
        Use random number for student-program pairs where we do not have historical preferences.

        Returns:
            np.ndarray: (num students) by (num programs) array with multiple tiebreaking
                lottery number
        """
        mtb = np.zeros((self.market.students.n, self.market.num_programs))
        selected_preferences = self.market.students.selected_preferences(
            self.market.programs.index_list
        )
        student_data = self.market.students.student_data
        required = {"selected_randomnumber", "selected_designation_randomnumber"}
        missing = sorted(required - set(student_data.columns))
        if missing:
            raise ValueError(
                f"Historical MTB lottery requires selected columns: {missing}."
            )
        for row_id, (program_ids, random_numbers, d_val) in enumerate(
            zip(
                selected_preferences,
                student_data.selected_randomnumber.to_numpy(),
                student_data.selected_designation_randomnumber,
            )
        ):
            program_ids = np.array([x for x in program_ids if x != 0])
            if len(program_ids):
                if np.any((program_ids < 1) | (program_ids > self.market.num_programs)):
                    raise ValueError(
                        "Historical preferences contain invalid program IDs."
                    )
                try:
                    designation_number = float(d_val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Historical designation lottery numbers must be numeric."
                    ) from exc
                if not np.isfinite(designation_number):
                    raise ValueError(
                        "Historical designation lottery numbers must be finite."
                    )
                mtb[row_id, :] = designation_number
                if not isinstance(random_numbers, (list, tuple, np.ndarray)):
                    raise ValueError(
                        "Historical lottery numbers must be loader-normalized lists."
                    )
                try:
                    parsed_numbers = np.asarray(random_numbers, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Historical lottery numbers must be numeric."
                    ) from exc
                if (
                    parsed_numbers.shape != (len(program_ids),)
                    or not np.isfinite(parsed_numbers).all()
                ):
                    raise ValueError(
                        "Historical lottery numbers must align with ranked programs."
                    )
                mtb[row_id, program_ids - 1] = parsed_numbers

        return mtb

    def _stb_real(self) -> np.ndarray:
        """Use each student's first selected historical lottery number.

        Returns:
            np.ndarray: (num students) by (num programs) array with single tiebreaking lottery numbers
        """
        warnings.warn(
            "STB_REAL uses the first lottery number from each student's selected "
            "choice; true policy is multiple tiebreaking",
            stacklevel=2,
        )
        student_data = self.market.students.student_data
        if "selected_randomnumber" not in student_data:
            raise ValueError("Historical STB lottery requires selected_randomnumber.")
        numbers = []
        for value in student_data.selected_randomnumber:
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise ValueError(
                    "Historical selected lottery numbers must be loader-normalized "
                    "lists."
                )
            if len(value) == 0:
                raise ValueError(
                    "Historical selected lottery numbers must not be empty."
                )
            try:
                number = float(value[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Historical selected lottery numbers must be numeric."
                ) from exc
            if not np.isfinite(number):
                raise ValueError("Historical selected lottery numbers must be finite.")
            numbers.append(number)
        return np.asarray(numbers)[:, np.newaxis] * np.ones(
            [1, self.market.num_programs]
        )

    def _set_not_designation_priority(self, preferences):
        not_designation = np.zeros([self.market.n, self.market.num_programs])
        pref_lengths = np.asarray(
            self.market.preference_generator.pref_length, dtype=int
        )
        pref_lengths = np.clip(pref_lengths, 0, preferences.shape[1])
        preference_positions = np.arange(preferences.shape[1])
        listed_mask = preference_positions < pref_lengths[:, np.newaxis]
        program_idxs = preferences[listed_mask].astype(int) - 1
        student_idxs = np.repeat(np.arange(self.market.n), pref_lengths)
        valid_programs = np.logical_and(
            program_idxs >= 0, program_idxs < self.market.num_programs
        )
        not_designation[student_idxs[valid_programs], program_idxs[valid_programs]] = 1
        return not_designation

    def set_policy_specific_priorities(
        self,
        policy: Policy,
        preferences: np.ndarray,
        iteration: int | None = None,
    ) -> np.ndarray:
        """Set priority component from a specific policy.

        Args:
            policy (Policy): policy data object
            preferences (np.ndarray): (num students) by (num programs) array
                with student preferences

        Returns:
            np.ndarray: (num students) by (num programs) array with policy priority component
        """
        priorities = self.get_priorities_without_lottery(policy, preferences)
        lottery = self._set_tiebreaker(policy.tiebreaker, iteration=iteration)
        return priorities + lottery

    def get_priorities_with_lottery(
        self, policy: Policy, preferences: np.ndarray
    ) -> np.ndarray:
        """Get priority component from a specific policy without lottery.

        Args:
            policy (Policy): policy data object
            preferences (np.ndarray): (num students) by (num programs) array with student
                preferences

        Returns:
            np.ndarray: (num students) by (num programs) array with policy priority component
        """
        round_priorities = self._set_rounds_merged(policy.rounds_merged)
        priorities = self._set_policy_priorities(policy.ctip, policy.name)
        not_designation_mask = self._set_not_designation_priority(preferences)
        non_designation_boost = self.market.config.get("non_designation_boost", 100)
        final = (
            round_priorities
            + np.multiply(priorities, not_designation_mask)
            + not_designation_mask * non_designation_boost
        )
        if self.market.config["restrict-zone"]:
            zone_mask = self.market.zones.zone_eligibility_matrix
            return np.multiply(final, zone_mask) - (1 - zone_mask) * 500
        return final

    def get_priorities_without_lottery(
        self, policy: Policy, preferences: np.ndarray
    ) -> np.ndarray:
        """Get priority component from a specific policy without lottery.

        Args:
            policy (Policy): policy data object
            preferences (np.ndarray): (num students) by (num programs) array with student
                preferences

        Returns:
            np.ndarray: (num students) by (num programs) array with policy priority component
        """
        round_priorities = self._set_rounds_merged(policy.rounds_merged)
        priorities = self._set_policy_priorities(policy.ctip, policy.name)
        not_designation_mask = self._set_not_designation_priority(preferences)
        non_designation_boost = self.market.config.get("non_designation_boost", 100)
        final = (
            round_priorities
            + np.multiply(priorities, not_designation_mask)
            + not_designation_mask * non_designation_boost
        )
        if self.market.config["restrict-zone"]:
            zone_mask = self.market.zones.zone_eligibility_matrix
            return np.multiply(final, zone_mask) - (1 - zone_mask) * 500
        return final
