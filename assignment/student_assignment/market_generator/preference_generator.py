import numpy as np
import pandas as pd


class PreferenceGenerator:
    def __init__(self, market):
        self.market = market
        self._designation_ordering = None
        self._designation_ordering_cache = {}
        self._eligibility_cache = {}
        self._real_preferences_cache = {}

    def _cache_context(self):
        return getattr(self.market, "_active_policy_cache_context", None)

    def initialize_real_preferences(self, designate: bool = True) -> np.ndarray:
        """Initialize historical preferences, using the first round a student participated in.

        Args:
            designate (bool, optional): Whether to include designation programs in the preferences. Defaults to True.

        Returns:
            np.ndarray: A (num students) by (num programs) matrix, where the entry (i, j) is the program index of the
                jth program on the preference list of the ith student.
        """
        cache_context = self._cache_context()
        add_aa_schools = self.market.config.get("add_aa_schools", False)
        drop_below_aa = self.market.config.get("drop_below_aa", False)
        remove_non_aa_or_citywide = self.market.config.get(
            "remove_non_aa_or_citywide", False
        )
        cache_key = (
            cache_context,
            designate,
            add_aa_schools,
            drop_below_aa,
            remove_non_aa_or_citywide,
        )
        if cache_context is not None and cache_key in self._real_preferences_cache:
            prefs, pref_length = self._real_preferences_cache[cache_key]
            self.pref_length = pref_length.copy()
            return prefs

        prefs = np.zeros((self.market.n, self.market.num_programs))
        main_round = self.market.students.first_round

        for round in np.unique(main_round):
            round_preferences = self.market.students.student_preferences(
                round + 1, self.market.programs.index_list
            )
            round_idxs = np.where(main_round == round)[0]
            prefs[round_idxs, :] = round_preferences[round_idxs, :]

        self.pref_length = np.count_nonzero(prefs, axis=1)

        if add_aa_schools:
            prefs = self._add_attendance_area_schools_to_preferences(prefs)
        if drop_below_aa:
            prefs = self._drop_preferences_below_attendance_area(prefs)
        if designate:
            eligible = self._get_eligibility()
            prefs = self._add_designation_programs_to_preferences(
                prefs, eligible
            )
        if remove_non_aa_or_citywide:
            prefs = self._remove_non_aa_or_citywide_programs(prefs)
        if cache_context is not None:
            self._real_preferences_cache[cache_key] = (
                prefs,
                self.pref_length.copy(),
            )
        return prefs

    def _add_attendance_area_schools_to_preferences(
        self, prefs: np.ndarray
    ) -> np.ndarray:
        """Append each student's attendance-area GE program when not already ranked."""
        combined_prefs = prefs.copy()
        pref_lengths = np.count_nonzero(combined_prefs, axis=1)
        attendance_areas = self.market.students.attendance_area
        grade = self.market.config["grade"]

        for student_idx, studentno in self.market.students.idx2studentno.items():
            attendance_area = attendance_areas.get(studentno, 0)
            program_idx = self.market.programs.indices.get(
                f"{attendance_area}-GE-{grade}"
            )
            pref_length = pref_lengths[student_idx]
            if (
                program_idx is None
                or program_idx in combined_prefs[student_idx, :pref_length]
                or pref_length >= combined_prefs.shape[1]
            ):
                continue
            combined_prefs[student_idx, pref_length] = program_idx
            pref_lengths[student_idx] += 1

        self.pref_length = pref_lengths
        return combined_prefs

    def _drop_preferences_below_attendance_area(
        self, prefs: np.ndarray
    ) -> np.ndarray:
        """Drop programs ranked after each student's attendance-area GE program."""
        truncated_prefs = prefs.copy()
        pref_lengths = np.count_nonzero(truncated_prefs, axis=1)
        attendance_areas = self.market.students.attendance_area
        grade = self.market.config["grade"]

        for student_idx, studentno in self.market.students.idx2studentno.items():
            attendance_area = attendance_areas.get(studentno, 0)
            program_idx = self.market.programs.indices.get(
                f"{attendance_area}-GE-{grade}"
            )
            if program_idx is None:
                continue

            aa_ranks = np.flatnonzero(
                truncated_prefs[student_idx] == program_idx
            )
            if not aa_ranks.size:
                continue

            pref_length = int(aa_ranks[0]) + 1
            truncated_prefs[student_idx, pref_length:] = 0
            pref_lengths[student_idx] = pref_length

        self.pref_length = pref_lengths
        return truncated_prefs

    def _get_aa_or_citywide_eligibility(self) -> np.ndarray:
        """Identify programs at each student's AA school or a citywide school."""
        eligible = np.zeros((self.market.n, self.market.num_programs), dtype=bool)
        citywide_programs = self.market.programs.citywide_program_indices(
            self.market.schools.citywide_schools
        )
        if citywide_programs:
            eligible[:, np.asarray(citywide_programs, dtype=int) - 1] = True

        attendance_areas = self.market.students.attendance_area
        school_to_indices = self.market.programs.school_to_indices
        for student_idx, studentno in self.market.students.idx2studentno.items():
            attendance_area = attendance_areas.get(studentno, 0)
            aa_programs = school_to_indices.get(attendance_area, [])
            if aa_programs:
                eligible[student_idx, np.asarray(aa_programs, dtype=int) - 1] = True
        return eligible

    def _remove_non_aa_or_citywide_programs(
        self,
        preferences: np.ndarray,
        eligible: np.ndarray | None = None,
    ) -> np.ndarray:
        """Keep only programs at each student's AA school or citywide schools."""
        if eligible is None:
            eligible = self._get_aa_or_citywide_eligibility()

        ranked_lengths = self.pref_length.copy()
        filtered_preferences = np.zeros_like(preferences)
        filtered_ranked_lengths = np.zeros(self.market.n, dtype=int)
        for student_idx, student_prefs in enumerate(preferences):
            ranked_prefs = student_prefs[: int(ranked_lengths[student_idx])]
            ranked_prefs = ranked_prefs[ranked_prefs != 0].astype(int)
            if len(ranked_prefs):
                filtered_ranked_lengths[student_idx] = np.count_nonzero(
                    eligible[student_idx, ranked_prefs - 1]
                )

            student_prefs = student_prefs[student_prefs != 0].astype(int)
            allowed_prefs = student_prefs[eligible[student_idx, student_prefs - 1]]
            filtered_preferences[student_idx, : len(allowed_prefs)] = allowed_prefs

        self.pref_length = filtered_ranked_lengths
        return filtered_preferences

    def _generate_designation_program_ordering(self):
        """For each student, create the ordering of programs for designation consideration.

        If the student requested program designation, the list is the closest language
        programs followed by the closest GE programs. If the student did not request program
        designation, the list is the closest GE programs. These lists are added to the end of
        the student's preference list when designation is allowed. Note that language eligibility
        testing is done when the designation ordering is appended to preferences.
        """
        designation_ordering_type = self.market.config.get(
            "designation-ordering-type", "in_zone"
        )
        cache_key = (self._cache_context(), designation_ordering_type)
        if (
            cache_key[0] is not None
            and cache_key in self._designation_ordering_cache
        ):
            self._designation_ordering = self._designation_ordering_cache[
                cache_key
            ]
            return

        distances = self.market.students.distance_data
        map_matrix = self.market.zones.zone_eligibility_matrix
        # sort distances by closest to furthest for every student
        lp_distances = distances[
            [x for x in distances.columns if x.split("-")[1] != "GE"]
        ]
        lp_sorted_distances = lp_distances.apply(
            lambda row: pd.Series(lp_distances.columns[np.argsort(row.values)]),
            axis=1,
        )
        ge_distances = distances[
            [x for x in distances.columns if x.split("-")[1] == "GE"]
        ]
        ge_sorted_distances = ge_distances.apply(
            lambda row: pd.Series(ge_distances.columns[np.argsort(row.values)]),
            axis=1,
        )

        def _get_student_designation_ordering(studentno):
            # language programs if student requested language program designation
            if self.market.students.language_designation[studentno] or (
                self.market.config["utility-model"].get(
                    "designate-lp-for-all", False
                )
                and self.market.config["utility-model"]["enable"]
            ):
                o_designated = set(lp_sorted_distances.loc[studentno])
                types = set(
                    self.market.students.student_data.loc[
                        studentno
                    ].program_types
                )
                designated = [
                    x for x in o_designated if x.split("-")[1] in types
                ]
            else:
                designated = []

            # closest GE programs
            eligible_ges = list(ge_sorted_distances.loc[studentno])
            designated += eligible_ges
            return self.market.programs.index_list(designated)

        def _get_student_designation_ordering_in_zone(studentno):
            # Step 1: Get student index for NumPy-based indexing
            student_idx = self.market.students.studentno2idx[studentno]

            program_to_index = self.market.programs.indices

            ge_eligible = list(
                ge_sorted_distances.loc[studentno]
            )  # Sorted GE programs
            lp_eligible = list(
                lp_sorted_distances.loc[studentno]
            )  # Sorted LP programs

            designated = []

            types = set(
                self.market.students.student_data.loc[studentno].program_types
            )

            # 1s mean the student qualifies for program, meaning
            if self.market.students.language_designation[studentno] or (
                self.market.config["utility-model"].get(
                    "designate-lp-for-all", False
                )
                and self.market.config["utility-model"]["enable"]
            ):
                lp_eligible_ones = [
                    lp
                    for lp in lp_eligible
                    if map_matrix[student_idx, program_to_index[lp] - 1]
                ]
                lp_eligible_ones = [
                    x for x in lp_eligible_ones if x.split("-")[1] in types
                ]
                designated += lp_eligible_ones

            ge_eligible_ones = [
                ge
                for ge in ge_eligible
                if map_matrix[student_idx, program_to_index[ge] - 1]
            ]
            designated += ge_eligible_ones

            # zeros are out inelligible, IE allows students to choose out of zone designation schools
            if self.market.students.language_designation[studentno] or (
                self.market.config["utility-model"].get(
                    "designate-lp-for-all", False
                )
                and self.market.config["utility-model"]["enable"]
            ):
                lp_eligible_zeros = [
                    lp
                    for lp in lp_eligible
                    if not map_matrix[student_idx, program_to_index[lp] - 1]
                ]
                lp_eligible_zeros = [
                    x for x in lp_eligible_zeros if x.split("-")[1] in types
                ]
                designated += lp_eligible_zeros

            ge_eligible_zeros = [
                ge
                for ge in ge_eligible
                if not map_matrix[student_idx, program_to_index[ge] - 1]
            ]
            designated += ge_eligible_zeros

            return self.market.programs.index_list(designated)

        if designation_ordering_type == "in_zone":
            ordering_func = _get_student_designation_ordering_in_zone
        elif designation_ordering_type == "simple":
            ordering_func = _get_student_designation_ordering
        else:
            raise ValueError(
                f"Unknown designation ordering type: {designation_ordering_type}"
            )

        self._designation_ordering = {
            studentno: ordering_func(studentno)
            for studentno in self.market.students.studentno2idx.keys()
        }
        if cache_key[0] is not None:
            self._designation_ordering_cache[cache_key] = (
                self._designation_ordering
            )
    def _add_designation_programs_to_preferences(
        self, prefs: np.ndarray, eligible: np.ndarray
    ) -> np.ndarray:
        """Add designation programs to the end of the preference list for each student.

        Args:
            prefs (np.ndarray): A (num students) by (num programs) matrix, where the entry (i, j) is the program index
                of the jth program on the preference list of the ith student.
            eligible (np.ndarray): A 0-1 (num students) by (num programs) matrix, where 1 indicates the student is
                eligible for that program.

        Returns:
            np.ndarray: A (num students) by (num programs) matrix, the preferences input with designation programs
                added to the end of each student's preference list.
        """
        self._generate_designation_program_ordering()

        combined_prefs = np.zeros_like(prefs)
        pref_lengths = np.zeros(self.market.n, dtype=int)

        for i, studentno in self.market.students.idx2studentno.items():
            ranked = list(
                dict.fromkeys(int(program) for program in prefs[i] if program != 0)
            )
            pref_lengths[i] = sum(
                bool(eligible[i, program - 1]) for program in ranked
            )
            combined = list(
                dict.fromkeys(ranked + self._designation_ordering[studentno])
            )
            combined_prefs[i, : len(combined)] = combined

        self.pref_length = pref_lengths
        return self._remove_ineligible_programs(combined_prefs, eligible)

    def _get_program_type_eligibility_matrix(self) -> np.ndarray:
        """Create a matrix of program type eligibility for each student.

        Returns:
            np.ndarray: A 0-1 (num students) by (num programs) matrix, where True indicates the student is eligible for
                that program
        """
        eligibility_map = self.market.students.get_qualified_programs_dict()
        program_type_idxs = self.market.programs.program_type_to_indices
        eligible = np.zeros((self.market.n, self.market.num_programs))
        for (
            student_idx,
            studentno,
        ) in self.market.students.idx2studentno.items():
            for program_type in eligibility_map[studentno]:
                if (
                    program_type
                    in self.market.programs.program_df["program_type"].unique()
                ):
                    eligible[
                        student_idx,
                        [x - 1 for x in program_type_idxs[program_type]],
                    ] = 1

        return eligible

    def _remove_ineligible_programs(
        self, preferences: np.ndarray, eligible: np.ndarray
    ) -> np.ndarray:
        """Remove ineligible programs from the preference list of each student.

        Args:
            preferences (np.ndarray): A (num students) by (num programs) matrix, where the entry (i, j) is the program
                index of the jth program on the preference list of the ith student.
            eligible (np.ndarray): A 0-1 (num students) by (num programs) matrix, where 1 indicates the student is
                eligible for that program.

        Returns:
            np.ndarray: A (num students) by (num programs) matrix, the preferences input with ineligible programs
                removed from each student's preference list.
        """
        filtered_preferences = np.zeros_like(preferences)
        for student_idx, student_prefs in enumerate(preferences):
            program_prefs = student_prefs[student_prefs != 0].astype(int)
            if np.any((program_prefs < 1) | (program_prefs > eligible.shape[1])):
                raise ValueError("preference contains an invalid program index")
            allowed_prefs = program_prefs[
                eligible[student_idx, program_prefs - 1].astype(bool)
            ]
            filtered_preferences[student_idx, : len(allowed_prefs)] = allowed_prefs
        return filtered_preferences

    def _truncate_utility_model_preferences(
        self, eligible: np.ndarray
    ) -> np.ndarray:
        """Truncate utility model preferences to the list length specified in the config.

        Args:
            eligible (np.ndarray): A 0-1 (num students) by (num programs) matrix, where 1 indicates the student is
                eligible for that program.

        Returns:
            np.ndarray: A (num students) by (num programs) matrix, where the appropriate number
                of programs are listed after removing ineligible programs. Does not include
                designation programs.
        """
        num_eligible = eligible.sum(axis=1)
        filtered_preferences = self._remove_ineligible_programs(
            self.market.umodel.original_preferences, eligible
        )

        num_ranked_array = self.set_number_programs_ranked()
        truncated_prefs = np.zeros_like(filtered_preferences)
        students_aa = self.market.students.attendance_area
        grade = self.market.config["grade"]
        truncate_at_aa = self.market.config.get("truncate-at-AA-GE", False)
        for i, student_prefs in enumerate(filtered_preferences):
            num_progs = int(min(num_eligible[i], num_ranked_array[i]))
            if truncate_at_aa:
                # Truncate program at AA GE program
                cur_student_aa = students_aa[
                    self.market.students.idx2studentno[i]
                ]
                cur_aa_program_id = f"{cur_student_aa}-GE-{grade}"
                program_idx = self.market.programs.indices[cur_aa_program_id]
                program_idx = np.argwhere(
                    student_prefs == program_idx
                ).flatten()[0]
                # Truncate at program_idx + 1 to include AA GE program.
                num_progs = min(num_progs, program_idx + 1)

            if num_progs == 0:
                # print(
                #     f"Not ranking any programs for student {self.market.students.idx2studentno[i]}"
                # )
                continue
            truncated_prefs[i, :num_progs] = student_prefs[:num_progs]
        return truncated_prefs

    def set_number_programs_ranked(self) -> np.ndarray:
        """Set the number of programs ranked for each student based on the config.

        Returns:
            np.ndarray: A (num students) by 1 array, where the entry i is the number of
                programs ranked by the ith student.
        """
        # TODO: refactor
        num_students = self.market.n
        student_data = self.market.students.student_data
        length = self.market.config["utility-model"]["list-length"]
        if length == "real_length":
            num_ranked_array = np.array(student_data["num_ranked"])
        elif length == "real_length_x2":
            num_ranked_array = 2 * np.array(student_data["num_ranked"])
        elif length == "0.8*round(real_length)":
            num_ranked_array = np.round(
                0.8 * np.array(student_data["num_ranked"])
            )
            num_ranked_array = np.maximum(3, num_ranked_array)
        elif length == "0.7*round(real_length)":
            num_ranked_array = np.round(
                0.7 * np.array(student_data["num_ranked"])
            )
            num_ranked_array = np.maximum(3, num_ranked_array)
        elif length == "0.6*round(real_length)":
            num_ranked_array = np.round(
                0.6 * np.array(student_data["num_ranked"])
            )
            num_ranked_array = np.maximum(3, num_ranked_array)
        elif length == "0.5*round(real_length)":
            num_ranked_array = np.ceil(
                0.5 * np.array(student_data["num_ranked"])
            )
        elif length == "real_length_+3":
            num_ranked_array = np.array(
                student_data["num_ranked"]
            ) + 3 * np.ones([student_data])
        # If option is a number, all students list option number of programs
        elif length.isnumeric():
            num_ranked_array = int(length) * np.ones(num_students)
        elif length == "length_by_ethn":
            num_ranked_array = np.zeros(num_students)
            numerator_dict = {}
            denominator_dict = {}
            for i in range(num_students):
                ethn = student_data.resolved_ethnicity.iloc[i]
                num_ranked = student_data["num_ranked"].iloc[i]
                if ethn in numerator_dict:
                    numerator_dict[ethn] += num_ranked
                    denominator_dict[ethn] += 1
                else:
                    numerator_dict[ethn] = 0
                    denominator_dict[ethn] = 1
            for i in range(num_students):
                ethn = student_data.resolved_ethnicity.iloc[i]
                num_ranked_array[i] = (
                    numerator_dict[ethn] / denominator_dict[ethn]
                )
        elif length == "length_by_ctip":
            num_ranked_array = np.zeros(num_students)
            student_data["ctip1"] = student_data["ctip1"].fillna(0)
            ctip1 = student_data[student_data["ctip1"] == 1]["num_ranked"]
            ctip1_length = ctip1.sum() / ctip1.count()

            ctip0 = student_data[student_data["ctip1"] == 0]["num_ranked"]
            ctip0_length = ctip0.sum() / ctip0.count()

            for i in range(num_students):
                ctip = student_data.ctip1.iloc[i]
                num_ranked_array[i] = (
                    ctip1_length if ctip == 1 else ctip0_length
                )

        elif length == "length_by_frl":
            num_ranked_array = np.zeros(num_students)
            frl1 = student_data[student_data["FRL Score"] >= 0.5]["num_ranked"]
            frl1_length = frl1.sum() / frl1.count()

            frl0 = student_data[student_data["FRL Score"] < 0.5]["num_ranked"]
            frl0_length = frl0.sum() / frl0.count()

            for i in range(num_students):
                frl_score = student_data["FRL Score"].iloc[i]
                num_ranked_array[i] = (
                    frl1_length if frl_score >= 0.5 else frl0_length
                )

        elif length == "length_by_income":
            num_ranked_array = np.zeros(num_students)
            high_income = student_data[
                student_data["median_hh_income"] >= 95292
            ]["num_ranked"]
            high_income_length = high_income.sum() / high_income.count()

            low_income = student_data[student_data["median_hh_income"] < 95292][
                "num_ranked"
            ]
            low_income_length = low_income.sum() / low_income.count()

            for i in range(num_students):
                income = student_data["median_hh_income"].iloc[i]
                num_ranked_array[i] = (
                    high_income_length if income >= 95292 else low_income_length
                )

        elif length == "all_eligible":
            num_ranked_array = np.ones(num_students) * self.market.num_programs
        else:
            raise ValueError(
                f"Utility model preference length {length} (type {type(length)} not recognized."
            )
        return num_ranked_array

    def _get_eligibility(self) -> np.ndarray:
        """Get a matrix identifying which students are eligible for which programs.

        Includes both program/language eligibility and zone eligibility.

        Returns:
            np.ndarray: A 0-1 (num students) by (num programs) matrix, where 1 indicates
                the student is eligible for that program.
        """
        cache_context = self._cache_context()
        if (
            cache_context is not None
            and cache_context in self._eligibility_cache
        ):
            return self._eligibility_cache[cache_context]

        program_eligibility = self._get_program_type_eligibility_matrix()

        if hasattr(
            self.market.zones, "area_id2prog_list"
        ):  # check that zones have been set
            zone_eligibility = self.market.zones.zone_eligibility_matrix
            eligible = np.logical_and(program_eligibility, zone_eligibility)
            if cache_context is not None:
                self._eligibility_cache[cache_context] = eligible
            return eligible
        if cache_context is not None:
            self._eligibility_cache[cache_context] = program_eligibility
        return program_eligibility

    def get_utility_model_preferences_after_truncation(self) -> np.ndarray:
        """Get utility model preferences after removing ineligible programs and truncating the list length.

        Returns:
            np.ndarray: A (num students) by (num programs) matrix, where the appropriate number
                of programs are listed after removing ineligible programs.
        """
        eligible = self._get_eligibility()
        aa_or_citywide_eligible = None
        remove_non_aa_or_citywide = self.market.config.get(
            "remove_non_aa_or_citywide", False
        )
        if remove_non_aa_or_citywide:
            aa_or_citywide_eligible = self._get_aa_or_citywide_eligibility()
            eligible = np.logical_and(eligible, aa_or_citywide_eligible)
        prefs = self._truncate_utility_model_preferences(eligible)
        self.pref_length = np.count_nonzero(prefs, axis=1)
        if self.market.config.get("add_aa_schools", False):
            prefs = self._add_attendance_area_schools_to_preferences(prefs)
        if self.market.config.get("drop_below_aa", False):
            prefs = self._drop_preferences_below_attendance_area(prefs)
        if self.market.config["designate"]:
            prefs = self._add_designation_programs_to_preferences(
                prefs, eligible
            )
        if remove_non_aa_or_citywide:
            prefs = self._remove_non_aa_or_citywide_programs(
                prefs, aa_or_citywide_eligible
            )
        return prefs
