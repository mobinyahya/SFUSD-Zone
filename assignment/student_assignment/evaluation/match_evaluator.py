import csv
import math

import numpy as np
import pandas as pd

LANG_CODES = (
    "CN",
    "CE",
    "CB",
    "CT",
    "NC",
    "JB",
    "JE",
    "JN",
    "KN",
    "KE",
    "MN",
    "ME",
    "SB",
    "SN",
    "SE",
    "NS",
    "FB",
)
ELL_CODES = (
    "CN",
    "CE",
    "CB",
    "CT",
    "JE",
    "JN",
    "KN",
    "KE",
    "MN",
    "ME",
    "SB",
    "SN",
    "SE",
    "NS",
)


class MatchEvaluator:
    def __init__(self, students, assignments, distances):
        self.students = students
        self.student_data = students.student_data
        self.distance_data = distances

        if "assigned_utility" in assignments.columns:
            assignments["assigned_utility"] = assignments[
                "assigned_utility"
            ].replace(-np.inf, np.nan)
            self.assigned_utility = assignments["assigned_utility"]
            self.assigned_utility.min()
            self.utility_exists = 1
        else:
            self.utility_exists = 0

        self.assignments = assignments  # .set_index('studentno')
        self.student_data = self.student_data.merge(
            self.assignments, how="left", right_index=True, left_index=True
        )
        self.student_data.rename(
            columns={"programcodes": "assignment"}, inplace=True
        )
        self.match_ranks = self.assignments["rank"]
        self.student_data["main"] = pd.Series(
            np.argmax(students.round_participation, axis=1) + 1,
            index=students.student_data.index,
        )  # should be done in Students
        self.student_data["programtype"] = self.student_data["assignment"].str[
            4:6
        ]
        self.student_data["frl"] = (
            self.student_data["freelunch_prob"]
            + self.student_data["reducedlunch_prob"]
        )
        self.student_data["assigned school"] = self.student_data[
            "assignment"
        ].str[:3]
        if "In-Zone Rank" not in self.student_data.columns:
            self.student_data["In-Zone Rank"] = np.nan
        self.num_students = self.student_data.shape[0]
        self.num_schools = self.student_data["assigned school"].nunique()
        self.eval_distance()
        self.eval_matrix = self._make_eval_matrix()
        (
            self.ethnic_groups,
            self.ethnic_matrix,
            self.ethnic_matrix_norm,
            self.ethnic_total,
            self.ethnic_total_norm,
            self.populations,
        ) = self._make_ethnic_matrix()
        self.eths = [
            "Black or African American",
            "Asian",
            "Hispanic/Latino",
            "Two or More Races",
            "Pacific Islander",
            "White",
        ]
        self.eth_labels = [
            "Black",
            "Asian",
            "Hispanic",
            "Multiracial",
            "PI",
            "White",
        ]

    def eval_assignment_paper_metrics(self):
        metrics = {}
        student_data = self.student_data
        assigned_students = student_data[student_data["programno"] > 0]
        assigned_students = assigned_students.reindex()
        assigned_students.shape[0]

        school_groups = assigned_students.groupby("assigned school")
        enrollment = school_groups.count()[
            "SES_category"
        ]  # Just picking an arbitrary column to get the count

        # PROXIMITY
        metrics["Distance Av"] = self.metric_dist_av(assigned_students)
        metrics["Distance < 0.5"] = self.metric_dist_threshold(
            assigned_students, 0.5, False
        )
        metrics["Distance > 3"] = self.metric_dist_threshold(
            assigned_students, 3, True
        )

        # DIVERSITY
        metrics["Schools above 10% district FRL"] = (
            self.metric_school_frl_above_district(0.1)
        )
        metrics["Schools above 15% district FRL"] = (
            self.metric_school_frl_above_district(0.15)
        )
        AALPI = [
            "Black or African American",
            "Hispanic/Latino",
            "Pacific Islander",
        ]
        aalpi_students = assigned_students[
            assigned_students["resolved_ethnicity"].apply(lambda x: x in AALPI)
        ]
        metrics["AALPI in school with +10% FRL"] = (
            self.metric_FRL_concentration(
                assigned_students, aalpi_students, 0.1
            )
        )
        metrics["AALPI in school with +15% FRL"] = (
            self.metric_FRL_concentration(
                assigned_students, aalpi_students, 0.15
            )
        )
        metrics["Dissimilarity AALPI"] = self.metric_dissimilarity(
            aalpi_students, enrollment
        )
        ses3_students = assigned_students[
            assigned_students["SES_category"] == 3
        ]
        metrics["Dissimilarity SES3"] = self.dissimilarity(
            ses3_students, enrollment
        )
        # black_isolated = sum([1 for x in self.ethnic_matrix['Black or African American'] if 1 <= x <= 4])
        # metrics["Programs with 1-4 AA"] = black_isolated/len(self.ethnic_matrix)
        metrics["Programs with 1-4 AA"] = self.metric_isolation(
            assigned_students[
                assigned_students["resolved_ethnicity"]
                == "Black or African American"
            ],
            5,
        )

        # CHOICE
        metrics["Unassigned"] = self.metric_unassigned(student_data)
        metrics["Designated"] = self.metric_designated(assigned_students)

        metrics["Top 3 choice"] = self.metric_top_choice(assigned_students, 3)
        metrics["Top 1 choice"] = self.metric_top_choice(assigned_students, 1)
        (assigned_students["In-Zone Rank"] == 1).mean()
        (assigned_students["In-Zone Rank"] <= 3).mean()
        metrics["Top 3 in-zone choice"] = self.metric_top_in_zone_choice(
            assigned_students, 3
        )
        metrics["Top 1 in-zone choice"] = self.metric_top_in_zone_choice(
            assigned_students, 1
        )
        metrics["Dist >= 3, Rank >= 5"] = self.metric_dist_and_rank(
            assigned_students, 3, 5
        )
        if "assigned_utility" in assigned_students.columns:
            average_utility = assigned_students["assigned_utility"].mean()
        else:
            average_utility = np.nan
        metrics["Avg utility"] = average_utility

        # COMMUNITY COHESION
        metrics["BG Cohesion (3)"] = self.metric_BG_cohesion(
            assigned_students, 3
        )

        # EQUITY OF ACCESS
        high_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) > 0.5  # High FRL students have > 50% estimated chance of being FRL
        low_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) <= 0.5  # Low FRL students have <= 50% estimated chance of being FRL
        high_frl_students = student_data[high_frl_prob]
        low_frl_students = student_data[low_frl_prob]
        high_frl_assigned = high_frl_students[
            high_frl_students["programno"] > 0
        ]
        low_frl_assigned = low_frl_students[low_frl_students["programno"] > 0]
        groups = [
            "Black or African American",
            "Asian",
            "Hispanic/Latino",
            "Pacific Islander",
            "White",
            "High FRL",
            "Low FRL",
        ]
        for group in groups:
            if group == "High FRL":
                students = high_frl_assigned
            elif group == "Low FRL":
                students = low_frl_assigned
            else:
                students = assigned_students[
                    assigned_students["resolved_ethnicity"] == group
                ]

            metrics[f"Top 3 choice {group}"] = self.metric_top_choice(
                students, 3
            )
            metrics[f"Distance Av {group}"] = self.metric_dist_av(students)
            metrics[f"{group} in school with +15% FRL"] = (
                self.metric_FRL_concentration(assigned_students, students, 0.15)
            )
            metrics[f"{group} Dist >= 3, Rank >= 5"] = (
                self.metric_dist_and_rank(students, 3, 5)
            )

        return pd.Series(metrics)

    #    BASE METRICS
    # --------------------

    #     Proximity
    # --------------------

    # Name: Distance Av.
    # Inputs: Assigned students
    # Average straight-line distance of assigned students to their assigned school
    def metric_dist_av(self, assigned_students):
        return assigned_students["assignment_dist"].mean()

    # Name: Distance < X or Distance > X
    # Inputs: Assigned students, threshold distance X, above = True if Distance > X
    # Fraction of assigned students with straight-line distance to their assigned school above or below given threshold
    def metric_dist_threshold(self, assigned_students, threshold, above):
        if above:
            return (assigned_students["assignment_dist"] > threshold).mean()
        else:
            return (assigned_students["assignment_dist"] < threshold).mean()

    #     Diversity
    # --------------------

    # Name: Schools above X% district FRL
    # Inputs: Threshold X
    # Fraction of schools where average FRL status of students assigned to the school is at least X% above the fraction of all students with FRL status
    def metric_school_frl_above_district(self, threshold, student_data=None):
        if student_data is None:
            student_data = self.student_data
        district_avg = student_data["frl"].mean()
        school_frl = student_data.groupby("assigned school").mean(
            numeric_only=True
        )
        school_frl["in_range"] = school_frl["frl"].apply(
            lambda x: 1 if x >= district_avg + threshold else 0
        )
        return school_frl["in_range"].mean()

    # Name: [Student group] in School with +X% FRL
    # Inputs: Assigned students, assigned students in student group, threshold X
    # Fraction of assigned students in student group who are assigned to a school where average FRL status of students assigned to the school is
    #   at least X% above the fraction of all students with FRL status
    def metric_FRL_concentration(self, all_students, group_students, threshold):
        if group_students.empty:
            return np.nan
        schools_frl = all_students.groupby("assigned school").mean(
            numeric_only=True
        )["frl"]
        district_avg = all_students["frl"].mean()
        num_students = group_students.shape[0]
        count = 0
        for i in range(num_students):
            school = group_students["assigned school"].iloc[i]
            if isinstance(school, str):
                school_frl = schools_frl.loc[school]
                if school_frl > district_avg + threshold:
                    count += 1
        return count / num_students

    # Name: Dissimilarity [student group]
    # Inputs: students in group, total number of students assigned to each school
    # Fraction of students in student group who would have to be assigned to a different school so that the student group would be
    #   uniformly distributed across schools.
    def metric_dissimilarity(self, group_students, total_enrollment):
        students = group_students
        n = students.shape[0]
        total_n = pd.to_numeric(
            pd.Series(np.asarray(total_enrollment).reshape(-1)), errors="coerce"
        ).sum()
        if n == 0 or total_n == 0:
            return np.nan
        ratio = n / total_n
        school_groups = students.groupby("assigned school")
        enrollment = school_groups.count()[
            "SES_category"
        ]  # Just picking an arbitrary column to get the count
        dissimilarity_total = 0
        for i in range(enrollment.shape[0]):
            num_students = enrollment.iloc[i]
            total_students = total_enrollment.iloc[i]
            dissimilarity_total += (
                abs(num_students - total_students * ratio) / 2
            )
        return dissimilarity_total / n

    # Name: Isolation [student group] < X
    # Inputs: students in group, theshold X
    # Number of schools with at least one and less than X students assigned from the student group
    def metric_isolation(self, group_students, threshold):
        school_groups = group_students.groupby("assigned school")
        enrollment = school_groups.count()[
            "SES_category"
        ]  # Just picking an arbitrary column to get the count
        count = 0
        for i in range(enrollment.shape[0]):
            num_students = enrollment.iloc[i]
            if num_students < threshold and num_students >= 1:
                count += 1
        return count

    #      Choice
    # --------------------

    # Name: Unassigned
    # Inputs: students
    # Fraction of students who are not assigned to any program
    def metric_unassigned(self, students):
        return students[students["programno"] == 0].shape[0] / students.shape[0]

    # Name: Designated
    # Inputs: assigned students
    # Fraction of assigned students who are assigned through designation
    def metric_designated(self, assigned_students):
        return assigned_students["designation"].mean()

    # Name: Top X choice
    # Inputs: assigned students, threshold X
    # Fraction of assigned students who are assigned to one of their top X choices out of *all* choices
    def metric_top_choice(self, assigned_students, threshold):
        return (assigned_students["rank"] <= threshold).mean()

    # Name: Top X in-zone choice
    # Inputs: assigned students, threshold X
    # Fraction of assigned students who are assigned to one of their top X choices out of the choices in their zone
    def metric_top_in_zone_choice(self, assigned_students, threshold):
        return (assigned_students["In-Zone Rank"] <= threshold).mean()

    # Name: Dist >= X, Rank >= Y
    # Inputs: assigned students, threshold X, threshold Y
    # Fraction of assigned students who have both straight-line distance to school >= X and receiving choice >= Y out of *all* choices
    def metric_dist_and_rank(self, assigned_students, X, Y):
        return np.logical_and(
            assigned_students["assignment_dist"] >= X,
            assigned_students["rank"] >= Y,
        )

    # Community Cohesion
    # --------------------

    # Name: BG cohesion (X)
    # Inputs: assigned_students, number X
    # Fraction of assigned students who are assigned to a school with at least X students from their block group (including themself)
    def metric_BG_cohesion(self, assigned_students, num):
        if assigned_students.empty:
            return np.nan
        cohesion = sum(
            self._bgcohesion(group, num)
            for _, group in assigned_students.groupby("census_blockgroup")
        )
        return cohesion / assigned_students.shape[0]

    def eval_assignment_full(self, school_data, real_match=False):
        metrics = {}
        student_data = self.student_data
        assigned_students = student_data[student_data["programno"] > 0]
        assigned_students = assigned_students.reindex()
        assigned_students.shape[0]

        school_groups = assigned_students.groupby("assigned school")
        enrollment = school_groups.count()[
            "SES_category"
        ]  # Just picking an arbitrary column to get the count

        metrics["Unassigned"] = (
            student_data[student_data["programno"] == 0].shape[0]
            / student_data.shape[0]
        )
        all_designated = assigned_students["designation"].mean()
        metrics["Designated"] = all_designated
        choice_3 = (assigned_students["rank"] <= 3).mean()
        metrics["Top 3 choice"] = choice_3

        """
        if real_match is False:
            average_utility = assigned_students['assigned_utility'].mean()
            metrics['Average utility'] = average_utility
        """

        metrics["Distance Av"] = assigned_students["assignment_dist"].mean()
        metrics["Distance < 0.5"] = (
            assigned_students["assignment_dist"] < 0.5
        ).mean()
        metrics["Distance > 3"] = (
            assigned_students["assignment_dist"] > 3
        ).mean()
        metrics["All AvgColorIndex"] = self.avg_color_index(
            assigned_students, school_data
        )

        high_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) > 0.5
        low_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) <= 0.5
        high_frl_students = student_data[high_frl_prob]
        low_frl_students = student_data[low_frl_prob]
        high_frl_assigned = high_frl_students[
            high_frl_students["programno"] > 0
        ]
        low_frl_assigned = low_frl_students[low_frl_students["programno"] > 0]

        groups = [
            "All",
            "Black or African American",
            "Asian",
            "Hispanic/Latino",
            "Pacific Islander",
            "White",
            "High FRL",
            "Low FRL",
        ]
        groups = ["Round One"]
        for group in groups:
            if group == "All":
                students = assigned_students
                full_students = student_data
            elif group == "High FRL":
                students = high_frl_assigned
                full_students = high_frl_students
            elif group == "Low FRL":
                students = low_frl_assigned
                full_students = low_frl_students
            elif group == "Round One":
                full_students = student_data[student_data["main"] == 1]
                students = full_students[full_students["programno"] > 0]
            else:
                students = assigned_students[
                    assigned_students["resolved_ethnicity"] == group
                ]
                full_students = student_data[
                    student_data["resolved_ethnicity"] == group
                ]

            metrics[f"Unassigned {group}"] = (
                full_students.shape[0] - students.shape[0]
            ) / (full_students.shape[0])

            metrics[f"Distance Av {group}"] = students["assignment_dist"].mean()
            metrics[f"Distance < 0.5 {group}"] = (
                students["assignment_dist"] < 0.5
            ).mean()
            metrics[f"Distance > 3 {group}"] = (
                students["assignment_dist"] > 3
            ).mean()
            metrics[f"Top 3 choice {group}"] = (students["rank"] <= 3).mean()
            if group != "All":
                metrics[f"{group} in school with +10% FRL"] = (
                    self.poverty_concentration(assigned_students, students, 0.1)
                )
                metrics[f"{group} in school with +15% FRL"] = (
                    self.poverty_concentration(
                        assigned_students, students, 0.15
                    )
                )
            metrics[f"{group} AvgColorIndex"] = self.avg_color_index(
                students, school_data
            )
            """
            if real_match is False:
                average_utility = students['assigned_utility'].mean()
                metrics['{} Average utility'.format(eth)] = average_utility
            """

            metrics[f"Dissimilarity {group}"] = self.dissimilarity(
                students, enrollment
            )

        metrics["Schools above 10% district FRL"] = (
            self.school_frl_range_district(0.1, above=True)
        )
        metrics["Schools above 15% district FRL"] = (
            self.school_frl_range_district(0.15, above=True)
        )

        ge_students = assigned_students[
            assigned_students["programtype"] == "GE"
        ]
        ge_students.groupby("assignment")
        ge_students["assigned school"].nunique()

        AALPI = [
            "Black or African American",
            "Hispanic/Latino",
            "Pacific Islander",
        ]
        aalpi_students = assigned_students[
            assigned_students["resolved_ethnicity"].apply(lambda x: x in AALPI)
        ]
        metrics["AALPI in school with +10% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, 0.1
        )
        metrics["AALPI in school with +15% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, 0.15
        )

        metrics["BG Cohesion (3)"] = (
            assigned_students.groupby("census_blockgroup")
            .apply(lambda x: self._bgcohesion(x, 3))
            .sum()
            / assigned_students.shape[0]
        )

        return pd.Series(metrics)

        # ctip_students = student_data[student_data['ctip1'] > 0]
        # ctip_assigned = assigned_students[assigned_students['ctip1'] > 0]
        # metrics['Low FRL +10% FRL'] = self.poverty_concentration(low_frl_assigned,0.1)

    def eval_assignment_overview(self, school_data, real_match=False):
        metrics = {}
        student_data = self.student_data
        assigned_students = student_data[student_data["programno"] > 0]
        assigned_students = assigned_students.reindex()
        assigned_students.shape[0]

        school_groups = assigned_students.groupby("assigned school")
        enrollment = school_groups.count()[
            "SES_category"
        ]  # Just picking an arbitrary column to get the count

        metrics["Unassigned"] = (
            student_data[student_data["programno"] == 0].shape[0]
            / student_data.shape[0]
        )
        all_designated = assigned_students["designation"].mean()
        metrics["Designated"] = all_designated
        choice_3 = (assigned_students["rank"] <= 3).mean()
        metrics["Top 3 choice"] = choice_3

        """
        if real_match is False:
            average_utility = assigned_students['assigned_utility'].mean()
            metrics['Average utility'] = average_utility
        """

        metrics["Distance Av"] = assigned_students["assignment_dist"].mean()
        metrics["Distance < 0.5"] = (
            assigned_students["assignment_dist"] < 0.5
        ).mean()
        metrics["Distance > 3"] = (
            assigned_students["assignment_dist"] > 3
        ).mean()

        high_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) > 0.5
        low_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) <= 0.5
        high_frl_students = student_data[high_frl_prob]
        low_frl_students = student_data[low_frl_prob]
        high_frl_assigned = high_frl_students[
            high_frl_students["programno"] > 0
        ]
        low_frl_assigned = low_frl_students[low_frl_students["programno"] > 0]

        groups = ["Black or African American", "Hispanic/Latino", "High FRL"]
        for group in groups:
            if group == "All":
                students = assigned_students
            elif group == "High FRL":
                students = high_frl_assigned
            elif group == "Low FRL":
                students = low_frl_assigned
            else:
                students = assigned_students[
                    assigned_students["resolved_ethnicity"] == group
                ]

            metrics[f"Dissimilarity {group}"] = self.dissimilarity(
                students, enrollment
            )

        metrics["Schools above 10% district FRL"] = (
            self.school_frl_range_district(0.1, above=True)
        )
        metrics["Schools above 15% district FRL"] = (
            self.school_frl_range_district(0.15, above=True)
        )

        ge_students = assigned_students[
            assigned_students["programtype"] == "GE"
        ]
        ge_students.groupby("assignment")
        ge_students["assigned school"].nunique()

        AALPI = [
            "Black or African American",
            "Hispanic/Latino",
            "Pacific Islander",
        ]
        aalpi_students = assigned_students[
            assigned_students["resolved_ethnicity"].apply(lambda x: x in AALPI)
        ]
        metrics["AALPI in school with +10% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, 0.1
        )
        metrics["AALPI in school with +15% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, 0.15
        )

        return pd.Series(metrics)

    def eval_assignment_equity(self, school_data, real_match=False):
        metrics = {}
        student_data = self.student_data
        assigned_students = student_data[student_data["programno"] > 0]
        assigned_students = assigned_students.reindex()
        assigned_students.shape[0]

        school_groups = assigned_students.groupby("assigned school")
        enrollment = school_groups.count()[
            "SES_category"
        ]  # Just picking an arbitrary column to get the count

        high_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) > 0.5
        low_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) <= 0.5
        high_frl_students = student_data[high_frl_prob]
        low_frl_students = student_data[low_frl_prob]
        high_frl_assigned = high_frl_students[
            high_frl_students["programno"] > 0
        ]
        low_frl_assigned = low_frl_students[low_frl_students["programno"] > 0]

        groups = [
            "Black or African American",
            "Asian",
            "Hispanic/Latino",
            "Pacific Islander",
            "White",
            "High FRL",
            "Low FRL",
        ]
        for group in groups:
            if group == "All":
                students = assigned_students
                full_students = student_data
            elif group == "High FRL":
                students = high_frl_assigned
                full_students = high_frl_students
            elif group == "Low FRL":
                students = low_frl_assigned
                full_students = low_frl_students
            else:
                students = assigned_students[
                    assigned_students["resolved_ethnicity"] == group
                ]
                full_students = student_data[
                    student_data["resolved_ethnicity"] == group
                ]

            metrics[f"Unassigned {group}"] = (
                full_students.shape[0] - students.shape[0]
            ) / (full_students.shape[0])
            metrics[f"Distance Av {group}"] = students["assignment_dist"].mean()
            metrics[f"Top 3 choice {group}"] = (students["rank"] <= 3).mean()
            metrics[f"{group} in school with +10% FRL"] = (
                self.poverty_concentration(assigned_students, students, 0.1)
            )
            metrics[f"{group} in school with +15% FRL"] = (
                self.poverty_concentration(assigned_students, students, 0.15)
            )
            metrics[f"{group} AvgColorIndex"] = self.avg_color_index(
                students, school_data
            )
            metrics[f"Dissimilarity {group}"] = self.dissimilarity(
                students, enrollment
            )

        return pd.Series(metrics)

    def _pctunderserved(self, x, frl=0.67, aalpi=0.5, el=0.2):
        frl_pct = x["high_poverty"].mean()
        if frl_pct < frl:
            return False
        aalpi_pct = x["aalpi"].mean()
        el_pct = x["ell"].mean()
        if aalpi_pct > aalpi:
            return True
        if el_pct > el:
            return True
        return False

    def _ethnicisolation(self, x, threshold=0.5):
        eth_counts = x["resolved_ethnicity"].value_counts()
        # print(eth_counts.index[0])
        if not eth_counts.shape[0] > 0:
            return 0
        top_eth = eth_counts.iloc[0]
        pct_isolation = top_eth / eth_counts.sum()
        return pct_isolation > threshold

    def _bgcohesion(self, gp, n):
        count_byschool = gp["assigned school"].value_counts()
        over_thresh = count_byschool[(count_byschool >= n)].sum()
        return over_thresh

    def capacity_unfilled(self, schools, programs, market, priority_weights):
        student_data = self.student_data
        assigned_students = student_data[student_data["programno"] > 0]
        assigned_ge = assigned_students[
            assigned_students["programtype"] == "GE"
        ]
        school_groups = assigned_students.groupby("assigned school")
        school_groups_ge = assigned_ge.groupby("assigned school")
        enrollments = school_groups.size()
        enrollments_ge = school_groups_ge.size()
        programs = programs.program_df
        programs["school_id"] = programs["school_id"].astype(str)
        school_capacities = programs.groupby("school_id")["capacity"].sum()
        programs_ge = programs[programs["program_type"] == "GE"]
        school_capacities_ge = programs_ge.groupby("school_id")[
            "capacity"
        ].sum()
        unfilled = school_capacities - enrollments
        unfilled_ge = school_capacities_ge - enrollments_ge
        designated_students = student_data[student_data["designation"] == 1]
        designated_liveinaa = designated_students.groupby("aa").size()
        designated_toaa = designated_students.groupby("assigned school").size()
        data = {}
        for sch in enrollments.index:
            try:
                data["Assigned<" + sch + ">"] = enrollments.loc[sch]
                data["AssignedGE<" + sch + ">"] = enrollments_ge.loc[sch]
                data["Capacity<" + sch + ">"] = school_capacities.loc[sch]
                data["CapacityGE<" + sch + ">"] = school_capacities_ge.loc[sch]
                data["Unfilled<" + sch + ">"] = unfilled.loc[sch]
                data["UnfilledGE<" + sch + ">"] = unfilled_ge.loc[sch]
                aacode = "[" + str(sch) + "]"
                data["DesignatedLiveIn<" + sch + ">"] = designated_liveinaa.loc[
                    aacode
                ]
                data["DesignatedTo<" + sch + ">"] = designated_toaa.loc[sch]
            except Exception:
                continue
        return pd.Series(data)

    def distance_rank_CDF(self, students, d, k):
        count = 0
        n = students.shape[0]
        for i in range(n):
            if students["assignment_dist"].iloc[i] >= d:
                if (
                    students["rank"].iloc[i] >= k
                    or students["designation"].iloc[i] == 1
                ):
                    count += 1
        if n == 0:
            return 0
        else:
            return count / n

    def eval_assignment_Oct7(self, schools, programs, market, priority_weights):
        student_data = self.student_data

        (
            eth_groups,
            eth_matrix,
            eth_matrix_norm,
            eth_total,
            eth_total_norm,
            populations,
        ) = self._make_ethnic_matrix(student_data)
        assigned_students = student_data[student_data["programno"] > 0]
        ge_students = assigned_students[
            assigned_students["programtype"] == "GE"
        ]
        school_groups = assigned_students.groupby("assigned school")
        program_groups = assigned_students.groupby("assignment")

        # 1. Diversity
        # a) FRL
        frl_schools = school_groups.apply(lambda x: x["frl"].mean())
        average_frl = school_groups.apply(lambda x: x["frl"].mean()).mean()

        frl_schools_10 = 0
        frl_schools_15 = 0
        s = frl_schools.shape[0]
        for i in range(s):
            frl = frl_schools.iloc[i]
            if abs(frl - average_frl) < 0.1:
                frl_schools_10 += 1
            if abs(frl - average_frl) < 0.15:
                frl_schools_15 += 1
        frl_schools_10 = frl_schools_10 / s
        frl_schools_15 = frl_schools_15 / s

        # b) ethnicity
        eth_schools_10 = 0
        eth_schools_15 = 0
        GE_count = 0
        for i in range(eth_matrix_norm.shape[0]):
            program = eth_matrix_norm.index[i]
            code = program[4:6]
            if code == "GE":
                GE_count += 1
                in_threshold_10 = True
                in_threshold_15 = True
                for eth in eth_groups:
                    if eth in eth_matrix_norm.columns:
                        if (
                            abs(
                                eth_matrix_norm[eth].iloc[i]
                                - eth_total_norm.loc[eth]
                            )
                            > 0.1
                        ):
                            in_threshold_10 = False
                        if (
                            abs(
                                eth_matrix_norm[eth].iloc[i]
                                - eth_total_norm.loc[eth]
                            )
                            > 0.15
                        ):
                            in_threshold_15 = False
                if in_threshold_10:
                    eth_schools_10 += 1
                if in_threshold_15:
                    eth_schools_15 += 1

        eth_schools_10 = eth_schools_10 / GE_count
        eth_schools_15 = eth_schools_15 / GE_count

        diversity_metrics = {
            "FRL < +-10%": frl_schools_10,
            "FRL < +-15%": frl_schools_15,
            "Ethn < +-10% (GE)": eth_schools_10,
            "Ethn < +-15% (GE)": eth_schools_15,
        }

        for eth in eth_groups:
            vals = []
            for sch in market.schools.school_ids:
                code = str(sch) + "-GE-KG"
                if (
                    code in eth_matrix_norm.index
                    and eth in eth_matrix_norm.columns
                ):
                    vals.append(eth_matrix_norm[eth].loc[code])
            if len(vals) > 0:
                max_val = max(vals)
                med_val = np.median(vals)
                diversity_metrics["Max " + eth + " (GE)"] = max_val
                diversity_metrics["Med " + eth + " (GE)"] = med_val

        high_frl_schools = []  # schools with above average frl
        frl_60_schools = []
        frl_65_schools = []
        for i in range(frl_schools.shape[0]):
            if frl_schools.iloc[i] > average_frl:
                high_frl_schools.append(frl_schools.index[i])
            if frl_schools.iloc[i] > 0.6:
                frl_60_schools.append(frl_schools.index[i])
            if frl_schools.iloc[i] > 0.65:
                frl_65_schools.append(frl_schools.index[i])

        # AALPI
        aalpi = [
            "Hispanic/Latino",
            "Black or African American",
            "Pacific Islander",
        ]
        aalpi_students = student_data[
            student_data["resolved_ethnicity"].apply(lambda x: x in aalpi)
        ]
        not_aalpi_students = student_data[
            student_data["resolved_ethnicity"].apply(lambda x: x not in aalpi)
        ]
        aalpi_in_high_frl = aalpi_students[
            aalpi_students["assigned school"].apply(
                lambda x: x in high_frl_schools
            )
        ]
        not_aalpi_in_high_frl = not_aalpi_students[
            not_aalpi_students["assigned school"].apply(
                lambda x: x in high_frl_schools
            )
        ]

        aalpi_high_frl_frac = (
            aalpi_in_high_frl.shape[0] / aalpi_students.shape[0]
        )
        not_aalpi_high_frl_frac = (
            not_aalpi_in_high_frl.shape[0] / not_aalpi_students.shape[0]
        )
        diversity_metrics["AALPI in higher FRL sch"] = aalpi_high_frl_frac
        diversity_metrics["non-AALPI in higher FRL sch"] = (
            not_aalpi_high_frl_frac
        )

        # ELL
        ell_students = student_data[
            student_data["englprof_desc"].apply(
                lambda x: x == "N-Non English" or x == "L-Limited English"
            )
        ]
        not_ell_students = student_data[
            student_data["englprof_desc"].apply(
                lambda x: x != "N-Non English" and x != "L-Limited English"
            )
        ]
        ell_in_high_frl = ell_students[
            ell_students["assigned school"].apply(
                lambda x: x in high_frl_schools
            )
        ]
        not_ell_in_high_frl = not_ell_students[
            not_ell_students["assigned school"].apply(
                lambda x: x in high_frl_schools
            )
        ]
        ell_high_frl_frac = ell_in_high_frl.shape[0] / ell_students.shape[0]
        not_ell_high_frl_frac = (
            not_ell_in_high_frl.shape[0] / not_ell_students.shape[0]
        )
        diversity_metrics["ELL in higher FRL sch"] = ell_high_frl_frac
        diversity_metrics["non-ELL in higher FRL sch"] = not_ell_high_frl_frac

        # SpEd
        speced_students = student_data[student_data["speced"] == "Yes"]
        not_speced_students = student_data[student_data["speced"] == "No"]
        speced_in_high_frl = speced_students[
            speced_students["assigned school"].apply(
                lambda x: x in high_frl_schools
            )
        ]
        not_speced_in_high_frl = not_speced_students[
            not_speced_students["assigned school"].apply(
                lambda x: x in high_frl_schools
            )
        ]
        speced_high_frl_frac = (
            speced_in_high_frl.shape[0] / speced_students.shape[0]
        )
        not_speced_high_frl_frac = (
            not_speced_in_high_frl.shape[0] / not_speced_students.shape[0]
        )
        diversity_metrics["SpEd in higher FRL sch"] = speced_high_frl_frac
        diversity_metrics["non-SpEd in higher FRL sch"] = (
            not_speced_high_frl_frac
        )

        for eth in eth_groups:
            eth_students = student_data[
                student_data["resolved_ethnicity"] == eth
            ]
            eth_students_frl_60 = eth_students[
                eth_students["assigned school"].apply(
                    lambda x: x in frl_60_schools
                )
            ]
            eth_students_frl_65 = eth_students[
                eth_students["assigned school"].apply(
                    lambda x: x in frl_65_schools
                )
            ]
            eth_60_frl_frac = (
                eth_students_frl_60.shape[0] / eth_students.shape[0]
            )
            eth_65_frl_frac = (
                eth_students_frl_65.shape[0] / eth_students.shape[0]
            )
            diversity_metrics[">60% FRL " + eth] = eth_60_frl_frac
            diversity_metrics[">65% FRL " + eth] = eth_65_frl_frac

        for eth in eth_groups:
            var = 0
            av = 0
            total = 0
            if eth in eth_matrix.columns:
                for i in range(eth_matrix.shape[0]):
                    num = eth_matrix[eth].iloc[i]
                    if num > 0:
                        total += num
                        av += num * (num - 1)
                        var += num * (num - 1) * (num - 1)
                if total > 0:
                    var = var / total
                    av = av / total
                    std = np.sqrt(var - av * av)
                    diversity_metrics["std peers " + eth] = std

                frac = eth_total_norm[eth]
                diversity_metrics["% of district " + eth] = frac

        diversity_metrics = pd.Series(diversity_metrics)
        # print(diversity_metrics)
        # 2. Proximity
        # a) Distance
        walkzone = (assigned_students["assignment_dist"] < 0.5).mean()
        dist_15 = (assigned_students["assignment_dist"] > 1.5).mean()
        dist_30 = (assigned_students["assignment_dist"] > 3).mean()

        proximity_metrics = {
            "Walkzone": walkzone,
            "Distance > 1.5": dist_15,
            "Distance > 3": dist_30,
        }

        # b) Travel Time
        if hasattr(self, "travel_times"):
            max_GE_time = ge_students[
                "assignment_time"
            ].max()  # Maximum travel time for a student to a GE program
            time_250 = (assigned_students["assignment_time"] > 25).mean()
            avg_time = assigned_students["assignment_time"].mean()

            proximity_metrics["Average Time"] = avg_time
            proximity_metrics["Time > 25"] = time_250
            proximity_metrics["Max GE time"] = max_GE_time

        proximity_metrics = pd.Series(proximity_metrics)

        # 3. Community cohesion
        bgs_school = school_groups["census_blockgroup"].nunique().mean()
        bgcohesion_2 = (
            assigned_students.groupby("census_blockgroup")
            .apply(lambda x: self._bgcohesion(x, 2))
            .sum()
            / assigned_students.shape[0]
        )

        predictability_metrics = pd.Series(
            {
                "BGs per school": bgs_school,
                "BG cohesion (2)": bgcohesion_2,
            }
        )

        # 4. Predictability / Choice

        choice_3 = (assigned_students["rank"] <= 3).mean()
        inzone_choice_1 = (assigned_students["In-Zone Rank"] == 1).mean()
        inzone_choice_3 = (assigned_students["In-Zone Rank"] <= 3).mean()
        designated = student_data["designation"].mean()

        enrollments = program_groups.size()
        tmp = programs.program_df.merge(
            pd.DataFrame(data=enrollments, columns=["assigned_count"]),
            how="left",
            left_on="program_id",
            right_index=True,
        )
        pctfull = tmp["assigned_count"].fillna(0) / tmp["capacity"]
        enrolled_under_80 = (pctfull < 0.8).mean()

        choice_metrics = {
            "Top 3 choice": choice_3,
            "Top in-zone choice": inzone_choice_1,
            "Top 3 in-zone choice": inzone_choice_3,
            "Designated": designated,
            "<80% enrolled": enrolled_under_80,
        }

        for e in range(len(self.eths)):
            subset = assigned_students[
                assigned_students["resolved_ethnicity"] == self.eths[e]
            ]
            eth_choice_3 = (subset["rank"] <= 3).mean()
            choice_metrics["Top 3 choice - " + self.eth_labels[e]] = (
                eth_choice_3
            )

        choice_metrics = pd.Series(choice_metrics)

        # all_metrics = pd.concat([diversity_metrics,  proximity_metrics,
        #    predictability_metrics, enrollment_metrics, choice_metrics])
        all_metrics = pd.concat(
            [
                proximity_metrics,
                predictability_metrics,
                choice_metrics,
                diversity_metrics,
            ]
        )

        return all_metrics

    def updated_eval_assignment(
        self, schools, programs, market, priority_weights
    ):
        student_data = self.student_data
        (
            eth_groups,
            eth_matrix,
            eth_matrix_norm,
            eth_total,
            eth_total_norm,
            populations,
        ) = self._make_ethnic_matrix(student_data)

        # 1. Diversity
        frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) > 0.5
        student_data["high_poverty"] = frl_prob
        frl_students = student_data[frl_prob]
        aalpi = [
            "Hispanic/Latino",
            "Black or African American",
            "Pacific Islander",
        ]
        aalpi_labels = ["Hispanic", "Black", "PI"]
        aalpi_students = student_data["resolved_ethnicity"].apply(
            lambda x: x in aalpi
        )
        student_data["aalpi"] = aalpi_students
        aalpi_students = student_data[student_data["aalpi"]]
        ell_students = student_data["englprof_desc"].apply(
            lambda x: x == "N-Non English" or x == "L-Limited English"
        )
        student_data["ell"] = ell_students
        ell_students = student_data[student_data["ell"]]

        student_data["speced"] = student_data["speced"] == "Yes"
        speced_students = student_data[student_data["speced"]]

        assigned_students = student_data[student_data["programno"] > 0]
        self.assigned_students = assigned_students
        ell_assigned = ell_students[ell_students["programno"] > 0]
        frl_assigned = frl_students[frl_students["programno"] > 0]
        speced_assigned = speced_students[speced_students["programno"] > 0]
        program_groups = assigned_students.groupby("assignment")
        ge_students = assigned_students[
            assigned_students["programtype"] == "GE"
        ]
        ge_groups = ge_students.groupby("assignment")
        school_groups = assigned_students.groupby("assigned school")

        diversity_metrics = {}

        # a) Socioeconomic
        average_frl = program_groups.apply(lambda x: x["frl"].mean()).mean()
        median_frl = program_groups.apply(lambda x: x["frl"].mean()).median()
        max_frl = program_groups.apply(lambda x: x["frl"].mean()).max()
        std_frl = program_groups.apply(lambda x: x["frl"].mean()).std()
        frl_peers_std = school_groups.apply(lambda x: x["frl"].sum()).std()
        diversity_metrics["Average FRL"] = average_frl
        diversity_metrics["Median FRL"] = median_frl
        diversity_metrics["Max FRL"] = max_frl
        diversity_metrics["STD FRL"] = std_frl
        diversity_metrics["STD peers FRL"] = frl_peers_std

        # b) Socioeconomic + Racial/Ethnic
        underserved_concentrated = program_groups.apply(
            lambda x: self._pctunderserved(x, frl=0.67, aalpi=0.5, el=0.2)
        ).mean()
        underserved_concentrated_ge = ge_groups.apply(
            lambda x: self._pctunderserved(x, frl=0.67, aalpi=0.5, el=0.2)
        ).mean()
        diversity_metrics["Underserved concentration"] = (
            underserved_concentrated
        )
        diversity_metrics["Underserved concentration - GE"] = (
            underserved_concentrated_ge
        )

        # c) Racial/Ethnic
        thiel = self.theil(eth_matrix, eth_total, eth_total_norm)
        thiel_ge = self.theil(
            eth_matrix, eth_total, eth_total_norm, GE_only=True
        )
        hellinger = self.hellinger_avg()
        hellinger_ge = self.hellinger_avg(GE_only=True)
        ethiso_45_ge = ge_groups.apply(
            lambda x: self._ethnicisolation(x, 0.45)
        ).mean()
        ethiso_50_ge = ge_groups.apply(
            lambda x: self._ethnicisolation(x, 0.50)
        ).mean()
        ethiso_60_ge = ge_groups.apply(
            lambda x: self._ethnicisolation(x, 0.60)
        ).mean()
        ethiso_60 = program_groups.apply(
            lambda x: self._ethnicisolation(x, 0.60)
        ).mean()
        diversity_metrics["Thiel"] = thiel
        diversity_metrics["Thiel - GE"] = thiel_ge
        diversity_metrics["Hellinger"] = hellinger
        diversity_metrics["Hellinger - GE"] = hellinger_ge
        diversity_metrics["Ethnic isolation 45% - GE"] = ethiso_45_ge
        diversity_metrics["Ethnic isolation 50% - GE"] = ethiso_50_ge
        diversity_metrics["Ethnic isolation 60% - GE"] = ethiso_60_ge
        diversity_metrics["Ethnic isolation 60%"] = ethiso_60
        for e in range(3):
            min_to_self = self.mintoself(assigned_students, aalpi[e])
            diversity_metrics["Min-to-self " + aalpi_labels[e]] = min_to_self
            eth_std = program_groups.apply(
                lambda x: (x["resolved_ethnicity"] == aalpi[e]).mean()
            ).std()
            diversity_metrics["STD " + aalpi_labels[e]] = eth_std
            eth_peers_std = school_groups.apply(
                lambda x: (x["resolved_ethnicity"] == aalpi[e]).sum()
            ).std()
            diversity_metrics["STD peers " + aalpi_labels[e]] = eth_peers_std

        # d) Language
        el_33 = program_groups.apply(lambda x: x["ell"].mean() > 0.33).mean()
        el_50 = program_groups.apply(lambda x: x["ell"].mean() > 0.50).mean()
        diversity_metrics["EL concentration 33%"] = el_33
        diversity_metrics["EL concentration 50%"] = el_50

        # e) Special Education
        speced_20 = program_groups.apply(
            lambda x: x["speced"].mean() > 0.2
        ).mean()
        diversity_metrics["SpecEd concentration 20%"] = speced_20

        diversity_metrics = pd.Series(diversity_metrics)

        # 2. Proximity

        # a) Distance
        avg_dist = assigned_students["assignment_dist"].mean()
        median_dist = assigned_students["assignment_dist"].median()
        dist_15 = (assigned_students["assignment_dist"] > 1.5).mean()
        dist_30 = (assigned_students["assignment_dist"] > 3).mean()
        el_dist_15 = (ell_students["assignment_dist"] > 1.5).mean()
        frl_dist_15 = (frl_students["assignment_dist"] > 1.5).mean()
        speced_dist_15 = (speced_students["assignment_dist"] > 1.5).mean()

        proximity_metrics = {
            "Average distance": avg_dist,
            "Median distance": median_dist,
            "Distance > 1.5": dist_15,
            "Distance > 3": dist_30,
            "Distance > 1.5 - EL": el_dist_15,
            "Distance > 1.5 - FRL": frl_dist_15,
            "Distance > 1.5 - SpecEd": speced_dist_15,
        }

        for e in range(len(self.eths)):
            subset = assigned_students[
                assigned_students["resolved_ethnicity"] == self.eths[e]
            ]
            eth_dist_15 = (subset["assignment_dist"] > 1.5).mean()
            proximity_metrics["Distance > 1.5 - " + self.eth_labels[e]] = (
                eth_dist_15
            )

        proximity_metrics = pd.Series(proximity_metrics)

        # b) Travel Time
        if hasattr(self, "travel_times"):
            max_GE_time = ge_students[
                "assignment_time"
            ].max()  # Maximum travel time for a student to a GE program
            time_250 = (assigned_students["assignment_time"] > 25).mean()

            proximity_metrics["Max GE time"] = max_GE_time
            proximity_metrics["Time > 25"] = time_250

        # 3. Predictability

        # a) Range of outcomes

        # b) Community cohesion
        bgs_school = school_groups["census_blockgroup"].nunique().mean()
        bgcohesion_2 = (
            assigned_students.groupby("census_blockgroup")
            .apply(lambda x: self._bgcohesion(x, 2))
            .sum()
            / assigned_students.shape[0]
        )
        bgcohesion_3 = (
            assigned_students.groupby("census_blockgroup")
            .apply(lambda x: self._bgcohesion(x, 3))
            .sum()
            / assigned_students.shape[0]
        )
        bgcohesion_ct = assigned_students.apply(
            lambda x: self._cohesionct(x), axis=1
        )
        median_bgc = bgcohesion_ct.median()

        predictability_metrics = pd.Series(
            {
                "BGs per school": bgs_school,
                "BG cohesion (2)": bgcohesion_2,
                "BG cohesion (3)": bgcohesion_3,
                "Median BG cohesion": median_bgc,
            }
        )

        # 4. Enrollment

        # a) Percent of capacity
        enrollments = program_groups.size()
        tmp = programs.program_df.merge(
            pd.DataFrame(data=enrollments, columns=["assigned_count"]),
            how="left",
            left_on="program_id",
            right_index=True,
        )
        pctfull = tmp["assigned_count"].fillna(0) / tmp["capacity"]
        enrolled_80 = (pctfull > 0.8).mean()
        enrolled_90 = (pctfull > 0.9).mean()
        enrolled_95 = (pctfull > 0.95).mean()
        lowest_enrollment = pctfull.min()
        lowest_enrollment_code = pctfull.idxmin()

        enrollment_metrics = {
            "Above 80% capacity": enrolled_80,
            "Above 90% capacity": enrolled_90,
            "Above 95% capacity": enrolled_95,
            "Lowest % of capacity": lowest_enrollment,
            "Lowest % of capacity code": lowest_enrollment_code,
        }

        # b) School KG enrollments
        target_schools = {
            "Carver": 625,
            "Drew": 507,
            "El Dorado": 521,
            "Malcolm X": 830,
            "Muir": 650,
            "Spring Valley Science": 834,
            "Visitacion": 867,
            "Parker": 638,
            "Harte": 453,
            "Cobb": 525,
        }
        for sch in target_schools:
            sch_assn = (
                assigned_students["assigned school"] == str(target_schools[sch])
            ).sum()
            enrollment_metrics[sch + " enrollment"] = sch_assn

        enrollment_metrics = pd.Series(enrollment_metrics)

        # 5. Choice
        choice_1 = (assigned_students["rank"] == 1).mean()
        choice_3 = (assigned_students["rank"] <= 3).mean()
        inzone_choice_1 = (assigned_students["In-Zone Rank"] == 1).mean()
        inzone_choice_3 = (assigned_students["In-Zone Rank"] <= 3).mean()
        designated = student_data["designation"].mean()
        unassigned = (student_data["programno"] == 0).mean()

        choice_metrics = {
            "Top choice": choice_1,
            "Top 3 choice": choice_3,
            "Top in-zone choice": inzone_choice_1,
            "Top 3 in-zone choice": inzone_choice_3,
            "Designated": designated,
            "Unassigned": unassigned,
        }

        for e in range(len(self.eths)):
            subset = assigned_students[
                assigned_students["resolved_ethnicity"] == self.eths[e]
            ]
            eth_choice_1 = (subset["rank"] == 1).mean()
            eth_choice_3 = (subset["rank"] <= 3).mean()
            eth_iz_choice_1 = (subset["In-Zone Rank"] == 1).mean()
            eth_iz_choice_3 = (subset["In-Zone Rank"] <= 3).mean()
            subset = student_data[
                student_data["resolved_ethnicity"] == self.eths[e]
            ]
            eth_designated = subset["designation"].mean()
            eth_unassigned = (subset["programno"] == 0).mean()
            choice_metrics["Top choice - " + self.eth_labels[e]] = eth_choice_1
            choice_metrics["Top 3 choice - " + self.eth_labels[e]] = (
                eth_choice_3
            )
            choice_metrics["Top in-zone choice - " + self.eth_labels[e]] = (
                eth_iz_choice_1
            )
            choice_metrics["Top 3 in-zone choice - " + self.eth_labels[e]] = (
                eth_iz_choice_3
            )
            choice_metrics["Designated - " + self.eth_labels[e]] = (
                eth_designated
            )
            choice_metrics["Unassigned - " + self.eth_labels[e]] = (
                eth_unassigned
            )

        el_choice_1 = (ell_assigned["rank"] == 1).mean()
        el_choice_3 = (ell_assigned["rank"] <= 3).mean()
        el_iz_choice_1 = (ell_assigned["In-Zone Rank"] == 1).mean()
        el_iz_choice_3 = (ell_assigned["In-Zone Rank"] <= 3).mean()
        el_designated = ell_students["designation"].mean()
        el_unassigned = (ell_students["programno"] == 0).mean()
        choice_metrics["Top choice - EL"] = el_choice_1
        choice_metrics["Top 3 choice - EL"] = el_choice_3
        choice_metrics["Top in-zone choice - EL"] = el_iz_choice_1
        choice_metrics["Top 3 in-zone choice - EL"] = el_iz_choice_3
        choice_metrics["Designated - EL"] = el_designated
        choice_metrics["Unassigned - EL"] = el_unassigned
        frl_choice_1 = (frl_assigned["rank"] == 1).mean()
        frl_choice_3 = (frl_assigned["rank"] <= 3).mean()
        frl_iz_choice_1 = (frl_assigned["In-Zone Rank"] == 1).mean()
        frl_iz_choice_3 = (frl_assigned["In-Zone Rank"] <= 3).mean()
        frl_designated = frl_students["designation"].mean()
        frl_unassigned = (frl_students["programno"] == 0).mean()
        choice_metrics["Top choice - FRL"] = frl_choice_1
        choice_metrics["Top 3 choice - FRL"] = frl_choice_3
        choice_metrics["Top in-zone choice - FRL"] = frl_iz_choice_1
        choice_metrics["Top 3 in-zone choice - FRL"] = frl_iz_choice_3
        choice_metrics["Designated - FRL"] = frl_designated
        choice_metrics["Unassigned - FRL"] = frl_unassigned
        speced_choice_1 = (speced_assigned["rank"] == 1).mean()
        speced_choice_3 = (speced_assigned["rank"] <= 3).mean()
        speced_iz_choice_1 = (speced_assigned["In-Zone Rank"] == 1).mean()
        speced_iz_choice_3 = (speced_assigned["In-Zone Rank"] <= 3).mean()
        speced_designated = speced_students["designation"].mean()
        speced_unassigned = (speced_students["programno"] == 0).mean()
        choice_metrics["Top choice - SpecEd"] = speced_choice_1
        choice_metrics["Top 3 choice - SpecEd"] = speced_choice_3
        choice_metrics["Top in-zone choice - SpecEd"] = speced_iz_choice_1
        choice_metrics["Top 3 in-zone choice - SpecEd"] = speced_iz_choice_3
        choice_metrics["Designated - SpecEd"] = speced_designated
        choice_metrics["Unassigned - SpecEd"] = speced_unassigned

        aa_groups = assigned_students.dropna(subset=["aa"])
        aa_groups = aa_groups.groupby("aa")
        for aa in aa_groups:
            aa_name = aa[0]
            if len(aa_name) != 5:
                continue
            aa_name = aa_name[1:-1]
            for k in range(1, 4):
                aa_topk = (aa[1]["rank"] <= k).mean()
                choice_metrics["Top " + str(k) + " choice - " + aa_name] = (
                    aa_topk
                )

        choice_metrics = pd.Series(choice_metrics)

        all_metrics = pd.concat(
            [
                diversity_metrics,
                proximity_metrics,
                predictability_metrics,
                enrollment_metrics,
                choice_metrics,
            ]
        )

        if "assigned_utility_x" not in student_data.columns:
            # print('XYXYXY')
            return all_metrics

        # . Utility
        utility_metrics = {}

        average_utility = student_data["assigned_utility_x"].mean()
        utility_metrics["Average utility"] = average_utility

        for e in range(len(self.eths)):
            subset = student_data[
                student_data["resolved_ethnicity"] == self.eths[e]
            ]
            eth_utility = subset["assigned_utility_x"].mean()
            utility_metrics["Average utility - " + self.eth_labels[e]] = (
                eth_utility
            )

        aa_utility_min = aa_groups["assigned_utility_x"].min()
        utility_metrics["Min average utility"] = aa_utility_min

        utility_metrics = pd.Series(utility_metrics)

        all_metrics = pd.concat([all_metrics, utility_metrics])
        # all_metrics.to_csv('~/Desktop/test.csv')
        return all_metrics

    def lightweight_eval_assignment(
        self, schools, programs, market, priority_weights
    ):
        student_data = self.student_data

        aalpi = [
            "Hispanic/Latino",
            "Black or African American",
            "Pacific Islander",
        ]

        aalpi_students = student_data["resolved_ethnicity"].apply(
            lambda x: x in aalpi
        )
        student_data["aalpi"] = aalpi_students
        frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        )
        student_data["frlprob"] = frl_prob > 0.5

        has_aa = student_data.dropna(subset=["aa"])
        has_aa["isunassigned"] = has_aa["programno"] == 0
        by_aa = has_aa.groupby("aa")["isunassigned"].sum()
        most_unassigned = by_aa.max()
        most_unassigned_aa = by_aa.idxmax()
        by_aa = has_aa.groupby("aa")["isunassigned"].mean()
        by_aa.max()
        by_aa.idxmax()

        assigned_students = student_data[student_data["programno"] > 0]
        (
            ethnic_groups,
            ethnic_matrix,
            ethnic_matrix_norm,
            ethnic_total,
            ethnic_total_norm,
            populations,
        ) = self._make_ethnic_matrix(student_data)
        self._get_quality_schools(
            schools, programs, "math_scores_1819", pct=0.33
        )
        # print('quality_GEprog_idx_math',quality_GEprog_idx_math)
        self._colors_quality_schools(schools, programs, pct=0.33)
        eval_metrics = pd.Series(
            {
                "Avg Dist": assigned_students["assignment_dist"].mean(),
                "Dist > 3": (assigned_students["assignment_dist"] > 3).sum()
                / assigned_students.shape[0],
                "Dist < 0.5": (assigned_students["assignment_dist"] < 0.5).sum()
                / assigned_students.shape[0],
                "Dist > 1.5": (assigned_students["assignment_dist"] > 1.5).sum()
                / assigned_students.shape[0],
                "Avg In-Zone Rank": (assigned_students["In-Zone Rank"]).mean(),
                "Avg Rank": (assigned_students["rank"]).mean(),
                "In-Zone Top 3": (
                    assigned_students["In-Zone Rank"] <= 3
                ).mean(),
                "Rank Top 3": (assigned_students["rank"] <= 3).mean(),
                "In-Zone Top 1": (
                    assigned_students["In-Zone Rank"] == 1
                ).mean(),
                "Rank Top 1": (assigned_students["rank"] == 1).mean(),
                "Worst avg rank": (
                    assigned_students.groupby("idschoolattendance")[
                        "rank"
                    ].mean()
                ).max(),
                "Max School FRL": self.max_frl(),
                "Median School FRL": self.median_frl(),
                "School w/in 10% district FRL": self.school_frl_range_district(
                    0.10
                ),
                "School w/in 15% district FRL": self.school_frl_range_district(
                    0.15
                ),
                "Unassigned": (student_data["programno"] == 0).mean(),
                "Max unassigned ct": most_unassigned,
                "Max unassigned ct aa": most_unassigned_aa,
                #'Max unassigned pct': most_unassigned_pct,
                #'Max unassigned pct aa': most_unassigned_pct_aa,
                "Thiel": self.theil(
                    ethnic_matrix, ethnic_total, ethnic_total_norm
                ),
                "GE-only Thiel": self.theil(
                    ethnic_matrix, ethnic_total, ethnic_total_norm, GE_only=True
                ),
                "Avg Hellinger": self.hellinger_avg(),
                "GE-only Avg Hellinger": self.hellinger_avg(GE_only=True),
                #'Min Quality Access':self.likelihood_of_quality_school(
                #    quality_GEprog_idx_math,market,priority_weights),
                #'Min Quality Color': self.likelihood_of_quality_school(
                #    quality_GEprog_idx_color,market,priority_weights),
                "Designated": student_data["designation"].mean(),
                "BG isolation (2)": self.without_blockgroup(threshold=2),
                "BG isolation (3)": self.without_blockgroup(threshold=3),
            }
        )
        eval_metrics = pd.concat(
            [
                eval_metrics,
                pd.Series(
                    self.rank_cdf(student_data, "In-Zone Rank"),
                    index=[f"In-Zone Rank <= {rank}" for rank in range(1, 6)],
                ),
            ]
        )
        eval_metrics = pd.concat(
            [
                eval_metrics,
                pd.Series(
                    self.rank_cdf(
                        student_data, "In-Zone Rank", rank_clusters=1
                    ),
                    index=["In-Zone First Choice"],
                ),
            ]
        )
        student_data[student_data["programtype"] == "GE"]
        # eval_metrics = pd.concat([eval_metrics, pd.Series(
        #    self.rank_cdf(ge_data, 'In-Zone Rank'),
        #    index= ['In-Zone Rank (GE) <= {}'.format(rank) for rank in range(1,6)])])
        # eval_metrics = pd.concat([eval_metrics, pd.Series(self.rank_cdf(ge_data, 'rank'),
        #   index= ['Rank (GE) <= {}'.format(rank) for rank in range(1,6)])])
        # eval_metrics = pd.concat([eval_metrics, pd.Series(self.rank_cdf(student_data, 'rank'),
        #    index= ['Rank <= {}'.format(rank) for rank in range(1,6)])])

        assigned_ge = student_data[student_data["programtype"] == "GE"]
        student_data[student_data["programtype"] != "GE"]

        # choice_metrics = self.subset_choice(student_data)
        quality_bypop = self.subset_quality(student_data, schools.school_data)
        live_nearby = self.live_nearby(assigned_students)
        live_nearby_ge = self.live_nearby(assigned_ge)
        live_nearby_ge.index = [x + " - GE only" for x in live_nearby_ge.index]

        nschools = student_data["assigned school"].nunique()
        aalpi_iso_50 = (
            self.group_isolation(student_data, "aalpi", 0.5) / nschools
        )
        aalpi_iso_60 = (
            self.group_isolation(student_data, "aalpi", 0.6) / nschools
        )
        aalpi_iso_70 = (
            self.group_isolation(student_data, "aalpi", 0.7) / nschools
        )
        aalpi_iso_50_ge = (
            self.group_isolation(assigned_ge, "aalpi", 0.5) / nschools
        )
        aalpi_iso_60_ge = (
            self.group_isolation(assigned_ge, "aalpi", 0.6) / nschools
        )
        aalpi_iso_70_ge = (
            self.group_isolation(assigned_ge, "aalpi", 0.7) / nschools
        )
        frl_iso_50 = (
            self.group_isolation(student_data, "frlprob", 0.5) / nschools
        )
        frl_iso_60 = (
            self.group_isolation(student_data, "frlprob", 0.6) / nschools
        )
        frl_iso_70 = (
            self.group_isolation(student_data, "frlprob", 0.7) / nschools
        )
        frl_iso_50_ge = (
            self.group_isolation(assigned_ge, "frlprob", 0.5) / nschools
        )
        frl_iso_60_ge = (
            self.group_isolation(assigned_ge, "frlprob", 0.6) / nschools
        )
        frl_iso_70_ge = (
            self.group_isolation(assigned_ge, "frlprob", 0.7) / nschools
        )
        group_isolation = pd.Series(
            {
                "aalpi iso 50": aalpi_iso_50,
                "aalpi iso 60": aalpi_iso_60,
                "aalpi iso 70": aalpi_iso_70,
                "aalpi iso 50 - GE": aalpi_iso_50_ge,
                "aalpi iso 60 - GE": aalpi_iso_60_ge,
                "aalpi iso 70 - GE": aalpi_iso_70_ge,
                "frl iso 50": frl_iso_50,
                "frl iso 60": frl_iso_60,
                "frl iso 70": frl_iso_70,
                "frl iso 50 - GE": frl_iso_50_ge,
                "frl iso 60 - GE": frl_iso_60_ge,
                "frl iso 70 - GE": frl_iso_70_ge,
            }
        )

        """
        min_to_self_hisp = self.mintoself(assigned_students, 'Hispanic/Latino')
        min_to_self_black = self.mintoself(assigned_students, 'Black or African American')
        min_to_self_hisp_ge = self.mintoself(assigned_ge, 'Hispanic/Latino')
        min_to_self_black_ge = self.mintoself(assigned_ge, 'Black or African American')
        min_to_self = pd.Series({
            'min-to-self Hisp.': min_to_self_hisp,
            'min-to-self Black': min_to_self_black,
            'min-to-self Hisp.': min_to_self_hisp_ge,
            'min-to-self Black': min_to_self_black_ge
        })

        assigned_ge = assigned_ge['assignment'].value_counts()
        assigned_lp = assigned_lp['assignment'].value_counts()
        assigned_ge = pd.DataFrame(assigned_ge).merge(
                programs.program_df,how='left',left_index=True,right_on='program_id')


        """
        """
        ge_diffs = assigned_ge['capacity']-assigned_ge['assignment']
        assigned_lp = pd.DataFrame(assigned_lp).merge(
                programs.program_df,how='left',left_index=True,right_on='program_id')
        lp_diffs = assigned_lp['capacity']-assigned_lp['assignment']
        ge_max_diffs = ge_diffs.nlargest(3).to_frame()
        ge_max_diffs['program'] = ge_max_diffs.index
        ge_max_diffs = ge_max_diffs.values
        lp_max_diffs = lp_diffs.nlargest(3).to_frame()
        lp_max_diffs['program'] = lp_max_diffs.index
        lp_max_diffs = lp_max_diffs.values
        capacity_diffs = pd.Series({
            'Unfilled GE 1 ct': ge_max_diffs[0][0],
            'Unfilled GE 1 prog': ge_max_diffs[0][1],
            'Unfilled GE 2 ct': ge_max_diffs[1][0],
            'Unfilled GE 2 prog': ge_max_diffs[1][1],
            'Unfilled GE 3 ct': ge_max_diffs[2][0],
            'Unfilled GE 3 prog': ge_max_diffs[2][1],
            'Unfilled LP 1 ct': lp_max_diffs[0][0],
            'Unfilled LP 1 prog': lp_max_diffs[0][1],
            'Unfilled LP 2 ct': lp_max_diffs[1][0],
            'Unfilled LP 2 prog': lp_max_diffs[1][1],
            'Unfilled LP 3 ct': lp_max_diffs[2][0],
            'Unfilled LP 3 prog': lp_max_diffs[2][1]
        })
        """

        self.assigned_students = assigned_students
        # ell_assigned = ell_students[ell_students['programno'] > 0]
        # frl_assigned = frl_students[frl_students['programno'] > 0]
        # speced_assigned = speced_students[speced_students['programno'] > 0]
        program_groups = assigned_students.groupby("assignment")
        ge_students = assigned_students[
            assigned_students["programtype"] == "GE"
        ]
        ge_groups = ge_students.groupby("assignment")
        school_groups = assigned_students.groupby("assigned school")

        aalpi_labels = ["Hispanic", "Black", "PI"]
        diversity_metrics = {}

        # a) Socioeconomic
        average_frl = program_groups.apply(lambda x: x["frl"].mean()).mean()
        median_frl = program_groups.apply(lambda x: x["frl"].mean()).median()
        max_frl = program_groups.apply(lambda x: x["frl"].mean()).max()
        std_frl = program_groups.apply(lambda x: x["frl"].mean()).std()
        frl_peers_std = school_groups.apply(lambda x: x["frl"].sum()).std()
        diversity_metrics["Average FRL"] = average_frl
        diversity_metrics["Median FRL"] = median_frl
        diversity_metrics["Max FRL"] = max_frl
        diversity_metrics["STD FRL"] = std_frl
        diversity_metrics["STD peers FRL"] = frl_peers_std

        # b) Socioeconomic + Racial/Ethnic
        # underserved_concentrated = program_groups.apply(lambda x:
        #        self._pctunderserved(x, frl=0.67, aalpi=0.5, el=0.2)).mean()
        # underserved_concentrated_ge = ge_groups.apply(lambda x:
        #        self._pctunderserved(x, frl=0.67, aalpi=0.5, el=0.2)).mean()
        # diversity_metrics['Underserved concentration'] = underserved_concentrated
        # diversity_metrics['Underserved concentration - GE'] = underserved_concentrated_ge

        # c) Racial/Ethnic
        # thiel = self.theil(eth_matrix, eth_total, eth_total_norm)
        # thiel_ge = self.theil(eth_matrix, eth_total, eth_total_norm, GE_only=True)
        # hellinger = self.hellinger_avg()
        # hellinger_ge = self.hellinger_avg(GE_only=True)
        # ethiso_45_ge = ge_groups.apply(lambda x: self._ethnicisolation(x, 0.45)).mean()
        # ethiso_50_ge = ge_groups.apply(lambda x: self._ethnicisolation(x, 0.50)).mean()
        # ethiso_60_ge = ge_groups.apply(lambda x: self._ethnicisolation(x, 0.60)).mean()
        # ethiso_60 = program_groups.apply(lambda x: self._ethnicisolation(x, 0.60)).mean()
        # diversity_metrics['Thiel'] = thiel
        # diversity_metrics['Thiel - GE'] = thiel_ge
        # diversity_metrics['Hellinger'] = hellinger
        # diversity_metrics['Hellinger - GE'] = hellinger_ge
        # diversity_metrics['Ethnic isolation 45% - GE'] = ethiso_45_ge
        # diversity_metrics['Ethnic isolation 50% - GE'] = ethiso_50_ge
        # diversity_metrics['Ethnic isolation 60% - GE'] = ethiso_60_ge
        # diversity_metrics['Ethnic isolation 60%'] = ethiso_60

        for e in range(len(self.eths)):
            if e % 2 != 0:
                eth_peers_std = school_groups.apply(
                    lambda x: (
                        x["resolved_ethnicity"] == self.eth_labels[e]
                    ).sum()
                ).std()
                diversity_metrics["STD peers " + self.eth_labels[e]] = (
                    eth_peers_std
                )

        for e in range(3):
            # min_to_self = self.mintoself(assigned_students, aalpi[e])
            # diversity_metrics['Min-to-self ' + aalpi_labels[e]] = min_to_self
            eth_std = program_groups.apply(
                lambda x: (x["resolved_ethnicity"] == aalpi[e]).mean()
            ).std()
            diversity_metrics["STD " + aalpi_labels[e]] = eth_std
            eth_peers_std = school_groups.apply(
                lambda x: (x["resolved_ethnicity"] == aalpi[e]).sum()
            ).std()
            diversity_metrics["STD peers " + aalpi[e]] = eth_peers_std

        # for e in range(len(self.eths)):
        #    eth_peers_sum = school_groups.apply(lambda x:
        #            (x['resolved_ethnicity'] == self.eth_labels[e]).sum()).mean()
        #    diversity_metrics['Mean  peers ' + self.eth_labels[e]] = eth_peers_sum

        # d) Language
        # el_33 = program_groups.apply(lambda x: x['ell'].mean() > 0.33).mean()
        # el_50 = program_groups.apply(lambda x: x['ell'].mean() > 0.50).mean()
        # diversity_metrics['EL concentration 33%'] = el_33
        # diversity_metrics['EL concentration 50%'] = el_50

        # e) Special Education
        # speced_20 = program_groups.apply(lambda x: x['speced'].mean() > 0.2).mean()
        # diversity_metrics['SpecEd concentration 20%'] = speced_20

        ethiso_45_ge = ge_groups.apply(
            lambda x: self._ethnicisolation(x, 0.45)
        ).mean()
        ethiso_50_ge = ge_groups.apply(
            lambda x: self._ethnicisolation(x, 0.50)
        ).mean()
        ethiso_60 = program_groups.apply(
            lambda x: self._ethnicisolation(x, 0.60)
        ).mean()

        diversity_metrics["Ethnic isolation 45% - GE"] = ethiso_45_ge
        diversity_metrics["Ethnic isolation 50% - GE"] = ethiso_50_ge
        diversity_metrics["Ethnic isolation 60%"] = ethiso_60

        bgs_school = school_groups["census_blockgroup"].nunique().mean()
        bgcohesion_2 = (
            assigned_students.groupby("census_blockgroup")
            .apply(lambda x: self._bgcohesion(x, 2))
            .sum()
            / assigned_students.shape[0]
        )
        bgcohesion_3 = (
            assigned_students.groupby("census_blockgroup")
            .apply(lambda x: self._bgcohesion(x, 3))
            .sum()
            / assigned_students.shape[0]
        )
        bgcohesion_ct = assigned_students.apply(
            lambda x: self._cohesionct(x), axis=1
        )
        median_bgc = bgcohesion_ct.median()
        averaege_bgc = bgcohesion_ct.median()

        predictability_metrics = pd.Series(
            {
                "BGs per school": bgs_school,
                "BG cohesion (2)": bgcohesion_2,
                "BG cohesion (3)": bgcohesion_3,
                "Median BG cohesion": median_bgc,
                "Average BG cohesion": averaege_bgc,
            }
        )

        diversity_metrics = pd.Series(diversity_metrics)

        peer_counts = self.peers_same_ethnicity()
        eval_metrics = pd.concat([eval_metrics, peer_counts])
        # eval_metrics = pd.concat([eval_metrics, capacity_diffs])
        eval_metrics = pd.concat([eval_metrics, quality_bypop])
        eval_metrics = pd.concat(
            [eval_metrics, live_nearby, live_nearby_ge]
        )  # min_to_self])
        eval_metrics = pd.concat([eval_metrics, group_isolation])

        eval_metrics = pd.concat(
            [eval_metrics, diversity_metrics, predictability_metrics]
        )

        for threshold in [3, 5, 7]:
            eval_metrics = pd.concat(
                [eval_metrics, self.race_isolation(threshold=threshold)]
            )

        choice_1 = (assigned_students["rank"] == 1).mean()
        choice_3 = (assigned_students["rank"] == 3).mean()
        inzone_choice_1 = (assigned_students["In-Zone Rank"] == 1).mean()
        inzone_choice_3 = (assigned_students["In-Zone Rank"] <= 3).mean()
        designated = student_data["designation"].mean()
        unassigned = (student_data["programno"] == 0).mean()

        choice_metrics = {
            "Top choice": choice_1,
            "Top 3 choice": choice_3,
            "Top in-zone choice": inzone_choice_1,
            "Top 3 in-zone choice": inzone_choice_3,
            "Designated": designated,
            "Unassigned": unassigned,
        }

        for e in range(len(self.eths)):
            subset = assigned_students[
                assigned_students["resolved_ethnicity"] == self.eths[e]
            ]
            eth_choice_1 = (subset["rank"] == 1).mean()
            eth_choice_3 = (subset["rank"] <= 3).mean()
            eth_iz_choice_1 = (subset["In-Zone Rank"] == 1).mean()
            eth_iz_choice_3 = (subset["In-Zone Rank"] <= 3).mean()
            subset = student_data[
                student_data["resolved_ethnicity"] == self.eths[e]
            ]
            eth_designated = subset["designation"].mean()
            eth_unassigned = (subset["programno"] == 0).mean()
            choice_metrics["Top choice - " + self.eth_labels[e]] = eth_choice_1
            choice_metrics["Top 3 choice - " + self.eth_labels[e]] = (
                eth_choice_3
            )
            choice_metrics["Top in-zone choice - " + self.eth_labels[e]] = (
                eth_iz_choice_1
            )
            choice_metrics["Top 3 in-zone choice - " + self.eth_labels[e]] = (
                eth_iz_choice_3
            )
            choice_metrics["Designated - " + self.eth_labels[e]] = (
                eth_designated
            )
            choice_metrics["Unassigned - " + self.eth_labels[e]] = (
                eth_unassigned
            )

        choice_metrics = pd.Series(choice_metrics)

        eval_metrics = pd.concat([eval_metrics, choice_metrics])

        """
        if 'program_cutoff' in student_data.columns:
            quality_GEprog_idx = self._get_quality_schools(schools,programs,'math_scores_1819',pct=0.33)
            quality_math_33 = self.likelihood_of_quality_school(quality_GEprog_idx, market,priority_weights)
            quality_GEprog_idx = self._get_quality_schools(schools,programs,'math_scores_1819',pct=0.50)
            quality_math_50 = self.likelihood_of_quality_school(quality_GEprog_idx, market,priority_weights)
            quality_GEprog_idx = self._colors_quality_schools(schools,programs,pct=0.33)
            quality_color_33 = self.likelihood_of_quality_school(quality_GEprog_idx, market,priority_weights)
            quality_GEprog_idx = self._colors_quality_schools(schools,programs,pct=0.50)
            quality_color_50 = self.likelihood_of_quality_school(quality_GEprog_idx, market,priority_weights)

            cuttoff_metrics = pd.Series({'Min Quality Math 33%':quality_math_33,
                                    'Min Quality Math 50%':quality_math_50,
                                    'Min Quality Colors 33%':quality_color_33,
                                    'Min Quality Colors 50%':quality_color_50})


        eval_metrics = pd.concat([eval_metrics, cuttoff_metrics])
        """
        # distance by race
        proximity_metrics = {}
        for e in range(len(self.eths)):
            subset = assigned_students[
                assigned_students["resolved_ethnicity"] == self.eths[e]
            ]
            eth_dist_15 = (subset["assignment_dist"] > 1.5).mean()
            eth_dist_05 = (subset["assignment_dist"] < 0.5).mean()
            eth_avg_dist = subset["assignment_dist"].mean()
            proximity_metrics["Distance > 1.5 - " + self.eth_labels[e]] = (
                eth_dist_15
            )
            proximity_metrics["Distance < 0.5 - " + self.eth_labels[e]] = (
                eth_dist_05
            )
            proximity_metrics["Avg distance - " + self.eth_labels[e]] = (
                eth_avg_dist
            )
        proximity_metrics = pd.Series(proximity_metrics)

        # percent of race at poverty school
        pov_sch_metrics = self.poverty_schools_by_race()
        eval_metrics = pd.concat(
            [eval_metrics, proximity_metrics, pov_sch_metrics]
        )

        if "assigned_utility_x" not in student_data.columns:
            # print('XYXYXY')
            return eval_metrics

        # . Utility
        utility_metrics = {}

        average_utility = student_data["assigned_utility_x"].mean()
        utility_metrics["Average utility"] = average_utility

        for e in range(len(self.eths)):
            subset = student_data[
                student_data["resolved_ethnicity"] == self.eths[e]
            ]
            eth_utility = subset["assigned_utility_x"].mean()
            utility_metrics["Average utility - " + self.eth_labels[e]] = (
                eth_utility
            )

        # aa_utility_min = aa_groups['averaassigned_utility_xge_utility'].min()
        # utility_metrics['Min average utility'] = aa_utility_min

        utility_metrics = pd.Series(utility_metrics)

        eval_metrics = pd.concat([eval_metrics, utility_metrics])
        return eval_metrics

    def _cohesionct(self, x):
        assignment = x["assignment"]
        bg = x["census_blockgroup"]
        students = self.assigned_students
        matches = students[students["assignment"] == assignment]
        matches = matches[matches["census_blockgroup"] == bg]
        return matches.shape[0] - 1

    def group_isolation(self, student_data, col, threshold):
        school_groups = student_data.groupby("assigned school")
        group_pct = school_groups[col].mean()
        return (group_pct > threshold).sum()

    def group_isolation_absolute(self, student_data, threshold):
        counts = np.zeros([1000])
        for i in range(student_data.shape[0]):
            if isinstance(
                student_data["assigned school"].iloc[i], int
            ) or isinstance(student_data["assigned school"].iloc[i], str):
                school_num = int(student_data["assigned school"].iloc[i])
                print(school_num)
                counts[school_num] += 1
        num_schools_isolated = 0
        for i in range(len(counts)):
            if counts[i] > 0 and counts[i] <= threshold:
                num_schools_isolated += 1
        return num_schools_isolated

    def mintoself(self, student_data, ethnicity):
        school_groups = student_data.groupby("assigned school")
        total = 0
        X = (student_data["resolved_ethnicity"] == ethnicity).sum()
        for group in school_groups:
            xi = (group[1]["resolved_ethnicity"] == ethnicity).sum()
            ti = group[1].shape[0]
            total += (xi / X) * (xi / ti)
        return total

    def live_nearby(self, student_data):
        school_groups = student_data.groupby("assigned school")
        min_06 = 1
        min_1 = 1
        for gp in school_groups:
            if gp[0] == 0:
                continue
            nearby_06 = (gp[1]["assignment_dist"] < 0.6).mean()
            nearby_1 = (gp[1]["assignment_dist"] < 1).mean()
            if nearby_06 < min_06:
                min_06 = nearby_06
            if nearby_1 < min_1:
                min_1 = nearby_1
        mins = {"min within 0.6mi": min_06, "min within 1mi": min_1}
        return pd.Series(mins)

    def subset_quality(self, student_data, school_data):
        colormap = {
            "Red": 0,
            "Orange": 1,
            "Yellow": 2,
            "Green": 3,
            "Blue": 4,
            "None": 0,
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }
        quality_metrics = {}
        met33t = school_data["MetStandards"].quantile(0.666)
        met33 = school_data[school_data["MetStandards"] > met33t]
        met33 = list(met33.index)
        met50t = school_data["MetStandards"].quantile(0.5)
        met50 = school_data[school_data["MetStandards"] > met50t]
        met50 = list(met50.index)
        school_data["ela_color"] = school_data["ela_color"].apply(
            lambda x: colormap[x]
        )
        school_data["math_color"] = school_data["math_color"].apply(
            lambda x: colormap[x]
        )
        school_data["chronic_color"] = school_data["chronic_color"].apply(
            lambda x: colormap[x]
        )
        school_data["suspension_color"] = school_data["suspension_color"].apply(
            lambda x: colormap[x]
        )
        school_data["color_total"] = (
            school_data["ela_color"]
            + school_data["math_color"]
            + school_data["chronic_color"]
            + school_data["suspension_color"]
        )
        color33t = school_data["color_total"].quantile(0.666)
        color33 = school_data[school_data["color_total"] > color33t]
        color33 = list(color33.index)
        color50t = school_data["color_total"].quantile(0.5)
        color50 = school_data[school_data["color_total"] > color50t]
        color50 = list(color50.index)
        math33t = school_data["math_scores_1819"].quantile(0.666)
        math33 = school_data[school_data["math_scores_1819"] > math33t]
        math33 = list(math33.index)
        math50t = school_data["math_scores_1819"].quantile(0.50)
        math50 = school_data[school_data["math_scores_1819"] > math50t]
        math50 = list(math50.index)

        student_data["assigned school"] = student_data[
            "assigned school"
        ].fillna(0)

        def makemetrics(subset, name):
            met33subset = (
                subset["assigned school"]
                .apply(lambda x: int(x) in met33)
                .mean()
            )
            met50subset = (
                subset["assigned school"]
                .apply(lambda x: int(x) in met50)
                .mean()
            )
            color33subset = (
                subset["assigned school"]
                .apply(lambda x: int(x) in color33)
                .mean()
            )
            color50subset = (
                subset["assigned school"]
                .apply(lambda x: int(x) in color50)
                .mean()
            )
            math33subset = (
                subset["assigned school"]
                .apply(lambda x: int(x) in math33)
                .mean()
            )
            math50subset = (
                subset["assigned school"]
                .apply(lambda x: int(x) in math50)
                .mean()
            )
            quality_metrics["met33-" + name] = met33subset
            quality_metrics["met50-" + name] = met50subset
            quality_metrics["color33-" + name] = color33subset
            quality_metrics["color50-" + name] = color50subset
            quality_metrics["math33-" + name] = math33subset
            quality_metrics["math50-" + name] = math50subset

        # 1. by ethnicity
        eth_names = ["white", "asian", "hisp", "black", "decline"]
        eth_cols = [
            "White",
            "Asian",
            "Hispanic/Latino",
            "Black or African American",
            "Decline to State",
        ]
        for eth in range(4):
            subset = student_data[
                student_data["resolved_ethnicity"] == eth_cols[eth]
            ]
            makemetrics(subset, eth_names[eth])
        # 2. by FRL
        student_data["frl_prob"] = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        )
        subset = student_data[student_data["frl_prob"] < 0.5]
        makemetrics(subset, "frl<0.5")
        subset = student_data[student_data["frl_prob"] > 0.5][
            student_data["frl_prob"] < 0.7
        ]
        makemetrics(subset, "frl0.5-0.7")
        subset = student_data[student_data["frl_prob"] > 0.7]
        makemetrics(subset, "frl>0.7")
        # 3. by ELL
        subset = student_data[
            student_data["englprof_desc"] == "L-Limited English"
        ]
        subset2 = student_data[student_data["englprof_desc"] == "N-Non English"]
        subset = pd.concat([subset, subset2])
        makemetrics(subset, "ELL")
        subset = student_data[
            student_data["englprof_desc"] != "L-Limited English"
        ]
        subset = subset[subset["englprof_desc"] != "N-Non English"]
        makemetrics(subset, "nonELL")
        return pd.Series(quality_metrics)

    def filter_students(self, filter):
        """If filter['Students']:
        return self.student_data.loc[filter['Students'], :].
        """
        if filter is None:
            return self.student_data
        rounds = self.student_data["mainround"].isin(filter["Rounds"])
        designated = ethnic = enrolled = ses = ctip = np.ones(
            self.student_data.shape[0], dtype=bool
        )  # default includes all students
        if filter["Designation"] == 1:
            designated = self.is_designated()  # filter designated students
        if filter["Designation"] == 0:
            designated = ~self.is_designated()  # filter non-designated students
        if filter["Ethnicity"] is not None:
            ethnic = (
                self.student_data["resolved_ethnicity"] == filter["Ethnicity"]
            )  # filter students by ethnicity
        if filter["Enrolled"]:
            enrolled = self.student_data["enrolled_idschool"].notnull()
        if filter["SES Quantile"]:
            baseline = (
                self.student_data[["freelunch_prob", "reducedlunch_prob"]]
                .sum(axis=1)
                .quantile(filter["SES Quantile"])
            )
            ses = (
                self.student_data[["freelunch_prob", "reducedlunch_prob"]].sum(
                    axis=1
                )
                >= baseline
            )
        if filter["HasCTIP"] == 1:
            ctip = self.has_ctip()  # filter students with ctip
        if filter["HasCTIP"] == 0:
            ctip = ~self.has_ctip()  # filter students without ctip

        kept_indices = rounds & designated & ethnic & enrolled & ses & ctip

        return self.student_data.loc[kept_indices, :]

    def has_ctip(self, student_data=None):
        """Output: Series where index = student code, value = True (if student had ctip1) or False( if not)."""
        if student_data is None:
            student_data = self.student_data
        return student_data["ctip1"].astype(bool)

    def is_designated(self, student_data=None):
        """Output: Series where index = student code, value = True (if student was assigned) or False( if not)."""
        if student_data is None:
            student_data = self.student_data
        return student_data["designation"].astype(bool)

    def is_assigned(self, student_data=None):
        """Output: Series where index = student code, value = True (if student was assigned) or False (if not)."""
        if student_data is None:
            student_data = self.student_data
        return student_data["assignment"].notnull()

    def is_outside_zone(self, student_data):
        """Output: Series where index = student code, value = True (if student match is outside zone) or False (if not)."""

        def check_zones(student):
            aa = student["idschoolattendance"]
            if (
                aa.notnull()
                and student_data.loc[student.name, "assignment"] != 0
            ):
                return np.any(
                    [
                        False
                        for zone in self.zones
                        if student_data.loc[student.name, "assignment"] in zone
                        and aa in zone
                    ]
                )
            else:
                return False  # could also be np.nan

        return student_data.apply(check_zones, axis=1)

    def eval_distance(self):
        """Output: 'distances' -> Series where index = student code, value = distance to matched school."""
        distances = pd.melt(
            self.distance_data.reset_index(),
            id_vars="studentno",
            var_name="assignment",
            value_name="assignment_dist",
        )
        self.student_data = self.student_data.merge(
            distances, how="left", on=["studentno", "assignment"]
        )
        self.distances = self.student_data.assignment_dist.to_numpy()
        # distances = []
        # assignments = self.student_data['assignment']
        # for i in range(self.student_data.shape[0]):
        #     assignment = assignments.iloc[i]
        #     if isinstance(assignment,str):
        #         distance = self.distance_data[assignment].iloc[i]
        #     else:
        #         distance = 0
        #     distances.append(distance)
        # self.distances = np.array(distances)
        # self.student_data['assignment_dist'] = self.distances

        """
        def get_distance(row):
            return self.distance_data.loc[row['studentno'], row['assignment']]
        print(self.student_data['assignment'])
        match_mask = self.student_data['assignment'].notnull()
        distance_mask = [True if code in self.distance_data.index else False for code in self.student_data.index]
        filtered = self.student_data[match_mask & distance_mask]
        self.distances = pd.Series(filtered.reset_index().apply(get_distance, axis=1).values,
                                   index=filtered.index).replace(0, np.nan)
        self.student_data['assignment_dist'] = self.distances

        if hasattr(self,'travel_times'):
            def get_time(row):
                return self.travel_times.loc[row['studentno'], row['assignment']]

            match_mask = self.student_data['assignment'].notnull()
            distance_mask = [True if code in self.travel_times.index else False for code in self.student_data.index]
            filtered = self.student_data[match_mask & distance_mask]
            self.travel_times = pd.Series(filtered.reset_index().apply(get_time, axis=1).values,
                                       index=filtered.index).replace(0, np.nan)
            self.student_data['assignment_time'] = self.travel_times
        """
        return self.distances

    def rank_cdf(self, student_data, col, rank_clusters=5):
        """Input: 'rank_clusters' -> integer indicating how many buckets to consider when making the rank_cdf
        Output: 'rank_cdf' -> ndarray representing cumulative distribution of student assignment ranks.
        """
        student_data = student_data[student_data["programno"] > 0]
        rank_cdf = np.zeros(rank_clusters)
        for rank in range(1, rank_clusters + 1):
            rank_cdf[rank - 1] = np.sum((student_data[col] <= rank).astype(int))
        rank_cdf /= np.sum(
            (student_data["assignment"] != 0).astype(int)
        )  # normalize

        return rank_cdf

    def _make_eval_matrix(self, student_data=None, distance_threshold=3):
        """Input: 'distance_threshold' -> integer indicating the cutoff for distance calculations."""
        if student_data is None:
            student_data = self.student_data
        eval_matrix = pd.DataFrame(
            {
                "Distance": student_data["assignment_dist"],
                f"Dist. > {distance_threshold}": student_data["assignment_dist"]
                > distance_threshold,
                "Rank": student_data["rank"],
                "Designated": self.is_designated(student_data),
                "Unassigned": ~self.is_assigned(student_data),
            }
        )
        #'Outside Zone': self.is_outside_zone(), will add back in
        return eval_matrix

    def aa_metrics(
        self,
        student_data,
        assignment=False,
        program_level=False,
        distance_threshold=3,
    ):
        """Input: 'distance_threshold' -> integer indicating the cutoff for distance calculations
        Output: 'aa_metrics' -> DataFrame with index = attendance area, columns = student match metrics.
        """
        group_str = "assigned school" if assignment else "idschoolattendance"
        if program_level:
            group_str = "assignment"
        aa_groups = student_data.groupby(group_str)
        aa_metrics = aa_groups.mean(numeric_only=True)
        aa_metrics[f"Avg Dist. > {distance_threshold}"] = (
            aa_metrics["assignment_dist"] > distance_threshold
        )
        aa_metrics["Max Distance"] = aa_groups["assignment_dist"].max()
        aa_metrics["Min Distance"] = aa_groups["assignment_dist"].min()
        aa_metrics["STD Distance"] = aa_groups["assignment_dist"].std()
        aa_metrics["Avg Utility"] = aa_groups["assigned_utility"].std()

        return aa_metrics

    def group_metrics(
        self, student_data=None, group="idschoolattendance", zone_file=""
    ):
        """Args:
            student_data: matrix from Students class
            group: group to perform analysis, 4 options: 'assignment', 'assigned school', 'idschoolattendance', or 'assigned_zone'.

        Returns:
            DataFrame with index=group, columns=ethnicities, ctip, FRL > median, total
        """
        if student_data is None:
            student_data = self.student_data
        if group == "zone_id" or group == "assigned_zone":
            student_data, zone_dicts = self._student_zone_cols(
                zone_file, student_data
            )
        groups = student_data.groupby(group)
        ethnics = (
            student_data.pivot_table(
                index=group,
                columns="resolved_ethnicity",
                values="grade",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )
        counts = pd.Series(ethnics.sum(axis=1).astype(int), name="total")
        ethnics = ethnics.div(ethnics.sum(axis=1), axis=0)  # normalize
        ctip = groups["ctip1"].mean()

        """
        high_frl = pd.Series(student_data.loc[student_data['frl'] > student_data['frl'].median(),
                                              ['frl', group]].groupby(group).count()['frl'] / counts, name='FRL > median')
        """
        dist = groups["assignment_dist"].mean()
        nblockgroups = groups.apply(
            lambda x: x["census_blockgroup"].value_counts().shape[0]
        )
        nblockgroups.name = "Number of BGs"
        hellinger = self.hellinger_at_level(group, student_data)

        def pct_des(x):
            init = x.shape[0]
            x = x["designation"] == 1
            end = x.shape[0]
            if end > 0:
                return init / end
            else:
                return 1

        pctdesignated = groups.apply(lambda x: pct_des(x))

        def des_notzone(x):
            init = x.shape[0]
            x = x[x["designation"] == 1]
            x = x[x["inZone"] == 0]
            end = x.shape[0]
            if end > 0:
                return init / end
            else:
                return 1

        groups.apply(lambda x: des_notzone(x))

        if isinstance(ethnics, pd.Series):
            ethnics = ethnics.to_frame()
        if isinstance(ctip, pd.Series):
            ctip = ctip.to_frame()
        if isinstance(dist, pd.Series):
            dist = dist.to_frame()
        if isinstance(counts, pd.Series):
            counts = counts.to_frame()
        if isinstance(nblockgroups, pd.Series):
            nblockgroups = nblockgroups.to_frame()
        if isinstance(pctdesignated, pd.Series):
            pctdesignated = pctdesignated.to_frame()

        # return ethnics.join(ctip).join(high_frl).join(dist).join(counts).join(nblockgroups).fillna(0)
        return (
            ethnics.join(ctip)
            .join(dist)
            .join(counts)
            .join(nblockgroups)
            .join(hellinger)
            .join(pctdesignated)
            .fillna(0)
        )

    def aa_summary(self, student_data=None):
        """Input: 'student_data' -> matrix from Students class
        Output: -> DataFrame with index=aa, columns=[1,2,3,'% Left AA For Lang, '% Left AA For GE'].
        """
        if student_data is None:
            student_data = self.student_data
        lang_codes = [
            "CN",
            "CE",
            "CB",
            "CT",
            "NC",
            "JB",
            "JE",
            "JN",
            "KN",
            "KE",
            "MN",
            "ME",
            "SB",
            "SN",
            "SE",
            "NS",
            "FB",
        ]

        aa_school_info = (
            student_data.pivot_table(
                index="idschoolattendance",
                values="grade",
                columns="assigned school",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )
        top3 = aa_school_info.apply(
            lambda row: pd.Series(row.index[np.argsort(-row)].values), axis=1
        ).iloc[:, :3]
        top3.rename(
            columns={0: "school_1", 1: "school_2", 2: "school_3"}, inplace=True
        )
        for i in range(1, 4):
            top3[f"school_{i}"] = top3[f"school_{i}"].astype("int64")
        aa_program_info = (
            student_data.pivot_table(
                index="idschoolattendance",
                values="grade",
                columns="assignment",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )
        aa_counts = aa_program_info.sum(axis=1)

        def make_nan(row):
            row[row.index.str.contains(str(int(row.name)))] = np.nan
            return row

        aa_program_info = aa_program_info.apply(make_nan, axis=1)
        aa_lang_program_info = aa_program_info.loc[
            :, aa_program_info.columns.str.contains("|".join(lang_codes))
        ]
        left_for_lang = pd.Series(
            aa_lang_program_info.sum(axis=1) / aa_counts,
            name="Left AA For Lang",
        )
        aa_ge_program_info = aa_program_info.loc[
            :, aa_program_info.columns.str.contains("GE")
        ]
        left_for_ge = pd.Series(
            aa_ge_program_info.sum(axis=1) / aa_counts, name="Left GE For Lang"
        )

        return pd.concat([top3, left_for_lang, left_for_ge], axis=1)

    def school_summary(self, student_data=None, program_level=False):
        if student_data is None:
            student_data = self.student_data
        group = "assignment" if program_level else "assigned school"
        school_aa_info = (
            student_data.pivot_table(
                index=group,
                values="grade",
                columns="idschoolattendance",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )
        top3 = school_aa_info.apply(
            lambda row: pd.Series(row.index[np.argsort(-row)].values), axis=1
        ).iloc[:, :3]
        top3.rename(columns={0: 1, 1: 2, 2: 3}, inplace=True)
        avg_dist = student_data.groupby(group)["assignment_dist"].mean()

        return top3.join(avg_dist)

    def _student_zone_cols(self, zone_file, student_data):
        # get school to zone translation dictionary
        with open(zone_file) as f:
            reader = csv.reader(f)
            zones = list(reader)
        zone_dict = {}
        for idx, schools in enumerate(zones):
            zone_dict = {
                **zone_dict,
                **{int(float(s)): idx for s in schools if s != ""},
            }

        # get students' zone and calc basic stats
        student_data["zone_id"] = student_data["idschoolattendance"].replace(
            zone_dict
        )
        student_data["assigned_zone"] = student_data["assigned school"].apply(
            lambda x: (
                zone_dict[int(x)]
                if (not pd.isnull(x) and int(x) in zone_dict)
                else x
                if pd.isnull(x)
                else int(x)
            )
        )
        return student_data, zone_dict

    def zone_summary(self, zone_file, student_data=None, group="assigned_zone"):
        """Calculate summary statistics for students assigned to a zone given
        an assignment. Note that the zone file should match the zones used to
        create the assignment! There is a function to find this in Simulator.
        """
        if student_data is None:
            student_data = self.student_data

        # get students' zone and calc basic stats
        student_data, zone_dict = self._student_zone_cols(
            zone_file, student_data
        )
        student_data.loc[:, "num_students"] = 1
        num_students = student_data.groupby(group)["num_students"].sum()
        # avg_dist = student_data.groupby(group)['assignment_dist'].mean()
        assigned_avg_rank = student_data.groupby(group)["rank"].mean()
        assigned_frl = student_data.groupby(group)["frl"].mean()
        assigned_frl = pd.DataFrame(assigned_frl).rename(
            columns={"frl": "avg_frl"}
        )

        # calc number >50% isolated GE programs in each zone
        (
            ethnic_groups,
            ethnic_matrix,
            ethnic_matrix_norm,
            ethnic_total,
            ethnic_total_norm,
            populations,
        ) = self._make_ethnic_matrix(
            student_data.loc[student_data["assignment"].str[4:6] == "GE", :]
        )
        ethnic_max = ethnic_matrix_norm.max(axis=1)
        num_isolated = pd.DataFrame(
            ethnic_max[ethnic_max >= 0.50], columns=["num_>50%_GE_iso"]
        )
        num_isolated["school_id"] = [int(x[:3]) for x in num_isolated.index]
        num_isolated["assigned_zone"] = num_isolated["school_id"].replace(
            zone_dict
        )
        num_isolated = num_isolated.groupby("assigned_zone").count()[
            "num_>50%_GE_iso"
        ]

        # get num schools per zone
        num_schools = student_data.groupby("assigned school").first()
        num_schools = num_schools.groupby("assigned_zone").count()
        num_schools = num_schools[["num_students"]].rename(
            columns={"num_students": "num_schools"}
        )

        # get with_blockgroup
        block_groups = student_data.groupby("census_blockgroup")

        def blockgroup_pct(bg, threshold):
            by_school = bg["assigned school"].value_counts()
            min_students = int(threshold * bg.shape[0]) + 1
            over_threshold = by_school[by_school > min_students].sum()
            return over_threshold

        n_to_same = pd.DataFrame(
            block_groups.apply(lambda x: blockgroup_pct(x, 0.1)),
            columns=["Count 10% BG Cohesion"],
        )
        cohesion = (
            student_data[["census_blockgroup", "assigned_zone"]]
            .groupby("census_blockgroup")
            .first()
        )
        cohesion = cohesion.join(n_to_same)
        cohesion = cohesion.groupby("assigned_zone").sum()[
            "Count 10% BG Cohesion"
        ]

        # get % 1st and 3rd choice
        student_data["top1_choice"] = np.where(student_data["rank"] == 1, 1, 0)
        student_data["top3_choice"] = np.where(student_data["rank"] <= 3, 1, 0)
        topchoice = student_data.groupby("assigned_zone").mean(numeric_only=True)[
            ["top1_choice", "top3_choice"]
        ]

        # % designated
        designated = student_data.groupby("assigned_zone").mean(numeric_only=True)[
            "designation"
        ]

        table = pd.concat(
            [
                num_students,
                num_schools,
                assigned_avg_rank,
                assigned_frl,
                num_isolated,
                cohesion,
                topchoice,
                designated,
            ],
            axis=1,
        )
        table.rename(columns={"rank": "avg_rank"}, inplace=True)
        # print(table)
        # print(pd.concat([num_students,num_schools,assigned_avg_rank,avg_dist],axis=1))
        # print(pd.concat([assigned_frl,num_isolated,cohesion,topchoice,designated],axis=1))
        return table

    def average_metrics(self, student_data, quantile=0.9):
        """Input: 'percentile' -> integer indicating percentile cutoff for distance calculation
        Output: 'av_metrics' -> Series with index = average student match metrics.
        """
        return student_data.mean(numeric_only=True)

    def ell_to_lang(self, student_data=None, ranked=False):
        """Input: 'student_data' -> student_data table with format from Students class
               'ranked' -> boolean indicating if calculation is only to programs on students' ranked lists
        Output: -> if 'ranked' == False: percentage of ELL students who are matched to ANY language program
                   if 'ranked' == True: percentage of ELL students who are matched to a language program they ranked.
        """
        if student_data is None:
            student_data = self.student_data

        ell = student_data["englprof_desc"] == "L-Limited English"
        if ranked:
            return (
                student_data.loc[ell, :]
                .apply(
                    lambda row: (
                        row["programtype"] in set(row["r1_programs"]) - {"GE"}
                    ),
                    axis=1,
                )
                .sum()
                / ell.sum()
            )
        else:
            return (
                student_data.loc[ell, "programtype"].isin(LANG_CODES).sum()
                / ell.sum()
            )

    def ell_dist_to_lang(
        self, student_data=None, distance_data=None, max_dist=1
    ):
        """Calculate the percentage of ELL students who are matched to a language program within max_dist miles.

        Args:
             student_data (pd.DataFrame): student_data DataFrame with format from Students class
             distance_data (pd.DataFrame): distances DataFrame with format from Students class
             max_dist (int or float): maximum distance to consider in calculation

        Returns:
             float: percentage of ELL students with a matching language program within max_dist miles
        """
        if student_data is None:
            student_data = self.student_data
        if distance_data is None:
            distance_data = self.distance_data

        ell = student_data["englprof_desc"].isin(
            ["L-Limited English", "N-Non English", "L", "N"]
        )
        ell_homelangs = student_data.loc[ell, "homelang_desc"]
        ell_distances = distance_data.loc[ell, :]

        def func(row, dist):
            homelang = ell_homelangs.loc[row.name]
            if homelang == "SP-Spanish":
                lang_match = ell_distances.columns.str[4:6].isin(
                    ["SB", "SN", "SE", "NS"]
                )
                return ell_distances.loc[row.name, lang_match].min() <= dist
            if homelang == "CC-Chinese Cantonese":
                lang_match = ell_distances.columns.str[4:6].isin(
                    ["CN", "CE", "CB", "CT", "NC"]
                )
                return ell_distances.loc[row.name, lang_match].min() <= dist
            if homelang == "CM-Chinese Mandarin":
                lang_match = ell_distances.columns.str[4:6].isin(["MN", "ME"])
                return ell_distances.loc[row.name, lang_match].min() <= dist
            if homelang == "JA-Japanese":
                lang_match = ell_distances.columns.str[4:6].isin(
                    ["JB", "JE", "JN"]
                )
                return ell_distances.loc[row.name, lang_match].min() <= dist
            if homelang == "KO-Korean":
                lang_match = ell_distances.columns.str[4:6].isin(["KN", "KE"])
                return ell_distances.loc[row.name, lang_match].min() <= dist
            if homelang == "FT-Filipino Tagalog":
                lang_match = ell_distances.columns.str[4:6] == "FB"
                return ell_distances.loc[row.name, lang_match].min() <= dist
            return False

        return (
            ell_distances.apply(func, dist=max_dist, axis=1).sum() / ell.sum()
        )

    def _make_ethnic_matrix(self, student_data=None):
        if student_data is None:
            student_data = self.student_data

        ethnic_groups = student_data["resolved_ethnicity"].dropna().unique()
        ethnic_total = student_data["resolved_ethnicity"].value_counts()
        ethnic_matrix = (
            student_data.pivot_table(
                index="assignment",
                columns="resolved_ethnicity",
                values="grade",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )
        ethnic_matrix_norm = ethnic_matrix.div(
            ethnic_matrix.sum(axis=1), axis=0
        )
        ethnic_total_norm = student_data["resolved_ethnicity"].value_counts(
            normalize=True
        )
        populations = ethnic_matrix.sum(axis=1)

        return (
            ethnic_groups,
            ethnic_matrix,
            ethnic_matrix_norm,
            ethnic_total,
            ethnic_total_norm,
            populations,
        )

    def diversity_metrics(
        self, group1, group2, isolation_threshold=(0.5, 0.55, 0.6, 0.65, 0.7)
    ):
        """Args:
            group1: list of ethnic groups to consider as first group
            group2: list of ethnic groups to consider as second group
            isolation_threshold: list of isolation fractions to consider in isolation metric.

        Returns:
            Series with index = diversity metrics of group1, group2, and entire cohort
        """
        diversity_summary = pd.Series(dtype=float)
        for threshold in isolation_threshold:
            diversity_summary[f"Isolation ({threshold * 100}%)"] = (
                self.isolation(threshold)
            )
        diversity_summary["Dissimilarity"] = self.dissimilarity(group1, group2)
        diversity_summary["Thiel H"] = self.theil()
        diversity_summary["Interaction"] = self.interaction(group1, group2)
        program_isolation = self.ethnic_matrix_norm.max(axis=1)

        return diversity_summary, program_isolation

    def avg_color_index(self, students, school_data):
        total = 0
        count = 0
        color_index_values = school_data["AvgColorIndex"]
        n = students.shape[0]
        for i in range(n):
            assigned_school = self.student_data["assigned school"].iloc[i]
            if isinstance(assigned_school, str):
                color_index = color_index_values.loc[int(assigned_school)]
                total += color_index
                count += 1
        return total / n

    def poverty_concentration(self, all_students, students, threshold):
        if students.empty:
            return np.nan
        schools_frl = all_students.groupby("assigned school").mean(
            numeric_only=True
        )["frl"]
        district_avg = all_students["frl"].mean()
        num_students = students.shape[0]
        count = 0
        for i in range(num_students):
            school = students["assigned school"].iloc[i]
            if isinstance(school, str):
                school_frl = schools_frl.loc[school]
                if school_frl > district_avg + threshold:
                    count += 1
        return count / num_students

    def max_frl(self, student_data=None):
        """Calculate the highest FRL school in given assignment."""
        if student_data is None:
            student_data = self.student_data
        school_frl = student_data.groupby("assigned school").mean(
            numeric_only=True
        )
        return school_frl["frl"].max()

    def median_frl(self, student_data=None):
        """Calculate the median FRL school in given assignment."""
        if student_data is None:
            student_data = self.student_data
        school_frl = student_data.groupby("assigned school").mean(
            numeric_only=True
        )
        return school_frl["frl"].median()

    def isolation(self, ethnic_matrix_norm=None, threshold=0.6, percent=False):
        """Args:
            threshold: isolation percentage to consider as cutoff
            percent: boolean indicating if the percentage of isolated programs is output (True) or number (False).

        Returns:
            float or int: the number of isolated programs (if percent = False) or percentage otherwise
        """
        if ethnic_matrix_norm is None:
            ethnic_matrix_norm = self.ethnic_matrix_norm

        ethnic_max = ethnic_matrix_norm.max(axis=1)
        num_isolated = ethnic_max[ethnic_max >= threshold].count()
        return (
            num_isolated / ethnic_matrix_norm.shape[0]
            if percent
            else num_isolated
        )

    def dissimilarity(self, students, total_enrollment):
        n = students.shape[0]
        total_n = pd.to_numeric(
            pd.Series(np.asarray(total_enrollment).reshape(-1)), errors="coerce"
        ).sum()
        if n == 0 or total_n == 0:
            return np.nan
        ratio = n / total_n
        school_groups = students.groupby("assigned school")
        enrollment = school_groups.count()[
            "SES_category"
        ]  # Just picking an arbitrary column to get the count

        dissimilarity_total = 0
        for i in range(enrollment.shape[0]):
            num_students = enrollment.iloc[i]
            total_students = total_enrollment.iloc[i]
            dissimilarity_total += (
                abs(num_students - total_students * ratio) / 2
            )

        return dissimilarity_total / n

    def interaction(
        self, group1, group2, ethnic_total=None, ethnic_matrix=None
    ):
        """Args:
            group1: list of ethnic groups to consider as first group
            group2: list of ethnic groups to consider as second group.

        Returns:
            interaction index between group1 and group2
        """
        if ethnic_total is None and ethnic_matrix is None:
            ethnic_total = self.ethnic_total
            ethnic_matrix = self.ethnic_matrix

        total1 = ethnic_total[group1].sum()
        return ethnic_matrix.apply(
            lambda row: (
                (row[group1].sum() * row[group2].sum()) / (total1 * row.sum())
            ),
            axis=1,
        ).sum()

    def entropy(
        self, ethnic_total_norm=None, evenness=False, max_entropy=False
    ):
        """Args:
            ethnic_array: array of proportions of each ethnic group
            evenness: boolean indicating if metric is being used to measure only evennes.

        Returns:
            entropy metric of all ethnic groups in school district
        """
        if ethnic_total_norm is None:
            ethnic_total_norm = self.ethnic_total_norm
        if max_entropy:
            return math.log(ethnic_total_norm.size)

        entropy = (ethnic_total_norm * np.log(1 / ethnic_total_norm)).sum()
        return (
            entropy / math.log(ethnic_total_norm.shape[1])
            if evenness
            else entropy
        )

    def theil(
        self,
        ethnic_matrix=None,
        ethnic_total=None,
        ethnic_total_norm=None,
        GE_only=False,
    ):
        """Output: Thiel's H index of all ethnic groups in school district."""
        if (
            ethnic_matrix is None
            and ethnic_total is None
            and ethnic_total_norm is None
        ):
            ethnic_matrix = self.ethnic_matrix
            ethnic_total = self.ethnic_total
            ethnic_total_norm = self.ethnic_total_norm

        if GE_only:
            ge_idxs = [
                True if x[4:6] == "GE" else False for x in ethnic_matrix.index
            ]
            ethnic_matrix = ethnic_matrix[ge_idxs]

            student_data = self.student_data
            student_data["filter"] = student_data["assignment"].apply(
                lambda x: 1 if not pd.isnull(x) and x[4:6] == "GE" else 0
            )
            student_data = student_data.loc[student_data["filter"] == 1]
            ethnic_total = student_data["resolved_ethnicity"].value_counts()
            ethnic_total_norm = student_data["resolved_ethnicity"].value_counts(
                normalize=True
            )

        district_entropy = self.entropy(ethnic_total_norm)
        return ethnic_matrix.apply(
            lambda row: (
                row.sum() * (district_entropy - self.entropy(row / row.sum()))
            ),
            axis=1,
        ).sum() / (district_entropy * ethnic_total.sum())

    def hellinger(
        self, school_code, ethnic_matrix_norm=None, ethnic_total_norm=None
    ):
        if ethnic_matrix_norm is None:
            ethnic_matrix_norm = self.ethnic_matrix_norm
        if ethnic_total_norm is None:
            ethnic_total_norm = self.ethnic_total_norm

        # if school_code in self.student_data['assignment'].unique():
        if school_code in ethnic_matrix_norm.index:
            return math.sqrt(
                0.5
                * ethnic_matrix_norm.loc[school_code, :]
                .reset_index()
                .apply(
                    lambda row: (
                        (
                            math.sqrt(row.iloc[1])
                            - math.sqrt(
                                ethnic_total_norm[row["resolved_ethnicity"]]
                            )
                        )
                        ** 2
                    ),
                    axis=1,
                )
                .sum()
            )
        else:
            return 0

    def hellinger_at_level(self, level_col, student_data=None):
        """Compute hellinger metric at attendance area, school, program, or zone level."""
        if student_data is None:
            student_data = self.student_data

        ethnic_matrix = (
            student_data.pivot_table(
                index=level_col,
                columns="resolved_ethnicity",
                values="grade",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )
        ethnic_matrix_norm = ethnic_matrix.div(
            ethnic_matrix.sum(axis=1), axis=0
        )
        ethnic_total_norm = student_data["resolved_ethnicity"].value_counts(
            normalize=True
        )

        hellinger = [
            self.hellinger(x, ethnic_matrix_norm, ethnic_total_norm)
            for x in ethnic_matrix.index
        ]
        hellinger = pd.DataFrame(
            columns=["Hellinger"], data=hellinger, index=ethnic_matrix.index
        )
        return hellinger

    def hellinger_avg(
        self,
        ethnic_matrix_norm=None,
        ethnic_total_norm=None,
        populations=None,
        GE_only=False,
    ):
        if ethnic_matrix_norm is None:
            ethnic_matrix_norm = self.ethnic_matrix_norm
        if ethnic_total_norm is None:
            ethnic_total_norm = self.ethnic_total_norm
        if populations is None:
            populations = self.populations

        total = populations.sum()
        h = ethnic_matrix_norm.apply(
            lambda row: (
                (populations[row.name] / total)
                * self.hellinger(
                    row.name, ethnic_matrix_norm, ethnic_total_norm
                )
            ),
            axis=1,
        )
        if GE_only:
            ge_idxs = [
                True if x[4:6] == "GE" else False
                for x in ethnic_matrix_norm.index
            ]
            return h[ge_idxs].sum()
        else:
            return h.sum()

    def hellinger_min(self, ethnic_matrix_norm=None, ethnic_total_norm=None):
        if ethnic_matrix_norm is None:
            ethnic_matrix_norm = self.ethnic_matrix_norm
        if ethnic_total_norm is None:
            ethnic_total_norm = self.ethnic_total_norm

        h = ethnic_matrix_norm.apply(
            lambda row: self.hellinger(
                row.name, ethnic_matrix_norm, ethnic_total_norm
            ),
            axis=1,
        )
        ge_idxs = [
            True if x[4:6] == "GE" else False for x in ethnic_matrix_norm.index
        ]
        return h[ge_idxs].min()

    def hellinger_max(self, ethnic_matrix_norm=None, ethnic_total_norm=None):
        if ethnic_matrix_norm is None:
            ethnic_matrix_norm = self.ethnic_matrix_norm
        if ethnic_total_norm is None:
            ethnic_total_norm = self.ethnic_total_norm

        h = ethnic_matrix_norm.apply(
            lambda row: self.hellinger(
                row.name, ethnic_matrix_norm, ethnic_total_norm
            ),
            axis=1,
        )
        ge_idxs = [
            True if x[4:6] == "GE" else False for x in ethnic_matrix_norm.index
        ]
        return h[ge_idxs].max()

    def edi(self, code, ethnic_matrix=None):
        """Output: -> Ethnic Diversity Index at the school level, school/program code required as input."""
        if ethnic_matrix is None:
            ethnic_matrix = self.ethnic_matrix
        groups = [
            "Asian",
            "Hispanic/Latino",
            "White",
            "Two or More Races",
            "Black or African American",
            "Filipino",
            "Pacific Islander",
            "American Indian or Alaskan Native",
        ]
        ethnic_matrix_norm = ethnic_matrix.loc[:, groups].div(
            ethnic_matrix.loc[:, groups].sum(axis=1), axis=0
        )
        d = math.sqrt(
            ethnic_matrix_norm.loc[code, :]
            .apply(lambda ethnic: (ethnic - (1 / 8)) ** 2)
            .sum()
        )
        return 100 - 100 * (math.sqrt(8 * 7) / 7) * d

    def edi_district(self, ethnic_total=None):
        """Output: -> Ethnic Diversity Index at the district level."""
        if ethnic_total is None:
            ethnic_total = self.ethnic_total
        groups = [
            "Asian",
            "Hispanic/Latino",
            "White",
            "Two or More Races",
            "Black or African American",
            "Filipino",
            "Pacific Islander",
            "American Indian or Alaskan Native",
        ]
        ethnic_total_norm = ethnic_total[groups] / ethnic_total[groups].sum()
        d = math.sqrt(
            ethnic_total_norm.apply(
                lambda ethnic: (ethnic - (1 / 8)) ** 2
            ).sum()
        )
        return 100 - 100 * (math.sqrt(8 * 7) / 7) * d

    def ell_lang_access(self, student_data=None):
        """Input: 'student_data' -> student_data table with format from Students class."""
        if student_data is None:
            student_data = self.student_data
        ell = student_data[student_data["englprof_desc"] == "L-Limited English"]
        return ell["programtype"].apply(
            lambda x: x in set(ELL_CODES)
        ).sum() / float(ell.shape[0])

    def ranked_got_lang(self, student_data=None):
        if student_data is None:
            student_data = self.student_data

        def rankedlp3(x):
            for i in range(3):
                if i >= len(x) or x[i] == "":
                    return False
                if x[i][1:-1] in set(LANG_CODES):
                    return True
            return False

        rankedlp3_mask = student_data["r1_programs"].apply(
            lambda x: rankedlp3(x)
        )
        want_lang = student_data[rankedlp3_mask]
        got_lang = want_lang[
            want_lang["programtype"].apply(lambda x: x in set(LANG_CODES))
        ]
        if want_lang.shape[0] != 0:
            return got_lang.shape[0] / want_lang.shape[0]
        else:
            return 0

    def with_blockgroup(self, threshold=0.05, student_data=None):
        """Input: 'threshold' -> pct of students from the block group going to same school
        'student_data' -> student_data table with format from Students class.
        """
        if student_data is None:
            student_data = self.student_data
        block_groups = student_data.groupby("census_blockgroup")

        def blockgroup_pct(bg, threshold):
            by_school = bg["assigned school"].value_counts()
            min_students = int(threshold * bg.shape[0]) + 1
            over_threshold = by_school[by_school > min_students].sum()
            return over_threshold

        n_to_same = block_groups.apply(lambda x: blockgroup_pct(x, threshold))
        return n_to_same.sum() / student_data.shape[0]

    def without_blockgroup(self, threshold=3, student_data=None):
        """Input: 'threshold' -> number of students from the block group going to same school
        'student_data' -> student_data table with format from Students class.
        """
        if student_data is None:
            student_data = self.student_data
        block_groups = student_data.groupby("census_blockgroup")

        def blockgroup_min(bg, threshold):
            by_school = bg["assigned school"].value_counts()
            over_threshold = by_school[by_school < threshold].sum()
            return over_threshold

        n_to_same = block_groups.apply(lambda x: blockgroup_min(x, threshold))
        return n_to_same.sum() / student_data.shape[0]

    def race_isolation(self, threshold=3, ethnic_matrix=None):
        if ethnic_matrix is None:
            ethnic_matrix = self.ethnic_matrix
        hispanic_isolated = [
            x for x in ethnic_matrix["Hispanic/Latino"] if x < threshold
        ]
        hispanic_isolated_pct = (
            sum(hispanic_isolated) / ethnic_matrix["Hispanic/Latino"].sum()
        )
        black_isolated = [
            x
            for x in ethnic_matrix["Black or African American"]
            if x < threshold
        ]
        black_isolated_pct = (
            sum(black_isolated)
            / ethnic_matrix["Black or African American"].sum()
        )
        race_isolated = pd.Series(
            {
                f"Hispanic Isolated ({threshold})": hispanic_isolated_pct,
                f"Black Isolated ({threshold})": black_isolated_pct,
            }
        )
        return race_isolated

    def peers_same_ethnicity(self, ethnic_matrix=None):
        """Calculate avg number of peers of the same ethnicity."""
        if ethnic_matrix is None:
            ethnic_matrix = self.ethnic_matrix
        asian_counts = ethnic_matrix[["Asian", "Chinese"]].sum(axis=1)

        def nonzero_avg(counts):
            return np.mean([x for x in counts if x > 0])

        asian_peers = (
            nonzero_avg(asian_counts) - 1
        )  # -1 for not counting yourself as your own peer
        black_peers = (
            nonzero_avg(ethnic_matrix["Black or African American"]) - 1
        )
        white_peers = nonzero_avg(ethnic_matrix["White"]) - 1
        hispanic_peers = nonzero_avg(ethnic_matrix["Hispanic/Latino"]) - 1
        avg_peers = pd.Series(
            {
                "Avg Asian Peers": asian_peers,
                "Avg Black Peers": black_peers,
                "Avg Hispanic Peers": hispanic_peers,
                "Avg White Peers": white_peers,
            }
        )
        return avg_peers

    def _nonzero_mean(self, col, student_data=None):
        if student_data is None:
            student_data = self.student_data
        data = student_data[student_data[col] > 0]
        return data[col].mean()

    def _colors_quality_schools(self, schools, programs, pct=0.33):
        # identify quality schools
        sch = schools.school_df
        color_values = {
            "Red": 0,
            "Orange": 1,
            "Yellow": 2,
            "Green": 3,
            "Blue": 4,
            "None": 0,
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }
        if "quality" not in sch.columns:
            sch.loc[:, "quality"] = 0
            for col in [
                "ela_color",
                "math_color",
                "chronic_color",
                "suspension_color",
            ]:
                sch["quality"] += sch[col].replace(color_values).astype("int64")
        sch.sort_values("quality", ascending=False, inplace=True)
        num_quality = int(pct * len(sch.index))
        quality_GEprog_idx = []
        for i, row in sch.iterrows():
            if programs.index(str(i) + "-GE-KG", quiet=True) != -1:
                quality_GEprog_idx.append(programs.index(str(i) + "-GE-KG"))
                if len(quality_GEprog_idx) == num_quality:
                    break
        return quality_GEprog_idx

    def _get_quality_schools(self, schools, programs, col, pct=0.33):
        # identify quality schools
        sch = schools.school_df
        sch.sort_values(col, ascending=False, inplace=True)
        num_quality = int(pct * len(sch.index))
        quality_GEprog_idx = []
        for i, row in sch.iterrows():
            if programs.index(str(i) + "-GE-KG", quiet=True) != -1:
                quality_GEprog_idx.append(programs.index(str(i) + "-GE-KG"))
                if len(quality_GEprog_idx) == num_quality:
                    break
        return quality_GEprog_idx

    def _get_cutoff_matrix(self, priorities):
        cutoffs = self.student_data
        cutoffs["programno"] = cutoffs["programno"].astype("int64")
        cutoffs = cutoffs.groupby("programno", as_index=False).mean(
            numeric_only=True
        )
        cutoffs = cutoffs[["programno", "program_cutoff"]]
        cutoffs["program_cutoff"] = cutoffs["program_cutoff"].apply(
            lambda x: max(x - 64, 0)
        )  # round 1 priorities boosted by 64
        cutoffs["programno"] = cutoffs["programno"].astype("int64")
        cutoff_matrix = np.zeros(priorities.shape[1])
        for i, row in cutoffs.iterrows():
            cutoff_matrix[int(row["programno"] - 1)] = row["program_cutoff"]
        cutoff_matrix = np.tile(cutoff_matrix, (priorities.shape[0], 1))

        return cutoff_matrix

    def likelihood_of_quality_school(
        self, quality_GEprog_idx, market, priority_weights
    ):
        """Identify the probability that a student could get into a 'quality'
        school if they ranked it first in round 1.
        """
        # get student priorities
        market.setPriorities(priority_weights, designation_antipriority=0)
        priorities = market.priorities

        # get quality program cutoffs
        cutoff_matrix = self._get_cutoff_matrix(priorities)

        # compare priorities to cutoffs
        accessible_probability = np.clip(cutoff_matrix - priorities, 0, 1)
        shifted = [x - 1 for x in quality_GEprog_idx]
        quality_accessible_probability = accessible_probability[:, shifted]
        self.student_data["quality_access"] = 1 - np.min(
            quality_accessible_probability, axis=1
        )

        # 2 students with no attendance area have no zone and no priority everywhere.
        df = self.student_data.dropna(subset=["idschoolattendance"])
        return np.min(df["quality_access"])

    def ell_likelihood_of_language_prog(
        self, programs, market, priority_weights, language
    ):
        """Determine how likely each ELL student is to get access to a program
        in their language if they were to rank it first in round 1
        INPUT: language - 'S' for Spanish, 'C' for Chinese, immersion or biliteracy.
        """
        # get language program indices
        lps = programs.program_df
        prog_types = [f"{language}B", f"{language}N"]
        lps["filter"] = lps["program_type"].apply(
            lambda x: 1 if x in prog_types else 0
        )
        lps = lps.loc[lps["filter"] == 1]
        lp_idxs = np.array(lps.index)

        # get ELL students in that language
        ell = self.student_data
        ell["filter"] = ell["englprof_desc"].apply(
            lambda x: 1 if x in ["L-Limited English", "N-Non English"] else 0
        )
        ell = ell.loc[ell["filter"] == 1]
        self.students.get_qualified_programs_dict()
        eligible = self.students.qualified_program_dict
        ell_idxs = []
        for i, row in ell.iterrows():
            # if student is eligible for type of language program
            if set(eligible[i]) & set(prog_types):
                ell_idxs.append(self.students.studentno2idx[i])

        # get student priorities
        market.setPriorities(priority_weights, designation_antipriority=0)
        priorities = market.priorities

        # get quality program cutoffs
        cutoff_matrix = self._get_cutoff_matrix(priorities)

        # compare priorities to cutoffs
        accessible_probability = np.clip(cutoff_matrix - priorities, 0, 1)
        lp_accessible_probability = accessible_probability[ell_idxs, :][
            :, lp_idxs
        ]
        by_student = 1 - np.min(lp_accessible_probability, axis=1)
        # print(priorities[ell_idxs[0],lp_idxs])
        # print(cutoff_matrix[ell_idxs[0],lp_idxs])
        # print(accessible_probability[ell_idxs[0],lp_idxs])
        # print('SMALLEST CHANCE:',np.min(by_student))
        return np.min(by_student)

    def poverty_schools_by_race(self, student_data=None):
        if student_data is None:
            student_data = self.student_data
        sch_frl = student_data.groupby("assigned school", as_index=False).mean(
            numeric_only=True
        )
        quantile66 = sch_frl["frl"].quantile(0.66)
        poverty_schools = list(
            sch_frl.loc[sch_frl["frl"] >= quantile66]["assigned school"]
        )

        student_data["high_pov"] = student_data["assigned school"].apply(
            lambda x: 1 if x in poverty_schools else 0
        )
        pov_df = student_data.loc[student_data["high_pov"] == 1]

        pct_pov = {}
        for e in range(len(self.eths)):
            subset = pov_df[pov_df["resolved_ethnicity"] == self.eths[e]]
            total = student_data[
                student_data["resolved_ethnicity"] == self.eths[e]
            ]
            pct_pov["% at poverty school - " + self.eth_labels[e]] = len(
                subset.index
            ) / len(total.index)

        return pd.Series(pct_pov)

    def max_aalpi_pct_GE(self, student_data=None):
        if student_data is None:
            student_data = self.student_data
        # identify aalpi students
        if "aalpi" not in student_data.columns:
            aalpi = [
                "Hispanic/Latino",
                "Black or African American",
                "Pacific Islander",
            ]
            student_data["aalpi"] = student_data["resolved_ethnicity"].apply(
                lambda x: 1 if x in aalpi else 0
            )

        progs = student_data.groupby("assignment").mean(numeric_only=True)
        ge_idxs = [True if x[4:6] == "GE" else False for x in progs.index]
        progs = progs[ge_idxs]
        return max(progs["aalpi"])

    def school_frl_range_district(self, pct, student_data=None, above=False):
        """Fraction of schools where average FRL status of students assigned
        to the school is at least X% above or in the range of X% below to X%
        above the fraction of all students with FRL status.
        """
        if student_data is None:
            student_data = self.student_data
        district_avg = student_data["frl"].mean()
        school_frl = student_data.groupby("assigned school").mean(
            numeric_only=True
        )
        if above is False:
            school_frl["in_range"] = school_frl["frl"].apply(
                lambda x: (
                    1
                    if x <= district_avg + pct and x >= district_avg - pct
                    else 0
                )
            )
        else:
            school_frl["in_range"] = school_frl["frl"].apply(
                lambda x: 1 if x >= district_avg + pct else 0
            )
        return school_frl["in_range"].mean()
