"""Paper-metrics match evaluator used by scripts/analysis/analyze_trends.py.

Created 12/11/2022
@author vravoson

Moved from Model_Analysis/Python_Scripts/Short_Match_Evaluator.py and
parametrized: the schools lat/lon file and the optional equity-block
``.npy`` are constructor arguments instead of hardcoded cluster paths
(configure them via ``schools_data`` / ``new_ctip_path`` in the
analyze_trends config). Legacy cluster defaults are kept below as
clearly-marked fallbacks so older local workflows keep working.
"""

import csv
import logging
import os
from math import asin, cos, isnan, radians, sin, sqrt

import numpy as np
import pandas as pd

from ..definitions.constants import (
    SPECIAL_PROGRAMS,
    ZONE_COLORS,
)

logger = logging.getLogger(__name__)

AALPI = ["Black", "Hispanic", "Pacific Islander"]

# Legacy cluster paths, used only when the caller does not provide the
# corresponding constructor argument. Prefer setting ``schools_data`` /
# ``new_ctip_path`` in the analyze_trends config (see
# scripts/settings/models_cluster.env).
LEGACY_SCHOOLS_LATLON_PATH = (
    "/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/"
    "schools_rehauled_withMissionBay_2324.csv"
)
LEGACY_NEW_CTIP_PATH = (
    "/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/ETB_2024.npy"
)


class MatchEvaluator:
    def __init__(
        self,
        student_data,
        assignments,
        first_round,
        dropout,
        low_income,
        medium_income,
        high_income,
        year,
        grade=None,
        no_special_program=False,
        program_file=None,
        schools_latlon_path=None,
        new_ctip_path=None,
    ):
        self.student_data = student_data
        self.low_income = low_income
        self.medium_income = medium_income
        self.high_income = high_income
        self.grade = grade
        self.year = year

        self.immersion_programs = [
            "CE",
            "CN",
            "CT",
            "JE",
            "JN",
            "KE",
            "KN",
            "ME",
            "MN",
            "SE",
            "SN",
        ]
        self.non_immersion_programs = ["CB", "FB", "NC", "NS", "SB"]
        self.ge = ["GE"]
        self.ce_se = ["CN", "CE", "CB", "SN", "SB", "SE"]
        self.ge_immer_notimmer = (
            self.immersion_programs + self.non_immersion_programs + self.ge
        )

        if schools_latlon_path is None:
            schools_latlon_path = LEGACY_SCHOOLS_LATLON_PATH
            logger.warning(
                "No schools_latlon_path provided — falling back to legacy "
                "cluster path %s. Set `schools_data` in the analysis config.",
                schools_latlon_path,
            )
        self.schools_latlon = pd.read_csv(schools_latlon_path, sep=",")

        if new_ctip_path is None and os.path.exists(LEGACY_NEW_CTIP_PATH):
            new_ctip_path = LEGACY_NEW_CTIP_PATH
        if new_ctip_path:
            new_ctip_list = np.load(new_ctip_path)
        else:
            # No equity-block list available: et2 is 0 for every student.
            new_ctip_list = np.array([])

        if program_file is None:
            raise ValueError(
                "program_file is required (path to the programs CSV, e.g. "
                "programs_without_specialprogs_<year>.csv)"
            )
        self.programs = pd.read_csv(program_file)
        # Normalize program_id type to string for consistent merging
        if "program_id" in self.programs.columns:
            self.programs["program_id"] = self.programs["program_id"].astype(
                str
            )

        if first_round:
            self.student_data = student_data[
                ~student_data["r1_ranked_idschool"].isna()
            ]  # Filter round 1 students only

        self.assignments = assignments
        self.student_data = self.student_data.merge(
            self.assignments, on="studentno"
        )
        # Normalize programcodes to string for consistent merging downstream
        if "programcodes" in self.student_data.columns:
            self.student_data["programcodes"] = (
                self.student_data["programcodes"].fillna("").astype(str)
            )

        if no_special_program:
            student_data = self.student_data
            student_data["is_special"] = student_data["r1_programs"].apply(
                lambda x: (
                    False
                    if str(x) == "nan"
                    else len(set(eval(x)).intersection(SPECIAL_PROGRAMS)) > 0
                )
            )
            # Record the rows that we keep to filter utility models later.
            student_data = student_data[student_data["is_special"] == 0].drop(
                columns=["is_special"]
            )
            self.student_data = student_data

        self.student_data["et2"] = self.student_data["census_block"].apply(
            lambda x: 1 if x in new_ctip_list else 0
        )
        self.student_data["assignment"] = self.student_data[
            "programcodes"
        ].apply(lambda x: "000-ZZ" if pd.isna(x) or x == "" else x)
        # lambda x: np.nan if pd.isna(x) else int(x.split("-")[0])

        self.student_data["assigned_school"] = self.student_data[
            "assignment"
        ].apply(
            lambda x: 0 if pd.isna(x) else int(x.split("-")[0])
            # lambda x: np.nan if pd.isna(x) else int(x.split("-")[0])
        )
        self.student_data["programtype"] = self.student_data["assignment"].str[
            4:6
        ]
        self.student_data["frl"] = (
            self.student_data["freelunch_prob"]
            + self.student_data["reducedlunch_prob"]
        )
        self.student_data["is_frl"] = self.student_data["frl"].apply(
            lambda x: 1 if x >= 0.5 else 0
        )
        self.student_data["assigned school"] = self.student_data[
            "assignment"
        ].str[:3]
        self.map_ethnicity()
        self.eval_distance()

    def map_ethnicity(self):
        def ethn(x):
            if x in [
                "Asian",
                "Asian Indian",
                "Chinese",
                "Vietnamese",
                "Filipino",
                "Japanese",
                "Korean",
                "Hmong",
                "Other Asian",
                "Cambodian",
                "Laotian",
            ]:
                return "Asian"
            elif x in ["Hispanic/Latino", "Hispanic/Latinx", "Hispanic"]:
                return "Hispanic"
            elif x in ["White", "Middle Eastern/Arabic"]:
                return "White"
            elif x in ["Black or African American", "Black/African American"]:
                return "Black"
            elif x in [
                "Other Pacific Islander",
                "Pacific Islander",
                "Samoan",
                "Hawaiian Native",
            ]:
                return "Pacific Islander"
            elif x in ["Two or More Races", "Multi-Racial", "Two or More"]:
                return "Two or More Races"
            elif x in ["Decline to State", "Decline To State"]:
                return "Decline to State"
            return "Other"

        self.student_data["ethnicity"] = self.student_data[
            "resolved_ethnicity"
        ].apply(ethn)

    def eval_distance(self):
        """Output: 'distances' -> Series where index = student code, value = distance to matched school."""

        def haversine_dist(lat1: float, lon1: float, school):
            if pd.isna(school) or school <= 0:
                return np.nan

            sk_latlon = self.schools_latlon[
                self.schools_latlon["school_id"] == school
            ]
            if sk_latlon.empty:
                return np.nan
            lat2 = float(sk_latlon["lat"].iloc[0])
            lon2 = float(sk_latlon["lon"].iloc[0])

            lat1, lat2, lon1, lon2 = [
                radians(_) for _ in [lat1, lat2, lon1, lon2]
            ]
            c = 2 * asin(
                sqrt(
                    sin((lat2 - lat1) / 2) ** 2
                    + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
                )
            )  # angle
            r = 3958.8  # earth radius (in miles)

            return r * c

        self.student_data["assignment_dist"] = self.student_data.apply(
            lambda x: haversine_dist(
                x["latitude"], x["longitude"], x["assigned_school"]
            ),
            axis=1,
        )

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

    def school_income_range(
        self, all_students, income_threshold, prop_threshold
    ):
        """Proportion of schools above/below pct
        the average proportion of low income students
        in the district.
        """
        prop_district_below_thres = np.mean(
            all_students.median_hh_income <= income_threshold
        )
        schools_nb_low = (
            all_students[all_students.median_hh_income <= income_threshold]
            .groupby("assigned school")
            .count()["studentno"]
        )
        schools_prop_low = (
            schools_nb_low
            / all_students.groupby("assigned school").count()["studentno"]
        )
        schools_prop_low = pd.DataFrame(schools_prop_low).reset_index()
        if prop_threshold > 0:
            return np.mean(
                schools_prop_low["studentno"]
                >= prop_district_below_thres + prop_threshold
            )
        else:
            return np.mean(
                schools_prop_low["studentno"]
                < prop_district_below_thres + prop_threshold
            )

    def high_income_school_range(
        self, all_students, income_threshold, prop_threshold
    ):
        """(#Schools, Proportion of schools) where the average income is
        above a specified threshold compared to the district average,
        adjusted by a proportion threshold.
        """
        prop_district_above_thres = np.mean(
            all_students.median_hh_income >= income_threshold
        )
        schools_nb_high = (
            all_students[all_students.median_hh_income >= income_threshold]
            .groupby("assigned school")
            .count()["studentno"]
        )
        schools_prop_high = (
            schools_nb_high
            / all_students.groupby("assigned school").count()["studentno"]
        )
        schools_prop_high = pd.DataFrame(schools_prop_high).reset_index()
        if prop_threshold > 0:
            schools_matching = (
                schools_prop_high["studentno"]
                >= prop_district_above_thres + prop_threshold
            )
            return np.sum(schools_matching), np.mean(schools_matching)
        else:
            schools_matching = (
                schools_prop_high["studentno"]
                < prop_district_above_thres + prop_threshold
            )
            return np.sum(schools_matching), np.mean(schools_matching)

    def aalpi_in_high_income_schools(
        self, all_students, income_threshold, prop_threshold
    ):
        """Computes the proportion of AALPI (African American, Latino, Pacific Islander) students
        in schools with average income above a specified threshold.
        """
        prop_district_above_thres = np.mean(
            all_students.median_hh_income >= income_threshold
        )
        schools_nb_high = (
            all_students[all_students.median_hh_income >= income_threshold]
            .groupby("assigned school")
            .count()["studentno"]
        )
        schools_prop_high = (
            schools_nb_high
            / all_students.groupby("assigned school").count()["studentno"]
        )
        schools_prop_high = pd.DataFrame(schools_prop_high).reset_index()
        if prop_threshold > 0:
            high_income_schools = schools_prop_high[
                schools_prop_high.studentno
                >= prop_district_above_thres + prop_threshold
            ]["assigned school"].to_list()
        else:
            high_income_schools = schools_prop_high[
                schools_prop_high.studentno
                < prop_district_above_thres + prop_threshold
            ]["assigned school"].to_list()
        student_data_hi = all_students[
            all_students["assigned school"].isin(high_income_schools)
        ]
        prop_aalpi_hi_schools = (
            student_data_hi[student_data_hi.ethnicity.isin(AALPI)]
            .groupby("assigned school")
            .count()["studentno"]
            .sum()
            / student_data_hi.groupby("assigned school")
            .count()["studentno"]
            .sum()
        )
        return prop_aalpi_hi_schools

    def aalpi_in_low_income_schools(
        self, all_students, income_threshold, prop_threshold
    ):
        """Computes the proportion of AALPI (African American, Latino, Pacific Islander) students
        in schools with average income below a specified threshold.
        """
        prop_district_below_thres = np.mean(
            all_students.median_hh_income <= income_threshold
        )
        schools_nb_low = (
            all_students[all_students.median_hh_income <= income_threshold]
            .groupby("assigned school")
            .count()["studentno"]
        )
        schools_prop_low = (
            schools_nb_low
            / all_students.groupby("assigned school").count()["studentno"]
        )
        schools_prop_low = pd.DataFrame(schools_prop_low).reset_index()
        if prop_threshold > 0:
            low_income_schools = schools_prop_low[
                schools_prop_low.studentno
                >= prop_district_below_thres + prop_threshold
            ]["assigned school"].to_list()
        else:
            low_income_schools = schools_prop_low[
                schools_prop_low.studentno
                < prop_district_below_thres + prop_threshold
            ]["assigned school"].to_list()
        student_data_li = all_students[
            all_students["assigned school"].isin(low_income_schools)
        ]
        prop_aalpi_li_schools = (
            student_data_li[student_data_li.ethnicity.isin(AALPI)]
            .groupby("assigned school")
            .count()["studentno"]
            .sum()
            / student_data_li.groupby("assigned school")
            .count()["studentno"]
            .sum()
        )
        return prop_aalpi_li_schools

    def avg_aalpi_in_high_income_schools(
        self, all_students, income_threshold, prop_threshold
    ):
        """Calculates the average proportion of AALPI (African American, Latino, Pacific Islander)
        students across high income schools with average income above a threshold.
        """
        prop_district_above_thres = np.mean(
            all_students.median_hh_income >= income_threshold
        )
        schools_nb_high = (
            all_students[all_students.median_hh_income >= income_threshold]
            .groupby("assigned school")
            .count()["studentno"]
        )
        schools_prop_high = (
            schools_nb_high
            / all_students.groupby("assigned school").count()["studentno"]
        )
        schools_prop_high = pd.DataFrame(schools_prop_high).reset_index()
        if prop_threshold > 0:
            high_income_schools = schools_prop_high[
                schools_prop_high.studentno
                >= prop_district_above_thres + prop_threshold
            ]["assigned school"].to_list()
        else:
            high_income_schools = schools_prop_high[
                schools_prop_high.studentno
                < prop_district_above_thres + prop_threshold
            ]["assigned school"].to_list()
        student_data_hi = all_students[
            all_students["assigned school"].isin(high_income_schools)
        ]
        avg_prop_aalpi_hi_schools = (
            student_data_hi[student_data_hi.ethnicity.isin(AALPI)]
            .groupby("assigned school")
            .count()["studentno"]
            .mean()
            / student_data_hi.groupby("assigned school")
            .count()["studentno"]
            .mean()
        )
        return avg_prop_aalpi_hi_schools

    def avg_frl_in_high_income_schools(
        self, all_students, income_threshold, prop_threshold
    ):
        """Calculates the average proportion of FRL students across high
        income schools with average income above a threshold.
        """
        prop_district_above_thres = np.mean(
            all_students.median_hh_income >= income_threshold
        )
        schools_nb_high = (
            all_students[all_students.median_hh_income >= income_threshold]
            .groupby("assigned school")
            .count()["studentno"]
        )
        schools_prop_high = (
            schools_nb_high
            / all_students.groupby("assigned school").count()["studentno"]
        )
        schools_prop_high = pd.DataFrame(schools_prop_high).reset_index()
        if prop_threshold > 0:
            high_income_schools = schools_prop_high[
                schools_prop_high.studentno
                >= prop_district_above_thres + prop_threshold
            ]["assigned school"].to_list()
        else:
            high_income_schools = schools_prop_high[
                schools_prop_high.studentno
                < prop_district_above_thres + prop_threshold
            ]["assigned school"].to_list()
        student_data_hi = all_students[
            all_students["assigned school"].isin(high_income_schools)
        ]
        avg_prop_frl_hi_schools = (
            student_data_hi[student_data_hi.is_frl == 1]
            .groupby("assigned school")
            .count()["studentno"]
            .mean()
            / student_data_hi.groupby("assigned school")
            .count()["studentno"]
            .mean()
        )
        return avg_prop_frl_hi_schools

    def poverty_concentration(
        self, all_students, students, threshold, return_count=False
    ):
        """Proportion of students in schools where the percentage of FRL students
        exceeds or falls below the district average by a certain threshold.
        """
        schools_frl = (
            all_students[["assigned school", "frl"]]
            .groupby("assigned school")
            .mean()["frl"]
        )
        district_avg = all_students["frl"].mean()
        num_students = students.shape[0]
        count = 0
        for i in range(num_students):
            school = students["assigned school"].iloc[i]
            if isinstance(school, str):
                school_frl = schools_frl.loc[str(school)]
                if (
                    (threshold > 0) and (school_frl > district_avg + threshold)
                ) or (
                    (threshold < 0) and (school_frl < district_avg + threshold)
                ):
                    count += 1
        if return_count:
            return count
        else:
            return count / num_students

    def ge_frl_range(self, ge_students, ge_groups, threshold):
        ges_frl = ge_groups[["frl"]].mean()
        district_avg = ge_students["frl"].mean()
        if threshold >= 0:
            ges_matching = ges_frl["frl"] >= district_avg + threshold
            return np.sum(ges_matching), np.mean(ges_matching)
        else:
            ges_matching = ges_frl["frl"] <= district_avg + threshold
            return np.sum(ges_matching), np.mean(ges_matching)

    def metric_ge_FRL_concentration(
        self, ge_students, group_students, ge_groups, threshold
    ):
        """Proportion of students in GE programs where % of FRL students
        exceeds or falls below the district average by a certain threshold.
        """
        ges_frl = ge_groups[["frl"]].mean()["frl"]
        district_avg = ge_students["frl"].mean()
        num_students = group_students.shape[0]
        count = 0
        for i in range(num_students):
            ge = group_students["assignment"].iloc[i]
            if isinstance(ge, str):
                ge_frl = ges_frl.loc[ge]
                if (
                    (threshold > 0) and (ge_frl > district_avg + threshold)
                ) or ((threshold < 0) and (ge_frl < district_avg + threshold)):
                    count += 1
        return count / num_students

    def ge_AAPI_range(
        self,
        higher_threshold,
        lower_threshold=0,
        smaller_strict=True,
        percentage=True,
        student_data=None,
    ):
        """Compute the number of GE programs that have less than threshold of their
        capacity or a certain number African American or Pacific Islander students.
        """
        if student_data is None:
            student_data = self.student_data
        program_aapi_cnts = (
            student_data[
                student_data["ethnicity"].isin(["Black", "Pacific Islander"])
            ][["programcodes", "studentno"]]
            .groupby("programcodes")
            .count()
            .reset_index()
        )
        if not percentage:
            # Return counts based on number of students.
            if smaller_strict:
                return len(
                    program_aapi_cnts[
                        (program_aapi_cnts["studentno"] >= lower_threshold)
                        & (program_aapi_cnts["studentno"] < higher_threshold)
                    ]
                )
            else:
                return len(
                    program_aapi_cnts[
                        (program_aapi_cnts["studentno"] >= lower_threshold)
                        & (program_aapi_cnts["studentno"] <= higher_threshold)
                    ]
                )

        # Return number based on percentage of capacity.
        ge_programs = self.programs[
            self.programs["program_type"] == "GE"
        ].copy()
        # Ensure types are consistent for merge
        program_aapi_cnts["programcodes"] = program_aapi_cnts[
            "programcodes"
        ].astype(str)
        ge_programs["program_id"] = ge_programs["program_id"].astype(str)
        ge_programs = ge_programs.merge(
            program_aapi_cnts,
            how="left",
            left_on="program_id",
            right_on="programcodes",
        )
        if smaller_strict:
            ge_programs["below_threshold"] = ge_programs[
                ["studentno", "capacity"]
            ].apply(
                lambda x: (
                    x["studentno"] < higher_threshold * x["capacity"]
                    and x["studentno"] >= lower_threshold * x["capacity"]
                ),
                axis=1,
            )
        else:
            ge_programs["below_threshold"] = ge_programs[
                ["studentno", "capacity"]
            ].apply(
                lambda x: (
                    x["studentno"] <= higher_threshold * x["capacity"]
                    and x["studentno"] >= lower_threshold * x["capacity"]
                ),
                axis=1,
            )
        return ge_programs["below_threshold"].sum()

    def load_zone_file_dict(self, zone_file):
        """Load a dictionary for zone_file mapping zone building block to zone id.

        Input:
            zone_file: path to the zone file
        Returns:
            the dictionary mapping zone building block to zone id
        """
        if zone_file is None:
            logger.warning("Zone file is None; map metrics may be incomplete.")
            return {}
        with open(zone_file) as f:
            reader = csv.reader(f)
            zones = list(reader)

        zone_dict = {}
        for idx, schools in enumerate(zones):
            zone_dict = {
                **zone_dict,
                **{int(float(s)): idx for s in schools if s != ""},
            }
        return zone_dict

    def eval_assignment_metrics_by_student_area(
        self,
        program_data,  # not used here.
        building_block="idschoolattendance",
        zone_file=None,
        school_order=None,
        zone_order=None,
    ):
        """Generate the dataframe for map-level metrics for 2024 summer SFUSD
        dashboard based on student area.

        TODO: similar to the function eval_assignment_metrics_by_assigned_area
        with only differences on grouping students. Refactor if have time.

        Input:
            program_data: program df used for in capacity and empty seats
            building_block: one of "idschoolattendance", "Block" or "BlockGroup"
            zone_file: zone file as used in configs. Can only be none if using "aa" as
                zone building_block.
            school_order: a list of ordered schools to order the columns in the outputs
            zone_order: a list of ordered zones to order the columns in the outputs
        Returns:
            the dataframe where rows are metrics, and columns are the zones or AA.
        """
        student_data = self.student_data.copy()
        school_id_to_name = {
            x: y
            for [x, y] in self.schools_latlon[
                ["school_id", "school_name"]
            ].to_numpy()
        }
        # Map student to their map area by their building block.
        if building_block == "idschoolattendance":
            student_data["map_area"] = student_data["idschoolattendance"]
            school_to_zone_dict = {
                x: x for x in self.schools_latlon["school_id"].to_numpy()
            }
        elif building_block in ["Block", "BlockGroup"]:
            zone_dict = self.load_zone_file_dict(zone_file)
            building_block_col = "census_" + building_block.lower()
            student_data["map_area"] = student_data[building_block_col].apply(
                lambda x: zone_dict[x] if not isnan(x) else np.nan
            )

            # Count attendance school to zone, and citywide school as itself.
            school_to_zone_dict = self.schools_latlon[
                ["school_id", building_block, "category"]
            ].apply(
                lambda x: [
                    x[0],
                    zone_dict[x[1]] if x[2] == "Attendance" else x[0],
                ],
                axis=1,
            )
            school_to_zone_dict = {
                x: y for [x, y] in school_to_zone_dict.to_numpy()
            }
        else:
            logger.warning(
                "Expected building_block to be one of idschoolattendance, "
                "Block, or BlockGroup; returning None."
            )
            return None

        # Build up program data matched to compute program capacity.
        program_data = program_data.copy()
        program_data["program_area"] = program_data["school_id"].apply(
            lambda x: school_to_zone_dict[x]
        )
        # Gather the area list for building blocks.
        if building_block == "BlockGroup":
            # For formatting, have 18 columns no matter how many zones we have.
            area_list = [x for x in range(18)]
        elif building_block == "Block":
            # Adding missing zones
            area_list = [x for x in range(59)]
        elif building_block == "idschoolattendance":
            area_list = [
                x
                for x in self.schools_latlon["school_id"].to_numpy()
                if x != 909
            ]

        list_metrics = []
        for area in area_list:
            cur_students = student_data[student_data["map_area"] == area]
            # cur_programs = program_data[program_data["program_area"] == area]
            metrics = self.generate_metrics_by_aera(cur_students, None)
            list_metrics.append(pd.Series(metrics))

        # Format to return
        area_list = [
            self.format_cur_col_name(x, school_id_to_name) for x in area_list
        ]
        col_names = dict(zip(range(len(area_list)), area_list))
        all_metrics_df = pd.concat(list_metrics, axis=1).rename(
            columns=col_names
        )
        all_metrics_df = self.reorder_schools_zones(
            all_metrics_df, school_order=school_order, zone_order=zone_order
        )
        return all_metrics_df

    def eval_assignment_metrics_by_assigned_area(
        self,
        program_data,
        building_block="idschoolattendance",
        zone_file=None,
        school_order=None,
        zone_order=None,
    ):
        """Generate the dataframe for map-level metrics for 2024 summer SFUSD
        dashboard based on assigned school area.

        TODO: the current codes are very messy, we need to clean it up and/or optimize
            it when we have time.

        Input:
            program_data: program df used for in capacity and empty seats
            building_block: one of "idschoolattendance", "Block" or "BlockGroup"
            zone_file: zone file as used in configs. Can only be none if using "aa" as
                zone building_block.
            school_order: a list of ordered schools to order the columns in the outputs
            zone_order: a list of ordered zones to order the columns in the outputs
        Returns:
            the dataframe where rows are metrics, and columns are the zones or AA.
        """
        student_data = self.student_data.copy()
        school_id_to_name = {
            x: y
            for [x, y] in self.schools_latlon[
                ["school_id", "school_name"]
            ].to_numpy()
        }
        # Map student to their assigned area by assigned_school.
        if building_block == "idschoolattendance":
            school_to_zone_dict = {
                x: x for x in self.schools_latlon["school_id"].to_numpy()
            }
        elif building_block in ["Block", "BlockGroup"]:
            zone_dict = self.load_zone_file_dict(zone_file)
            # Count attendance school to zone, and citywide school as itself.
            school_to_zone_dict = self.schools_latlon[
                ["school_id", building_block, "category"]
            ].apply(
                lambda x: [
                    x[0],
                    zone_dict[x[1]] if x[2] == "Attendance" else x[0],
                ],
                axis=1,
            )
            school_to_zone_dict = {
                x: y for [x, y] in school_to_zone_dict.to_numpy()
            }
        else:
            logger.warning(
                "Expected building_block to be one of idschoolattendance, "
                "Block, or BlockGroup; returning None."
            )
            return None
        # Count unassigned student to their located zone based on attandance area.
        student_data["assigned_area"] = student_data[
            ["assigned_school", "idschoolattendance"]
        ].apply(
            lambda x: (
                school_to_zone_dict[x[0]]
                if not isnan(x[0])
                else school_to_zone_dict[x[1]]
            ),
            axis=1,
        )
        # Build up program data matched to compute program capacity.
        program_data = program_data.copy()
        program_data["program_area"] = program_data["school_id"].apply(
            lambda x: school_to_zone_dict[x]
        )
        # area_list = ["All"] + list(np.unique(student_data["assigned_area"].to_numpy()))
        area_list = ["All"]
        aa_schools = self.schools_latlon[
            self.schools_latlon["category"] == "Attendance"
        ]["school_id"].to_numpy()
        aa_schools = [x for x in aa_schools if x != 909]
        all_schools = [
            x for x in self.schools_latlon["school_id"].to_numpy() if x != 909
        ]
        citywide_schools = list(set(all_schools) - set(aa_schools))
        # Fill in what metrics need to be generated for each config.
        # Zone ids + citywide schools for block or blockgroup and all schools for AA.
        if building_block == "BlockGroup":
            # For formatting, have 18 columns no matter how many zones we have.
            area_list += [x for x in range(18) if x not in area_list]
            area_list += citywide_schools
        elif building_block == "Block":
            # Adding missing zones
            area_list += [x for x in range(59) if x not in area_list]
            area_list += citywide_schools
        elif building_block == "idschoolattendance":
            area_list += all_schools

        list_metrics = []
        for area in area_list:
            if area == "All":
                cur_students = student_data
                cur_programs = program_data
            else:
                cur_students = student_data[
                    student_data["assigned_area"] == area
                ]
                cur_programs = program_data[
                    program_data["program_area"] == area
                ]

            metrics = self.generate_metrics_by_aera(cur_students, cur_programs)
            list_metrics.append(pd.Series(metrics))

        # Add metrics at attendance school level.
        # TODO: currently code is messy and redundant. We should clean this up
        # and/or optimize when we have time. Ideally, we should not seperate
        # Attendance area school from the citywide schools.
        if building_block in ["Block", "BlockGroup"]:
            aa_schools = self.schools_latlon[
                self.schools_latlon["category"] == "Attendance"
            ]["school_id"].to_numpy()
            # 909 is duplicate of 999 and not used in assignment.
            aa_schools = [x for x in aa_schools if x != 909]
            area_list += list(aa_schools)
            for aa_school in aa_schools:
                cur_students = student_data[
                    student_data["assigned_school"] == aa_school
                ]
                cur_programs = program_data[
                    program_data["school_id"] == aa_school
                ]
                metrics = self.generate_metrics_by_aera(
                    cur_students, cur_programs
                )
                list_metrics.append(pd.Series(metrics))

        area_list = [
            self.format_cur_col_name(x, school_id_to_name) for x in area_list
        ]
        col_names = dict(zip(range(len(area_list)), area_list))
        all_metrics_df = pd.concat(list_metrics, axis=1).rename(
            columns=col_names
        )
        all_metrics_df = self.reorder_schools_zones(
            all_metrics_df, school_order=school_order, zone_order=zone_order
        )
        return all_metrics_df

    def reorder_schools_zones(
        self, all_metrics_df, school_order=None, zone_order=None
    ):
        """Reorder columns based on input school and/or zone orders. Assume that
        the columns of all_metrics_df contains "All", and a list of zones with format
        "Zone <zone id>" (e.g. "Zone 1") and/or a list of schools in the format
        "<School id> <school name>".
        TODO: optimize if have time.
        """
        columns = all_metrics_df.columns
        new_columns = ["All"] if "All" in columns else []
        # Separate list of zone and schools.
        list_zones = [x for x in columns if "Zone" in x]
        list_schools = [x for x in columns if "Zone" not in x and x != "All"]
        # TODO: zone order if we have.
        new_columns += list_zones
        # School orders, need to rename as the school name are different in our
        # records v.s. the ones given by SFUSD.
        if school_order is None:
            new_columns += list_schools
        else:
            # Exclude schools not in results from school order
            existing_schools = set([x[:3] for x in list_schools])
            school_order = [
                x for x in school_order if x[:3] in existing_schools
            ]
            # Rename columns to the new school names.
            new_schoolid2school = {x[:3]: x for x in school_order}
            school_old2new = {
                x: new_schoolid2school[x[:3]] for x in list_schools
            }
            all_metrics_df = all_metrics_df.rename(columns=school_old2new)
            new_columns += school_order
        return all_metrics_df[new_columns]

    def format_cur_col_name(self, area, school_id_to_name):
        """Helper function for eval_assignment_metrics_by_area to decide the column
        name for the map-level metrics.
        """
        if area == "All":
            return area
        # With time limit, assume no zone has id > 100
        if area > 100:
            return f"{area} {school_id_to_name[area]}"
        else:
            return f"Zone {area + 1} {ZONE_COLORS[area]}"

    def generate_metrics_by_aera(
        self,
        cur_students,
        cur_programs,
        all_students=None,
        ge_over_assigned=True,
    ):
        if all_students is None:
            all_students = self.student_data.copy()
        ignore_program_related = cur_programs is None
        assigned_students = all_students[all_students["programno"] > 0]

        metrics = {}
        cur_assigned_students = cur_students[cur_students["programno"] > 0]
        cur_aalpi_students = cur_assigned_students[
            cur_assigned_students["ethnicity"].apply(lambda x: x in AALPI)
        ]
        # Counts of assigned, capacity, designated, unassigned.
        if not ignore_program_related:
            metrics["Capacity"] = cur_programs["capacity"].sum()
        metrics["Students"] = len(cur_students)
        if not ignore_program_related:
            metrics["# schools"] = len(
                np.unique(cur_programs["school_id"].to_numpy())
            )
        metrics["Assigned students"] = len(cur_assigned_students)
        metrics["Assigned to language program"] = len(
            cur_students[
                cur_students["programtype"].isin(
                    self.immersion_programs + self.non_immersion_programs
                )
            ]
        )
        metrics["Not assigned"] = len(
            cur_students[cur_students["programno"] == 0]
        )
        metrics["Designated"] = len(
            cur_students[(cur_students["designation"] != 0)]
        )
        if not ignore_program_related:
            metrics["Empty Seats"] = (
                metrics["Capacity"] - metrics["Assigned students"]
            )
            if ge_over_assigned:
                # Given time limit on implementation, assume over-assignment is only for GE.
                ge_capacity = cur_programs[
                    cur_programs["program_type"] == "GE"
                ][["capacity", "school_id"]]
                ge_students = (
                    cur_students[cur_students["programtype"] == "GE"][
                        ["assigned_school", "studentno"]
                    ]
                    .groupby("assigned_school")
                    .count()
                    .reset_index()
                    .rename(
                        columns={
                            "assigned_school": "school_id",
                            "studentno": "student_count",
                        }
                    )
                )
                ge_empty_seats = ge_capacity.merge(
                    ge_students, how="outer", on="school_id"
                )
                ge_empty_seats["empty_seats"] = (
                    ge_empty_seats["capacity"] - ge_empty_seats["student_count"]
                )
                metrics["# GE Empty Seats"] = np.sum(
                    ge_empty_seats[ge_empty_seats["empty_seats"] >= 0][
                        "empty_seats"
                    ].to_numpy()
                )
                metrics["# GE Over Assigned Seats"] = -np.sum(
                    ge_empty_seats[ge_empty_seats["empty_seats"] < 0][
                        "empty_seats"
                    ].to_numpy()
                )
        # School choice.
        metrics["Assigned to 1st choice"] = (
            cur_assigned_students["rank"] <= 1
        ).mean()
        metrics["Assigned top-3 choice"] = (
            cur_assigned_students["rank"] <= 3
        ).mean()
        # Distances
        metrics["Avg. Distance"] = cur_assigned_students[
            "assignment_dist"
        ].mean()
        metrics["Median Distance"] = cur_assigned_students[
            "assignment_dist"
        ].median()
        metrics["% assigned within 0.5mi"] = (
            cur_assigned_students["assignment_dist"] < 0.5
        ).mean()
        metrics["% assigned beyond 3mi"] = (
            cur_assigned_students["assignment_dist"] > 3
        ).mean()
        # Schools
        (
            metrics["# high-poverty schools"],
            _,
            metrics["# students in high-poverty schools"],
        ) = self.school_frl_range(
            0.15,
            student_data=cur_assigned_students,
            all_student_data=assigned_students,
        )
        metrics["# AALPI students in high-poverty schools"] = (
            self.poverty_concentration(
                assigned_students, cur_aalpi_students, 0.15, return_count=True
            )
        )
        metrics["# of students from high-poverty area"] = len(
            cur_assigned_students[cur_assigned_students["frl"] > 0.5]
        )
        # Count of students ethnicity groups.
        groups = ["Black", "Asian", "Hispanic", "White"]
        for group in groups:
            group_name = group if group != "Black" else "African American"
            metrics[f"# {group_name} students"] = len(
                cur_students[(cur_students["ethnicity"] == group)]
            )
        return metrics

    def school_frl_range(
        self,
        pct,
        student_data=None,
        all_student_data=None,
        non_desig_only=False,
    ):
        """(#of schools, proportion of schools, #of students in those schools) above/below pct
        the average proportion of FRL students in the district.
        """
        if student_data is None:
            student_data = self.student_data
        if all_student_data is None:
            all_student_data = self.student_data
            # TODO: district_avg and school_frl should remains the same within the
            # same ME (based on same set of assignment results) if all_student_data
            # is the whole student_data. Double check and maybe save and reuse instead
            # of computing it everytime.
        district_avg = all_student_data["frl"].mean()

        # consider average frl only of non-designated students
        if non_desig_only:
            student_data = student_data[student_data["designation"] == 0]
            school_frl = (
                student_data[["assigned school", "frl"]]
                .groupby("assigned school")
                .mean()
            )
        else:
            school_frl = (
                all_student_data[["assigned school", "frl"]]
                .groupby("assigned school")
                .mean()
            )

        if pct >= 0:
            schools_matching = school_frl["frl"] >= district_avg + pct
            school_ids = schools_matching[schools_matching].index.tolist()
            return (
                np.sum(schools_matching),
                np.mean(schools_matching),
                np.sum(student_data["assigned school"].isin(school_ids)),
            )
        else:
            schools_matching = school_frl["frl"] <= district_avg + pct
            school_ids = schools_matching[schools_matching].index.tolist()
            return (
                np.sum(schools_matching),
                np.mean(schools_matching),
                np.sum(student_data["assigned school"].isin(school_ids)),
            )

    def metric_dissimilarity(self, students, total_enrollment):
        n = students.shape[0]
        total_n = total_enrollment.sum()
        ratio = n / total_n
        enrollment = students.groupby("assigned school").count()["studentno"]
        dissimilarity_total = 0
        for school in total_enrollment.index:
            num_students = enrollment.get(school, 0)
            total_students = total_enrollment.loc[school]
            dissimilarity_total += (
                abs(num_students - total_students * ratio) / 2
            )
        if n == 0:
            return -1.0
        else:
            return dissimilarity_total / n

    def metrics_segregation(
        self, group_a, group_b, proportion, total_enrollment
    ):
        segregation_total = 0
        en_a = group_a.groupby("assigned school").count()["studentno"]
        en_a_sum = en_a.sum()
        en_b = group_b.groupby("assigned school").count()["studentno"]
        en_b_sum = en_b.sum()
        prop = proportion.groupby("assigned school").count()["studentno"]
        for school in total_enrollment.index:
            num_a = en_a.get(school, 0)
            num_b = en_b.get(school, 0)
            prop_school = prop.get(school, 0) / total_enrollment.loc[school]
            contri = ((num_a / en_a_sum) - (num_b / en_b_sum)) * prop_school
            segregation_total += contri
        return segregation_total

    def metrics_exposure(self, group, proportion, total_enrollment):
        """Compute the absolute exposure of a group to a proportion.

        This calculates the average proportion of 'proportion' students in the
        schools attended by students in 'group'. Unlike metrics_segregation,
        this does not compute a difference between two groups.

        Args:
            group: DataFrame of students in the group of interest.
            proportion: DataFrame of students representing the proportion
                (e.g., high FRL students, AALPI students).
            total_enrollment: Series with school as index and total enrollment
                as value.

        Returns:
            float: The weighted average exposure.
        """
        en_group = group.groupby("assigned school").count()["studentno"]
        en_group_sum = en_group.sum()
        if en_group_sum == 0:
            return 0.0
        prop = proportion.groupby("assigned school").count()["studentno"]
        exposure_total = 0
        for school in total_enrollment.index:
            num_group = en_group.get(school, 0)
            prop_school = prop.get(school, 0) / total_enrollment.loc[school]
            contri = (num_group / en_group_sum) * prop_school
            exposure_total += contri
        return exposure_total

    def metric_theil(self, student_data=None):
        """Compute Theil's H (entropy-based) segregation index.

        Measures how evenly ethnic groups are distributed across schools
        relative to the district-wide composition.

        Returns:
            float: Theil H index (0 = perfect integration, 1 = complete segregation)
        """
        if student_data is None:
            student_data = self.student_data
        assigned = student_data[student_data["programno"] > 0].copy()
        if len(assigned) == 0:
            return 0.0

        # Build ethnic matrix: rows=schools, cols=ethnicities
        ethnic_matrix = (
            assigned.pivot_table(
                index="assigned school",
                columns="ethnicity",
                values="studentno",
                aggfunc="count",
            )
            .fillna(0)
            .astype(int)
        )
        ethnic_total = assigned["ethnicity"].value_counts()
        ethnic_total_norm = assigned["ethnicity"].value_counts(normalize=True)

        # District entropy
        district_entropy = (
            ethnic_total_norm * np.log(1.0 / ethnic_total_norm)
        ).sum()
        if district_entropy == 0:
            return 0.0

        # School-level entropy weighted by school size
        total_students = ethnic_total.sum()
        theil_sum = 0.0
        for school_id, row in ethnic_matrix.iterrows():
            school_total = row.sum()
            if school_total == 0:
                continue
            school_props = row / school_total
            # Avoid log(0)
            school_props = school_props[school_props > 0]
            school_entropy = (school_props * np.log(1.0 / school_props)).sum()
            theil_sum += school_total * (district_entropy - school_entropy)

        return theil_sum / (district_entropy * total_students)

    def eval_assignment_paper_metrics(self):
        low_income, medium_income, high_income = (
            self.low_income,
            self.medium_income,
            self.high_income,
        )
        metrics = {}
        student_data = self.student_data
        assigned_students = student_data[student_data["programno"] > 0]
        assigned_students = assigned_students.reindex()

        designated_students = student_data[(student_data["designation"] != 0)]

        # Add total AALPI counts for both all students and assigned students
        metrics["Total AALPI students"] = len(
            student_data[student_data["ethnicity"].isin(AALPI)]
        )
        metrics["Total AALPI assigned students"] = len(
            assigned_students[assigned_students["ethnicity"].isin(AALPI)]
        )

        school_groups = assigned_students.groupby("assigned school")
        enrollment = school_groups.count()["studentno"]

        non_designated_students = student_data[
            (student_data["programno"] > 0) & (student_data["designation"] == 0)
        ]
        class_below_medium_income = student_data[
            (student_data["programno"] > 0)
            & (student_data["median_hh_income"] <= medium_income)
        ]
        class_above_medium_income = student_data[
            (student_data["programno"] > 0)
            & (student_data["median_hh_income"] > medium_income)
        ]
        student_data[
            (student_data["programno"] > 0)
            & (student_data["median_hh_income"] <= low_income)
        ]
        student_data[
            (student_data["programno"] > 0)
            & (student_data["median_hh_income"] > low_income)
            & (student_data["median_hh_income"] <= high_income)
        ]
        student_data[
            (student_data["programno"] > 0)
            & (student_data["median_hh_income"] > low_income)
            & (student_data["median_hh_income"] <= medium_income)
        ]
        student_data[
            (student_data["programno"] > 0)
            & (student_data["median_hh_income"] > medium_income)
            & (student_data["median_hh_income"] <= high_income)
        ]
        student_data[
            (student_data["programno"] > 0)
            & (student_data["median_hh_income"] > high_income)
        ]
        high_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) > 0.5
        low_frl_prob = (
            student_data["freelunch_prob"] + student_data["reducedlunch_prob"]
        ) <= 0.5
        high_frl_students = student_data[high_frl_prob]
        low_frl_students = student_data[low_frl_prob]

        # PROXIMITY METRICS

        non_designated = self.student_data[
            self.student_data["designation"] == 0
        ]
        designated = self.student_data[self.student_data["designation"] == 1]
        metrics["Total Designated"] = len(designated)
        metrics["Not designated"] = len(non_designated)

        for group in ["Black", "Asian", "Hispanic", "White"]:
            metrics[f" Non-designated {group} students"] = len(
                non_designated[non_designated["ethnicity"] == group]
            )
            metrics[f" Designated {group} students"] = len(
                designated[designated["ethnicity"] == group]
            )

        metrics["Tot Nb Students (Round 1)"] = len(student_data)
        metrics["Tot Nb Assigned (Round 1)"] = len(assigned_students)
        metrics["Tot Nb Designated (Round 1)"] = len(designated_students)
        metrics["Distance Av (All Assigned)"] = assigned_students[
            "assignment_dist"
        ].mean()
        metrics["Distance Median (All Assigned)"] = assigned_students[
            "assignment_dist"
        ].median()
        metrics["Distance < 0.5 (All Assigned)"] = (
            assigned_students["assignment_dist"] < 0.5
        ).mean()
        metrics["Distance < 1 (All Assigned)"] = (
            assigned_students["assignment_dist"] < 1.0
        ).mean()
        metrics["Distance > 3 (All Assigned)"] = (
            (assigned_students["assignment_dist"] >= 0.5)
            & (assigned_students["assignment_dist"] > 3)
        ).mean()

        # DIVERSITY METRICS
        metrics[
            "#GE programs that have more than 0% and less than 10% of their capacity as African American or Pacific Islander students"
        ] = self.ge_AAPI_range(
            0.1,
            lower_threshold=0.0001,
            smaller_strict=True,
            percentage=True,
        )
        metrics[
            "#GE programs that have exactly 0 African American or Pacific Islander students"
        ] = self.ge_AAPI_range(
            1,
            lower_threshold=0,
            smaller_strict=True,
            percentage=False,
        )
        metrics[
            "#GE programs that have 1-4 African American or Pacific Islander students"
        ] = self.ge_AAPI_range(
            4,
            lower_threshold=1,
            smaller_strict=False,
            percentage=False,
        )

        for x in [10, 15, -10, -15]:
            (col1, col2, col3) = (
                f"#Schools {'above' if x >= 0 else 'below'} {x}% district FRL",
                f"Schools {'above' if x >= 0 else 'below'} {x}% district FRL",
                f"#Students in schools {'above' if x >= 0 else 'below'} {x}% district FRL",
            )
            (metrics[col1], metrics[col2], metrics[col3]) = (
                self.school_frl_range(x / 100.0)
            )
            (col1, col2, col3) = (
                f"#Schools {'above' if x >= 0 else 'below'} {x}% district FRL (Non-Designated)",
                f"Schools {'above' if x >= 0 else 'below'} {x}% district FRL (Non-Designated)",
                f"#Students in schools {'above' if x >= 0 else 'below'} {x}% district FRL  (Non-Designated)",
            )
            (metrics[col1], metrics[col2], metrics[col3]) = (
                self.school_frl_range(x / 100.0, non_desig_only=True)
            )

        aalpi_students = assigned_students[
            assigned_students["ethnicity"].apply(lambda x: x in AALPI)
        ]
        metrics["AALPI in school with +10% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, 0.1
        )
        metrics["AALPI in school with +15% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, 0.15
        )
        metrics["AALPI in school with -10% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, -0.1
        )
        metrics["AALPI in school with -15% FRL"] = self.poverty_concentration(
            assigned_students, aalpi_students, -0.15
        )

        ge_students = assigned_students[
            assigned_students["programtype"] == "GE"
        ]
        ge_aalpi_students = ge_students[
            ge_students["ethnicity"].apply(lambda x: x in AALPI)
        ]
        ge_groups = ge_students.groupby("assignment")

        (
            metrics["#GE above +10% district FRL"],
            metrics["GE above +10% district FRL"],
        ) = self.ge_frl_range(ge_students, ge_groups, 0.1)
        (
            metrics["#GE above +15% district FRL"],
            metrics["GE above +15% district FRL"],
        ) = self.ge_frl_range(ge_students, ge_groups, 0.15)
        (
            metrics["#GE below -10% district FRL"],
            metrics["GE below -10% district FRL"],
        ) = self.ge_frl_range(ge_students, ge_groups, -0.1)
        (
            metrics["#GE below -15% district FRL"],
            metrics["GE below -15% district FRL"],
        ) = self.ge_frl_range(ge_students, ge_groups, -0.15)

        metrics["AALPI in GE with +10% FLR"] = self.metric_ge_FRL_concentration(
            ge_students, ge_aalpi_students, ge_groups, 0.1
        )
        metrics["AALPI in GE with +15% FLR"] = self.metric_ge_FRL_concentration(
            ge_students, ge_aalpi_students, ge_groups, 0.15
        )
        metrics["AALPI in GE with -10% FLR"] = self.metric_ge_FRL_concentration(
            ge_students, ge_aalpi_students, ge_groups, -0.1
        )

        metrics["AALPI in GE with -15% FLR"] = self.metric_ge_FRL_concentration(
            ge_students, ge_aalpi_students, ge_groups, -0.15
        )

        designated_aalpi_students = designated_students[
            designated_students["ethnicity"].apply(lambda x: x in AALPI)
        ]
        if len(designated_students) == 0:
            metrics["AALPI in designated"] = 0
        else:
            metrics["AALPI in designated"] = len(
                designated_aalpi_students
            ) / len(designated_students)

        (
            metrics[f"#Schools with +10% High Income ({medium_income})"],
            metrics[f"Prop schools with +10% High Income ({medium_income})"],
        ) = self.high_income_school_range(assigned_students, medium_income, 0.1)
        (
            metrics[f"#Schools with +15% High Income ({medium_income})"],
            metrics[f"Prop schools with +15% High Income ({medium_income})"],
        ) = self.high_income_school_range(
            assigned_students, medium_income, 0.15
        )
        (
            metrics[f"#Schools with -10% High Income ({medium_income})"],
            metrics[f"Prop schools with -10% High Income ({medium_income})"],
        ) = self.high_income_school_range(
            assigned_students, medium_income, -0.1
        )
        (
            metrics[f"#Schools with -15% High Income ({medium_income})"],
            metrics[f"Prop schools with -15% High Income ({medium_income})"],
        ) = self.high_income_school_range(
            assigned_students, medium_income, -0.15
        )

        metrics[f"Prop AALPI in +10% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students, medium_income, 0.1
            )
        )
        metrics[f"Prop AALPI in +15% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students, medium_income, 0.15
            )
        )
        metrics[f"Prop AALPI in -10% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students, medium_income, -0.1
            )
        )
        metrics[f"Prop AALPI in -15% High Income Schools ({medium_income})"] = (
            self.aalpi_in_high_income_schools(
                assigned_students, medium_income, -0.15
            )
        )

        metrics[
            f"Avg Prop AALPI in +10% High Income Schools ({medium_income})"
        ] = self.avg_aalpi_in_high_income_schools(
            assigned_students, medium_income, 0.1
        )
        metrics[
            f"Avg Prop AALPI in +15% High Income Schools ({medium_income})"
        ] = self.avg_aalpi_in_high_income_schools(
            assigned_students, medium_income, 0.15
        )
        metrics[
            f"Avg Prop AALPI in -10% High Income Schools ({medium_income})"
        ] = self.avg_aalpi_in_high_income_schools(
            assigned_students, medium_income, -0.1
        )
        metrics[
            f"Avg Prop AALPI in -15% High Income Schools ({medium_income})"
        ] = self.avg_aalpi_in_high_income_schools(
            assigned_students, medium_income, -0.15
        )

        metrics[f"Prop AALPI in -10% Low Income Schools ({low_income})"] = (
            self.aalpi_in_low_income_schools(
                assigned_students, low_income, -0.10
            )
        )

        metrics[f"Prop AALPI in -15% Low Income Schools ({low_income})"] = (
            self.aalpi_in_low_income_schools(
                assigned_students, low_income, -0.15
            )
        )

        metrics[
            f"Avg Prop FRL in +10% High Income Schools ({medium_income})"
        ] = self.avg_frl_in_high_income_schools(
            assigned_students, medium_income, 0.1
        )
        metrics[
            f"Avg Prop FRL in +15% High Income Schools ({medium_income})"
        ] = self.avg_frl_in_high_income_schools(
            assigned_students, medium_income, 0.15
        )

        # CHOICE METRICS

        metrics["#Unassigned"] = student_data[
            student_data["programno"] == 0
        ].shape[0]
        metrics["Unassigned"] = (
            student_data[student_data["programno"] == 0].shape[0]
            / student_data.shape[0]
        )
        all_designated = assigned_students["designation"].mean()
        metrics["#Designated"] = assigned_students["designation"].sum()
        metrics["Designated"] = all_designated

        dico_student_type = {
            "All Assigned": assigned_students,
            "Non-Designated": non_designated_students,
            #  "Designated": designated_students,
        }

        for student_type, data_student in dico_student_type.items():
            metrics[f"Prop Top 1 choice ({student_type})"] = (
                data_student["rank"] <= 1
            ).mean()
            metrics[f"Prop Top 2 choice ({student_type})"] = (
                data_student["rank"] <= 2
            ).mean()
            metrics[f"Prop Top 3 choice ({student_type})"] = (
                data_student["rank"] <= 3
            ).mean()
            metrics[f"Mean Choice ({student_type})"] = data_student[
                "rank"
            ].mean()
            metrics[f"Median Choice ({student_type})"] = data_student[
                "rank"
            ].median()
            metrics[f"Top 1 in-zone choice ({student_type})"] = (
                data_student["In-Zone Rank"] == 1
            ).mean()
            metrics[f"Top 2 in-zone choice ({student_type})"] = (
                data_student["In-Zone Rank"] <= 2
            ).mean()
            metrics[f"Top 3 in-zone choice ({student_type})"] = (
                data_student["In-Zone Rank"] <= 3
            ).mean()

            metrics[f"Distance Av ({student_type})"] = data_student[
                "assignment_dist"
            ].mean()
            metrics[f"Distance Median ({student_type})"] = data_student[
                "assignment_dist"
            ].median()

        high_frl_assigned = high_frl_students[
            high_frl_students["programno"] > 0
        ]
        low_frl_assigned = low_frl_students[low_frl_students["programno"] > 0]

        ctip_assigned = student_data[
            (student_data["ctip1"] == 1) & (student_data["programno"] > 0)
        ]
        non_ctip_assigned = student_data[
            (student_data["ctip1"] == 0) & (student_data["programno"] > 0)
        ]

        et2_assigned = student_data[
            (student_data["et2"] == 1) & (student_data["programno"] > 0)
        ]
        non_et2_assigned = student_data[
            (student_data["et2"] == 0) & (student_data["programno"] > 0)
        ]

        groups = [
            "All Assigned",
            "Black",
            "Asian",
            "Hispanic",
            "White",
            # "Pacific Islander",
            "Two or More Races",
            "Decline to State",
            "High FRL",
            "Low FRL",
            "CTIP",
            "non-CTIP",
            "ET (2024)",
            "non-ET (2024)",
            "AALPI",
        ]

        dico_income = {
            f"Income below {medium_income}": class_below_medium_income,
            f"Income above {medium_income}": class_above_medium_income,
        }

        white_students = assigned_students[
            assigned_students["ethnicity"] == "White"
        ]
        for ethnicity in ["Black", "Hispanic"]:
            ethnic_students = assigned_students[
                assigned_students["ethnicity"] == ethnicity
            ]
            metrics[f"{ethnicity}/White exposure to AALPI"] = (
                self.metrics_segregation(
                    ethnic_students, white_students, aalpi_students, enrollment
                )
            )
            metrics[f"{ethnicity}/White exposure to poverty"] = (
                self.metrics_segregation(
                    ethnic_students,
                    white_students,
                    high_frl_students,
                    enrollment,
                )
            )
            # New: exposure difference to low FRL
            metrics[f"{ethnicity}/White exposure to low FRL"] = (
                self.metrics_segregation(
                    ethnic_students,
                    white_students,
                    low_frl_students,
                    enrollment,
                )
            )

        # Absolute exposure metrics (no difference between groups)
        # Exposure to high FRL and low FRL for key groups
        exposure_groups = {
            "AALPI": aalpi_students,
            "White": white_students,
            "Black": assigned_students[
                assigned_students["ethnicity"] == "Black"
            ],
            "Hispanic": assigned_students[
                assigned_students["ethnicity"] == "Hispanic"
            ],
            "High FRL": high_frl_assigned,
            "Low FRL": low_frl_assigned,
        }

        for group_name, group_students in exposure_groups.items():
            # Exposure to AALPI
            metrics[f"{group_name} exposure to AALPI"] = self.metrics_exposure(
                group_students, aalpi_students, enrollment
            )
            # Exposure to high FRL (poverty)
            metrics[f"{group_name} exposure to high FRL"] = (
                self.metrics_exposure(
                    group_students, high_frl_students, enrollment
                )
            )
            # Exposure to low FRL
            metrics[f"{group_name} exposure to low FRL"] = (
                self.metrics_exposure(
                    group_students, low_frl_students, enrollment
                )
            )

        # Precompute school FRL probabilities for efficiency
        # Average probability of FRL students in each school
        school_frl_probs = (
            assigned_students[["assigned school", "frl"]]
            .groupby("assigned school")["frl"]
            .mean()
        )
        district_frl_prob = assigned_students[
            "frl"
        ].mean()  # District average FRL prob

        # Precalculate schools meeting relative thresholds
        # Schools with FRL > District Avg + 10%
        schools_high_frl_rel = school_frl_probs[
            school_frl_probs > district_frl_prob + 0.1
        ].index
        # Schools with FRL > District Avg + 15%
        schools_high_frl_rel_15 = school_frl_probs[
            school_frl_probs > district_frl_prob + 0.15
        ].index
        # Schools with FRL > District Avg
        schools_avg_frl_rel = school_frl_probs[
            school_frl_probs > district_frl_prob
        ].index

        for group in groups + list(dico_income.keys()):
            if group == "All Assigned":
                students = self.student_data[self.student_data["programno"] > 0]
            elif group == "High FRL":
                students = high_frl_assigned
            elif group == "Low FRL":
                students = low_frl_assigned
            elif group == "CTIP":
                students = ctip_assigned
            elif group == "non-CTIP":
                students = non_ctip_assigned
            elif group == "ET (2024)":
                students = et2_assigned
            elif group == "non-ET (2024)":
                students = non_et2_assigned
            elif group == "AALPI":
                students = aalpi_students
            elif group in groups:
                students = assigned_students[
                    assigned_students["ethnicity"] == group
                ]
            else:
                students = dico_income[group]

            metrics[f"Number of assigned students ({group})"] = len(students)

            # --- New FRL Metrics ---
            if len(students) > 0:
                # 1. Exposure to High FRL (threshold 0.5)
                # Note: This is exposure to PEERS who are High FRL
                metrics[f"{group} exposure to high FRL"] = (
                    self.metrics_exposure(
                        students, high_frl_students, enrollment
                    )
                )

                # 2. Exposure to FRL (prob)
                # Average school FRL probability for students in this group
                # Maps each student's school to its avg FRL prob, then takes mean over group
                metrics[f"{group} exposure to FRL prob"] = (
                    students["assigned school"].map(school_frl_probs).mean()
                )

                # 3. Relative High FRL (>+10% dist FRL)
                # Proportion of group in schools with > Dist + 10% FRL
                metrics[f"{group} in school with +10% FRL"] = (
                    students["assigned school"]
                    .isin(schools_high_frl_rel)
                    .mean()
                )

                metrics[f"{group} in school with +15% FRL"] = (
                    students["assigned school"]
                    .isin(schools_high_frl_rel_15)
                    .mean()
                )

                # 4. FRL district average (Relative FRL > Avg)
                # Proportion of group in schools with > Dist Avg FRL
                metrics[f"{group} in school with > avg FRL"] = (
                    students["assigned school"].isin(schools_avg_frl_rel).mean()
                )
            else:
                metrics[f"{group} exposure to high FRL"] = 0
                metrics[f"{group} exposure to FRL prob"] = 0
                metrics[f"{group} in school with +10% FRL"] = 0
                metrics[f"{group} in school with +15% FRL"] = 0
                metrics[f"{group} in school with > avg FRL"] = 0
            # -----------------------

            # --- Added Choice Metrics for Subgroups ---
            if len(students) > 0:
                metrics[f"Prop Top 1 choice ({group})"] = (
                    students["rank"] <= 1
                ).mean()
                metrics[f"Prop Top 2 choice ({group})"] = (
                    students["rank"] <= 2
                ).mean()
                metrics[f"Prop Top 3 choice ({group})"] = (
                    students["rank"] <= 3
                ).mean()
                metrics[f"Distance Av ({group})"] = students[
                    "assignment_dist"
                ].mean()
                metrics[f"Distance Median ({group})"] = students[
                    "assignment_dist"
                ].median()
            else:
                metrics[f"Prop Top 1 choice ({group})"] = 0
                metrics[f"Prop Top 2 choice ({group})"] = 0
                metrics[f"Prop Top 3 choice ({group})"] = 0
                metrics[f"Distance Av ({group})"] = 0
                metrics[f"Distance Median ({group})"] = 0
            # ------------------------------------------

            len(students[students["programtype"].isin(self.ge_immer_notimmer)])
            len(
                students[
                    (students["programtype"].isin(self.ge_immer_notimmer))
                    & (students["designation"] == 1)
                ]
            )
            len(
                students[
                    (students["programtype"].isin(self.ge_immer_notimmer))
                    & (students["designation"] == 0)
                ]
            )

            metrics[f"Dissimilarity ({group})"] = self.metric_dissimilarity(
                students, enrollment
            )

            # if n_ge_immer_notimmer > 0:
            metrics[f"Nb assigned students ({group}) to LP program"] = len(
                students[
                    students["programtype"].isin(
                        self.immersion_programs + self.non_immersion_programs
                    )
                ]
            )
            metrics[f"Nb assigned students ({group}) to immersion program"] = (
                len(
                    students[
                        students["programtype"].isin(self.immersion_programs)
                    ]
                )
            )
            metrics[
                f"Nb assigned students ({group}) to non-immersion program"
            ] = len(
                students[
                    students["programtype"].isin(self.non_immersion_programs)
                ]
            )
            metrics[f"Nb assigned students ({group}) to GE program"] = len(
                students[students["programtype"].isin(self.ge)]
            )

            for program in self.ce_se:
                metrics[f"Nb assigned students ({group}) to {program}"] = len(
                    students[students["programtype"].isin([program])]
                )

            metrics[f"Nb designated students ({group}) to LP program"] = len(
                students[
                    (
                        students["programtype"].isin(
                            self.immersion_programs
                            + self.non_immersion_programs
                        )
                    )
                    & (students["designation"] == 1)
                ]
            )
            metrics[
                f"Nb designated students ({group}) to immersion program"
            ] = len(
                students[
                    (students["programtype"].isin(self.immersion_programs))
                    & (students["designation"] == 1)
                ]
            )  # /n_ge_immer_notimmer_designated
            metrics[
                f"Nb designated students ({group}) to non-immersion program"
            ] = len(
                students[
                    (students["programtype"].isin(self.non_immersion_programs))
                    & (students["designation"] == 1)
                ]
            )  # /n_ge_immer_notimmer_designated
            metrics[f"Nb designated students ({group}) to GE program"] = len(
                students[
                    (students["programtype"].isin(self.ge))
                    & (students["designation"] == 1)
                ]
            )

            for program in self.ce_se:
                metrics[f"Nb designated students ({group}) to {program}"] = len(
                    students[
                        (students["programtype"].isin([program]))
                        & (students["designation"] == 1)
                    ]
                )

            metrics[f"Nb designated students ({group}) to LP program"] = len(
                students[
                    (
                        students["programtype"].isin(
                            self.immersion_programs
                            + self.non_immersion_programs
                        )
                    )
                    & (students["designation"] == 1)
                ]
            )
            metrics[
                f"Nb non-designated students ({group}) to immersion program"
            ] = len(
                students[
                    (students["programtype"].isin(self.immersion_programs))
                    & (students["designation"] == 0)
                ]
            )
            metrics[
                f"Nb non-designated students ({group}) to non-immersion program"
            ] = len(
                students[
                    (students["programtype"].isin(self.non_immersion_programs))
                    & (students["designation"] == 0)
                ]
            )
            metrics[f"Nb non-designated students ({group}) to GE program"] = (
                len(
                    students[
                        (students["programtype"].isin(self.ge))
                        & (students["designation"] == 0)
                    ]
                )
            )

            for program in self.ce_se:
                metrics[
                    f"Nb non-designated students ({group}) to {program}"
                ] = len(
                    students[
                        (students["programtype"].isin([program]))
                        & (students["designation"] == 0)
                    ]
                )

            (
                _,
                _,
                metrics[
                    f"#Students in schools above +15% district FRL ({group})"
                ],
            ) = self.school_frl_range(0.15, student_data=students)
            metrics[
                f"Prop students in schools above +15% district FRL ({group})"
            ] = metrics[
                f"#Students in schools above +15% district FRL ({group})"
            ] / len(students)

            # when students are designated, assign rank 999
            students.loc[students["designation"] == 1, "rank"] = 999

            if group != "All Assigned":
                metrics[f"Number of designated students ({group})"] = len(
                    students[students["designation"] == 1]
                )
                metrics[f"Prop Top 1 choice Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["rank"] <= 1
                ).mean()

                metrics[f"Prop Top 2 choice Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["rank"] <= 2
                ).mean()

                for i in range(1, 11):
                    metrics[f"Proportion of students in top {i} ({group})"] = (
                        students["rank"] <= i
                    ).mean()
                    metrics[
                        f"Proportion of students in top {i} (All Students)"
                    ] = (student_data["rank"] <= i).mean()
                metrics[f"Prop Top 3 choice Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["rank"] <= 3
                ).mean()
                metrics[f"Top 1 in-zone choice Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["In-Zone Rank"] == 1
                ).mean()
                metrics[f"Top 2 in-zone choice Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["In-Zone Rank"] <= 2
                ).mean()
                metrics[f"Top 3 in-zone choice Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["In-Zone Rank"] <= 3
                ).mean()
                metrics[f"Mean Choice Non-Designated ({group})"] = students[
                    students["designation"] == 0
                ]["rank"].mean()
                metrics[f"Median Choice Non-Designated ({group})"] = students[
                    students["designation"] == 0
                ]["rank"].median()
                metrics[f"Distance Av Non-Designated ({group})"] = students[
                    students["designation"] == 0
                ]["assignment_dist"].mean()
                metrics[f"Distance Median Non-Designated ({group})"] = students[
                    students["designation"] == 0
                ]["assignment_dist"].median()
                metrics[f"Distance < 0.5 Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["assignment_dist"]
                    < 0.5
                ).mean()
                metrics[f"Distance > 3 Non-Designated ({group})"] = (
                    students[students["designation"] == 0]["assignment_dist"]
                    > 3
                ).mean()

            # Evaluate for all groups, append together as distances.
            metrics[f"Prop Distance > 3 and Rank>=5 ({group})"] = (
                (students["assignment_dist"] > 3) & (students["rank"] >= 5)
            ).mean()
            metrics[f"Prop Distance > 3 and in-zone Rank>=5 ({group})"] = (
                (students["assignment_dist"] > 3)
                & (students["In-Zone Rank"] >= 5)
            ).mean()

            cur_non_designatd_student = students[students["designation"] == 0]
            metrics[
                f"Prop Distance > 3 and Rank>=5 Non-Designated ({group})"
            ] = (
                (cur_non_designatd_student["assignment_dist"] > 3)
                & (cur_non_designatd_student["rank"] >= 5)
            ).mean()
            metrics[
                f"Prop Distance > 3 and in-zone Rank>=5 Non-Designated ({group})"
            ] = (
                (cur_non_designatd_student["assignment_dist"] > 3)
                & (cur_non_designatd_student["In-Zone Rank"] >= 5)
            ).mean()

            # --- Prop Distance > 3 and Rank>=4 ---
            metrics[f"Prop Distance > 3 and Rank>=4 ({group})"] = (
                (students["assignment_dist"] > 3) & (students["rank"] >= 4)
            ).mean()
            metrics[
                f"Prop Distance > 3 and Rank>=4 Non-Designated ({group})"
            ] = (
                (cur_non_designatd_student["assignment_dist"] > 3)
                & (cur_non_designatd_student["rank"] >= 4)
            ).mean()
            # Prop Distance > 3 and (Rank>=4 or designated)
            metrics[
                f"Prop Distance > 3 and (Rank>=4 or designated) ({group})"
            ] = (
                (students["assignment_dist"] > 3)
                & ((students["rank"] >= 4) | (students["designation"] == 1))
            ).mean()
            # Prop Distance > 3 and (Rank>=5 or designated)
            metrics[
                f"Prop Distance > 3 and (Rank>=5 or designated) ({group})"
            ] = (
                (students["assignment_dist"] > 3)
                & ((students["rank"] >= 5) | (students["designation"] == 1))
            ).mean()
            # Prop Distance > 3 and (in-zone Rank>=5 or designated)
            metrics[
                f"Prop Distance > 3 and (in-zone Rank>=5 or designated) ({group})"
            ] = (
                (students["assignment_dist"] > 3)
                & (
                    (students["In-Zone Rank"] >= 5)
                    | (students["designation"] == 1)
                )
            ).mean()

            # --- Variance metrics ---
            if len(students) > 0:
                metrics[f"Variance of rank ({group})"] = students["rank"].var()
                metrics[f"Variance of in-zone rank ({group})"] = students[
                    "In-Zone Rank"
                ].var()
                metrics[f"Variance of distance ({group})"] = students[
                    "assignment_dist"
                ].var()
            else:
                metrics[f"Variance of rank ({group})"] = 0
                metrics[f"Variance of in-zone rank ({group})"] = 0
                metrics[f"Variance of distance ({group})"] = 0

        # --- Theil's H segregation index ---
        metrics["Theil H"] = self.metric_theil()

        # Add diagnostic metrics
        diagnostic_metrics = self.eval_diagnostic_metrics()
        metrics.update(diagnostic_metrics)

        return pd.Series(metrics)

    def eval_diagnostic_metrics(self):
        """Compute diagnostic metrics for trend analysis.

        Returns:
            Dict[str, float]: Dictionary of diagnostic metrics including:
                - count_students_{ethnicity}: Count per ethnic group
                - count_students_Total: Total student count
                - enrollment_count_{ethnicity}: Assigned students per group
                - enrollment_rate_{ethnicity}: Assignment rate per group
                - utilization_{program_type}: Capacity utilization per type
        """
        metrics = {}
        student_data = self.student_data
        programs = self.programs

        # ===== Demographic Metrics =====
        # Use existing 'ethnicity' column (already standardized by map_ethnicity)
        if "ethnicity" in student_data.columns:
            # Student counts by ethnicity
            counts = student_data["ethnicity"].value_counts()
            for group, count in counts.items():
                metrics[f"count_students_{group}"] = count

            # Total students
            metrics["count_students_Total"] = len(student_data)

            # Enrolled (assigned) students
            assigned = student_data[student_data["programno"] > 0]
            enrolled_counts = assigned["ethnicity"].value_counts()
            for group, count in enrolled_counts.items():
                metrics[f"enrollment_count_{group}"] = count

            metrics["enrollment_count_Total"] = len(assigned)

            # Enrollment rates
            for group, total in counts.items():
                enrolled = enrolled_counts.get(group, 0)
                if total > 0:
                    metrics[f"enrollment_rate_{group}"] = enrolled / total

            total_students = len(student_data)
            if total_students > 0:
                metrics["enrollment_rate_Total"] = (
                    len(assigned) / total_students
                )

        # ===== Capacity Utilization Metrics =====
        if "programno" in student_data.columns and not programs.empty:
            # Count enrollment per program
            # Use programcodes from student_data to match program_id in programs
            if (
                "programcodes" in student_data.columns
                and "program_id" in programs.columns
            ):
                enrollment = (
                    student_data[student_data["programno"] > 0]
                    .groupby("programcodes")
                    .size()
                    .reset_index(name="enrollment")
                )
                # Ensure types match for merge
                enrollment["programcodes"] = enrollment["programcodes"].astype(
                    str
                )
                programs_copy = programs.copy()
                programs_copy["program_id"] = programs_copy[
                    "program_id"
                ].astype(str)
                stats = programs_copy.merge(
                    enrollment,
                    left_on="program_id",
                    right_on="programcodes",
                    how="left",
                )
            elif "programno" in programs.columns:
                enrollment = (
                    student_data[student_data["programno"] > 0]
                    .groupby("programno")
                    .size()
                    .reset_index(name="enrollment")
                )
                stats = programs.merge(
                    enrollment,
                    left_on="programno",
                    right_on="programno",
                    how="left",
                )
            else:
                # Cannot compute utilization without matching keys
                stats = pd.DataFrame()

            if not stats.empty:
                stats["enrollment"] = stats["enrollment"].fillna(0)
                total_cap = (
                    stats["capacity"].sum()
                    if "capacity" in stats.columns
                    else 0
                )
                if total_cap > 0:
                    metrics["utilization_rate_avg"] = (
                        stats["enrollment"] / stats["capacity"]
                    ).mean()

                # By program type
                type_col = (
                    "program_type"
                    if "program_type" in stats.columns
                    else "type"
                )
                if type_col in stats.columns:
                    for p_type, group in stats.groupby(type_col):
                        cap = group["capacity"].sum()
                        enr = group["enrollment"].sum()
                        if cap > 0:
                            metrics[f"utilization_{p_type}"] = enr / cap

        return metrics
