import glob
import os
import csv
import re
import string
from collections import defaultdict
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import geopandas as gpd
from shapely.geometry import Point

from Zone_Generation.Optimization_IP.generate_zones import (
    DesignZones,
    load_zones_from_file,
    ETHNICITY_DICT,
    K8_SCHOOLS,
)
from Graphic_Visualization.zone_viz_KLM import ZoneVisualizer

SPED_PROGRAMS = ["SA", "MS", "MM", "TC", "AF", "ED", "AO"]
ETHNICITY_COLS = [
    "resolved_ethnicity_American Indian",
    "resolved_ethnicity_Asian",
    "resolved_ethnicity_Black or African American",
    # "resolved_ethnicity_Decline to state",
    "resolved_ethnicity_Filipino",
    "resolved_ethnicity_Hispanic/Latinx",
    "resolved_ethnicity_Pacific Islander",
    "resolved_ethnicity_Two or More Races",
    "resolved_ethnicity_White",
]
CENSUS_TRANSLATOR_PATH = "~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv"
NEW_GUARDRAIL_PATH = "/Users/katherinementzer/SFUSD/Data/NewBlockData_10.21.21.xlsx"
CBEDS_SBAC_PATH = (
    "/Users/katherinementzer/SFUSD/Data/SFUSD_Demographics_SBAC_byblock.xlsx"
)
CENSUS_SHAPEFILE_PATH = os.path.expanduser(
    "~/SFUSD/Census 2010_ Blocks for San Francisco/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp"
)

FRL_MAP = {"1": 0.15, "2": 0.10}
RACIAL_MAP = {"1": 0.15, "2": 0.12}
NUM_ZONES_MAP = {"1": 6, "2": 7, "3": 8}
SCHOOL_QUALITY_MAP = {"1": "Yes", "2": "No"}
K8_MAP = {"1": "Yes", "2": "No"}


class DashboardCreator:
    def __init__(
        self,
        year: Union[int, list] = 21,
        capacity_scenario="A",
        level="BlockGroup",
        program_type="GE",
        drop_optout=True,
        new_schools=True,
        individual_years=False,
    ):
        self.year = year
        self.capacity_scenario = capacity_scenario
        self.level = level
        self.program_type = program_type
        self.drop_optout = drop_optout
        self.new_schools = new_schools
        self.individual_years = individual_years

        self.guardrail_data = self._load_guardrail_data()
        self.cbeds_data = self._load_cbeds_data()
        (
            self.area_data_k8,
            self.student_data,
            self.school_data_k8,
            self.year_area_data_k8,
            self.year_student_data,
        ) = self._load_area_data(include_k8=True)
        (
            self.area_data_no_k8,
            _,
            self.school_data_no_k8,
            self.year_area_data_no_k8,
            _,
        ) = self._load_area_data(include_k8=False)
        self.school_counts, self.district_avg = self._load_current_cbeds_data()
        (
            self.modality_zones,
            self.walk_time,
            self.transit_time,
        ) = self._load_transportation_data()
        self.distances = self._load_distance_data()
        with open("centroids.yaml", "r") as f:
            self.centroid_options = yaml.safe_load(f)

    def _load_guardrail_data(self):
        guardrail = pd.read_excel(
            NEW_GUARDRAIL_PATH,
            usecols=[
                "2010 Block ID",
                # "Pct_Bachelors_Degree",
                "Median_HH_Income",
                "Families in Poverty",
                "Total Population",
            ],
            sheet_name="NEW Block Database",
        )
        guardrail["Median_HH_Income"] = guardrail["Median_HH_Income"].apply(
            lambda x: int(str(x)[:-1].replace(",", "")) if str(x)[-1] == "+" else x
        )
        guardrail["Median_HH_Income"] = (
            guardrail["Median_HH_Income"].replace({"-": np.nan}).astype("Int64")
        )
        guardrail.rename(
            columns={
                "2010 Block ID": "census_block",
                "Total Population": "ses_total_count",
            },
            inplace=True,
        )
        guardrail = guardrail.groupby("census_block", as_index=False).mean()
        return guardrail

    def _load_cbeds_data(self):
        cbeds = pd.read_excel(CBEDS_SBAC_PATH, sheet_name="CBEDS2122")
        sbac = pd.read_excel(CBEDS_SBAC_PATH, sheet_name="SBAC1819")

        # block aalpi and frl
        cbeds.rename(
            columns={
                "FRPM by block": "frl_count",
                "Distinct count of Student No": "frl_total_count",
                "Geoid10": "census_block",
            },
            inplace=True,
        )
        cbeds = cbeds[["census_block", "frl_count", "frl_total_count"]]

        # L1 ttest scores
        sbac.rename(
            columns={
                "Geoid10 (Geo Export 417C2E4A-3D91-4F53-B352-A60394Fca356.Shp1)": "census_block",
                "Total L1 Tests": "L1_test_count",
                "Total Tests": "total_test_count",
            },
            inplace=True,
        )
        sbac = sbac[["census_block", "L1_test_count", "total_test_count"]]

        return cbeds.merge(sbac, how="outer", on="census_block")

    def _load_area_data(self, include_k8):
        translator = pd.read_csv(CENSUS_TRANSLATOR_PATH)
        translator.rename(
            columns={"Block": "census_block", "BlockGroup": "census_blockgroup"},
            inplace=True,
        )
        cbeds = self.cbeds_data.merge(translator, how="outer", on="census_block")
        cbeds = cbeds.merge(self.guardrail_data, how="outer", on="census_block")
        area_data = cbeds.groupby("census_blockgroup", as_index=False).sum()

        school = self._load_school_data(include_k8)
        school["school_name"] = (
            school[["census_blockgroup", "school_name"]]
            .groupby("census_blockgroup")["school_name"]
            .transform(lambda x: "\n".join(x))
        )
        school_area = school.groupby("census_blockgroup", as_index=False).sum()
        school_names = school.groupby("census_blockgroup").first()[["school_name"]]
        school_area = school_area.merge(
            school_names, how="left", left_on="census_blockgroup", right_index=True
        )
        area_data = area_data.merge(school_area, how="outer", on="census_blockgroup")

        area_data = self._add_all_school_aged_children_count(area_data, translator)

        year_area_data = {}
        year_student_data = {}
        if type(self.year) == int:
            student = self._load_student_data(year=self.year)
        else:
            student = pd.DataFrame()
            for y in self.year:
                student_year = self._load_student_data(year=y)
                year_student_data[y] = student_year
                student = student.append(student_year)

                student_area_year = student_year.groupby(
                    "census_blockgroup", as_index=False
                ).sum()
                year_area = area_data.merge(
                    student_area_year, how="outer", on="census_blockgroup"
                )
                year_area_data[y] = year_area.copy()
        student_area = student.groupby("census_blockgroup", as_index=False).sum()
        area_data = area_data.merge(student_area, how="outer", on="census_blockgroup")
        area_data.to_csv("~/Desktop/area_data.csv")
        exit()

        return area_data, student, school, year_area_data, year_student_data

    def _add_all_school_aged_children_count(self, area_data, translator):
        census_student = pd.read_excel(
            "/Users/katherinementzer/Downloads/ACS 5yr 2019 Child Count by Tract.xls",
            header=1,
            usecols=["id", "Draft Estimate"],
        )
        census_student.loc[:, "Tract"] = census_student["id"].apply(
            lambda x: int(x[9:])
        )

        num_blockgroups = translator.groupby(
            "census_blockgroup", as_index=False
        ).first()
        num_blockgroups = (
            area_data[["census_blockgroup"]]
            .merge(num_blockgroups, how="left", on="census_blockgroup")
            .groupby("Tract")
            .count()
        )
        census_student = census_student.merge(
            num_blockgroups, left_on="Tract", right_index=True
        )

        census_student["all_school_aged_children"] = (
            census_student["Draft Estimate"] / census_student["census_block"]
        )
        census_student = census_student[["all_school_aged_children", "Tract"]]
        census_student = (
            translator.groupby("census_blockgroup", as_index=False)
            .first()
            .merge(census_student, how="left", on="Tract")
        )
        area_data = area_data.merge(census_student, how="left", on="census_blockgroup")
        return area_data

    @staticmethod
    def _make_program_type_lists(df):
        """ create column with each type of program applied to """
        for round in range(1, 4):
            col = "r{}_programs".format(round)
            if col in df.columns:
                df[col] = df[col].fillna("[]")
                df[col] = df[col].apply(lambda x: eval(x))
                if round == 1:
                    df["program_types"] = df[col]
                else:
                    df["program_types"] = df["program_types"] + df[col]
        df["program_types"] = df["program_types"].apply(lambda x: np.unique(x))
        return df

    def _load_student_data(self, year=21):
        # get student data
        student_data = pd.read_csv(
            f"~/SFUSD/Data/Cleaned/enrolled_{year}{year + 1}.csv", low_memory=False
        )
        student_data = student_data.loc[student_data["grade"] == "KG"]
        student_data.loc[:, "num_with_ethnicity"] = student_data[
            "resolved_ethnicity"
        ].apply(lambda x: 0 if pd.isna(x) else 1)
        student_data["resolved_ethnicity"] = student_data["resolved_ethnicity"].replace(
            ETHNICITY_DICT
        )
        # student_data = self._make_program_type_lists(student_data)

        if year in [21, 22]:
            student_data.loc[:, "ell"] = student_data["englprof"].apply(
                lambda x: 1 if x in ["N", "L"] else 0
            )
            student_data.loc[:, "ell_total_count"] = student_data["englprof"].apply(
                lambda x: 1 if not pd.isna(x) else 0
            )
        else:
            student_data.loc[:, "sped"] = np.where(
                student_data["speced"] == "Yes", 1, 0
            )
            student_data.loc[:, "ell"] = student_data["englprof_desc"].apply(
                lambda x: 1 if (x == "N-Non English" or x == "L-Limited English") else 0
            )
            student_data.loc[:, "ell_total_count"] = student_data[
                "englprof_desc"
            ].apply(lambda x: 1 if not pd.isna(x) else 0)

        student_data = pd.get_dummies(student_data, columns=["resolved_ethnicity"])

        for col in ETHNICITY_COLS:
            if col not in student_data.columns:
                student_data[col] = 0

        student_data["all_students"] = 1
        student_data.loc[:, "enrolled_students"] = np.where(
            student_data["enrolled_idschool"].isna(), 0, 1
        )

        for rnd in range(2, 6):
            if f"r{rnd}_programs" in student_data.columns:
                student_data["r1_programs"] = student_data["r1_programs"].fillna(
                    student_data[f"r{rnd}_programs"]
                )
        student_data.r1_programs = student_data.r1_programs.apply(
            lambda x: eval(x) if not pd.isna(x) else []
        )
        student_data["enrolled_and_ge_applied"] = student_data.apply(
            lambda x: sum([i == "GE" for i in x["r1_programs"]]) / len(x["r1_programs"])
            if x["enrolled_students"] == 1 and len(x["r1_programs"]) > 0
            else 0,
            axis=1,
        )
        student_data = student_data[
            [
                "census_blockgroup",
                "all_students",
                "enrolled_students",
                "enrolled_and_ge_applied",
                "num_with_ethnicity",
                "sped",
                "ell",
                "ell_total_count",
                "enrolled_idschool",
            ]
            + ETHNICITY_COLS
        ]

        return student_data.rename(columns={"sped": "sped_count", "ell": "ell_count"})

    def _load_school_data(self, include_k8):
        sc_df = pd.read_csv(
            "~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv"
        )
        sc_df.rename(columns={"BlockGroup": "census_blockgroup"}, inplace=True)
        sc_df["K-8"] = sc_df["school_id"].apply(lambda x: 1 if x in K8_SCHOOLS else 0)

        change_scores = pd.read_csv(
            "/Users/katherinementzer/Dropbox/SFUSD/Data/State Accountability Dashboard Scores 2019 - 3yr Stanford Sheet.csv",
            usecols=["School Code", "All Students"],
        )
        change_scores.rename(
            columns={"School Code": "school_id", "All Students": "change_score"},
            inplace=True,
        )
        sc_df = sc_df.merge(change_scores, how="left", on="school_id")
        sc_df.loc[:, "dashboard_school_count"] = sc_df["change_score"].apply(
            lambda x: 1 if not pd.isna(x) else 0
        )

        cap_data = pd.read_csv("~/Dropbox/SFUSD/Data/School_Capacity/stanford_capacities_12.23.21.csv")
        cap_data.rename(
            columns={
                "SchNum": "school_id",
                "PathwayCode": "program_type",
                f"Scenario_{self.capacity_scenario}_Capacity": "r3_capacity",
            },
            inplace=True,
        )

        prog_ge = cap_data.loc[cap_data["program_type"] == "GE"][
            ["school_id", "r3_capacity"]
        ].rename(columns={"r3_capacity": "ge_capacity"})
        prog_all = cap_data.loc[~cap_data["program_type"].isin(SPED_PROGRAMS)][
            ["school_id", "r3_capacity"]
        ].rename(columns={"r3_capacity": "all_nonsped_cap"})

        tk5_cap = pd.read_csv(
            "~/Dropbox/SFUSD/Data/School_Capacity/stanford_capacities_12.23.21_TK-5.csv",
            usecols=["School ID", "School Capacity"],
        )
        tk5_cap.rename(
            columns={"School ID": "school_id", "School Capacity": "TK-5_capacity"},
            inplace=True,
        )

        prog_all = prog_all.groupby("school_id", as_index=False).sum()
        prog_ge = prog_ge.groupby("school_id", as_index=False).sum()
        sc_df = sc_df.merge(prog_all, how="left", on="school_id")
        sc_df = sc_df.merge(prog_ge, how="left", on="school_id")
        sc_df = sc_df.merge(tk5_cap, how="left", on="school_id")
        sc_df.loc[:, "ge_schools"] = sc_df["ge_capacity"].apply(
            lambda x: 1 if (not pd.isna(x) and x > 0) else 0
        )
        sc_df["all_schools"] = 1
        self.ge_schools_with_k8 = list(sc_df.loc[sc_df["ge_schools"] == 1]["school_id"])
        self.ge_schools_no_k8 = list(
            sc_df.loc[sc_df["ge_schools"] == 1].loc[sc_df["K-8"] == 0]["school_id"]
        )

        sc_df = sc_df[
            [
                "school_id",
                "school_name",
                "census_blockgroup",
                "all_nonsped_cap",
                "ge_capacity",
                "TK-5_capacity",
                "all_schools",
                "ge_schools",
                "K-8",
                "AvgColorIndex",
                "change_score",
                "dashboard_school_count",
            ]
        ]
        if not include_k8:
            return sc_df.loc[sc_df["K-8"] == 0]
        return sc_df

    def _load_distance_data(self):
        save_path = "~/Dropbox/SFUSD/Optimization/block2block_distances.csv"
        distances = pd.read_csv(save_path, index_col=self.level)
        distances.columns = [str(int(float(x))) for x in distances.columns]
        return distances

    @staticmethod
    def _load_transportation_data():
        maps = {}
        for time in [20, 30]:
            for mode in ["walk", "bike", "transit"]:
                with open(
                    os.path.expanduser(
                        f"~/Dropbox/SFUSD/Data/Cleaned/bgs_within_{time}_{mode}_time_1.4_adjusted.yaml"
                    ),
                    "r",
                ) as f:
                    maps[f"{mode}_{time}"] = yaml.safe_load(f)
        walk = pd.read_csv(
            "~/Dropbox/SFUSD/Data/Cleaned/bg_to_school_walk_time.csv",
            index_col="origin_bg",
        )
        transit = pd.read_csv(
            "~/Dropbox/SFUSD/Data/Cleaned/bg_to_school_transit_time.csv",
            index_col="origin_bg",
        )
        return maps, walk, transit

    def _add_frl_and_census_blocks(self, st_df):
        st_df.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)
        st_df["latitude"].replace({0.0: np.nan}, inplace=True)
        st_df["longitude"].replace({0.0: np.nan}, inplace=True)
        st_df.dropna(subset=["latitude", "longitude"], inplace=True)

        block_sf = gpd.read_file(CENSUS_SHAPEFILE_PATH)
        block_sf["geoid10"] = (
            block_sf["geoid10"].fillna(value=0).astype("int64", copy=False)
        )
        # make geo data frame
        geometry = [Point(xy) for xy in zip(st_df["longitude"], st_df["latitude"])]
        st_df = gpd.GeoDataFrame(st_df, crs="epsg:4326", geometry=geometry)
        st_df = gpd.sjoin(
            st_df, block_sf[["geometry", "geoid10"]], how="left", op="intersects"
        )

        # get frl data
        st_df = st_df.merge(
            self.cbeds_data[["census_block", "frl_count", "frl_total_count"]],
            how="left",
            left_on="geoid10",
            right_on="census_block",
        )
        st_df["frl"] = st_df.frl_count / st_df.frl_total_count
        st_df.drop(columns=["geoid10", "frl_count", "frl_total_count"], inplace=True)

        # add census blockgroup and tract
        census = pd.read_csv(CENSUS_TRANSLATOR_PATH)
        census.rename(
            columns={
                "Block": "census_block",
                "BlockGroup": "census_blockgroup",
                "Tract": "census_tract",
            },
            inplace=True,
        )
        census = census.drop_duplicates()
        st_df = st_df.merge(census, how="left", on="census_block")
        st_df.drop(columns=["index_right", "geometry"], inplace=True)
        return st_df

    def _load_current_cbeds_data(self):
        if os.path.isfile(
            os.path.expanduser("~/SFUSD/Data/Cleaned/current_students_2122.csv")
        ):
            df = pd.read_csv("~/SFUSD/Data/Cleaned/current_students_2122.csv")
        else:
            df = pd.read_csv(
                "/Users/katherinementzer/SFUSD/Data/Student and family data/cbeds_2122.csv"
            )
            df.rename(columns={"race": "resolved_ethnicity"}, inplace=True)
            df["resolved_ethnicity"].replace(ETHNICITY_DICT, inplace=True)
            df = pd.get_dummies(df, columns=["resolved_ethnicity"])
            df = self._add_frl_and_census_blocks(df)
            df["num_students_w_frl"] = df.frl.apply(
                lambda x: 1 if not pd.isna(x) else 0
            )
            df.loc[:, "ell"] = df["englprof"].apply(
                lambda x: 1 if x in ["N", "L"] else (0 if not pd.isna(x) else np.nan)
            )
            df.loc[:, "sped"] = np.where(df["sped"] == "Y", 1, 0)
            df[
                [
                    "schno",
                    "ell",
                    "frl",
                    "num_students_w_frl",
                    "sped",
                    "census_blockgroup",
                ]
                + ETHNICITY_COLS
            ].to_csv("~/SFUSD/Data/Cleaned/current_students_2122.csv", index=False)

        df.loc[:, "num_students_w_frl"] = df["frl"].apply(
            lambda x: 1 if not pd.isna(x) else 0
        )
        df.loc[:, "num_students"] = 1
        all_eth_cols = [i for i in df.columns if re.match("resolved_ethnicity*", i)]
        df.loc[:, "num_with_ethnicity"] = df.apply(
            lambda x: 1 if sum(x[all_eth_cols]) > 0 else 0, axis=1
        )
        school_counts = df.groupby("schno").sum()

        l1_tests = pd.read_csv(
            "~/Dropbox/SFUSD/Data/school_l1_test_scores.csv", index_col="school_id"
        )
        school_counts = school_counts.join(l1_tests, how="left")
        # school_counts["L1_test%"] = (school_counts.math_l1_tests + school_counts.ela_l1_tests)/ (school_counts.num_math_tests + school_counts.num_ela_tests)
        # school_counts["frl%"] = school_counts.frl / school_counts.num_students_w_frl
        # school_counts["ell%"] = school_counts.ell / school_counts.num_students
        # school_counts["sped%"] = school_counts.sped/ school_counts.num_students

        ethnicities_short = [x.split("_")[-1] + "%" for x in ETHNICITY_COLS]
        for long, short in zip(ETHNICITY_COLS, ethnicities_short):
            school_counts[short] = (
                school_counts[long] / school_counts["num_with_ethnicity"]
            )

        sch2bg = dict(
            zip(self.school_data_k8.school_id, self.school_data_k8.census_blockgroup)
        )
        school_counts["census_blockgroup"] = [
            sch2bg[x] if x in sch2bg else np.nan for x in school_counts.index
        ]
        school_counts.dropna(subset=["census_blockgroup"], inplace=True)

        l1_pct = (
            self.cbeds_data.L1_test_count.sum() / self.cbeds_data.total_test_count.sum()
        )

        district_avg = df[
            ["frl", "num_students_w_frl", "ell", "sped", "num_students"]
            + ETHNICITY_COLS
        ].sum(axis=0)
        # district_avg = df[
        #     ["frl%", "ell%", "L1_test_count%", "sped%"] + ETHNICITY_COLS
        # ].sum(axis=0)
        district_avg["frl%"] = district_avg.frl / district_avg.num_students_w_frl
        district_avg["ell%"] = district_avg.ell / district_avg["num_students"]
        district_avg["sped%"] = district_avg.sped / district_avg["num_students"]

        # print(district_avg["num_students"], district_avg)
        district_avg = pd.concat([district_avg, pd.Series({"L1_test%": l1_pct})])

        # print(district_avg)
        eth_sum = district_avg[ETHNICITY_COLS].sum()
        for col in ETHNICITY_COLS:
            district_avg[col.split("_")[-1] + "%"] = district_avg[col] / eth_sum
        # print(district_avg)
        # exit()
        return school_counts, district_avg

    def _calculate_metrics_category(self, df, metrics_fn):
        all_zone_metrics = pd.DataFrame()
        for zone, areas in df.groupby("zone_id"):
            proximity_metrics = metrics_fn(areas)
            zone_metrics = pd.DataFrame(proximity_metrics, index=[zone])
            all_zone_metrics = all_zone_metrics.append(zone_metrics)
        return all_zone_metrics

    def _proximity_metrics(self, args):
        areas = args
        students = areas.loc[areas["all_students"] > 0][
            ["census_blockgroup", "all_students"]
        ].rename(columns={"census_blockgroup": "student_blockgroup"})
        students["key"] = 0
        schools = areas.loc[areas["ge_schools"] > 0][
            ["census_blockgroup", "ge_schools"]
        ].rename(columns={"census_blockgroup": "school_blockgroup"})
        schools["key"] = 0
        pairs = students.merge(schools, how="outer", on="key")
        pairs.loc[:, "dist"] = pairs.apply(
            lambda x: self.distances.loc[
                x["student_blockgroup"], str(int(x["school_blockgroup"]))
            ],
            axis=1,
        )
        pairs.loc[:, "within_1mi"] = pairs["dist"].apply(lambda x: 1 if x <= 1 else 0)

        metrics = {}
        metrics["avg_num_ge_within_1mi"] = (
            sum(pairs["within_1mi"] * pairs["all_students"])
            / students["all_students"].sum()
        )
        metrics["max_min_dist"] = (
            pairs.groupby("student_blockgroup").min()["dist"].max()
        )
        metrics["avg_min_dist"] = (
            pairs.groupby("student_blockgroup").min()["dist"].mean()
        )
        metrics["max_max_dist"] = (
            pairs.groupby("student_blockgroup").max()["dist"].max()
        )
        metrics["avg_max_dist"] = (
            pairs.groupby("student_blockgroup").max()["dist"].mean()
        )
        return metrics

    def _disruption_metrics(self, zone_dict, include_k8, specific_year=None):
        # percentage of students currently attending school in this zone
        if specific_year is None:
            stu = self.student_data
        else:
            stu = self.year_student_data[specific_year]
        stu.loc[:, "home_zone"] = stu["census_blockgroup"].replace(zone_dict)
        sch = self.school_data_k8 if include_k8 else self.school_data_no_k8
        sch["school_zone"] = sch["census_blockgroup"].replace(zone_dict)
        df = stu.merge(
            sch[["school_id", "school_zone"]],
            how="left",
            left_on="enrolled_idschool",
            right_on="school_id",
        )
        if not include_k8:
            df.loc[:, "school_zone"] = df.apply(
                lambda x: x["home_zone"]
                if x["enrolled_idschool"] in K8_SCHOOLS
                else x["school_zone"],
                axis=1,
            )

        movement = pd.pivot_table(
            df,
            values="all_students",
            index="home_zone",
            columns="school_zone",
            aggfunc="sum",
            fill_value=0,
        )
        movement.loc[:, :] = movement.loc[:, :].div(movement.sum(axis=1), axis=0)

        # pad columns for dashboard formatting
        if len(movement.columns < 8):
            for i in range(len(movement.columns), 8):
                movement.loc[:, i] = 0
        movement.columns = [f"school_zone_{x}" for x in movement.columns]
        return movement

    def _transportation_metrics(self, zone_dict):
        sch2bg = dict(
            zip(self.school_data_k8.school_id, self.school_data_k8.census_blockgroup)
        )
        bg2num_students = dict(
            zip(
                self.area_data_k8.census_blockgroup,
                self.area_data_k8.enrolled_students.fillna(0),
            )
        )
        n = self.area_data_k8.enrolled_students.sum()
        metrics = {}

        for time in [20, 30]:
            for mode in ["walk", "bike", "transit"]:
                preserved = 0
                sch_student_modality_pairs = 0
                options_per_bg = defaultdict(lambda: 0)
                for sch, block_groups in self.modality_zones[f"{mode}_{time}"].items():
                    sch_zone = zone_dict[sch2bg[sch]]
                    blocks_in_zone = [
                        x for x in block_groups if zone_dict[x] == sch_zone
                    ]
                    for bg in blocks_in_zone:
                        options_per_bg[bg] += 1
                    preserved += sum([bg2num_students[x] for x in blocks_in_zone])
                    sch_student_modality_pairs += sum(
                        [bg2num_students[x] for x in block_groups]
                    )

                metrics[f"%_{time}_{mode}_bgs_preserved_adjusted"] = (
                    float(preserved) / sch_student_modality_pairs
                )
                for i in range(3):
                    if i == 2:
                        options = sum(
                            [
                                bg2num_students[bg]
                                for bg in zone_dict.keys()
                                if options_per_bg[bg] >= i
                            ]
                        )
                    else:
                        options = sum(
                            [
                                bg2num_students[bg]
                                for bg in zone_dict.keys()
                                if options_per_bg[bg] == i
                            ]
                        )
                    metrics[
                        f"%_with_{i if i < 2 else '2+'}_{time}min_{mode}_options_adjusted"
                    ] = (options / n)
                metrics[f"num_options_in_{time}mins_{mode}_adjusted"] = (
                    sum(
                        [
                            bg2num_students[bg] * num
                            for bg, num in options_per_bg.items()
                        ]
                    )
                    / n
                )

        return pd.Series(metrics)

    def _zone_level_transportation_metrics(self, df, include_k8, zone_dict):
        students = df.loc[df["all_students"] > 0][
            ["census_blockgroup", "all_students", "zone_id"]
        ].rename(columns={"census_blockgroup": "student_blockgroup"})
        schools = self.school_data_k8 if include_k8 else self.school_data_no_k8
        schools = schools.loc[schools["ge_capacity"] > 0][
            ["school_id", "census_blockgroup"]
        ]
        schools.loc[:, "zone_id"] = schools["census_blockgroup"].replace(zone_dict)
        pairs = students.merge(schools, how="inner", on="zone_id")
        pairs["walk_time"] = pairs.apply(
            lambda x: self.walk_time.loc[
                int(x["student_blockgroup"]), str(int(x["school_id"]))
            ]
            if (int(x["student_blockgroup"]) != 60750179021)
            else np.nan,
            axis=1,
        )

        # pairs.loc[pairs.student_blockgroup == 60750179021, ['walk_time']] = np.nan
        # pairs.loc[pairs.student_blockgroup == 60750179021, ['transit_time']] = np.nan
        max_walk = (
            pairs.groupby("student_blockgroup")
            .max()
            .rename(columns={"walk_time": "avg_max_walk_time"})
        )
        max_walk["avg_max_walk_time"] = (
            max_walk["avg_max_walk_time"] * max_walk["all_students"]
        )
        max_walk = max_walk.groupby("zone_id").sum()
        max_walk["avg_max_walk_time"] = (
            max_walk["avg_max_walk_time"] / max_walk["all_students"]
        )

        avg_walk = (
            pairs.groupby("student_blockgroup")
            .mean()
            .rename(columns={"walk_time": "avg_walk_time"})
        )
        avg_walk["avg_walk_time"] = avg_walk["avg_walk_time"] * avg_walk["all_students"]
        avg_walk = avg_walk.groupby("zone_id").sum()
        avg_walk["avg_walk_time"] = avg_walk["avg_walk_time"] / avg_walk["all_students"]
        return max_walk[["avg_max_walk_time"]].join(avg_walk[["avg_walk_time"]])

    def _supplementary_zone_level_transportation_metrics(
        self, df, include_k8, zone_dict, walk_multiplier=1.3
    ):
        students = df.loc[df["all_students"] > 0][
            ["census_blockgroup", "all_students", "zone_id"]
        ].rename(columns={"census_blockgroup": "student_blockgroup"})
        schools = self.school_data_k8 if include_k8 else self.school_data_no_k8
        schools = schools.loc[schools["ge_capacity"] > 0][
            ["school_id", "census_blockgroup"]
        ]
        schools.loc[:, "zone_id"] = schools["census_blockgroup"].replace(zone_dict)
        pairs = students.merge(schools, how="inner", on="zone_id")
        pairs["walk_time"] = pairs.apply(
            lambda x: self.walk_time.loc[
                int(x["student_blockgroup"]), str(int(x["school_id"]))
            ],  # if (int(x["student_blockgroup"]) != 60750179021) else np.nan,
            axis=1,
        )
        pairs["transit_time"] = pairs.apply(
            lambda x: self.transit_time.loc[
                int(x["student_blockgroup"]), str(int(x["school_id"]))
            ],  # if (int(x["student_blockgroup"]) != 60750179021) else np.nan,
            axis=1,
        )
        pairs.loc[:, "30min_walk_adjusted"] = pairs["walk_time"].apply(
            lambda x: 1 if x <= 30 / walk_multiplier else 0
        )
        pairs.loc[:, "30min_transit"] = pairs["transit_time"].apply(
            lambda x: 1 if x <= 30 else 0
        )
        pairs.loc[:, "20min_walk_adjusted"] = pairs["walk_time"].apply(
            lambda x: 1 if x <= 20 / walk_multiplier else 0
        )
        pairs.loc[:, "20min_transit"] = pairs["transit_time"].apply(
            lambda x: 1 if x <= 20 else 0
        )

        num_options = pairs.groupby("student_blockgroup", as_index=False).sum()[
            [
                "student_blockgroup",
                "30min_walk_adjusted",
                "30min_transit",
                "20min_walk_adjusted",
                "20min_transit",
            ]
        ]
        num_options = num_options.merge(students, how="left", on="student_blockgroup")
        num_options["2+_30min_walk_adjusted"] = num_options[
            "30min_walk_adjusted"
        ].apply(lambda x: 1 if x >= 2 else 0)
        num_options["2+_30min_transit"] = num_options["30min_transit"].apply(
            lambda x: 1 if x >= 2 else 0
        )
        num_options["2+_20min_walk_adjusted"] = num_options[
            "20min_walk_adjusted"
        ].apply(lambda x: 1 if x >= 2 else 0)
        num_options["2+_20min_transit"] = num_options["20min_transit"].apply(
            lambda x: 1 if x >= 2 else 0
        )
        columns = [
            "2+_30min_walk_adjusted",
            "2+_30min_transit",
            "30min_walk_adjusted",
            "30min_transit",
            "2+_20min_walk_adjusted",
            "2+_20min_transit",
            "20min_walk_adjusted",
            "20min_transit",
        ]
        for col in columns:
            num_options[col] *= num_options["all_students"]
        num_options = num_options.groupby("zone_id").sum()
        for col in columns:
            num_options[col] /= num_options["all_students"]
        num_options = num_options[columns]

        pairs.loc[pairs.student_blockgroup == 60750179021, ["walk_time"]] = np.nan
        pairs.loc[pairs.student_blockgroup == 60750179021, ["transit_time"]] = np.nan
        max_walk = (
            pairs.groupby("student_blockgroup")
            .max()
            .rename(columns={"walk_time": "avg_max_walk_time"})
        )
        max_walk["avg_max_walk_time"] = (
            max_walk["avg_max_walk_time"] * max_walk["all_students"]
        )
        max_walk = max_walk.groupby("zone_id").sum()
        max_walk["avg_max_walk_time"] = (
            max_walk["avg_max_walk_time"] / max_walk["all_students"]
        )
        avg_walk = (
            pairs.groupby("student_blockgroup")
            .mean()
            .rename(columns={"walk_time": "avg_walk_time"})
        )
        avg_walk["avg_walk_time"] = avg_walk["avg_walk_time"] * avg_walk["all_students"]
        avg_walk = avg_walk.groupby("zone_id").sum()
        avg_walk["avg_walk_time"] = avg_walk["avg_walk_time"] / avg_walk["all_students"]
        avg_max = max_walk[["avg_max_walk_time"]].join(avg_walk[["avg_walk_time"]])
        avg_max["avg_max_walk_adjusted"] = (
            avg_max["avg_max_walk_time"] * walk_multiplier
        )
        avg_max["avg_walk_adjusted"] = avg_max["avg_walk_time"] * walk_multiplier

        return num_options.join(avg_max)

    def _dissimilarity_metrics(self, df):
        zone_eths = df.groupby("zone_id").sum()[ETHNICITY_COLS]
        zone_total = zone_eths.sum(axis=1)
        ethnicity_total = zone_eths.sum(axis=0)

        ethnicity_dissimilarity = {}
        dissimilarity_total = 0
        for eth in ETHNICITY_COLS:
            zone_eth_distn = zone_eths[eth]
            uniform_distn = zone_total * (
                ethnicity_total[eth] / df["num_with_ethnicity"].sum()
            )
            difference_norm = np.linalg.norm(zone_eth_distn - uniform_distn, 1)
            dissimilarity_total += difference_norm / 2
            eth_name = eth.split("_")[-1] + "_dissimilarity"
            ethnicity_dissimilarity[eth_name] = (difference_norm / 2) / ethnicity_total[
                eth
            ]
        metrics = ethnicity_dissimilarity
        metrics["dissimilarity"] = dissimilarity_total / df["num_with_ethnicity"].sum()

        zone_grouped = df.groupby("zone_id").sum()
        for col, col_total in [
            ("frl_count", "frl_total_count"),
            ("sped_count", "all_students"),
            ("ell_count", "ell_total_count"),
            ("L1_test_count", "total_test_count"),
        ]:
            zone_frl = zone_grouped[col]
            zone_total = zone_grouped[col_total]
            uniform_col = zone_total * (df[col].sum() / df[col_total].sum())
            difference_norm = np.linalg.norm(zone_frl - uniform_col, 1)
            if col == "frl_count":
                metrics[f"{col.split('_')[0]}_dissimilarity"] = (
                    difference_norm / df[col_total].sum()
                )
            else:
                metrics[f"{col.split('_')[0]}_dissimilarity"] = (
                    difference_norm / 2
                ) / df[col].sum()
        return pd.Series(metrics)

    def _comparison_to_current_schools(self, df, zone_dict, include_k8):
        if include_k8:
            school_counts = self.school_data_k8[["school_id"]].merge(
                self.school_counts, how="left", left_on="school_id", right_index=True
            )
        else:
            school_counts = self.school_data_no_k8[["school_id"]].merge(
                self.school_counts, how="left", left_on="school_id", right_index=True
            )
        school_counts["zone_id"] = school_counts.census_blockgroup.replace(zone_dict)
        zone_enrollment = school_counts.groupby("zone_id").sum()

        ethnicities_short = [x.split("_")[-1] + "%" for x in ETHNICITY_COLS]
        for long, short in zip(ETHNICITY_COLS, ethnicities_short):
            zone_enrollment[short] = (
                zone_enrollment[long] / zone_enrollment["num_with_ethnicity"]
            )

        zone_enrollment["L1_test%"] = (
            zone_enrollment.math_l1_tests + zone_enrollment.ela_l1_tests
        ) / (zone_enrollment.num_math_tests + zone_enrollment.num_ela_tests)
        zone_enrollment["frl%"] = (
            zone_enrollment.frl / zone_enrollment.num_students_w_frl
        )
        zone_enrollment["ell%"] = zone_enrollment.ell / zone_enrollment.num_students
        zone_enrollment["sped%"] = zone_enrollment.sped / zone_enrollment.num_students

        zone_enrollment = zone_enrollment[
            ["frl%", "ell%", "sped%", "L1_test%"] + ethnicities_short
        ]
        # print("Zone enrollment", zone_enrollment)
        # print(zone_enrollment.columns)
        district_avg = self.district_avg[
            ["frl%", "ell%", "sped%", "L1_test%"] + ethnicities_short
        ]  # / len(
        #     zone_enrollment
        # )
        # df1 = pd.DataFrame(df.all_students / df.all_students.sum())
        zone_weights = df.all_students / df.all_students.sum()
        # print(df1.shape)
        # df2 = pd.DataFrame(district_avg)
        # print(df2.T.shape)
        # district_avg = pd.DataFrame(np.outer(zone_weights, district_avg), index=zone_weights.index, columns=district_avg.index)
        # print("district avg:\n", district_avg)
        # print("zone_enrollment\n", zone_enrollment)
        enrollment_diff = zone_enrollment.subtract(district_avg)
        enrollment_diff = enrollment_diff.mul(zone_weights, axis=0)
        # print(enrollment_diff)
        enrollment_frl_diff = (
            np.linalg.norm(enrollment_diff["frl%"], 1)
            # / school_counts.num_students_w_frl.sum()
        )
        enrollment_ell_diff = np.linalg.norm(
            enrollment_diff["ell%"], 1
        )  # / school_counts.num_students.sum()
        enrollment_sped_diff = np.linalg.norm(
            enrollment_diff["sped%"], 1
        )  # / school_counts.num_students.sum()
        enrollment_l1_diff = np.linalg.norm(enrollment_diff["L1_test%"], 1)
        enrollment_race_diff = (
            np.linalg.norm(enrollment_diff[ethnicities_short], 1) / 2
        )  # / df[ethnicities_short].sum().sum()
        # print('zone:\n',df[["frl%", "ell%", "sped%", "L1_test%", "all_students"] + ethnicities_short])

        zone_diff = (
            df[
                ["frl%", "ell%", "sped%", "L1_test%", "all_students"]
                + ethnicities_short
            ]
            # .rename(columns={"ell_count": "ell", "frl_count": "frl"})
            .subtract(district_avg)
        )
        zone_diff = zone_diff.mul(zone_weights, axis=0)
        # print(zone_diff)
        # print(zone_diff.mul(df.all_students/df.all_students.sum(), axis=0))
        # print(df.all_students/df.all_students.sum())
        zone_frl_diff = np.linalg.norm(
            zone_diff["frl%"], 1
        )  # / school_counts.num_students_w_frl.sum()
        zone_ell_diff = np.linalg.norm(
            zone_diff["ell%"], 1
        )  # / school_counts.num_students.sum()
        zone_sped_diff = np.linalg.norm(
            zone_diff["sped%"], 1
        )  # / school_counts.num_students.sum()
        zone_l1_diff = np.linalg.norm(zone_diff["L1_test%"], 1)
        zone_race_diff = np.linalg.norm(zone_diff[ethnicities_short], 1) / 2  # / df[
        #     ETHNICITY_COLS
        # ].sum().sum()
        metrics = {
            "frl_improvement": enrollment_frl_diff - zone_frl_diff,
            "ell_improvement": enrollment_ell_diff - zone_ell_diff,
            "sped_improvement": enrollment_sped_diff - zone_sped_diff,
            "l1_improvement": enrollment_l1_diff - zone_l1_diff,
            "race_improvement": enrollment_race_diff - zone_race_diff,
        }
        # print(metrics)
        return pd.Series(metrics)

    def zone_metrics(
        self, zone_file, include_k8, all_students=True, specific_year=None
    ):
        df = self.get_appropriate_area_data(all_students, include_k8, specific_year)

        zone_dict = load_zones_from_file(zone_file)
        df["zone_id"] = df["census_blockgroup"].replace(zone_dict)
        proximity = self._calculate_metrics_category(df, self._proximity_metrics)
        disruption = self._disruption_metrics(
            zone_dict, include_k8, specific_year=specific_year
        )
        zone_transportation = self._zone_level_transportation_metrics(
            df, include_k8, zone_dict
        )
        # violations = self._capacity_violation_metrics(df, include_k8, specific_year, zone_dict)
        ge_var = self._student_count_variability_metrics(df, specific_year, zone_dict)

        ethnicities_short = [x.split("_")[-1] + "%" for x in ETHNICITY_COLS]
        names = df.groupby("zone_id")["school_name"].apply(
            lambda x: "\n".join([i for i in x if not pd.isna(i)])
        )

        metrics = df.groupby("zone_id").sum()
        metrics["frl%"] = metrics["frl_count"] / metrics["frl_total_count"]
        metrics["sped%"] = metrics["sped_count"] / metrics["all_students"]
        metrics["ell%"] = metrics["ell_count"] / metrics["ell_total_count"]
        metrics["L1_test%"] = metrics["L1_test_count"] / metrics["total_test_count"]
        metrics["median_hh_income"] = (
            metrics["Median_HH_Income"] / metrics["ses_total_count"]
        )
        metrics["poverty%"] = (
            metrics["Families in Poverty"] / metrics["ses_total_count"]
        )
        metrics["AvgColorIndex"] = (
            metrics["AvgColorIndex"] / metrics["dashboard_school_count"]
        )
        metrics["change_score"] = (
            metrics["change_score"] / metrics["dashboard_school_count"]
        )
        for long, short in zip(ETHNICITY_COLS, ethnicities_short):
            metrics[short] = metrics[long] / metrics["num_with_ethnicity"]

        current_sch_comparison = self._comparison_to_current_schools(
            metrics, zone_dict, include_k8
        )

        if type(self.year) is list and specific_year is None:
            for col in ["all_students", "enrolled_students", "enrolled_and_ge_applied"]:
                metrics[col] = metrics[col] / len(self.year)

        zone_level_metrics = (
            metrics[
                [
                    "all_schools",
                    "ge_schools",
                    "K-8",
                    "AvgColorIndex",
                    "change_score",
                    "TK-5_capacity",
                    "all_nonsped_cap",
                    "ge_capacity",
                    "all_school_aged_children",
                    "all_students",
                    "enrolled_students",
                    "enrolled_and_ge_applied",
                    "frl%",
                    "sped%",
                    "ell%",
                    "L1_test%",
                ]
                + ethnicities_short
            ]
            .join(proximity)
            .join(disruption)
            .join(names)
            .join(ge_var)
            .join(zone_transportation)
        )

        transportation = self._transportation_metrics(zone_dict)
        dissimilarity = self._dissimilarity_metrics(df)
        overall_metrics = pd.concat(
            [transportation, dissimilarity, current_sch_comparison]
        )

        return zone_level_metrics, overall_metrics

    def _capacity_violation_metrics(self, df, include_k8, specific_year, zone_dict):
        if type(self.year) is list and not specific_year:
            violations = pd.DataFrame()
            for y in self.year:
                year_df = (
                    self.year_area_data_k8[y]
                    if include_k8
                    else self.year_area_data_no_k8[y]
                )
                year_df.loc[:, "zone_id"] = year_df["census_blockgroup"].replace(
                    zone_dict
                )
                zone_year = year_df.groupby("zone_id", as_index=False).sum()
                violations = violations.append(zone_year)
            violations[">20%_shortage"] = violations.apply(
                lambda x: 1
                if x.ge_capacity - x.enrolled_and_ge_applied
                < -0.20 * x.enrolled_and_ge_applied
                else 0,
                axis=1,
            )
            violations = violations.groupby("zone_id").mean()[[">20%_shortage"]]
        else:
            violations = df.groupby("zone_id", as_index=False).sum()
            violations[">20%_shortage"] = violations.apply(
                lambda x: 1
                if x.ge_capacity - x.enrolled_and_ge_applied
                < -0.20 * x.enrolled_and_ge_applied
                else 0,
                axis=1,
            )
            violations = violations[[">20%_shortage"]]
        return violations

    def _student_count_variability_metrics(self, df, specific_year, zone_dict):
        if type(self.year) is list and not specific_year:
            ge_var = pd.DataFrame()
            for y in self.year:
                year_df = self.year_area_data_k8[y]
                year_df.loc[:, "zone_id"] = year_df["census_blockgroup"].replace(
                    zone_dict
                )
                zone_year = year_df.groupby("zone_id", as_index=False).sum()
                ge_var = ge_var.append(zone_year)
            ge_var = (
                ge_var.groupby("zone_id")
                .std()[["enrolled_and_ge_applied"]]
                .rename(columns={"enrolled_and_ge_applied": "ge_estimate_variance"})
            )
        else:
            ge_var = df.groupby("zone_id", as_index=False).sum()
            ge_var.loc[:, "ge_estimate_variance"] = 1
            ge_var = ge_var[["ge_estimate_variance"]]
        return ge_var

    def get_appropriate_area_data(self, all_students, include_k8, specific_year):
        if all_students and include_k8:
            if specific_year is not None:
                df = self.year_area_data_k8[specific_year]
            else:
                df = self.area_data_k8
        elif all_students and not include_k8:
            if specific_year is not None:
                df = self.year_area_data_no_k8[specific_year]
            else:
                df = self.area_data_no_k8
        else:
            dz = DesignZones(
                level=self.level,
                include_k8=include_k8,
                program_type=self.program_type,
                drop_optout=self.drop_optout,
                new_schools=self.new_schools,
                year=self.year,
            )
            df = dz.area_data
        return df

    def _serialize_metrics(self, df):
        flat = {}
        for idx, row in df.iterrows():
            for col, val in row.iteritems():
                flat[f"Zone {idx} {col}"] = val
        return pd.Series(flat)

    def _translate_filename_to_levers(self, file_path):
        """
        Lever numbers follow this document:
        https://docs.google.com/document/d/1zH6R9LTk0iaJzH8MkE3-ilq2dkZyHptqSDfzvvRRaOs/edit

        Example filename:
        GE_Zoning_1_1_3_2_1_centroids 8-zone-28_BG.csv
        """
        levers = file_path.split("_")[3:6] + file_path.split("_")[7:8]
        centroids = file_path.split()[-1].split("_")[0]
        mapping = [
            FRL_MAP,
            RACIAL_MAP,
            NUM_ZONES_MAP,
            K8_MAP,
        ]  # , SCHOOL_QUALITY_MAP, K8_MAP]
        names = [
            "FRL Balance",
            "Racial Representativeness",
            "Number of Zones",
            # "Consider School Quality",
            "Include K-8",
        ]
        file_name = file_path.split("/")[-1][:-4]
        image_command = f'=IMAGE("https://web.stanford.edu/~kmentzer/{file_name}.png")'.replace(
            " ", "%20"
        )
        lever_settings = {
            "Zone Name": file_name,
            "Image": image_command,
            "Centroids": centroids,
        }
        for lever_name, lever_setting, lever_map in zip(names, levers, mapping):
            lever_settings[lever_name] = lever_map[lever_setting]

        return pd.Series(lever_settings).reindex(
            index=[
                "Zone Name",
                "Image",
                "Centroids",
                "Number of Zones",
                "FRL Balance",
                "Racial Representativeness",
                "Include K-8",
                # "Consider School Quality",
            ]
        )

    def compute_metrics_on_file_list(
        self, file_list, img_save_location, output_save_file, append=True
    ):
        zv = ZoneVisualizer("BlockGroup")
        output = []

        for zonefile in file_list:
            print(zonefile.split("/")[-1][:-4])
            levers = self._translate_filename_to_levers(zonefile)

            include_k8 = levers["Include K-8"] == "Yes"
            school_list = (
                self.ge_schools_with_k8 if include_k8 else self.ge_schools_no_k8
            )
            zv.visualize_zones(zonefile, show=False, centroids=school_list)
            plt.savefig(img_save_location + zonefile.split("/")[-1][:-4] + ".png")
            plt.close()

            metrics, overall_metrics = self.zone_metrics(
                zonefile, include_k8, all_students=True
            )
            serialized_metrics = self._serialize_metrics(metrics)
            output.append(
                pd.concat(
                    [
                        levers,
                        serialized_metrics,
                        overall_metrics,
                        pd.Series({"year": str(self.year)}),
                    ]
                )
            )
            if type(self.year) is list and self.individual_years:
                for y in self.year:
                    year_metrics, year_overall_metrics = self.zone_metrics(
                        zonefile, include_k8, all_students=True, specific_year=y
                    )
                    serialized_year_metrics = self._serialize_metrics(year_metrics)
                    output.append(
                        pd.concat(
                            [
                                levers,
                                serialized_year_metrics,
                                year_overall_metrics,
                                pd.Series({"year": str(y)}),
                            ]
                        )
                    )

        if append and os.path.isfile(output_save_file):
            preexisting = pd.read_csv(output_save_file)
            new = pd.concat(output, axis=1).T
            table = preexisting.append(new)
            table.drop(columns=["Option"])
        else:
            table = pd.concat(output, axis=1).T
        dissimilarity_metrics = [
            x.split("_")[-1] + "_dissimilarity" for x in ETHNICITY_COLS
        ] + [
            "frl_dissimilarity",
            "sped_dissimilarity",
            "ell_dissimilarity",
            "L1_dissimilarity",
            "dissimilarity",
        ]
        transportation_metrics = (
            [
                f"%_{time}_{mode}_bgs_preserved_adjusted"
                for mode in ["walk", "bike", "transit"]
                for time in [20, 30]
            ]
            + [
                f"%_with_{i if i < 2 else '2+'}_{time}min_{mode}_options_adjusted"
                for i in range(3)
                for mode in ["walk", "bike", "transit"]
                for time in [20, 30]
            ]
            + [
                f"num_options_in_{time}mins_{mode}_adjusted"
                for mode in ["walk", "bike", "transit"]
                for time in [20, 30]
            ]
        )
        improvement_metrics = [
            "frl_improvement",
            "ell_improvement",
            "race_improvement",
            "sped_improvement",
            "l1_improvement",
        ]
        ordered_cols = (
            transportation_metrics
            + dissimilarity_metrics
            + improvement_metrics
            + ["year", "Option"]
        )
        table.sort_values("Zone Name", inplace=True)
        table["Option"] = (
            table.groupby(
                [
                    "FRL Balance",
                    "Racial Representativeness",
                    "Number of Zones",
                    # "Consider School Quality",
                    "Include K-8",
                    "year",
                ]
            ).cumcount()
            + 1
        )

        table = table[
            [x for x in table.columns if x not in ordered_cols] + ordered_cols
        ]
        table.to_csv(output_save_file, index=False)

    def compute_transit_subset_metrics_on_file_list(
        self,
        file_list,
        output_save_file,
        append=True,
        all_students=True,
        specific_year=None,
        walk_multiplier=1.3,
    ):
        output = []
        # for mode in ['walk', "bike", "transit"]:
        #     with open(
        #             os.path.expanduser(f"~/Dropbox/SFUSD/Data/Cleaned/bgs_within_20_{mode}_time.yaml"), "r"
        #     ) as f:
        #         setattr(self, f'20min_{mode}_zones', yaml.safe_load(f))

        for zonefile in file_list:
            print(zonefile.split("/")[-1][:-4])
            levers = self._translate_filename_to_levers(zonefile)

            include_k8 = levers["Include K-8"] == "Yes"
            df = self.get_appropriate_area_data(all_students, include_k8, specific_year)
            zone_dict = load_zones_from_file(zonefile)
            df["zone_id"] = df["census_blockgroup"].replace(zone_dict)

            metrics = self._supplementary_zone_level_transportation_metrics(
                df, include_k8, zone_dict, walk_multiplier
            )
            serialized_metrics = self._serialize_metrics(metrics)
            output.append(
                pd.concat(
                    [levers, serialized_metrics, pd.Series({"year": str(self.year)}),]
                )
            )

        if append and os.path.isfile(output_save_file):
            preexisting = pd.read_csv(output_save_file)
            new = pd.concat(output, axis=1).T
            table = preexisting.append(new)
            table.drop(columns=["Option"])
        else:
            table = pd.concat(output, axis=1).T

        ordered_cols = ["year", "Option"]
        table.sort_values("Zone Name", inplace=True)
        table["Option"] = (
            table.groupby(
                [
                    "FRL Balance",
                    "Racial Representativeness",
                    "Number of Zones",
                    # "Consider School Quality",
                    "Include K-8",
                    "year",
                ]
            ).cumcount()
            + 1
        )

        table = table[
            [x for x in table.columns if x not in ordered_cols] + ordered_cols
        ]
        table.to_csv(output_save_file, index=False)

    def compute_benchmarks(self, out_file):
        single_zone = os.path.expanduser(
            "~/Dropbox/SFUSD/Optimization/Zones/benchmark.csv"
        )
        metrics, _ = self.zone_metrics(single_zone, include_k8=True, all_students=True)
        metrics.T.to_csv(out_file)


def print_google_sheets_formulas(save_path, num_metrics=14, hci_version=False):
    s = string.ascii_uppercase
    l = [""] + [s[i] for i in range(len(s))]
    col_order = []
    for n1 in l[:15]:
        for n2 in l[1:]:
            col_order.append(n1 + n2)

    if hci_version:

        def make_command(col):
            return f"=INDEX(Data!${col}:${col},MATCH(1,($B$32=Data!$A:$A)*1,0))"

    else:

        def make_command(col):
            return f"=INDEX(Data!${col}:${col},MATCH(1,($C$8=Data!$D:$D)*($C$9=Data!$E:$E)*($C$10=Data!$F:$F)*($C$11=Data!$G:$G)*($C$12=Data!$LY:$LY),0))"

    out = [[] for _ in range(num_metrics)]
    for i in range(8):
        for j in range(num_metrics):
            idx = num_metrics * i + j
            col = col_order[idx + 7]
            out[j].append(make_command(col))

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out)


def print_google_sheets_formula_predictability(save_path, num_metrics=39):
    s = string.ascii_uppercase
    l = [""] + [s[i] for i in range(len(s))]
    col_order = []
    for n1 in l[:15]:
        for n2 in l[1:]:
            col_order.append(n1 + n2)

    out1 = []
    for i in range(8):
        out1.append(
            f"=(Data!${col_order[13 + i * num_metrics]}2-Data!${col_order[17 + i * num_metrics]}2)/Data!${col_order[17 + i * num_metrics]}2"
        )
    out2 = []
    for i in range(8):
        out2.append(
            f"=(Data!${col_order[14 + i * num_metrics]}2-Data!${col_order[18 + i * num_metrics]}2)/Data!${col_order[18 + i * num_metrics]}2"
        )

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([out1, out2])


def generate_file_list():
    files = glob.glob(
        "/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/Visualization_Tool/*_BG.csv"
    )
    df = pd.DataFrame(files, columns=["filename"])
    df.loc[:, "settings"] = df["filename"].apply(lambda x: x.split("/")[-1][10:19])
    one_per_setting = df.groupby("settings").first()
    return list(one_per_setting["filename"])


def get_missing_patterns_file_list(output_save_file, file_list):
    finished = pd.read_csv(output_save_file)
    shortened = pd.DataFrame(
        [x.split("/")[-1][:-4] for x in file_list], columns=["Zone Name"]
    )
    remaining = pd.merge(
        shortened, finished, on="Zone Name", how="outer", indicator=True
    )
    remaining = remaining[remaining["_merge"] == "left_only"]
    path = "/".join(file_list[0].split("/")[:-1])
    return [path + "/" + x + ".csv" for x in remaining["Zone Name"]]


def compute_school_dissimilarity_benchmark():
    df = pd.DataFrame()
    for f in glob.glob("/Users/katherinementzer/SFUSD/Data/school_frl/20*.xlsx"):
        print(f)
        yr = pd.read_excel(f)
        df = df.append(yr)
    elem_sch = pd.read_csv(
        "~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv"
    )
    df = df.groupby("Sch No", as_index=False).sum()
    df = elem_sch[["school_id"]].merge(
        df, how="inner", left_on="school_id", right_on="Sch No"
    )
    assert len(df) == 72  # check correct schools
    n = df["TotalCount"].sum()

    ethnicity_dissimilarity = {}
    dissimilarity_total = 0
    for eth in ["AsianCount", "BlackCount", "HispanicCount", "PICount", "WhiteCount"]:
        uniform_distn = df["TotalCount"] * (df[eth].sum() / n)
        difference_norm = np.linalg.norm(df[eth] - uniform_distn, 1)
        dissimilarity_total += difference_norm / 2
        eth_name = eth[:-5] + "_dissimilarity"
        ethnicity_dissimilarity[eth_name] = (difference_norm / 2) / df[eth].sum()
    metrics = ethnicity_dissimilarity
    metrics["all_race_dissimilarity"] = dissimilarity_total / n

    uniform_frl = df["TotalCount"] * (df["FRLCount"].sum() / n)
    difference_norm = np.linalg.norm(df["FRLCount"] - uniform_frl, 1)
    metrics["FRL_dissimilarity"] = difference_norm / n
    for k, v in metrics.items():
        print(f"{k}: \t{np.round(v, decimals=4)}")


if __name__ == "__main__":
    # file_list = generate_file_list()

    # file_list = glob.glob(
    #     "/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/Visualization_Tool 05:16:22/GE_Zoning_*_BG.csv"
    # ) + glob.glob(
    #     "/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/Visualization_Tool 05:16:22/GE_Zoning_*_BG_ls.csv"
    # )
    img_save_location = "/Users/katherinementzer/code/personal-website/WWW/"
    # output_save_file = "/Users/katherinementzer/SFUSD/dashboard_metrics.csv"
    # file_list = get_missing_patterns_file_list(output_save_file, file_list)
    dc = DashboardCreator(year=[14, 15, 16, 17, 18, 21, 22], individual_years=False)
    # dc.compute_metrics_on_file_list(
    #     file_list, img_save_location, output_save_file, append=False
    # )

    # selected_zones = ['GE_Zoning_1_1_1_2_2_centroids 6-zone-3_BG',
    #                   'GE_Zoning_1_1_1_2_2_centroids 6-zone-9_BG',
    #                   'GE_Zoning_1_1_2_2_1_centroids 7-zone-14_BG',
    #                   'GE_Zoning_1_1_2_2_1_centroids 7-zone-15_BG',
    #                   'GE_Zoning_1_2_1_2_2_centroids 6-zone-10_BG',
    #                   'GE_Zoning_1_2_1_2_2_centroids 6-zone-3_BG_ls',
    #                   'GE_Zoning_1_2_1_2_2_centroids 6-zone-7_BG',
    #                   'GE_Zoning_1_2_1_2_2_centroids 6-zone-9_BG',
    #                   'GE_Zoning_1_2_2_2_2_centroids 7-zone-14_BG',
    #                   'GE_Zoning_1_2_2_2_2_centroids 7-zone-17_BG',
    #                   'GE_Zoning_2_1_1_2_2_centroids 6-zone-10_BG',
    #                   'GE_Zoning_2_1_1_2_2_centroids 6-zone-2_BG_ls',
    #                   'GE_Zoning_2_1_1_2_2_centroids 6-zone-9_BG_ls']
    selected_zones = [
        "GE_Zoning_1_1_1_2_1_centroids 6-zone-9_BG",
        "GE_Zoning_1_2_1_2_2_centroids 6-zone-3_BG",
        "GE_Zoning_2_1_1_2_1_centroids 6-zone-9_BG",
    ]
    file_list = [
        f"/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/Visualization_Tool 05:16:22/{x}.csv"
        for x in selected_zones
    ]
    dc.compute_transit_subset_metrics_on_file_list(
        file_list,
        "/Users/katherinementzer/SFUSD/transit_metrics.csv",
        append=False,
        walk_multiplier=1.4,
    )
    dc.compute_benchmarks("/Users/katherinementzer/SFUSD/benchmark.csv")

    # save_path = os.path.expanduser("~/Desktop/commands.csv")
    # print_google_sheets_formulas(save_path, num_metrics=39, hci_version=True)
    # print_google_sheets_formula_predictability(save_path, num_metrics=39)

    # compute_school_dissimilarity_benchmark()
