import sys
import random, math, gc, os, csv
import pickle
from typing import Union

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from gurobipy import GRB
import geopandas as gpd

sys.path.append("../..")
from util import get_distance, make_school_geodataframe
from summary_statistics.zone_viz_mobin import ZoneVisualizer
from optimization.zone_opt.zone_eval import *
from optimization.zone_opt.Tuning_param import *
from optimization.zone_opt.graph_shortest_path import Shortest_Path

K8_SCHOOLS = [676, 449, 479, 760, 796, 493]
SF_Montessori = 814
ETHNICITY_DICT = {
    'Chinese': 'Asian',
    'Two or More': 'Two or More Races',
    'Middle Eastern/Arab': 'White',
    'Decline To State': 'Decline to state',
    'American Indian or Alaska Native': 'American Indian',
    'Korean': 'Asian',
    'Hispanic': 'Hispanic/Latinx',
    'Cambodian': 'Asian',
    'Japanese': 'Asian',
    'Other Pacific Islander': 'Pacific Islander',
    'Hawaiian': 'Pacific Islander',
    'Black or African American': 'Black or African American',
    'Vietnamese': 'Asian',
    'Samoan': 'Pacific Islander',
    'Tahitian': 'Pacific Islander',
    'Laotian': 'Asian',
    'Asian Indian': 'Asian',
    'Not Specified': 'Not specified',
    'Other Asian': 'Asian',
    'Hmong': 'Asian',
    'Middle Eastern/Arabic': 'White',
    'American Indian or Alaskan Native': 'American Indian',
    'Hispanic/Latino': 'Hispanic/Latinx',
    'Two or more races': 'Two or More Races'
}
CBEDS_SBAC_PATH = os.path.expanduser("~/SFUSD/Data/SFUSD_Demographics_SBAC_byblock.xlsx")


def _get_constraints_from_filename(filename):
    l = filename[:-4].split("_")
    constrs = {}
    for i in range(len(l) // 2):
        name = l[2 * i + 1]
        value = l[2 * i + 2].split("-")
        if len(value) > 1:
            constrs[name] = (float(value[0]), float(value[1]))
        else:
            constrs[name] = float(value[0])
    return constrs


def _get_zone_dict(assignment_name, return_list=False):
    with open(assignment_name, "r") as f:
        reader = csv.reader(f)
        zones = list(reader)
    zone_dict = {}
    for idx, schools in enumerate(zones):
        zone_dict = {
            **zone_dict,
            **{int(float(s)): idx for s in schools if s != ""},
        }
    if return_list:
        return zone_dict, zones
    return zone_dict


class DesignZones:
    def __init__(
            self,
            M=20,
            level="BlockGroup",
            centroids_type: Union[int, str] = -1,
            include_k8=False,
            population_type="GE",
            program_type="GE",
            drop_optout=True,
            capacity_scenario="Old",
            new_schools=False,
            year=18,
            move_SpEd=False,
            use_loaded_data=True
    ):
        self.use_loaded_data = use_loaded_data
        self.program_type = program_type
        self.population_type = population_type
        self.drop_optout = drop_optout
        self.year = year
        self.move_SpEd = move_SpEd
        allowed_capacity_scenarios = ["Old", "A", "B", "C", "D"]
        if capacity_scenario not in allowed_capacity_scenarios:
            raise ValueError(
                f"Unrecognized capacity scenario {capacity_scenario}. Please use one of {allowed_capacity_scenarios}."
            )
        self.capacity_scenario = capacity_scenario
        self.new_schools = new_schools
        self.allowed_building_blocks = ["BlockGroup", "idschoolattendance"]
        self.set_basic_parameters(M, centroids_type, include_k8, level)

        self._load_bg2att()
        self.load_all_data()


        self.N = sum(self.area_data["num_students"])
        self.A = len(self.area_data.index)
        self.studentsInArea = self.area_data["num_students"]

        print("self.A:       " + str(self.A))
        print("Number of students:       " + str(self.N))
        print("Number of total students: " + str(sum(self.area_data["total_students"])))
        print("Number of total seats:    " + str(sum(self.area_data["all_program_cap"])))
        print("Number of GE seats:       " + str(sum(self.seats)))

        self.area_data[self.level] = self.area_data[self.level].astype("int64")
        self.area2idx = dict(zip(self.area_data[self.level], self.area_data.index))
        self.idx2area = dict(zip(self.area_data.index, self.area_data[self.level]))
        self.euc_distances = self._load_euc_distance_data()
        # self.drive_distances = self._load_driving_distance_data()

        self.initialize_area_neighbor_dict()

        if type(self.centroids_type) == str or self.centroids_type >= 0:
            self.initialize_centroids()
            self.initialize_centroid_neighbors()

    def load_all_data(self):
        pd.set_option('display.max_rows', None)

        if self.use_loaded_data:
            self.ethnicity_cols = [
                "resolved_ethnicity_American Indian",
                "resolved_ethnicity_Asian",
                "resolved_ethnicity_Black or African American",
                "resolved_ethnicity_Filipino",
                "resolved_ethnicity_Hispanic/Latinx",
                "resolved_ethnicity_Pacific Islander",
                "resolved_ethnicity_Two or More Races",
                "resolved_ethnicity_White"]

            if self.include_k8:
                self.area_data = pd.read_csv(f"~/Dropbox/SFUSD/Data/final_area_data/area_data_k8.csv", low_memory=False)
            else:
                self.area_data = pd.read_csv(f"~/Dropbox/SFUSD/Data/final_area_data/area_data_no_k8.csv", low_memory=False)

            #####################  rename columns of input data so it matches with current format ###################
            self.area_data.rename(columns={"enrolled_and_ge_applied": "num_students",
                                       "enrolled_students": "total_students", "all_nonsped_cap": "all_program_cap",
                                        "census_blockgroup": "BlockGroup", "ge_schools":"num_schools" }, inplace=True)

            for metric in ['frl_count', 'sped_count', 'ell_count', 'num_students', 'total_students',
                           'all_program_cap', 'num_schools', 'num_with_ethnicity', 'K-8'] + self.ethnicity_cols:
                self.area_data[metric].fillna(value=0, inplace=True)
            self.area_data.dropna(subset=['BlockGroup'], inplace=True)

            self.area_data["num_students"] = self.area_data["num_students"] / 6
            self.area_data["total_students"] = self.area_data["total_students"] / 6



            self.FRL_ratio = sum(self.area_data["frl_count"]) / sum(self.area_data["frl_total_count"])
            print("FRL ratio is " + str(self.FRL_ratio))

            # self.area_data["frl%"] = self.area_data['frl_count'] / self.area_data['frl_total_count']
            # self.area_data["frl%"].fillna(value=0, inplace=True)
            #
            # for eth in self.ethnicity_cols:
            #     self.area_data[str(eth) + "%"] = self.area_data[eth] / self.area_data['num_with_ethnicity']
            #     self.area_data[str(eth) + "%"].fillna(value=0, inplace=True)
            # print("lll")
            # print(self.area_data["frl%"])
            #########################################################################################################



            #####################  repeating essential dictionary builds #######################
            ####################################################################################
            if self.new_schools:
                sc_df = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv")
            else:
                sc_df = pd.read_csv(f"~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv")
            self.sch2aa = dict(zip(sc_df["school_id"], sc_df["attendance_area"]))
            if self.level == "idschoolattendance":
                self.sch2block = self.sch2aa
            else:
                self.sch2block = dict(zip(sc_df["school_id"], sc_df[self.level]))
            sc_df["K-8"] = sc_df["school_id"].apply(lambda x: 1 if ((x in K8_SCHOOLS)) else 0)
            self.sc_df = self._load_capacities(sc_df)
            ####################################################################################

            if self.level == 'idschoolattendance':
                # TODO 497 --> 0
                self.area_data['idschoolattendance'] =  self.area_data['BlockGroup'].apply(lambda x: self.bg2att[int(x)] if int(x) in self.bg2att else 497)
                # print(self.area_data[['idschoolattendance', 'BlockGroup', 'num_students']])

                self.area_data = self.area_data.groupby(self.level, as_index=False).sum()
                self.area_data.reset_index(inplace=True)

            self.seats = (self.area_data["ge_capacity"].fillna(value=0).astype("int64").to_numpy())
            self.schools = self.area_data['num_schools'].fillna(value=0)


        else:
            self._load_area_data()
            self._load_auxilariy_areas()
            self._load_school_data()
            self._normalize_population_capacity_telemetry()
            self.F = sum(self.area_data["frl"])
            for metric in ['frl', 'sped_count', 'ell_count', 'num_students', 'total_students', 'all_program_cap', 'num_schools']:
                self.area_data[metric].fillna(value=0, inplace=True)

    def set_optimization_parameters(
            self,
            M=13,
            level="idschoolattendance",
            centroids_type=-1,
            include_k8=False,
    ):
        self.set_basic_parameters(M, centroids_type, include_k8, level)
        if self.centroids_type >= 0:
            self.initialize_centroids()
            self.initialize_centroid_neighbors()

    def set_basic_parameters(self, M, centroids_type, include_k8, level):
        self.M = M  # number of possible zones
        self.num_zones_for_programtype()  # set M according to program type, if not GE
        if level in self.allowed_building_blocks:
            self.level = level  # 'BlockGroup' or 'idschoolattendance'
        else:
            raise ValueError(
                f"Unrecognized level parameter. Please use one of {self.allowed_building_blocks}."
            )
        self.include_k8 = include_k8
        self.constraints = {"include_k8": include_k8}
        self.centroids_type = centroids_type

    def num_zones_for_programtype(self):
        if self.program_type == "GE":
            return
        # DLI - one option per zone
        elif self.program_type == "SE" or self.program_type == "SN":
            self.M = 9  # 9 typically, 3 to add more options
        elif self.program_type == "CE" or self.program_type == "CN":
            self.M = 4
        elif self.program_type == "ME" or self.program_type == "MN":
            self.M = 2
        # FLES - one option per zone
        elif self.program_type == "FB":
            self.M = 2
        elif self.program_type == "JE":
            self.M = 2
        # Biliteracy - 2-3 options per zone
        elif self.program_type == "SB":
            self.M = 5  # 13 programs
        elif self.program_type == "CB":
            self.M = 4  # 11 programs

    def _make_program_type_lists(self, df):
        # look at rounds 1-4, and all the programs listed in those rounds
        # make a new column program_types, which is a list of all such program types over different rounds
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

    def _program_student_filter(self, df, year):
        """Identify students relevant to the program of interest"""
        if self.population_type == "SB":
            df["filter"] = df.apply(
                lambda row: 1
                if row["homelang_desc"] == "SP-Spanish" or "SB" in row["program_types"]
                else 0,
                axis=1,
            )
        elif self.population_type == "CB":
            df["filter"] = df.apply(
                lambda row: 1
                if row["homelang_desc"] == "CC-Chinese Cantonese"
                   or "CB" in row["program_types"]
                else 0,
                axis=1,
            )
        elif self.population_type == "GE":
            if year != 18:
                df['filter'] = df.apply(
                    lambda row: 1
                    if "GE" in row["program_types"]
                    else 0,
                    axis=1
                )
            else:
                def filter_students(row):
                    if (
                            row["r1_idschool"] == row["enrolled_idschool"]
                            and row["r1_programcode"] == "GE"
                    ):
                        return 1
                    elif (
                            row["r3_idschool"] == row["enrolled_idschool"]
                            and row["r3_programcode"] == "GE"
                    ):
                        return 1
                    elif (
                            row["r1_idschool"] != row["enrolled_idschool"]
                            and row["r3_idschool"] != row["enrolled_idschool"]
                    ):
                        return 1
                    else:
                        return 0

                df["filter"] = df.apply(filter_students, axis=1)
        # if program type is "All"
        elif self.population_type == "All":
            return df
        return df.loc[df["filter"] == 1]

    def _load_student_data(self, year):
        # get student data
        if self.drop_optout:
            if year == 19:
                print("WARNING: Due to limited data, switching to using student_1920.csv instead of "
                      "drop_optout_1920.csv and including all students instead of just enrolled.")
                student_data = pd.read_csv(
                    f"~/SFUSD/Data/Cleaned/student_1920.csv", low_memory=False
                )
            else:
                student_data = pd.read_csv(
                    # f"~/SFUSD/Data/Cleaned/drop_optout_{year}{year + 1}.csv", low_memory=False
                    f"~/SFUSD/Data/Cleaned/enrolled_{year}{year + 1}.csv", low_memory = False
                )
        else:
            student_data = pd.read_csv(
                f"~/SFUSD/Data/Cleaned/student_{year}{year + 1}.csv", low_memory=False
            )
        student_data = student_data.loc[student_data["grade"] == "KG"]
        student_data['resolved_ethnicity'] = student_data['resolved_ethnicity'].replace(ETHNICITY_DICT)
        student_data.rename(columns={"census_blockgroup": "BlockGroup"}, inplace=True)
        self.all_student_data = self._make_program_type_lists(student_data)
        if year != 19:
            student_data = self.all_student_data.dropna(subset=["enrolled_idschool"])
        else:
            student_data = self.all_student_data
        student_data = self._program_student_filter(student_data, year)
        student_data.rename(
            columns={"census_blockgroup": "BlockGroup"}, inplace=True
        )
        if year == 21:
            student_data.loc[:, "ell"] = student_data["englprof"].apply(lambda x: 1 if x in ["N", "L"] else 0)
        else:
            student_data.loc[:, "sped"] = np.where(student_data["speced"] == "Yes", 1, 0)
            student_data.loc[:, "ell"] = student_data["englprof_desc"].apply(
                lambda x: 1 if (x == "N-Non English" or x == "L-Limited English") else 0
            )
        student_data.loc[:, "enrolled_students"] = np.where(
            student_data["enrolled_idschool"].isna(), 0, 1)

        for rnd in range(2, 6):
            if f"r{rnd}_programs" in student_data.columns:
                for idx, row in student_data.iterrows():
                    if len(student_data["r1_programs"][idx]) <= 0:
                        if type(student_data[f"r{rnd}_programs"][idx]) is list:
                            student_data["r1_programs"][idx] = student_data[f"r{rnd}_programs"][idx]

        student_data.r1_programs = student_data.r1_programs.apply(lambda x: x if len(x)>0 else [])
        student_data["enrolled_and_ge_applied"] = student_data.apply(
            lambda x: sum([i == "GE" for i in x["r1_programs"]]) / len(x["r1_programs"])
            if x["enrolled_students"] == 1 and len(x["r1_programs"]) > 0
            else 0,
            axis=1
        )
        print("student_data.enrolled_and_ge_applied  " + str(sum(student_data.enrolled_and_ge_applied)))
        return student_data

    def group_student_data_by_level(self, student_data):
        student_data = pd.get_dummies(
            student_data, columns=["resolved_ethnicity"]
        )
        area_data = student_data.groupby(self.level, as_index=False).mean()

        ethnicity_cols = [
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

        self.ethnicity_cols = ethnicity_cols
        old_cols = [
                       "Unnamed: 0",
                       "studentno",
                       "randomnumber",
                       "requestprogramdesignation",
                       "latitude",
                       "longitude",
                       "r1_idschool",
                       "r1_rank",
                       "r1_isdesignation",
                       "r1_distance",
                       "ctip1",
                       "r3_idschool",
                       "r3_rank",
                       "r3_isdesignation",
                       "r3_distance",
                       "zipcode",
                       "disability",
                       "enrolled_idschool",
                       "math_scalescore",
                       "ela_scalescore",
                       "r2_idschool",
                       "final_school",
                       "num_ranked",
                       "census_block",
                       "filter",
                       "sped",
                       "ell",
                       "r2_rank",
                       "r2_isdesignation",
                       "r2_distance",
                       "trans_sped"
                   ] + ethnicity_cols
        area_data.drop(
            columns=[x for x in old_cols if x in area_data.columns],
            inplace=True,
        )
        student_data["num_students"] = student_data["enrolled_and_ge_applied"]
        student_data["total_students"] = 1
        # student_data["num_students"] = 1

        cbeds = pd.read_excel(CBEDS_SBAC_PATH, sheet_name="CBEDS2122")
        cbeds.rename(columns={'Geoid10': 'census_block'}, inplace=True)
        cbeds['frl'] = cbeds["FRPM by block"]/cbeds["Distinct count of Student No"]
        if 'frl' in student_data.columns:
            student_data.drop(columns=["frl"], inplace=True)
        student_data = student_data.merge(cbeds[['census_block', 'frl']], how='left', on='census_block')

        # for col in ethnicity_cols:
        for col in ethnicity_cols + ['frl']:
            if col not in student_data.columns:
                student_data[col] = 0
            student_data[col] = student_data.apply(
                lambda x: x[col] * x["num_students"],
                axis=1
            )
        student_data["frl"].fillna(value=0, inplace=True)
        print("total_students")
        print(sum(student_data["total_students"]))

        #*student_data[self.level].fillna(value=0, inplace=True)
        # student_data["frl"] = (
        #         student_data["reducedlunch_prob"] + student_data["freelunch_prob"]
        # )
        df = (
            student_data[
                [self.level, "total_students", "enrolled_and_ge_applied", "num_students", "frl", "sped", "ell"] + ethnicity_cols
                ]
                .groupby(self.level, as_index=False)
                .sum()
        ).rename(columns={"sped": "sped_count", "ell": "ell_count"})

        area_data.drop(columns=["enrolled_and_ge_applied"], inplace=True)
        area_data = area_data.merge(df, how="left", on=self.level)
        area_data = area_data.sort_values(self.level)
        area_data.reset_index(inplace=True)

        print("total_students")
        print(sum(area_data["total_students"]))

        return area_data


    def _load_area_data(self):
        # years = [15]
        years = [14, 15, 16, 17, 18, 21]
        student_data_years = [0] * len(years)
        area_data_years = [0] * len(years)
        for i in range(len(years)):
            # load student data of year i
            student_data_years[i] = self._load_student_data(year=years[i])
            # groupby the student data of year i, suchh that we have the information only on area level
            area_data_years[i] = self.group_student_data_by_level(student_data_years[i])

        # average the student data over all years
        if len(years) > 1:
            self.area_data = pd.concat([area_data_years[0], area_data_years[1]]).groupby([self.level]).sum().reset_index()
            for i in range(2,len(years)):
                self.area_data = pd.concat([area_data_years[i], self.area_data]).groupby([self.level]).sum().reset_index()
            self.area_data.loc[:, self.area_data.columns != self.level] = self.area_data.loc[:, self.area_data.columns != self.level].div(len(years))
        else:
            self.area_data = area_data_years[0]

        self.area_data = self.area_data.sort_values(self.level)
        self.area_data.reset_index(inplace=True)


    def _load_auxilariy_areas(self):
        if (self.level=='BlockGroup') | (self.level=='Block'):
            # we add areas (blockgroups/blocks) that were missed from guardrail, since there was no student or school in them.
            # print(self.census_sf.head(10))
            current_area_list = set(self.area_data[self.level])
            df = pd.read_csv('~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv')
            valid_area_list = set(df[self.level])

            self._load_census_shapefile()
            auxilary_area = self.census_sf.copy()
            for idx, row in auxilary_area.iterrows():
                if (row[self.level] in (current_area_list.union({60759804011000, 60759804011001, 60759804011002, 60759804011003}))) \
                        or (row[self.level] not in valid_area_list):
                # if row[self.level] in area_list:
                        auxilary_area[self.level][idx] = 0
                else:
                    current_area_list.add(row[self.level])
            # auxilary_area = auxilary_area.loc[auxilary_area[self.level]!= 0][[self.level, 'geometry']]
            auxilary_area = auxilary_area.loc[auxilary_area[self.level]!= 0][[self.level]]
            self.area_data = self.area_data.append(auxilary_area, ignore_index=True)

            # self.area_data.fillna(0)


    def _load_census_shapefile(self):
        # get census block shapefile
        path = os.path.expanduser(
            "~/SFUSD/Census 2010_ Blocks for San Francisco/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp"
        )
        census_sf = gpd.read_file(path)
        census_sf["Block"] = (
            census_sf["geoid10"].fillna(value=0).astype("int64", copy=False)
        )

        df = pd.read_csv("~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv")
        df["Block"] = df["Block"].fillna(value=0).astype("int64", copy=False)
        self.census_sf = census_sf.merge(df, how="left", on="Block")

        self.census_sf.dropna(subset=['BlockGroup'], inplace=True)
        self.census_sf.dropna(subset=['Block'], inplace=True)
        self.census_sf[self.level] = self.census_sf[self.level].astype('int64')

    def _load_capacities(self, sc_df):
        # add on capacity
        if self.capacity_scenario != "Old":
            prog = self._load_dec2021_capacities()
        else:
            prog = pd.read_csv("~/SFUSD/Data/Cleaned/programs_1819.csv")
        self.prog = prog
        prog_ge = prog.loc[prog["program_type"] == self.program_type][
            ["school_id", "r3_capacity"]
        ]
        prog_all = prog[["school_id", "r3_capacity"]].rename(
            columns={"r3_capacity": "all_program_cap"}
        )
        prog_all = prog_all.groupby("school_id", as_index=False).sum()
        sc_df = sc_df.merge(prog_all, how="inner", on="school_id")
        sc_df = sc_df.merge(prog_ge, how="inner", on="school_id")
        sc_df = sc_df.loc[sc_df['r3_capacity'] > 0]
        # sc_df.rename(columns={"school_id": "idschoolattendance"}, inplace=True)
        return sc_df

    def _load_dec2021_capacities(self):
        cap_data = pd.read_csv("~/Dropbox/SFUSD/Data/stanford_capacities_12.23.21.csv")
        cap_data.rename(
            columns={
                "SchNum": "school_id",
                "PathwayCode": "program_type",
                f"Scenario_{self.capacity_scenario}_Capacity": "r3_capacity",
            }, inplace=True
        )
        return cap_data

    # rename columns & drop k8 schools if include_k8 is false & filter columns to be less busy
    def _eligible_school_data(self, sc_df):
        if self.include_k8:
            schools = sc_df.loc[:, :]
        else:
            schools = sc_df.loc[sc_df["K-8"] == 0]
        schools["num_schools"] = 1
        if self.level == "idschoolattendance":
            schools.rename(columns={"attendance_area": "idschoolattendance"}, inplace=True)

        return schools[[self.level, "school_name", "lon", "lat"] + [str(t + "_classes") for t in self.SpEd_types]]


    def _aggregate_school_data_to_area(self, sc_df):
        if self.include_k8:
            schools = sc_df.loc[:, :]
        else:
            schools = sc_df.loc[sc_df["K-8"] == 0]
        schools["num_schools"] = 1
        if self.level == "idschoolattendance":
            schools.rename(columns={"attendance_area": "idschoolattendance"}, inplace=True)

        sum_columns = [self.level, "all_program_cap", "r3_capacity", "num_schools"]

        sum_schools = (
            schools[sum_columns]
                .groupby(self.level, as_index=False)
                .sum()
        )
        mean_schools = (
            schools[
                [
                    self.level,
                    "eng_scores_1819",
                    "math_scores_1819",
                    "greatschools_rating",
                    "MetStandards",
                    "AvgColorIndex",
                ]
            ]
                .groupby(self.level, as_index=False)
                .sum()
        )
        return mean_schools.merge(sum_schools, how="left", on=self.level)


    def _load_classroom_data(self):
        sc_names = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv")
        sc_classrooms = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/Stanford_programs_1819_classrooms.csv")
        sc_classrooms = sc_classrooms.merge(sc_names, on='school_id')[['school_id', 'program_type', 'Classrooms', 'Capacity per classroom']]
        print("sc_classroomsss")
        print(sc_classrooms)
        print(sc_classrooms.loc[sc_classrooms['school_id'] == 413])
        # *** This list should start with 'GE'
        self.SpEd_types = ['GE', 'MS', 'MM', 'AF', 'ED']
        # self.SpEd_types = ['GE', 'MS', 'MM']
        # self.SpEd_types = ['GE', 'AF']
        # self.SpEd_types = ['GE', 'MM', 'AF']
        self.class_cap = {}

        for t in self.SpEd_types:
            t_schools = sc_classrooms.loc[sc_classrooms['program_type'] == t]
            self.class_cap[t] = int(t_schools['Capacity per classroom'].mean())
            self.sc_df[str(t + "_classes")] = 0
            print("school data of type  " + str(t) + "  is: " + str(t_schools))
            for idx, row in self.sc_df.iterrows():
                if row['school_id'] in set(t_schools['school_id']):
                    self.sc_df[str(t + "_classes")][idx] = t_schools.loc[t_schools['school_id'] == row['school_id']]['Classrooms']
                else:
                    self.sc_df[str(t + "_classes")][idx] = 0

        # Note, in sc_df, "school_id"s are distincts, but "attendance_area"s are not distinct
        print("sc_df.columns: " + str((self.sc_df.columns)))


    def _load_school_data(self):
        if self.new_schools:
            sc_df = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv")
        else:
            sc_df = pd.read_csv(f"~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv")
        self.sch2aa = dict(zip(sc_df["school_id"], sc_df["attendance_area"]))
        if self.level == "idschoolattendance":
            self.sch2block = self.sch2aa
        else:
            self.sch2block = dict(zip(sc_df["school_id"], sc_df[self.level]))

        sc_df["K-8"] = sc_df["school_id"].apply(lambda x: 1 if ((x in K8_SCHOOLS)) else 0)
        self.sc_df = self._load_capacities(sc_df)
        if self.move_SpEd:
            self._load_classroom_data()
            self.eligible_sc_df = self._eligible_school_data(self.sc_df)
            self.eligible_sc_df.reset_index(inplace=True)
            print("eligible_sc_df.columns: " + str((self.eligible_sc_df.columns)))

        school_area = self._aggregate_school_data_to_area(self.sc_df)
        self.school_area = school_area

        # Note: in school_area, "idschoolattendance"s (previously known as "attendance_area" in sc_df) are distinct
        print("school_area.columns: " + str(self.school_area.columns))


    def _normalize_population_capacity_telemetry(self):
        if self.move_SpEd:
            self.type_total_cap = {}
            for t in self.SpEd_types:
                self.type_total_cap[t] = self.eligible_sc_df[str(t + "_classes")].sum() * self.class_cap[t]
            total_capacity = sum(self.type_total_cap[t] for t in self.SpEd_types)
        else:
            total_capacity = self.school_area['r3_capacity'].sum()

        total_population = self.area_data['num_students'].sum()

        print("Pre-calculating total GE population:  " + str(total_population))
        print("Pre-calculating total GE capacity:  " + str(total_capacity))

        if self.move_SpEd:
            self.pop_ratio = [0] * len(self.SpEd_types)
            for t in range(len(self.SpEd_types)):
                self.pop_ratio[t] =  0.9999 * self.type_total_cap[self.SpEd_types[t]] / total_capacity
            print('pop_ratio: ' + str(self.pop_ratio))

        # *** uncomment the following lines when:
        # you want to normalize the total population
        # self.area_data.loc[:, self.area_data.columns != self.level] = \
        #     self.area_data.loc[:, self.area_data.columns != self.level].div(float(total_population)/total_capacity)


        self.area_data = self.area_data.merge(self.school_area, how="outer", on=self.level)
        self.seats = (
            self.area_data["r3_capacity"].fillna(value=0).astype("int64").to_numpy()
        )

        self.area_data['num_schools'].fillna(value=0)
        self.schools = self.area_data['num_schools'].fillna(value=0)



    def _get_first_choice(self):
        """ make column in student data table containing school first choice"""
        df = self.student_data
        df["firstchoice"] = np.nan
        for round in range(1, 6):
            col = "r{}_ranked_idschool".format(round)
            if col in df.columns:
                # format column names
                df[col] = df[col].fillna("[]")
                df[col] = df[col].apply(lambda x: eval(x))
                tmp = "r{}_tmp".format(round)
                df[tmp] = [
                    df.at[i, col][0]
                    if len(df.at[i, col]) > 0 and df.at[i, col][0] != ""
                    else np.nan
                    for i in df.index
                ]
                df["firstchoice"] = df["firstchoice"].fillna(df[tmp])
                df = df.drop(columns=[tmp])
        self.student_data = df

    def _load_bg2att(self):
        savename = '/Users/mobin/Dropbox/SFUSD/Optimization/bg2aa_mapping.pkl'

        # # This mapping is based on polygon shapefile information (not the students info)
        # if self.level=='Block':
        #     savename ='/Users/mobin/Dropbox/SFUSD/Optimization/b2aa_mapping.pkl'
        # if self.level=='BlockGroup':
        #     savename ='/Users/mobin/Dropbox/SFUSD/Optimization/bg2aa_mapping.pkl'
        # elif self.level=='idschoolattendance':
        #     # We don't need a mapping in this case
        #     return

        if os.path.exists(os.path.expanduser(savename)):
            file = open(savename, "rb")
            self.bg2att = pickle.load(file)
            print("File was already saved, and loaded faster")
            # print(self.bg2att)
            return

        # load attendance area geometry + its id in a single dataframe
        path = os.path.expanduser('~/Downloads/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
        sf = gpd.read_file(path)
        sf = sf.to_crs('epsg:4326')
        sc_merged = make_school_geodataframe()
        translator = sc_merged.loc[sc_merged['category'] == 'Attendance'][['school_id', 'index_right']]
        translator['school_id'] = translator['school_id'].fillna(value=0).astype('int64', copy=False)
        sf = sf.merge(translator, how='left', left_index=True, right_on='index_right')

        # load blockgroup/block  geometry + its id in a single dataframe
        df = self.census_sf.dissolve(by=self.level, as_index=False)

        self.bg2att = {}
        for i in range(len(df.index)):
            area_c = df['geometry'][i].centroid
            for z, row in sf.iterrows():
                aa_poly = row['geometry']
                # if aa_poly.contains(area_c) | aa_poly.touches(area_c):
                if aa_poly.contains(area_c):
                    self.bg2att[df[self.level][i]] = row['school_id']

        file = open(savename, "wb")
        pickle.dump(self.bg2att, file)
        file.close()


    def _load_euc_distance_data(self):
        if self.level == "BlockGroup":
            save_path = "~/Dropbox/SFUSD/Optimization/block2block_distances.csv"
        else:
            save_path = "~/Dropbox/SFUSD/Optimization/aa2aa_distances.csv"

        if os.path.exists(os.path.expanduser(save_path)):
            distances = pd.read_csv(save_path, index_col=self.level)
            distances.columns = [str(int(float(x))) for x in distances.columns]
            return distances

        if self.level == "BlockGroup":
            # self._load_census_shapefile()
            df = self.census_sf.dissolve(by="BlockGroup", as_index=False)
            df["centroid"] = df.centroid
            df["Lat"] = df["centroid"].apply(lambda x: x.y)
            df["Lon"] = df["centroid"].apply(lambda x: x.x)
            df = df[["BlockGroup", "Lat", "Lon"]]
            df.loc[:, "key"] = 0
            df = df.merge(df, how="outer", on="key")
            df.rename(
                columns={
                    "Lat_x": "Lat",
                    "Lon_x": "Lon",
                    "Lat_y": "st_lat",
                    "Lon_y": "st_lon",
                    "BlockGroup_x": "BlockGroup",
                },
                inplace=True,
            )
        else:
            df = self.area_data[["idschoolattendance", "lat", "lon"]]
            df.loc[:, "key"] = 0
            df = df.merge(df, how="outer", on="key")
            df.rename(
                columns={
                    "lat_x": "Lat",
                    "lon_x": "Lon",
                    "lat_y": "st_lat",
                    "lon_y": "st_lon",
                    "idschoolattendance_x": "idschoolattendance",
                },
                inplace=True,
            )
        df["distance"] = df.apply(get_distance, axis=1)
        df[self.level] = df[self.level].astype('Int64')
        table = pd.pivot_table(
            df,
            values="distance",
            index=[self.level],
            columns=[self.level + "_y"],
            aggfunc=np.sum,
        )
        table.to_csv(save_path)
        return table

    def _load_driving_distance_data(self, destinations = None):
        if self.level == 'BlockGroup':
            savename = '~/Dropbox/SFUSD/Optimization/OD_drive_time_cut60.csv'

        if os.path.exists(os.path.expanduser(savename)):
            drive_time = pd.read_csv(savename)
        if destinations == None:
            destinations = self.choices

        drive_time_distance = pd.DataFrame(index=sorted(list(self.area_data['BlockGroup'])))
        for school_id in destinations:
            # make sure this school_id was found by the system
            if len(drive_time.loc[drive_time['Name_1'] == school_id]) != 0:
                distance_array = []
                for bg in sorted(list(self.area_data['BlockGroup'])):
                    # make sure this bg id was found by the system, and the lack of distance info
                    # is not just due to the cut-off
                    if len(drive_time.loc[drive_time['Name_12'] == bg]) != 0:
                        dist = drive_time.loc[drive_time['Name_1'] == school_id].loc[drive_time['Name_12'] == bg]['Total_Trav']
                        if len(dist) == 1:
                            dist = float(dist)
                        elif len(dist) == 0:
                            # dist = 100
                            dist = math.inf
                        else:
                            print("duplicate distance data, error")
                            print(dist)

                    else:
                        # value -1 represents an unassigned value for now, we later
                        # construct value for these indices, using their neighbor avg
                        dist = -1
                        # print("missing bg init: " + str(bg))

                    distance_array.append(dist)
                print("school_id  " + str(school_id))
                print(" this is the sch2block  " + str(self.sch2level[school_id]))
                drive_time_distance[str(self.sch2level[school_id])] = distance_array

    def initialize_centroids(self):
        """set the centroids - each one is a block or attendance area depends on the method
        probably best to make it a school"""
        # set up centroids for language programs
        if self.program_type != "GE":
            if self.program_type in ["CE", "CN", "ME", "MN", "FB", "JE"]:
                # one per zone, take program locations
                locations = self.sc_df[self.sc_df["r3_capacity"] > 0][
                    "idschoolattendance"
                ]
                self.centroids = np.unique(
                    [self.area2idx[self.sch2block[i]] for i in locations]
                )
                assert len(self.centroids) == self.M
            if self.program_type in ["SE", "SN"]:
                locations = self.sc_df.sample(self.M)["idschoolattendance"]
                self.centroids = [self.area2idx[self.sch2block[i]] for i in locations]
            else:
                # if multiple programs per zone, sample M as centroids
                locations = self.sc_df[self.sc_df["r3_capacity"] > 0]
                locations = locations.sample(self.M)["idschoolattendance"]
                self.centroids = [self.area2idx[self.sch2block[i]] for i in locations]
            return

        if type(self.centroids_type) == str:
            with open("centroids.yaml", "r") as f:
                centroid_options = yaml.safe_load(f)
            if self.centroids_type not in centroid_options:
                raise ValueError(
                    "The centroids type specified is not defined in centroids.yaml."
                )
            for x in centroid_options[self.centroids_type]:
                if x not in self.sch2aa:
                    print(str(x) + '  not found')
            choices = [self.sch2aa[x] for x in centroid_options[self.centroids_type]]
            if len(choices) != self.M:
                print(f"WARNING: The value of M and the centroids policy does not match. Setting M to {len(choices)} for consistency.")
                self.M = len(choices)
        elif self.centroids_type == 2:
            centMet = [
                539,
                644,
                735,
                786,
                790,
                801,
                723,
                513,
                481,
                842,
                876,
                838,
                614,
                525,
                729,
                650,
                718,
            ]
            choices = np.random.choice(centMet, self.M, replace=False)

        elif self.centroids_type == 0:
            centMet = [
                435,
                735,
                859,
                872,
                838,
                842,
                746,
                420,
                507,
                488,
                862,
                722,
                656,
                478,
                650,
                481,
            ]
            choices = np.random.choice(centMet, self.M, replace=False)

        elif self.centroids_type == 3:
            centMet = [
                435,
                735,
                490,
                842,
                722,
                823,
                589,
                735,
                478,
                488,
                644,
                718,
                801,
                838,
                876,
                842,
                746,
                420,
            ]
            choices = np.random.choice(centMet, self.M, replace=False)

        elif self.centroids_type == 5:
            centMet = [
                435,
                569,
                750,
                862,
                722,
                664,
                823,
                872,
                589,
                735,
                478,
                413,
                488,
                644,
                718,
                801,
                838,
                876,
                842,
                746,
                420,
            ]
            choices = np.random.choice(centMet, self.M, replace=False)

        elif self.centroids_type == 4:
            centMet = [
                435,
                735,
                650,
                490,
                723,
                838,
                656,
                507,
                729,
                842,
                539,
                782,
                644,
                876,
                456,
            ]
            choices = np.random.choice(centMet, self.M, replace=False)

        elif self.centroids_type == 1:
            centMet = [488, 507, 453, 420, 497, 490, 670, 481, 413, 544, 478, 435, 456]
            choices = np.random.choice(centMet, self.M, replace=False)
        else:
            print("No centroids specified.")
            return
        self.sc_df['is_centroid'] = self.sc_df['school_id'].apply(lambda x: 1 if x in choices else 0)

        # self.centroid_location = self.sc_df.loc[self.sc_df['is_centroid'] == 1][['lon', 'lat']]
        if self.include_k8:
            self.centroid_location = self.sc_df[['lon', 'lat']]
        else:
            self.centroid_location = self.sc_df.loc[self.sc_df['K-8'] != 1][['lon', 'lat']]

        # pd.set_option("display.max_rows", None, "display.max_columns", None)

        if self.level == "idschoolattendance":
            self.centroids = [self.area2idx[j] for j in choices]
        else:
            self.centroids = [self.area2idx[self.sch2block[j]] for j in choices]

        self.constraints["centroidsType"] = self.centroids_type

    def initialize_area_neighbor_dict(self):
        """ build a dictionary mapping a block group/attendance area to a list
        of its neighboring block groups/attendnace areas"""

        if self.level == "BlockGroup":
            file = os.path.expanduser(
                "~/Dropbox/SFUSD/Optimization/block_group_adjacency_matrix.csv"
            )

        else:  # self.level == "idschoolattendance":
            file = os.path.expanduser(
                "~/Dropbox/SFUSD/Optimization/attendance_area_adjacency_matrix.csv"
            )

        with open(file, "r") as f:
            reader = csv.reader(f)
            nbhd = list(reader)

        # create dictionary mapping attendance area school id to list of neighbor
        # attendance area ids (similarly, block group number)
        self.neighbors = {}
        for units in nbhd:
            indices = [
                self.area2idx[int(x)]
                for x in units
                if x != ''
                   and int(x) in list(self.area2idx.keys())
            ]
            self.neighbors[indices[0]] = [x for x in indices[1:]]

    def load_geodesic_neighbors(self):
        self.shortestpath = Shortest_Path(self.neighbors, self.centroids)
        # print("pairwise distance")
        # print(self.shortestpath)

        for c in self.centroids:
            # for area in range(self.A):
            for area in self.candidate_idx:
                closer_geod = []
                for n in self.neighbors[area]:
                    if self.shortestpath[n, c] < self.shortestpath[area, c]:
                        closer_geod.append(n)
                self.closer_geodesic_neighbors[area, c] = closer_geod

    def initialize_centroid_neighbors(self):
        """ for each centroid c and each area j, define a set n(j,c) to be all neighbors of j such that are closer to c than j"""
        self.closer_euc_neighbors = {}
        self.closer_geodesic_neighbors = {}
        # self.load_geodesic_neighbors()

        self.euc_distances.dropna(inplace=True)
        for c in self.centroids:
            for area in range(self.A):
                n = self.neighbors[area]
                closer = [x for x in n
                    if self.euc_distances.loc[self.idx2area[c], str(self.idx2area[area])]
                       >= self.euc_distances.loc[self.idx2area[c], str(self.idx2area[x])]
                ]
                self.closer_euc_neighbors[area, c] = closer



    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def set_y_distance(self):
        self.y_distance = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="distance distortion")

        for z in range(self.M):
            zone_area =  str(self.idx2area[self.centroids[z]])
            zone_dist_sum = gp.quicksum([((self.euc_distances.loc[self.idx2area[j], zone_area]) ** 2) * self.x[j, z] for j in range(self.A)])
            # zone_dist_sum = gp.quicksum([((self.drive_distances.loc[self.idx2area[j], zone_area]) ** 2) * self.x[j, z] for j in range(self.A)])
            # zone_dist_sum = gp.quicksum([((self.drive_distances.loc[self.idx2area[j], zone_area]) ** 2) * (self.studentsInArea[j]) * self.x[j, z] for j in range(self.A)])
            self.m.addConstr(zone_dist_sum <= self.y_distance)


    def set_y_balance(self):
        self.y_balance = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="balance distortion")

        # minimize the maximum distortion from average number of students (across zones)
        for z in range(self.M):
            zone_stud = gp.quicksum([self.studentsInArea[j]*self.x[j,z] for j in range(self.A)])
            self.m.addConstr(self.N/self.M - zone_stud <= self.y_balance)
            self.m.addConstr(zone_stud - self.N/self.M <= self.y_balance)


    def set_y_shortage(self):
        self.y_shortage = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="shortage distortion")

        # minimize the maximum distortion from average student
        # deficit (student capacity -  number of seats) (across zones)
        for z in range(self.M):
            zone_stud = gp.quicksum([self.studentsInArea[j]*self.x[j,z] for j in range(self.A)])
            zone_seats = gp.quicksum([self.seats[j]*self.x[j,z] for j in range(self.A)])
            self.m.addConstr(zone_stud - zone_seats <= self.y_shortage)


    def _add_boundary_constraint(self):
        for i in range(self.A):
            for j in range(i+1, self.A):
                # if i and j are neighbors, check if they are boundaries of different zones
                if j in self.neighbors[i]:
                    # print("this is i:   " +str(i) + " and this is j:   " + str(j))
                    for z in range(self.M):
                        self.m.addConstr(gp.quicksum([self.x[i, z], -1 * self.x[j, z]]) <= self.b[i, j])
                        self.m.addConstr(gp.quicksum([-1 * self.x[i, z], self.x[j, z]]) <= self.b[i, j])
                else:
                    self.m.addConstr(self.b[i, j] == 0)
    def set_y_boundary(self):
        self.b = self.m.addVars(self.A, self.A, vtype=GRB.BINARY, name="boundary_vars")
        # self.b = self.m.addVars(self.A, self.A, lb=0.0, ub= 1.0, vtype=GRB.CONTINUOUS, name="boundary_vars")

        self.y_boundary = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="boundary distortion")
        # self.y_boundary = sum(sum(self.b[i, j] for j in range(i+1, self.A)) for i in range(self.A))
        self.m.addConstr(gp.quicksum([gp.quicksum([self.b[i, j] for j in range(i+1, self.A)]) for i in range(self.A)]) == self.y_boundary)
        self._add_boundary_constraint()

    def construct_general_objective_model(self):
        self.m = gp.Model("unknown_zones_feasibility")

        self.x = self.m.addVars(self.A, self.M, vtype=GRB.BINARY, name="x")
        # self.x = self.m.addVars(self.A, self.M, lb=0.0, ub= 1.0, vtype=GRB.CONTINUOUS, name="x")
        # self.x[1, 0].vtype = GRB.INTEGER

        self.constraints['M'] = self.M
        # # for z in range(self.M):
        # for j in range(self.A):
        #     if j %10 == 0:
        #         z = random.randint(0, self.M - 1)
        #         self.x[j,z].vtype = GRB.BINARY


        self.set_y_distance()
        self.distance_coef = 1

        self.set_y_boundary()
        self.boundary_coef = 10

        # self.set_y_balance()
        # self.balance_coef = 0

        # self.set_y_shortage()
        # self.shortage_coef = 0

        self.m.setObjective(self.distance_coef * self.y_distance + self.boundary_coef * self.y_boundary , GRB.MINIMIZE)
        # self.m.setObjective( self.distance_coef * self.y_distance +  self.shortage_coef * self.y_shortage +
        #                      self.balance_coef * self.y_balance + self.boundary_coef * self.y_boundary , GRB.MINIMIZE)

        self._shortage_and_balance_constraints(balance_ = False)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def moving_sped_objective_model(self):
        self.m = gp.Model("unknown_zones_feasibility")
        self.S = len(self.eligible_sc_df)
        self.T = len(self.SpEd_types)
        print("S is: " + str(self.S))
        print("T is: " + str(self.T))
        self.q = self.m.addVars(self.T, self.S, lb=0.0, vtype=GRB.INTEGER, name="x")
        self.y = self.m.addVars(self.T, self.S, vtype=GRB.BINARY, name="y")
        self.p = self.m.addVars(self.T, self.A, self.S, lb=0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name="p")

        self.set_y_SpEd()
        self.m.setObjective(self.y_SpEd , GRB.MINIMIZE)
        # self.m.setObjective(1 , GRB.MINIMIZE)

        self.assignment_contstraint()
        self.school_class_fix()
        self.program_class_fix()
        self.school_valid_class_count()
        self.school_type_limit()



    def set_y_SpEd(self):
        self.y_SpEd = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="objective")
        self.m.addConstr(
            gp.quicksum([((self.euc_distances.loc[self.idx2area[j], str(self.eligible_sc_df[self.level][s])]) ** 2) *
                         (self.studentsInArea[j] * self.pop_ratio[t]) * self.p[t, j, s]
                         for s in range(self.S)
                         for j in range(self.A)
                         for t in range(1, self.T)])
            == self.y_SpEd
        )
    def assignment_contstraint(self):
        # each students of each area (for each program) are assigned to 1 school
        for t in range(self.T):
            for j in range(self.A):
                self.m.addConstr(
                    gp.quicksum([self.p[t, j, s]
                                 for s in range(self.S)])
                    == 1
                )

        # no school is assigned more students than it fits
        for t in range(self.T):
            for s in range(self.S):
                self.m.addConstr(
                    gp.quicksum([self.studentsInArea[j] * self.pop_ratio[t] * self.p[t, j, s]
                                 for j in range(self.A)])
                    <=
                    self.class_cap[self.SpEd_types[t]] * self.q[t, s]
                )


    def school_class_fix(self):
        # number of classes for each school over all the programs remains the same
        for s in range(self.S):
            self.m.addConstr(
                gp.quicksum([self.q[t,s]
                             for t in range(self.T)])
                ==
                sum([self.eligible_sc_df[str(self.SpEd_types[t] + "_classes")][s]
                             for t in range(self.T)])
            )

    def program_class_fix(self):
        # number of classes for each program over the schools remains the same
        for t in range(self.T):
            self.m.addConstr(
                gp.quicksum([self.q[t,s]
                             for s in range(self.S)])
                ==
                sum([self.eligible_sc_df[str(self.SpEd_types[t] + "_classes")][s]
                             for s in range(self.S)])
            )

    def school_valid_class_count(self):
        # make sure the capacity for program type t in school s is more than zero,
        # if variable y for it is non-zero
        for t in range(self.T):
            for s in range(self.S):
                self.m.addConstr(self.q[t, s] <= 10 * self.y[t, s])

    def school_type_limit(self):
        # Each school has max 2 classes for a single SpEd program type
        for s in range(self.S):
            self.m.addConstr(
                # gp.quicksum([self.y[t,s] for t in range(self.T)])
                gp.quicksum([self.q[t,s] for t in range(1,self.T)])
                <= 2)


        # No school has more than 1 type of SpEd classes in it
        # Note TOD: if you change to 2 program type at a school ==> update visualization to incorporate this new assumption
        for s in range(self.S):
            self.m.addConstr(
                gp.quicksum([self.y[t,s] for t in range(1,self.T)])
                <= 1)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def set_model(self, shortage=0.15, overage = 0.2, all_cap_shortage = 0.8, balance=1000):
        self.all_cap_shortage = all_cap_shortage
        self.shortage = shortage
        self.overage = overage
        self.balance = balance

        if self.move_SpEd:
            self.moving_sped_objective_model()
        else:
            self.construct_general_objective_model()


    def construct_feasibility_model(self):
        self.m = gp.Model("unknown_zones_feasibility")
        self.x = self.m.addVars(self.A, self.M, vtype=GRB.BINARY, name="x")
        self.constraints["M"] = self.M

        # get any feasible solution
        self.m.setObjective(1, GRB.MAXIMIZE)

        self._shortage_and_balance_constraints()


    def construct_frl_objective_model(self):
        self.constraints = {}
        self.m = gp.Model("unknown_zones_frl_minimax")

        self.x = self.m.addVars(self.A, self.M, vtype=GRB.BINARY, name="x")
        self.y = self.m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
        self.constraints["M"] = self.M

        # minimize the maximum number of FRL students across zones
        self.area_data["frl"] = self.area_data["frl"]
        for z in range(self.M):
            zone_sum = gp.quicksum(
                [self.area_data["frl"][j] * self.x[j, z] for j in range(self.A)]
            )
            self.m.addConstr(zone_sum <= self.y)
        self.m.setObjective(self.y, GRB.MINIMIZE)

        self._shortage_and_balance_constraints()

    def construct_balance_objective_model(self):
        self.constraints = {}
        self.m = gp.Model("unknown_zones_frl_minimax")

        self.x = self.m.addVars(self.A, self.M, vtype=GRB.BINARY, name="x")
        self.y = self.m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
        self.constraints["M"] = self.M

        # minimize the maximum seat deficit across zones
        for z in range(self.M):
            firstZone = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in range(self.A)]
            )
            for q in range(z + 1, self.M):
                secondZone = gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, q] for j in range(self.A)]
                )
                self.m.addConstr(firstZone - secondZone <= self.y)
                self.m.addConstr(firstZone - secondZone >= -self.y)
        self.m.setObjective(self.y, GRB.MINIMIZE)

        self._shortage_and_balance_constraints(balance_=False)

    def construct_shortage_objective_model(self):
        self.constraints = {}
        self.m = gp.Model("unknown_zones_frl_minimax")

        self.x = self.m.addVars(self.A, self.M, vtype=GRB.BINARY, name="x")
        self.y = self.m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
        self.constraints["M"] = self.M

        # minimize the maximum seat deficit across zones
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [
                        (self.seats[j] - self.studentsInArea[j]) * self.x[j, z]
                        for j in range(self.A)
                    ]
                )
                >= self.y)
        self.m.setObjective(self.y, GRB.MAXIMIZE)

        self._shortage_and_balance_constraints(shortage_=False)


    def all_capacity_proportional_shortage_const(self):
        # No zone has shortage more than self.all_cap_shortage percentage of its total student population
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [(self.area_data["total_students"][j] - self.area_data["all_program_cap"][j]) * self.x[j, z]
                     for j in range(self.A)]
                )
                <=
                self.all_cap_shortage *
                gp.quicksum(
                    [self.area_data["total_students"][j] * self.x[j, z]
                     for j in range(self.A)]
                )
            )
        self.constraints["All_Cap_Propotional_Shortage"] = self.all_cap_shortage

    def proportional_shortage_const(self):
        # No zone has shortage more than self.shortage percentage of its population
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [(self.studentsInArea[j] - self.seats[j]) * self.x[j, z]
                        for j in range(self.A)]
                )
                <=
                self.shortage *
                gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z]
                     for j in range(self.A)]
                )
            )
        self.constraints["Propotional_Shortage"] = self.shortage


    def proportional_overage_const(self):
        # No zone has shortage more than self.shortage percentage of its population
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [(-self.studentsInArea[j] + self.seats[j]) * self.x[j, z]
                        for j in range(self.A)]
                )
                <=
                self.overage *
                gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z]
                     for j in range(self.A)]
                )
            )
        self.constraints["Propotional_Overage"] = self.overage

    def fixed_shortage_const(self):
        # each zone has at least the shortage
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [(self.studentsInArea[j] - self.seats[j]) * self.x[j, z]
                        for j in range(self.A)]
                )
                <= self.shortage)
        self.constraints["Fixed_Shortage"] = self.shortage

    def _shortage_and_balance_constraints(self, shortage_=True, balance_=True):
        # every area has to belong to one zone
        self.m.addConstrs(
            gp.quicksum(self.x[j, z] for z in range(self.M)) == 1
                for j in range(self.A)
        )

        if shortage_:
            # self.fixed_shortage_const()
            self.proportional_shortage_const()
            self.proportional_overage_const()
            self.all_capacity_proportional_shortage_const()

        if balance_:
            # add number of students balance constraint
            for z in range(self.M):
                firstZone = gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z] for j in range(self.A)]
                )
                for q in range(z + 1, self.M):
                    secondZone = gp.quicksum(
                        [self.studentsInArea[j] * self.x[j, q] for j in range(self.A)]
                    )
                    self.m.addConstr(firstZone - secondZone <= self.balance)
                    self.m.addConstr(firstZone - secondZone >= -self.balance)
            self.constraints["Balance"] = self.balance



    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def _add_geo_constraints(
            self,
            max_distance=-1,
            real_distance=True,
            cover_distance=-1,
            contiguity=False,
            neighbor=False,
    ):

        # distance constraint
        if max_distance > 0:
            self._add_distance_constraint(max_distance, real_distance)
        if cover_distance > 0:
            self._add_cover_distance(cover_distance)
        else:
            self.constraints["cover"] = -1
        if contiguity:
            self._add_contiguity_constraint()
        else:
            self.constraints["contiguity"] = 0
        if neighbor:
            self._add_neighbor_constraint(1)

    def _add_distance_constraint(self, distance_bound, real=False):
        self.farAwayLists = []
        for j in range(0, self.A - 1):
            farAwayList = []
            for k in range(j + 1, self.A):
                if (
                        self.euc_distances.loc[self.idx2area[j], str(self.idx2area[k])]
                        > distance_bound
                ):
                    farAwayList.append(k)
                    if real:
                        for z in range(self.M):
                            self.m.addConstr(
                                self.x[j, z] + self.x[k, z] <= 1, name="Distance"
                            )
            self.farAwayLists.append(farAwayList)
        self.constraints["distance"] = distance_bound


    def _add_cover_distance(self, coverDistance):
        """  """
        for z in range(self.M):
            c = str(self.idx2area[self.centroids[z]])
            zone_sum = gp.quicksum(
                [
                    self.euc_distances.loc[self.idx2area[j], c]
                    * self.studentsInArea[j]
                    * self.x[j, z]
                    for j in range(self.A)
                ]
            )
            students_sum = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in range(self.A)]
            )
            self.m.addConstr(
                zone_sum <= coverDistance * students_sum, name="Cover Distance"
            )
        self.constraints["cover"] = coverDistance

    def _add_contiguity_constraint(self):
        """ initialization - every centroid belongs to its own zone"""
        for z in range(self.M):
            self.m.addConstr(
                self.x[self.centroids[z], z] == 1, name="Centroids to Zones"
            )

        """ x[j,z] \leq sum over all x[j',z] where j'  is in self.closer_neighbors_per_centroid[area,c]  where c is cetnroid for z"""
        for j in range(self.A):
            for z in range(self.M):
                if j == self.centroids[z]:
                    continue
                if len(self.closer_euc_neighbors[j,self.centroids[z]]) >= 1:
                    neighbor_sum = gp.quicksum(
                        [
                            self.x[k, z]
                            for k in self.closer_euc_neighbors[
                            j, self.centroids[z]
                        ]
                        ]
                    )
                    self.m.addConstr(self.x[j, z] <= neighbor_sum, name="Contiguity")
        self.constraints["contiguity"] = 1

    def _add_neighbor_constraint(self, radius):
        neighbors = np.zeros((self.A, self.A))
        if self.level == "BlockGroup":
            for i in range(self.A):
                for j in range(self.A):
                    dist = self.euc_distances.loc[self.idx2area[i], str(self.idx2area[j])]
                    if (
                            self.idx2area[j] == 60750179021
                            or self.idx2area[i] == 60750179021
                    ) and dist <= 2:
                        neighbors[i, j] = 1
                    elif dist <= radius:
                        neighbors[i, j] = 1
                neighbors[i, i] = 0
        if self.level == "idschoolattendance":
            for i in range(self.A):
                for j in range(self.A):
                    dist = self.euc_distances.loc[self.idx2area[i], str(self.idx2area[j])]
                    if dist <= radius:
                        neighbors[i, j] = 1
                neighbors[i, i] = 0

        for z in range(self.M):
            for j in range(self.A):
                neighbor_count = gp.quicksum(
                    self.x[k, z] * neighbors[k, j] for k in range(self.A)
                )
                is_active = self.x[j, z]
                self.m.addConstr(neighbor_count >= is_active, name="Neighbor")
        self.constraints["neighbor"] = radius



    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def _add_diversity_constraints(
            self, racial_dev=1, frl_dev=1, HOCidx1_dev=1, aalpi_dev=1
    ):
        # racial balance constraint
        if racial_dev < 1:
            self._add_racial_constraint(racial_dev)
        # frl constraint
        if frl_dev < 1:
            if hasattr(self, "gamma"):
                self._add_frl_constraint(1 - self.gamma, 1 + self.gamma)
            else:
                self._add_frl_constraint(frl_dev)
        # guardrail constraints
        if HOCidx1_dev < 1:
            if hasattr(self, "gamma"):
                self._add_guardrail_constraint(
                    "HOCidx1", 1 - self.gamma, 1 + self.gamma
                )
            else:
                self._add_guardrail_constraint("HOCidx1", HOCidx1_dev)
        # aalpi constraint
        if aalpi_dev < 1:
            if hasattr(self, "gamma"):
                self._add_guardrail_constraint(
                    "AALPI Score", 1 - self.gamma, 1 + self.gamma
                )
            else:
                self._add_guardrail_constraint("AALPI Score", aalpi_dev)


    def _add_racial_constraint(self, race_dev=1):
        for race in \
                self.ethnicity_cols:
                # ['resolved_ethnicity_White',
                #      'resolved_ethnicity_Hispanic/Latinx',
                #      'resolved_ethnicity_Asian']:
            if self.use_loaded_data:
                race_ratio = sum(self.area_data[race]) / sum(self.area_data["num_with_ethnicity"])
                print(str(race) + " is " + str(race_ratio))

                for z in range(self.M):
                    zone_sum = gp.quicksum(
                        [self.area_data[race][j] * self.x[j, z] for j in range(self.A)])
                    zone_total = gp.quicksum(
                        [self.area_data["num_with_ethnicity"][j] * self.x[j, z] for j in range(self.A)])

                    self.m.addConstr(zone_sum >= (race_ratio - race_dev) * zone_total, name= str(race) + " LB")
                    self.m.addConstr(zone_sum <= (race_ratio + race_dev) * zone_total, name= str(race) + " UB")

            else:
                race_ratio = sum(self.area_data[race]) / float(self.N)
                print(str(race) + " is " + str(race_ratio))

                for z in range(self.M):
                    zone_sum = gp.quicksum(
                        [self.area_data[race][j] * self.x[j, z] for j in range(self.A)]
                    )
                    district_students = gp.quicksum(
                        [self.studentsInArea[j] * self.x[j, z] for j in range(self.A)]
                    )
                    self.m.addConstr(zone_sum >= (race_ratio - race_dev) * district_students, name= str(race) + " LB")
                    self.m.addConstr(zone_sum <= (race_ratio + race_dev) * district_students, name= str(race) + " UB")



    def _add_frl_constraint(self, frl_dev=1):
        for z in range(self.M):
            if self.use_loaded_data:
                zone_sum = gp.quicksum([self.area_data["frl_count"][j] * self.x[j, z] for j in range(self.A)])
                zone_total = gp.quicksum([self.area_data["frl_total_count"][j] * self.x[j, z] for j in range(self.A)])
                self.m.addConstr(zone_sum >= (zone_total * (self.FRL_ratio - frl_dev)), name="FRL LB")
                self.m.addConstr(zone_sum <= (zone_total * (self.FRL_ratio + frl_dev)), name="FRL UB")
            else:
                zone_sum = gp.quicksum(
                    [self.area_data["frl"][j] * self.x[j, z] for j in range(self.A)]
                )
                district_students = gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z] for j in range(self.A)]
                )
                self.m.addConstr(zone_sum >= (float(self.F) / (self.N) - frl_dev) * district_students, name="FRL LB")
                self.m.addConstr(zone_sum <= (float(self.F) / (self.N) + frl_dev) * district_students, name="FRL UB")

        self.constraints["frl_dev"] = frl_dev

    def _add_guardrail_constraint(self, metric, allowed_dev):
        scores = self.area_data[metric]

        for z in range(self.M):
            zone_sum = gp.quicksum(
                [scores[j] * self.studentsInArea[j] * self.x[j, z] for j in range(self.A)]
            )
            district_students = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in range(self.A)]
            )

            self.m.addConstr(zone_sum >= (sum(scores * self.studentsInArea) / float(self.N) - allowed_dev) * district_students, name=str(metric) + " LB")
            self.m.addConstr(zone_sum <= (sum(scores * self.studentsInArea) / float(self.N) + allowed_dev) * district_students, name=str(metric) + " UB")


    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    def _add_school_count_constraint(self):
        zone_school_count = {}
        avg_school_count = sum([self.schools[j] for j in range(self.A)]) / self.M + 0.0001
        # note: although we enforce max deviation of 1 from avg, in practice,
        # no two zones will have more than 1 difference in school count
        # Reason: school count is int. Observe the avg_school_count +-1,
        # if avg_school_count is not int, and see how the inequalities will look like
        # * I implemented the code this way (instead of pairwise comparison), since it is faster
        for z in range(self.M):
            zone_school_count[z] = gp.quicksum([self.schools[j] * self.x[j, z] for j in range(self.A)])
            self.m.addConstr(zone_school_count[z] <= avg_school_count + 1)
            self.m.addConstr(zone_school_count[z] >= avg_school_count - 1)

        # # Pairwise comparison
        # for z1 in range(self.M - 1):
        #     for z2 in range(z1 + 1, self.M):
        #         self.m.addConstr(zone_school_count[z1] <= zone_school_count[z2]+1)
        #         self.m.addConstr(zone_school_count[z1] >= zone_school_count[z2]-1)

        # if K8 schools are included,
        # make sure no zone has more than one K8 schools
        if self.include_k8:
            zone_k8_count = {}
            for z in range(self.M):
                if self.use_loaded_data:
                    zone_k8_count[z] = gp.quicksum([self.area_data["K-8"][j] * self.x[j, z]
                                                    for j in range(self.A)])
                else:
                    zone_k8_count[z] = gp.quicksum([
                                                    self.sc_df["K-8"][j] *
                                                    self.x[self.area2idx[self.sc_df[self.level][j]], z]
                                                    for j in range(len(self.sc_df.index))
                                                    ])

                self.m.addConstr(zone_k8_count[z] <= 1)


    def _add_school_quality_constraint(self, min_pct, max_pct=None):
        scores = self.area_data["eng_scores_1819"].fillna(value=0)
        schools = self.area_data["num_schools"].fillna(value=0)
        for z in range(self.M):
            zone_sum = gp.quicksum(
                [scores[j] * schools[j] * self.x[j, z] for j in range(self.A)]
            )
            district_average = (
                    sum(scores * schools)
                    / sum(schools)
                    * gp.quicksum([self.x[j, z] * schools[j] for j in range(self.A)])
            )

            self.m.addConstr(zone_sum >= min_pct * district_average)
            if max_pct != None:
                self.m.addConstr(zone_sum <= max_pct * district_average)
                self.constraints["engscores1819"] = str(min_pct) + "-" + str(max_pct)
            else:
                self.constraints["engscores1819"] = min_pct

        scores = self.area_data["math_scores_1819"].fillna(value=0)

        for z in range(self.M):
            zone_sum = gp.quicksum(
                [scores[j] * schools[j] * self.x[j, z] for j in range(self.A)]
            )
            district_average = (
                    sum(scores * schools)
                    / sum(schools)
                    * gp.quicksum([self.x[j, z] * schools[j] for j in range(self.A)])
            )

            self.m.addConstr(zone_sum >= min_pct * district_average)
            if max_pct != None:
                self.m.addConstr(zone_sum <= max_pct * district_average)
                self.constraints["math_scores_1819"] = str(min_pct) + "-" + str(max_pct)
            else:
                self.constraints["math_scores_1819"] = min_pct

    def _add_met_quality_constraint(self, min_pct = 0, max_pct=None, topX=0):

        scores = self.area_data["AvgColorIndex"].fillna(value=0)

        if min_pct > 0:
            for z in range(self.M):
                zone_sum = gp.quicksum(
                    [scores[j] * self.schools[j] * self.x[j, z] for j in range(self.A)]
                )
                district_average = (
                        sum(scores * self.schools) / sum(self.schools)
                        * gp.quicksum([self.x[j, z] * self.schools[j] for j in range(self.A)])
                )

                self.m.addConstr(zone_sum >= min_pct * district_average)
                if max_pct is not None:
                    self.m.addConstr(zone_sum <= max_pct * district_average)
                    self.constraints["AvgColorIndex"] = (str(min_pct) + "-" + str(max_pct))
                else:
                    self.constraints["AvgColorIndex"] = min_pct

        if topX > 0:
            top_schools = np.zeros([self.A])
            top = np.percentile(scores, 100 * (1 - self.M / self.A) - 0.05)
            top = np.percentile(scores, topX)
            print(top)
            for j in range(self.A):
                if scores[j] > top:
                    top_schools[j] = 1
            for z in range(self.M):
                topz = gp.quicksum(
                    [self.x[j, z] * top_schools[j] for j in range(self.A)]
                )
                self.m.addConstr(topz >= 0.8)
                self.constraints["AvgColorIndex"] = topX


    def add_favorite_school_constraint(self, k=3):
        """ add constraint that one of top k favorite schools are in your zone """
        # get top 3 favorite schools for each attendance area
        if "firstchoice" not in self.student_data.columns:
            self._get_first_choice()
        gk = self.student_data.groupby("idschoolattendance", as_index=False)
        aa2top3 = {}
        for name, group in gk:
            subgroup = group.groupby("firstchoice", as_index=False).count()
            subgroup.sort_values("studentno", ascending=False, inplace=True)
            aa2top3[name] = subgroup["firstchoice"][:k]
        # add constraint that one of top 3 in your zone
        # schno2aa = dict(
        #     zip(self.sc_df["original_schno"], self.sc_df["idschoolattendance"])
        # )
        for aa, top3 in aa2top3.items():
            aa_idx = self.area2idx[int(aa)]
            top3idxs = [self.area2idx[self.sch2aa[int(x)]] for x in top3]
            for z in range(self.M):
                fave_in_zone = gp.quicksum(self.x[i, z] for i in top3idxs)
                self.m.addConstr(fave_in_zone >= self.x[aa_idx, z])

    def _boundary_threshold_constraint(self, boundary_threshold):
        # make sure areas that are closer than boundary_threshold distance
        # to a school, are matched to the same zone as that school.
        for i, row in self.sc_df.iterrows():
            if (row["K-8"] == 1) & (self.include_k8 == False):
                continue
            s = row["school_id"]
            for area_idx in range(self.A):

                if self.euc_distances.loc[self.sch2block[s], str(self.idx2area[area_idx])] < boundary_threshold:
                    for z in range(self.M):
                        self.m.addConstr(self.x[area_idx, z] == self.x[self.area2idx[self.sch2block[s]], z])

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def save(self, path,  rand=random.randint(0, 400000), save_opt_params = False, name = "", solve_success = 1):

        filename = os.path.expanduser(path + self.program_type)
        filename += name
        filename += ".csv"

        # save zones themselves
        with open(filename, "w") as outFile:
            writer = csv.writer(outFile, lineterminator="\n")
            if solve_success == 1:
                for z in self.z:
                    writer.writerow(z)
            else:
                writer.writerow({})

        # if solve_success == 1:
        #     # save zones themselves
        #     with open(filename, "w") as outFile:
        #         writer = csv.writer(outFile, lineterminator="\n")
        #         for z in self.z:
        #             writer.writerow(z)
        # elif solve_success == -1:
        #     with open(filename, "w") as outFile:
        #         writer = csv.writer(outFile, lineterminator="\n")
        #         for z in self.z:
        #             writer.writerow(z)
        #     writer.writerow({})

        # save optimization parameters
        if save_opt_params == True:
            with open(filename[:-4] + "_params.txt", "w") as f:
                for k, v in self.constraints.items():
                    f.write("".join(k.split()) + " " + str(v) + "\n")
                if type(self.centroids_type) == str or self.centroids_type >= 0:
                    f.write("centroids " + str(self.centroids_type))

    def save_language_zone(
            self, rand=random.randint(0, 400000), save_path="", prog="SB"
    ):
        progs = self.prog.loc[self.prog["program_type"] == self.program_type]
        progs["aa"] = progs["school_id"].apply(lambda x: self.sch2aa[x])
        if prog == "SE":
            progs["zone_id"] = range(len(progs.index))
        else:
            progs["zone_id"] = progs["aa"].apply(lambda x: self.zone_dict[x])
        df = progs.groupby("zone_id")["program_id"].apply(list)
        zone2prog = df.to_dict()
        aa2prog = {t: zone2prog[v] for t, v in self.zone_dict.items()}

        if save_path == "":
            save_path = "~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/"
        filename = os.path.expanduser(save_path + "opt" + self.program_type)
        filename += str(rand) + ".txt"

        # save zones themselves
        with open(filename, "w") as f:
            f.write(str(aa2prog))

        # save optimization parameters
        with open(filename[:-4] + "_params.txt", "w") as f:
            for k, v in self.constraints.items():
                f.write("".join(k.split()) + " " + str(v) + "\n")
            f.write("centroids " + str([self.idx2area[x] for x in self.centroids]))

    def solve(self, write=False, save_path="~/SFUSD/"):
        self.filename = ""
        self.zone_dict = {}

        try:
            self.m.Params.TimeLimit = 350
            self.m.optimize()
            if self.move_SpEd:
                for t in range(self.T):
                    self.eligible_sc_df[str(self.SpEd_types[t] + "_classes_new")] = 0
                    for s in range(self.S):
                        self.eligible_sc_df[str(self.SpEd_types[t] + "_classes_new")][s] = self.q[t,s].X
                self.eligible_sc_df.to_csv('~/Desktop/SpEd_relocation.csv')

                # P1 = []
                # for s in range(self.S):
                #     P1.append([])
                #     for j in range(self.A):
                #         P1[s].append(self.p[1, j, s].X)
                # print("p matrix for t=1  " + str(P1))
                print("y_SpEd Objective:  " + str(self.y_SpEd.X))

            else:
                zones = []
                for z in range(0, self.M):
                    zone = []
                    for j in range(0, self.A):
                        if self.x[j, z].X >= 0.999:
                            self.zone_dict[self.idx2area[j]] = z
                            zone.append(self.area_data[self.level][j])
                            # add City wide school SF Montessori, even if we are not including city wide schools
                            # 823 is the aa level of SF Montessori school (which has school id 814)
                            if self.idx2area[j] in [823, 60750132001]:
                                self.zone_dict[self.idx2area[j]] = z
                                if self.level == "idschoolattendance":
                                    zone.append(SF_Montessori)
                    if not zone == False:
                        zones.append(zone)
                zone_dict = {}
                for idx, schools in enumerate(zones):
                    zone_dict = {
                        **zone_dict,
                        **{int(float(s)): idx for s in schools if s != ""},
                    }
                # add K-8 schools to dict if using them
                if (self.level == 'idschoolattendance') & (self.include_k8):
                    cw = self.sc_df.loc[self.sc_df["K-8"] == 1]
                    for i, row in cw.iterrows():
                        k8_schno = row["school_id"]
                        z = zone_dict[self.sch2block[int(float(k8_schno))]]
                        zone_dict = {**zone_dict, **{int(float(k8_schno)): z}}
                        zones[z].append(k8_schno)
                self.zd = zone_dict
                self.z = zones

                if write:
                    self.save(save_path)

            return 1

        except gp.GurobiError as e:
            print("gurobi error #" + str(e.errno) + ": " + str(e))
            return -1
        except AttributeError:
            print("attribute error")
            return -1



def zone_assignment_process(param):
    name = compute_name(param)
    print(name)
    if os.path.exists(param.path +"GE" + name + "_AA" +".csv"):
        return

    input_level = 'idschoolattendance'
    # input_level = 'BlockGroup'
    dz = DesignZones(
        M=param.zone_count,
        level=input_level,
        centroids_type=param.centroids_type,
        include_k8=param.include_k8,
        population_type=param.population_type,
        program_type="GE",
        drop_optout=True,
        capacity_scenario="A",
        new_schools=True,
        move_SpEd=param.move_SpEd,
        use_loaded_data=True
    )
    dz.set_model(shortage=param.shortage, overage= param.overage, all_cap_shortage=param.all_cap_shortage, balance=param.balance)

    if param.move_SpEd:
        dz.solve()
        zv = ZoneVisualizer('BlockGroup')
        zv.visualize_SpEd(sped_df=dz.eligible_sc_df, sped_types=dz.SpEd_types, centroid_location=dz.centroid_location)
    else:
        dz._add_geo_constraints(max_distance=param.max_distance, contiguity=True)
        dz._add_diversity_constraints(racial_dev=param.racial_dev, frl_dev=param.frl_dev)
        dz._add_school_count_constraint()
        # if param.include_sch_qlty_balance == True:
        #     dz._add_met_quality_constraint(min_pct = param.lbscqlty)
        # dz._boundary_threshold_constraint(boundary_threshold = param.boundary_threshold)

        solve_success = dz.solve()

        if solve_success == 1:
            print("Resulting zone dictionary:")
            print(dz.zd)

            dz.save(path=param.path, name = name + "_AA")

            zv = ZoneVisualizer(input_level)
            zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location, save_name= "GE" + name + "_AA")

            # stats_evaluation(dz, dz.zd)
        elif solve_success == -1:
            dz.save(path=param.path, name = name + "_AA", solve_success = solve_success)

        del dz
        gc.collect()

if __name__ == "__main__":
    param = Tuning_param()
    # zone_assignment_process(param)

    for frl_dev in [0.15, 0.1]:
        param.frl_dev = frl_dev
        for racial_dev in [0.15, 0.12]:
            param.racial_dev = racial_dev
            for include_k8 in [True, False]:
                param.include_k8 = include_k8
                with open("centroids.yaml", "r") as f:
                    centroid_options = yaml.safe_load(f)
                    for centroids in centroid_options:
                        param.zone_count = int(centroids.split("-")[0])
                        param.centroids_type = centroids
                        print("param: " + str(param.frl_dev) + " " + str(param.racial_dev)
                              + " " + str(param.include_k8))
                        zone_assignment_process(param)




# Note: Total number of students in aa level is not the same as blockgroup level.
# Reason: some students, do not have their bg info available
# (but they do have their aa info, and also they pass every other filter, i.e. enrollment)