import sys
import ast
import yaml
import pickle
import random, math, gc, os, csv
from typing import Union
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
import geopandas as gpd

sys.path.append("../..")
from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.graph_shortest_path import Shortest_Path
from Zone_Generation.Config.Constants import *
# from Helper_Functions.util import get_distance, make_school_geodataframe, load_bg2att
from Helper_Functions.util import *



def Compute_Name(config):
    name = str(config["centroids_type"])
    # # add frl deviation
    # name += "_frl_" + str(config["frl_dev"])
    # # add shortage
    # # name += "_shortage_" + str(config["shortage"])

    return name

def load_zones_from_file(file_path):
    zone_lists = []
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            # Convert each element in the row to an integer and store it in the list
            zone_row = []
            for cell in row:
                # Split the cell content by commas, convert to integers, and append to the row
                cell_values = [int(val.strip()) for val in cell.split(',') if val.strip()]  #
                zone_row.extend(cell_values)
            zone_lists.append(zone_row)

    # build a zone dictionary based on zone_list
    zone_dict = {}
    for index, sublist in enumerate(zone_lists):
        for item in sublist:
            zone_dict[item] = index

    return zone_lists, zone_dict


class DesignZones:
    def __init__(
            self,
            config,
            level="BlockGroup",
            drop_optout=True,
            capacity_scenario="A",
            new_schools=True,
            use_loaded_data=True
    ):
        self.config = config
        self.M = int(config["centroids_type"].split("-")[0])  # number of possible zones
        self.level = level  # 'Block', 'BlockGroup' or 'attendance_area'
        self.centroid_sch_option = config["centroids_type"]
        self.include_k8 = config["include_k8"]
        self.population_type = config["population_type"]

        # allowed capacity scenarios
        if capacity_scenario not in ["Old", "A", "B", "C", "D"]:
            raise ValueError(
                f"Unrecognized capacity scenario {capacity_scenario}. Please use one of allowed capacity scenarios.")
        self.capacity_scenario = capacity_scenario

        self.drop_optout = drop_optout
        self.new_schools = new_schools
        self.use_loaded_data = use_loaded_data

        self.load_students_and_schools()
        self.construct_datastructures()
        self.load_neighborhood_dict()
        self.initialize_centroids()
        self.initialize_centroid_neighbors()


    def construct_datastructures(self):


        self.seats = (self.area_data["ge_capacity"].astype("int64").to_numpy())
        self.schools = self.area_data['num_schools']

        self.N = sum(self.area_data["ge_students"])
        self.F = sum(self.area_data["FRL"]) / (self.N)

        self.A = len(self.area_data.index)
        self.studentsInArea = self.area_data["ge_students"]
        self.constraints = {"include_k8": self.include_k8}

        print("Average FRL ratio:       " + str(self.F))
        print("Number of Areas:       " + str(self.A))
        print("Number of GE students:       " + str(self.N))
        print("Number of total students: " + str(sum(self.area_data["all_prog_students"])))
        print("Number of total seats:    " + str(sum(self.area_data["all_prog_capacity"])))
        print("Number of GE seats:       " + str(sum(self.seats)))


        self.area_data[self.level] = self.area_data[self.level].astype("int64")
        self.area2idx = dict(zip(self.area_data[self.level], self.area_data.index))
        self.idx2area = dict(zip(self.area_data.index, self.area_data[self.level]))

        self.euc_distances = load_euc_distance_data(self.level)
        # self.save_partial_distances()

        # self.drive_distances = self.load_driving_distance_data()
        area2schools = {}

        for area in self.area2idx:
            idx = self.area2idx[area]
            area2schools[area] = self.schools[idx]

    def save_partial_distances(self):
        self.euc_distances = load_euc_distance_data(self.level, complete_bg=True)

        print("len(self.euc_distances))  ", len(self.euc_distances))
        school_blocks = list(self.sch2b.values())

        existing_school_blocks = [block for block in school_blocks if block in self.euc_distances.index]

        # pd.set_option('display.max_rows', None)
        print("self.euc_distances.index", list(self.euc_distances.index))
        print("school_blocks ", school_blocks)
        print("len(existing_school_blocks): ", len(existing_school_blocks))
        print("len((school_blocks)): ",  len((school_blocks)))
        distances = self.euc_distances.loc[existing_school_blocks]

        save_path = "~/Dropbox/SFUSD/Optimization/distances_b2b_schools.csv"
        distances.to_csv(save_path)

    def load_student_data(self):
        self.years = [14, 15, 16, 17, 18, 21, 22]

        cleaned_student_path = ("/Users/mobin/SFUSD/Data/Cleaned/Cleaned_Students_" +
                                '_'.join([str(year) for year in self.years]) + ".csv")
        if os.path.exists(cleaned_student_path):
            student_df = pd.read_csv(cleaned_student_path, low_memory=False)
            return student_df

        student_data_years = [0] * len(self.years)
        for i in range(len(self.years)):
            # load student data of year i
            student_data_years[i] = self._load_student_data(year=self.years[i])
        student_df = pd.concat(student_data_years, ignore_index=True)

        student_df.to_csv(cleaned_student_path, index=False)

        return student_df




    def load_students_and_schools(self):
        if self.use_loaded_data:
            self.student_df = self.load_student_data()
            self.school_df = self.load_school_data()


            student_stats = self._aggregate_student_data_to_area(self.student_df)
            school_stats = self._aggregate_school_data_to_area(self.school_df)

            self.area_data = student_stats.merge(school_stats, how='outer', on=self.level)

            self._load_auxilariy_areas()

            self.area_data.fillna(value=0, inplace=True)
            if self.level == "BlockGroup":
                self.bg2att = load_bg2att(self.level)

        else:
            if self.include_k8:
                self.area_data = pd.read_csv(f"~/Dropbox/SFUSD/Data/Cleaned/final_area_data/area_data_k8.csv", low_memory=False)
            else:
                self.area_data = pd.read_csv(f"~/Dropbox/SFUSD/Data/Cleaned/final_area_data/area_data_no_k8.csv", low_memory=False)

            #####################  rename columns of input data so it matches with current format ###################
            self.area_data.drop(["all_nonsped_cap", "ge_schools", "ge_capacity"], axis='columns', inplace=True)
            self.area_data.rename(columns={"enrolled_and_ge_applied": "ge_students",
                                       "enrolled_students": "all_prog_students",
                                       "census_blockgroup": "BlockGroup"}, inplace=True)

            self.area_data.dropna(subset=['BlockGroup'], inplace=True)

            self.area_data["ge_students"] = self.area_data["ge_students"] / 6
            self.area_data["all_prog_students"] = self.area_data["all_prog_students"] / 6


            sch_stats = self._load_school_data()
            self.area_data = self.area_data.merge(sch_stats, how='left', on="BlockGroup")


            for metric in ['frl_count', 'sped_count', 'ell_count',
                       'ge_students', 'all_prog_students', "all_prog_capacity", "ge_capacity",
                        'num_with_ethnicity', 'K-8'] + ETHNICITY_COLS:
                self.area_data[metric].fillna(value=0, inplace=True)


            # bg2att has thr following structural issue:
            # School S, might be in attendance area A, and in Blockgroup B.
            # But bg2att[B] != A. This is due to city structure, and that Blockgroups
            # are not always a perfect subset of only 1 attendance area.
            self.bg2att = load_bg2att(self.level)
            # We fix the structural issue in bg2aa, manually only for school locations.
            # Mapping of BGs to AA, for the location of schools
            bg2att_schools = dict(zip(self.school_df["BlockGroup"], self.school_df["attendance_area"]))
            self.bg2att.update(bg2att_schools)

            if self.level == 'attendance_area':
                # TODO 497 --> 0
                self.area_data['attendance_area'] =  self.area_data['BlockGroup'].apply(lambda x: self.bg2att[int(x)] if int(x) in self.bg2att else 497)
                self.area_data = self.area_data.groupby(self.level, as_index=False).sum()
                # both schools 603 and 723 are in the same bg 60750228031. while they are in different aa (603 and 723)
                # Increase 'ge_capacity' by 44 and (capacity of school 603) for rows where 'attendance_area' equals 603
                self.area_data.loc[self.area_data['attendance_area'] == 603, 'ge_capacity'] += 44
                self.area_data.loc[self.area_data['attendance_area'] == 603, 'all_prog_capacity'] += 92
                self.area_data.loc[self.area_data['attendance_area'] == 603, 'num_schools'] += 1
                self.area_data.loc[self.area_data['attendance_area'] == 723, 'ge_capacity'] -= 44
                self.area_data.loc[self.area_data['attendance_area'] == 723, 'all_prog_capacity'] -= 92
                self.area_data.loc[self.area_data['attendance_area'] == 723, 'num_schools'] -= 1

                self.area_data.reset_index(inplace=True)


    def _make_program_types(self, df):
        # look at rounds 1-4, and all the programs listed in those rounds
        # make a new column program_types, which is a list of all such program types over different rounds
        """ create column with each type of program applied to """
        for round in range(1, 4):
            col = "r{}_programs".format(round)
            if col in df.columns:
                if round == 1:
                    df["program_types"] = df[col]
                else:
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
                    df["program_types"] = df["program_types"] + df[col]
        df["program_types"] = df["program_types"].apply(lambda x: np.unique(x))
        return df

    def _filter_program_types(self, df, year):
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
                    f"~/SFUSD/Data/Cleaned/student_1920.csv", low_memory=False)
                student_data = student_data.dropna(subset=["enrolled_idschool"])

            else:
                student_data = pd.read_csv(
                    # f"~/SFUSD/Data/Cleaned/drop_optout_{year}{year + 1}.csv", low_memory=False
                    f"~/SFUSD/Data/Cleaned/enrolled_{year}{year + 1}.csv", low_memory = False)
        else:
            student_data = pd.read_csv(
                f"~/SFUSD/Data/Cleaned/student_{year}{year + 1}.csv", low_memory=False)

        student_data = student_data.loc[student_data["grade"] == "KG"]
        student_data['resolved_ethnicity'] = student_data['resolved_ethnicity'].replace(ETHNICITY_DICT)
        student_data.rename(columns={"census_block": "Block", "census_blockgroup": "BlockGroup", "idschoolattendance": "attendance_area", "FRL Score":"FRL"}, inplace=True)
        student_data.dropna(subset=BUILDING_BLOCKS, inplace=True)


        if year in [21,22]:
            student_data["ell_count"] = student_data["englprof"].apply(lambda x: 1 if x in ["N", "L"] else 0)
        else:
            student_data["ell_count"] = student_data["englprof_desc"].apply(
                lambda x: 1 if (x == "N-Non English" or x == "L-Limited English") else 0
            )
        student_data["enrolled_students"] = np.where(student_data["enrolled_idschool"].isna(), 0, 1)



        student_data['r1_programs'] = student_data['r1_programs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        for rnd in range(2, 6):
            col_name = f"r{rnd}_programs"

            # Check if the column exists
            if col_name in student_data.columns:
                # Identify rows where r1_programs is empty and the current round column contains a list
                condition = student_data['r1_programs'].apply(lambda x: len(x) <= 0) & student_data[col_name].apply(lambda x: isinstance(x, list))

                # For rows meeting the condition, update r1_programs with the value from the current round column
                student_data.loc[condition, 'r1_programs'] = student_data.loc[condition, col_name]
        #
        student_data = student_data.loc[student_data['r1_programs'].apply(lambda x: x != [])]

        # Each student counts as a full all_prog_student
        # and a partial (between [0,1]) ge_student
        student_data["all_prog_students"] = 1
        student_data["ge_students"] = student_data.apply(
            lambda x: sum([i == "GE" for i in x["r1_programs"]]) / len(x["r1_programs"])
            if x["enrolled_students"] == 1 and len(x["r1_programs"]) > 0
            else 0,
            axis=1
        )

        student_data = pd.get_dummies(
            student_data, columns=["resolved_ethnicity"]
        )

        #  Make sure we only count the students statistics
        #  (i.e. frl, racial) if the student is GE
        for col in ETHNICITY_COLS + ['FRL']:
            student_data[col] = student_data.apply(
                lambda x: x[col] * x["ge_students"],
                axis=1
            )

        # # Load FRL
        # # TODO check if I need this new approach to compute FRL
        # cbeds = pd.read_excel(CBEDS_SBAC_PATH, sheet_name="CBEDS2122")
        # cbeds.rename(columns={'Geoid10': 'census_block'}, inplace=True)
        # cbeds['frl'] = cbeds["FRPM by block"]/cbeds["Distinct count of Student No"]
        # print("student_data.columns ", student_data.columns)
        # student_data = student_data.merge(cbeds[['census_block', 'frl']], how='left', on='census_block')



        student_data = self._make_program_types(student_data)
        student_data = self._filter_program_types(student_data, year)

        student_data = student_data[IMPORTANT_COLS + ETHNICITY_COLS]

        # Fill NaN values in columns with the mean value
        for col in ["FRL", "AALPI Score"]:
            mean_value = student_data[col].mean()
            student_data['FRL'].fillna(value=mean_value, inplace=True)

        # Fill NaN values in the remaining columns with 0
        student_data.fillna(value=0, inplace=True)

        return student_data


    # groupby the student data of year i, such that we have the information only on area level
    def _aggregate_student_data_to_area(self, student_df):
        # sum_columns = list(student_df.columns)
        # sum_columns.remove("FRL")
        # mean_columns = [self.level, "FRL"]
        #
        # sum_students = student_df[sum_columns].groupby(self.level, as_index=False).sum()
        # mean_students = student_df[mean_columns].groupby(self.level, as_index=False).mean()
        #
        # student_stats = mean_students.merge(sum_students, how="left", on=self.level)
        student_stats =  student_df.groupby(self.level, as_index=False).sum()
        student_stats = student_stats[AREA_COLS + [self.level]]

        for col in student_stats.columns:
            if col not in BUILDING_BLOCKS:
                student_stats[col] /= len(self.years)
        return student_stats

    def _aggregate_school_data_to_area(self, school_df):

        sum_columns = [self.level, "all_prog_capacity", "ge_capacity", "num_schools"]
        mean_columns = [self.level, "eng_scores_1819", "math_scores_1819", "greatschools_rating", "MetStandards", "AvgColorIndex"]

        sum_schools = school_df[sum_columns].groupby(self.level, as_index=False).sum()
        mean_schools = school_df[mean_columns].groupby(self.level, as_index=False).mean()

        return mean_schools.merge(sum_schools, how="left", on=self.level)


    def _load_auxilariy_areas(self):
        # we add areas (blockgroups/blocks) that were missed from guardrail, since there was no student or school in them.
        if (self.level=='BlockGroup') | (self.level=='Block'):
            valid_areas = set(pd.read_csv('~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv')[self.level])
            census_areas = load_census_shapefile(self.level)[self.level]
            census_areas = set(census_areas)
            census_areas = census_areas - set(AUX_BG)

            common_areas = census_areas.intersection(valid_areas)

            current_areas = set(self.area_data[self.level])

            auxiliary_areas = common_areas - current_areas

            auxiliary_areas_df = pd.DataFrame({self.level: list(auxiliary_areas)})
            self.area_data = self.area_data.append(auxiliary_areas_df, ignore_index=True)
            self.area_data.fillna(value=0, inplace=True)



    def _load_capacity(self, school_df):
        # add on capacity
        programs = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/stanford_capacities_12.23.21.csv")
        programs.rename(
            columns={
                "SchNum": "school_id",
                "PathwayCode": "program_type",
                f"Scenario_{self.capacity_scenario}_Capacity": "Capacity",
            }, inplace=True
        )

        prog_ge = programs.loc[programs["program_type"] == "GE"][["school_id", "Capacity"]]
        prog_ge.rename(columns={"Capacity": "ge_capacity"}, inplace=True)

        prog_all = programs[["school_id", "Capacity"]].rename(columns={"Capacity": "all_prog_capacity"})
        prog_all = prog_all.groupby("school_id", as_index=False).sum()

        school_df = school_df.merge(prog_all, how="inner", on="school_id")
        school_df = school_df.merge(prog_ge, how="inner", on="school_id")

        school_df = school_df.loc[school_df['ge_capacity'] > 0]

        if self.include_k8:
            school_df = school_df.loc[:, :]
        else:
            school_df = school_df.loc[school_df["K-8"] == 0]

        return school_df



    def load_school_data(self):
        # Load School Dataframe. Map School Name to AA number
        if self.new_schools:
            # school_df = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv")
            school_df = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development_updated.csv")
        else:
            school_df = pd.read_csv(f"~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv")

        school_df.rename(columns={"attendance_area": "attendance_area"}, inplace=True)

        school_df[self.level] = school_df[self.level].astype('Int64')

        if self.level == "attendance_area":
            self.sch2aa = dict(zip(school_df["school_id"], school_df["attendance_area"]))
        elif self.level == "BlockGroup":
            self.sch2bg = dict(zip(school_df["school_id"], school_df["BlockGroup"]))
        elif self.level == "Block":
            self.sch2b = dict(zip(school_df["school_id"], school_df["Block"]))

        school_df = school_df.loc[school_df["category"] != "Citywide"]

        # Load School Capacities according to capacity policy
        school_df["K-8"] = school_df["school_id"].apply(lambda x: 1 if ((x in K8_SCHOOLS)) else 0)
        # TODO Manually check to make make sure only one of schools 999 or 909 are included in the data
        # school_df = school_df.loc[school_df["school_id"] != 999]

        school_df = self._load_capacity(school_df)
        school_df = self._compute_school_popularity(school_df)
        school_df = self._inflate_capacity(school_df)
        school_df["num_schools"] = 1

        return school_df


    # Compute the popularity of each school, and add it as an additional column, ge_popularity
    # Popularity computation for each school s:
    # (Number of GE students that selected school s as top choice) / (real capacity of school s)
    def _compute_school_popularity(self, school_df):
        # Convert "r1_ranked_idschool" from string to list
        self.student_df['r1_ranked_idschool'] = self.student_df['r1_ranked_idschool'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

        # Extract top 1 school for each student and map it to its zone
        self.student_df['top1_school'] = self.student_df['r1_ranked_idschool'].apply(lambda x: x[0] if x else None)

        # Sum the ge_students values for each school ranked as top choice
        top1_school_weights = self.student_df.groupby('top1_school')['ge_students'].sum().reset_index()
        # rename the ge_demand column, and normalize the demand by the number of years.
        top1_school_weights["ge_student_demand"] = top1_school_weights["ge_students"].astype(float) / len(self.years)

        top1_school_weights["school_id"] = top1_school_weights["top1_school"].astype(int)
        top1_school_weights = top1_school_weights[['school_id', 'ge_student_demand']]

        # Merge this weighted sum with the school_df DataFrame
        school_df = school_df.merge(top1_school_weights, on='school_id', how='left')

        # For schools in Mission_Bay, there is no historical data, so manually inflate popularity
        school_df.loc[school_df['school_id'].isin(Mission_Bay), 'ge_student_demand'] =\
            school_df.loc[school_df['school_id'].isin(Mission_Bay), 'ge_capacity']

        # Calculate the weighted popularity of each school
        school_df['ge_popularity'] = school_df['ge_student_demand'] / school_df['ge_capacity']

        return school_df

    def calculate_inflated_capacity(self, row):
        decay_power = 1.2
        if row['ge_popularity'] > 1:
            return row['ge_capacity']
        elif 0.5 < row['ge_popularity'] <= 1:
            return row['ge_capacity'] *  (1/row['ge_popularity']) ** decay_power
        elif row['ge_popularity'] <= 0.5:
            return 2**decay_power * row['ge_capacity']
    def _inflate_capacity(self, school_df):
        # Apply the function to calculate inflated_ge_capacity row by row, for each school
        school_df['inflated_ge_capacity'] = school_df.apply(self.calculate_inflated_capacity, axis=1)
        return school_df

    def initialize_centroids(self):
        """set the centroids - each one is a block or attendance area depends on the method
        probably best to make it a school"""

        with open("../Config/centroids.yaml", "r") as f:
            centroid_configs = yaml.safe_load(f)
        if self.centroid_sch_option not in centroid_configs:
            raise ValueError(
                "The centroids type specified is not defined in centroids.yaml.")

        self.centroid_sch = centroid_configs[self.centroid_sch_option]

        self.school_df['is_centroid'] = self.school_df['school_id'].apply(lambda x: 1 if x in self.centroid_sch else 0)
        if self.include_k8:
            self.centroid_location = self.school_df[self.school_df['is_centroid'] == 1][['lon', 'lat', 'school_id']]
        else:
            self.centroid_location = self.school_df[(self.school_df['is_centroid'] == 1) & (self.school_df['K-8'] != 1)][['lon', 'lat', 'school_id']]


        if self.level == "attendance_area":
            centroid_aa = [self.sch2aa[x] for x in self.centroid_sch]
            self.centroids = [self.area2idx[j] for j in centroid_aa]
        elif self.level == "BlockGroup":
            centroid_bg = [self.sch2bg[x] for x in self.centroid_sch]
            self.centroids = [self.area2idx[j] for j in centroid_bg]
        elif self.level == "Block":
            centroid_b = [self.sch2b[x] for x in self.centroid_sch]
            self.centroids = [self.area2idx[j] for j in centroid_b]

        self.constraints["centroidsType"] = self.centroid_sch_option

    def load_neighborhood_dict(self):
        """ build a dictionary mapping a block group/attendance area to a list
        of its neighboring block groups/attendnace areas"""
        if self.level == "Block":
            file = os.path.expanduser("~/Dropbox/SFUSD/Optimization/adjacency_matrix_b.csv")

        elif self.level == "BlockGroup":
            file = os.path.expanduser("~/Dropbox/SFUSD/Optimization/adjacency_matrix_bg.csv")

        elif self.level == "attendance_area":
            file = os.path.expanduser("~/Dropbox/SFUSD/Optimization/adjacency_matrix_aa.csv")

        with open(file, "r") as f:
            reader = csv.reader(f)
            neighborhoods = list(reader)

        # create dictionary mapping attendance area school id to list of neighbor
        # attendance area ids (similarly, block group number)
        self.neighbors = {}
        for row in neighborhoods:
            # Potential Issue: row[0] is an area number from the neighborhood adjacency matrix,
            # and it should be included as a key in area2idx map.
            if int(row[0]) not in self.area2idx:
                continue
            u = self.area2idx[int(row[0])]
            ngbrs = [
                self.area2idx[int(n)]
                for n in row
                if n != ''
                   and int(n) in list(self.area2idx.keys())
            ]
            ngbrs.remove(u)
            self.neighbors[u] = [n for n in ngbrs]
            for n in ngbrs:
                if n in self.neighbors:
                    if u not in self.neighbors[n]:
                        self.neighbors[n].append(u)
                else:
                    self.neighbors[n] = [u]

        # self.check_neighbormap()

    # check validity of neighborhood map
    def check_neighbormap(self):
        counter = 0
        for area in self.neighbors:
            for neighbor in self.neighbors[area]:
                if area not in self.neighbors[neighbor]:
                    raise ValueError(
                        "The Neighborhood Map Is Not Valid. There Is a Directed Neighboring Relationship.")
    def load_geodesic_neighbors(self):
        # self.closer_geodesic_neighbors = {}

        self.shortestpath = Shortest_Path(self.neighbors, self.centroids)
        print("Pairwise Shortestpath Distance: ", self.shortestpath)

        for c in self.centroids:
            # for area in range(self.A):
            for area in self.candidate_idx:
                closer_geod = []
                for n in self.neighbors[area]:
                    if self.shortestpath[n, c] < self.shortestpath[area, c]:
                        closer_geod.append(n)
                self.closer_geodesic_neighbors[area, c] = closer_geod

    def initialize_centroid_neighbors(self):
        """ for each centroid c and each area j, define a set n(j,c) to be all neighbors of j that are closer to c than j"""

        # pd.set_option('display.max_rows', None)

        # self.load_geodesic_neighbors()
        # A = 60750101001017
        # idx_A = self.area2idx[A]
        # if idx_A in range(self.A):
        #     print("Yes")
        # if A in self.euc_distances.index:
        #     print("Also in distances")
        # print("self.euc_distances.columns", list(self.euc_distances.columns))
        # print("self.euc_distances.loc[60750327004002]:  ", self.euc_distances.loc[60750327004002])

        self.euc_distances.dropna(inplace=True)

        save_path = os.path.expanduser("~/Dropbox/SFUSD/Optimization/59zone_contiguity_constraint.pkl")

        if (self.level == "Block") and (self.centroid_sch_option == '59-zone-1'):
            if os.path.exists(os.path.expanduser(save_path)):
                with open(save_path, 'rb') as file:
                    self.closer_euc_neighbors = pickle.load(file)
                return


        self.closer_euc_neighbors = {}
        for c in self.centroids:
            for area in range(self.A):
                n = self.neighbors[area]
                # for x in n:
                    # print("self.idx2area[c]:  ", self.idx2area[c], "   self.idx2area[area]: ", self.idx2area[area], "  self.idx2area[x]):  ", self.idx2area[x])
                    # print("self.euc_distances.loc[self.idx2area[c], self.idx2area[area]]  ", self.euc_distances.loc[self.idx2area[c], str(self.idx2area[area])])
                    # print("self.euc_distances.loc[self.idx2area[c], (self.idx2area[x])]  ", self.euc_distances.loc[self.idx2area[c], str(self.idx2area[x])])
                # print(self.euc_distances)
                # print(self.euc_distances.columns)
                closer = [x for x in n
                    if self.euc_distances.loc[self.idx2area[c], str(self.idx2area[area])]
                       >= self.euc_distances.loc[self.idx2area[c], str(self.idx2area[x])]
                ]
                self.closer_euc_neighbors[area, c] = closer

        if (self.level == "Block") and (self.centroid_sch_option == '59-zone-1'):
            with open(save_path, 'wb') as file:
                pickle.dump(self.closer_euc_neighbors, file)

    # ---------------------------------------------------------------------------

    def set_y_distance(self):
        self.y_distance = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="distance distortion")

        for z in range(self.M):
            centroid_area =  self.idx2area[self.centroids[z]]
            zone_dist_sum = gp.quicksum([((self.euc_distances.loc[centroid_area, str(self.idx2area[j])]) ** 2) * self.x[j, z] for j in range(self.A)])
            # zone_dist_sum = gp.quicksum([((self.drive_distances.loc[centroid_area, str(self.idx2area[j])]) ** 2) * self.x[j, z] for j in range(self.A)])
            # zone_dist_sum = gp.quicksum([((self.drive_distances.loc[centroid_area, str(self.idx2area[j])]) * (self.studentsInArea[j]) * self.x[j, z] for j in range(self.A)])
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



    def set_y_boundary(self):
        neighboring_tuples = []
        for i in range(self.A):
            for j in self.neighbors[i]:
                if i >= j:
                    continue
                if ((i,j) in self.samezone_pairs) or ((j,i) in self.samezone_pairs):
                    continue
                neighboring_tuples.append((i,j))

        print("Setting boundary objective function")
        self.b = self.m.addVars(neighboring_tuples, vtype=GRB.BINARY, name="boundary_vars")
        self.y_boundary = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="boundary distortion")
        self.m.addConstr(gp.quicksum(self.b[i, j] for i, j in neighboring_tuples) == self.y_boundary)
        self._add_boundary_constraint()

    def _add_boundary_constraint(self):
        # if i and j are neighbors, check if they are boundaries of different zones
        for i in range(self.A):
            for j in self.neighbors[i]:
                if i >= j:
                    continue
                if ((i,j) in self.samezone_pairs) or ((j,i) in self.samezone_pairs):
                    continue
                for z in range(self.M):
                    if (i in self.valid_area_per_zone[z]) and (j in self.valid_area_per_zone[z]):
                        self.m.addConstr(gp.quicksum([self.x[i, z], -1 * self.x[j, z]]) <= self.b[i, j])
                        self.m.addConstr(gp.quicksum([-1 * self.x[i, z], self.x[j, z]]) <= self.b[i, j])
                    elif (i in self.valid_area_per_zone[z]):
                        self.m.addConstr(self.x[i, z] <= self.b[i, j])
                    elif (j in self.valid_area_per_zone[z]):
                        self.m.addConstr(self.x[j, z] <= self.b[i, j])



    def _set_objective_model(self, max_distance=-1):
        # closest_sch_dist = {}
        # for i in range(self.A):
        #     min_dist = 10000
        #     for centroid_z in self.centroids:
        #         distance_z = self.euc_distances.loc[self.idx2area[centroid_z], str(self.idx2area[i])]
        #         if distance_z < min_dist:
        #             min_dist = distance_z
        #             closest_sch_dist[i] = distance_z
        # sorted_dict = sorted(closest_sch_dist.items(), key=lambda item: item[1])
        # print(sorted_dict)
        # exit()

        corner_blocks = [60750127001008, 60750601001058, 60750601001092, 60750601001013, 60750601001028, 60750127001009, 60750601001053, 60750601001047, 60750601001091, 60750601001059, 60750604001011, 60750601001080, 60750127001007, 60750601001073, 60750601001032, 60750601001042, 60750601001009, 60750601001060,
                         60750601001082, 60750601001081, 60750601001049, 60750601001057, 60750127001001, 60759806001055, 60750601001065, 60750601001046, 60759806001041, 60750601001056, 60750601001052, 60750604001013, 60750601001035, 60759806001040, 60750601001084, 60750601001051, 60750601001055, 60750601001014,
                         60750601001048, 60759806001052, 60750601001050, 60750601001003, 60750601001054, 60759806001043, 60750601001004, 60750601001015, 60750601001085, 60750601001005, 60750601001031, 60750601001037, 60750601001006, 60750601001030, 60750604001032, 60750601001034, 60750604001012, 60750601001036,
                         60750601001038, 60750601001039, 60750601001002, 60750601001008, 60750601001029, 60750601001040, 60750604001033, 60750601001041, 60750601001019, 60750601001186, 60750601001018, 60750601001007, 60750601001010, 60750179021022, 60750179021020, 60750179021021, 60750179021079, 60750179021063,
                         60750179021011, 60750179021006, 60750179021062, 60750179021061, 60750179021040, 60750179021059, 60750179021018, 60750179021031, 60750179021058, 60750179021012, 60750179021060, 60750179021015, 60750179021037, 60750179021055, 60750179021016, 60750179021030, 60750179021050, 60750179021056,
                         60750179021053, 60750179021054, 60750179021010, 60750179021057, 60750179021052, 60750179021001, 60750179021045, 60750179021017, 60750179021036, 60750179021038, 60750179021041, 60750179021044, 60750179021008, 60750179021023, 60750179021034, 60750179021042, 60750179021014, 60750179021051,
                         60750179021043, 60750179021026, 60750179021048, 60750179021046, 60750179021035, 60750179021049, 60750179021047, 60750179021039, 60750179021032, 60750179021033, 60750179021025, 60750179021028, 60750179021029, 60750179021013, 60750179021027, 60750179021019, 60750179021024]


        self.m = gp.Model("Zone model")
        valid_assignments = []
        # if a max distance constraint is given, allow areas to be matched only to
        # zone centroids that are closer than max_distance
        if max_distance > 0:
            for z in range(self.M):
                centroid_z = self.centroids[z]
                # zone_max_distance = max_distance
                for i in range(self.A):
                    if (self.euc_distances.loc[self.idx2area[centroid_z], str(self.idx2area[i])] < max_distance):
                        valid_assignments.append((i,z))
                    else:
                        if (self.euc_distances.loc[self.idx2area[centroid_z], str(self.idx2area[i])] > 3.5):
                            continue
                        area_i = self.idx2area[i]
                        if area_i not in corner_blocks:
                            continue
                        valid_assignments.append((i,z))

        else:
            for z in range(self.M):
                for i in range(self.A):
                        valid_assignments.append((i,z))

        self.x = self.m.addVars(valid_assignments, vtype=GRB.BINARY, name="x")


        # Initialize a dictionary to hold valid zones for each area
        self.valid_area_per_zone = {z: [] for z in range(self.M)}
        # Initialize a dictionary to hold valid zones for each area
        self.valid_zone_per_area = {i: [] for i in range(self.A)}

        # Populate the dictionary with valid zones for each area
        for i, z in valid_assignments:
            self.valid_area_per_zone[z].append(i)
            self.valid_zone_per_area[i].append(z)

        west_blocks = [60750426011002, 60750427001001, 60750427001002, 60750427001003, 60750427001004, 60750427001005, 60750427002000, 60750427002001, 60750427002002, 60750427002003, 60750427002004, 60750427002006, 60750427003000, 60750427003001, 60750427003002, 60750427003003, 60750427003004, 60750427003005, 60750427003007, 60750427003008, 60750428001004, 60750428001006, 60750428001009, 60750428001010, 60750428002006, 60750428002008, 60750428002014, 60750477011001, 60750477011002, 60750477011004, 60750477011006, 60750477011008, 60750477011010, 60750477012006, 60750477012007, 60750477013000, 60750477013001, 60750477013002, 60750477013003, 60750477013004, 60750477013005, 60750477021000, 60750477021001, 60750477021002, 60750477021003, 60750477021004, 60750477021005, 60750477022002, 60750477023000, 60750477023001, 60750477023002, 60750477023003, 60750477023004, 60750477023005, 60750478011000, 60750478011001, 60750478011002, 60750478011003, 60750478011004, 60750478011005, 60750478012000, 60750478012001, 60750478012002, 60750478012003, 60750478012004, 60750478012005, 60750478013001, 60750478013002, 60750478013004, 60750478013005, 60750478013006, 60750478013007, 60750478013009, 60750478013010, 60750478013011, 60750478021000, 60750478021001, 60750478021002, 60750478021003, 60750478021004, 60750478021005, 60750478021008, 60750478022000, 60750478022001, 60750478022002, 60750478022003, 60750478022004, 60750478022005, 60750478023001, 60750478023002, 60750478023004, 60750478023006, 60750478023008, 60750478023010, 60750479011000, 60750479011001, 60750479011002, 60750479011003, 60750479011004, 60750479011005, 60750479012009, 60750479012010, 60750479012011, 60750479012013, 60750479012014, 60750479012015, 60750479012016, 60750479013000, 60750479013001, 60750479013002, 60750479013003, 60750479013004, 60750479013005, 60750479013006, 60750479013007, 60750479014000, 60750479014001, 60750479014002, 60750479014003, 60750479014004, 60750479014005, 60750479014006, 60750479015000, 60750479015001, 60750479015002, 60750479015003, 60750479015004, 60750479015005, 60750479015006, 60750479015007, 60750479021000, 60750479021001, 60750479021002, 60750479021003, 60750479021005, 60750479021006, 60750479021007, 60750479021008, 60750479021009, 60750479022000, 60750479022001, 60750479022002, 60750479022003, 60750479022007, 60750479022008, 60750479022011, 60750479022012, 60750479023000, 60750479023001, 60750479023002, 60750479023004, 60750479023005, 60750479023008, 60759802001004, 60750351001000, 60750351001003, 60750351001006, 60750327007000, 60750327007003, 60750327007005, 60759803001013, 60759803001014, 60759803001015, 60759803001016, 60759803001017, 60759803001018, 60759803001019, 60759803001020, 60759803001026, 60759803001027, 60759803001029, 60759803001030, 60759803001031, 60759803001032, 60759803001033, 60750477011000, 60750477011003, 60750477011005, 60750477011007, 60750477011009, 60750477011011, 60750477012008, 60750479012004, 60750479012005, 60750479012006, 60750479012007, 60750479012008, 60750479012012, 60750428001007, 60750478021006, 60750478021007, 60750478013000, 60750478013003, 60750478013008, 60750478013012, 60750428001005, 60750428001008, 60750479021004, 60750427002005, 60759802001001, 60759802001002, 60759802001003, 60750428002004, 60750428002007, 60750428002009, 60750427003006, 60759803001061, 60759803001072, 60750478023000, 60750478023003, 60750478023005, 60750478023007, 60750478023009, 60750479023003, 60750479023006, 60750479023007, 60750479022004, 60750479022005, 60750479022006, 60750479022009, 60750479022010, 60750327001004, 60750327001006, 60750327001008, 60750327001010, 60750327002001, 60750327002002, 60750327002003, 60750327002004, 60750327002005, 60750327003000, 60750327003001, 60750327003002, 60750327003003, 60750327003004, 60750327003005, 60750327004000, 60750327004001, 60750327004002, 60750327004003, 60750327004004, 60750327004005, 60750327005001, 60750327005002, 60750327005003, 60750327005004, 60750327005005, 60750327006000, 60750327006001, 60750327006002, 60750327006003, 60750327006004, 60750327006005, 60750327007001, 60750327007002, 60750327007007, 60750327007008, 60750327007013, 60750329011005, 60750329011007, 60750329021001, 60750329021002, 60750329021003, 60750329021004, 60750329021005, 60750329021006, 60750329021007, 60750329022002, 60750329022003, 60750329022004, 60750329022005, 60750329022006, 60750329022007, 60750329023003, 60750329023004, 60750329023005, 60750329023006, 60750329023007, 60750351001002, 60750351001005, 60750351001007, 60750351001009, 60750351001011, 60750351001013, 60750351002001, 60750351002002, 60750351002003, 60750351002004, 60750351002005, 60750351002006, 60750351003001, 60750351003002, 60750351003003, 60750351003004, 60750351003005, 60750351003006, 60750351004001, 60750351004002, 60750351004003, 60750351004004, 60750351004005, 60750351004006, 60750351005001, 60750351005002, 60750351005003, 60750351005004, 60750351005005, 60750351005006, 60750351006001, 60750351006002, 60750351006003, 60750351006004, 60750351006005, 60750351006006, 60750351007001, 60750351007002, 60750351007003, 60750351007004, 60750351007005, 60750351007006, 60750351007007, 60750351007008, 60750352011000, 60750352011001, 60750352011002, 60750352011003, 60750352011004, 60750352011005, 60750352012005, 60750352012006, 60750352012007, 60750352012008, 60750352012009, 60750352012010, 60750352013000, 60750352013001, 60750352013002, 60750352013003, 60750352013004, 60750352013005, 60750352014000, 60750352014001, 60750352014002, 60750352014003, 60750352014004, 60750352014005, 60750352015000, 60750352015001, 60750352015002, 60750352015003, 60750352015004, 60750352015005, 60750352021000, 60750352021002, 60750352021004, 60750352021006, 60750352021008, 60750352021010, 60750352022000, 60750352022001, 60750352022002, 60750352022003, 60750352022004, 60750352022005, 60750352022006, 60750352023000, 60750352023001, 60750352023002, 60750352023003, 60750352023004, 60750352023005, 60750353001008, 60750353001009, 60750353006000, 60750353006001, 60750353006002, 60750353006003, 60750354001000, 60750354001001, 60750354001002, 60750354001004, 60750354001009, 60750354001010, 60750354002000, 60750354002001, 60750354002002, 60750354002003, 60750354002004, 60750354002005, 60750329011006, 60750353001001, 60750329021008, 60750329021009, 60750327006006, 60750327006007, 60750327006008, 60750327006009, 60750351001000, 60750351001001, 60750351001003, 60750351001004, 60750351001006, 60750351001008, 60750351001010, 60750351001012, 60750351001014, 60750327007000, 60750327007003, 60750327007004, 60750327007005, 60750327007006, 60750327007009, 60750327007010, 60750327007011, 60750327007012, 60750327007014, 60759803001019, 60759803001024, 60759803001025, 60759803001026, 60759803001027, 60759803001028, 60759803001029, 60759803001032, 60750351002000, 60750329023008, 60750329023009, 60750353001000, 60750353001002, 60750353001003, 60750351003000, 60750327001005, 60750327001007, 60750327001009, 60750327001011, 60750351004000, 60750352012000, 60750352012002, 60750352012003, 60750352012001, 60750351005000, 60750352021001, 60750352021003, 60750352021005, 60750352021007, 60750352021009, 60750352021011, 60750352021012, 60750352021013, 60759803001061, 60750351006000, 60750329011008, 60750329011009, 60750351007000, 60750352023006, 60750327005000, 60750329022008, 60750329022009, 60750326021010, 60750327003004, 60750327003005, 60750327004001, 60750327004002, 60750327004003, 60750327004004, 60750327004005, 60750327005001, 60750327005002, 60750327005003, 60750327005004, 60750327005005, 60750327006000, 60750327006001, 60750327006002, 60750327006003, 60750327006004, 60750327006005, 60750327007013, 60750328012006, 60750328012007, 60750328013002, 60750328013003, 60750328013004, 60750328013005, 60750328013006, 60750328013007, 60750328021003, 60750328023004, 60750328023005, 60750328023006, 60750328023007, 60750329011000, 60750329011001, 60750329011002, 60750329011003, 60750329011004, 60750329011005, 60750329011007, 60750329012000, 60750329012001, 60750329012002, 60750329012003, 60750329012004, 60750329012005, 60750329012006, 60750329012007, 60750329013000, 60750329013001, 60750329013002, 60750329013003, 60750329013004, 60750329013005, 60750329013006, 60750329013007, 60750329014000, 60750329014001, 60750329014002, 60750329014003, 60750329014004, 60750329014005, 60750329014006, 60750329014007, 60750329021000, 60750329021001, 60750329021002, 60750329021003, 60750329021004, 60750329021005, 60750329021006, 60750329021007, 60750329022000, 60750329022001, 60750329022002, 60750329022003, 60750329022004, 60750329022005, 60750329022006, 60750329022007, 60750329023000, 60750329023001, 60750329023002, 60750329023003, 60750329023004, 60750329023005, 60750329023006, 60750329023007, 60750330004011, 60750330004012, 60750330004017, 60750330005000, 60750330005001, 60750330005002, 60750330005003, 60750330005004, 60750330005005, 60750330005006, 60750330005007, 60750330006002, 60750330006003, 60750330006004, 60750330006005, 60750330006006, 60750330006007, 60750351003001, 60750351003002, 60750351003003, 60750351003004, 60750351003005, 60750351003006, 60750351004001, 60750351004002, 60750351004003, 60750351004004, 60750351004005, 60750351004006, 60750351005001, 60750351005002, 60750351005003, 60750351005004, 60750351005005, 60750351005006, 60750351006001, 60750351006002, 60750351006003, 60750351006004, 60750351006005, 60750351006006, 60750351007001, 60750351007002, 60750351007003, 60750351007004, 60750351007005, 60750351007006, 60750351007007, 60750351007008, 60750352011000, 60750352011001, 60750352011002, 60750352011003, 60750352011004, 60750352011005, 60750352012005, 60750352012006, 60750352012007, 60750352012008, 60750352012009, 60750352012010, 60750352013000, 60750352013001, 60750352013002, 60750352013003, 60750352013004, 60750352013005, 60750352014000, 60750352014001, 60750352014002, 60750352014003, 60750352014004, 60750352014005, 60750352015000, 60750352015001, 60750352015002, 60750352015003, 60750352015004, 60750352015005, 60750352023000, 60750352023001, 60750352023002, 60750353001005, 60750353001006, 60750353001007, 60750353001008, 60750353001009, 60750353001010, 60750353002001, 60750353002002, 60750353002003, 60750353002004, 60750353002005, 60750353002006, 60750353002007, 60750353002008, 60750353003001, 60750353003002, 60750353003003, 60750353003004, 60750353003006, 60750353003007, 60750353003008, 60750353005000, 60750353005001, 60750353005002, 60750353005003, 60750353005004, 60750353006000, 60750353006001, 60750353006002, 60750353006003, 60750353006004, 60750353006005, 60750353006006, 60750353006007, 60750354001000, 60750354001001, 60750354001002, 60750354001004, 60750354001005, 60750354001009, 60750354001010, 60750354001011, 60750354001012, 60750354002000, 60750354002001, 60750354002002, 60750354002003, 60750354002004, 60750354002005, 60750354002006, 60750354002007, 60750354002008, 60750354003000, 60750354003001, 60750354003003, 60750354003004, 60750354003006, 60750354003007, 60750354004000, 60750354004001, 60750354004002, 60750354004003, 60750354004004, 60750354004005, 60750354004006, 60750354004007, 60750354004008, 60750354005000, 60750329011006, 60750353001001, 60750353003005, 60750328021004, 60750330004018, 60750330004019, 60750329021008, 60750329021009, 60750329013008, 60750329013009, 60750327006006, 60750327006007, 60750327006008, 60750327006009, 60750330005008, 60750330005009, 60750329014008, 60750329014009, 60750327007012, 60750327007014, 60750330006008, 60750330006009, 60750329023008, 60750329023009, 60750353001000, 60750353001002, 60750353001003, 60750353001004, 60750354001003, 60750354001006, 60750354001007, 60750351003000, 60750353002000, 60750353002009, 60750351004000, 60750353003000, 60750353003009, 60750352012002, 60750354003002, 60750354003005, 60750352012001, 60750351005000, 60750351006000, 60750329011008, 60750329011009, 60750351007000, 60750329012008, 60750329012009, 60750327005000, 60750329022008, 60750329022009, 60750329011007, 60750329012004, 60750329012005, 60750329012006, 60750329012007, 60750329013001, 60750329013002, 60750329013003, 60750329013004, 60750329013005, 60750329013006, 60750329013007, 60750329014000, 60750329014001, 60750329014002, 60750329014003, 60750329014004, 60750329014005, 60750329014006, 60750329014007, 60750330001008, 60750330002008, 60750330003006, 60750330003007, 60750330003008, 60750330003010, 60750330004005, 60750330004007, 60750330004008, 60750330004009, 60750330004010, 60750330004011, 60750330004012, 60750330004014, 60750330004015, 60750330004016, 60750330004017, 60750330004020, 60750330005000, 60750330005001, 60750330005002, 60750330005003, 60750330005004, 60750330005005, 60750330005006, 60750330005007, 60750330006000, 60750330006001, 60750330006002, 60750330006003, 60750330006004, 60750330006005, 60750330006006, 60750330006007, 60750331002007, 60750331003005, 60750331003006, 60750331003010, 60750331003012, 60750331003014, 60750331003015, 60750331003016, 60750331003017, 60750331004001, 60750331004003, 60750331004005, 60750331004009, 60750331004010, 60750331004011, 60750331004012, 60750331004016, 60750331004017, 60750331004018, 60750331004019, 60750331004020, 60750331004021, 60750331004022, 60750331004023, 60750331004024, 60750331004026, 60750353001005, 60750353001006, 60750353001007, 60750353001008, 60750353001009, 60750353001010, 60750353002001, 60750353002002, 60750353002003, 60750353002004, 60750353002005, 60750353002006, 60750353002007, 60750353002008, 60750353003001, 60750353003002, 60750353003003, 60750353003004, 60750353003006, 60750353003007, 60750353003008, 60750353004001, 60750353004002, 60750353004003, 60750353004004, 60750353004005, 60750353004006, 60750353005000, 60750353005001, 60750353005002, 60750353005003, 60750353005004, 60750353005005, 60750353005006, 60750353006000, 60750353006001, 60750353006002, 60750353006003, 60750353006004, 60750353006005, 60750353006006, 60750353006007, 60750354001000, 60750354001004, 60750354001005, 60750354001009, 60750354001010, 60750354001011, 60750354001012, 60750354002000, 60750354002001, 60750354002002, 60750354002003, 60750354002004, 60750354002005, 60750354002006, 60750354002007, 60750354002008, 60750354003000, 60750354003001, 60750354003003, 60750354003004, 60750354003006, 60750354003007, 60750354004000, 60750354004001, 60750354004002, 60750354004003, 60750354004004, 60750354004005, 60750354004006, 60750354004007, 60750354004008, 60750354005000, 60750354005001, 60750354005002, 60750354005003, 60750354005004, 60750354005005, 60750354005006, 60750354005007, 60750354005009, 60750354005010, 60750353001001, 60750353003005, 60750331003007, 60750331003008, 60750331003009, 60750331003011, 60750331003013, 60750330004004, 60750330004006, 60750330004013, 60750330004018, 60750330004019, 60750330004021, 60750330004022, 60750329013008, 60750329013009, 60750331004000, 60750331004002, 60750331004004, 60750331004006, 60750331004007, 60750331004008, 60750331004013, 60750331004014, 60750331004015, 60750331004025, 60750331004027, 60750331004028, 60750331004029, 60750331004030, 60750331004031, 60750331004032, 60750331004033, 60750331004034, 60750330005008, 60750330005009, 60750329014008, 60750329014009, 60750330006008, 60750330006009, 60750353001002, 60750353001003, 60750353001004, 60750354001003, 60750354001006, 60750354001007, 60750353002000, 60750353002009, 60750353003000, 60750353003009, 60750354003002, 60750354003005, 60750331003000, 60750331003001, 60750331003002, 60750331003003, 60750331003004, 60750353004000, 60750353004007, 60750353004008, 60750604001037, 60750329011008, 60750329011009, 60750354005008, 60750354005011, 60750331002006, 60750604001006, 60750604001007, 60750604001008, 60750604001009, 60750604001010, 60750604001015, 60750604001016, 60750604001039, 60750329012008, 60750329012009]
        # Feasiblity Constraint: every area has to belong to one zone
        self.m.addConstrs(
            (gp.quicksum(self.x[i, z] for z in self.valid_zone_per_area[i]) == 1
            for i in range(self.A)
             # ),
             # if self.idx2area[i] in west_blocks),
             if self.idx2area[i] in left_half_blocks3),
            "FeasibilityConstraint"
        )


        # self.x = self.m.addVars(self.A, self.M, lb=0.0, ub= 1.0, vtype=GRB.CONTINUOUS, name="x")
        # self.x[1, 0].vtype = GRB.INTEGER

        # self.constraints['M'] = self.M
        # # for z in range(self.M):
        # for j in range(self.A):
        #     if j %10 == 0:
        #         z = random.randint(0, self.M - 1)
        #         self.x[j,z].vtype = GRB.BINARY


        # self.set_y_distance()
        # self.distance_coef = 1

        # self.set_y_balance()
        # self.balance_coef = 0

        # self.set_y_shortage()
        # self.shortage_coef = 2

        self.set_y_boundary()
        self.boundary_coef = 10
        self.m.setObjective(self.boundary_coef * self.y_boundary, GRB.MINIMIZE)
        # self.m.setObjective(1 , GRB.MINIMIZE)
        # self.m.setObjective(self.distance_coef * self.y_distance + self.shortage_coef * self.y_shortage +
        #                     self.boundary_coef * self.y_boundary , GRB.MINIMIZE)
        # self.m.setObjective( self.distance_coef * self.y_distance +  self.shortage_coef * self.y_shortage +
        #                      self.balance_coef * self.y_balance + self.boundary_coef * self.y_boundary , GRB.MINIMIZE)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def _all_cap_proportional_shortage_constraint(self, all_cap_shortage):
        # No zone has shortage more than all_cap_shortage percentage of its total student population
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [(self.area_data["all_prog_students"][j] - self.area_data["all_prog_capacity"][j]) * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                all_cap_shortage *
                gp.quicksum(
                    [self.area_data["all_prog_students"][j] * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
            )
        self.constraints["All_Cap_Propotional_Shortage"] = all_cap_shortage

    def _proportional_shortage_constraint(self, shortage):
        # No zone has shortage more than shortage percentage of its population
        for z in range(self.M):
            self.m.addConstr(
                (1 - shortage) *
                gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                gp.quicksum(
                    [self.seats[j] * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
            )
        self.constraints["Propotional_Shortage"] = shortage


    def _proportional_overage_constraint(self, overage):
        # No zone has shortage more than shortage percentage of its population
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [(-self.studentsInArea[j] + self.seats[j]) * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                overage *
                gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
            )
        self.constraints["Propotional_Overage"] = overage

    def fixed_shortage_const(self, shortage):
        # each zone has at least the shortage
        for z in range(self.M):
            self.m.addConstr(
                gp.quicksum(
                    [(self.studentsInArea[j] - self.seats[j]) * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <= shortage)
        self.constraints["Fixed_Shortage"] = shortage

    def _shortage_and_balance_constraints(self, shortage_=True, balance_=True, shortage=0.15, overage=0.2, all_cap_shortage=0.8, balance=1000):
        if shortage_:
            # self.fixed_shortage_const()
            if shortage != -1:
                self._proportional_shortage_constraint(shortage)
            if overage != -1:
                self._proportional_overage_constraint(overage)
            if all_cap_shortage != -1:
                self._all_cap_proportional_shortage_constraint()

        if balance_:
            # add number of students balance constraint
            for z in range(self.M):
                firstZone = gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                for q in range(z + 1, self.M):
                    secondZone = gp.quicksum(
                        [self.studentsInArea[j] * self.x[j, q] for j in self.valid_area_per_zone[z]]
                    )
                    self.m.addConstr(firstZone - secondZone <= balance)
                    self.m.addConstr(firstZone - secondZone >= -balance)
            self.constraints["Balance"] = balance


    def _add_geo_constraints(
            self,
            cover_distance=-1,
            contiguity=False,
    ):
        if cover_distance > 0:
            self._add_cover_distance(cover_distance)
        else:
            self.constraints["cover"] = -1
        if contiguity:
            self._add_contiguity_constraint()
        else:
            self.constraints["contiguity"] = 0

    def _add_cover_distance(self, coverDistance):
        """  """
        for z in range(self.M):
            c = str(self.idx2area[self.centroids[z]])
            zone_sum = gp.quicksum(
                [
                    self.euc_distances.loc[self.idx2area[j], c]
                    * self.studentsInArea[j]
                    * self.x[j, z]
                    for j in self.valid_area_per_zone[z]
                ]
            )
            students_sum = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
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
                if j not in self.valid_area_per_zone[z]:
                    continue
                if len(self.closer_euc_neighbors[j, self.centroids[z]]) >= 1:  # TODO
                    neighbor_sum = gp.quicksum(
                        self.x[k, z]
                        for k in self.closer_euc_neighbors[j, self.centroids[z]]
                        if k in self.valid_area_per_zone[z]
                    )
                    self.m.addConstr(self.x[j, z] <= neighbor_sum, name="Contiguity")
                else:
                    any_neighbor_sum = gp.quicksum(
                        [
                            self.x[k, z]
                            for k in self.neighbors[j] if k in self.valid_area_per_zone[z]
                        ]
                    )
                    self.m.addConstr(self.x[j, z] <= any_neighbor_sum, name="Contiguity")
                    # z_area = self.idx2area[self.centroids[z]]
                    # j_area = self.idx2area[j]
                    # if self.euc_distances.loc[z_area, str(j_area)] > 1.5:
                    #     self.m.addConstr(self.x[j, z] == 0, name="Contiguity")

        self.constraints["contiguity"] = 1

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def _add_diversity_constraints(self, racial_dev=1, frl_dev=1, aalpi_dev=1):
        # racial balance constraint
        if racial_dev < 1:
            self._add_racial_constraint(racial_dev)

        # frl constraint
        if frl_dev < 1:
                self._add_frl_constraint(frl_dev)

        # aalpi constraint
        if aalpi_dev < 1:
            self._add_aalpi_constraint(aalpi_dev)


    def _add_population_balance_constraint(self, population_dev=1):
        average_population = sum(self.area_data["all_prog_students"])/self.M
        for z in range(self.M):
            zone_sum = gp.quicksum(
                [self.area_data["all_prog_students"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])

            self.m.addConstr(zone_sum >= (1 - population_dev) * average_population, name= "Population LB")
            self.m.addConstr(zone_sum <= (1 + population_dev) * average_population, name= "Population UB")




    def _add_racial_constraint(self, race_dev=1):
        for race in ETHNICITY_COLS:
                # ['resolved_ethnicity_White',
                #      'resolved_ethnicity_Hispanic/Latinx',
                #      'resolved_ethnicity_Asian']:
            if self.use_loaded_data:
                race_ratio = sum(self.area_data[race]) / float(self.N)

                for z in range(self.M):
                    zone_sum = gp.quicksum(
                        [self.area_data[race][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                    )
                    district_students = gp.quicksum(
                        [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                    )
                    self.m.addConstr(zone_sum >= (race_ratio - race_dev) * district_students, name= str(race) + " LB")
                    self.m.addConstr(zone_sum <= (race_ratio + race_dev) * district_students, name= str(race) + " UB")

            else:
                race_ratio = sum(self.area_data[race]) / sum(self.area_data["num_with_ethnicity"])
                for z in range(self.M):
                    zone_sum = gp.quicksum(
                        [self.area_data[race][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])
                    zone_total = gp.quicksum(
                        [self.area_data["num_with_ethnicity"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])

                    self.m.addConstr(zone_sum >= (race_ratio - race_dev) * zone_total, name= str(race) + " LB")
                    self.m.addConstr(zone_sum <= (race_ratio + race_dev) * zone_total, name= str(race) + " UB")




    def _add_frl_constraint(self, frl_dev=1):
        for z in range(self.M):
            if self.use_loaded_data:
                zone_sum = gp.quicksum(
                    [self.area_data["FRL"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                district_students = gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                self.m.addConstr(zone_sum >= (self.F - frl_dev) * district_students, name="FRL LB")
                self.m.addConstr(zone_sum <= (self.F + frl_dev) * district_students, name="FRL UB")

            else:
                zone_sum = gp.quicksum([self.area_data["frl_count"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])
                zone_total = gp.quicksum([self.area_data["frl_total_count"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])
                self.m.addConstr(zone_sum >= (zone_total * (self.FRL_ratio - frl_dev)), name="FRL LB")
                self.m.addConstr(zone_sum <= (zone_total * (self.FRL_ratio + frl_dev)), name="FRL UB")

        self.constraints["frl_dev"] = frl_dev


    def _add_aalpi_constraint(self, aalpi_dev):
        district_average = sum(self.area_data["AALPI Score"]) / self.N
        for z in range(self.M):
            zone_sum = gp.quicksum(
                [self.area_data["AALPI Score"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            district_students = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            self.m.addConstr(zone_sum >= (district_average - aalpi_dev) * district_students, name="AALPI LB")
            self.m.addConstr(zone_sum <= (district_average  + aalpi_dev) * district_students, name="AALPI UB")

        self.constraints["aalpi_dev"] = aalpi_dev

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
            zone_school_count[z] = gp.quicksum([self.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]])
            self.m.addConstr(zone_school_count[z] <= avg_school_count + 1)
            self.m.addConstr(zone_school_count[z] >= avg_school_count - 1)

        # if K8 schools are included,
        # make sure no zone has more than one K8 schools
        if self.include_k8:
            zone_k8_count = {}
            for z in range(self.M):
                if self.use_loaded_data:
                    zone_k8_count[z] = gp.quicksum([self.area_data["K-8"][j] * self.x[j, z]
                                                    for j in self.valid_area_per_zone[z]])
                else:
                    zone_k8_count[z] = gp.quicksum([
                        self.school_df["K-8"][j] *
                        self.x[self.area2idx[self.school_df[self.level][j]], z]
                                                    for j in range(len(self.school_df.index))])

                self.m.addConstr(zone_k8_count[z] <= 1)


    def _add_school_quality_constraint(self, min_pct, max_pct=None):
        scores = self.area_data["eng_scores_1819"].fillna(value=0)
        schools = self.area_data["num_schools"].fillna(value=0)
        for z in range(self.M):
            zone_sum = gp.quicksum(
                [scores[j] * schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            district_average = (
                    sum(scores * schools)
                    / sum(schools)
                    * gp.quicksum([self.x[j, z] * schools[j] for j in self.valid_area_per_zone[z]])
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
                [scores[j] * schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            district_average = (
                    sum(scores * schools)
                    / sum(schools)
                    * gp.quicksum([self.x[j, z] * schools[j] for j in self.valid_area_per_zone[z]])
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
                    [scores[j] * self.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                district_average = (
                        sum(scores * self.schools) / sum(self.schools)
                        * gp.quicksum([self.x[j, z] * self.schools[j] for j in self.valid_area_per_zone[z]])
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
                    [self.x[j, z] * top_schools[j] for j in self.valid_area_per_zone[z]]
                )
                self.m.addConstr(topz >= 0.8)
                self.constraints["AvgColorIndex"] = topX


    def _boundary_threshold_constraint(self, boundary_threshold):
        # make sure areas that are closer than boundary_threshold distance
        # to a school, are matched to the same zone as that school.
        for z in range(self.M):
            for i in range(self.A):
                if i not in self.valid_area_per_zone[z]:
                    continue
                area = self.idx2area[i]

                for idx, row in self.school_df.iterrows():
                    if (row["K-8"] == 1) & (self.include_k8 == False):
                        continue

                    if self.level == "BlockGroup":
                        sch_area = self.sch2bg[row["school_id"]]
                    elif self.level == "Block":
                        sch_area = self.sch2b[row["school_id"]]
                    else:
                        raise ValueError("It is not recommended to have boundary threshold constraint for Attendance Area Zones")
                    sch_idx = self.area2idx[sch_area]
                    if sch_idx not in self.valid_area_per_zone[z]:
                        continue

                    if self.euc_distances.loc[sch_area, str(area)] < boundary_threshold:
                            # self.m.addConstr(self.x[i, z] == self.x[sch_idx, z])
                            self.m.addConstr(self.x[i, z] == 0)
                            self.m.addConstr(self.x[sch_idx, z] == 0)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def save(self, path,  rand=random.randint(0, 400000), save_opt_params = False, name = "", solve_success = 1):
        filename = os.path.expanduser(path)
        filename += name
        filename += ".csv"

        # save zones themselves
        with open(filename, "w") as outFile:
            writer = csv.writer(outFile, lineterminator="\n")
            if solve_success == 1:
                for z in self.zone_lists:
                    writer.writerow(z)
            else:
                writer.writerow({})


        # save optimization parameters
        if save_opt_params == True:
            with open(filename[:-4] + "_params.txt", "w") as f:
                for k, v in self.constraints.items():
                    f.write("".join(k.split()) + " " + str(v) + "\n")
                if type(self.centroid_sch_option) == str or self.centroid_sch_option >= 0:
                    f.write("centroids " + str(self.centroid_sch_option))

    def solve(self, write=False, save_path="~/SFUSD/"):
        self.filename = ""
        self.zone_dict = {}

        try:
            self.m.Params.TimeLimit = 10000
            self.m.optimize()

            zone_lists = []
            for z in range(0, self.M):
                zone = []
                for j in range(0, self.A):
                    if j not in self.valid_area_per_zone[z]:
                        continue
                    if self.x[j, z].X >= 0.999:
                        self.zone_dict[self.idx2area[j]] = z
                        zone.append(self.area_data[self.level][j])
                        # add City wide school SF Montessori, even if we are not including city wide schools
                        # 823 is the aa level of SF Montessori school (which has school id 814)
                        if self.idx2area[j] in [823, 60750132001]:
                            self.zone_dict[self.idx2area[j]] = z
                            if self.level == "attendance_area":
                                zone.append(SF_Montessori)
                if not zone == False:
                    zone_lists.append(zone)
            zone_dict = {}
            for idx, schools in enumerate(zone_lists):
                zone_dict = {
                    **zone_dict,
                    **{int(float(s)): idx for s in schools if s != ""},
                }
            # add K-8 schools to dict if using them
            if (self.level == 'attendance_area') & (self.include_k8):
                cw = self.school_df.loc[self.school_df["K-8"] == 1]
                for i, row in cw.iterrows():
                    k8_schno = row["school_id"]
                    z = zone_dict[self.sch2bg[int(float(k8_schno))]]
                    zone_dict = {**zone_dict, **{int(float(k8_schno)): z}}
                    zone_lists[z].append(k8_schno)
            self.zone_dict = zone_dict
            self.zone_lists = zone_lists

            if write:
                self.save(save_path)

            return 1

        except gp.GurobiError as e:
            print("gurobi error #" + str(e.errno) + ": " + str(e))
            return -1
        except AttributeError:
            print("attribute error")
            return -1



def zone_assignment_process(config):
    name = Compute_Name(config)
    print("name: ", name)
    # if os.path.exists(config["path"] + name + "_AA" +".csv"):
    #     return

    # input_level = 'attendance_area'
    # input_level = 'BlockGroup'
    input_level = 'Block'


    dz = DesignZones(
        config = config,
        level=input_level,
    )
    dz._set_objective_model(max_distance=config["max_distance"])
    dz._shortage_and_balance_constraints(shortage_=True, balance_= False,
                     shortage=config["shortage"], overage= config["overage"], all_cap_shortage=config["all_cap_shortage"])

    dz._add_geo_constraints(contiguity=True)
    # dz._add_diversity_constraints(racial_dev=config["racial_dev"], frl_dev=config["frl_dev"])
    # dz._add_school_count_constraint()
    # if param.include_sch_qlty_balance == True:
    #     dz._add_met_quality_constraint(min_pct = param.lbscqlty)
    # dz._boundary_threshold_constraint(boundary_threshold = param.boundary_threshold)

    solve_success = dz.solve()

    if solve_success == 1:
        print("Resulting zone dictionary: ", dz.zone_dict)

        dz.save(path=config["path"], name = name + "_AA")

        zv = ZoneVisualizer(input_level)
        zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location, save_name="GE" + name + "_AA")

        # stats_evaluation(dz, dz.zd)
    elif solve_success == -1:
        dz.save(path=config["path"], name = name + "_AA", solve_success = solve_success)

    del dz
    gc.collect()


if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    zone_assignment_process(config)



    # for frl_dev in [0.15, 0.1]:
    #     param.frl_dev = frl_dev
    #     for racial_dev in [0.15, 0.12]:
    #         param.racial_dev = racial_dev
    #         for include_k8 in [True, False]:
    #             param.include_k8 = include_k8
    #             with open("../Config/centroids.yaml", "r") as f:
    #                 centroid_options = yaml.safe_load(f)
    #                 for centroids in centroid_options:
    #                     param.zone_count = int(centroids.split("-")[0])
    #                     param.centroids_type = centroids
    #                     print("param: " + str(param.frl_dev) + " " + str(param.racial_dev)
    #                           + " " + str(param.include_k8))
    #                     zone_assignment_process(param)
    print("done")
# Note: Total number of students in aa level is not the same as blockgroup level.
# Reason: some students, do not have their bg info available
# (but they do have their aa info, and also they pass every other filter, i.e. enrollment)

