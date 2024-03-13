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
from Zone_Generation.Optimization_IP.load_optimization_data_old import *



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
            level="attendance_area",
            drop_optout=True,
            capacity_scenario="A",
            new_schools=True,
            use_loaded_data=True
    ):
        self.config = config
        # self.M: number of zones requested (The number of zones that we need to divide the city into)
        self.M = int(config["centroids_type"].split("-")[0])  # number of possible zones
        # The building blocks of zones. As a defualt, this is attendance_area
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

        # self.seats: A dictionary, for every area index:
        # (keys: area index j), (values: number of seats for GE students in area index j)
        self.seats = (self.area_data["ge_capacity"].astype("int64").to_numpy())

        # self.schools: A dictionary, for every area:
        # (keys: area index j), (values: number of schools in area index j) this value is usually 0 or 1
        self.schools = self.area_data['num_schools']

        # self.N: Total number of GE students
        self.N = sum(self.area_data["ge_students"])

        # self.F: Average percentage of FRL students
        # (students that are eligible for Free or reduced price lunch)
        self.F = sum(self.area_data["FRL"]) / (self.N)

        # self.A: Total number of areas (number of distinct area indices)
        self.A = len(self.area_data.index)

        # self.studentsInArea: A dictionary, for every area index:
        # (keys: area index j), (values: number of GE students in area index j)
        self.studentsInArea = self.area_data["ge_students"]
        self.constraints = {"include_k8": self.include_k8}

        print("Average FRL ratio:       ", self.F)
        print("Number of Areas:       ", self.A)
        print("Number of GE students:       ", self.N)
        print("Number of total students: ", sum(self.area_data["all_prog_students"]))
        print("Number of total seats:    ", sum(self.area_data["all_prog_capacity"]))
        print("Number of GE seats:       ", sum(self.seats))
        print("Number of zones:       ", self.M)

        self.area_data[self.level] = self.area_data[self.level].astype("int64")

        # self.area2idx: A dictionary, mapping each census area code, to its index in our data
        # (keys: census area code AA), (values: index of area AA, in our data set)
        # Note that we can access our dictionaries only using the area index, and not the area code
        self.area2idx = dict(zip(self.area_data[self.level], self.area_data.index))

        # self.area2idx: A dictionary, mapping each area index in our data, to its census area code
        # (keys: area index j), (values: census area code fo the area with index j)
        self.idx2area = dict(zip(self.area_data.index, self.area_data[self.level]))

        self.euc_distances = load_euc_distance_data(self.level)
        # self.save_partial_distances()
        # self.drive_distances = self.load_driving_distance_data()


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
            self.area_data = load_data_old()


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
        print("Number of centroid schools ", len(self.centroid_sch))

        self.school_df['is_centroid'] = self.school_df['school_id'].apply(lambda x: 1 if x in self.centroid_sch else 0)

        if self.include_k8:
            self.centroid_location = self.school_df[self.school_df['is_centroid'] == 1][['lon', 'lat', 'school_id']]
        else:
            self.centroid_location = self.school_df[(self.school_df['is_centroid'] == 1) & (self.school_df['K-8'] != 1)][['lon', 'lat', 'school_id']]
            self.schools_locations = self.school_df[['lon', 'lat', 'school_id']]


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


    # This function constructs the boundary cost variables.
    # Boundary cost variables are used in the optimization model objective
    def set_y_boundary(self):
        neighboring_tuples = []
        for i in range(self.A):
            for j in self.neighbors[i]:
                if i >= j:
                    continue
                if ((i,j) in self.samezone_pairs) or ((j,i) in self.samezone_pairs):
                    continue
                neighboring_tuples.append((i,j))

        # self.b[i, j]: a binary boundary variable. This variable will be 1,
        # if area with index i, and area with index j, are adjacent areas, that
        # are assigned to different zones (hence, they will be part of boundary cost)
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



    def _set_objective_model(self, loaded_zd, max_distance=-1):
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
                    if self.idx2area[i] in loaded_zd:
                        continue
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

        # Feasiblity Constraint: every area has to belong to one zone
        self.m.addConstrs(
            (gp.quicksum(self.x[i, z] for z in self.valid_zone_per_area[i]) == 1
            for i in range(self.A)
             # ),
             if self.idx2area[i] not in loaded_zd),
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

        # set the objective of the Integer Program.
        # The integer program will try to minimize the cost of boundary,
        # which will result into compact and nice looking shapes for zones.
        self.m.setObjective(self.boundary_coef * self.y_boundary, GRB.MINIMIZE)
        # self.m.setObjective(1 , GRB.MINIMIZE)
        # self.m.setObjective(self.distance_coef * self.y_distance + self.shortage_coef * self.y_shortage +
        #                     self.boundary_coef * self.y_boundary , GRB.MINIMIZE)
        # self.m.setObjective( self.distance_coef * self.y_distance +  self.shortage_coef * self.y_shortage +
        #                      self.balance_coef * self.y_balance + self.boundary_coef * self.y_boundary , GRB.MINIMIZE)



    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # All programs proportional shortage for each zone =
    # percentage of all-program-students in the zone, that don't get any seat from all-program-capacities.
    # all-program-students =
    # (Total number of students, across all program types, in the zones)
    # all-program-capacities =
    # (Total number of seats for all programs (not just GE) in schools within the zone)
    # The following constraint makes sure no zone has an All programs proportional shortage
    # larger than the given input, all_cap_shortage
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


    # proportional shortage for each zone =
    # percentage of students (GE students) in the zone, that don't get any seat (from GE capacities)
    # students in the zone
    # The following constraint makes sure no zone has a shortage
    # larger than the given input "shortage"
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

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_constraint(self, overage):
        # No zone has overage more than overage percentage of its population
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



    # Designing contiguous school zones is desirable for practical reasons,
    # i.e. school commutes and policy communication.
    # Make sure areas assigned to each zone form a contiguous zone as follows:
    # assign unit 𝑗 to zone with centroid area 𝑧, only if
    # there is a ‘path’ of closer neighboring areas also assigned
    # to the same zone that connects area 𝑗 to the centroid area 𝑧.
    def _add_contiguity_constraint(self, loaded_szd):
        # initialization - every centroid belongs to its own zone
        for z in range(self.M):
            self.m.addConstr(
                self.x[self.centroids[z], z] == 1, name="Centroids to Zones"
            )

        # (x[j,z] (and indicator that unit j is assigned to zone z)) \leq
        # (sum of all x[j',z] where j' is in self.closer_neighbors_per_centroid[area,c] where c is centroid for z)
        for j in range(self.A):
            for z in range(self.M):
                if j == self.centroids[z]:
                    continue
                if j not in self.valid_area_per_zone[z]:
                    continue
                X = self.closer_euc_neighbors[j, self.centroids[z]]
                Y = [neighbor for neighbor in X if self.idx2area[neighbor] not in loaded_szd]
                # only impose the contiguity as we said, if the area j has a neighbor that is closer to centroid z.
                # otherwise, just make sure j has at least another neighbor assigned tot the same zone z, so that
                # j is not an island assigned to z.
                if len(Y) >= 1:  # TODO
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

        self.constraints["contiguity"] = 1

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Add constraints related to diversity such as: racial balance,
    # frl balance (balance in free or reduced priced lunch eligibility)
    # and aalpi balance, across all the zones.
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

    # Enforce zones to have almost the same number of students
    # Make sure the average population of each zone, is within a given
    # population_dev% of average population for zones
    def _add_population_balance_constraint(self, population_dev=1):
        average_population = sum(self.area_data["all_prog_students"])/self.M
        for z in range(self.M):
            zone_sum = gp.quicksum(
                [self.area_data["all_prog_students"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])

            self.m.addConstr(zone_sum >= (1 - population_dev) * average_population, name= "Population LB")
            self.m.addConstr(zone_sum <= (1 + population_dev) * average_population, name= "Population UB")


    # Make sure students of racial groups are fairly distributed among zones.
    # For specific racial minority, make sure the percentage of students in each zone, is within an additive
    #  race_dev% of percentage of total students of that race.
    def _add_racial_constraint(self, race_dev=1):
        for race in ETHNICITY_COLS:
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


    # Make sure students of low socioeconomic status groups are fairly distributed among zones.
    # Our only metric to measure socioeconomic status, is FRL, which is the students eligibility for
    # Free or Reduced Price Lunch.
    # make sure the total FRL for students in each zone, is within an additive
    #  frl_dev% of average FRL over zones..
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
    # This following constraint makes sure all zones have almost similar number of schools.
    # First compute the average number of schools per zone,
    # by computing the total number of schools in the city and dividing it by the number of zones.
    # Next, add a constraint to make sure the number of schools in each zone
    # is within average number of schools per zone + or - 1
    def _add_school_count_constraint(self, loaded_zd):
        zone_school_count = {}
        #TODO change
        avg_school_count = sum([self.schools[j] for j in range(self.A)
                                if self.idx2area[j] not in loaded_zd]
                               ) / self.M + 0.0001
        # note: although we enforce max deviation of 1 from avg, in practice,
        # no two zones will have more than 1 difference in school count
        # Reason: school count is int. Observe the avg_school_count +-1,
        # if avg_school_count is not int, and see how the inequalities will look like
        # * I implemented the code this way (instead of pairwise comparison), since it is faster
        for z in range(self.M):
            zone_school_count[z] = gp.quicksum([self.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]])
            # TODO
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

    # We want to make sure families have the option to go to a school,
    # if they are only a block away from that school.
    # make sure areas that are closer than boundary_threshold distance
    # to a school, are matched to the same zone as that school.
    def _boundary_threshold_constraint(self, boundary_threshold):
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
        config=config,
        level=input_level,
    )
    dz._set_objective_model(max_distance=config["max_distance"])
    dz._shortage_and_balance_constraints(shortage_=True, balance_= False,
                     shortage=config["shortage"], overage= config["overage"], all_cap_shortage=config["all_cap_shortage"])

    dz._add_contiguity_constraint()
    dz._add_diversity_constraints(racial_dev=config["racial_dev"], frl_dev=config["frl_dev"])
    dz._add_school_count_constraint()

    solve_success = dz.solve()

    if solve_success == 1:
        print("Resulting zone dictionary: ", dz.zone_dict)
        dz.save(path=config["path"], name = name + "_AA")

        zv = ZoneVisualizer(input_level)
        zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location, save_name="GE" + name + "_AA")
        # stats_evaluation(dz, dz.zd)

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

