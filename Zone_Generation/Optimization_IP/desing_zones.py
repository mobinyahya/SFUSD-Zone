import sys
import ast
import yaml
import pickle
import random, math, gc, os, csv
from typing import Union
import gurobipy as gp
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
from Zone_Generation.Optimization_IP.schools import Schools
from Zone_Generation.Optimization_IP.students import Students
from Zone_Generation.Optimization_IP.integer_program import Integer_Program



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
    ):
        self.config = config
        # self.M: number of zones requested (The number of zones that we need to divide the city into)
        self.M = int(config["centroids_type"].split("-")[0])  # number of possible zones
        # The building blocks of zones. As a defualt, this is attendance_area
        self.level = config["level"]  # 'Block', 'BlockGroup' or 'attendance_area'

        self.centroid_type = config["centroids_type"]
        self.include_k8 = config["include_k8"]
        self.population_type = config["population_type"]

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

        self.sch2area = dict(zip(self.school_df["school_id"], self.school_df[self.level]))

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






    def load_students_and_schools(self):
        students_data = Students(self.config)
        schools_data = Schools(self.config)
        self.student_df = students_data.load_student_data()
        self.school_df = schools_data.load_school_data()

        student_stats = self._aggregate_student_data_to_area(self.student_df, students_data.years)
        school_stats = self._aggregate_school_data_to_area(self.school_df)

        self.area_data = student_stats.merge(school_stats, how='outer', on=self.level)

        self._load_auxilariy_areas()

        self.area_data.fillna(value=0, inplace=True)
        if self.level == "BlockGroup":
            self.bg2att = load_bg2att(self.level)





    # groupby the student data of year i, such that we have the information only on area level
    def _aggregate_student_data_to_area(self, student_df, years):
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
                student_stats[col] /= len(years)
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







    def initialize_centroids(self):
        """set the centroids - each one is a block or attendance area depends on the method
        probably best to make it a school"""

        with open("../Config/centroids.yaml", "r") as f:
            centroid_configs = yaml.safe_load(f)
        if self.centroid_type not in centroid_configs:
            raise ValueError(
                "The centroids type specified is not defined in centroids.yaml.")

        self.centroid_sch = centroid_configs[self.centroid_type]
        print("Number of centroid schools ", len(self.centroid_sch))

        self.school_df['is_centroid'] = self.school_df['school_id'].apply(lambda x: 1 if x in self.centroid_sch else 0)

        if self.include_k8:
            self.centroid_location = self.school_df[self.school_df['is_centroid'] == 1][['lon', 'lat', 'school_id']]
        else:
            self.centroid_location = self.school_df[(self.school_df['is_centroid'] == 1) & (self.school_df['K-8'] != 1)][['lon', 'lat', 'school_id']]
            self.schools_locations = self.school_df[['lon', 'lat', 'school_id']]


        centroid_areas = [self.sch2area[x] for x in self.centroid_sch]
        self.centroids = [self.area2idx[j] for j in centroid_areas]


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

        if (self.level == "Block") and (self.centroid_type == '59-zone-1'):
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

        if (self.level == "Block") and (self.centroid_type == '59-zone-1'):
            with open(save_path, 'wb') as file:
                pickle.dump(self.closer_euc_neighbors, file)





if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    name = Compute_Name(config)
    print("name: ", name)
    # if os.path.exists(config["path"] + name + "_AA" +".csv"):
    #     return


    dz = DesignZones(config=config)
    dz.samezone_pairs = []
    IP = Integer_Program(dz)
    IP._set_objective_model(max_distance=config["max_distance"])
    IP._shortage_and_balance_constraints(shortage_=True, balance_= False,
                     shortage=config["shortage"], overage= config["overage"], all_cap_shortage=config["all_cap_shortage"])

    IP._add_contiguity_constraint()
    IP._add_diversity_constraints(racial_dev=config["racial_dev"], frl_dev=config["frl_dev"])
    IP._add_school_count_constraint()

    solve_success = IP.solve()

    if solve_success == 1:
        print("Resulting zone dictionary: ", IP.zone_dict)
        IP.save(path=config["path"], name = name + "_AA")

        zv = ZoneVisualizer(config["level"])
        zv.zones_from_dict(IP.zone_dict, centroid_location=dz.centroid_location, save_path=config["path"]+name+"_"+SUFFIX[config["level"]])
        # stats_evaluation(dz, dz.zd)



# Note: Total number of students in aa level is not the same as blockgroup level.
# Reason: some students, do not have their bg info available
# (but they do have their aa info, and also they pass every other filter, i.e. enrollment)

