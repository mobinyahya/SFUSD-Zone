import csv
import json
import os
import pickle
from collections import defaultdict

import pandas as pd
import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import load_euc_distance_data, load_bg2att, load_b2bg, load_census_shapefile
from Zone_Generation.Config.Constants import AREA_ETHNICITIES, BUILDING_BLOCKS, AUX_BG, AREA_COLS, get_dropbox_path
from Zone_Generation.Optimization.schools import Schools
from Zone_Generation.Optimization.students import Students


class DesignZones:
    def __init__(
            self,
            config,
    ):
        self.config = config
        # self.Z: number of zones requested (The number of zones that we need to divide the city into)
        self.Z = int(config["centroids_type"].split("-")[0])  # number of possible zones
        # The building blocks of zones. As a defualt, this is attendance_area
        self.level = config["level"]  # 'Block', 'BlockGroup' or 'attendance_area'
        self.is_local = config["is_local"]

        self.centroid_type = config["centroids_type"]
        self.include_k8 = config["include_k8"]
        self.population_type = config["population_type"]
        self.capacity_scenario = config["capacity_scenario"]

        self.load_students_and_schools()
        self.construct_datastructures()

        self.load_neighborhood_dict()
        self.initialize_centroids()
        self.initialize_centroid_neighbors()

    def construct_datastructures(self):

        self.A = len(self.area_data.index)
        self.schools = self.area_data['num_schools']
        self.area_data[self.level] = self.area_data[self.level].astype("int64")

        self.area2idx = dict(zip(self.area_data[self.level], self.area_data.index))
        self.idx2area = dict(zip(self.area_data.index, self.area_data[self.level]))
        self.sch2area = dict(zip(self.school_df["school_id"], self.school_df[self.level]))

        self.euc_distances = load_euc_distance_data(self.level, self.area2idx, self.is_local)

        if self.capacity_scenario != "Closure":
            self.seats = (self.area_data["ge_capacity"].astype("int64").to_numpy())
            self.studentsInArea = self.area_data["ge_students"]
            self.N = sum(self.area_data["ge_students"])

        else:
            imbalance_ratio = 3700 / sum(self.area_data["all_prog_students"])
            self.area_data["all_prog_students"] = self.area_data["all_prog_students"] * imbalance_ratio

            self.seats = (self.area_data["all_prog_capacity"].astype("int64").to_numpy())
            self.studentsInArea = self.area_data["all_prog_students"]
            self.N = sum(self.area_data["all_prog_students"])
            self.area_data["FRL"] = 3700 / 2460 * self.area_data["FRL"]
            for ethnicity in AREA_ETHNICITIES:
                self.area_data[ethnicity] = 3700 / 2460 * self.area_data[ethnicity]
            #     print("ethnicity ", ethnicity, "percentage is: ", sum(self.area_data[ethnicity]))

        self.F = sum(self.area_data["FRL"]) / (self.N)
        self.R = {}
        for ethnicity in AREA_ETHNICITIES:
            self.R[ethnicity] = sum(self.area_data[ethnicity]) / (self.N)

        # print("Average FRL ratio:       ", self.F)
        print("Number of Areas:       ", self.A)
        # max,min number of students in an area
        print("Max number of students in an area: ", max(self.studentsInArea))
        print("Min number of students in an area: ", min(self.studentsInArea))
        print('Avg number of students in an area: ', sum(self.studentsInArea) / self.A)
        print('Median number of students in an area: ', self.area_data['ge_students'].median())

        print("Number of GE students:       ", sum(self.area_data["ge_students"]))
        # print("Number of GE seats:       ", sum(self.area_data["ge_capacity"]))
        print("Number of total students: ", sum(self.area_data["all_prog_students"]))
        # print("Number of total seats:    ", sum(self.area_data["all_prog_capacity"]))
        # print("Number of Schools:       ", sum(self.schools))
        # print("Number of zones:       ", self.Z)

        # self.save_partial_distances()
        # self.drive_distances = self.load_driving_distance_data()

    def save_partial_distances(self):
        self.euc_distances = load_euc_distance_data(self.level, self.area2idx, self.is_local, complete_bg=True)

        # print("len(self.euc_distances))  ", len(self.euc_distances))
        school_blocks = list(self.sch2b.values())

        existing_school_blocks = [block for block in school_blocks if block in self.euc_distances.index]

        # pd.set_option('display.max_rows', None)
        # print("self.euc_distances.index", list(self.euc_distances.index))
        # print("school_blocks ", school_blocks)
        # print("len(existing_school_blocks): ", len(existing_school_blocks))
        # print("len((school_blocks)): ", len((school_blocks)))
        distances = self.euc_distances.loc[existing_school_blocks]

        save_path = f"{get_dropbox_path(self.is_local)}/Optimization/distances_b2b_schools.csv"
        distances.to_csv(save_path)

    def load_students_and_schools(self):
        students_data = Students(self.config)
        schools_data = Schools(self.config)
        self.student_df = students_data.load_student_data()

        self.school_df = schools_data.load_school_data()

        student_stats = self._aggregate_student_data_to_area(self.student_df)
        school_stats = self._aggregate_school_data_to_area(self.school_df)

        student_stats[self.level] = student_stats[self.level].astype(int)
        self.area_data = student_stats.merge(school_stats, how='outer', on=self.level)

        self._load_auxilariy_areas()

        self.area_data.fillna(value=0, inplace=True)
        if self.level == "BlockGroup":
            self.bg2att = load_bg2att(self.is_local)
        elif self.level == "Block":
            self.b2bg = load_b2bg(self.is_local)

    # groupby the student data by area level
    def _aggregate_student_data_to_area(self, student_df):
        # sum_columns = list(student_df.columns)
        # sum_columns.remove("FRL")
        # mean_columns = [self.level, "FRL"]
        #
        # sum_students = student_df[sum_columns].groupby(self.level, as_index=False).sum()
        # mean_students = student_df[mean_columns].groupby(self.level, as_index=False).mean()
        #
        # student_stats = mean_students.merge(sum_students, how="left", on=self.level)
        student_stats = student_df.groupby(self.level, as_index=False).sum()
        student_stats = student_stats[AREA_COLS + [self.level] + AREA_ETHNICITIES]

        for col in student_stats.columns:
            if col not in BUILDING_BLOCKS:
                student_stats[col] /= len(self.config["years"])
        return student_stats

    def _aggregate_school_data_to_area(self, school_df):

        sum_columns = [self.level, "all_prog_capacity", "ge_capacity", "num_schools", "english_score",
                       "math_score", "greatschools_rating", "AvgColorIndex"]
        mean_columns = [self.level, "MetStandards", ]

        sum_schools = school_df[sum_columns].groupby(self.level, as_index=False).sum()
        mean_schools = school_df[mean_columns].groupby(self.level, as_index=False).mean()

        return mean_schools.merge(sum_schools, how="left", on=self.level)

    def _load_auxilariy_areas(self):
        # we add areas (blockgroups/blocks) that were missed from guardrail, since there was no student or school in them.
        if (self.level == 'BlockGroup') | (self.level == 'Block'):
            valid_areas = set(
                pd.read_csv(f'{get_dropbox_path(self.is_local)}/Optimization/block_blockgroup_tract.csv')[self.level])
            census_areas = load_census_shapefile(self.level, self.is_local)[self.level]
            census_areas = set(census_areas)
            census_areas = census_areas - set(AUX_BG)

            common_areas = census_areas.intersection(valid_areas)

            current_areas = set(self.area_data[self.level])

            auxiliary_areas = common_areas - current_areas

            auxiliary_areas_df = pd.DataFrame({self.level: list(auxiliary_areas)})

            self.area_data[self.level] = self.area_data[self.level].astype(int)
            auxiliary_areas_df[self.level] = auxiliary_areas_df[self.level].astype(int)
            self.area_data = pd.merge(self.area_data, auxiliary_areas_df, how='outer', on=self.level)
            self.area_data.fillna(value=0, inplace=True)

    def initialize_centroids(self):
        """set the centroids - each one is a block or attendance area depends on the method
        probably best to make it a school"""

        with open("../Config/centroids.yaml", "r") as f:
            # with open("../Config/school_closure_centroids.yaml", "r") as f:
            centroid_configs = yaml.safe_load(f)
        if self.centroid_type not in centroid_configs:
            raise ValueError(
                "The centroids type specified is not defined in centroids.yaml.")

        self.centroid_sch = centroid_configs[self.centroid_type]

        self.school_df['is_centroid'] = self.school_df['school_id'].apply(lambda x: 1 if x in self.centroid_sch else 0)

        if self.include_k8:
            self.centroid_location = self.school_df[self.school_df['is_centroid'] == 1][['lon', 'lat', 'school_id']]
        else:
            self.centroid_location = self.school_df[self.school_df['is_centroid'] == 1][['lon', 'lat', 'school_id']]
            self.schools_locations = self.school_df[['lon', 'lat', 'school_id']]

        centroid_areas = [self.sch2area[x] for x in self.centroid_sch]
        self.centroids = [self.area2idx[j] for j in centroid_areas]

    def load_neighborhood_dict(self):
        """ build a dictionary mapping a block group/attendance area to a list
        of its neighboring block groups/attendnace areas"""
        if self.level == "Block":
            file = os.path.expanduser(f"{get_dropbox_path(self.is_local)}/Optimization/adjacency_matrix_b.csv")

        elif self.level == "BlockGroup":
            file = os.path.expanduser(f"{get_dropbox_path(self.is_local)}/Optimization/adjacency_matrix_bg.csv")

        elif self.level == "attendance_area":
            file = os.path.expanduser(f"{get_dropbox_path(self.is_local)}/Optimization/adjacency_matrix_aa.csv")

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

    def initialize_centroid_neighbors(self):
        """ for each centroid c and each area j, define a set n(j,c) to be all neighbors of j that are closer to c than j"""
        save_path = os.path.expanduser("~/Dropbox/SFUSD/Optimization/59zone_contiguity_constraint.pkl")

        if (self.level == "Block") and (self.centroid_type == '59-zone-1'):
            if os.path.exists(os.path.expanduser(save_path)):
                with open(save_path, 'rb') as file:
                    self.closer_euc_neighbors = pickle.load(file)
                return

        self.closer_euc_neighbors = {}
        for z in self.centroids:
            for idx in range(self.A):
                n = self.neighbors[idx]
                closer = [x for x in n
                          if self.euc_distances[z][idx]
                          >= self.euc_distances[z][x]
                          ]
                self.closer_euc_neighbors[idx, z] = closer

        if (self.level == "Block") and (self.centroid_type == '59-zone-1'):
            with open(save_path, 'wb') as file:
                pickle.dump(self.closer_euc_neighbors, file)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def save(self, path, name="", solve_success=1):
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


class SolutionOutput:
    def __init__(self, zone_dict, objective_value, status, wall_time, dz: DesignZones):
        self.zone_dict = zone_dict
        self.objective_value = objective_value
        self.status = status
        self.wall_time = wall_time
        self.dz = dz

    def get_boundary_cost(self):
        boundary_cost = 0
        for i in range(self.dz.A):
            a = self.dz.idx2area[i]
            for j in self.dz.neighbors[i]:
                b = self.dz.idx2area[j]
                if self.zone_dict[a] != self.zone_dict[b]:
                    boundary_cost += 1
        return boundary_cost / 2  # each boundary counted twice

    # Assuming AREA_ETHNICITIES is defined somewhere and is a list of strings like ['Ethnicity_White', 'Ethnicity_Black', ...]

    def get_zone_demographics(self):
        """
        Calculates aggregated and proportional demographics (total students, FRL proportion,
        and racial/ethnic proportions) for each zone.

        Assumes self.dz.area_data is a pandas DataFrame and self.zone_dict maps area to zone_id.
        """

        # 1. Define the columns to aggregate and the corresponding keys for the output dictionary
        # The keys will be like 'White', 'Black', etc.
        ETHNICITY_KEYS = [col[len('Ethnicity_'):] for col in AREA_ETHNICITIES]

        # 2. Use a defaultdict to simplify accumulation and avoid checking for key existence
        # We initialize with a lambda that returns a dict structure for aggregation
        zone_aggregates = defaultdict(lambda: {
            "total_students": 0.0,
            "FRL": 0.0,
            **{key: 0.0 for key in ETHNICITY_KEYS}
        })

        # 3. Iterate over the area data and aggregate by zone
        # Using iterrows or a direct index-based loop can be fine, but pandas' built-in
        # grouping/aggregation methods are often more efficient for large data.
        # We'll stick to a loop here since the original code did, but optimize the lookups.

        # Iterate over the area data index
        for i, row in self.dz.area_data.iterrows():
            area = self.dz.idx2area[i]
            zone_id = self.zone_dict[area]

            # Reference the aggregate dictionary for the current zone
            agg = zone_aggregates[zone_id]

            # Accumulate total students and FRL counts
            agg["total_students"] += float(row["ge_students"])
            agg["FRL"] += float(row["FRL"])

            # Accumulate ethnicity counts
            for ethnicity_col, key in zip(AREA_ETHNICITIES, ETHNICITY_KEYS):
                agg[key] += float(row[ethnicity_col])

        # 4. Convert aggregated counts to proportions
        final_zone_demographics = {}
        for zone_id, data in zone_aggregates.items():
            total_students = data["total_students"]

            # Create a new dictionary for the final, proportional results
            final_demographics = {"total_students": round(total_students, 2)}

            if total_students > 0:
                # Calculate FRL proportion
                final_demographics["FRL"] = round(data["FRL"] / total_students, 2)

                # Calculate ethnicity proportions
                for key in ETHNICITY_KEYS:
                    final_demographics[key] = round(data[key] / total_students, 2)
            else:
                # Handle zones with zero students (FRL and ethnic proportions are 0)
                final_demographics["FRL"] = 0.0
                for key in ETHNICITY_KEYS:
                    final_demographics[key] = 0.0

            final_zone_demographics[zone_id] = final_demographics

        return final_zone_demographics

    def save_output(self, folder_path):
        # save the image, dict_solution, objective_value, status, wall_time
        save_path = os.path.expanduser(folder_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save zone dict as json file
        filename = os.path.join(save_path, "zone_dict.json")
        with open(filename, "w") as f:
            json.dump(self.zone_dict, f)

        boundary_cost = -1
        if self.zone_dict is not None and len(self.zone_dict) > 0:
            boundary_cost = self.get_boundary_cost()
            file_name = os.path.join(save_path, "zones_visualization")
            zv = ZoneVisualizer(self.dz.level, self.dz.is_local)
            zv.zones_from_dict(self.zone_dict, save_path=file_name, label=False)

        output_info = {
            "boundary_cost": boundary_cost,
            "status": self.status,
            "wall_time": self.wall_time
        }
        filename = os.path.join(save_path, "solution_info.json")
        with open(filename, "w") as f:
            json.dump(output_info, f)


class Optimizer:
    def __init__(self, dz: DesignZones, config):
        self.dz = dz

        self.config = config

    def add_constraints(self):
        raise NotImplementedError('Subclasses must implement add_constraints')

    def add_objective(self):
        raise NotImplementedError('Subclasses must implement add_objective')

    def solve(self) -> SolutionOutput:
        raise NotImplementedError('Subclasses must implement solve')

    def fix_areas(self, fixed_zone_dict):
        raise NotImplementedError('Subclasses must implement fix_areas')

    def _add_hints(self):
        raise NotImplementedError('Subclasses must implement _add_hints')

    @staticmethod
    def get_optimizer(dz: DesignZones, config):
        # mip, cp_int, cp_bool
        if config["optimizer"] == "mip":
            from Zone_Generation.Optimization.integer_program import Integer_Program
            return Integer_Program(dz, config)
        elif config["optimizer"] == "cp_int":
            from Zone_Generation.Optimization.constraint_program_integer import IntegerConstraintProgram
            return IntegerConstraintProgram(dz, config)
        elif config["optimizer"] == "cp_bool":
            from Zone_Generation.Optimization.constraint_program_boolean import BooleanConstraintProgram
            return BooleanConstraintProgram(dz, config)
        else:
            raise ValueError("The optimizer type specified is not recognized.")
