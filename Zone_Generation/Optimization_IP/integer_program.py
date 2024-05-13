import gurobipy as gp
import pandas as pd
import random, math, gc, os, csv
import numpy as np
from gurobipy import GRB
from Zone_Generation.Config.Constants import *

class Integer_Program(object):
    def __init__(self, Area_Data):
        # Number of zones. This is given to us as input.
        # We are trying to divide the city into self.Z number of zones
        self.Z = Area_Data.Z

        # self.N: Total number of GE students.
        # Computation based on area_data file: self.N = sum(self.area_data["ge_students"])
        self.N = Area_Data.N

        # self.A: Total number of areas (number of distinct area indices)
        # Computation based on area_data file: self.A = len(self.area_data.index)
        self.A = Area_Data.A

        # self.F: Average percentage of FRL students
        # (students that are eligible for Free or reduced price lunch)
        # Computation based on area_data file: self.F = sum(self.area_data["FRL"]) / (self.N)
        self.F = Area_Data.F

        # Should K8 schools be considered into the calculations or not
        self.include_k8 = Area_Data.include_k8

        # self.studentsInArea is a dictionary of:
        # (keys: area index j), (values: number of GE students in area index j)
        # Example: self.studentsInArea[41] == number of GE students in area index 41
        # Computation based on area_data file: self.studentsInArea = self.area_data["ge_students"]
        self.studentsInArea = Area_Data.studentsInArea

        # self.seats is a dictionary of:
        # (keys: area index j), (values: number of seats for GE students in area index j)
        # Example: self.seats[41] == number of seats for GE students in area index 41
        # Computation based on area_data file: self.seats = self.area_data["ge_capacity"].to_numpy()
        self.seats = Area_Data.seats

        # self.schools is a dictionary of:
        # (keys: area index j), (values: number of schools in area index j) this value is usually 0 or 1
        # Example: self.schools[41] == number of schools in area index 41 (this value is usually 0 or 1)
        # Computation based on area_data file: self.schools = self.area_data['num_schools']
        self.schools = Area_Data.schools
        self.school_df = Area_Data.school_df

        # Most important, and most comprehensive data file.
        # dataself.area_data is a pandas dataframe. Each column shows a metric for areas
        # Each row, represent an area.
        # Example: self.area_data['ge_students'][41] == number of GE students in area with index 41.
        self.area_data = Area_Data.area_data

        self.centroids = Area_Data.centroids

        # self.euc_distances is a dictionary of:
        # (keys: a pair of (area index i, area index j)), (values: euclidean distance, in miles, between area i and area j
        # Example: self.euc_distances[41, 13] == 3.42, which means area 41 and area 13 are 3.42 miles away
        self.euc_distances = Area_Data.euc_distances

        # self.neighbors is a dictionary of,
        # (keys: area index j), (values: a list of indices of neighboring areas, to area j.
        # Example: self.neighbors[41] == [32, 12, 52, 2], which is a list of indices of areas, adjacent to area 41.
        self.neighbors = Area_Data.neighbors

        # self.closer_euc_neighbors is a dictionary of:
        # (keys: a pair of (area index j, zone index z)), (values: a list of indices of
        # neighboring areas, to area j, that are closer to the area of centroid z than araa j
        # Example: self.closer_euc_neighbors[41, 3] == [32, 2], which is a list of indices of areas,
        # adjacent to area 41, that are closer to the area of centroid 3 than araa 41
        self.closer_euc_neighbors = Area_Data.closer_euc_neighbors

        self.level = Area_Data.level
        # self.idx2area: A dictionary, mapping each area index in our data, to its census area code
        # (keys: area index j), (values: census area code for the area with index j)
        # Example: self.idx2area[41] == census area code for the area with index 41)
        # Computation based on area_data file: self.idx2area = dict(zip(self.area_data.index, self.area_data[self.level]))
        self.idx2area = Area_Data.idx2area



        # self.area2idx: A dictionary, mapping each census area code, to its index in our data
        # (keys: census area code AA), (values: index of area AA, in our data set)
        # Example: self.area2idx[area code AA] == index of area AA in our data set
        # Note that we can access our dictionaries only using the area index, and not the area code
        # Computation based on area_data file: self.area2idx = dict(zip(self.area_data[self.level], self.area_data.index))
        self.area2idx = Area_Data.area2idx

        # self.sch2area: A dictionary, mapping each school id, to its census area code
        # Example: self.sch2area[644] == sensus area code for the school, with school id 644
        # Computation based on area_data file: self.sch2area = dict(zip(self.school_df["school_id"], self.school_df[self.level]))
        self.sch2area = Area_Data.sch2area


    def _feasibility_const(self, sub_units=None, max_distance=float('inf')):
        valid_assignments = []
        # if a max distance constraint is given, allow areas to be matched only to
        # zone centroids that are closer than max_distance
        for z in range(self.Z):
            centroid_z = self.centroids[z]
            # zone_max_distance = max_distance
            for i in range(self.A):
                if sub_units != None:
                    if self.idx2area[i] not in sub_units:
                        # print("Error! ", i)
                        continue
                if (self.euc_distances[centroid_z][i] < max_distance):
                    valid_assignments.append((i,z))


        # Initialize a dictionary to hold valid zones for each area
        self.valid_area_per_zone = {z: [] for z in range(self.Z)}
        # Initialize a dictionary to hold valid zones for each area
        self.valid_zone_per_area = {i: [] for i in range(self.A)}

        # Populate the dictionary with valid zones for each area
        for i, z in valid_assignments:
            self.valid_area_per_zone[z].append(i)
            self.valid_zone_per_area[i].append(z)

        self.m = gp.Model("Zone model")

        # Variable self.x[i,z]: is a binary variable. It indicates
        # whether area with index i is assigned to zone z or not.
        # Example: if self.x[41,2] == 0, it means area with index 41 is not assigned to zone 2.
        self.x = self.m.addVars(valid_assignments, vtype=GRB.BINARY, name="x")

        # Feasiblity Constraint: every area must  belong to exactly one zone
        self.m.addConstrs(
            (gp.quicksum(self.x[i, z] for z in self.valid_zone_per_area[i]) == 1
            for i in range(self.A)
             # if self.idx2area[i] in sub_units
             ),
        )


    def set_y_distance(self):
        y_distance = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="distance distortion")

        for z in range(self.Z):
            zone_dist_sum = gp.quicksum([((self.euc_distances[self.centroids[z]][j]) ** 2) * self.x[j, z] for j in range(self.A)])
            # zone_dist_sum = gp.quicksum([((self.drive_distances.loc[centroid_area, str(self.idx2area[j])]) ** 2) * self.x[j, z] for j in range(self.A)])
            self.m.addConstr(zone_dist_sum <= y_distance)
        return y_distance


    def set_y_balance(self):
        y_balance = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="balance distortion")

        # minimize the maximum distortion from average number of students (across zones)
        for z in range(self.Z):
            zone_stud = gp.quicksum([self.studentsInArea[j]*self.x[j,z] for j in range(self.A)])
            self.m.addConstr(self.N/self.Z - zone_stud <= y_balance)
            self.m.addConstr(zone_stud - self.N/self.Z <= y_balance)
        return y_balance

    def set_y_shortage(self):
        y_shortage = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="shortage distortion")

        # minimize the maximum distortion from average student
        # deficit (student capacity -  number of seats) (across zones)
        for z in range(self.Z):
            zone_stud = gp.quicksum([self.studentsInArea[j]*self.x[j,z] for j in range(self.A)])
            zone_seats = gp.quicksum([self.seats[j]*self.x[j,z] for j in range(self.A)])
            self.m.addConstr(zone_stud - zone_seats <= y_shortage)
        return y_shortage

    # This function constructs the boundary cost variables.
    # Boundary cost variables are used in the optimization model objective
    def set_y_boundary(self):
        neighboring_tuples = []
        for i in range(self.A):
            for j in self.neighbors[i]:
                if i >= j:
                    continue
                neighboring_tuples.append((i,j))

        # self.b[i, j]: a binary boundary variable. This variable will be 1,
        # if area with index i, and area with index j, are adjacent areas, that
        # are assigned to different zones (hence, they will be part of boundary cost)
        self.b = self.m.addVars(neighboring_tuples, vtype=GRB.BINARY, name="boundary_vars")
        y_boundary = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="boundary distortion")
        self.m.addConstr(gp.quicksum(self.b[i, j] for i, j in neighboring_tuples) == y_boundary)
        self._add_boundary_constraint()
        return y_boundary

    def _add_boundary_constraint(self):
        # if i and j are neighbors, check if they are boundaries of different zones
        for i in range(self.A):
            for j in self.neighbors[i]:
                if i >= j:
                    continue
                for z in range(self.Z):
                    if (i in self.valid_area_per_zone[z]) and (j in self.valid_area_per_zone[z]):
                        self.m.addConstr(gp.quicksum([self.x[i, z], -1 * self.x[j, z]]) <= self.b[i, j])
                        self.m.addConstr(gp.quicksum([-1 * self.x[i, z], self.x[j, z]]) <= self.b[i, j])
                    elif (i in self.valid_area_per_zone[z]):
                        self.m.addConstr(self.x[i, z] <= self.b[i, j])
                    elif (j in self.valid_area_per_zone[z]):
                        self.m.addConstr(self.x[j, z] <= self.b[i, j])


    def _set_objective_model(self):
        # y_distance = self.set_y_distance()
        # distance_coef = 1
        #
        # y_balance = self.set_y_balance()
        # balance_coef = 0
        #
        # y_shortage = self.set_y_shortage()
        # shortage_coef = 2

        y_boundary = self.set_y_boundary()
        boundary_coef = 10

        # set the objective of the Integer Program.
        # The integer program will try to minimize the cost of boundary,
        # which will result into compact and nice looking shapes for zones.
        self.m.setObjective(boundary_coef * y_boundary, GRB.MINIMIZE)
        # self.m.setObjective(boundary_coef * y_shortage, GRB.MINIMIZE)
        # self.m.setObjective(1 , GRB.MINIMIZE)
        # self.m.setObjective(distance_coef * y_distance +  shortage_coef * y_shortage +
        #                      balance_coef * y_balance + boundary_coef * y_boundary , GRB.MINIMIZE)


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
    def _all_cap_proportional_shortage_const(self, all_cap_shortage):
        # No zone has shortage more than all_cap_shortage percentage of its total student population
        for z in range(self.Z):
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


    # proportional shortage for each zone =
    # percentage of students (GE students) in the zone, that don't get any seat (from GE capacities)
    # students in the zone
    # The following constraint makes sure no zone has a shortage
    # larger than the given input "shortage"
    def _proportional_shortage_const(self, shortage):
        # No zone has shortage more than shortage percentage of its population
        for z in range(self.Z):
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

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_constraint(self, overage):
        # No zone has overage more than overage percentage of its population
        for z in range(self.Z):
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

    def _absolute_shortage_const(self, shortage):
        # each zone has at least the shortage
        for z in range(self.Z):
            self.m.addConstr(
                gp.quicksum(
                    [(self.studentsInArea[j] - self.seats[j]) * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <= shortage)

    def _shortage_const(self, shortage=0.15, overage=0.2, all_cap_shortage=0.8):
        if shortage <= 1:
            self._proportional_shortage_const(shortage)
        if overage <= 1:
            self._proportional_overage_constraint(overage)
        if all_cap_shortage <= 1:
            self._all_cap_proportional_shortage_const(all_cap_shortage)




    # Designing contiguous school zones is desirable for practical reasons,
    # i.e. school commutes and policy communication.
    # Make sure areas assigned to each zone form a contiguous zone as follows:
    # assign unit 𝑗 to zone with centroid area 𝑧, only if
    # there is a ‘path’ of closer neighboring areas also assigned
    # to the same zone that connects area 𝑗 to the centroid area 𝑧.
    def _contiguity_const(self, sub_units=None):
        # initialization - every centroid belongs to its own zone
        for z in range(self.Z):
            self.m.addConstr(
                self.x[self.centroids[z], z] == 1, name="Centroids to Zones"
            )

        # (x[j,z] (and indicator that unit j is assigned to zone z)) \leq
        # (sum of all x[j',z] where j' is in self.closer_neighbors_per_centroid[area,c] where c is centroid for z)
        for j in range(self.A):
            if sub_units != None:
                if self.idx2area[j] not in sub_units:
                    continue
            for z in range(self.Z):
                if j == self.centroids[z]:
                    continue
                if self.centroids[z] in self.neighbors[j]:
                    continue
                if j not in self.valid_area_per_zone[z]:
                    continue
                # only impose the contiguity if the area j has a neighbor that is closer to centroid z.
                # otherwise, just make sure j has at least another neighbor assigned tot the same zone z, so that
                # j is not an island assigned to z.
                if len(self.closer_euc_neighbors[j, self.centroids[z]]) >= 1:
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


    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Add constraints related to diversity such as: racial balance,
    # frl balance (balance in free or reduced priced lunch eligibility)
    # and aalpi balance, across all the zones.
    def _diversity_const(self, racial_dev=1, frl_dev=1, aalpi_dev=1):
        # racial balance constraint
        if racial_dev < 1:
            self._racial_const(racial_dev)

        # frl constraint
        if frl_dev < 1:
                self._add_frl_constraint(frl_dev)

        # aalpi constraint
        if aalpi_dev < 1:
            self._add_aalpi_constraint(aalpi_dev)

    # Enforce zones to have almost the same number of students
    # Make sure the difference between total population of GE students
    # among two different zone is at most _balance.
    def _absolute_population_const(self, _balance=1000):
        # add number of students balance constraint
        for z in range(self.Z):
            firstZone = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            for q in range(z + 1, self.Z):
                secondZone = gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, q] for j in self.x[z]]
                )
                self.m.addConstr(firstZone - secondZone <= _balance)
                self.m.addConstr(firstZone - secondZone >= -_balance)


    # Enforce zones to have almost the same number of students
    # Make sure the average population of each zone, is within a given
    # population_dev% of average population over zones
    def _proportional_population_const(self, population_dev=1):
        average_population = sum(self.area_data["all_prog_students"])/self.Z
        for z in range(self.Z):
            zone_sum = gp.quicksum(
                [self.area_data["all_prog_students"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])

            self.m.addConstr(zone_sum >= (1 - population_dev) * average_population, name= "Population LB")
            self.m.addConstr(zone_sum <= (1 + population_dev) * average_population, name= "Population UB")


    # Make sure students of racial groups are fairly distributed among zones.
    # For specific racial minority, make sure the percentage of students in each zone, is within an additive
    #  race_dev% of percentage of total students of that race.
    def _racial_const(self, race_dev=1):
        for race in ETHNICITY_COLS:
            race_ratio = sum(self.area_data[race]) / float(self.N)

            for z in range(self.Z):
                zone_sum = gp.quicksum(
                    [self.area_data[race][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                district_students = gp.quicksum(
                    [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                self.m.addConstr(zone_sum >= (race_ratio - race_dev) * district_students, name= str(race) + " LB")
                self.m.addConstr(zone_sum <= (race_ratio + race_dev) * district_students, name= str(race) + " UB")



    # Make sure students of low socioeconomic status groups are fairly distributed among zones.
    # Our only metric to measure socioeconomic status, is FRL, which is the students eligibility for
    # Free or Reduced Price Lunch.
    # make sure the total FRL for students in each zone, is within an additive
    #  frl_dev% of average FRL over zones..
    def _add_frl_constraint(self, frl_dev=1):
        for z in range(self.Z):
            zone_sum = gp.quicksum(
                [self.area_data["FRL"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            district_students = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            self.m.addConstr(zone_sum >= (self.F - frl_dev) * district_students, name="FRL LB")
            self.m.addConstr(zone_sum <= (self.F + frl_dev) * district_students, name="FRL UB")




    def _add_aalpi_constraint(self, aalpi_dev):
        district_average = sum(self.area_data["AALPI Score"]) / self.N
        for z in range(self.Z):
            zone_sum = gp.quicksum(
                [self.area_data["AALPI Score"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            district_students = gp.quicksum(
                [self.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            self.m.addConstr(zone_sum >= (district_average - aalpi_dev) * district_students, name="AALPI LB")
            self.m.addConstr(zone_sum <= (district_average  + aalpi_dev) * district_students, name="AALPI UB")


    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # This following constraint makes sure all zones have almost similar number of schools.
    # First compute the average number of schools per zone,
    # by computing the total number of schools in the city and dividing it by the number of zones.
    # Next, add a constraint to make sure the number of schools in each zone
    # is within average number of schools per zone + or - 1
    def _add_school_count_const(self, sub_units=None):
        zone_school_count = {}
        if sub_units != None:
            avg_school_count = sum([self.schools[j] for j in range(self.A) if self.idx2area[j] in sub_units]) / self.Z + 0.0001
        else:
            avg_school_count = sum([self.schools[j] for j in range(self.A)]) / self.Z + 0.0001
        print("avg_school_count ", avg_school_count)

        # note: although we enforce max deviation of 1 from avg, in practice,
        # no two zones will have more than 1 difference in school count
        # Reason: school count is int. Observe the avg_school_count +-1,
        # if avg_school_count is not int, and see how the inequalities will look like
        # * I implemented the code this way (instead of pairwise comparison), since it is faster
        for z in range(self.Z):
            zone_school_count[z] = gp.quicksum([self.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]])
            self.m.addConstr(zone_school_count[z] <= avg_school_count + 1)
            self.m.addConstr(zone_school_count[z] >= avg_school_count - 1)

        # if K8 schools are included,
        # make sure no zone has more than one K8 schools
        if self.include_k8:
            zone_k8_count = {}
            for z in range(self.Z):
                zone_k8_count[z] = gp.quicksum([self.area_data["K-8"][j] * self.x[j, z]
                                                for j in self.valid_area_per_zone[z]])
                self.m.addConstr(zone_k8_count[z] <= 1)


    # Enforce a balance in english score over schools of different zones as follows:
    # Compute the average: average english score over all schools in the district.
    # Sum up english scores for schools of each zone. Divide the english score for each zone,
    # by total number of schools within that zone.
    # Make sure the average english score for each zone,
    # is between (1-score_dev) * average and (1+score_dev) * average
    def _school_eng_score_quality_const(self, score_dev=-1):
        if not (1 > score_dev > -1):
            return
        eng_scores = self.area_data["english_score"].fillna(value=0)
        school_average = sum(eng_scores) / sum(self.schools)

        for z in range(self.Z):
            zone_sum = gp.quicksum(
                [eng_scores[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            zone_schools = gp.quicksum(
                [self.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            self.m.addConstr(zone_sum >= (1 - score_dev) * school_average * zone_schools)
            self.m.addConstr(zone_sum <= (1 + score_dev) * school_average * zone_schools)

    def _school_math_score_quality_const(self, score_dev=-1):
        if not (1 > score_dev > -1):
            return

        math_scores = self.area_data["math_score"].fillna(value=0)
        school_average = sum(math_scores) / sum(self.schools)

        for z in range(self.Z):
            zone_sum = gp.quicksum(
                [math_scores[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            zone_schools = gp.quicksum(
                [self.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            self.m.addConstr(zone_sum >= (1 - score_dev) * school_average * zone_schools)
            self.m.addConstr(zone_sum <= (1 + score_dev) * school_average * zone_schools)



    # Enforce school quality balance constraint, using "AvgColorIndex" metric, which is:
    # Average of ela_color, math_color, chronic_color, and suspension_color, where Red=1 and Blue=5
    # Make sure all zones are within min_pct and max_pct of average of AvgColorIndex for each zone
    # min_pct: min percentage. max_pct: max percentage
    def _color_quality_const(self, score_dev=-1, topX=0):
        if not (1 > score_dev > -1):
            return
        color_scores = self.area_data["AvgColorIndex"].fillna(value=0)
        school_average = sum(color_scores) / sum(self.schools)

        for z in range(self.Z):
            zone_sum = gp.quicksum(
                [color_scores[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            zone_schools = gp.quicksum(
                [self.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            self.m.addConstr(zone_sum >= (1 - score_dev) * school_average * zone_schools)
            self.m.addConstr(zone_sum <= (1 + score_dev) * school_average * zone_schools)


        if topX > 0:
            top_schools = np.zeros([self.A])
            top = np.percentile(color_scores, 100 * (1 - self.Z / self.A) - 0.05)
            top = np.percentile(color_scores, topX)
            print(top)
            for j in range(self.A):
                if color_scores[j] > top:
                    top_schools[j] = 1
            for z in range(self.Z):
                topz = gp.quicksum(
                    [self.x[j, z] * top_schools[j] for j in self.valid_area_per_zone[z]]
                )
                self.m.addConstr(topz >= 0.8)
