import gurobipy as gp
import numpy as np
from gurobipy import GRB

from Zone_Generation.Config.Constants import *
from Zone_Generation.Optimization.optimizer import Optimizer, DesignZones, SolutionOutput


class Integer_Program(Optimizer):
    def __init__(self, Area_Data: DesignZones, config):
        super().__init__(Area_Data, config)

    def _feasibility_const(self, sub_units=None, max_distance=float('inf')):
        valid_assignments = []
        # if a max distance constraint is given, allow areas to be matched only to
        # zone centroids that are closer than max_distance
        for z in range(self.dz.Z):
            centroid_z = self.dz.centroids[z]
            # zone_max_distance = max_distance
            for i in range(self.dz.A):
                if sub_units != None:
                    if self.dz.idx2area[i] not in sub_units:
                        # print("Error! ", i)
                        continue
                if (self.dz.euc_distances[centroid_z][i] < max_distance):
                    valid_assignments.append((i, z))

        # Initialize a dictionary to hold valid zones for each area
        self.valid_area_per_zone = {z: [] for z in range(self.dz.Z)}
        # Initialize a dictionary to hold valid zones for each area
        self.valid_zone_per_area = {i: [] for i in range(self.dz.A)}

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
             for i in range(self.dz.A)
             # if self.idx2area[i] in sub_units
             ),
        )

    def set_y_distance(self):
        y_distance = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="distance distortion")

        for z in range(self.dz.Z):
            zone_dist_sum = gp.quicksum(
                [((self.dz.euc_distances[self.dz.centroids[z]][j]) ** 2) * self.x[j, z] for j in range(self.dz.A)])
            # zone_dist_sum = gp.quicksum([((self.drive_distances.loc[centroid_area, str(self.idx2area[j])]) ** 2) * self.x[j, z] for j in range(self.A)])
            self.m.addConstr(zone_dist_sum <= y_distance)
        return y_distance

    def set_y_balance(self):
        y_balance = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="balance distortion")

        # minimize the maximum distortion from average number of students (across zones)
        for z in range(self.dz.Z):
            zone_stud = gp.quicksum([self.dz.studentsInArea[j] * self.x[j, z] for j in range(self.dz.A)])
            self.m.addConstr(self.dz.N / self.dz.Z - zone_stud <= y_balance)
            self.m.addConstr(zone_stud - self.dz.N / self.dz.Z <= y_balance)
        return y_balance

    def set_y_shortage(self):
        y_shortage = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="shortage distortion")

        # minimize the maximum distortion from average student
        # deficit (student capacity -  number of seats) (across zones)
        for z in range(self.Z):
            zone_stud = gp.quicksum([self.dz.studentsInArea[j] * self.x[j, z] for j in range(self.dz.A)])
            zone_seats = gp.quicksum([self.dz.seats[j] * self.x[j, z] for j in range(self.dz.A)])
            self.m.addConstr(zone_stud - zone_seats <= y_shortage)
        return y_shortage

    # This function constructs the boundary cost variables.
    # Boundary cost variables are used in the optimization model objective
    def set_y_boundary(self):
        neighboring_tuples = []
        for i in range(self.dz.A):
            for j in self.dz.neighbors[i]:
                if i >= j:
                    continue
                neighboring_tuples.append((i, j))

        # self.b[i, j]: a binary boundary variable. This variable will be 1,
        # if area with index i, and area with index j, are adjacent areas, that
        # are assigned to different zones (hence, they will be part of boundary cost)
        self.b = self.m.addVars(neighboring_tuples, vtype=GRB.BINARY, name="boundary_vars")
        y_boundary = self.m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="boundary distortion")
        self.m.addConstr(gp.quicksum(self.b[i, j] for i, j in neighboring_tuples) == y_boundary)
        self._boundary_constraint()
        return y_boundary

    def _boundary_constraint(self):
        # if i and j are neighbors, check if they are boundaries of different zones
        for i in range(self.dz.A):
            for j in self.dz.neighbors[i]:
                if i >= j:
                    continue
                for z in range(self.dz.Z):
                    if (i in self.valid_area_per_zone[z]) and (j in self.valid_area_per_zone[z]):
                        self.m.addConstr(gp.quicksum([self.x[i, z], -1 * self.x[j, z]]) <= self.b[i, j])
                        self.m.addConstr(gp.quicksum([-1 * self.x[i, z], self.x[j, z]]) <= self.b[i, j])
                    elif (i in self.valid_area_per_zone[z]):
                        self.m.addConstr(self.x[i, z] <= self.b[i, j])
                    elif (j in self.valid_area_per_zone[z]):
                        self.m.addConstr(self.x[j, z] <= self.b[i, j])

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
        for z in range(self.dz.Z):
            self.m.addConstr(
                gp.quicksum(
                    [(self.dz.area_data["all_prog_students"][j] - self.dz.area_data["all_prog_capacity"][j]) * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                all_cap_shortage *
                gp.quicksum(
                    [self.dz.area_data["all_prog_students"][j] * self.x[j, z]
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
        for z in range(self.dz.Z):
            self.m.addConstr(
                (1 - shortage) *
                gp.quicksum(
                    [self.dz.studentsInArea[j] * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                gp.quicksum(
                    [self.dz.seats[j] * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
            )

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_constraint(self, overage):
        # No zone has overage more than overage percentage of its population
        for z in range(self.dz.Z):
            self.m.addConstr(
                gp.quicksum(
                    [(-self.dz.studentsInArea[j] + self.dz.seats[j]) * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                overage *
                gp.quicksum(
                    [self.dz.studentsInArea[j] * self.x[j, z]
                     for j in self.valid_area_per_zone[z]]
                )
            )

    def _absolute_shortage_const(self, shortage):
        # each zone has at least the shortage
        for z in range(self.dz.Z):
            self.m.addConstr(
                gp.quicksum(
                    [(self.dz.studentsInArea[j] - self.dz.seats[j]) * self.x[j, z]
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
        for z in range(self.dz.Z):
            self.m.addConstr(
                self.x[self.dz.centroids[z], z] == 1, name="Centroids to Zones"
            )

        # (x[j,z] (and indicator that unit j is assigned to zone z)) \leq
        # (sum of all x[j',z] where j' is in self.closer_neighbors_per_centroid[area,c] where c is centroid for z)
        for j in range(self.dz.A):
            if sub_units != None:
                if self.dz.idx2area[j] not in sub_units:
                    continue
            for z in range(self.dz.Z):
                if j == self.dz.centroids[z]:
                    continue
                if self.dz.centroids[z] in self.dz.neighbors[j]:
                    continue
                if j not in self.valid_area_per_zone[z]:
                    continue
                # only impose the contiguity if the area j has a neighbor that is closer to centroid z.
                # otherwise, just make sure j has at least another neighbor assigned tot the same zone z, so that
                # j is not an island assigned to z.
                if len(self.dz.closer_euc_neighbors[j, self.dz.centroids[z]]) >= 1:
                    neighbor_sum = gp.quicksum(
                        self.x[k, z]
                        for k in self.dz.closer_euc_neighbors[j, self.dz.centroids[z]]
                        if k in self.valid_area_per_zone[z]
                    )
                    self.m.addConstr(self.x[j, z] <= neighbor_sum, name="Contiguity")
                else:
                    any_neighbor_sum = gp.quicksum(
                        [
                            self.x[k, z]
                            for k in self.dz.neighbors[j] if k in self.valid_area_per_zone[z]
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
            self._frl_constraint(frl_dev)

        # aalpi constraint
        if aalpi_dev < 1:
            self._aalpi_constraint(aalpi_dev)

    # Enforce zones to have almost the same number of students
    # Make sure the difference between total population of GE students
    # among two different zone is at most _balance.
    def _absolute_population_const(self, _balance=1000):
        # add number of students balance constraint
        for z in range(self.dz.Z):
            firstZone = gp.quicksum(
                [self.dz.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            for q in range(z + 1, self.dz.Z):
                secondZone = gp.quicksum(
                    [self.dz.studentsInArea[j] * self.x[j, q] for j in self.x[z]]
                )
                self.m.addConstr(firstZone - secondZone <= _balance)
                self.m.addConstr(firstZone - secondZone >= -_balance)

    # Enforce zones to have almost the same number of students
    # Make sure the average population of each zone, is within a given
    # population_dev% of average population over zones
    def _proportional_population_const(self, population_dev=1):
        average_population = sum(self.dz.area_data["all_prog_students"]) / self.dz.Z
        for z in range(self.dz.Z):
            zone_sum = gp.quicksum(
                [self.dz.area_data["all_prog_students"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]])

            self.m.addConstr(zone_sum >= (1 - population_dev) * average_population, name="Population LB")
            self.m.addConstr(zone_sum <= (1 + population_dev) * average_population, name="Population UB")

    # Make sure students of racial groups are fairly distributed among zones.
    # For specific racial minority, make sure the percentage of students in each zone, is within an additive
    #  race_dev% of percentage of total students of that race.
    def _racial_const(self, race_dev=1):
        for race in AREA_ETHNICITIES:
            race_ratio = sum(self.dz.area_data[race]) / float(self.dz.N)

            for z in range(self.dz.Z):
                zone_sum = gp.quicksum(
                    [self.dz.area_data[race][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                district_students = gp.quicksum(
                    [self.dz.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
                )
                self.m.addConstr(zone_sum >= (race_ratio - race_dev) * district_students, name=str(race) + " LB")
                self.m.addConstr(zone_sum <= (race_ratio + race_dev) * district_students, name=str(race) + " UB")

    # Make sure students of low socioeconomic status groups are fairly distributed among zones.
    # Our only metric to measure socioeconomic status, is FRL, which is the students eligibility for
    # Free or Reduced Price Lunch.
    # make sure the total FRL for students in each zone, is within an additive
    #  frl_dev% of average FRL over zones..
    def _frl_constraint(self, frl_dev=1):
        for z in range(self.dz.Z):
            zone_sum = gp.quicksum(
                [self.dz.area_data["FRL"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            district_students = gp.quicksum(
                [self.dz.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            self.m.addConstr(zone_sum >= (self.dz.F - frl_dev) * district_students, name="FRL LB")
            self.m.addConstr(zone_sum <= (self.dz.F + frl_dev) * district_students, name="FRL UB")

    def _aalpi_constraint(self, aalpi_dev):
        district_average = sum(self.dz.area_data["AALPI Score"]) / self.dz.N
        for z in range(self.dz.Z):
            zone_sum = gp.quicksum(
                [self.dz.area_data["AALPI Score"][j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            district_students = gp.quicksum(
                [self.dz.studentsInArea[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            self.m.addConstr(zone_sum >= (district_average - aalpi_dev) * district_students, name="AALPI LB")
            self.m.addConstr(zone_sum <= (district_average + aalpi_dev) * district_students, name="AALPI UB")

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # This following constraint makes sure all zones have almost similar number of schools.
    # First compute the average number of schools per zone,
    # by computing the total number of schools in the city and dividing it by the number of zones.
    # Next, add a constraint to make sure the number of schools in each zone
    # is within average number of schools per zone + or - 1
    def _school_count_const(self, sub_units=None):
        zone_school_count = {}
        if sub_units != None:
            avg_school_count = sum(
                [self.dz.schools[j] for j in range(self.dz.A) if self.dz.idx2area[j] in sub_units]) / self.dz.Z + 0.0001
        else:
            avg_school_count = sum([self.dz.schools[j] for j in range(self.dz.A)]) / self.dz.Z + 0.0001
        print("avg_school_count ", avg_school_count)

        # note: although we enforce max deviation of 1 from avg, in practice,
        # no two zones will have more than 1 difference in school count
        # Reason: school count is int. Observe the avg_school_count +-1,
        # if avg_school_count is not int, and see how the inequalities will look like
        # * I implemented the code this way (instead of pairwise comparison), since it is faster
        for z in range(self.dz.Z):
            zone_school_count[z] = gp.quicksum([self.dz.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]])
            self.m.addConstr(zone_school_count[z] <= avg_school_count + 1)
            self.m.addConstr(zone_school_count[z] >= avg_school_count - 1)

        # if K8 schools are included,
        # make sure no zone has more than one K8 schools
        if self.dz.include_k8:
            zone_k8_count = {}
            for z in range(self.dz.Z):
                zone_k8_count[z] = gp.quicksum([self.dz.area_data["K-8"][j] * self.x[j, z]
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
        eng_scores = self.dz.area_data["english_score"].fillna(value=0)
        school_average = sum(eng_scores) / sum(self.dz.schools)

        for z in range(self.dz.Z):
            zone_sum = gp.quicksum(
                [eng_scores[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            zone_schools = gp.quicksum(
                [self.dz.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            self.m.addConstr(zone_sum >= (1 - score_dev) * school_average * zone_schools)
            self.m.addConstr(zone_sum <= (1 + score_dev) * school_average * zone_schools)

    def _school_math_score_quality_const(self, score_dev=-1):
        if not (1 > score_dev > -1):
            return

        math_scores = self.dz.area_data["math_score"].fillna(value=0)
        school_average = sum(math_scores) / sum(self.dz.schools)

        for z in range(self.dz.Z):
            zone_sum = gp.quicksum(
                [math_scores[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            zone_schools = gp.quicksum(
                [self.dz.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
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
        color_scores = self.dz.area_data["AvgColorIndex"].fillna(value=0)
        school_average = sum(color_scores) / sum(self.dz.schools)

        for z in range(self.dz.Z):
            zone_sum = gp.quicksum(
                [color_scores[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )
            zone_schools = gp.quicksum(
                [self.dz.schools[j] * self.x[j, z] for j in self.valid_area_per_zone[z]]
            )

            self.m.addConstr(zone_sum >= (1 - score_dev) * school_average * zone_schools)
            self.m.addConstr(zone_sum <= (1 + score_dev) * school_average * zone_schools)

        if topX > 0:
            top_schools = np.zeros([self.dz.A])
            top = np.percentile(color_scores, 100 * (1 - self.Z / self.A) - 0.05)
            top = np.percentile(color_scores, topX)
            print(top)
            for j in range(self.dz.A):
                if color_scores[j] > top:
                    top_schools[j] = 1
            for z in range(self.dz.Z):
                topz = gp.quicksum(
                    [self.x[j, z] * top_schools[j] for j in self.valid_area_per_zone[z]]
                )
                self.m.addConstr(topz >= 0.8)

    def solve(self):
        self.m.update()  # Update the model
        print(f"Total number of variables: {self.m.numVars}")
        print(f"Total number of constraints: {self.m.numConstrs}")

        self.m.Params.TimeLimit = self.config['solve_time_limit']
        self.m.Params.OutputFlag = 0
        # optional: prevent writing a log file on macOS
        self.m.Params.LogFile = '/dev/null'
        if self.config['relative_gap_limit']> 0:
            self.m.Params.MIPGap = self.config['relative_gap_limit']
        self.m.setParam("Seed", self.config['random_seed'])
        if self.config['is_local']:
            self.m.setParam("Threads", 6)
        else:
            self.m.setParam("Threads", 16)
        if self.config['use_hints']:
            self._add_hints()
        self.m.optimize()



        status_map = {
            gp.GRB.OPTIMAL: "OPTIMAL",
            gp.GRB.INFEASIBLE: "INFEASIBLE",
            gp.GRB.UNBOUNDED: "UNBOUNDED",
            gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
            gp.GRB.INTERRUPTED: "INTERRUPTED",
            gp.GRB.TIME_LIMIT: "TIME_LIMIT",
            gp.GRB.CUTOFF: "CUTOFF",
            gp.GRB.NODE_LIMIT: "NODE_LIMIT",
            gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        }

        status_code = int(self.m.Status)
        status_name = status_map.get(status_code, f"STATUS_{status_code}")

        obj_value = None
        zone_dict = None
        if self.m.SolCount > 0:
            obj_value = self.m.ObjVal
            zone_dict = self._generate_zone_dict()

        return SolutionOutput(
            zone_dict=zone_dict,
            objective_value=obj_value,
            status=status_name,
            wall_time=self.m.Runtime,
            dz = self.dz
        )



    def add_constraints(self):
        self._feasibility_const(max_distance=self.config["max_distance"])
        self._shortage_const(shortage=self.config["shortage"], overage=self.config["overage"],
                             all_cap_shortage=self.config["all_cap_shortage"])

        self._contiguity_const()
        self._diversity_const(racial_dev=self.config["racial_dev"], frl_dev=self.config["frl_dev"])
        self._school_count_const()

    def add_objective(self):
        y_boundary = self.set_y_boundary()

        # set the objective of the Integer Program.
        # The integer program will try to minimize the cost of boundary,
        # which will result into compact and nice looking shapes for zones.
        self.m.setObjective(y_boundary, GRB.MINIMIZE)

    def fix_areas(self, zone_dict):
        if zone_dict is None:
            return
        for i in range(self.dz.A):
            area = self.dz.idx2area[i]
            if area in zone_dict:
                z = zone_dict[area]
                if (i, z) in self.x:
                    self.m.addConstr(self.x[i, z] == 1, 'Fix area to zone {}, {}'.format(area, z))

    def _add_hints(self):
        for i in range(self.dz.A):
            closest_centroid = None
            closest_distance = float('inf')
            for z in range(self.dz.Z):
                centroid_z = self.dz.centroids[z]
                dist = self.dz.euc_distances[centroid_z][i]
                if dist < closest_distance:
                    closest_distance = dist
                    closest_centroid = z
            if (i, closest_centroid) in self.x:
                self.x[i, closest_centroid].Start = 1

    def _generate_zone_dict(self):
        zone_dict = {}
        zone_lists = []
        for z in range(0, self.dz.Z):
            zone = []
            for j in range(0, self.dz.A):
                if j not in self.valid_area_per_zone[z]:
                    continue
                if self.x[j, z].X >= 0.999:
                    zone_dict[self.dz.idx2area[j]] = z
                    zone.append(self.dz.area_data[self.dz.level][j])
                    # add City wide school SF Montessori, even if we are not including city wide schools
                    # 823 is the aa level of SF Montessori school (which has school id 814)
                    if self.dz.idx2area[j] in [823, 60750132001]:
                        zone_dict[self.dz.idx2area[j]] = z
                        if self.dz.level == "attendance_area":
                            zone.append(SF_Montessori)
            if not zone == False:
                zone_lists.append(zone)
        temp_zone_dict = {}
        for idx, schools in enumerate(zone_lists):
            temp_zone_dict = {
                **temp_zone_dict,
                **{int(float(s)): idx for s in schools if s != ""},
            }
        # add K-8 schools to dict if using them
        if (self.dz.level == 'attendance_area') & (self.dz.include_k8):
            cw = self.dz.school_df.loc[self.dz.school_df["K-8"] == 1]
            for i, row in cw.iterrows():
                k8_schno = row["school_id"]
                z = temp_zone_dict[self.dz.sch2area[int(float(k8_schno))]]
                temp_zone_dict = {**temp_zone_dict, **{int(float(k8_schno)): z}}
                zone_lists[z].append(k8_schno)

        return temp_zone_dict