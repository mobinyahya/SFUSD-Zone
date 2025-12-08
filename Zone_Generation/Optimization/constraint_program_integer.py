from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import Domain

from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.Optimization.optimizer import Optimizer, DesignZones, SolutionOutput


class IntegerConstraintProgram(Optimizer):
    def __init__(self, dz: DesignZones, config):
        super().__init__(dz, config)
        self.m = cp_model.CpModel()
        self.valid_area_per_zone, self.valid_zone_per_area, self.x, self.y = self.add_variables()

    def add_variables(self):
        max_distance = self.config.get('max_distance')

        sub_units = self.config.get('sub_units')
        if max_distance is None:
            max_distance = float('inf')
        valid_area_per_zone = {}
        valid_zone_per_area = {}
        x = {}
        y = []

        for z in range(self.dz.Z):
            valid_area_per_zone[z] = []
            x[z] = {}
        for i in range(self.dz.A):
            valid_zone_per_area[i] = []

        for z in range(self.dz.Z):
            centroid_z = self.dz.centroids[z]
            for i in range(self.dz.A):
                if sub_units is not None:
                    if self.dz.idx2area[i] not in sub_units:
                        continue
                if self.dz.euc_distances[centroid_z][i] < max_distance:
                    var = self.m.NewBoolVar(f"zone_{centroid_z}_area_{self.dz.idx2area[i]}")
                    valid_area_per_zone[z].append(i)
                    valid_zone_per_area[i].append(z)
                    x[z][i] = var

        for i in range(self.dz.A):
            var = self.m.NewIntVarFromDomain(Domain.FromValues(valid_zone_per_area[i]),
                                             f"area_{self.dz.idx2area[i]}_zone_idx")
            for z in valid_zone_per_area[i]:
                self.m.Add(var == z).OnlyEnforceIf(x[z][i])
                self.m.Add(var != z).OnlyEnforceIf(x[z][i].Not())

            y.append(var)

        return valid_area_per_zone, valid_zone_per_area, x, y

    def _feasibility_const(self):
        # each centroid belong to its own zone
        for z in range(self.dz.Z):
            centroid_z = self.dz.centroids[z]
            self.m.Add(self.y[centroid_z] == z)

    def _school_count_const(self):
        sub_units = self.config.get('sub_units')

        if sub_units is not None:
            avg_school_count = sum(
                [self.dz.schools[j] for j in range(self.dz.A) if self.dz.idx2area[j] in sub_units]) / self.dz.Z
        else:
            avg_school_count = sum([self.dz.schools[j] for j in range(self.dz.A)]) / self.dz.Z
        print("avg_school_count ", avg_school_count)

        school_ub = int(avg_school_count + 1)
        school_lb = int(avg_school_count)

        for z in range(self.dz.Z):
            school_coefs = []
            school_vars = []
            for j in self.valid_area_per_zone[z]:
                school_coefs.append(int(self.dz.schools[j]))
                school_vars.append(self.x[z][j])
            zone_school_count = cp_model.LinearExpr.WeightedSum(school_vars, school_coefs)
            # self.m.Add(zone_school_count[z] <= school_ub)
            # self.m.Add(zone_school_count[z] >= school_lb)
            self.m.AddLinearConstraint(zone_school_count, school_lb, school_ub)

        # if K8 schools are included,
        # make sure no zone has more than one K8 schools
        if self.dz.include_k8:
            for z in range(self.dz.Z):
                zone_k8_count = [self.dz.area_data["K-8"][j] * self.x[z][j] for j in self.valid_area_per_zone[z]]
                self.m.AddAtMostOne(zone_k8_count)

    def _school_count_const_primes(self):
        pass

    def _contiguity_const(self):
        # (x[j,z] (and indicator that unit j is assigned to zone z)) \leq
        # (sum of all x[j',z] where j' is in self.closer_neighbors_per_centroid[area,c] where c is centroid for z)
        sub_units = self.config.get('sub_units')

        for j in range(self.dz.A):
            if sub_units is not None:
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
                    neighbors = [
                        self.x[z][k]
                        for k in self.dz.closer_euc_neighbors[j, self.dz.centroids[z]]
                        if k in self.valid_area_per_zone[z]
                    ]
                    self.m.AddBoolOr(neighbors).OnlyEnforceIf(self.x[z][j])
                else:
                    neighbors = [
                        self.x[z][k]
                        for k in self.dz.neighbors[j] if k in self.valid_area_per_zone[z]
                    ]
                    self.m.AddBoolOr(neighbors).OnlyEnforceIf(self.x[z][j])

    def _contiguity_const_primes(self):
        pass
        # do the same thing except just ensure that the product of neighbors mod the current value is 0

    def _racial_const(self):
        for race in AREA_ETHNICITIES:
            race_dev = int(100 * self.config['racial_dev'])
            race_ratio = int(100 * sum(self.dz.area_data[race]) / float(self.dz.N))

            for z in range(self.dz.Z):
                zone_sum = sum(
                    [int(10000 * self.dz.area_data[race][j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
                )
                district_students = sum(
                    [int(100 * self.dz.studentsInArea[j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
                )
                self.m.Add(zone_sum >= (race_ratio - race_dev) * district_students)
                self.m.Add(zone_sum <= (race_ratio + race_dev) * district_students)

    # Make sure students of low socioeconomic status groups are fairly distributed among zones.
    # Our only metric to measure socioeconomic status, is FRL, which is the students eligibility for
    # Free or Reduced Price Lunch.
    # make sure the total FRL for students in each zone, is within an additive
    #  frl_dev% of average FRL over zones..
    def _frl_const(self):
        frl_dev = int(100 * self.config['frl_dev'])
        f = int(100 * self.dz.F)
        for z in range(self.dz.Z):
            zone_sum = sum(
                [int(10000 * self.dz.area_data["FRL"][j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
            )
            district_students = sum(
                [int(100 * self.dz.studentsInArea[j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
            )

            self.m.Add(zone_sum >= (f - frl_dev) * district_students)
            self.m.Add(zone_sum <= (f + frl_dev) * district_students)

    def _proportional_shortage_const(self):
        # No zone has shortage more than shortage percentage of its population
        shortage = int(100 * self.config['shortage'])
        for z in range(self.dz.Z):
            self.m.Add(
                (100 - shortage) *
                sum(
                    [int(100 * self.dz.studentsInArea[j]) * self.x[z][j]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                sum(
                    [int(10000 * self.dz.seats[j]) * self.x[z][j]
                     for j in self.valid_area_per_zone[z]]
                )
            )

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_const(self):
        # No zone has overage more than overage percentage of its population
        overage = int(100 * self.config['overage'])
        for z in range(self.dz.Z):
            self.m.Add(
                sum(
                    [(-int(10000 * self.dz.studentsInArea[j]) + self.dz.seats[j]) * self.x[z][j]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                overage *
                sum(
                    [int(100 * self.dz.studentsInArea[j]) * self.x[z][j]
                     for j in self.valid_area_per_zone[z]]
                )
            )

    def add_constraints(self):
        self._feasibility_const()
        self._school_count_const()
        self._contiguity_const()
        self._racial_const()
        self._frl_const()
        self._proportional_shortage_const()
        self._proportional_overage_const()

    def add_objective(self):
        boundary_vars = []
        for area in range(self.dz.A):
            for neighbor in self.dz.neighbors[area]:
                if area < neighbor:
                    boundary_var = self.m.NewBoolVar(
                        f"boundary_area_{self.dz.idx2area[area]}_neighbor_{self.dz.idx2area[neighbor]}")
                    boundary_vars.append(boundary_var)
                    # if area and neighbor are assigned to different zones, then boundary_var = 1
                    self.m.Add(self.y[area] != self.y[neighbor]).OnlyEnforceIf(boundary_var)
                    self.m.Add(self.y[area] == self.y[neighbor]).OnlyEnforceIf(boundary_var.Not())

        self.m.Minimize(sum(boundary_vars))

    def solve(self):
        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = self.config['solve_time_limit']
        solver.parameters.num_search_workers = 5

        status = solver.Solve(self.m)

        return SolutionOutput(self._generate_zone_dict(solver), solver.ObjectiveValue(), solver.StatusName(status),
                              solver.UserTime(), self.dz)

    def _generate_zone_dict(self, solver):
        zone_dict = {}
        for i in range(self.dz.A):
            zone_dict[self.dz.idx2area[i]] = solver.Value(self.y[i])
        return zone_dict

    def fix_areas(self, fixed_zone_dict):
        if fixed_zone_dict is None:
            return
        for area, zone in fixed_zone_dict.items():
            area_idx = self.dz.area2idx[area]
            self.m.Add(self.y[area_idx] == zone)
