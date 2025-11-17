from dataclasses import dataclass
from typing import Optional

from ortools.sat.python import cp_model

from Zone_Generation.Config.Constants import ETHNICITY_COLS, AREA_ETHNICITIES, SCALING_CONST
from Zone_Generation.Optimization.optimizer import Optimizer, DesignZones


@dataclass(frozen=True)
class OptimizationConfig:
    sub_units: Optional
    max_distance: Optional[float]


class BooleanConstraintProgram(Optimizer):
    def __init__(self, Area_Data: DesignZones, optimization_config: OptimizationConfig):
        super().__init__(Area_Data, optimization_config)
        self.m = cp_model.CpModel()
        self.valid_area_per_zone, self.valid_zone_per_area, self.x = self.add_variables()

    def add_variables(self):
        max_distance = self.config.get('max_distance')

        sub_units = self.config.get('sub_units')
        if max_distance is None:
            max_distance = float('inf')
        valid_area_per_zone = {}
        valid_zone_per_area = {}
        x = {}
        for z in range(self.Z):
            valid_area_per_zone[z] = []
            x[z] = {}
        for i in range(self.A):
            valid_zone_per_area[i] = []

        for z in range(self.Z):
            centroid_z = self.centroids[z]
            for i in range(self.A):
                if sub_units is not None:
                    if self.idx2area[i] not in sub_units:
                        continue
                if self.euc_distances[centroid_z][i] < max_distance:
                    var = self.m.NewBoolVar(f"zone_{centroid_z}_area_{self.idx2area[i]}")
                    valid_area_per_zone[z].append(i)
                    valid_zone_per_area[i].append(z)
                    x[z][i] = var
        return valid_area_per_zone, valid_zone_per_area, x

    def _feasibility_const(self):
        # Each area assigned to exactly one zone
        for i in range(self.A):
            self.m.Add(sum([self.x[z][i] for z in self.valid_zone_per_area[i]]) == 1)

        # each centroid belong to its own zone
        for z in range(self.Z):
            centroid_z = self.centroids[z]
            self.m.Add(self.x[z][centroid_z] == 1)

    def _school_count_const(self):
        zone_school_count = {}
        sub_units = self.config.get('sub_units')

        if sub_units is not None:
            avg_school_count = sum(
                [self.schools[j] for j in range(self.A) if self.idx2area[j] in sub_units]) / self.Z + 0.0001
        else:
            avg_school_count = sum([self.schools[j] for j in range(self.A)]) / self.Z + 0.0001
        print("avg_school_count ", avg_school_count)

        school_ub = int(avg_school_count + 1)
        school_lb = int(avg_school_count - 1)

        # note: although we enforce max deviation of 1 from avg, in practice,
        # no two zones will have more than 1 difference in school count
        # Reason: school count is int. Observe the avg_school_count +-1,
        # if avg_school_count is not int, and see how the inequalities will look like
        # * I implemented the code this way (instead of pairwise comparison), since it is faster
        for z in range(self.Z):
            school_coefs = []
            school_vars = []
            for j in self.valid_area_per_zone[z]:
                school_coefs.append(int(self.schools[j]))
                school_vars.append(self.x[z][j])
            zone_school_count[z] = cp_model.LinearExpr.WeightedSum(school_vars, school_coefs)
            self.m.Add(zone_school_count[z] <= school_ub)
            self.m.Add(zone_school_count[z] >= school_lb)

        # if K8 schools are included,
        # make sure no zone has more than one K8 schools
        if self.include_k8:
            zone_k8_count = {}
            for z in range(self.Z):
                zone_k8_count[z] = sum([self.area_data["K-8"][j] * self.x[z][j] for j in self.valid_area_per_zone[z]])
                self.m.Add(zone_k8_count[z] <= 1)

    def _contiguity_const(self):

        # (x[j,z] (and indicator that unit j is assigned to zone z)) \leq
        # (sum of all x[j',z] where j' is in self.closer_neighbors_per_centroid[area,c] where c is centroid for z)
        sub_units = self.config.get('sub_units')

        for j in range(self.A):
            if sub_units is not None:
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
                    neighbor_sum = sum(
                        self.x[z][k]
                        for k in self.closer_euc_neighbors[j, self.centroids[z]]
                        if k in self.valid_area_per_zone[z]
                    )
                    self.m.Add(self.x[z][j] <= neighbor_sum)
                else:
                    any_neighbor_sum = sum(
                        [
                            self.x[z][k]
                            for k in self.neighbors[j] if k in self.valid_area_per_zone[z]
                        ]
                    )
                    self.m.Add(self.x[z][j] <= any_neighbor_sum)

    def _racial_const(self):
        for race in AREA_ETHNICITIES:
            race_dev = int(100 * self.config['racial_dev'])
            race_ratio = int(100 * sum(self.area_data[race]) / float(self.N))

            for z in range(self.Z):
                zone_sum = sum(
                    [int(10000 * self.area_data[race][j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
                )
                district_students = sum(
                    [int(100 * self.studentsInArea[j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
                )
                self.m.Add(zone_sum >= (race_ratio - race_dev) * district_students)
                self.m.Add(zone_sum <= (race_ratio + race_dev) * district_students)

    # Make sure students of low socioeconomic status groups are fairly distributed among zones.
    # Our only metric to measure socioeconomic status, is FRL, which is the students eligibility for
    # Free or Reduced Price Lunch.
    # make sure the total FRL for students in each zone, is within an additive
    #  frl_dev% of average FRL over zones..
    def _frl_const(self):
        frl_dev = int(SCALING_CONST * self.config['frl_dev'])
        f = int(SCALING_CONST * self.F)
        for z in range(self.Z):
            zone_sum = sum(
                [int(SCALING_CONST**2 * self.area_data["FRL"][j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
            )
            district_students = sum(
                [int(SCALING_CONST *self.studentsInArea[j]) * self.x[z][j] for j in self.valid_area_per_zone[z]]
            )

            self.m.Add(zone_sum >= (f - frl_dev) * district_students)
            self.m.Add(zone_sum <= (f + frl_dev) * district_students)

    def _proportional_shortage_const(self):
        # No zone has shortage more than shortage percentage of its population
        shortage = int(SCALING_CONST * self.config['shortage'])
        for z in range(self.Z):
            self.m.Add(
                (SCALING_CONST - shortage) *
                sum(
                    [int(SCALING_CONST * self.studentsInArea[j]) * self.x[z][j]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                sum(
                    [int(SCALING_CONST**2 *self.seats[j]) * self.x[z][j]
                     for j in self.valid_area_per_zone[z]]
                )
            )

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_const(self):
        # No zone has overage more than overage percentage of its population
        overage = int(SCALING_CONST * self.config['overage'])
        for z in range(self.Z):
            self.m.Add(
                sum(
                    [(-int(SCALING_CONST**2 * self.studentsInArea[j]) + self.seats[j]) * self.x[z][j]
                     for j in self.valid_area_per_zone[z]]
                )
                <=
                overage *
                sum(
                    [int(SCALING_CONST *self.studentsInArea[j]) * self.x[z][j]
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
        #all cap shortage

    def add_objective(self):
        boundary_vars = []
        for zone in range(self.Z):
            for i in self.valid_area_per_zone[zone]:
                for j in self.neighbors[i]:
                    if i >= j:
                        continue
                    if j not in self.valid_area_per_zone[zone]:
                        # always going to be assigned to a different zone, so just add 1 to the boundary cost
                        boundary_vars.append(1)
                        continue
                    b = self.m.NewBoolVar(f"boundary_{i}_{j}")
                    self.m.AddAbsEquality(b, self.x[zone][i] - self.x[zone][j])
                    boundary_vars.append(b)
        self.m.Minimize(sum(boundary_vars))

    def solve(self):
        solver = cp_model.CpSolver()
        status = solver.Solve(self.m)
        solver.parameters.max_time_in_seconds = 60
        solver.parameters.num_search_workers = 5

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Solution found with objective value {solver.ObjectiveValue()}")
            zone_dict = {}
            for i in range(self.A):
                for z in self.valid_zone_per_area[i]:
                    if solver.BooleanValue(self.x[z][i]) == 1:
                        zone_dict[self.idx2area[i]] = z
                        break
            return zone_dict
        else:
            print("No solution found.")
            return None
