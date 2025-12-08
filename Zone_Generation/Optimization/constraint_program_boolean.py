import time
from dataclasses import dataclass
from typing import Optional

from ortools.sat.python import cp_model

from Zone_Generation.Config.Constants import AREA_ETHNICITIES, SCALING_CONST
from Zone_Generation.Optimization.optimizer import Optimizer, DesignZones, SolutionOutput


@dataclass(frozen=True)
class OptimizationConfig:
    sub_units: Optional
    max_distance: Optional[float]


class BooleanConstraintProgram(Optimizer):
    def __init__(self, dz: DesignZones, config):
        super().__init__(dz, config)
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
        return valid_area_per_zone, valid_zone_per_area, x

    def _feasibility_const(self):
        # Each area assigned to exactly one zone
        for i in range(self.dz.A):
            self.m.AddExactlyOne([self.x[z][i] for z in self.valid_zone_per_area[i]])

        # each centroid belong to its own zone
        for z in range(self.dz.Z):
            centroid_z = self.dz.centroids[z]
            self.m.Add(self.x[z][centroid_z] == 1)

    def _school_count_const(self):
        sub_units = self.config.get('sub_units')

        if sub_units is not None:
            avg_school_count = sum(
                [self.dz.schools[j] for j in range(self.dz.A) if self.dz.idx2area[j] in sub_units]) / self.dz.Z + 0.0001
        else:
            avg_school_count = sum([self.dz.schools[j] for j in range(self.dz.A)]) / self.dz.Z + 0.0001
        print("avg_school_count ", avg_school_count)

        school_ub = int(avg_school_count + 1)
        school_lb = int(avg_school_count)

        # note: although we enforce max deviation of 1 from avg, in practice,
        # no two zones will have more than 1 difference in school count
        # Reason: school count is int. Observe the avg_school_count +-1,
        # if avg_school_count is not int, and see how the inequalities will look like
        # * I implemented the code this way (instead of pairwise comparison), since it is faster
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
                # self.m.Add(zone_k8_count[z] <= 1)
                self.m.AddAtMostOne(zone_k8_count)

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
                    if len(neighbors) > 0:
                        # self.m.Add(self.x[z][j] <= sum(neighbors))
                        self.m.AddBoolOr(neighbors).OnlyEnforceIf(self.x[z][j])
                else:
                    any_neighbors = [
                        self.x[z][k]
                        for k in self.dz.neighbors[j] if k in self.valid_area_per_zone[z]
                    ]

                    # self.m.Add(self.x[z][j] <= sum(any_neighbors))
                    if len(any_neighbors) > 0:
                        self.m.AddBoolOr(any_neighbors).OnlyEnforceIf(self.x[z][j])

    def _racial_const(self):
        for race in AREA_ETHNICITIES:
            race_dev = int(SCALING_CONST * self.config['racial_dev'])
            r = int(SCALING_CONST * self.dz.R[race])

            for z in range(self.dz.Z):
                vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]
                race_coefs = [int(SCALING_CONST * self.dz.area_data[race][j]) for j in self.valid_area_per_zone[z]]
                pop_coefs = [int(SCALING_CONST * self.dz.area_data["ge_students"][j]) for j in
                             self.valid_area_per_zone[z]]

                race_sum = cp_model.LinearExpr.WeightedSum(vars_list, race_coefs)
                pop_students = cp_model.LinearExpr.WeightedSum(vars_list, pop_coefs)

                self.m.Add(race_sum * SCALING_CONST >= (r - race_dev) * pop_students)
                self.m.Add(race_sum * SCALING_CONST <= (r + race_dev) * pop_students)

    # Make sure students of low socioeconomic status groups are fairly distributed among zones.
    # Our only metric to measure socioeconomic status, is FRL, which is the students eligibility for
    # Free or Reduced Price Lunch.
    # make sure the total FRL for students in each zone, is within an additive
    #  frl_dev% of average FRL over zones..
    def _frl_const(self):


        frl_dev = int(SCALING_CONST * self.config['frl_dev'])
        f = int(SCALING_CONST * self.dz.F)
        for z in range(self.dz.Z):
            vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]
            frl_coefs = [int(SCALING_CONST * self.dz.area_data['FRL'][j]) for j in self.valid_area_per_zone[z]]
            pop_coefs = [int(SCALING_CONST * self.dz.area_data["ge_students"][j]) for j in
                         self.valid_area_per_zone[z]]

            race_sum = cp_model.LinearExpr.WeightedSum(vars_list, frl_coefs)
            pop_students = cp_model.LinearExpr.WeightedSum(vars_list, pop_coefs)

            self.m.Add(race_sum * SCALING_CONST >= (f - frl_dev) * pop_students)
            self.m.Add(race_sum * SCALING_CONST <= (f + frl_dev) * pop_students)

    def _proportional_shortage_const(self):
        # No zone has shortage more than shortage percentage of its population
        for z in range(self.dz.Z):
            min_seats = sum(
                [int(SCALING_CONST * (1 - self.config['shortage']) * self.dz.seats[j]) * self.x[z][j]
                 for j in self.valid_area_per_zone[z]]
            )
            total_students = sum(
                [int(SCALING_CONST * self.dz.area_data["ge_students"][j]) * self.x[z][j]
                 for j in self.valid_area_per_zone[z]]
            )
            self.m.Add(
                total_students >= min_seats
            )

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_const(self):
        # No zone has overage more than overage percentage of its population
        for z in range(self.dz.Z):
            max_seats = sum(
                [int(SCALING_CONST * (1 + self.config['overage']) * self.dz.seats[j]) * self.x[z][j]
                 for j in self.valid_area_per_zone[z]]
            )
            total_students = sum(
                [int(SCALING_CONST * self.dz.area_data["ge_students"][j]) * self.x[z][j]
                 for j in self.valid_area_per_zone[z]]
            )
            self.m.Add(
                total_students <= max_seats
            )

    def add_constraints(self):
        self._feasibility_const()
        self._school_count_const()
        self._contiguity_const()
        self._racial_const()
        self._frl_const()
        self._proportional_shortage_const()
        self._proportional_overage_const()

    def _add_hints(self):
        # add hint that each area will be assigned to the closest centroid
        for i in range(self.dz.A):
            closest_centroid = None
            closest_distance = float('inf')
            for z in range(self.dz.Z):
                centroid_z = self.dz.centroids[z]
                dist = self.dz.euc_distances[centroid_z][i]
                if dist < closest_distance:
                    closest_distance = dist
                    closest_centroid = z
            if closest_centroid in self.valid_zone_per_area[i]:
                self.m.AddHint(self.x[closest_centroid][i], 1)

    def _add_search_strategy(self):
        # define search strategy
        # for each zone, try to assign areas closer to centroid first
        areas_by_constraint = sorted(
            range(self.dz.A),
            key=lambda i: len(self.valid_zone_per_area[i])
        )

        vars_ordered = []
        for i in areas_by_constraint:
            # For each area, order zones by distance
            zones_sorted = sorted(
                self.valid_zone_per_area[i],
                key=lambda z: self.dz.euc_distances[self.dz.centroids[z]][i]
            )
            vars_ordered.extend(self.x[z][i] for z in zones_sorted)

        self.m.AddDecisionStrategy(
            vars_ordered,
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE
        )



    # def add_objective_old(self):
    #     self._add_hints()
    #     # self._add_search_strategy()
    #     boundary_vars = []
    #
    #     for zone in range(self.dz.Z):
    #         for i in self.valid_area_per_zone[zone]:
    #             for j in self.dz.neighbors[i]:
    #                 if i >= j:
    #                     continue
    #                 if j not in self.valid_area_per_zone[zone]:
    #                     # always going to be assigned to a different zone so add own zone
    #                     boundary_vars.append(self.x[zone][i])
    #                     continue
    #                 b = self.m.NewBoolVar(f"boundary_{i}_{j}")
    #                 self.m.Add(self.x[zone][i] != self.x[zone][j]).OnlyEnforceIf(b)
    #                 self.m.Add(self.x[zone][i] == self.x[zone][j]).OnlyEnforceIf(b.Not())
    #                 boundary_vars.append(b)
    #
    #
    #     self.m.Minimize(sum(boundary_vars))

    def add_objective(self):
        self._add_hints()
        boundary_vars = []

        for zone in range(self.dz.Z):
            for i in self.valid_area_per_zone[zone]:
                for j in self.dz.neighbors[i]:
                    # 1. Enforce undirected edge check to avoid double processing
                    if i >= j:
                        continue

                    # 2. Case: Neighbor j is not in this zone's valid area
                    # The boundary exists solely if i is selected for this zone.
                    if j not in self.valid_area_per_zone[zone]:
                        boundary_vars.append(self.x[zone][i])
                        continue

                    # 3. Case: Both i and j are valid candidates for this zone
                    # We want b = |x[i] - x[j]|.
                    # Instead of expensive logic, we use linear inequalities.
                    b = self.m.NewBoolVar(f"boundary_{zone}_{i}_{j}")

                    # Constraint: b >= x[i] - x[j]
                    self.m.Add(b >= self.x[zone][i] - self.x[zone][j])

                    # Constraint: b >= x[j] - x[i]
                    self.m.Add(b >= self.x[zone][j] - self.x[zone][i])

                    boundary_vars.append(b)

        self.m.Minimize(sum(boundary_vars))

    # def add_objective(self):
    #     self._add_hints()
    #     boundary_terms = []
    #
    #     processed_edges = set()
    #     for i in range(self.dz.A):
    #         for j in self.dz.neighbors[i]:
    #             if i >= j or (i, j) in processed_edges:
    #                 continue
    #             processed_edges.add((i, j))
    #
    #             # Find common zones
    #             common_zones = set(self.valid_zone_per_area[i]) & set(self.valid_zone_per_area[j])
    #             if not common_zones:
    #                 boundary_terms.append(1)  # Always a boundary
    #             else:
    #                 match_vars = []
    #
    #                 # 2. Create "Match" variables for each potential zone
    #                 # Logic: match_z implies (i is in z AND j is in z)
    #                 for z in common_zones:
    #                     match_z = self.m.NewBoolVar(f"match_{z}_{i}_{j}")
    #
    #                     # Replaces: match_z = x[z][i] * x[z][j]
    #                     # We only enforce: match_z => (x[i] AND x[j])
    #                     # Because we maximize match_z (via minimizing cost), this is sufficient.
    #                     self.m.AddBoolAnd([self.x[z][i], self.x[z][j]]).OnlyEnforceIf(match_z)
    #
    #                     match_vars.append(match_z)
    #
    #                 # 3. Create "Same Zone" variable
    #                 # Logic: same implies (match_z1 OR match_z2 OR ...)
    #                 same_zone = self.m.NewBoolVar(f"same_{i}_{j}")
    #
    #                 # Replaces: sum(match_vars) >= 1
    #                 # Enforce: same_zone => (match_z1 OR match_z2 ...)
    #                 self.m.AddBoolOr(match_vars).OnlyEnforceIf(same_zone)
    #
    #                 # 4. Add to objective (Cost = NOT same)
    #                 boundary_terms.append(same_zone.Not())
    #     boundary_sum = cp_model.LinearExpr.Sum(boundary_terms)
    #     self.m.Minimize(boundary_sum)

    def solve(self):

        class StalledSearchCallback(cp_model.CpSolverSolutionCallback):
            """Stops the solver if no improvement in the objective is found for a set duration."""

            def __init__(self, stall_limit_seconds: float):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.__stall_limit = stall_limit_seconds
                self.__start_time = time.time()
                self.__last_best_time = self.__start_time
                self.__best_objective = float('inf')  # Assuming minimization

                # For logging/display (optional)
                print(f"Solver started with a stall limit of {self.__stall_limit} seconds.")

            def on_solution_callback(self):
                # The objective value for the current solution
                current_objective = self.ObjectiveValue()
                current_time = time.time()

                # Check for improvement (assuming minimization)
                if current_objective < self.__best_objective:
                    self.__best_objective = current_objective
                    self.__last_best_time = current_time
                    # Optional: Log the improvement
                    # print(f"New best objective: {current_objective} at {current_time - self.__start_time:.2f}s")
                else:
                    # Check if the search has stalled
                    time_since_last_improvement = current_time - self.__last_best_time

                    if time_since_last_improvement >= self.__stall_limit:
                        print(f"\nSearch stalled! No improvement in {self.__stall_limit}s.")
                        print(f"Stopping search. Best objective found: {self.__best_objective}")
                        self.StopSearch()  # This is the crucial line to stop the solver

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = self.config['solve_time_limit']
        presolve_iterations = 0
        gap_limit = 0
        if self.dz.level != 'attendance_area':
            presolve_iterations = 20
            gap_limit = 0.15
        solver.parameters.max_presolve_iterations = presolve_iterations
        solver.parameters.relative_gap_limit = gap_limit
        solver.parameters.random_seed = 42
        solver.parameters.num_search_workers = 6
        solver.parameters.log_search_progress = True
        #important to think about this parameter and thourhgly test later. for now leave at 1
        solver.parameters.linearization_level = 0
        solver.parameters.symmetry_level = 2

        solution_callback = StalledSearchCallback(30)

        status = solver.Solve(self.m, solution_callback)

        objective_value = solver.ObjectiveValue()
        best_bound = solver.BestObjectiveBound()

        # 2. Calculate the Relative Gap
        absolute_gap = abs(objective_value - best_bound)

        # Use the denominator max(1, abs(objective_value)) to avoid division by zero
        # and to handle cases where the objective is close to zero.
        relative_gap = absolute_gap / max(1.0, abs(objective_value))

        user_time = solver.UserTime()

        print("-" * 30)
        print(f"**Objective Value (Best Solution):** {objective_value:,.2f}")
        print(f"**Best Bound Found (Theoretical Limit):** {best_bound:,.2f}")
        print(f"**Absolute Gap:** {absolute_gap:,.2f}")
        print(f"**Relative Gap:** ({relative_gap * 100:.2f}%)")
        print(f"**Status:** {solver.StatusName(status)}")
        print(f"**User Time:** {user_time:.2f} seconds")
        print("-" * 30)

        return SolutionOutput(self._generate_zone_dict(solver), objective_value,
                              solver.StatusName(status), user_time, self.dz)

    def fix_areas(self, fixed_zone_dict):
        if fixed_zone_dict is None:
            return
        fixed_areas = 0
        for area, zone in fixed_zone_dict.items():
            area_idx = self.dz.area2idx[area]
            if area_idx in self.valid_zone_per_area:
                if zone in self.valid_zone_per_area[area_idx]:
                    self.m.Add(self.x[zone][area_idx] == 1)
                    fixed_areas += 1
        print(f"Fixed areas: {fixed_areas}")

    def _generate_zone_dict(self, solver):
        zone_dict = {}
        for i in range(self.dz.A):
            for z in self.valid_zone_per_area[i]:
                if solver.BooleanValue(self.x[z][i]) == 1:
                    zone_dict[self.dz.idx2area[i]] = z
                    break
        return zone_dict
