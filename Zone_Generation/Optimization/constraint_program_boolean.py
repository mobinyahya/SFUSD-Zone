from ortools.sat.python import cp_model

from Zone_Generation.Config.Constants import AREA_ETHNICITIES, SCALING_CONST
from Zone_Generation.Optimization.optimizer import Optimizer, DesignZones, SolutionOutput


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
            self.m.AddLinearConstraint(zone_school_count, school_lb, school_ub)

        # if K8 schools are included,
        # make sure no zone has more than one K8 schools
        if self.dz.include_k8:
            for z in range(self.dz.Z):
                zone_k8_count = [self.dz.area_data["K-8"][j] * self.x[z][j] for j in self.valid_area_per_zone[z]]
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
                    # ensure that all neighbors of centroid are also assigned to the same zone
                    for neighbor in self.dz.neighbors[j]:
                        if neighbor not in self.valid_area_per_zone[z]:
                            continue
                        self.m.Add(self.x[z][neighbor] == 1)
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

                    # TODO: Deal with these stupid edge cases

    def _racial_const(self):
        for race_col in AREA_ETHNICITIES:
            race_dev = self.config['racial_dev']
            r = self.dz.R[race_col]

            for z in range(self.dz.Z):
                lb_coefs = []
                ub_coefs = []

                for j in self.valid_area_per_zone[z]:
                    students = self.dz.area_data["ge_students"][j]
                    race = self.dz.area_data[race_col][j]

                    # Lower Bound Coefficient: (FRL * SCALE) - (LowerBoundRatio * Students)
                    # Note: (f - frl_dev) already contains one factor of SCALING_CONST
                    lb_c = (race - (r - race_dev) * students) * SCALING_CONST
                    lb_coefs.append(int(lb_c))

                    # Upper Bound Coefficient: (FRL * SCALE) - (UpperBoundRatio * Students)
                    ub_c = (race - (r + race_dev) * students) * SCALING_CONST
                    ub_coefs.append(int(ub_c))

                vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]

                # The entire ratio logic is now contained within these weighted sums
                self.m.Add(cp_model.LinearExpr.WeightedSum(vars_list, lb_coefs) >= 0)
                self.m.Add(cp_model.LinearExpr.WeightedSum(vars_list, ub_coefs) <= 0)

    # Make sure students of low socioeconomic status groups are fairly distributed among zones.
    # Our only metric to measure socioeconomic status, is FRL, which is the students eligibility for
    # Free or Reduced Price Lunch.
    # make sure the total FRL for students in each zone, is within an additive
    #  frl_dev% of average FRL over zones..
    def _frl_const(self):
        frl_dev = self.config['frl_dev']
        f = self.dz.F
        for z in range(self.dz.Z):
            lb_coefs = []
            ub_coefs = []

            for j in self.valid_area_per_zone[z]:
                students = self.dz.area_data["ge_students"][j]
                frl = self.dz.area_data['FRL'][j]

                # Lower Bound Coefficient: (FRL * SCALE) - (LowerBoundRatio * Students)
                # Note: (f - frl_dev) already contains one factor of SCALING_CONST
                lb_c = (frl - (f - frl_dev) * students) * SCALING_CONST
                lb_coefs.append(int(lb_c))

                # Upper Bound Coefficient: (FRL * SCALE) - (UpperBoundRatio * Students)
                ub_c = (frl - (f + frl_dev) * students) * SCALING_CONST
                ub_coefs.append(int(ub_c))

            vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]

            # The entire ratio logic is now contained within these weighted sums
            self.m.Add(cp_model.LinearExpr.WeightedSum(vars_list, lb_coefs) >= 0)
            self.m.Add(cp_model.LinearExpr.WeightedSum(vars_list, ub_coefs) <= 0)

    def _proportional_shortage_const(self):
        # No zone has shortage more than shortage percentage of its population
        for z in range(self.dz.Z):
            coefs = []
            for j in self.valid_area_per_zone[z]:
                lb_seats = (1 - self.config['shortage']) * self.dz.seats[j]
                students = self.dz.area_data["ge_students"][j]
                coef = int(SCALING_CONST * (students - lb_seats))
                coefs.append(coef)
            vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]
            total_expr = cp_model.LinearExpr.WeightedSum(vars_list, coefs)
            self.m.Add(total_expr >= 0)

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_const(self):
        # No zone has overage more than overage percentage of its population
        for z in range(self.dz.Z):
            coefs = []
            for j in self.valid_area_per_zone[z]:
                ub_seats = (1 + self.config['overage']) * self.dz.seats[j]
                students = self.dz.area_data["ge_students"][j]
                coef = int(SCALING_CONST * (students - ub_seats))
                coefs.append(coef)
            vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]
            total_expr = cp_model.LinearExpr.WeightedSum(vars_list, coefs)
            self.m.Add(total_expr <= 0)

    def add_constraints(self):
        self._school_count_const()
        self._feasibility_const()
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

    def add_objective(self):
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

    def solve(self):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config['solve_time_limit']
        solver.parameters.max_presolve_iterations = 10
        # if self.config['relative_gap_limit'] > 0:
        #     solver.parameters.relative_gap_limit = self.config['relative_gap_limit']
        if self.config['level'] == 'Block':
            solver.parameters.relative_gap_limit = 0.1
        elif self.config['level'] == 'BlockGroup':
            solver.parameters.relative_gap_limit = 0.3
        else:
            solver.parameters.relative_gap_limit = 0
        solver.parameters.random_seed = self.config['random_seed']
        if self.config['is_local']:
            solver.parameters.num_search_workers = 6
        else:
            solver.parameters.num_search_workers = 16
        if self.config['use_hints']:
            self._add_hints()

        solver.parameters.log_search_progress = True
        # important to think about this parameter and thourhgly test later. for now leave at 1
        solver.parameters.linearization_level = 2
        solver.parameters.symmetry_level = 4

        log_file_path = f"{self.config['level']}.txt"

        class FileLogger(cp_model.CpSolverSolutionCallback):
            def __init__(self, log_file):
                super().__init__()
                self.log_file = log_file
                with open(self.log_file, "w") as f:
                    f.write("")

            def OnLogMessage(self, message: str):
                with open(self.log_file, "a") as f:
                    f.write(message + "\n")

        # Assign the callback and solve
        logger = FileLogger(log_file_path)
        solver.log_callback = logger.OnLogMessage
        status = solver.Solve(self.m)

        objective_value = solver.ObjectiveValue()
        wall_time = solver.WallTime()

        return SolutionOutput(self._generate_zone_dict(solver), objective_value,
                              solver.StatusName(status), wall_time, self.dz)

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
