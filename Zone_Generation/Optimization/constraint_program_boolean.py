from ortools.sat.python import cp_model

from Zone_Generation.Config.Constants import AREA_ETHNICITIES, SCALING_CONST
from Zone_Generation.Optimization.optimizer import Optimizer, SolutionOutput


class BooleanConstraintProgram(Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.m = cp_model.CpModel()

        self.valid_area_per_zone, self.valid_zone_per_area, self.x = self.add_variables()

    def add_variables(self):
        max_distance = self.config.get('max_distance')

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
                # ignore if far away from centroid
                if self.G.graph['distance_dict'][centroid_z][i] > max_distance:
                    continue

                # ignore if centroid of a different zone
                if i in self.centroids and i != centroid_z:
                    continue

                var = self.m.NewBoolVar(f"zone_{centroid_z}_area_{i}")
                valid_area_per_zone[z].append(i)
                valid_zone_per_area[i].append(z)
                x[z][i] = var
        return valid_area_per_zone, valid_zone_per_area, x

    def _feasibility_const(self):
        # Each area assigned to exactly one zone
        for i in range(self.A):
            self.m.AddExactlyOne([self.x[z][i] for z in self.valid_zone_per_area[i]])

        # each centroid belong to its own zone, as well as its neighbors
        for z in range(self.Z):
            centroid_z = self.centroids[z]
            self.m.Add(self.x[z][centroid_z] == 1)

            # we do this so that when we trim the zones later, we don't end up removing centroids
            # in the future consider adding a parameter to set the distance, so that we will not fail
            # if we have more aggressive trimming
            for neighbor in self.G.neighbors(centroid_z):
                if neighbor in self.valid_area_per_zone[z]:
                    self.m.Add(self.x[z][neighbor] == 1)

    def _school_count_const(self):
        avg_school_count = sum([len(node[1]['school_ids']) for node in self.G.nodes(data=True)]) / self.Z

        # TODO: ask sfusd if this is ok
        school_ub = int(avg_school_count + 2)
        school_lb = int(avg_school_count)

        # note: although we enforce max deviation of 1 from avg, in practice,
        # no two zones will have more than 1 difference in school count
        # Reason: school count is int. Observe the avg_school_count +-1,
        # if avg_school_count is not int, and see how the inequalities will look like
        # * I implemented the code this way (instead of pairwise comparison), since it is faster
        for z in range(self.Z):
            school_coefs = []
            school_vars = []
            for j in self.valid_area_per_zone[z]:
                school_coefs.append(len(self.G.nodes[j]['school_ids']))
                school_vars.append(self.x[z][j])
            zone_school_count = cp_model.LinearExpr.WeightedSum(school_vars, school_coefs)
            self.m.AddLinearConstraint(zone_school_count, school_lb, school_ub)

    def _contiguity_const(self):
        # (x[j,z] (and indicator that unit j is assigned to zone z)) \leq
        # (sum of all x[j',z] where j' is in self.closer_neighbors_per_centroid[area,c] where c is centroid for z)

        closer_neighbors = {}
        all_neighbors = {}
        for j in range(self.A):
            closer_neighbors[j] = {}
            all_neighbors[j] = {}
            for z in range(self.Z):
                closer_neighbors[j][z] = [
                    neighbor for neighbor in self.G.neighbors(j)
                    if self.G.graph['distance_dict'][self.centroids[z]][neighbor] <
                       self.G.graph['distance_dict'][self.centroids[z]][j]
                       and neighbor in self.valid_area_per_zone[z]
                ]
                all_neighbors[j][z] = [
                    neighbor for neighbor in self.G.neighbors(j)
                    if neighbor in self.valid_area_per_zone[z]
                ]

        # a good neighbor is a closer neighbor that also has a closer neighbor (or is a centroid)
        # an ok neighbor is any neighbor that also has a closer neighbor (or is a centroid)
        good_neighbors = {}
        ok_neighbors = {}
        for j in range(self.A):
            good_neighbors[j] = {}
            ok_neighbors[j] = {}
            for z in range(self.Z):
                good_neighbors[j][z] = []
                ok_neighbors[j][z] = []
                for neighbor in closer_neighbors[j][z]:
                    if len(closer_neighbors[neighbor][z]) >= 1:
                        good_neighbors[j][z].append(neighbor)
                    elif neighbor == self.centroids[z]:
                        good_neighbors[j][z].append(neighbor)
                for neighbor in all_neighbors[j][z]:
                    if len(closer_neighbors[neighbor][z]) >= 1:
                        ok_neighbors[j][z].append(neighbor)
                    elif neighbor == self.centroids[z]:
                        ok_neighbors[j][z].append(neighbor)

        def try_to_add_neighbor_constraint(neighbor_set, cur_zone, cur_block):
            cur_neighbors = [
                self.x[z][k]
                for k in neighbor_set
            ]
            if len(cur_neighbors) > 0:
                self.m.AddBoolOr(cur_neighbors).OnlyEnforceIf(self.x[cur_zone][cur_block])
                return True
            return False

        for j in range(self.A):
            for z in range(self.Z):
                if j not in self.valid_area_per_zone[z]:
                    continue
                if self.centroids[z] in self.G.neighbors(j):
                    continue
                # if j is a centroid, any_neighbor will do
                if j == self.centroids[z]:
                    continue

                # only impose the contiguity if the area j has a neighbor that is closer to centroid z.
                # otherwise, just make sure j has at least another neighbor assigned tot the same zone z, so that
                # j is not an island assigned to z.
                # TODO: Deal with these stupid edge cases
                if try_to_add_neighbor_constraint(good_neighbors[j][z], z, j):
                    continue
                if try_to_add_neighbor_constraint(ok_neighbors[j][z], z, j):
                    continue
                if try_to_add_neighbor_constraint(closer_neighbors[j][z], z, j):
                    continue
                if try_to_add_neighbor_constraint(all_neighbors[j][z], z, j):
                    continue
                # print('no acceptable neighbors at all!!! area ', j, ' zone ', z)

    def _racial_const(self):
        for race_col in AREA_ETHNICITIES:
            race_dev = self.config['racial_dev']

            r = self.G.graph['R'][race_col]

            for z in range(self.Z):
                lb_coefs = []
                ub_coefs = []

                for j in self.valid_area_per_zone[z]:
                    # students = self.dz.area_data["ge_students"][j]
                    # race = self.dz.area_data[race_col][j]
                    students = self.G.nodes[j]["ge_students"]
                    race = self.G.nodes[j][race_col]

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
        f = self.G.graph['F']

        for z in range(self.Z):
            lb_coefs = []
            ub_coefs = []

            for j in self.valid_area_per_zone[z]:
                # students = self.dz.area_data["ge_students"][j]
                # frl = self.dz.area_data['FRL'][j]
                students = self.G.nodes[j]["ge_students"]
                frl = self.G.nodes[j]['FRL']

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
        for z in range(self.Z):
            coefs = []
            for j in self.valid_area_per_zone[z]:
                lb_seats = (1 - self.config['shortage']) * self.G.nodes[j]['ge_capacity']
                students = self.G.nodes[j]["ge_students"]
                coef = int(SCALING_CONST * (students - lb_seats))
                coefs.append(coef)
            vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]
            total_expr = cp_model.LinearExpr.WeightedSum(vars_list, coefs)
            self.m.Add(total_expr >= 0)

    # percentage of students (GE students) in the zone, that we need to add to fill all the GE seats in the zone
    def _proportional_overage_const(self):
        # No zone has overage more than overage percentage of its population
        for z in range(self.Z):
            coefs = []
            for j in self.valid_area_per_zone[z]:
                ub_seats = (1 + self.config['overage']) * self.G.nodes[j]['ge_capacity']
                students = self.G.nodes[j]["ge_students"]
                coef = int(SCALING_CONST * (students - ub_seats))
                coefs.append(coef)
            vars_list = [self.x[z][j] for j in self.valid_area_per_zone[z]]
            total_expr = cp_model.LinearExpr.WeightedSum(vars_list, coefs)
            self.m.Add(total_expr <= 0)

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

        for zone in range(self.Z):
            for i in self.valid_area_per_zone[zone]:
                for j in self.G.neighbors(i):
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
        if self.config['use_hints']:
            self._add_hints()

        solver = cp_model.CpSolver()
        self._add_solver_parameters(solver)

        status = solver.Solve(self.m)

        objective_value = solver.ObjectiveValue()
        wall_time = solver.WallTime()

        return SolutionOutput(self._generate_zone_dict(solver), objective_value,
                              solver.StatusName(status), wall_time, self.G, self.config['is_local'])

    def fix_areas(self, fixed_zone_dict):
        if fixed_zone_dict is None:
            return
        for area, zone in fixed_zone_dict.items():
            centroid_idx = self.centroid_schools.index(zone)
            if centroid_idx in self.valid_zone_per_area[area]:
                self.m.Add(self.x[centroid_idx][area] == 1)

    def _add_solver_parameters(self, solver):
        solver.parameters.max_time_in_seconds = self.config['solve_time_limit']
        solver.parameters.max_presolve_iterations = 5
        solver.parameters.relative_gap_limit = self.config.get('relative_gap_limit', 0)

        solver.parameters.random_seed = self.config['random_seed']
        if self.config['is_local']:
            solver.parameters.num_search_workers = 6
        else:
            solver.parameters.num_search_workers = 32

        # important to think about this parameter and thourhgly test later. for now leave at 1
        solver.parameters.linearization_level = 0
        solver.parameters.symmetry_level = 4

        log_folder = self.config.get('log_folder')
        if log_folder is not None:
            solver.parameters.log_to_stdout = False
            solver.parameters.log_search_progress = True

            log_file_path = f"{log_folder}/{self.config['level']}_log.txt"

            with open(log_file_path, "w") as f:
                f.write("")

            def on_log_message(message: str):
                with open(log_file_path, "a") as f:
                    f.write(message + "\n")

            # Assign the callback and solve
            solver.log_callback = on_log_message
        else:
            solver.parameters.log_to_stdout = True
            solver.parameters.log_search_progress = True

    def _add_hints(self):
        # add hint that each area will be assigned to the closest centroid
        for i in range(self.A):
            closest_centroid = None
            closest_distance = float('inf')
            for z in range(self.Z):
                centroid_z = self.centroids[z]
                dist = self.G.graph['distance_dict'][centroid_z][i]
                if dist < closest_distance:
                    closest_distance = dist
                    closest_centroid = z
            if closest_centroid in self.valid_zone_per_area[i]:
                self.m.AddHint(self.x[closest_centroid][i], 1)
                for z in self.valid_zone_per_area[i]:
                    if z != closest_centroid:
                        self.m.AddHint(self.x[z][i], 0)

        # use metis to generate initial zones
        # super_nodes = partition_graph_metis_constrained(self.G, self.Z, self.centroids)
        # for i in range(self.A):
        #     assigned_zone = None
        #     for z in range(self.Z):
        #         if i in super_nodes[z]:
        #             assigned_zone = z
        #             break
        #     if assigned_zone is not None and assigned_zone in self.valid_zone_per_area[i]:
        #         self.m.AddHint(self.x[assigned_zone][i], 1)
        #         for z in self.valid_zone_per_area[i]:
        #             if z != assigned_zone:
        #                 self.m.AddHint(self.x[z][i], 0)


    def _generate_zone_dict(self, solver):
        zone_dict = {}
        for i in range(self.A):
            for z in self.valid_zone_per_area[i]:
                if solver.BooleanValue(self.x[z][i]) == 1:
                    zone_dict[i] = self.centroid_schools[z]
                    break
        return zone_dict
