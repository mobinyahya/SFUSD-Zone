import time
from collections import defaultdict

from ortools.sat.python import cp_model

from Zone_Generation.Config.Constants import AREA_ETHNICITIES, SCALING_CONST
from Zone_Generation.Optimization.optimizer import Optimizer, SolutionOutput


class BooleanConstraintProgram(Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.m = cp_model.CpModel()
        self.forbidden_assignments = defaultdict(set)
        self.valid_area_per_zone, self.valid_zone_per_area, self.x = None, None, None

    def add_variables(self, fixed_areas: dict[int, int] = None):
        if fixed_areas is None:
            fixed_areas = {}
        max_distance = self.config.get('max_distance')

        if max_distance is None:
            max_distance = float('inf')
        valid_area_per_zone = {}
        valid_zone_per_area = {}
        x = {}
        zone_utility_vars = {}
        for z in range(self.Z):
            valid_area_per_zone[z] = set()
            x[z] = {}
            zone_utility_vars[z] = {}
        for i in range(self.A):
            valid_zone_per_area[i] = set()

        # identify the centroid neighbors
        centroid_neighbor_assignments = {}
        for i in self.centroids:
            zone = self.centroids.index(i)
            for neighbor in self.G.neighbors(i):
                centroid_neighbor_assignments[neighbor] = zone

        for z in range(self.Z):
            centroid_z = self.centroids[z]
            # create variable for the centroid
            var = self.m.NewBoolVar(f"zone_{z}_area_{centroid_z}")
            valid_area_per_zone[z].add(centroid_z)
            valid_zone_per_area[centroid_z].add(z)
            x[z][centroid_z] = var

            for i in range(self.A):
                # ignore if centroid
                if i in self.centroids:
                    continue

                if i in fixed_areas:
                    # only allow assignment to the fixed zone
                    fixed_zone = self.centroid_schools.index(fixed_areas[i])
                    if fixed_zone != z:
                        continue
                    # this is bad style to repeat the same code, but we do this to avoid cases where
                    # fixed areas are far from centroids or in centroid neighbors
                    var = self.m.NewBoolVar(f"zone_{z}_area_{i}")
                    valid_area_per_zone[z].add(i)
                    valid_zone_per_area[i].add(z)
                    x[z][i] = var
                    continue
                # ignore if far away from centroid
                if self.G.graph['distance_dict'][centroid_z][i] > max_distance:
                    continue

                if i in centroid_neighbor_assignments:
                    assigned_zone = centroid_neighbor_assignments[i]
                    if assigned_zone != z:
                        continue

                var = self.m.NewBoolVar(f"zone_{z}_area_{i}")
                valid_area_per_zone[z].add(i)
                valid_zone_per_area[i].add(z)
                x[z][i] = var

        # check if any area has no valid zones
        for i in range(self.A):
            if len(valid_zone_per_area[i]) == 0:
                print(f"Area {i} has no valid zones! Infeasible model likely.")
                self.m.Add(False)
        self.valid_area_per_zone = valid_area_per_zone
        self.valid_zone_per_area = valid_zone_per_area
        self.x = x

    def _feasibility_const(self):
        # Each area assigned to exactly one zone
        for i in range(self.A):
            self.m.AddExactlyOne([self.x[z][i] for z in self.valid_zone_per_area[i]])

    def _school_count_const(self):
        avg_school_count = sum([len(node[1]['school_ids']) for node in self.G.nodes(data=True)]) / self.Z

        school_ub = int(avg_school_count + 1)
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

        # print out any areas that have no good or ok neighbors for any zone
        # for j in range(self.A):
        #     if j in self.centroids:
        #         continue
        #     has_any_good = False
        #     has_any_ok = False
        #
        #     for z in range(self.Z):
        #         if len(good_neighbors[j][z]) > 0:
        #             has_any_good = True
        #         if len(ok_neighbors[j][z]) > 0:
        #             has_any_ok = True
        #     if not has_any_good:
        #         print('Area ', j, ' has no good neighbors for any zone')
        #     if not has_any_ok:
        #         print('Area ', j, ' has no ok neighbors for any zone')

        def try_to_add_neighbor_constraint(neighbor_set, cur_zone, cur_block):
            cur_neighbors = [
                self.x[z][k]
                for k in neighbor_set if z not in self.forbidden_assignments[k]
            ]
            if len(cur_neighbors) > 0:
                self.m.AddBoolOr(cur_neighbors).OnlyEnforceIf(self.x[cur_zone][cur_block])
                return True
            return False

        for j in range(self.A):
            for z in range(self.Z):
                if j not in self.valid_area_per_zone[z]:
                    continue

                # only impose the contiguity if the area j has a neighbor that is closer to centroid z.
                # otherwise, just make sure j has at least another neighbor assigned tot the same zone z, so that
                # j is not an island assigned to z.
                if try_to_add_neighbor_constraint(good_neighbors[j][z], z, j):
                    continue

                if try_to_add_neighbor_constraint(ok_neighbors[j][z], z, j):
                    continue

                # if try_to_add_neighbor_constraint(closer_neighbors[j][z], z, j):
                #     continue
                # if try_to_add_neighbor_constraint(all_neighbors[j][z], z, j):
                #     continue
                # print('no acceptable neighbors at all!!! area ', j, ' zone ', z)
                # do not allow this area to be assigned to this zone
                # print('we have reached here for area ', j, ' zone ', z)
                # make sure were not adding duplicate forbidden assignments
                # make sure were able to forbid the assignment by checking that valid_zone_per_area has more than 1 zone after
                if len(self.valid_zone_per_area[j] - self.forbidden_assignments[j]) <= 1:
                    # print('Cannot forbid assignment of area ', j, ' to zone ', z, )
                    continue

                self.forbidden_assignments[j].add(z)
                self.m.Add(self.x[z][j] == 0)
                # TODO: Deal with these stupid edge cases

    def _contiguity_const_flow(self):
        # Total possible nodes that could be in a zone (upper bound for flow)
        max_nodes = self.A

        for z in range(self.Z):
            centroid_z = self.centroids[z]
            # flow_vars[(i, j)] is the amount of flow from node i to neighbor j
            flow_vars = {}

            nodes_in_zone = self.valid_area_per_zone[z]

            for i in nodes_in_zone:
                # Get neighbors that are also valid for this zone
                neighbors = [n for n in self.G.neighbors(i) if n in nodes_in_zone]
                for n in neighbors:
                    # Flow can only exist if both nodes are in the zone
                    # Max flow is bounded by total nodes
                    f_var = self.m.NewIntVar(0, max_nodes, f"flow_z{z}_i{i}_n{n}")
                    flow_vars[(i, n)] = f_var

                    # Capacity Constraint: flow only if arc is 'active'
                    # We assume x[z][i] and x[z][n] are the assignment variables
                    self.m.Add(f_var == 0).OnlyEnforceIf(self.x[z][i].Not())
                    self.m.Add(f_var == 0).OnlyEnforceIf(self.x[z][n].Not())

            for i in nodes_in_zone:
                # Outgoing flow from i
                out_flow = [flow_vars[(i, n)] for n in nodes_in_zone
                            if (i, n) in flow_vars]
                # Incoming flow to i
                in_flow = [flow_vars[(n, i)] for n in nodes_in_zone
                           if (n, i) in flow_vars]

                if i == centroid_z:
                    # The Centroid: Source of flow
                    # Sum(Out) - Sum(In) = Total Nodes in Zone - 1
                    # Since we don't know the count, we use the assignment variables:
                    total_assigned_minus_one = sum(self.x[z][j] for j in nodes_in_zone if j != centroid_z)
                    self.m.Add(sum(out_flow) - sum(in_flow) == total_assigned_minus_one)
                else:
                    # Every other node: Consume 1 unit if assigned, else 0
                    # Sum(In) - Sum(Out) = x[z][i]
                    self.m.Add(sum(in_flow) - sum(out_flow) == self.x[z][i])

    def _contiguity_const_circuit(self):
        for z in range(self.Z):
            arcs = []
            nodes = self.valid_area_per_zone[z]
            dummy_node = self.A + z
            centroid_z = self.centroids[z]

            # 1. The Dummy Node
            # The dummy node CANNOT have a self-loop. It must be in the circuit.
            # It connects ONLY to the centroid (start of the zone)
            start_arc = self.m.NewBoolVar(f"z{z}_dummy_to_centroid")
            arcs.append((dummy_node, centroid_z, start_arc))

            # 2. Potential Exit Points
            # Every node in the zone could potentially be the one to 'close' the loop back to dummy
            for i in nodes:
                exit_arc = self.m.NewBoolVar(f"z{z}_area{i}_to_dummy")
                # This arc is only possible if node i is in the zone
                self.m.Add(exit_arc <= self.x[z][i])
                arcs.append((i, dummy_node, exit_arc))

                # 3. Self-loops for inactive nodes
                # If x[z][i] is 0, then the self-loop MUST be 1.
                # If x[z][i] is 1, then the self-loop MUST be 0.
                self_loop = self.m.NewBoolVar(f"z{z}_area{i}_self_loop")
                self.m.Add(self_loop == 1).OnlyEnforceIf(self.x[z][i].Not())
                self.m.Add(self_loop == 0).OnlyEnforceIf(self.x[z][i])
                arcs.append((i, i, self_loop))

                # 4. Arcs between neighbors
                for n in self.G.neighbors(i):
                    if n in self.valid_area_per_zone[z]:
                        arc_var = self.m.NewBoolVar(f"arc_z{z}_from{i}_to{n}")
                        # Basic requirement: both nodes must be in the zone
                        self.m.Add(arc_var <= self.x[z][i])
                        self.m.Add(arc_var <= self.x[z][n])
                        arcs.append((i, n, arc_var))

            self.m.AddCircuit(arcs)

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

    # we already have implicit constraint on distance
    def _school_quality_const(self):
        # ensure that each zone has at least one school above a certain great schools rating
        threshold = self.config.get('great_schools_threshold')
        if not threshold:
            return

        good_area_idxs = []
        for i in range(self.A):
            school_ids = self.G.nodes[i]['school_ids']
            for school_id in school_ids:
                if self.G.graph['school_data'][school_id]['greatschools_rating'] >= threshold:
                    good_area_idxs.append(i)
        for z in range(self.Z):
            good_school_areas = []
            for i in good_area_idxs:
                if i in self.valid_area_per_zone[z]:
                    good_school_areas.append(self.x[z][i])
            if len(good_school_areas) > 0:
                self.m.AddBoolOr(good_school_areas)
            else:
                # infeasible constraint, no area in this zone has a good school
                print('Infeasible school quality constraint for zone ', z)
                self.m.Add(False)

    def _closest_school_const(self):
        raise NotImplementedError('Not implemented for boolean CP')

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
        # self._contiguity_const_flow()
        # self._contiguity_const_circuit()
        self._racial_const()
        self._frl_const()
        self._proportional_shortage_const()
        self._proportional_overage_const()
        self._boundary_const()

        # optional constraints
        # self._school_quality_const()
        # self._closest_school_const()

    def add_boundary_objective(self):
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
        self.m.Minimize(cp_model.LinearExpr.Sum(boundary_vars))

    def add_choice_objective(self):
        # for every area, we create a variable representing the value that the area gets from the choice set
        # for every indicator in self.x, we create a variable representing the value that the area gets from being assigned,
        # and add it to the objective
        
        for area in range(self.A):
            for zone in self.valid_zone_per_area[area]:
                lb = int(-100* self.G[area]['ge_students'])
                ub = int(100* self.G[area]['ge_students'])
                area_zone_utility = self.m.NewIntVar(lb, ub, f"area_zone_utility_{area}_{zone}")
                self.m.Add(area_zone_utility == 0).OnlyEnforceIf(self.x[zone][area].Not())
                self.zone_utility_vars[zone][area] = area_zone_utility

        self.m.Maximize(sum(self.zone_utility_vars.values()))

    def _boundary_const(self):
        # isntead of minimizing boundary cost, we can add a constraint that the boundary cost
        # must be below a certain threshold relative to the proportion of total edges
        max_boundary_proportion = 0.2
        if not max_boundary_proportion:
            return

        total_boundary_edges = 0
        for area in range(self.A):
            for neighbor in self.G.neighbors(area):
                if area < neighbor:
                    total_boundary_edges += 1
        boundary_vars = []
        for area in range(self.A):
            for neighbor in self.G.neighbors(area):
                if area < neighbor:
                    for zone in self.valid_zone_per_area[area]:
                        if zone in self.valid_zone_per_area[neighbor]:
                            boundary_var = self.m.NewBoolVar(f"boundary_area_{area}_neighbor_{neighbor}_zone_{zone}")
                            boundary_vars.append(boundary_var)
                            # if area and neighbor are assigned to different zones, then boundary_var = 1
                            self.m.Add(self.x[zone][area] != self.x[zone][neighbor]).OnlyEnforceIf(boundary_var)
                            self.m.Add(self.x[zone][area] == self.x[zone][neighbor]).OnlyEnforceIf(boundary_var.Not())
        boundary_sum = cp_model.LinearExpr.Sum(boundary_vars)
        self.m.Add(
            int(SCALING_CONST) * boundary_sum <= int(SCALING_CONST * max_boundary_proportion) * total_boundary_edges)


    def solve(self):
        if self.config['use_hints']:
            self._add_hints()

        solver = cp_model.CpSolver()
        log_file = self._add_solver_parameters(solver, objective_threshold=self.config.get('objective_threshold'))
        status = solver.Solve(self.m)
        if log_file is not None:
            print("Closing log file.")
            log_file.close()

        objective_value = solver.ObjectiveValue()
        wall_time = solver.WallTime()
        status_name = solver.StatusName(status)
        zone_dict = None
        if status_name == 'OPTIMAL' or status_name == 'FEASIBLE':
            zone_dict = self._generate_zone_dict(solver)

        return SolutionOutput(zone_dict, objective_value,
                              solver.StatusName(status), wall_time, self.G, self.config)

    def _add_solver_parameters(self, solver, objective_threshold=None, minimize=True):
        solver.parameters.max_time_in_seconds = self.config['solve_time_limit']
        solver.parameters.max_presolve_iterations = 10
        solver.parameters.relative_gap_limit = self.config.get('relative_gap_limit', 0)

        solver.parameters.random_seed = self.config['random_seed']
        if self.config['is_local']:
            solver.parameters.num_search_workers = 6
        else:
            solver.parameters.num_search_workers = 6

        # important to think about this parameter and thourhgly test later. for now leave at 1
        solver.parameters.linearization_level = 1
        solver.parameters.symmetry_level = 4
        # solver.parameters.keep_symmetry_in_presolve = True
        # solver.parameters.use_symmetry_in_lp = True

        log_file = None
        log_folder = self.config.get('log_folder', None)
        if log_folder is not None:
            solver.parameters.log_to_stdout = False
            solver.parameters.log_search_progress = True

            log_file_path = f"{log_folder}/{self.config['level']}_log.txt"

            class CallbackHandler:
                def __init__(self, config, objective_threshold=None, minimize=True):
                    self.log_file = open(log_file_path, "w", encoding='utf-8')
                    self.best_objective = float('inf') if minimize else float('-inf')
                    self.objective_threshold = objective_threshold
                    self.minimize = minimize
                    self.last_objective_time = time.time()
                    self.stall_limit = config.get('stall_time_limit', -1)  # in seconds
                    self.stopped = False
                    self.start_time = time.time()

                def on_log_message(self, message: str):
                    self.log_file.write(message + "\n")
                    self.log_file.flush()

                    cur_time = time.time()

                    # Parse current objective from log message
                    if 'best:' in message:
                        # take the number in between best: and next:
                        parts = message.split('best:')
                        cur_best_objective = float(parts[1].split('next:')[0].strip())

                        # Update best_objective whenever we find a better value
                        if (self.minimize and cur_best_objective < self.best_objective) or \
                           (not self.minimize and cur_best_objective > self.best_objective):
                            self.best_objective = cur_best_objective
                            self.last_objective_time = cur_time

                    # Check objective threshold
                    if self.objective_threshold is not None:
                        if self.minimize and self.best_objective < self.objective_threshold:
                            print(f"Stopping solver: objective {self.best_objective} reached threshold {self.objective_threshold}")
                            solver.StopSearch()
                        elif not self.minimize and self.best_objective > self.objective_threshold:
                            print(f"Stopping solver: objective {self.best_objective} reached threshold {self.objective_threshold}")
                            solver.StopSearch()

                    # Check stall time
                    if self.stall_limit > 0 and cur_time - self.last_objective_time > self.stall_limit:
                        if not self.stopped:
                            self.stopped = True
                            print(f"Stopping solver due to stall in objective improvement at time {cur_time - self.start_time}.")
                            solver.StopSearch()

            callback_handler = CallbackHandler(self.config, objective_threshold=objective_threshold, minimize=minimize)
            # Assign the callback and solve
            solver.log_callback = callback_handler.on_log_message
            log_file = callback_handler.log_file
        else:
            solver.parameters.log_to_stdout = True
            solver.parameters.log_search_progress = False

        return log_file

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

    def _generate_zone_dict(self, solver):
        zone_dict = {}
        for i in range(self.A):
            for z in self.valid_zone_per_area[i]:
                if solver.BooleanValue(self.x[z][i]) == 1:
                    zone_dict[i] = self.centroid_schools[z]
                    break
        return zone_dict
