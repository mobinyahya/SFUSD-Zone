from ortools.sat.python import cp_model
from Zone_Generation.Optimization.constraint_program_integer import IntegerConstraintProgram
from Zone_Generation.Optimization.utility_evaluation import UtilityEvaluator
from Zone_Generation.Config.Constants import SCALING_CONST
import pandas as pd

class IterativeChoiceOptimizer(IntegerConstraintProgram):
    def __init__(self, config):
        super().__init__(config)
        
        # Hardcoded paths as per plan/user context (or verify_modifications.py)
        # In a production setting, these should be in config, but for this task we use what we found.
        utility_path = "/share/data/school_choice/simulation-files/choice-model/estimates_2324_exp8_0514.csv"
        student_path = "/share/data/school_choice/Data/Cleaned/r1_filter_student_without_specialprogs_2324.csv"
        
        print(f"Initializing UtilityEvaluator with:\n utility: {utility_path}\n student: {student_path}")
        self.evaluator = UtilityEvaluator(utility_path, student_path)
        self.max_iterations = config.get('max_iterations', 5)
        
    # To store variables for Benders decomposition
        self.area_utility_vars = {}
        self.school_id_to_area = {}
        self.best_real_utility = -float('inf')
        self.best_zone_dict = None

    def add_variables(self, fixed_areas: dict[int, int] = None):
        super().add_variables(fixed_areas)
        # Build school to area mapping for easy lookup
        for area_id in range(self.A):
            if area_id in self.G.nodes:
                if 'school_ids' in self.G.nodes[area_id]:
                    for sid in self.G.nodes[area_id]['school_ids']:
                        self.school_id_to_area[sid] = area_id

    def add_choice_objective(self):
        """
        Adds the utility variables to the model and sets the objective.
        We strictly follow the boolean program structure but ensure area_utility_vars are correctly stored.
        """
        self.area_utility_vars = {} 
        all_utility_vars = []
        
        # We need a large enough range for utilities. 
        # Scaled utilities can be large.
        # Assuming +/- 1,000,000 is safe for scaled values.
        # UPDATE: Utilities can be very negative (-1e10) -> -200B.
        # CP-SAT supports up to roughly +/- 9e18.
        min_u = -1000000000
        max_u = 1000000000
        
        for i in range(self.A):
            var = self.m.NewIntVar(min_u, max_u, f"u_{i}")
            self.area_utility_vars[i] = var
            all_utility_vars.append(var)

        # Maximize sum of all utility variables
        self.m.Maximize(sum(all_utility_vars))

    def solve(self):
        """
        Overrides the solve method to implement the iterative loop.
        """
        if self.config['use_hints']:
            self._add_hints()

        solver = cp_model.CpSolver()
        log_file = self._add_solver_parameters(solver)
        
        # Iteration Loop
        best_solution = None
        
        for iteration in range(self.max_iterations):
            if iteration == 4 * self.max_iterations // 5:
                # doubling the solve time limit
                self.config['solve_time_limit'] *= 2
            print(f"\n=== Iteration {iteration + 1} / {self.max_iterations} ===")
            
            # Solve current model
            status = solver.Solve(self.m)
            status_name = solver.StatusName(status)
            
            model_obj = 0.0
            if status == cp_model.OPTIMAL:
                model_obj = solver.ObjectiveValue()
                print(f"Solver Status: OPTIMAL, Objective: {model_obj}")
            elif status == cp_model.FEASIBLE:
                model_obj = solver.ObjectiveValue()
                print(f"Solver Status: FEASIBLE, Objective: {model_obj}")
            else:
                print(f"Solver Status: {status_name}")
                break
            
            # Extract Zoning
            zone_dict = self._generate_zone_dict(solver)
            
            # Evaluate using Real Utility Evaluator
            # Note: We need to convert zone_dict keys to match what evaluator expects?
            # _generate_zone_dict returns {area_index: zone_id} or {area_id: zone_id}?
            # base class _generate_zone_dict uses keys as area INDICES (0..A-1).
            # But DesignZones/UtilityEvaluator might expect original Area IDs (e.g., census block IDs).
            # graph_utils/create_larger_areas uses indices mapped to area_id.
            
            # We need to pass a dict that maps Node ID (from graph) to Zone ID.
            # self.G nodes are indexed by integers 0..A-1.
            # so zone_dict from base class is correct for self.G.
            
            print("Evaluating zoning...")
            # Use 'max' method as standard for this project unless specified
            # Strip level suffix to ensure correct column selection in Evaluator (BlockGroup vs Block)
            eval_level = self.config['level'].split('_')[0]
            
            eval_metrics = self.evaluator.evaluate(zone_dict, self.G, level=eval_level, method='logsum')
            gradients = self.evaluator.get_utility_impact_gradients(zone_dict, self.G, level=eval_level, method='logsum')
            
            block_utils = eval_metrics['block_utilities'] # Index might be block ID strings?
            # Normalize block_utils index to strings
            block_utils.index = block_utils.index.map(lambda x: str(int(float(x))) if pd.notnull(x) else '')
            
            # Calculate and print Total Utility and Gap
            total_utility = block_utils.sum()
            print(f"Real Total Utility: {total_utility}")
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                model_obj_scaled = model_obj / SCALING_CONST
                gap = model_obj_scaled - total_utility
                print(f"Objective Gap (Model - Real): {gap} (Model: {model_obj_scaled}, Real: {total_utility})")
            
            print("Generating Benders cuts...")
            cuts_added = 0
            
            # Prepare data for faster access
            # Group impacts by block
            block_impacts_df = gradients['block_impacts']
            
            # block_impacts_df columns: [student_col, 'school_id', 'type', 'impact']
            student_col = block_impacts_df.columns[0]
            # Convert student_col to string for matching
            block_impacts_df['block_id_str'] = block_impacts_df[student_col].apply(lambda x: str(int(float(x))) if pd.notnull(x) else '')
            
            # Group impacts by block_id_str for fast lookup
            # Dictionary: block_id_str -> {school_id -> {type -> impact}}
            # This pre-processing is crucial for performance
            impacts_lookup = {}
            for block_id, group in block_impacts_df.groupby('block_id_str'):
                school_map = {}
                for _, row in group.iterrows():
                    sid = row['school_id']
                    if sid not in school_map: school_map[sid] = {}
                    school_map[sid][row['type']] = row['impact']
                impacts_lookup[block_id] = school_map

            # Identify current schools for each zone based on solver solution
            # We need this to determine the baseline 'current_schools' for each AREA.
            # An area 'i' is assigned to some zone 'z_assigned'. 
            # The baseline schools for 'i' are the schools in 'z_assigned'.
            zone_current_schools = {}
            for z in range(self.Z):
                sids = set()
                for i in self.valid_area_per_zone[z]:
                    if solver.BooleanValue(self.x[z][i]):
                        if i in self.G.nodes:
                             sids.update(self.G.nodes[i].get('school_ids', []))
                zone_current_schools[z] = sids

            # Iterate over Areas first
            for i in range(self.A):
                # 1. Identify constituent blocks for this graph node
                node_data = self.G.nodes[i]
                block_ids = []
                if 'block_ids' in node_data:
                    block_ids = [str(int(float(bid))) for bid in node_data['block_ids']]
                elif 'area_id' in node_data:
                    block_ids = [str(int(float(node_data['area_id'])))]
                
                if not block_ids:
                    continue

                # 2. Aggregate utility and impacts for this Block Group (node i)
                val_i = 0.0
                total_impacts = {} # sid -> {type -> sum_impact}
                has_data = False
                
                for bid in block_ids:
                    # Value
                    if bid in block_utils.index:
                        val = block_utils.loc[bid]
                        if not pd.isna(val):
                            val_i += val
                            has_data = True
                    
                    # Impacts
                    if bid in impacts_lookup:
                        for sid, types in impacts_lookup[bid].items():
                            if sid not in total_impacts: total_impacts[sid] = {'add': 0.0, 'remove': 0.0}
                            for t, imp in types.items():
                                total_impacts[sid][t] += imp

                # If no data, constrain ALL valid zone vars for this area to 0
                if not has_data and not total_impacts:
                    self.m.Add(self.area_utility_vars[i] == 0)
                    continue

                # 3. Determine Baseline Schools for Area i
                # Find which zone 'i' is currently assigned to
                assigned_z = -1
                for z in self.valid_zone_per_area[i]:
                    if solver.BooleanValue(self.x[z][i]):
                        assigned_z = z
                        break
                
                # If unassigned (should involve strict penalty elsewhere, but here we just need a set),
                # assume empty set or handle gracefully.
                # If assigned, use that zone's schools.
                if assigned_z != -1:
                    current_schools = zone_current_schools[assigned_z]
                else:
                    current_schools = set()

                # 4. Generate the Cut Logic
                # The cut calculates: Utility if in Zone Z <= Baseline Utility + Changes
                # Changes depend on: Is school S in Zone Z? (y_zs) vs Is school S in Baseline?
                
                # Pre-calculate linear terms
                # Linear expression = Constant + Sum(coeff * var)
                # Constant starts with val_i
                
                # We iterate over all schools that have impacts on i
                # For each school s:
                #   If s in current_schools (Existing):
                #       We look at REMOVE impact.
                #       Term: impact * (y_zs - 1). 
                #       Constant += -impact. Coeff for y_zs += impact.
                #   If s not in current_schools (Potential):
                #       We look at ADD impact.
                #       Term: impact * y_zs.
                #       Coeff for y_zs += impact.
                
                base_constant = int(val_i * SCALING_CONST)
                coeffs = {} # s_area_id -> cumulative coefficient
                
                for sid_code, impacts in total_impacts.items():
                    # Parse SID
                    try:
                        sid_int = int(float(sid_code))
                    except:
                        continue
                    
                    # Get area of school
                    if sid_int not in self.school_id_to_area:
                        continue
                    s_area = self.school_id_to_area[sid_int]
                    
                    if sid_int in current_schools:
                        # Baseline: Present. Consider removal.
                        grad = int(impacts.get('remove', 0.0) * SCALING_CONST)
                        # Term: grad * (y - 1) = grad*y - grad
                        base_constant -= grad
                        coeffs[s_area] = coeffs.get(s_area, 0) + grad
                    else:
                        # Baseline: Absent. Consider addition.
                        grad = int(impacts.get('add', 0.0) * SCALING_CONST)
                        # Term: grad * y
                        coeffs[s_area] = coeffs.get(s_area, 0) + grad

                # 5. Apply Cut to ALL valid zones for Area i
                for z in self.valid_zone_per_area[i]:
                    # Build CP-SAT expression
                    # Start with constant
                    linear_expr = base_constant
                    
                    # Add terms for schools possible in this zone
                    # Note: x[z][s_area] exists only if s_area is valid for z.
                    # If s_area is NOT valid for z, x[z][s_area] is implicitly 0.
                    # So we ignore terms where s_area is not in valid_area_per_zone[z]
                    
                    term_vars = []
                    term_coeffs = []
                    
                    for s_area, coeff in coeffs.items():
                        if s_area in self.valid_area_per_zone[z]:
                            # School area is valid for this zone, variable exists
                            term_vars.append(self.x[z][s_area])
                            term_coeffs.append(coeff)
                        else:
                            # School area cannot be in this zone. y=0.
                            # If we had a removal term (grad * (y-1) = -grad), we added -grad to base_constant via 'base_constant -= grad'.
                            # And we added +grad to coeff.
                            # Since y=0, the +grad*y part contributes 0. 
                            # So the component is just -grad (penalty for missing school). Correct.
                            pass
                            
                    # Create linear expression object or sum
                    if term_vars:
                        linear_expr += cp_model.LinearExpr.WeightedSum(term_vars, term_coeffs)
                    
                    # Apply cut: If assigned to zone z, then utility <= linear_expr
                    self.m.Add(self.area_utility_vars[i] <= linear_expr+10).OnlyEnforceIf(self.x[z][i])
                    cuts_added += 1
            
            print(f"Added {cuts_added} Benders cuts.")
            
            # Use current solution as hint for next iteration
            if total_utility > self.best_real_utility:
                self.best_real_utility = total_utility
                self.best_zone_dict = zone_dict.copy()
                self.m.ClearHints()
                for z in range(self.Z):
                    for i in self.valid_area_per_zone[z]:
                        val = solver.BooleanValue(self.x[z][i])
                        self.m.AddHint(self.x[z][i], val)
                        if val:
                            self.m.AddHint(self.y[i], z)
            
            # Optionally check convergence
            best_solution = self._generate_solution_output(solver, status, zone_dict, model_obj)
            
            # Debug: Save output to local folder
            import os
            debug_dir = f"debug_output/iter_{iteration}"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            print(f"Saving debug output to {debug_dir}")
            best_solution.save_output(debug_dir)
            
        return best_solution


    def _generate_solution_output(self, solver, status, zone_dict, obj_val):
        from Zone_Generation.Optimization.optimizer import SolutionOutput
        wall_time = solver.WallTime()
        status_name = solver.StatusName(status)
        return SolutionOutput(zone_dict, obj_val, status_name, wall_time, self.G, self.config)
