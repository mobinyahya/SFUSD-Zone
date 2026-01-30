import os
import sys
import yaml
import time
import pandas as pd

# Change to Optimization dir so relative paths in optimizer.py work
target_dir = "/home/kumarc/sfusd/SFUSD-Zone/Zone_Generation/Optimization"
if os.getcwd() != target_dir:
    os.chdir(target_dir)

# Add source root to path
sys.path.append("../../")

from Zone_Generation.Optimization.iterative_choice import IterativeChoiceOptimizer

# Mock config
config_path = "../Config/config.yaml"
# We define a custom config to avoid messing with real file if we want but reading is fine.
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Override for testing
config['max_iterations'] = 40
config['solve_time_limit'] = 60
config['level'] = 'BlockGroup_1' # Reduced problem size
config['is_local'] = False 
config['use_hints'] = True # Disable hints to speed up setup if they rely on other things

print(f"Using level: {config['level']}")

try:
    optimizer = IterativeChoiceOptimizer(config)
    print("Optimizer initialized.")
    print(f"Graph size: {len(optimizer.G)} nodes, {len(optimizer.G.edges())} edges.")
    
    # Setup model
    print("Adding variables...")
    optimizer.add_variables()
    print("Adding constraints...")
    optimizer.add_constraints()
    print("Adding choice objective...")
    optimizer.add_choice_objective() 
    
    # Run
    print("Starting solve...")
    # start_time = time.time()
    # solution = optimizer.solve()
    # end_time = time.time()
    
    # DEBUG: Check formats before solving
    first_node = list(optimizer.G.nodes(data=True))[0]
    print(f"Sample Graph Node 0: {first_node}")
    area_id = first_node[1].get('area_id')
    print(f"Sample area_id: {area_id} (Type: {type(area_id)})")
    
    optimizer.evaluator.load_data()
    print("Student DF columns:", optimizer.evaluator.student_df.columns)
    print("Student DF head sample (census_blockgroup):")
    print(optimizer.evaluator.student_df[['census_blockgroup']].head())
    print("Student DF head sample (census_block):")
    if 'census_block' in optimizer.evaluator.student_df.columns:
        print(optimizer.evaluator.student_df[['census_block']].head())

    start_time = time.time()
    solution = optimizer.solve()
    end_time = time.time()
    
    print(f"Optimization finished in {end_time - start_time:.2f}s")
    print(f"Status: {solution.status}")
    print(f"Objective: {solution.objective_value}")
    
    if solution.zone_dict:
        print(f"Assigned {len(solution.zone_dict)} areas.")
    else:
        print("No zone dict returned.")

except Exception as e:
    print(f"Test failed with exception: {e}")
    import traceback
    traceback.print_exc()
