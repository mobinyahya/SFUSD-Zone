import time

import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import convert_block_zone_to_zone_dict, Compute_Name, convert_to_block_zone_dict
from Zone_Generation.Optimization.optimizer import Optimizer
from Zone_Generation.Optimzation_Heuristics.zone_eval import drop_boundary_by_graph_distance, trim_noncontiguity_soft


def recursive_zoning(config):
    cur_block_zone_dict = None
    solutions = []
    is_local = config['is_local']

    for i in range(len(config['recursive_levels'])):
        cur_level = config['recursive_levels'][i]
        config['relative_gap_limit'] = config['relative_gap_limits'][i]
        config['level'] = cur_level
        solution_output = solve_level(config, is_local, cur_block_zone_dict)
        cur_block_zone_dict = solution_output.block_zone_dict
        solutions.append(solution_output)
        if solution_output.status in ['INFEASIBLE', 'MODEL_INVALID', 'UNKNOWN']:
            print(f"Zoning infeasible at level: {cur_level}")
            break

    return solutions


def solve_level(config, is_local, cur_block_zone_dict):
    optimizer = Optimizer.get_optimizer(config)
    optimizer.add_constraints()
    optimizer.add_objective()

    zone_dict = None
    if cur_block_zone_dict is not None:
        zone_dict = convert_block_zone_to_zone_dict(cur_block_zone_dict, optimizer.G)
        num_nodes = len(optimizer.G)
        # c = 0
        # if num_nodes > 1000:
        #     c = 0.5
        # print(c)
        zone_dict = drop_boundary_by_graph_distance(zone_dict, optimizer.G, optimizer.centroids)
        zone_dict = trim_noncontiguity_soft(zone_dict, optimizer.G, optimizer.centroids)

        block_zone_dict = convert_to_block_zone_dict(zone_dict, optimizer.G)
        zv = ZoneVisualizer('Block', is_local)
        zv.zones_from_dict(block_zone_dict, show_plot=True)

    optimizer.fix_areas(zone_dict)
    solution_output = optimizer.solve()
    solution_output.visualize_zones()

    return solution_output


if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    name = Compute_Name(base_config)
    print("name: ", name)

    start_time = time.time()
    solution_outputs = recursive_zoning(base_config)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    print(f"Total optimization taken: {sum([output.wall_time for output in solution_outputs])} seconds")
    print("Final Objective value: ", solution_outputs[-1].get_boundary_cost())
