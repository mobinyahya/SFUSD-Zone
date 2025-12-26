import time

import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import compute_zone_deviations
from Zone_Generation.Optimization.design_zones import Compute_Name
from Zone_Generation.Optimization.optimizer import DesignZones, Optimizer
from Zone_Generation.Optimzation_Heuristics.local_search_zoning import aa2bg_Zoning, bg2b_Zoning
from Zone_Generation.Optimzation_Heuristics.zone_eval import trim_noncontiguity, drop_boundary_by_graph_distance, \
    trim_noncontiguity_soft

AREA_HIERARCHY = ["attendance_area", "BlockGroup", "Block"]


def recursive_zone_supervised(config):
    total_time = 0
    cur_level_index = AREA_HIERARCHY.index(config["start_level"])
    cur_zone_dict = None
    is_local = config['is_local']
    while True:
        cur_level = AREA_HIERARCHY[cur_level_index]
        solution_output = solve_level(config, cur_level, is_local, cur_zone_dict)
        cur_zone_dict = solution_output.zone_dict
        total_time += solution_output.wall_time
        if solution_output.status == 'INFEASIBLE':
            print(f"Zoning infeasible at level: {cur_level}")
            break
        if cur_zone_dict is None:
            print(f"Zoning failed at level: {cur_level}")
            break
        if cur_level == config["end_level"]:
            break
        cur_level_index += 1

    return solution_output, total_time


def convert_to_lower_level_zones(dz, cur_zone_dict):
    if cur_zone_dict is None:
        return None
    if dz.level == 'BlockGroup':
        return aa2bg_Zoning(dz, cur_zone_dict)
    elif dz.level == 'Block':
        return bg2b_Zoning(dz, cur_zone_dict)
    else:
        return None


def solve_level(config, level, is_local, cur_zone_dict=None):
    print(f"Solving zoning for level: {level}")
    config["level"] = level
    dz = DesignZones(config=config)
    zone_dict = convert_to_lower_level_zones(dz, cur_zone_dict)
    zv = ZoneVisualizer(dz.level, is_local)
    if zone_dict is not None:
        zv.zones_from_dict(zone_dict, label=False, show_plot=True)
        zone_dict = drop_boundary_by_graph_distance(dz, zone_dict)
        zv.zones_from_dict(zone_dict, label=False, show_plot=True)
        zone_dict = trim_noncontiguity_soft(dz, zone_dict)
        zv.zones_from_dict(zone_dict, label=False, show_plot=True)

    optimizer = Optimizer.get_optimizer(dz, config)
    optimizer.add_constraints()
    optimizer.add_objective()
    optimizer.fix_areas(zone_dict)
    solution_output = optimizer.solve()

    return solution_output


if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    name = Compute_Name(base_config)
    print("name: ", name)

    start_time = time.time()
    output, user_time = recursive_zone_supervised(base_config)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    print(f"Total optimization taken: {user_time} seconds")
    print("Final Objective value: ", output.get_boundary_cost())
    if output.status != 'INFEASIBLE':
        zv = ZoneVisualizer(output.dz.level, base_config['is_local'])
        zv.zones_from_dict(output.zone_dict, label=False, show_plot=True)
        print("Final Demographics: ", compute_zone_deviations(output.get_zone_demographics()))
