import time

import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimization.design_zones import Compute_Name
from Zone_Generation.Optimization.optimizer import DesignZones, Optimizer
from Zone_Generation.Optimzation_Heuristics.local_search_zoning import aa2bg_Zoning, bg2b_Zoning
from Zone_Generation.Optimzation_Heuristics.zone_eval import trim_noncontiguity, drop_boundary


def compute_zone_deviations(zone_demographics):
    # get deviations in FRL and racial composition from average across zones
    if not zone_demographics:
        return {"frl_deviation": 0.0, "max_racial_deviation": 0.0, "per_ethnicity_deviation": {}}

    zones = list(zone_demographics.keys())
    num_zones = len(zones)

    if num_zones == 0:
        return {"frl_deviation": 0.0, "max_racial_deviation": 0.0, "per_ethnicity_deviation": {}}

    # infer ethnicity keys as everything except 'total_students' and 'FRL'
    sample = next(iter(zone_demographics.values()))
    ethnicity_keys = [k for k in sample.keys() if k not in ('total_students', 'FRL')]

    # compute average FRL and ethnicity proportions across zones
    avg_frl = sum(zone_demographics[z].get('FRL', 0) for z in zones) / num_zones
    avg_ethnicity = {eth: sum(zone_demographics[z].get(eth, 0) for z in zones) / num_zones
                     for eth in ethnicity_keys}

    # compute deviations
    frl_deviations = [abs(zone_demographics[z].get('FRL', 0) - avg_frl) for z in zones]
    frl_deviation = max(frl_deviations) if frl_deviations else 0.0

    per_ethnicity_deviation = {}
    for eth in ethnicity_keys:
        eth_deviations = [abs(zone_demographics[z].get(eth, 0) - avg_ethnicity[eth]) for z in zones]
        per_ethnicity_deviation[eth] = round(max(eth_deviations), 2) if eth_deviations else 0.0

    max_racial_deviation = max(per_ethnicity_deviation.values()) if per_ethnicity_deviation else 0.0

    return {
        "frl_deviation": round(frl_deviation, 2),
        "max_racial_deviation": max_racial_deviation,
        "per_ethnicity_deviation": per_ethnicity_deviation
    }


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
        total_time += solution_output.user_time
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
    # zv = ZoneVisualizer(dz.level, is_local)
    if zone_dict is not None:
        # zv.zones_from_dict(zone_dict, label=False)

        zone_dict = drop_boundary(dz, zone_dict)
        # zv.zones_from_dict(zone_dict, label=False)
        zone_dict = trim_noncontiguity(dz, zone_dict)
        # zv.zones_from_dict(zone_dict, label=False)

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
    if output.status != 'INFEASIBLE':
        zv = ZoneVisualizer(output.dz.level, base_config['is_local'])
        zv.zones_from_dict(output.zone_dict, label=False)
        print("Final Demographics: ", compute_zone_deviations(output.get_zone_demographics()))
