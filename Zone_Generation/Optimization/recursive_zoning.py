import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimization.design_zones import Compute_Name
from Zone_Generation.Optimization.optimizer import DesignZones, Optimizer
from Zone_Generation.Optimzation_Heuristics.local_search_zoning import aa2bg_Zoning, bg2b_Zoning
from Zone_Generation.Optimzation_Heuristics.zone_eval import drop_boundary, trim_noncontiguity

AREA_HIERARCHY = ["attendance_area", "BlockGroup", "Block"]


def recursive_zone_supervised(config):
    cur_level_index = AREA_HIERARCHY.index(config["start_level"])
    zone_dicts = {}
    cur_zone_dict = None
    while True:
        cur_level = AREA_HIERARCHY[cur_level_index]
        zone_dict = solve_level(config, cur_level, cur_zone_dict)

        if zone_dict is None:
            print(f"Zoning failed at level: {cur_level}")
            break
        zone_dicts[cur_level] = zone_dict
        cur_zone_dict = zone_dict
        if cur_level == config["end_level"]:
            print(f"Completed zoning down to level: {cur_level}")
            break
        cur_level_index += 1

    return zone_dicts


def convert_to_lower_level_zones(dz, cur_zone_dict):
    if cur_zone_dict is None:
        return None
    if dz.level == 'BlockGroup':
        return aa2bg_Zoning(dz, cur_zone_dict)
    elif dz.level == 'Block':
        return bg2b_Zoning(dz, cur_zone_dict)
    else:
        return None


def solve_level(config, level, cur_zone_dict=None):
    print(f"Solving zoning for level: {level}")
    config["level"] = level
    dz = DesignZones(config=config)
    zv = ZoneVisualizer(config["level"])
    zone_dict = convert_to_lower_level_zones(dz, cur_zone_dict)
    if zone_dict is not None:
        zone_dict = drop_boundary(dz, zone_dict)
        zv.zones_from_dict(zone_dict)

        zone_dict = trim_noncontiguity(dz, zone_dict)
        zv.zones_from_dict(zone_dict)

    optimizer = Optimizer.get_optimizer(dz, config)
    optimizer.add_constraints()
    optimizer.add_objective()
    optimizer.fix_areas(zone_dict)
    zone_dict = optimizer.solve()

    if zone_dict is not None:
        print(f"Resulting zone dictionary for {level}: ", zone_dict)
        zv = ZoneVisualizer(level)
        zv.zones_from_dict(zone_dict, label=False)

    return zone_dict

if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    name = Compute_Name(config)
    print("name: ", name)

    recursive_zone_supervised(config)
