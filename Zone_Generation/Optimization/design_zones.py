import sys

from Zone_Generation.Optimization.optimizer import DesignZones, Optimizer

sys.path.append("../..")
from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import *
from Zone_Generation.Optimization.load_optimization_data_old import *


def Compute_Name(config):
    name = str(config["centroids_type"])
    # # add frl deviation
    # name += "_frl_" + str(config["frl_dev"])
    # # add shortage
    # # name += "_shortage_" + str(config["shortage"])

    return name


def load_zones_from_file(file_path):
    zone_lists = []
    with open(file_path, 'r', newline='') as file:
        print("file_path ", file_path)
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            # Convert each element in the row to an integer and store it in the list
            zone_row = []
            for cell in row:
                # Split the cell content by commas, convert to integers, and append to the row
                cell_values = [int(val.strip()) for val in cell.split(',') if val.strip()]  #
                zone_row.extend(cell_values)
            zone_lists.append(zone_row)

    # build a zone dictionary based on zone_list
    zone_dict = {}
    for index, sublist in enumerate(zone_lists):
        for item in sublist:
            zone_dict[item] = index

    return zone_lists, zone_dict


if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    name = Compute_Name(config)
    print("name: ", name)

    dz = DesignZones(config=config)
    optimizer = Optimizer.get_optimizer(dz, config)
    optimizer.add_constraints()
    optimizer.add_objective()
    solution_output = optimizer.solve()

    if solution_output.status != 'INFEASIBLE':
        zone_dict = solution_output.zone_dict

        zv = ZoneVisualizer(config["level"])
        zv.zones_from_dict(zone_dict, save_path='output')

        print("Objective value: ", solution_output.objective_value)
        print('Boundary cost: ', solution_output.get_boundary_cost())
        print('Demographics: ', solution_output.get_zone_demographics())

# Note: when you update the distance/neighboring files, also update the closer_eucledian distance file
# Note: Total number of students in aa level is not the same as blockgroup level.
# Reason: some students, do not have their bg info available
# (but they do have their aa info, and also they pass every other filter, i.e. enrollment)
