import json
import os
import pickle

import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import Compute_Name, convert_to_block_zone_dict, compute_zone_deviations, \
    compute_zone_demographics
from Zone_Generation.Config.Constants import get_dropbox_path


class SolutionOutput:
    def __init__(self, zone_dict, objective_value, status, wall_time, G, config):
        self.zone_dict = zone_dict
        self.objective_value = objective_value
        self.status = status
        self.wall_time = wall_time
        self.G = G  # the graph of all the data we need
        self.config = config
        self.block_zone_dict = convert_to_block_zone_dict(zone_dict, G)

    def get_boundary_cost(self):
        if self.zone_dict is None or len(self.zone_dict) == 0:
            return -1
        boundary_cost = 0
        for i in range(len(self.G)):
            for j in self.G.neighbors(i):
                if i < j:  # prevent double counting
                    continue
                if self.zone_dict[i] != self.zone_dict[j]:
                    boundary_cost += 1
        return boundary_cost

    def get_base_boundary_cost(self, base_G):
        boundary_cost = 0
        for i in range(len(base_G)):
            for j in base_G.neighbors(i):
                if i < j:  # prevent double counting
                    continue
                if self.block_zone_dict[base_G.nodes[i]['area_id']] != self.block_zone_dict[base_G.nodes[j]['area_id']]:
                    boundary_cost += 1
        return boundary_cost

    def get_zone_demographics(self):
        return compute_zone_demographics(self.G, self.zone_dict)



    def visualize_zones(self):
        zv = ZoneVisualizer(self.config['level'].split('_')[0], self.config['is_local'])
        zv.zones_from_dict(self.block_zone_dict, show_plot=True)

    def save_output(self, folder_path):
        if self.zone_dict is None or len(self.zone_dict) == 0:
            print("No zones available to save.")
            return
        # save the image, dict_solution, objective_value, status, wall_time
        save_path = os.path.expanduser(folder_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        level = self.config['level']
        # save zone dict as json file
        filename = os.path.join(save_path, f"zone_dict_{level}.json")
        with open(filename, "w") as f:
            json.dump(self.zone_dict, f)

        boundary_cost = -1
        if self.zone_dict is not None and len(self.zone_dict) > 0:
            boundary_cost = self.get_boundary_cost()
            file_name = os.path.join(save_path, f"zones_visualization_{level}")
            zv = ZoneVisualizer(self.config['level'].split('_')[0], self.config['is_local'])
            zv.zones_from_dict(self.block_zone_dict, save_path=file_name)

        output_info = {
            "boundary_cost": boundary_cost,
            "status": self.status,
            "wall_time": self.wall_time,
            'config': self.config
        }
        filename = os.path.join(save_path, f"solution_info_{level}.json")
        with open(filename, "w") as f:
            json.dump(output_info, f)


class Optimizer:
    def __init__(self, config):
        self.config = config

        graph_folder = f'{get_dropbox_path(self.config["is_local"])}/Optimization/Zones/Graphs'

        graph_filename = os.path.join(graph_folder, f"{self.config['level']}.pickle")
        with open(graph_filename, "rb") as f:
            self.G = pickle.load(f)

        # open centroids file
        with open("../Config/centroids.yaml", "r") as f:
            # with open("../Config/school_closure_centroids.yaml", "r") as f:
            centroid_configs = yaml.safe_load(f)
        if self.config['centroids_type'] not in centroid_configs:
            raise ValueError("The centroids type specified is not defined in centroids.yaml.")

        self.centroid_schools = centroid_configs[self.config['centroids_type']]
        # search graph for centroid_school in node['school_ids']
        self.centroids = []
        for centroid_school in self.centroid_schools:
            for node in self.G.nodes(data=True):
                if centroid_school in node[1]['school_ids']:
                    self.centroids.append(node[0])
                    break
        self.Z = len(self.centroids)
        self.A = len(self.G)

    def add_constraints(self):
        raise NotImplementedError('Subclasses must implement add_constraints')

    def add_objective(self):
        raise NotImplementedError('Subclasses must implement add_objective')

    def solve(self) -> SolutionOutput:
        raise NotImplementedError('Subclasses must implement solve')

    def fix_areas(self, fixed_zone_dict):
        raise NotImplementedError('Subclasses must implement fix_areas')

    def _add_hints(self):
        raise NotImplementedError('Subclasses must implement _add_hints')

    @staticmethod
    def get_optimizer(config):
        # mip, cp_int, cp_bool
        if config["optimizer"] == "mip":
            # TODO: care about the integer program if i need to benchmark it
            from Zone_Generation.Optimization.integer_program import Integer_Program
            return Integer_Program(None, config)
        elif config["optimizer"] == "cp_int":
            from Zone_Generation.Optimization.constraint_program_integer import IntegerConstraintProgram
            return IntegerConstraintProgram(config)
        elif config["optimizer"] == "cp_bool":
            from Zone_Generation.Optimization.constraint_program_boolean import BooleanConstraintProgram
            return BooleanConstraintProgram(config)
        else:
            raise ValueError("The optimizer type specified is not recognized.")


if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    name = Compute_Name(config)
    print("name: ", name)

    optimizer = Optimizer.get_optimizer(config)
    optimizer.add_constraints()
    optimizer.add_objective()
    solution_output = optimizer.solve()

    graph_folder = f'{get_dropbox_path(config["is_local"])}/Optimization/Zones/Graphs'

    graph_filename = os.path.join(graph_folder, f"Block_0.pickle")
    with open(graph_filename, "rb") as f:
        base_G = pickle.load(f)

    if solution_output.status != 'INFEASIBLE':
        print("Objective value: ", solution_output.objective_value)
        print('Boundary cost: ', solution_output.get_boundary_cost())
        print('Base Boundary cost: ', solution_output.get_base_boundary_cost(base_G))
        print('Wall time: ', solution_output.wall_time)
        solution_output.visualize_zones()
