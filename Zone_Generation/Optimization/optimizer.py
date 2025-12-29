import json
import os
import pickle
from collections import defaultdict

import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import Compute_Name, convert_to_block_zone_dict, compute_zone_deviations
from Zone_Generation.Config.Constants import AREA_ETHNICITIES, get_dropbox_path


class SolutionOutput:
    def __init__(self, zone_dict, objective_value, status, wall_time, G, is_local):
        self.zone_dict = zone_dict
        self.objective_value = objective_value
        self.status = status
        self.wall_time = wall_time
        self.G = G  # the graph of all the data we need
        self.is_local = is_local
        self.block_zone_dict = convert_to_block_zone_dict(zone_dict, G)

    def get_boundary_cost(self):
        boundary_cost = 0
        for i in range(len(self.G)):
            for j in self.G.neighbors(i):
                if self.zone_dict[i] != self.zone_dict[j]:
                    boundary_cost += 1
        return boundary_cost / 2  # each boundary counted twice

    def get_zone_demographics(self):
        """
        Calculates aggregated and proportional demographics (total students, FRL proportion,
        and racial/ethnic proportions) for each zone.

        Assumes self.dz.area_data is a pandas DataFrame and self.zone_dict maps area to zone_id.
        """

        # 1. Define the columns to aggregate and the corresponding keys for the output dictionary
        # The keys will be like 'White', 'Black', etc.
        ETHNICITY_KEYS = [col[len('Ethnicity_'):] for col in AREA_ETHNICITIES]

        # 2. Use a defaultdict to simplify accumulation and avoid checking for key existence
        # We initialize with a lambda that returns a dict structure for aggregation
        zone_aggregates = defaultdict(lambda: {
            "total_students": 0.0,
            "FRL": 0.0,
            **{key: 0.0 for key in ETHNICITY_KEYS}
        })
        # 3. Iterate over each area in the graph and accumulate counts into the appropriate zone

        for node in self.G.nodes(data=True):
            area_idx = node[0]
            zone_id = self.zone_dict[area_idx]
            area_data = node[1]

            # Reference the aggregate dictionary for the current zone
            agg = zone_aggregates[zone_id]

            # Accumulate total students and FRL counts
            agg["total_students"] += float(area_data["ge_students"])
            agg["FRL"] += float(area_data["FRL"])
            for ethnicity_col, key in zip(AREA_ETHNICITIES, ETHNICITY_KEYS):
                agg[key] += float(area_data[ethnicity_col])

        # 4. Convert aggregated counts to proportions
        final_zone_demographics = {}
        for zone_id, data in zone_aggregates.items():
            total_students = data["total_students"]

            # Create a new dictionary for the final, proportional results
            final_demographics = {"total_students": round(total_students, 2)}

            if total_students > 0:
                # Calculate FRL proportion
                final_demographics["FRL"] = round(data["FRL"] / total_students, 2)

                # Calculate ethnicity proportions
                for key in ETHNICITY_KEYS:
                    final_demographics[key] = round(data[key] / total_students, 2)
            else:
                # Handle zones with zero students (FRL and ethnic proportions are 0)
                final_demographics["FRL"] = 0.0
                for key in ETHNICITY_KEYS:
                    final_demographics[key] = 0.0

            final_zone_demographics[zone_id] = final_demographics

        return final_zone_demographics

    def visualize_zones(self):
        zv = ZoneVisualizer('Block', self.is_local)
        zv.zones_from_dict(self.block_zone_dict, show_plot=True)

    def save_output(self, folder_path):
        # save the image, dict_solution, objective_value, status, wall_time
        save_path = os.path.expanduser(folder_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save zone dict as json file
        filename = os.path.join(save_path, "zone_dict.json")
        with open(filename, "w") as f:
            json.dump(self.zone_dict, f)

        boundary_cost = -1
        if self.zone_dict is not None and len(self.zone_dict) > 0:
            boundary_cost = self.get_boundary_cost()
            file_name = os.path.join(save_path, "zones_visualization")
            zv = ZoneVisualizer(self.dz.level, self.dz.is_local)
            zv.zones_from_dict(self.zone_dict, save_path=file_name)

        output_info = {
            "boundary_cost": boundary_cost,
            "status": self.status,
            "wall_time": self.wall_time
        }
        filename = os.path.join(save_path, "solution_info.json")
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

    if solution_output.status != 'INFEASIBLE':
        print("Objective value: ", solution_output.objective_value)
        print('Boundary cost: ', solution_output.get_boundary_cost())
        print('Wall time: ', solution_output.wall_time)
        demographics = solution_output.get_zone_demographics()
        print(compute_zone_deviations(demographics))
        solution_output.visualize_zones()
