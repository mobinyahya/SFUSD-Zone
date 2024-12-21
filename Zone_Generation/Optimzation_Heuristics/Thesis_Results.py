import math
import os, sys, yaml
from pathlib import Path

sys.path.append('../..')
sys.path.append('../../summary_statistics')
import pickle
from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimization_IP.design_zones import DesignZones, Compute_Name
from Helper_Functions.ReCom import *
from Helper_Functions.abstract_geography import *
from Zone_Generation.Config.Constants import *
from Zone_Generation.Optimization_IP.integer_program import Integer_Program
from Zone_Generation.Optimzation_Heuristics.local_search_zoning import *


class Generate_Request(object):
    def __init__(self, user_inputs):
        self.update_config(user_inputs)
        self.name = (str(self.config["centroids_type"]) + "_frl_" + str(self.config["frl_dev"])
                     + "_shortage_" + str(self.config["shortage"]))
        self.path = "/Users/mobin/SFUSD/Visualization_Tool_Data/Thesis/Results_Search/"

    def update_config(self, user_inputs):
        with open("../Config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.config["centroids_type"] = user_inputs["centroids_type"]
        self.config["shortage"] = user_inputs["shortage"]
        self.config["frl_dev"] = user_inputs["frl_dev"]
        self.config["Z"] = user_inputs["number_of_zones"]

    def generate_aa_zones(self):
        self.config["level"] = "attendance_area"
        DZ = DesignZones(config=self.config)
        IP = Integer_Program(DZ)
        IP._feasibility_const(max_distance=max_distance[self.config["Z"]])

        IP._set_objective_model()
        IP._shortage_const(shortage=self.config["shortage"], overage=self.config["overage"],
                           all_cap_shortage=self.config["all_cap_shortage"])
        IP._contiguity_const()
        IP._diversity_const(racial_dev=self.config["racial_dev"], frl_dev=self.config["frl_dev"])
        IP._school_count_const()

        solve_success = DZ.solve(IP)
        self.solve_AA_success = solve_success

        if solve_success:
            # DZ.save(path=self.path, name=self.name +"_" +SUFFIX[self.config["level"]])
            print("Resulting zone dictionary: ", DZ.zone_dict)
            self.zone_dict = DZ.zone_dict
            self.DZ = DZ

            zv = ZoneVisualizer(self.config["level"])
            zv.zones_from_dict(DZ.zone_dict, save_path=self.path + "_" + self.name +"_AA")


    def save_pkl(self):
        zone_dict_pkl = {}
        for key, value in self.zone_dict.items():
            zone_dict_pkl[key] = self.DZ.centroid_sch[value]

        with open(os.path.expanduser(self.path + "_" + self.name +"_AA.pkl"), 'wb') as file:
            pickle.dump(zone_dict_pkl, file)


if __name__ == "__main__":

    inputs = {}

    with open("../Config/automatic_centroids.yaml", "r") as f:
        centroid_options = yaml.safe_load(f)
        for centroids in centroid_options:
            inputs["number_of_zones"] = int(centroids.split("-")[0])
            inputs["centroids_type"] = centroids
            if inputs["number_of_zones"] != 10:
                continue

            search = [(0.15, 1)]
            # search = [(0.24, 1), (0.27, 1), (1, 0.1), (1, 0.07)]
            # search = [(0.15, 1), (0.05, 1), (0.025, 1)]
            for (frl_dev, shortage) in search:
                inputs["frl_dev"] = frl_dev
                inputs["shortage"] = shortage

                GR = Generate_Request(inputs)
                print(GR.name)
                GR.generate_aa_zones()
                if GR.solve_AA_success:
                    print("solved for frl: ", frl_dev, "  centroid: ", centroids)
                    GR.save_pkl()
                # else:
                #     break
