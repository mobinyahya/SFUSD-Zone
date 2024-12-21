import math
import os, sys, yaml
from pathlib import Path

sys.path.append('../..')
sys.path.append('../../summary_statistics')
import pickle
from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimization_IP.design_zones import DesignZones,  Compute_Name
from Helper_Functions.ReCom import *
from Helper_Functions.abstract_geography import *
from Zone_Generation.Config.Constants import *
from Zone_Generation.Optimization_IP.integer_program import Integer_Program
from Zone_Generation.Optimzation_Heuristics.local_search_zoning import *




class Generate_Request(object):
    def __init__(self, user_inputs):
        self.update_config(user_inputs)
        self.name = Compute_Name(self.config)
        self.path = " /Users/mobin/Documents/sfusd/local_runs/Zones/School_Closure/"
        self.path = os.path.abspath(self.path.strip())




    def update_config(self, user_inputs):
        with open("../Config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.config["shortage"] = user_inputs["shortage"]
        self.config["centroids_type"] = user_inputs["centroids_type"]
        self.config["frl_dev"] = user_inputs["frl_dev"]
        self.config["Z"] = user_inputs["number_of_zones"]
        self.solve_BG_success = False
        self.solve_AA_success = False


    def generate_aa_zones(self):
        self.config["level"] = "attendance_area"
        DZ = DesignZones(config=self.config)
        IP = Integer_Program(DZ)
        IP._feasibility_const(max_distance=max_distance[self.config["Z"]])
        self.generate_zones(DZ, IP)


    def generate_bg_zones(self):
        self.config["level"] = "BlockGroup"

        DZ = DesignZones(self.config)

        zv = ZoneVisualizer(self.config["level"])
        DZ.zone_dict = aa2bg_Zoning(DZ, self.AA_zd)
        zv.zones_from_dict(DZ.zone_dict)

        DZ.zone_dict = drop_boundary(DZ, DZ.zone_dict)
        zv.zones_from_dict(DZ.zone_dict)

        DZ.zone_dict = trim_noncontiguity(DZ, DZ.zone_dict)
        zv.zones_from_dict(DZ.zone_dict)

        IP = Integer_Program(DZ)
        IP._feasibility_const(max_distance=max_distance[self.config["Z"]])
        initialize_preassigned_units(IP, DZ.zone_dict)
        self.generate_zones(DZ, IP)

    def generate_zones(self, DZ, IP):
        IP._set_objective_model()
        IP._shortage_const(shortage=self.config["shortage"], overage=self.config["overage"],
                           all_cap_shortage=self.config["all_cap_shortage"])
        IP._contiguity_const()
        IP._diversity_const(frl_dev=self.config["frl_dev"])
        IP._school_count_const()

        solve_success = DZ.solve(IP)
        if  self.config["level"] == "attendance_area":
            print("solve_success ", solve_success)
            self.solve_AA_success = solve_success
            self.AA_zd = DZ.zone_dict

        if self.config["level"] == "BlockGroup":
            self.solve_BG_success = solve_success

        if solve_success:
            # DZ.save(path=self.path, name=self.name +"_" +SUFFIX[self.config["level"]])
            # print("Resulting zone dictionary: ", DZ.zone_dict)

            # if self.config["level"] == "BlockGroup":
            self.zone_dict = DZ.zone_dict
            self.DZ = DZ
            self.objevtive = IP.m.objVal
            print("IP.m.objVal ", IP.m.objVal)

    def save_pkl(self):
        zv = ZoneVisualizer(self.config["level"])
        file_name = self.name + "_" + self.config["level"] + "_" + str(self.config["frl_dev"]) + "_" + str(self.config["shortage"])
        # Create the full file path
        full_path = os.path.join(self.path, file_name)
        zv.zones_from_dict(self.DZ.zone_dict, save_path= full_path)

        zone_dict_pkl = {}
        for key,value in self.zone_dict.items():
            zone_dict_pkl[key] = self.DZ.centroid_sch[value]
        print("zone_dict_pkl ", zone_dict_pkl)
        with open(full_path+".pkl", 'wb') as file:
            pickle.dump(zone_dict_pkl, file)

    def load_pkl_aa(self):
        file_name = self.name + "_" + self.config["level"] + "_" + str(self.config["frl_dev"]) + "_" + str(self.config["shortage"])
        # Create the full file path
        full_path = os.path.join(self.path, file_name)
        if not os.path.exists(full_path + ".pkl"):
            return False
        with open(full_path + ".pkl", 'rb') as file:
            zone_dict_pkl = pickle.load(file)
        self.DZ = DesignZones(self.config)

        self.AA_zd = {}
        print("elf.DZ.centroid_sch", self.DZ.centroid_sch)
        for key,value in zone_dict_pkl.items():
            self.AA_zd[key] = self.DZ.centroid_sch.index(value)

        print("zone_dict_pkl ", zone_dict_pkl)
        return True

if __name__ == "__main__":
    inputs = {}
    # inputs["frl_dev"] = 0.25
    # inputs["centroids_type"] = "18-zone"
    # inputs["number_of_zones"] = int(inputs["centroids_type"].split("-")[0])
    # inputs["shortage"] = shortage_ratio[inputs["number_of_zones"]]

    # GR = Generate_Request(inputs)
    # GR.generate_aa_zones()
    # if GR.solve_AA_success:
    #     GR.generate_bg_zones()
    #     GR.save_pkl()

    # with open(os.path.expanduser(GR.path + "Objective_1130.0_4-zone-3"), 'rb') as file:
    #     zd = pickle.load(file)
    #     zv = ZoneVisualizer("BlockGroup")
    #     zv.zones_from_dict(zd)


    # with open("../Config/automatic_centroids.yaml", "r") as f:
    #     centroid_options = yaml.safe_load(f)
    #     for centroids in centroid_options:
    #         # solved until 10-zones-7, frl:0.3
    #         for frl_dev in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
    #         # for frl_dev in [0.15]:
    #             inputs["frl_dev"] = frl_dev
    #             print("frl: ", frl_dev, "  centroid: ", centroids)
    #             inputs["number_of_zones"] = int(centroids.split("-")[0])
    #             inputs["shortage"] = shortage_ratio[inputs["number_of_zones"]]
    #             inputs["centroids_type"] = centroids
    #
    #             GR = Generate_Request(inputs)
    #             GR.generate_aa_zones()
    #             if GR.solve_AA_success:
    #                 GR.generate_bg_zones()
    #                 GR.save_pkl()

    with open("../Config/school_closure_centroids.yaml", "r") as f:
        centroid_options = yaml.safe_load(f)
        for shortage in [0.22, 0.25]:
            inputs["shortage"] = shortage
            # solved until 10-zones-7, frl:0.3
            # for frl_dev in [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
            for frl_dev in [0.15, 0.2]:
                inputs["frl_dev"] = frl_dev
                for centroids in centroid_options:
                    print("frl: ", frl_dev, "  centroid: ", centroids)
                    inputs["number_of_zones"] = int(centroids.split("-")[0])
                    if inputs["number_of_zones"] != 6:
                        frl_dev += 0.02
                    inputs["centroids_type"] = centroids
                    GR = Generate_Request(inputs)
                    if not GR.load_pkl_aa():
                        GR.generate_aa_zones()
                        if GR.solve_AA_success:
                            GR.save_pkl()
                            GR.generate_bg_zones()
                    else:
                        GR.generate_bg_zones()
                    if GR.solve_BG_success:
                        GR.save_pkl()

# bug: frl:  0.25   centroid:  8-zone-26 no solution, but
#           frl:  0.2  centroid:  8-zone-26 has solution!