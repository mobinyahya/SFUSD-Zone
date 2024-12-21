import math
import os, sys, yaml
import pandas as pd

sys.path.append('../..')
sys.path.append('../../summary_statistics')
import pickle
from Graphic_Visualization.zone_viz import ZoneVisualizer
# from IP_Zoning import DesignZones
from Zone_Generation.Optimization_IP.design_zones import DesignZones, load_zones_from_file, Compute_Name
from Zone_Generation.Optimzation_Heuristics.zone_eval import * # evaluate_assignment_score, Tuning_param, boundary_trimming, evaluate_contiguity
from Helper_Functions.ReCom import *
from Helper_Functions.abstract_geography import *
from Zone_Generation.Config.Constants import *
from Helper_Functions.Relaxed_ReCom import Relaxed_ReCom
from Zone_Generation.Optimization_IP.integer_program import Integer_Program
from Zone_Generation.Optimzation_Heuristics.local_search_zoning import *


def load_pkl_maps(config, name):
    with open(os.path.expanduser(config["path"] + name), 'rb') as file:
        return pickle.load(file)


class Generate_Request(object):
    def __init__(self, user_inputs):
        self.update_config(user_inputs)
        self.name = Compute_Name(self.config)
        self.path = "/Users/mobin/SFUSD/Visualization_Tool_Data/Thesis/Hierarchical/"




    def update_config(self, user_inputs):
        with open("../Config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.config["centroids_type"] = user_inputs["centroids_type"]
        self.config["frl_dev"] = user_inputs["frl_dev"]
        self.config["Z"] = user_inputs["number_of_zones"]


    def generate_aa_zones(self):
        self.config["level"] = "attendance_area"
        DZ = DesignZones(config=self.config)
        self.DZ = DZ

        IP = Integer_Program(DZ)
        IP._feasibility_const(max_distance=max_distance[self.config["Z"]])
        self.generate_zones(DZ, IP)

    def generate_bg_zones(self):
        self.config["level"] = "BlockGroup"

        DZ = DesignZones(self.config)
        self.DZ = DZ

        zv = ZoneVisualizer(self.config["level"])
        DZ.zone_dict = aa2bg_Zoning(DZ, self.AA_zd)
        zv.zones_from_dict(DZ.zone_dict)

        DZ.zone_dict = drop_boundary(DZ, DZ.zone_dict)
        zv.zones_from_dict(DZ.zone_dict)

        DZ.zone_dict = trim_noncontiguity(DZ, DZ.zone_dict)
        zv.zones_from_dict(DZ.zone_dict, save_path=self.path + "_" + self.config["level"] + "_Trimmed")
        self.save_pkl(DZ.zone_dict, name="_Trimmed")

        IP = Integer_Program(DZ)
        IP._feasibility_const(max_distance=max_distance[self.config["Z"]])
        initialize_preassigned_units(IP, DZ.zone_dict)
        self.generate_zones(DZ, IP)

    def generate_b_zones(self):
        self.config["level"] = "Block"

        DZ = DesignZones(self.config)
        self.DZ = DZ

        zv = ZoneVisualizer(self.config["level"])
        DZ.zone_dict = bg2b_Zoning(DZ, self.BG_zd)
        # zv.zones_from_dict(DZ.zone_dict)

        DZ.zone_dict = drop_boundary(DZ, DZ.zone_dict)
        zv.zones_from_dict(DZ.zone_dict)
        DZ.zone_dict = drop_boundary(DZ, DZ.zone_dict)
        zv.zones_from_dict(DZ.zone_dict)

        # DZ.zone_dict = trim_noncontiguity(DZ, DZ.zone_dict)
        # zv.zones_from_dict(DZ.zone_dict, save_path=self.path + "_" + self.config["level"] + "_Trimmed")
        self.save_pkl(DZ.zone_dict, name="_Trimmed")

        IP = Integer_Program(DZ)
        IP._feasibility_const(max_distance=max_distance[self.config["Z"]])
        initialize_preassigned_units(IP, DZ.zone_dict)
        self.generate_zones(DZ, IP)


    def generate_zones(self, DZ, IP):
        IP._set_objective_model()
        IP._shortage_const(shortage=self.config["shortage"], overage=self.config["overage"],
                           all_cap_shortage=self.config["all_cap_shortage"])
        IP._contiguity_const()
        IP._diversity_const(racial_dev=self.config["racial_dev"], frl_dev=self.config["frl_dev"])
        IP._school_count_const()

        solve_success = DZ.solve(IP)
        if  self.config["level"] == "attendance_area":
            self.solve_AA_success = solve_success
            self.AA_zd = DZ.zone_dict
        if  self.config["level"] == "BlockGroup":
            self.solve_BG_success = solve_success
            self.BG_zd = DZ.zone_dict
        if  self.config["level"] == "Block":
            self.solve_B_success = solve_success
            self.B_zd = DZ.zone_dict

        if solve_success:
            # DZ.save(path=self.path, name=self.name +"_" +SUFFIX[self.config["level"]])
            print("Resulting zone dictionary: ", DZ.zone_dict)
            zv = ZoneVisualizer(self.config["level"])
            zv.zones_from_dict(DZ.zone_dict, save_path = self.path + "_" + self.config["level"])

    def save_pkl(self, zone_dict, name=""):
        zone_dict_pkl = {}
        for key,value in zone_dict.items():
            zone_dict_pkl[key] = self.DZ.centroid_sch[value]

        with open(os.path.expanduser(self.path + "Hierarchical"+ "_" + self.config["level"]+ name + ".pkl" ), 'wb') as file:
            pickle.dump(zone_dict_pkl, file)



if __name__ == "__main__":

    old_capacities = pd.read_csv('/Users/mobin/Desktop/untitled/Old_Capacities.csv')
    closure = pd.read_csv('/Users/mobin/Desktop/untitled/Closure.csv')


    # Create a unique identifier for each program
    old_capacities['program_id'] = old_capacities['SchNum'].astype(str) + '_' + old_capacities['PathwayCode']
    closure['program_id'] = closure['SchNum'].astype(str) + '_' + closure['PathwayCode']

    # Merge the dataframes
    merged = pd.merge(old_capacities,
                      closure[['program_id', 'Scenario_Closure_Capacity']],
                      on='program_id',
                      how='outer',
                      indicator=True)

    # Fill Scenario_Closure_Capacity with 0 for rows only in old_capacities
    merged['Scenario_Closure_Capacity'] = merged['Scenario_Closure_Capacity'].fillna(0).astype(int)

    # For rows only in closure, fill other columns from closure dataframe
    closure_only = merged['_merge'] == 'right_only'
    columns_to_fill = ['SchNum', 'SchoolName', 'Grade', 'PathwayCode']
    for col in columns_to_fill:
        merged.loc[closure_only, col] = closure.loc[closure['program_id'].isin(merged.loc[closure_only, 'program_id']), col].values

    # Fill remaining NaN values with 0 for numeric columns, and '' for string columns
    numeric_columns = merged.select_dtypes(include=['int64', 'float64']).columns
    merged[numeric_columns] = merged[numeric_columns].fillna(0)
    string_columns = merged.select_dtypes(include=['object']).columns
    merged[string_columns] = merged[string_columns].fillna('')


    # closure = closure.loc[closure["Type"] != "Citywide"] #60 without, 51 with this line
    closure = closure[closure['Scenario_Closure_Capacity'] != 0]
    schools = closure["SchNum"]
    print(len(set(schools)))
    exit()

    # Drop the temporary columns
    merged = merged.drop(['program_id', '_merge'], axis=1)

    # Reorder columns to match the desired output
    column_order = [col for col in old_capacities.columns if col != 'program_id'] + ['Scenario_Closure_Capacity']
    merged = merged[column_order]

    # Save the merged dataframe to a new CSV file
    merged.to_csv('/Users/mobin/Desktop/untitled/Merged_Capacities.csv', index=False)


    print("Hi")
    exit()
    # with open("../Config/config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    # zv = ZoneVisualizer(config["level"])
    #
    # path_name  = "/Users/mobin/Desktop/CandidateZones/18-zone-1_2_BG"
    # # path_name = "/Users/mobin/Documents/sfusd/local_runs/Zones/concept1zones"
    # dz = DesignZones(config=config)
    # zone_lists, zone_dict = load_zones_from_file(path_name+".csv")
    #
    # zv.zones_from_dict(zone_dict, centroid_location=dz.centroid_location, save_path=path_name+"_id")
    #
    #
    #
    # # with open(os.path.expanduser("/Users/mobin/Desktop/59-zone_B.pkl"), 'rb') as file:
    # #     zd = pickle.load(file)
    # # zv.zones_from_dict(zd)



    """ must vs buffer units for recursive """
   # for z in dz.centroids:
   #     for area in dz.area2idx:
   #         idx = dz.area2idx[area]
   #         if dz.euc_distances[z][idx] < 1.2:
   #             zd[area] = 625
   # for z in dz.centroids:
   #     for area in dz.area2idx:
   #         if area in zd:
   #             continue
   #         idx = dz.area2idx[area]
   #         if dz.euc_distances[z][idx] < 3:
   #             zd[area] =  834
   # print("zd ", zd)

    """ Transforming a .csv zones into pkl, without knowing the centroid order """
    # centroid_indices = {}
    # zl, old_zd = load_zones_from_file(config["path"] + "6-zone-9_1_BG.csv")
    # print("old_zd ", old_zd)
    # for z in range(len(dz.centroids)):
    #     sch = dz.centroid_sch[z]
    #     sch_idx = dz.centroids[z]
    #     sch_area = dz.idx2area[sch_idx]
    #     if sch_area not in old_zd:
    #         print("Error sch_area ", sch_area, "  sch_idx", sch_idx , "  sch", sch)
    #     else:
    #         centroid_indices[old_zd[sch_area]] = sch
    # zd = {key: centroid_indices[value] for key,value in old_zd.items()}
    # print("centroid_indices ", centroid_indices)


    # zd = {}
    # for idx in dz.idx2area:
    #     area = dz.idx2area[idx]
    #     if area not in old_zd:
    #         continue
    #     z = old_zd[area]
    #     zd[area] = z
    #     for n in dz.neighbors[idx]:
    #         n_area = dz.idx2area[n]
    #         if n_area in zd:
    #             continue
    #         zd[n_area] = z
    # zv.zones_from_dict(zd, centroid_location=dz.centroid_location, save_path=config["path"] + "temp") #_buffer_optimized
    #
    #
    # zd_idx = {key: dz.centroid_sch.index(value) for key,value in zd.items()}
    # zd_idx = trim_noncontiguity(dz, zd_idx)
    # zd = {key: dz.centroid_sch[value] for key,value in zd_idx.items()}
    #


    """ arbitrary cut through the mcmc"""
    # for idx in dz.idx2area:
    #     area = dz.idx2area[idx]
    #     zd[area] = old_zd[area]
    #     if area not in old_zd:
    #         continue
    #     if old_zd[area] not in [750]:
    #         continue
    #     count = 0
    #     for n in dz.neighbors[idx]:
    #         n_area = dz.idx2area[n]
    #         if n_area not in zd:
    #             continue
    #         if old_zd[n_area] == 435:
    #             count+=1
    #         else:
    #             count-=1
    #     if count > 1:
    #         zd[area] = 435

    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    zv = ZoneVisualizer(config["level"])
    dz = DesignZones(config=config)

    zd = load_pkl_maps(config, name = "MCMC_13001-Balance.pkl")
    print(set(zd.values()))
    # exit()
    zv.zones_from_dict(zd) #_buffer_optimized

    # exit()
    # zd = {}
    # for idx in dz.idx2area:
    #     area = dz.idx2area[idx]
    #
    #     for n in dz.neighbors[idx]:
    #         n_area = dz.idx2area[n]
    #         if n_area not in zd:
    #             continue
    #
    #
    # # # edited_zd = {key: value for key, value in zd.items() if value not in [539, 848]} #750
    # zv.zones_from_dict(zd, centroid_location=dz.centroid_location, save_path=config["path"] + "MCMC_Balance_15000") #_buffer_optimized
    # with open(os.path.expanduser(config["path"] + "6-zone-9_1_BG" + ".pkl"), 'wb') as file:
    #     pickle.dump(zd, file)
    # print("dz.student_df[af_students] " , dz.area_data["af_students"].sum())
    # zv.population_heatmap(dz.area_data, save_path=config["path"] + "MCMC_Balance_15000")

    # zv.zones_from_dict(zd, centroid_location=dz.centroid_location, save_path=config["path"] + "AF-relocated") #_buffer_optimized


