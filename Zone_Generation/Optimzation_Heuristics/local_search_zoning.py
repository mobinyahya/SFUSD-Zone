import math
import os, sys, yaml

sys.path.append('../..')
sys.path.append('../../summary_statistics')
import pickle
from Graphic_Visualization.zone_viz import ZoneVisualizer
# from IP_Zoning import DesignZones
from Zone_Generation.Optimization_IP.generate_zones import DesignZones, load_zones_from_file, Compute_Name
from Zone_Generation.Optimzation_Heuristics.zone_eval import * # evaluate_assignment_score, Tuning_param, boundary_trimming, evaluate_contiguity
from Helper_Functions.ReCom import *
from Helper_Functions.abstract_geography import *
from Zone_Generation.Config.Constants import *
from Helper_Functions.Relaxed_ReCom import Relaxed_ReCom

# in this code, we run the iterative local search algorithm, to get better zone assignments
# Assumptions:
# 1- This local Search is only for blockgroup level
# 2- The starting point of Local Search (initialization step), is a zoning solution in attendance area level
#    we then convert this zoning, to its equivalent blockgroup level. And start iterative local search from there



def compute_samezone_pairs(dz, zone_dict):
    samezone_pairs = set()
    samezone_pairs.add((-1, -1))
    for i in range(dz.A):
        bg_i = dz.idx2area[i]
        for j in dz.neighbors[i]:
            bg_j = dz.idx2area[j]
            if i >= j:
                continue
            if (bg_j in zone_dict) and (bg_i in zone_dict):
                if zone_dict[bg_j] == zone_dict[bg_i]:
                    samezone_pairs.add((i, j))
    return samezone_pairs



def iterative_lp():
    linear_solution = True
    if linear_solution:
        with open("../Config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        zv = ZoneVisualizer('BlockGroup')

        rounded_zd = {}
        for i in range(0, 4):
            dz = DesignZones(M=3, level='BlockGroup', centroidsType=0, include_citywide=True)
            print("linear rounding, iteration number " + str(i))

            dz.setmodel(shortage=config["shortage"], balance=config["balance"])
            # print("check initialization")
            # print(dz.area2idx[60750301013])
            # dz.m.addConstr(dz.x[dz.area2idx[60750301013],1] == 0)
            for bg in rounded_zd:
                print("not in the loop")
                for z in range(dz.M):
                    if (z == rounded_zd[bg]):
                        print("add blockgroup of  " + str(bg) + " to the zone of  " + str(z))
                        dz.m.addConstr(dz.x[dz.area2idx[bg], z] == 1)
                        # dz.m.addConstr(dz.x[dz.centroids[z], z] == 1)
                        # print("comparison, is " + str(dz.area2idx[bg]) + "  equal to  " +  str(dz.centroids[z]))
                    # else:
                    #     dz.m.addConstr(dz.x[dz.area2idx[bg],z] == 0)
            # if i == 0:
            dz.addGeoConstraints(maxdistance=-1, realDistance=True, coverDistance=False, strong_contiguity=True, neighbor=False, boundary=True)
            dz._add_diversity_constraints(lbfrl=config["lbfrl"], ubfrl=config["ubfrl"])

            dz.solve()
            print("this is the resulting zone dictionary dz.zd")
            print(dz.zone_dict)
            rounded_zd = dict(dz.zone_dict)
            # evaluate_assignment_score(dz, rounded_zd, shortage_limit = param.shortage, balance_limit = param.balance, lbfrl_limit= param.lbfrl, ubfrl_limit= param.ubfrl)

            zv.zones_from_dict(dz.zone_dict, show=True, label=False)
            del dz



# def load_rec_maps(dz):
#     zd = {}
#     for block, zone_numb_sch in load_zd.items():
#         if zone_numb_sch not in dz.centroid_sch:
#             continue
#         # if zone_numb_sch == 670:
#         #     if dz.euc_distances.loc[dz.sch2b[zone_numb_sch], str(block)] > 0.65:
#         #        continue
#         zone_idx = dz.centroid_sch.index(zone_numb_sch)
#         zd[block] = zone_idx
#     return zd

rec_path = "/Users/mobin/SFUSD/Visualization_Tool_Data/Rec_Zones/"
def load_recursive_maps(dz, name):
    pkl_path = os.path.expanduser(rec_path + name + ".pkl")
    with open(pkl_path, 'rb') as file:
    # with open(rec_path + name + ".pkl", 'rb') as file:
        zone_sch_dict = pickle.load(file)

    zd = {}
    for block, zone_sch in zone_sch_dict.items():
        zd[block] = zone_sch / 50
    print("loaded zone_sch_dict = ", zone_sch_dict)


    zv = ZoneVisualizer("Block")
    zv.zones_from_dict(zd, centroid_location=dz.centroid_location, save_path=rec_path + name + "_B")

    exit()
    pkl_path = os.path.expanduser(rec_path + name + ".pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(zd, file)

    return zd
def load_initial_assignemt(dz, name, path, load_level='attendance_area'):
    if load_level == "attendance_area":
        aa_zl, aa_zd = load_zones_from_file(path + name + "_AA.csv")

        aa_zv = ZoneVisualizer('attendance_area')
        # aa_zv.visualize_zones_from_dict(aa_zoning, centroid_location=dz.centroid_location, save_name= name + "_AA")
        aa_zv.zones_from_dict(aa_zd, centroid_location=dz.centroid_location)
        # dz.distance_and_neighbors_telemetry(centroid_choices)

        dz.zone_dict = aa2bg_Zoning(dz, aa_zd)

    elif load_level == "BlockGroup":
        dz.zone_lists, dz.zone_dict = load_zones_from_file(path + name + "_BG.csv")
    elif load_level == "Block":
        dz.zone_lists, dz.zone_dict = load_zones_from_file(path + name + "_B.csv")
        # TODO 59zone mode
        # zd = {}
        # east_schools = [453, 625, 830, 507, 513, 521, 867]
        # west_schools = [544, 664, 750,
        #                 782, 413, 435,
        #                 862, 539, 848]
        golden_gate_sch = [644]

        rec_zone_maps = {}
        for block, zone_numb in dz.zone_dict.items():
            zone_numb_sch = dz.centroid_sch[zone_numb]
            if zone_numb_sch in golden_gate_sch:
                rec_zone_maps[block] = zone_numb_sch
        print("rec_zone_maps = ", rec_zone_maps)

        zv = ZoneVisualizer("Block")

        vis_zd = {key: all_schools.index(value) for key, value in rec_zone_maps.items()}
        zv.zones_from_dict(vis_zd, centroid_location=dz.centroid_location)

        rec_path = "/Users/mobin/SFUSD/Visualization_Tool_Data/Rec_Zones/"
        with open(os.path.expanduser(rec_path + "zone-init-total-loaded_zd.pkl"), 'rb') as file:
            loaded_szd = pickle.load(file)
            loaded_szd = {key: value for key, value in loaded_szd.items()}
            vis_zd = {key: all_schools.index(value) for key, value in loaded_szd.items()}
            zv.zones_from_dict(vis_zd, centroid_location=dz.schools_locations)

        for block,zone_sch in  loaded_szd.items():
            rec_zone_maps[block] = zone_sch
        vis_zd = {key: all_schools.index(value) for key, value in rec_zone_maps.items()}
        zv.zones_from_dict(vis_zd, centroid_location=dz.schools_locations, save_path=config["path"] + "zones-init-total-x" + SUFFIX["BlockGroup"])

        save_path = os.path.expanduser(rec_path + "zones-init-total-x.pkl")
        with open(save_path, 'wb') as file:
            pickle.dump(rec_zone_maps, file)

        exit()
        dz.zone_dict = rec_zone_maps
    else:
        raise ValueError("Invalid Input Level")

    return dz


def aa2bg_Zoning(dz, aa_zd):
    blockgroup_zoning = {}
    # For the given attendance area level zoning, find it's equivalent assignment on blockgroup level.
    for bg in dz.area2idx:
        if bg in dz.bg2att:
            if dz.bg2att[bg] in aa_zd:
                blockgroup_zoning[bg] = aa_zd[dz.bg2att[bg]]
            else:
                print("BG ", bg, " is in AA ", dz.bg2att[bg], " which is not included in aa zoning")


    for bg in dz.area2idx:
        neighbor_count = {}
        # look over all neighbors that are assigned to a zone so far
        for neighbor in dz.neighbors[dz.area2idx[bg]]:
            if dz.idx2area[neighbor] in blockgroup_zoning:
                # count how many neighbors of this blockgroup are from each different zone
                if blockgroup_zoning[dz.idx2area[neighbor]] in neighbor_count:
                    temp = neighbor_count[blockgroup_zoning[dz.idx2area[neighbor]]]
                    neighbor_count[blockgroup_zoning[dz.idx2area[neighbor]]] = temp + 1
                else:
                    neighbor_count[blockgroup_zoning[dz.idx2area[neighbor]]] = 1

    # # not just count each neighbor that is attached to the similar zone center, but actually
    # # look at neighbors that are closer to the centroid, and count such neighbors (and do this for each centroid)
    # for blockgroup in dz.area2idx:
    #     #neighbor count will have: for each zone center c, how many neighbors of blockgroup
    #     # that are closer to c, are also assigned to c
    #     neighbor_count = {}
    #     for z in range(dz.M):
    #         closer_neighbors = dz.closer_euc_neighbors[dz.area2idx[blockgroup], dz.centroids[z]]
    #         print("for zone " + str(z) + " list of closer_neighbors:")
    #         print(closer_neighbors)
    #         for neighbor in closer_neighbors:
    #             if dz.idx2area[neighbor] in blockgroup_zoning:
    #                 if blockgroup_zoning[dz.idx2area[neighbor]] == z:
    #                     print("neighbor")
    #                     print(neighbor)
    #                     # count how many neighbors of this blockgroup are from each different zone
    #                     if z in neighbor_count:
    #                         temp = neighbor_count[z]
    #                         neighbor_count[z] = temp + 1
    #                     else:
    #                         neighbor_count[z] = 1
# todo check how it happened that blockgroups were not in the bg2att

        # if all neighbors of this blockgroup are in the same zone
        # assign this blockgroup to the same zone (even if based on bg2att, we had a different outcome)
        if (bg in dz.bg2att):
            if len(neighbor_count) == 1:
                # make sure this blockgroup is not the centroid
                if bg not in [dz.idx2area[dz.centroids[z]] for z in range(dz.M)]:
                    # select the first key in neighbor_count (we have made sure neighbor_count has only 1 key,
                    # which is the zone #of all neighbors of this blockgroup)
                    blockgroup_zoning[bg] = list(neighbor_count.keys())[0]

        # some blockgroups might have been missed in bg2att dict
        # (One of the reasons: this block might not have any students --> is not included in the data, that we compute bg2att based on)
        # else:
        #     # find which zone was most common among neighbors
        #     max_zone = -1
        #     if len(neighbor_count) == 0:
        #         print("size of neighbor count is 0, Error")
        #         print(blockgroup)
        #     for z in neighbor_count:
        #         if max_zone == -1:
        #             max_zone = z
        #         else:
        #             if neighbor_count[z] > neighbor_count[max_zone]:
        #                 max_zone = z
        #     # assign the same zone number that most of the neighbors have to this blockgroup
        #     blockgroup_zoning[blockgroup] = max_zone


    # if evaluate_contiguity(dz, blockgroup_zoning) == False:
    #     print("Pre-initalization is not strongly contiguis. We smooth the boundaries further, so that we get a strongly contiguis zone assignment in blockgroup level")
    # if evaluate_contiguity(dz, blockgroup_zoning) == True:
    #     print("strong contiguity test is satisfied")
    return blockgroup_zoning


# iteratively update the zoning using spanning tree graph cut.
# in each iteration:
def spanning_tree_cut(dz, config):
    geograph, shuffle_zones = construct_shuffle_geograph(zones=dz.zone_lists, area2idx=dz.area2idx, idx2area=dz.idx2area, neighbors=dz.neighbors)
    for i in range(1):
        spanning_tree = random_spanning_tree(geograph)
        evaluation_cost = random_cut_evaluation(dz, spanning_tree, shuffle_zones, config)
        if evaluation_cost < 1000000:
            break

# iteratively update the zoning. in each iteration:
# for each zone, go over all its neighbors, and for each of them, if adding it to that zone would increase the total objective value, change the zoning for that blockgroup
# otherwise, don't do anything and continue to the another neighbor of that zone
def iterative_update(config, dz, zone_dict, ZoneVisualizer, noiterations):
    for iteration in range(0, noiterations):
        # bcf: boundary cost fraction of this iteration
        iter_bcf = (iteration + 1)/noiterations
        # iter_dl = 0.15 + (noiterations-1 - iteration)/(noiterations-1) * 0.85
        # diversity limit of this iteration
        # iter_dl = 0
        # we find the value of the current assignment (given the ratio of boundary_cost)
        curnt_val = evaluate_assignment_score(config, dz, zone_dict, boundary_cost_fraction = iter_bcf)
        # ratio of boundary_cost tells us what fraction of boundary_coeff, should we consider
        # the idea is that in first rounds, the boundary cost should not be that important
        # -- so that it allows for more movement on the boundary of local search
        # and as time goes, we increase this cost, resulting in zones with more smooth boundary for the end result
        print("bcf value at the begining of iteration " + str(iteration) + "  is: " + str(curnt_val))
        if curnt_val > 1000000000000:
            print("ERROR")


        # randomly select two zones
        random.sample(range(100), 5)

        # if selected zones have less than a fixed shared boundary,
        # go back and re-select two new zones.


        for level_area in zone_dict:
            for neighbor_idx in dz.neighbors[dz.area2idx[level_area]]:
                neighbor = dz.idx2area[neighbor_idx]
                # make sure the neighbor and level_area(aa/bg/b) are on the boundaries of their zones,
                # and belong to different zones
                if zone_dict[level_area] != zone_dict[neighbor]:
                    temp = zone_dict[neighbor]
                    zone_dict[neighbor] = zone_dict[level_area]
                    new_val = evaluate_assignment_score(config, dz, zone_dict, boundary_cost_fraction = iter_bcf)
                    if new_val <= curnt_val:
                        curnt_val = new_val
                        # print("new curnt_val " + str(curnt_val))
                    else:
                        # changing the boundaries in this direction would decrease the objective value so reverse back
                        zone_dict[neighbor] = temp
        print("bcf alue at the end of iteration " + str(iteration) + "  is: " + str(evaluate_assignment_score(config, dz, zone_dict, boundary_cost_fraction = iter_bcf)))
        print("Total value at the end of iteration " + str(iteration) + "  is: " + str(evaluate_assignment_score(config, dz, zone_dict)))
        ZoneVisualizer.zones_from_dict(zone_dict, centroid_location= dz.centroid_location)
    return zone_dict



def initialize_preassigned_units(dz, zone_dict):
    for j in range(dz.A):
        for z in range(dz.M):
            if j not in dz.valid_area_per_zone[z]:
                continue
            bg_j = dz.idx2area[j]
            if bg_j in zone_dict:
                if zone_dict[bg_j] == z:
                    dz.m.addConstr(dz.x[j, z] == 1)
                else:
                    dz.m.addConstr(dz.x[j, z] == 0)


def handle_anomolies(dz, zone_dict):
    for j in range(dz.A):
        level_j = dz.idx2area[j]
        if level_j not in zone_dict:
            if j in dz.centroids:
                continue
            neighbors = dz.neighbors [j]
            if len(neighbors) >= 1:
                for neighbor_idx in neighbors:
                    if dz.idx2area[neighbor_idx] in zone_dict:
                        closer_neighbors = dz.closer_euc_neighbors[j, dz.centroids[zone_dict[dz.idx2area[neighbor_idx]]]]
                        if len(closer_neighbors) == 0:
                            zone_dict[level_j] = zone_dict[dz.idx2area[neighbor_idx]]
                            break

def heuristic(dz, zone_dict):
    for i in range(5):
        for j in range(dz.A):
            level_j = dz.idx2area[j]
            if level_j not in zone_dict:
                if j in dz.centroids:
                    continue
                for z in range(dz.M):
                    c = dz.centroids[z]
                    closer_neighbors = dz.closer_euc_neighbors[j, c]
                    if len(closer_neighbors) >= 1:
                        for neighbor_idx in closer_neighbors:
                            if dz.idx2area[neighbor_idx] in zone_dict:
                                if level_j == 60750124022013:
                                    print("lll zone center" + str(z))
                                    print(dz.idx2area[neighbor_idx])
                                    print(zone_dict[dz.idx2area[neighbor_idx]])
                                if zone_dict[dz.idx2area[neighbor_idx]] == z:
                                    zone_dict[level_j] = z
                                    break

def sanity_check(dz):
    print("Sanity Check")
    for j in range(dz.A):
        level_j = dz.idx2area[j]
        if j in dz.centroids:
            continue
        count = 0
        for c in dz.centroids:
                closer_neighbors = dz.closer_euc_neighbors[j, c]
                if len(closer_neighbors) >= 1:
                    count = 1
        if count == 0:
            print("******")
            print(level_j)
def update_z(dz, zone_dict):
    dz.zone_lists = []
    for z in range(dz.M):
        dz.zone_lists.append([])
    for b in zone_dict:
        dz.zone_lists[zone_dict[b]].append(b)

def local_search(config):
    name = Compute_Name(config)
    print("name: " + str(name))
    # if os.path.exists(config["path"] + name + "_BG" +".csv"):
    #     return

    # if there is no png, it means it was either not considered yet, or
    # we looked for solution and non exists
    # if not os.path.exists(param.path + name + "_AA" +".png"):
    #     return

    # input_level = 'BlockGroup'
    input_level = 'Block'
    dz = DesignZones(
        config=config,
        level=input_level,
    )
    # dz = load_initial_assignemt(dz, path=config["path"], name=name, load_level="attendance_area")
    # dz = load_initial_assignemt(dz, path=config["path"], name=name, load_level="Block")
    # dz.zone_dict = load_recursive_maps(dz=dz, name=name)
    # dz.zone_dict = load_rec_maps(dz)
    local_search_iteration(dz, config, name, input_level)
    # for new_shortage in [0.35, 0.29, 0.24]:
    #     if new_shortage < param.shortage:
    #         print("checking for shortage: " + str(new_shortage))
    #         param.shortage = new_shortage
    #         name = Compute_Name(param)
    #         if not os.path.exists(param.path + name + "_AA" + ".png"):
    #             local_search_iteration(dz, param, name, input_level)


def local_search_iteration(dz, config, name, input_level):
    print("New round of Local Search")
    zv = ZoneVisualizer(input_level)

    rec_path = "/Users/mobin/SFUSD/Visualization_Tool_Data/Rec_Zones/"
    with open(os.path.expanduser(rec_path + "6-zone-2x_loaded_zd.pkl"), 'rb') as file:
        loaded_szd = pickle.load(file)
        loaded_szd = {key: value for key, value in loaded_szd.items() if value != 589}
        # loaded_szd = {key: value for key, value in loaded_szd.items()}

        vis_zd = {key: all_schools.index(value) for key, value in loaded_szd.items()}
        zv.zones_from_dict(vis_zd, centroid_location=dz.schools_locations)
        # exit()

    # dz.zone_dict = trim_noncontiguity(dz, dz.zone_dict)
    # # # heuristic(dz, dz.zd)
    # # # handle_anomolies(dz, dz.zd)
    # # # sanity_check(dz)
    # zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location)
    # dz.zd = drop_boundary(dz, dz.zone_dict)
    # zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location)
    # dz.zd = drop_centroid_distant(dz, dz.zone_dict)
    # zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location)
    dz.zone_dict = {}
    # dz.samezone_pairs = set()
    dz.zone_dict = assign_centroid_vicinity(dz, dz.zone_dict, config, loaded_szd)
    dz.zone_dict = trim_noncontiguity(dz, dz.zone_dict)
    dz.samezone_pairs = compute_samezone_pairs(dz, dz.zone_dict)
    # # # TODO check if during the assignment, one centroid is assigned to another zone.
    zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location)

    print("1110")
    dz._set_objective_model(loaded_szd, max_distance=config["max_distance"])
    initialize_preassigned_units(dz, dz.zone_dict)
    dz._shortage_and_balance_constraints(shortage_=True, balance_= False,
                     shortage=config["shortage"], overage= config["overage"], all_cap_shortage=config["all_cap_shortage"])
    dz._add_contiguity_constraint(loaded_szd)

    # dz.m.update()  # Update the model
    # print(f"Total number of dz.m variables: {dz.m.numVars}")
    # print(f"Total number of dz.m constraints: {dz.m.numConstrs}")

    # dz._add_population_balance_constraint(population_dev=param.population_dev)
    dz._add_school_count_constraint(loaded_szd)
    # dz._add_met_quality_constraint(min_pct = config["lbscqlty"])
    # dz._boundary_threshold_constraint(boundary_threshold=config["boundary_threshold"])
    print("1114")
    solve_success = dz.solve()


    if solve_success == 1:
        # zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location, save_name= name + "_BG")
        # dz.save(path=param.path, name = name + "_BG")
        local_search_mode = "None"
        if local_search_mode == "spanning_tree_cut":
            for noiterations in range(3):
                print("run number " + str(noiterations) + " ---------------------------------------------")
                spanning_tree_cut(dz, config)
                update_z(dz, dz.zone_dict)
        # find local search solution, and save it
        elif local_search_mode == "local_swap":
            iterative_update(config, dz, dz.zone_dict, zv, noiterations=3)
        # zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location, save_path=config["path"]+name+"_"+SUFFIX[input_level])
        # dz.save(path=config["path"], name=name+"_"+SUFFIX[input_level])

        new_loaded_szd = {key: dz.centroid_sch[value] for key, value in dz.zone_dict.items()}
        for blocks in loaded_szd:
            new_loaded_szd[blocks] = loaded_szd[blocks]

        vis_zd = {key: all_schools.index(value) for key, value in new_loaded_szd.items()}
        zv.zones_from_dict(vis_zd, centroid_location=dz.schools_locations, save_path=config["path"]+name+"x_"+SUFFIX[input_level])
        # print("dz.zone_dictd_is = ", dz.zone_dict)
        # exit()
        with open(os.path.expanduser(rec_path + name + "x_loaded_zd.pkl"), 'wb') as file:
            pickle.dump(new_loaded_szd, file)

    # # stats_evaluation(dz, dz.zd)
    # dz = load_initial_assignemt(dz, name=name, load_level="BG")
    # zv = ZoneVisualizer(input_level)
    # dz.zd = {60750209001: 0, 60750209002: 0, 60750209003: 0, 60750209004: 0, 60750210001: 0, 60750210002: 0, 60750210003: 0, 60750210004: 0, 60750211001: 0, 60750211002: 0, 60750211003: 0, 60750212001: 0, 60750212002: 0, 60750212003: 0, 60750213001: 0, 60750213002: 0, 60750214001: 0, 60750214002: 0, 60750214003: 0, 60750215001: 0, 60750215002: 0, 60750215003: 0, 60750215004: 0, 60750215005: 0, 60750216001: 0, 60750216002: 0, 60750217001: 0, 60750217002: 0, 60750217003: 0, 60750218001: 0, 60750218002: 0, 60750218003: 0, 60750218004: 0, 60750228033: 0, 60750229011: 0, 60750229012: 0, 60750229013: 0, 60750229021: 0, 60750229022: 0, 60750229031: 0, 60750229032: 0, 60750229033: 0, 60750230011: 0, 60750230012: 0, 60750230013: 0, 60750230031: 0, 60750230032: 0, 60750231022: 0, 60750231032: 0, 60750232002: 0, 60750232003: 0, 60750251001: 0, 60750251002: 0, 60750251003: 0, 60750252001: 0, 60750252002: 0, 60750252003: 0, 60750252004: 0, 60750253001: 0, 60750253002: 0, 60750253003: 0, 60750253004: 0, 60750254011: 0, 60750254012: 0, 60750254013: 0, 60750254021: 0, 60750254022: 0, 60750254023: 0, 60750254031: 0, 60750254032: 0, 60750255001: 0, 60750255002: 0, 60750256001: 0, 60750256002: 0, 60750256003: 0, 60750256004: 0, 60750257011: 0, 60750257013: 0, 60750257021: 0, 60750257023: 0, 60750258001: 0, 60750259001: 0, 60750259002: 0, 60750307001: 0, 60750307002: 0, 60750311001: 0, 60750612001: 0, 60750612002: 0, 60759806001: 0, 60750255003: 1, 60750255004: 1, 60750255005: 1, 60750255006: 1, 60750260011: 1, 60750260012: 1, 60750260021: 1, 60750260022: 1, 60750260031: 1, 60750260032: 1, 60750260041: 1, 60750260042: 1, 60750261001: 1, 60750261002: 1, 60750261003: 1, 60750261004: 1, 60750263011: 1, 60750263023: 1, 60750264011: 1, 60750264012: 1, 60750264031: 1, 60750264041: 1, 60750264042: 1, 60750307003: 1, 60750308001: 1, 60750308002: 1, 60750308003: 1, 60750308004: 1, 60750309001: 1, 60750309002: 1, 60750309003: 1, 60750309004: 1, 60750309005: 1, 60750310001: 1, 60750310002: 1, 60750310003: 1, 60750311002: 1, 60750311003: 1, 60750311004: 1, 60750311005: 1, 60750312011: 1, 60750312012: 1, 60750312013: 1, 60750312014: 1, 60750312021: 1, 60750312022: 1, 60750330002: 1, 60750330005: 1, 60750330006: 1, 60750351001: 1, 60750351002: 1, 60750351003: 1, 60750351004: 1, 60750351005: 1, 60750351006: 1, 60750351007: 1, 60750352011: 1, 60750352012: 1, 60750352013: 1, 60750352014: 1, 60750352015: 1, 60750352021: 1, 60750352022: 1, 60750352023: 1, 60750353002: 1, 60750353003: 1, 60750353005: 1, 60750353006: 1, 60750354001: 1, 60750354002: 1, 60750354003: 1, 60750354004: 1, 60750354005: 1, 60750605021: 1, 60750605022: 1, 60750605023: 1, 60759805011: 1, 60750170003: 2, 60750171012: 2, 60750171022: 2, 60750204011: 2, 60750204012: 2, 60750204013: 2, 60750204021: 2, 60750204022: 2, 60750205003: 2, 60750206002: 2, 60750211004: 2, 60750262001: 2, 60750262002: 2, 60750262003: 2, 60750262004: 2, 60750262005: 2, 60750263012: 2, 60750263013: 2, 60750263021: 2, 60750263022: 2, 60750263031: 2, 60750263032: 2, 60750301012: 2, 60750301013: 2, 60750301014: 2, 60750301021: 2, 60750301022: 2, 60750301023: 2, 60750302011: 2, 60750302012: 2, 60750302013: 2, 60750302021: 2, 60750302022: 2, 60750302023: 2, 60750303011: 2, 60750303012: 2, 60750303013: 2, 60750303014: 2, 60750303021: 2, 60750303022: 2, 60750303023: 2, 60750304001: 2, 60750304002: 2, 60750304003: 2, 60750304004: 2, 60750304005: 2, 60750305001: 2, 60750305002: 2, 60750305003: 2, 60750306001: 2, 60750306002: 2, 60750306003: 2, 60750308005: 2, 60750309006: 2, 60750309007: 2, 60750313011: 2, 60750313012: 2, 60750313013: 2, 60750313021: 2, 60750313022: 2, 60750313023: 2, 60750314001: 2, 60750314002: 2, 60750314003: 2, 60750314004: 2, 60750314005: 2, 60750326011: 2, 60750326012: 2, 60750326013: 2, 60750326021: 2, 60750326022: 2, 60750326023: 2, 60750327001: 2, 60750327002: 2, 60750327003: 2, 60750327004: 2, 60750327005: 2, 60750327006: 2, 60750327007: 2, 60750328011: 2, 60750328012: 2, 60750328013: 2, 60750328021: 2, 60750328022: 2, 60750328023: 2, 60750329011: 2, 60750329012: 2, 60750329013: 2, 60750329014: 2, 60750329021: 2, 60750329022: 2, 60750329023: 2, 60750330001: 2, 60750330003: 2, 60750330004: 2, 60750331001: 2, 60750331002: 2, 60750331003: 2, 60750331004: 2, 60750332011: 2, 60750332031: 2, 60750332032: 2, 60750332041: 2, 60750332042: 2, 60750332043: 2, 60750353001: 2, 60750353004: 2, 60750604001: 2, 60750124011: 3, 60750124021: 3, 60750124022: 3, 60750124023: 3, 60750126011: 3, 60750126021: 3, 60750126022: 3, 60750127001: 3, 60750127002: 3, 60750127003: 3, 60750128001: 3, 60750128002: 3, 60750128003: 3, 60750128004: 3, 60750129022: 3, 60750129023: 3, 60750130002: 3, 60750130003: 3, 60750130004: 3, 60750131021: 3, 60750131022: 3, 60750132001: 3, 60750132002: 3, 60750132003: 3, 60750133001: 3, 60750133002: 3, 60750133003: 3, 60750133004: 3, 60750133005: 3, 60750134001: 3, 60750134002: 3, 60750134003: 3, 60750135001: 3, 60750135002: 3, 60750152001: 3, 60750152002: 3, 60750152003: 3, 60750153001: 3, 60750153002: 3, 60750154001: 3, 60750154002: 3, 60750154003: 3, 60750154004: 3, 60750154005: 3, 60750155001: 3, 60750155002: 3, 60750155003: 3, 60750156001: 3, 60750156002: 3, 60750156003: 3, 60750157001: 3, 60750157003: 3, 60750157004: 3, 60750158012: 3, 60750160001: 3, 60750165004: 3, 60750176013: 3, 60750176014: 3, 60750178011: 3, 60750178012: 3, 60750178021: 3, 60750178022: 3, 60750180001: 3, 60750180002: 3, 60750227021: 3, 60750401001: 3, 60750401002: 3, 60750401003: 3, 60750401004: 3, 60750402001: 3, 60750402002: 3, 60750402003: 3, 60750402004: 3, 60750426011: 3, 60750426012: 3, 60750426021: 3, 60750426022: 3, 60750426023: 3, 60750427001: 3, 60750427002: 3, 60750427003: 3, 60750428001: 3, 60750428002: 3, 60750428003: 3, 60750451001: 3, 60750451002: 3, 60750451003: 3, 60750452001: 3, 60750452002: 3, 60750452003: 3, 60750452004: 3, 60750452005: 3, 60750476001: 3, 60750476002: 3, 60750476003: 3, 60750476004: 3, 60750477011: 3, 60750477012: 3, 60750477013: 3, 60750477021: 3, 60750477022: 3, 60750477023: 3, 60750478011: 3, 60750478012: 3, 60750478013: 3, 60750478021: 3, 60750478022: 3, 60750478023: 3, 60750479011: 3, 60750479012: 3, 60750479013: 3, 60750479014: 3, 60750479015: 3, 60750479021: 3, 60750479022: 3, 60750479023: 3, 60750601001: 3, 60750607001: 3, 60750607002: 3, 60750607003: 3, 60750615002: 3, 60750615004: 3, 60750615005: 3, 60750615006: 3, 60759802001: 3, 60759803001: 3, 60750101001: 4, 60750101002: 4, 60750102001: 4, 60750102002: 4, 60750102003: 4, 60750103001: 4, 60750103002: 4, 60750103003: 4, 60750104001: 4, 60750104002: 4, 60750104003: 4, 60750104004: 4, 60750105001: 4, 60750105002: 4, 60750106001: 4, 60750106002: 4, 60750106003: 4, 60750107001: 4, 60750107002: 4, 60750107003: 4, 60750107004: 4, 60750108001: 4, 60750108002: 4, 60750108003: 4, 60750109001: 4, 60750109002: 4, 60750109003: 4, 60750110001: 4, 60750110002: 4, 60750110003: 4, 60750111001: 4, 60750111002: 4, 60750111003: 4, 60750112001: 4, 60750112002: 4, 60750112003: 4, 60750113001: 4, 60750113002: 4, 60750117001: 4, 60750117002: 4, 60750118001: 4, 60750119011: 4, 60750119012: 4, 60750119021: 4, 60750119022: 4, 60750120001: 4, 60750120002: 4, 60750121001: 4, 60750121002: 4, 60750122011: 4, 60750122012: 4, 60750122021: 4, 60750123011: 4, 60750123012: 4, 60750123021: 4, 60750123022: 4, 60750124012: 4, 60750125011: 4, 60750125012: 4, 60750125021: 4, 60750125022: 4, 60750129011: 4, 60750129012: 4, 60750129021: 4, 60750130001: 4, 60750131011: 4, 60750131012: 4, 60750151001: 4, 60750151002: 4, 60750157002: 4, 60750158011: 4, 60750158013: 4, 60750158021: 4, 60750158022: 4, 60750159001: 4, 60750159002: 4, 60750161001: 4, 60750161002: 4, 60750161003: 4, 60750161004: 4, 60750162001: 4, 60750162002: 4, 60750162003: 4, 60750163001: 4, 60750163002: 4, 60750163003: 4, 60750164001: 4, 60750164002: 4, 60750165001: 4, 60750165002: 4, 60750165003: 4, 60750166001: 4, 60750166002: 4, 60750166003: 4, 60750166004: 4, 60750167001: 4, 60750167002: 4, 60750167003: 4, 60750167004: 4, 60750168011: 4, 60750168012: 4, 60750171011: 4, 60750171013: 4, 60750171021: 4, 60750171023: 4, 60750176012: 4, 60750176015: 4, 60750179021: 4, 60750301011: 4, 60750611001: 4, 60750611002: 4, 60750611003: 4, 60750615001: 4, 60750615003: 4, 60750176011: 4, 60750168013: 5, 60750168021: 5, 60750168022: 5, 60750168023: 5, 60750169001: 5, 60750169002: 5, 60750170001: 5, 60750170002: 5, 60750177001: 5, 60750177002: 5, 60750201001: 5, 60750201002: 5, 60750201003: 5, 60750201004: 5, 60750202001: 5, 60750202002: 5, 60750202003: 5, 60750203001: 5, 60750203002: 5, 60750203003: 5, 60750205001: 5, 60750205002: 5, 60750206001: 5, 60750206003: 5, 60750206004: 5, 60750207001: 5, 60750207002: 5, 60750207003: 5, 60750208001: 5, 60750208002: 5, 60750208003: 5, 60750208004: 5, 60750226001: 5, 60750226002: 5, 60750227022: 5, 60750227041: 5, 60750227042: 5, 60750228011: 5, 60750228012: 5, 60750228013: 5, 60750228021: 5, 60750228022: 5, 60750228031: 5, 60750228032: 5, 60750231021: 5, 60750231031: 5, 60750232001: 5, 60750233001: 5, 60750234001: 5, 60750234002: 5, 60750257012: 5, 60750257022: 5, 60750258002: 5, 60750259003: 5, 60750264021: 5, 60750264022: 5, 60750264023: 5, 60750264032: 5, 60750610001: 5, 60750610002: 5, 60750614001: 5, 60750614002: 5, 60750614003: 5, 60759809001: 5}
    # iterative_update(dz, dz.zd, zv, noiterations=5)
    # zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location, save_name= name + "_BG")

def recom_search(config):
    input_level = 'BlockGroup'
    dz = DesignZones(
        config,
        level=input_level,
    )
    zv = ZoneVisualizer(input_level)


    # iterative_update(param, dz, dz.zd, zv, noiterations=3)
    dz.zone_dict = {60750102001: 0, 60750102002: 0, 60750102003: 0, 60750109003: 0, 60750110001: 0, 60750110002: 0, 60750110003: 0, 60750111001: 0, 60750126022: 0, 60750129011: 0, 60750129012: 0, 60750129021: 0, 60750129022: 0, 60750129023: 0, 60750130001: 0, 60750130002: 0, 60750130003: 0, 60750130004: 0, 60750131011: 0, 60750131012: 0, 60750131021: 0, 60750131022: 0, 60750132001: 0, 60750132002: 0, 60750133003: 0, 60750133005: 0, 60750134001: 0, 60750134002: 0, 60750135001: 0, 60750135002: 0, 60750152001: 0, 60750152002: 0, 60750152003: 0, 60750153001: 0, 60750153002: 0, 60750154001: 0, 60750154002: 0, 60750154003: 0, 60750154004: 0, 60750154005: 0, 60750155001: 0, 60750155002: 0, 60750155003: 0, 60750156001: 0, 60750156002: 0, 60750156003: 0, 60750157001: 0, 60750157002: 0, 60750157003: 0, 60750157004: 0, 60750158011: 0, 60750158012: 0, 60750158013: 0, 60750158021: 0, 60750158022: 0, 60750159001: 0, 60750159002: 0, 60750161001: 0, 60750161002: 0, 60750161003: 0, 60750161004: 0, 60750165001: 0, 60750165002: 0, 60750165003: 0, 60750165004: 0, 60750451001: 0, 60750451002: 0, 60750451003: 0, 60750101001: 1, 60750101002: 1, 60750103001: 1, 60750103002: 1, 60750103003: 1, 60750104001: 1, 60750104002: 1, 60750104003: 1, 60750104004: 1, 60750105001: 1, 60750105002: 1, 60750106001: 1, 60750106002: 1, 60750106003: 1, 60750107001: 1, 60750107002: 1, 60750107003: 1, 60750107004: 1, 60750108001: 1, 60750108002: 1, 60750108003: 1, 60750109001: 1, 60750109002: 1, 60750111002: 1, 60750111003: 1, 60750112001: 1, 60750112002: 1, 60750112003: 1, 60750113001: 1, 60750113002: 1, 60750117001: 1, 60750117002: 1, 60750118001: 1, 60750119011: 1, 60750119012: 1, 60750119021: 1, 60750119022: 1, 60750120001: 1, 60750120002: 1, 60750121001: 1, 60750121002: 1, 60750122011: 1, 60750122012: 1, 60750122021: 1, 60750123011: 1, 60750123012: 1, 60750123021: 1, 60750123022: 1, 60750125012: 1, 60750125021: 1, 60750125022: 1, 60750151001: 1, 60750151002: 1, 60750611001: 1, 60750611002: 1, 60750611003: 1, 60750126011: 2, 60750126021: 2, 60750127001: 2, 60750127002: 2, 60750127003: 2, 60750128001: 2, 60750128002: 2, 60750128003: 2, 60750128004: 2, 60750132003: 2, 60750133001: 2, 60750133002: 2, 60750133004: 2, 60750134003: 2, 60750401001: 2, 60750401002: 2, 60750401003: 2, 60750401004: 2, 60750402001: 2, 60750402002: 2, 60750402003: 2, 60750402004: 2, 60750426011: 2, 60750426012: 2, 60750426021: 2, 60750426022: 2, 60750426023: 2, 60750427001: 2, 60750427002: 2, 60750427003: 2, 60750428001: 2, 60750428002: 2, 60750428003: 2, 60750452001: 2, 60750452002: 2, 60750452003: 2, 60750452004: 2, 60750452005: 2, 60750476001: 2, 60750476002: 2, 60750476003: 2, 60750476004: 2, 60750477011: 2, 60750477012: 2, 60750477013: 2, 60750477021: 2, 60750477022: 2, 60750477023: 2, 60750478011: 2, 60750478012: 2, 60750478013: 2, 60750478021: 2, 60750478022: 2, 60750478023: 2, 60750479011: 2, 60750479012: 2, 60750479013: 2, 60750479014: 2, 60750479015: 2, 60750479021: 2, 60750479022: 2, 60750479023: 2, 60750601001: 2, 60759802001: 2, 60750124011: 3, 60750124012: 3, 60750124021: 3, 60750124022: 3, 60750124023: 3, 60750160001: 3, 60750162001: 3, 60750162002: 3, 60750162003: 3, 60750163001: 3, 60750163002: 3, 60750163003: 3, 60750164001: 3, 60750164002: 3, 60750166001: 3, 60750166002: 3, 60750166003: 3, 60750166004: 3, 60750167001: 3, 60750167002: 3, 60750167003: 3, 60750167004: 3, 60750168011: 3, 60750168012: 3, 60750168013: 3, 60750168021: 3, 60750168022: 3, 60750168023: 3, 60750169001: 3, 60750169002: 3, 60750170001: 3, 60750170002: 3, 60750170003: 3, 60750171011: 3, 60750171012: 3, 60750171013: 3, 60750171021: 3, 60750171022: 3, 60750171023: 3, 60750202001: 3, 60750204011: 3, 60750204012: 3, 60750204013: 3, 60750301011: 3, 60750301012: 3, 60750301013: 3, 60750301014: 3, 60750301021: 3, 60750301022: 3, 60750301023: 3, 60750302011: 3, 60750302012: 3, 60750302013: 3, 60750302021: 3, 60750302022: 3, 60750302023: 3, 60750303011: 3, 60750303012: 3, 60750303013: 3, 60750303014: 3, 60750303021: 3, 60750303022: 3, 60750303023: 3, 60750304001: 3, 60750304002: 3, 60750304003: 3, 60750305001: 3, 60750305002: 3, 60750305003: 3, 60750306001: 3, 60750306002: 3, 60750306003: 3, 60750326011: 3, 60750326012: 3, 60759803001: 3, 60750125011: 4, 60750176011: 4, 60750176012: 4, 60750176013: 4, 60750176014: 4, 60750176015: 4, 60750177001: 4, 60750177002: 4, 60750178011: 4, 60750178012: 4, 60750178021: 4, 60750178022: 4, 60750179021: 4, 60750180001: 4, 60750180002: 4, 60750201001: 4, 60750201002: 4, 60750201003: 4, 60750201004: 4, 60750202002: 4, 60750202003: 4, 60750203001: 4, 60750203002: 4, 60750203003: 4, 60750205001: 4, 60750205002: 4, 60750206001: 4, 60750206002: 4, 60750206003: 4, 60750206004: 4, 60750207001: 4, 60750207002: 4, 60750207003: 4, 60750208001: 4, 60750208002: 4, 60750208003: 4, 60750208004: 4, 60750226001: 4, 60750226002: 4, 60750227021: 4, 60750227022: 4, 60750227041: 4, 60750227042: 4, 60750228011: 4, 60750228012: 4, 60750228013: 4, 60750228021: 4, 60750228022: 4, 60750228032: 4, 60750228033: 4, 60750229031: 4, 60750229032: 4, 60750607001: 4, 60750607002: 4, 60750607003: 4, 60750614001: 4, 60750614002: 4, 60750614003: 4, 60750615001: 4, 60750615002: 4, 60750615003: 4, 60750615004: 4, 60750615005: 4, 60750615006: 4, 60750204021: 5, 60750204022: 5, 60750205003: 5, 60750209001: 5, 60750209002: 5, 60750209003: 5, 60750209004: 5, 60750210001: 5, 60750210002: 5, 60750210003: 5, 60750210004: 5, 60750211001: 5, 60750211002: 5, 60750211003: 5, 60750211004: 5, 60750212001: 5, 60750212002: 5, 60750212003: 5, 60750213001: 5, 60750213002: 5, 60750214001: 5, 60750214002: 5, 60750214003: 5, 60750215001: 5, 60750215002: 5, 60750215003: 5, 60750215004: 5, 60750215005: 5, 60750216001: 5, 60750216002: 5, 60750217001: 5, 60750217002: 5, 60750217003: 5, 60750218001: 5, 60750218002: 5, 60750218003: 5, 60750218004: 5, 60750228031: 5, 60750229011: 5, 60750229012: 5, 60750229013: 5, 60750229021: 5, 60750229022: 5, 60750229033: 5, 60750251001: 5, 60750251002: 5, 60750251003: 5, 60750252001: 5, 60750252002: 5, 60750252003: 5, 60750252004: 5, 60750253001: 5, 60750253002: 5, 60750253003: 5, 60750253004: 5, 60750254011: 5, 60750254012: 5, 60750254013: 5, 60750254021: 5, 60750254022: 5, 60750254023: 5, 60750254031: 5, 60750254032: 5, 60750255001: 5, 60750311001: 5, 60750304004: 6, 60750304005: 6, 60750308001: 6, 60750308004: 6, 60750308005: 6, 60750326013: 6, 60750326021: 6, 60750326022: 6, 60750326023: 6, 60750327001: 6, 60750327002: 6, 60750327003: 6, 60750327004: 6, 60750327005: 6, 60750327006: 6, 60750327007: 6, 60750328011: 6, 60750328012: 6, 60750328013: 6, 60750328021: 6, 60750328022: 6, 60750328023: 6, 60750329011: 6, 60750329012: 6, 60750329013: 6, 60750329014: 6, 60750329021: 6, 60750329022: 6, 60750329023: 6, 60750330001: 6, 60750330002: 6, 60750330003: 6, 60750330004: 6, 60750330005: 6, 60750330006: 6, 60750331003: 6, 60750331004: 6, 60750351001: 6, 60750351002: 6, 60750351003: 6, 60750351004: 6, 60750351005: 6, 60750351006: 6, 60750351007: 6, 60750352011: 6, 60750352012: 6, 60750352013: 6, 60750352014: 6, 60750352015: 6, 60750352021: 6, 60750352022: 6, 60750352023: 6, 60750353001: 6, 60750353002: 6, 60750353003: 6, 60750353004: 6, 60750353005: 6, 60750353006: 6, 60750354001: 6, 60750354002: 6, 60750354003: 6, 60750354004: 6, 60750354005: 6, 60750604001: 6, 60750230011: 7, 60750230012: 7, 60750230013: 7, 60750230031: 7, 60750230032: 7, 60750231021: 7, 60750231022: 7, 60750231031: 7, 60750231032: 7, 60750232001: 7, 60750232002: 7, 60750232003: 7, 60750233001: 7, 60750234001: 7, 60750234002: 7, 60750256001: 7, 60750256002: 7, 60750256003: 7, 60750256004: 7, 60750257011: 7, 60750257012: 7, 60750257013: 7, 60750257021: 7, 60750257022: 7, 60750257023: 7, 60750258001: 7, 60750258002: 7, 60750259001: 7, 60750259002: 7, 60750259003: 7, 60750260022: 7, 60750264021: 7, 60750264022: 7, 60750610001: 7, 60750612001: 7, 60750612002: 7, 60759806001: 7, 60759809001: 7, 60750255002: 8, 60750255003: 8, 60750255004: 8, 60750255005: 8, 60750255006: 8, 60750260011: 8, 60750260012: 8, 60750260021: 8, 60750260041: 8, 60750261001: 8, 60750261004: 8, 60750307001: 8, 60750307002: 8, 60750307003: 8, 60750308002: 8, 60750308003: 8, 60750309001: 8, 60750309002: 8, 60750309003: 8, 60750309004: 8, 60750309005: 8, 60750309006: 8, 60750309007: 8, 60750310001: 8, 60750310002: 8, 60750310003: 8, 60750311002: 8, 60750311003: 8, 60750311004: 8, 60750311005: 8, 60750312011: 8, 60750312012: 8, 60750312013: 8, 60750312014: 8, 60750312021: 8, 60750312022: 8, 60750313011: 8, 60750313012: 8, 60750313013: 8, 60750313021: 8, 60750313022: 8, 60750313023: 8, 60750314001: 8, 60750314005: 8, 60750331001: 8, 60750331002: 8, 60750332011: 8, 60750332031: 8, 60750332032: 8, 60750332041: 8, 60750332042: 8, 60750332043: 8, 60750260031: 9, 60750260032: 9, 60750260042: 9, 60750261002: 9, 60750261003: 9, 60750262001: 9, 60750262002: 9, 60750262003: 9, 60750262004: 9, 60750262005: 9, 60750263011: 9, 60750263012: 9, 60750263013: 9, 60750263021: 9, 60750263022: 9, 60750263023: 9, 60750263031: 9, 60750263032: 9, 60750264011: 9, 60750264012: 9, 60750264023: 9, 60750264031: 9, 60750264032: 9, 60750264041: 9, 60750264042: 9, 60750314002: 9, 60750314003: 9, 60750314004: 9, 60750605021: 9, 60750605022: 9, 60750605023: 9, 60750610002: 9, 60759805011: 9}
    dz.zone_lists = [[60750102001, 60750102002, 60750102003, 60750109003, 60750110001, 60750110002, 60750110003, 60750111001, 60750126022, 60750129011, 60750129012, 60750129021, 60750129022, 60750129023, 60750130001, 60750130002, 60750130003, 60750130004, 60750131011, 60750131012, 60750131021, 60750131022, 60750132001, 60750132002, 60750133003, 60750133005, 60750134001, 60750134002, 60750135001, 60750135002, 60750152001, 60750152002, 60750152003, 60750153001, 60750153002, 60750154001, 60750154002, 60750154003, 60750154004, 60750154005, 60750155001, 60750155002, 60750155003, 60750156001, 60750156002, 60750156003, 60750157001, 60750157002, 60750157003, 60750157004, 60750158011, 60750158012, 60750158013, 60750158021, 60750158022, 60750159001, 60750159002, 60750161001, 60750161002, 60750161003, 60750161004, 60750165001, 60750165002, 60750165003, 60750165004, 60750451001, 60750451002, 60750451003], [60750101001, 60750101002, 60750103001, 60750103002, 60750103003, 60750104001, 60750104002, 60750104003, 60750104004, 60750105001, 60750105002, 60750106001, 60750106002, 60750106003, 60750107001, 60750107002, 60750107003, 60750107004, 60750108001, 60750108002, 60750108003, 60750109001, 60750109002, 60750111002, 60750111003, 60750112001, 60750112002, 60750112003, 60750113001, 60750113002, 60750117001, 60750117002, 60750118001, 60750119011, 60750119012, 60750119021, 60750119022, 60750120001, 60750120002, 60750121001, 60750121002, 60750122011, 60750122012, 60750122021, 60750123011, 60750123012, 60750123021, 60750123022, 60750125012, 60750125021, 60750125022, 60750151001, 60750151002, 60750611001, 60750611002, 60750611003], [60750126011, 60750126021, 60750127001, 60750127002, 60750127003, 60750128001, 60750128002, 60750128003, 60750128004, 60750132003, 60750133001, 60750133002, 60750133004, 60750134003, 60750401001, 60750401002, 60750401003, 60750401004, 60750402001, 60750402002, 60750402003, 60750402004, 60750426011, 60750426012, 60750426021, 60750426022, 60750426023, 60750427001, 60750427002, 60750427003, 60750428001, 60750428002, 60750428003, 60750452001, 60750452002, 60750452003, 60750452004, 60750452005, 60750476001, 60750476002, 60750476003, 60750476004, 60750477011, 60750477012, 60750477013, 60750477021, 60750477022, 60750477023, 60750478011, 60750478012, 60750478013, 60750478021, 60750478022, 60750478023, 60750479011, 60750479012, 60750479013, 60750479014, 60750479015, 60750479021, 60750479022, 60750479023, 60750601001, 60759802001], [60750124011, 60750124012, 60750124021, 60750124022, 60750124023, 60750160001, 60750162001, 60750162002, 60750162003, 60750163001, 60750163002, 60750163003, 60750164001, 60750164002, 60750166001, 60750166002, 60750166003, 60750166004, 60750167001, 60750167002, 60750167003, 60750167004, 60750168011, 60750168012, 60750168013, 60750168021, 60750168022, 60750168023, 60750169001, 60750169002, 60750170001, 60750170002, 60750170003, 60750171011, 60750171012, 60750171013, 60750171021, 60750171022, 60750171023, 60750202001, 60750204011, 60750204012, 60750204013, 60750301011, 60750301012, 60750301013, 60750301014, 60750301021, 60750301022, 60750301023, 60750302011, 60750302012, 60750302013, 60750302021, 60750302022, 60750302023, 60750303011, 60750303012, 60750303013, 60750303014, 60750303021, 60750303022, 60750303023, 60750304001, 60750304002, 60750304003, 60750305001, 60750305002, 60750305003, 60750306001, 60750306002, 60750306003, 60750326011, 60750326012, 60759803001], [60750125011, 60750176011, 60750176012, 60750176013, 60750176014, 60750176015, 60750177001, 60750177002, 60750178011, 60750178012, 60750178021, 60750178022, 60750179021, 60750180001, 60750180002, 60750201001, 60750201002, 60750201003, 60750201004, 60750202002, 60750202003, 60750203001, 60750203002, 60750203003, 60750205001, 60750205002, 60750206001, 60750206002, 60750206003, 60750206004, 60750207001, 60750207002, 60750207003, 60750208001, 60750208002, 60750208003, 60750208004, 60750226001, 60750226002, 60750227021, 60750227022, 60750227041, 60750227042, 60750228011, 60750228012, 60750228013, 60750228021, 60750228022, 60750228032, 60750228033, 60750229031, 60750229032, 60750607001, 60750607002, 60750607003, 60750614001, 60750614002, 60750614003, 60750615001, 60750615002, 60750615003, 60750615004, 60750615005, 60750615006], [60750204021, 60750204022, 60750205003, 60750209001, 60750209002, 60750209003, 60750209004, 60750210001, 60750210002, 60750210003, 60750210004, 60750211001, 60750211002, 60750211003, 60750211004, 60750212001, 60750212002, 60750212003, 60750213001, 60750213002, 60750214001, 60750214002, 60750214003, 60750215001, 60750215002, 60750215003, 60750215004, 60750215005, 60750216001, 60750216002, 60750217001, 60750217002, 60750217003, 60750218001, 60750218002, 60750218003, 60750218004, 60750228031, 60750229011, 60750229012, 60750229013, 60750229021, 60750229022, 60750229033, 60750251001, 60750251002, 60750251003, 60750252001, 60750252002, 60750252003, 60750252004, 60750253001, 60750253002, 60750253003, 60750253004, 60750254011, 60750254012, 60750254013, 60750254021, 60750254022, 60750254023, 60750254031, 60750254032, 60750255001, 60750311001], [60750304004, 60750304005, 60750308001, 60750308004, 60750308005, 60750326013, 60750326021, 60750326022, 60750326023, 60750327001, 60750327002, 60750327003, 60750327004, 60750327005, 60750327006, 60750327007, 60750328011, 60750328012, 60750328013, 60750328021, 60750328022, 60750328023, 60750329011, 60750329012, 60750329013, 60750329014, 60750329021, 60750329022, 60750329023, 60750330001, 60750330002, 60750330003, 60750330004, 60750330005, 60750330006, 60750331003, 60750331004, 60750351001, 60750351002, 60750351003, 60750351004, 60750351005, 60750351006, 60750351007, 60750352011, 60750352012, 60750352013, 60750352014, 60750352015, 60750352021, 60750352022, 60750352023, 60750353001, 60750353002, 60750353003, 60750353004, 60750353005, 60750353006, 60750354001, 60750354002, 60750354003, 60750354004, 60750354005, 60750604001], [60750230011, 60750230012, 60750230013, 60750230031, 60750230032, 60750231021, 60750231022, 60750231031, 60750231032, 60750232001, 60750232002, 60750232003, 60750233001, 60750234001, 60750234002, 60750256001, 60750256002, 60750256003, 60750256004, 60750257011, 60750257012, 60750257013, 60750257021, 60750257022, 60750257023, 60750258001, 60750258002, 60750259001, 60750259002, 60750259003, 60750260022, 60750264021, 60750264022, 60750610001, 60750612001, 60750612002, 60759806001, 60759809001], [60750255002, 60750255003, 60750255004, 60750255005, 60750255006, 60750260011, 60750260012, 60750260021, 60750260041, 60750261001, 60750261004, 60750307001, 60750307002, 60750307003, 60750308002, 60750308003, 60750309001, 60750309002, 60750309003, 60750309004, 60750309005, 60750309006, 60750309007, 60750310001, 60750310002, 60750310003, 60750311002, 60750311003, 60750311004, 60750311005, 60750312011, 60750312012, 60750312013, 60750312014, 60750312021, 60750312022, 60750313011, 60750313012, 60750313013, 60750313021, 60750313022, 60750313023, 60750314001, 60750314005, 60750331001, 60750331002, 60750332011, 60750332031, 60750332032, 60750332041, 60750332042, 60750332043], [60750260031, 60750260032, 60750260042, 60750261002, 60750261003, 60750262001, 60750262002, 60750262003, 60750262004, 60750262005, 60750263011, 60750263012, 60750263013, 60750263021, 60750263022, 60750263023, 60750263031, 60750263032, 60750264011, 60750264012, 60750264023, 60750264031, 60750264032, 60750264041, 60750264042, 60750314002, 60750314003, 60750314004, 60750605021, 60750605022, 60750605023, 60750610002, 60759805011]]
    zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location)

    mcmc = Relaxed_ReCom(dz.zone_lists, dz)
    for iter in range(17000):
        zones = mcmc.ReCom_step(dz.zone_lists)
        for z in range(len(zones)):
            for area in zones[z]:
                dz.zone_dict[area] = z
        dz.zone_lists = zones

        print(iter)
        if iter >15000:
            dz.save(path=config["path"], name = "_BG_MCMC_" + str(iter))
            zv.zones_from_dict(dz.zone_dict, centroid_location=dz.centroid_location, save_name="BG_MCMC_" + str(iter))


if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # recom_search(param)
    local_search(config)

    # for frl_dev in [0.33, 0.27]:
    #     param.frl_dev = frl_dev
    #     for shortage in [0.35, 0.29, 0.24]:
    #         param.shortage = shortage
    #         with open("../Config/centroids.yaml", "r") as f:
    #             centroid_options = yaml.safe_load(f)
    #             for centroids in centroid_options:
    #                 param.zone_count = int(centroids.split("-")[0])
    #                 param.centroids_type = centroids
    #                 local_search(param)