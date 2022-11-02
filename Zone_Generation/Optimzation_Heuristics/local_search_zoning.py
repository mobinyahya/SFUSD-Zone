import math
import os, sys, yaml


sys.path.append('../..')
sys.path.append('../../summary_statistics')
from Graphic_Visualization.zone_viz import ZoneVisualizer
# from IP_Zoning import DesignZones
from Zone_Generation.Optimization_IP.generate_zones import DesignZones, _get_zone_dict
from Zone_Generation.Optimzation_Heuristics.zone_eval import * # evaluate_assignment_score, Tuning_param, boundary_trimming, evaluate_contiguity
from Zone_Generation.Config.Tuning_param import *

# in this code, we run the iterative local search algorithm, to get better zone assignments
# Assumptions:
# 1- This local Search is only for blockgroup level
# 2- The starting point of Local Search (initialization step), is a zoning solution in attendance area level
#    we then convert this zoning, to its equivalent blockgroup level. And start iterative local search from there
def iterative_lp():
    linear_solution = True
    if linear_solution:
        param = Tuning_param()
        zv = ZoneVisualizer('BlockGroup')

        rounded_zd = {}
        for i in range(0, 4):
            dz = DesignZones(M=3, level='BlockGroup', centroidsType=0, include_citywide=True)
            print("linear rounding, iteration number " + str(i))

            dz.setmodel(shortage=param.shortage, balance=param.balance)
            # print("check initialization")
            # print(dz.area2idx[60750301013])
            # dz.m.addConstr(dz.x[dz.area2idx[60750301013],1] == 0)
            # rounded_zd = {60750101001: 0, 60750101002: 0, 60750102001: 0, 60750102002: 0, 60750102003: 0, 60750103001: 0, 60750103002: 0}
            #    , 60750103003: 0, 60750104001: 0, 60750104002: 0, 60750104003: 0, 60750105001: 0, 60750105002: 0, 60750106001: 0, 60750106003: 0, 60750107002: 0, 60750107003: 0, 60750107004: 0, 60750108001: 0,
            # 60750108002: 0, 60750108003: 0, 60750109001: 0, 60750109002: 0, 60750109003: 0, 60750110001: 0, 60750110002: 0, 60750110003: 0, 60750111001: 0, 60750111002: 0, 60750112001: 0, 60750112002: 0, 60750113001: 0, 60750113002: 0, 60750117001: 0, 60750117002: 0, 60750118001: 0, 60750119011: 0, 60750119021: 0,
            # 60750120001: 0, 60750120002: 0, 60750121001: 0, 60750121002: 0, 60750122011: 0, 60750122012: 0, 60750122021: 0, 60750123012: 0, 60750123021: 0, 60750123022: 0, 60750124011: 0, 60750124012: 0, 60750124021: 0, 60750124022: 0, 60750124023: 0, 60750125011: 0, 60750125012: 0, 60750125021: 0, 60750125022: 0,
            # 60750126011: 0, 60750126021: 0, 60750126022: 0, 60750127001: 0, 60750127002: 0, 60750128001: 0, 60750128003: 0, 60750129011: 0, 60750129012: 0, 60750129021: 0, 60750129022: 0, 60750129023: 0, 60750130002: 0, 60750130003: 0, 60750130004: 0, 60750131012: 0, 60750131021: 0, 60750131022: 0, 60750135001: 0,
            # 60750151001: 0, 60750152001: 0, 60750152002: 0, 60750152003: 0, 60750155001: 0, 60750159001: 0, 60750160001: 0, 60750176012: 0, 60750176013: 0, 60750176014: 0, 60750176015: 0, 60750178021: 0, 60750178022: 0, 60750180001: 0, 60750180002: 0, 60750607001: 0, 60750607003: 0, 60750611001: 0, 60750611002: 0,
            # 60750611003: 0, 60750615001: 0, 60750615002: 0, 60750615003: 0, 60750615004: 0, 60750615006: 0, 60750106002: 0, 60750111003: 0, 60750130001: 0, 60750153001: 0, 60750119022: 0, 60750615005: 0, 60750132003: 0, 60750176011: 0, 60750127003: 0, 60750128004: 0, 60750123011: 0, 60750135002: 0, 60750131011: 0,
            # 60750132002: 0, 60750128002: 0, 60750151002: 0, 60750134001: 0, 60750132001: 0, 60750178012: 0, 60750107001: 0, 60750119012: 0, 60750178011: 0, 60750112003: 0, 60750104004: 0, 60750264012: 1, 60750301013: 2, 60750301014: 2, 60750302011: 2, 60750302012: 2, 60750302013: 2, 60750302021: 2, 60750302022: 2,
            # 60750302023: 2, 60750303011: 2, 60750303012: 2, 60750303013: 2, 60750303014: 2, 60750303021: 2, 60750303022: 2, 60750303023: 2, 60750304002: 2, 60750304003: 2, 60750304004: 2, 60750304005: 2, 60750326011: 2, 60750326012: 2, 60750326013: 2, 60750326021: 2, 60750326022: 2, 60750326023: 2, 60750327001: 2,
            # 60750327002: 2, 60750327003: 2, 60750327004: 2, 60750327005: 2, 60750327006: 2, 60750327007: 2, 60750328011: 2, 60750328012: 2, 60750328013: 2, 60750328021: 2, 60750328022: 2, 60750328023: 2, 60750329011: 2, 60750329012: 2, 60750329013: 2, 60750329014: 2, 60750329021: 2, 60750329022: 2, 60750329023: 2,
            # 60750330001: 2, 60750330005: 2, 60750351001: 2, 60750351002: 2, 60750351003: 2, 60750351004: 2, 60750351005: 2, 60750351006: 2, 60750351007: 2, 60750352011: 2, 60750352012: 2, 60750352013: 2, 60750352014: 2, 60750352015: 2, 60750352021: 2, 60750352022: 2, 60750352023: 2, 60750353001: 2, 60750353002: 2,
            # 60750353005: 2, 60750353006: 2, 60750354001: 2, 60750354002: 2, 60750354003: 2, 60750354004: 2, 60750354005: 2}
            # rounded_zd = {60750264012: 0, 60750104003: 1, 60750353001: 2}
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
            dz._add_diversity_constraints(lbfrl=param.lbfrl, ubfrl=param.ubfrl)

            dz.solve()
            print("this is the resulting zone dictionary dz.zd")
            print(dz.zd)
            rounded_zd = dict(dz.zd)
            # evaluate_assignment_score(dz, rounded_zd, shortage_limit = param.shortage, balance_limit = param.balance, lbfrl_limit= param.lbfrl, ubfrl_limit= param.ubfrl)

            zv.visualize_zones_from_dict(dz.zd, show=True, label=False)
            del dz


def load_initial_assignemt(dz, name, load_level = 'AA'):
    if load_level == "AA":
        aa_zoning = _get_zone_dict("/Users/mobin/SFUSD/Visualization_Tool_Data/GE" + name + "_AA" +".csv")
        print(aa_zoning)
        # aa_zv = ZoneVisualizer('idschoolattendance')
        # aa_zv.visualize_zones_from_dict(aa_zoning, centroid_location=dz.centroid_location, save_name= "GE" + name + "_AA")
        # dz.distance_and_neighbors_telemetry(centroid_choices)
        dz.zd = aa2level_Zoning(dz, aa_zoning)

    elif load_level == "BG":
        print("Loading assignment on Blockgroup level")
        dz.zd = _get_zone_dict("/Users/mobin/SFUSD/Visualization_Tool_Data/GE" + name + "_BG" +".csv")

    return dz


def aa2level_Zoning(dz, aa_zoning):
    blockgroup_zoning = {}

    # For the given attendance area level zoning, find it's equivalent assignment on blockgroup level.
    for blockgroup in dz.area2idx:
        if blockgroup in dz.bg2att:
            if dz.bg2att[blockgroup] in aa_zoning:
                blockgroup_zoning[blockgroup] = aa_zoning[dz.bg2att[blockgroup]]
            else:
                print("unexpected happened")
                # blockgroup_zoning[blockgroup] = 90

    for blockgroup in dz.area2idx:
        neighbor_count = {}
        # look over all neighbors that are assigned to a zone so far
        for neighbor in dz.neighbors[dz.area2idx[blockgroup]]:
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
        if (blockgroup in dz.bg2att):
            if len(neighbor_count) == 1:
                # make sure this blockgroup is not the centroid
                if blockgroup not in [dz.idx2area[dz.centroids[z]] for z in range(dz.M)]:
                    # select the first key in neighbor_count (we have made sure neighbor_count has only 1 key,
                    # which is the zone #of all neighbors of this blockgroup)
                    blockgroup_zoning[blockgroup] = list(neighbor_count.keys())[0]

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



# iteratively update the zoning. in each iteration:
# for each zone, go over all its neighbors, and for each of them, if adding it to that zone would increase the total objective value, change the zoning for that blockgroup
# otherwise, don't do anything and continue to the another neighbor of that zone
def iterative_update(param, dz, zone_dict, ZoneVisualizer, noiterations):
    for iteration in range(0, noiterations):
        # boundary cost fraction of this iteration
        iter_bcf = (iteration + 1)/noiterations
        # iter_dl = 0.15 + (noiterations-1 - iteration)/(noiterations-1) * 0.85
        # diversity limit of this iteration
        # iter_dl = 0
        # we find the value of the current assignment (given the ratio of boundary_cost)
        curnt_val = evaluate_assignment_score(param, dz, zone_dict, boundary_cost_fraction = iter_bcf)
        # ratio of boundary_cost tells us what fraction of boundary_coeff, should we consider
        # the idea is that in first rounds, the boundary cost should not be that important
        # -- so that it allows for more movement on the boundary of local search
        # and as time goes, we increase this cost, resulting in zones with more smooth boundary for the end result
        print("bcf value at the begining of iteration " + str(iteration) + "  is: " + str(curnt_val))
        if curnt_val > 1000000000000:
            print("ERROR")

        # _Zoning_1_1_1_2_1_centroids 6 - zone - 4
        # _Zoning_1_1_1_2_1_centroids 6 - zone - 4
        for level_area in zone_dict:
            for neighbor_idx in dz.neighbors[dz.area2idx[level_area]]:
                neighbor = dz.idx2area[neighbor_idx]
                # make sure the neighbor and level_area(aa/bg/b) are on the boundaries of their zones,
                # and belong to different zones
                if zone_dict[level_area] != zone_dict[neighbor]:
                    temp = zone_dict[neighbor]
                    zone_dict[neighbor] = zone_dict[level_area]
                    new_val = evaluate_assignment_score(param, dz, zone_dict, boundary_cost_fraction = iter_bcf)
                    if new_val <= curnt_val:
                        curnt_val = new_val
                        # print("new curnt_val " + str(curnt_val))
                    else:
                        # changing the boundaries in this direction would decrease the objective value so reverse back
                        zone_dict[neighbor] = temp
        print("bcf alue at the end of iteration " + str(iteration) + "  is: " + str(evaluate_assignment_score(param, dz, zone_dict, boundary_cost_fraction = iter_bcf)))
        print("Total value at the end of iteration " + str(iteration) + "  is: " + str(evaluate_assignment_score(param, dz, zone_dict)))
        ZoneVisualizer.visualize_zones_from_dict(zone_dict, centroid_location= dz.centroid_location)
    return zone_dict

def initialize_nonboundary(dz, zone_dict):
    for j in range(dz.A):
        for z in range(dz.M):
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
    dz.z = []
    for z in range(dz.M):
        dz.z.append([])
    for b in zone_dict:
        dz.z[zone_dict[b]].append(b)

def local_seach(param):
    name = compute_name(param)
    print(name)
    if os.path.exists(param.path + "GE" + name + "_BG_ls" +".csv"):
        return
    # if there is no png, it means it was either not considered yet, or
    # we looked for solution and non exists
    if not os.path.exists(param.path +"GE" + name + "_AA" +".png"):
        return

    input_level = 'BlockGroup'
    dz = DesignZones(
        M=param.zone_count,
        level=input_level,
        centroids_type=param.centroids_type,
        include_k8=param.include_k8,
        population_type=param.population_type,
        program_type="GE",
        drop_optout=True,
        capacity_scenario="A",
        new_schools=True,
    )
    dz = load_initial_assignemt(dz, name=name, load_level="AA")
    dz.zd = trim_boundary(dz, dz.zd)
    # heuristic(dz, dz.zd)
    # handle_anomolies(dz, dz.zd)
    # sanity_check(dz)
    # zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location)
    dz.zd = drop_boundary(dz, dz.zd)
    # zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location)

    dz.set_model(shortage=param.shortage, all_cap_shortage=param.all_cap_shortage, balance=param.balance)
    initialize_nonboundary(dz, dz.zd)

    dz._add_geo_constraints(max_distance=param.max_distance, contiguity=True)
    dz._add_diversity_constraints(racial_dev=param.racial_dev, frl_dev=param.frl_dev)
    dz._add_school_count_constraint()
    if param.include_sch_qlty_balance == True:
        dz._add_met_quality_constraint(min_pct = param.lbscqlty)
    # dz._boundary_threshold_constraint(boundary_threshold = param.boundary_threshold)

    # solve_success = 1
    solve_success = dz.solve()

    if solve_success == 1:
        zv = ZoneVisualizer(input_level)
        print("Resulting zone dictionary:")
        print(dz.zd)

        zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location, save_name="GE" + name + "_BG")
        dz.save(path=param.path, name = name + "_BG")

        # find local search solution, and save it
        iterative_update(param, dz, dz.zd, zv, noiterations = 3)
        update_z(dz, dz.zd)

        zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location, save_name= "GE" + name + "_BG_ls")
        dz.save(path=param.path, name = name + "_BG_ls")


    # # stats_evaluation(dz, dz.zd)
    # dz = load_initial_assignemt(dz, name=name, load_level="BG")
    # zv = ZoneVisualizer(input_level)
    # dz.zd = {60750209001: 0, 60750209002: 0, 60750209003: 0, 60750209004: 0, 60750210001: 0, 60750210002: 0, 60750210003: 0, 60750210004: 0, 60750211001: 0, 60750211002: 0, 60750211003: 0, 60750212001: 0, 60750212002: 0, 60750212003: 0, 60750213001: 0, 60750213002: 0, 60750214001: 0, 60750214002: 0, 60750214003: 0, 60750215001: 0, 60750215002: 0, 60750215003: 0, 60750215004: 0, 60750215005: 0, 60750216001: 0, 60750216002: 0, 60750217001: 0, 60750217002: 0, 60750217003: 0, 60750218001: 0, 60750218002: 0, 60750218003: 0, 60750218004: 0, 60750228033: 0, 60750229011: 0, 60750229012: 0, 60750229013: 0, 60750229021: 0, 60750229022: 0, 60750229031: 0, 60750229032: 0, 60750229033: 0, 60750230011: 0, 60750230012: 0, 60750230013: 0, 60750230031: 0, 60750230032: 0, 60750231022: 0, 60750231032: 0, 60750232002: 0, 60750232003: 0, 60750251001: 0, 60750251002: 0, 60750251003: 0, 60750252001: 0, 60750252002: 0, 60750252003: 0, 60750252004: 0, 60750253001: 0, 60750253002: 0, 60750253003: 0, 60750253004: 0, 60750254011: 0, 60750254012: 0, 60750254013: 0, 60750254021: 0, 60750254022: 0, 60750254023: 0, 60750254031: 0, 60750254032: 0, 60750255001: 0, 60750255002: 0, 60750256001: 0, 60750256002: 0, 60750256003: 0, 60750256004: 0, 60750257011: 0, 60750257013: 0, 60750257021: 0, 60750257023: 0, 60750258001: 0, 60750259001: 0, 60750259002: 0, 60750307001: 0, 60750307002: 0, 60750311001: 0, 60750612001: 0, 60750612002: 0, 60759806001: 0, 60750255003: 1, 60750255004: 1, 60750255005: 1, 60750255006: 1, 60750260011: 1, 60750260012: 1, 60750260021: 1, 60750260022: 1, 60750260031: 1, 60750260032: 1, 60750260041: 1, 60750260042: 1, 60750261001: 1, 60750261002: 1, 60750261003: 1, 60750261004: 1, 60750263011: 1, 60750263023: 1, 60750264011: 1, 60750264012: 1, 60750264031: 1, 60750264041: 1, 60750264042: 1, 60750307003: 1, 60750308001: 1, 60750308002: 1, 60750308003: 1, 60750308004: 1, 60750309001: 1, 60750309002: 1, 60750309003: 1, 60750309004: 1, 60750309005: 1, 60750310001: 1, 60750310002: 1, 60750310003: 1, 60750311002: 1, 60750311003: 1, 60750311004: 1, 60750311005: 1, 60750312011: 1, 60750312012: 1, 60750312013: 1, 60750312014: 1, 60750312021: 1, 60750312022: 1, 60750330002: 1, 60750330005: 1, 60750330006: 1, 60750351001: 1, 60750351002: 1, 60750351003: 1, 60750351004: 1, 60750351005: 1, 60750351006: 1, 60750351007: 1, 60750352011: 1, 60750352012: 1, 60750352013: 1, 60750352014: 1, 60750352015: 1, 60750352021: 1, 60750352022: 1, 60750352023: 1, 60750353002: 1, 60750353003: 1, 60750353005: 1, 60750353006: 1, 60750354001: 1, 60750354002: 1, 60750354003: 1, 60750354004: 1, 60750354005: 1, 60750605021: 1, 60750605022: 1, 60750605023: 1, 60759805011: 1, 60750170003: 2, 60750171012: 2, 60750171022: 2, 60750204011: 2, 60750204012: 2, 60750204013: 2, 60750204021: 2, 60750204022: 2, 60750205003: 2, 60750206002: 2, 60750211004: 2, 60750262001: 2, 60750262002: 2, 60750262003: 2, 60750262004: 2, 60750262005: 2, 60750263012: 2, 60750263013: 2, 60750263021: 2, 60750263022: 2, 60750263031: 2, 60750263032: 2, 60750301012: 2, 60750301013: 2, 60750301014: 2, 60750301021: 2, 60750301022: 2, 60750301023: 2, 60750302011: 2, 60750302012: 2, 60750302013: 2, 60750302021: 2, 60750302022: 2, 60750302023: 2, 60750303011: 2, 60750303012: 2, 60750303013: 2, 60750303014: 2, 60750303021: 2, 60750303022: 2, 60750303023: 2, 60750304001: 2, 60750304002: 2, 60750304003: 2, 60750304004: 2, 60750304005: 2, 60750305001: 2, 60750305002: 2, 60750305003: 2, 60750306001: 2, 60750306002: 2, 60750306003: 2, 60750308005: 2, 60750309006: 2, 60750309007: 2, 60750313011: 2, 60750313012: 2, 60750313013: 2, 60750313021: 2, 60750313022: 2, 60750313023: 2, 60750314001: 2, 60750314002: 2, 60750314003: 2, 60750314004: 2, 60750314005: 2, 60750326011: 2, 60750326012: 2, 60750326013: 2, 60750326021: 2, 60750326022: 2, 60750326023: 2, 60750327001: 2, 60750327002: 2, 60750327003: 2, 60750327004: 2, 60750327005: 2, 60750327006: 2, 60750327007: 2, 60750328011: 2, 60750328012: 2, 60750328013: 2, 60750328021: 2, 60750328022: 2, 60750328023: 2, 60750329011: 2, 60750329012: 2, 60750329013: 2, 60750329014: 2, 60750329021: 2, 60750329022: 2, 60750329023: 2, 60750330001: 2, 60750330003: 2, 60750330004: 2, 60750331001: 2, 60750331002: 2, 60750331003: 2, 60750331004: 2, 60750332011: 2, 60750332031: 2, 60750332032: 2, 60750332041: 2, 60750332042: 2, 60750332043: 2, 60750353001: 2, 60750353004: 2, 60750604001: 2, 60750124011: 3, 60750124021: 3, 60750124022: 3, 60750124023: 3, 60750126011: 3, 60750126021: 3, 60750126022: 3, 60750127001: 3, 60750127002: 3, 60750127003: 3, 60750128001: 3, 60750128002: 3, 60750128003: 3, 60750128004: 3, 60750129022: 3, 60750129023: 3, 60750130002: 3, 60750130003: 3, 60750130004: 3, 60750131021: 3, 60750131022: 3, 60750132001: 3, 60750132002: 3, 60750132003: 3, 60750133001: 3, 60750133002: 3, 60750133003: 3, 60750133004: 3, 60750133005: 3, 60750134001: 3, 60750134002: 3, 60750134003: 3, 60750135001: 3, 60750135002: 3, 60750152001: 3, 60750152002: 3, 60750152003: 3, 60750153001: 3, 60750153002: 3, 60750154001: 3, 60750154002: 3, 60750154003: 3, 60750154004: 3, 60750154005: 3, 60750155001: 3, 60750155002: 3, 60750155003: 3, 60750156001: 3, 60750156002: 3, 60750156003: 3, 60750157001: 3, 60750157003: 3, 60750157004: 3, 60750158012: 3, 60750160001: 3, 60750165004: 3, 60750176013: 3, 60750176014: 3, 60750178011: 3, 60750178012: 3, 60750178021: 3, 60750178022: 3, 60750180001: 3, 60750180002: 3, 60750227021: 3, 60750401001: 3, 60750401002: 3, 60750401003: 3, 60750401004: 3, 60750402001: 3, 60750402002: 3, 60750402003: 3, 60750402004: 3, 60750426011: 3, 60750426012: 3, 60750426021: 3, 60750426022: 3, 60750426023: 3, 60750427001: 3, 60750427002: 3, 60750427003: 3, 60750428001: 3, 60750428002: 3, 60750428003: 3, 60750451001: 3, 60750451002: 3, 60750451003: 3, 60750452001: 3, 60750452002: 3, 60750452003: 3, 60750452004: 3, 60750452005: 3, 60750476001: 3, 60750476002: 3, 60750476003: 3, 60750476004: 3, 60750477011: 3, 60750477012: 3, 60750477013: 3, 60750477021: 3, 60750477022: 3, 60750477023: 3, 60750478011: 3, 60750478012: 3, 60750478013: 3, 60750478021: 3, 60750478022: 3, 60750478023: 3, 60750479011: 3, 60750479012: 3, 60750479013: 3, 60750479014: 3, 60750479015: 3, 60750479021: 3, 60750479022: 3, 60750479023: 3, 60750601001: 3, 60750607001: 3, 60750607002: 3, 60750607003: 3, 60750615002: 3, 60750615004: 3, 60750615005: 3, 60750615006: 3, 60759802001: 3, 60759803001: 3, 60750101001: 4, 60750101002: 4, 60750102001: 4, 60750102002: 4, 60750102003: 4, 60750103001: 4, 60750103002: 4, 60750103003: 4, 60750104001: 4, 60750104002: 4, 60750104003: 4, 60750104004: 4, 60750105001: 4, 60750105002: 4, 60750106001: 4, 60750106002: 4, 60750106003: 4, 60750107001: 4, 60750107002: 4, 60750107003: 4, 60750107004: 4, 60750108001: 4, 60750108002: 4, 60750108003: 4, 60750109001: 4, 60750109002: 4, 60750109003: 4, 60750110001: 4, 60750110002: 4, 60750110003: 4, 60750111001: 4, 60750111002: 4, 60750111003: 4, 60750112001: 4, 60750112002: 4, 60750112003: 4, 60750113001: 4, 60750113002: 4, 60750117001: 4, 60750117002: 4, 60750118001: 4, 60750119011: 4, 60750119012: 4, 60750119021: 4, 60750119022: 4, 60750120001: 4, 60750120002: 4, 60750121001: 4, 60750121002: 4, 60750122011: 4, 60750122012: 4, 60750122021: 4, 60750123011: 4, 60750123012: 4, 60750123021: 4, 60750123022: 4, 60750124012: 4, 60750125011: 4, 60750125012: 4, 60750125021: 4, 60750125022: 4, 60750129011: 4, 60750129012: 4, 60750129021: 4, 60750130001: 4, 60750131011: 4, 60750131012: 4, 60750151001: 4, 60750151002: 4, 60750157002: 4, 60750158011: 4, 60750158013: 4, 60750158021: 4, 60750158022: 4, 60750159001: 4, 60750159002: 4, 60750161001: 4, 60750161002: 4, 60750161003: 4, 60750161004: 4, 60750162001: 4, 60750162002: 4, 60750162003: 4, 60750163001: 4, 60750163002: 4, 60750163003: 4, 60750164001: 4, 60750164002: 4, 60750165001: 4, 60750165002: 4, 60750165003: 4, 60750166001: 4, 60750166002: 4, 60750166003: 4, 60750166004: 4, 60750167001: 4, 60750167002: 4, 60750167003: 4, 60750167004: 4, 60750168011: 4, 60750168012: 4, 60750171011: 4, 60750171013: 4, 60750171021: 4, 60750171023: 4, 60750176012: 4, 60750176015: 4, 60750179021: 4, 60750301011: 4, 60750611001: 4, 60750611002: 4, 60750611003: 4, 60750615001: 4, 60750615003: 4, 60750176011: 4, 60750168013: 5, 60750168021: 5, 60750168022: 5, 60750168023: 5, 60750169001: 5, 60750169002: 5, 60750170001: 5, 60750170002: 5, 60750177001: 5, 60750177002: 5, 60750201001: 5, 60750201002: 5, 60750201003: 5, 60750201004: 5, 60750202001: 5, 60750202002: 5, 60750202003: 5, 60750203001: 5, 60750203002: 5, 60750203003: 5, 60750205001: 5, 60750205002: 5, 60750206001: 5, 60750206003: 5, 60750206004: 5, 60750207001: 5, 60750207002: 5, 60750207003: 5, 60750208001: 5, 60750208002: 5, 60750208003: 5, 60750208004: 5, 60750226001: 5, 60750226002: 5, 60750227022: 5, 60750227041: 5, 60750227042: 5, 60750228011: 5, 60750228012: 5, 60750228013: 5, 60750228021: 5, 60750228022: 5, 60750228031: 5, 60750228032: 5, 60750231021: 5, 60750231031: 5, 60750232001: 5, 60750233001: 5, 60750234001: 5, 60750234002: 5, 60750257012: 5, 60750257022: 5, 60750258002: 5, 60750259003: 5, 60750264021: 5, 60750264022: 5, 60750264023: 5, 60750264032: 5, 60750610001: 5, 60750610002: 5, 60750614001: 5, 60750614002: 5, 60750614003: 5, 60759809001: 5}
    # iterative_update(dz, dz.zd, zv, noiterations=5)
    # zv.visualize_zones_from_dict(dz.zd, centroid_location=dz.centroid_location, save_name= "GE" + name + "_BG_ls")




if __name__ == "__main__":
    param = Tuning_param()
    local_seach(param)

    # for frl_dev in [0.15, 0.1]:
    #     param.frl_dev = frl_dev
    #     for racial_dev in [0.15, 0.12]:
    #         param.racial_dev = racial_dev
    #         for include_k8 in [True, False]:
    #             param.include_k8 = include_k8
    #             with open("../Config/centroids.yaml", "r") as f:
    #                 centroid_options = yaml.safe_load(f)
    #                 for centroids in centroid_options:
    #                     param.zone_count = int(centroids.split("-")[0])
    #                     param.centroids_type = centroids
    #                     print("param: " + str(param.frl_dev) + " " + str(param.racial_dev)
    #                          + " " + str(param.include_k8))
    #                     local_seach(param)



    # TODO
    # make change to iteration:
    # 1-  swap 2, instead of increament/decreament
    # 2- to make 1 faster, local search updates, instead of recomputing

    # File "/Users/mobin/PycharmProjects/sfusd-project/optimization/Zone_Generation/generate_zones.py", line 65, in _get_zone_dict
    # with open(assignment_name, "r") as f:
    # FileNotFoundError: [Errno 2] No such file or directory: '/Users/mobin/SFUSD/Visualization_Tool_Data/GE_Zoning_1_2_3_2_2_centroids 8-zone-28_BG.csv'
