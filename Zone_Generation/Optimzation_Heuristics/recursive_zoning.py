import os, sys, json

sys.path.append('../..')
sys.path.append('../../summary_statistics')
from Graphic_Visualization.zone_viz import ZoneVisualizer
from IP_Zoning import DesignZones
from Local_Search_Zoning import evaluate_assignment_score
from Local_Search_Zoning import Tuning_param


# target_subdivision = 0
# max_number = 15
# old_zd, candidates = read_zoning(target_subdivision)
#
# def make_new_zone_dictionary(old_zd, change_zd, M, max_number):
#     new_zd = {}
#     for area in old_zd:
#         if int(area) in change_zd:
#             # new_zd[int(area)] = old_zd[area] +  change_zd[int(area)] / M
#             new_zd[int(area)] = max_number +  change_zd[int(area)] + 1
#         else:
#             new_zd[int(area)] = old_zd[area]
#     return new_zd
#
# def save_zoning(new_zd):
#     a_file = open("/Users/mobin/Documents/sfusd/local_runs/Recursive/zoning_assignment1.json", "w")
#     json.dump(new_zd, a_file)
#     a_file.close()
#
# def read_zoning(target_subdivision):
#     with open('/Users/mobin/Documents/sfusd/local_runs/Recursive/zoning_assignment.json') as f:
#         saved_zd = json.load(f)
#     print(saved_zd)
#     f.close()
#
#     # old_zd = {420: 0, 456: 0, 478: 0, 497: 0, 505: 0, 603: 0, 644: 0, 718: 0, 723: 0, 816: 0, 453: 1, 481: 1, 488: 1, 507: 1, 513: 1, 521: 1, 539: 1, 575: 1, 593: 1, 614: 1, 625: 1, 656: 1, 670: 1, 680: 1, 691: 1, 722: 1, 729: 1, 746: 1, 750: 1, 820: 1, 830: 1, 838: 1, 842: 1, 862: 1, 867: 1, 876: 1, 413: 2, 435: 2, 490: 2,
#     #  525: 2, 544: 2, 549: 2, 562: 2, 569: 2, 589: 2, 638: 2, 650: 2, 664: 2, 735: 2, 782: 2, 786: 2, 790: 2, 801: 2, 823: 2, 834: 2, 848: 2, 859: 2, 872: 2}
#
#     candidates = []
#     for area in saved_zd:
#         if saved_zd[area] == target_subdivision:
#             candidates.append(int(area))
#
#     print("candidates")
#     print(candidates)
#
#     return saved_zd, candidates

def Run_Zoning(centroid_choices = None, candidates = None, subdivision_count = 2, balance_limit = 200, shortage_limit = 100, lbfrl_limit = 0.5, ubfrl_limit = 1.5):
    print("step1")

    if candidates == None:
        dz= DesignZones(M= subdivision_count, level='BlockGroup', centroidsType=0, include_citywide=False)
    else:
        dz = DesignZones(M= subdivision_count, level='BlockGroup', centroidsType=0, include_citywide=False, candidates=candidates)

    if centroid_choices != None:
        dz.update_centroid_telemetry(centroid_choices)
        dz.initializeNeighborsPerCentroids()

    dz.setmodel(shortage = balance_limit, balance = shortage_limit)
    dz.addGeoConstraints(maxdistance=-1, realDistance=True, coverDistance=False, strong_contiguity=True, neighbor=False, boundary=True)
    dz._add_diversity_constraints(lbfrl= lbfrl_limit, ubfrl= ubfrl_limit)

    # make sure the original centroid schools are balanced
    dz.m.addConstr(dz.x[dz.area2idx[dz.sch2area[481]], 0] + dz.x[dz.area2idx[dz.sch2area[862]], 0] + dz.x[dz.area2idx[dz.sch2area[872]], 0] <= 2)
    dz.m.addConstr(dz.x[dz.area2idx[dz.sch2area[859]], 1] + dz.x[dz.area2idx[dz.sch2area[862]], 1] + dz.x[dz.area2idx[dz.sch2area[872]], 1] <= 2)

    dz.solve()

    zv = ZoneVisualizer('BlockGroup')
    zv.zones_from_dict(dz.zone_dict, show=True, label=False)
    return dz


def evalute_final_zoning(dz1, dz2):
    final_dz = DesignZones(M=4, level='BlockGroup', centroidsType=-1, include_citywide=False)

    final_dz.update_centroid_telemetry(dz1.centroid_aa + dz2.centroid_aa)
    final_dz.initializeNeighborsPerCentroids()

    param = Tuning_param()
    final_dz.setmodel(shortage = param.shortage, balance = param.balance)

    final_dz.zone_dict = {**dz1.zone_dict, **dz2.zone_dict}
    evaluate_assignment_score(final_dz, final_dz.zone_dict, shortage_limit = param.shortage, balance_limit = param.balance, lbfrl_limit= param.lbfrl, ubfrl_limit= param.ubfrl)

    zv = ZoneVisualizer('BlockGroup')
    zv.zones_from_dict(final_dz.zone_dict, show=True, label=False)

# def evalute_final_zoning(dz1, dz2, dz3, dz4):
#     final_dz = DesignZones(M=8, level='BlockGroup', centroidsType=-1, include_citywide=False)
#
#     final_dz.update_centroid_telemetry(dz1.choices + dz2.choices + dz3.choices + dz4.choices)
#     final_dz.initializeNeighborsPerCentroids()
#
#     param = Tuning_param()
#     final_dz.setmodel(shortage = param.shortage, balance = param.balance)
#
#     final_dz.zd = {**dz1.zd, **dz2.zd, **dz3.zd, **dz4.zd}
#     print("This is final_dz.zd")
#     print(final_dz.zd)
#     evaluate_assignment_score(final_dz, final_dz.zd, shortage_limit = param.shortage, balance_limit = param.balance, lbfrl_limit= param.lbfrl, ubfrl_limit= param.ubfrl)
#
#     zv = ZoneVisualizer('BlockGroup')
#     zv.visualize_zones_from_dict(final_dz.zd,show=True,label=False)


if __name__ == "__main__":
    print("Step 0")
    round1_dz = Run_Zoning(centroid_choices = [481, 859], subdivision_count= 2)
    # print("Step 1")
    # round2_dz_A = Run_Zoning(centroid_choices = [481, 862], candidates = {key:value for key, value in round1_dz.zd.items() if value == 0}, subdivision_count= 2)
    # print("Step 2")
    # round2_dz_B = Run_Zoning(centroid_choices = [859, 872], candidates = {key:value for key, value in round1_dz.zd.items() if value == 1}, subdivision_count= 2)
    # print("Step 3")
    # # update the value (~zone number) for blockgroup zone assignment dictionary,
    # # since we should start from zone number 2 (zone number 0,1 is already used in subdivision round2_dz_A)
    # round2_dz_B.zd = {key: (value + 2) for key, value in round2_dz_B.zd.items() if value in {0, 1}}
    # evalute_final_zoning(round2_dz_A, round2_dz_B)




    # round1_dz = Run_Zoning(centroid_choices = [481, 859], balance_limit = 50, shortage_limit = 30, lbfrl_limit = 0.7, ubfrl_limit = 1.3)
    #
    # round2_1_dz = Run_Zoning(candidates = {key:value for key, value in round1_dz.zd.items() if value == 0}, balance_limit = 80, shortage_limit = 50, lbfrl_limit = 0.7, ubfrl_limit = 1.3)
    # round2_2_dz = Run_Zoning(candidates = {key:value for key, value in round1_dz.zd.items() if value == 1}, balance_limit = 80, shortage_limit = 50, lbfrl_limit = 0.7, ubfrl_limit = 1.3)
    #
    # round3_1_1_dz = Run_Zoning(candidates = {key:value for key, value in round2_1_dz.zd.items() if value == 0}, balance_limit = 250, shortage_limit = 150, lbfrl_limit = 0.6, ubfrl_limit = 1.4)
    # round3_1_2_dz = Run_Zoning(candidates = {key:value for key, value in round2_1_dz.zd.items() if value == 1}, balance_limit = 250, shortage_limit = 150, lbfrl_limit = 0.6, ubfrl_limit = 1.4)
    # #
    # round3_1_2_dz.zd = {key: (value + 2) for key, value in round3_1_2_dz.zd.items() if value in {0, 1}}
    #
    # round3_2_1_dz = Run_Zoning(candidates = {key:value for key, value in round2_2_dz.zd.items() if value == 0}, balance_limit = 250, shortage_limit = 150, lbfrl_limit = 0.6, ubfrl_limit = 1.4)
    # round3_2_2_dz = Run_Zoning(candidates = {key:value for key, value in round2_2_dz.zd.items() if value == 1}, balance_limit = 250, shortage_limit = 150, lbfrl_limit = 0.6, ubfrl_limit = 1.4)
    # #
    # round3_2_1_dz.zd = {key: (value + 4) for key, value in round3_2_1_dz.zd.items() if value in {0, 1}}
    # round3_2_2_dz.zd = {key: (value + 6) for key, value in round3_2_2_dz.zd.items() if value in {0, 1}}
    #
    # evalute_final_zoning(round3_1_1_dz, round3_1_2_dz, round3_2_1_dz, round3_2_2_dz)




    # a = [481, 859]
    # b = [2, 1]
    # print(a + b)

    # a_dictionary = {"a": 1, "b": 2, "c": 3, "d": 4}
    # keys_to_extract = ["a", "c"]
    # a_subset = {key: (value+1) for key, value in a_dictionary.items() if value in {1, 2}}
    # print(a_subset)