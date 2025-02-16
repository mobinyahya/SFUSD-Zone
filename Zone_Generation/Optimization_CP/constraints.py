import csv
import json
import os
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

from Zone_Generation.Optimization_CP import constants
from Zone_Generation.Optimization_CP.constants import SCALING_FACTOR, RACES


def add_constraints(model, vm, school_df, bdf, centroids):
    # Every centroid must have the block the school is in assigned to it
    # for zone in centroids:
    # MBES block assigned to mission bay, webster block assigned to webster
    mbes_block = school_df[school_df['school_id'] == 999]['Block'].iloc[0]
    webster_block = school_df[school_df['school_id'] == 497]['Block'].iloc[0]
    model.Add(vm[mbes_block] == 1)
    model.Add(vm[webster_block] == 0)
    # add_school_number_constraints(model, vm, school_df, bdf, centroids)
    # add_zone_capacity_constraints(model, vm, school_df, bdf, centroids)
    # add_zone_duplicates_constraints(model, vm, school_df, bdf, centroids)
    add_contiguity_constraints(model, vm, school_df, bdf, centroids)
    # add_diversity_constraints(model, vm, school_df, bdf, centroids)
    add_boundary_constraint(model, vm, school_df, bdf, centroids)
    add_mission_bay(model, vm, school_df, bdf, centroids)
    add_potrero_hill(model, vm, school_df, bdf, centroids)
    add_dogpatch(model, vm, school_df, bdf, centroids)
    add_zone_capacity_ratio_constraints(model, vm, school_df, bdf, centroids)

def add_potrero_hill(model, vm, school_df, bg_df, centroids):
    neighborhood_blocks = pd.read_csv('Webster Census Blocks, grouped by neighborhood - Sheet1.csv')
    ph_blocks = neighborhood_blocks[neighborhood_blocks['Geoid20 (group) 11'] == 'Potrero Hill']['Geoid20'].values
    ph_blocks = [int(x) for x in ph_blocks]
    for block in ph_blocks:
        model.Add(vm[block] == 0)

def add_dogpatch(model, vm, school_df, bg_df, centroids):
    neighborhood_blocks = pd.read_csv('Webster Census Blocks, grouped by neighborhood - Sheet1.csv')
    dp_blocks = neighborhood_blocks[neighborhood_blocks['Geoid20 (group) 11'] == 'Dogpatch']['Geoid20'].values
    dp_blocks = [int(x) for x in dp_blocks]
#     set all variables as equal
    last_block = dp_blocks[0]
    for i in range(1, len(dp_blocks)):
        model.Add(vm[dp_blocks[i]] == vm[last_block])
        last_block = dp_blocks[i]



def add_zone_capacity_ratio_constraints(model, vm, school_df, bg_df, centroids):
    block_values = list(vm.values())

    bg_counts = (bg_df['total']).round().astype(int).tolist()

    mbes_students = cp_model.LinearExpr.WeightedSum(block_values, bg_counts)

    total_students = int(bg_df['total'].sum())
    #   the ratio between MBES should have at most 3x the number of students as Webster
    #   and at least the same number of students
    ratio_thing = constants.CONFIG['max_pop_ratio']
    # do the math on why this works
    model.Add(ratio_thing* total_students >= (1+ratio_thing) * mbes_students)
    model.Add(total_students <= 2 * mbes_students)



def add_mission_bay(model, vm, school_df, bdf, centroids):
    # mbes_blocks = [60750607013001, 60750607013007, 60750226002003, 60750607013015, 60750607013015, 60750226002004, 60750226002004, 60750607012002, 60750607012001, 60750607013013]
    neighborhood_blocks = pd.read_csv('Webster Census Blocks, grouped by neighborhood - Sheet1.csv')
    mbes_blocks = neighborhood_blocks[neighborhood_blocks['Geoid20 (group) 11'] == 'Mission Bay']['Geoid20'].values
    mbes_blocks = [int(x) for x in mbes_blocks]
    for block in mbes_blocks:
        model.Add(vm[block] == 1)


def add_boundary_constraint(model, vm, school_df, bg_df, centroids):
    boundary_vars = []

    with open('neighbors.json', 'r') as f:
        neighbors = json.load(f)
    for bg in vm:
        if str(bg) not in neighbors:
            print(bg)
            continue
        for neighbor in neighbors[str(bg)]:
            if neighbor == '':
                continue
            # if neighbor not in bg_df['Block'].values:
            #     continue
            # neighbor = float(neighbor)
            neighbor = int(neighbor)
            if neighbor not in vm:
                continue
            b = model.NewBoolVar(f"boundary_{bg}_{neighbor}")
            #             minimize the number of neighbors with different zoning
            model.Add(vm[bg] != vm[neighbor]).OnlyEnforceIf(b)
            model.Add(vm[bg] == vm[neighbor]).OnlyEnforceIf(b.Not())

            boundary_vars.append(b)

    model.Add(sum(boundary_vars) < 150)


def add_frl_constraints(model, vm, school_df, bg_df, centroids):
    #     The FRL percentage of the zone must be within 15% of the average FRL percentage

    frl_min = int(
        ((bg_df['FRL'].sum() / bg_df['student_count'].sum()) - 0.15) * SCALING_FACTOR)
    frl_max = int(
        ((bg_df['FRL'].sum() / bg_df['student_count'].sum()) + 0.15) * SCALING_FACTOR)
    for zone in centroids:
        frl_coef = (bg_df['FRL'] * SCALING_FACTOR).round().astype(int).tolist()
        tcoef = (bg_df['student_count']).round().astype(int).tolist()
        block_values = list(vm[zone].values())
        frl_block_sum = cp_model.LinearExpr.WeightedSum(block_values, frl_coef)
        total_block_sum = cp_model.LinearExpr.WeightedSum(block_values, tcoef)
        model.Add(frl_block_sum >= total_block_sum * frl_min)
        model.Add(frl_block_sum <= total_block_sum * frl_max)


def add_diversity_constraints(model, vm, school_df, bg_df, centroids):
    # All zones must have more than 15% less of the average number of any group (FRL, White, Asian, Latino)

    for zone in centroids:
        for race in RACES:
            # TODO: Check that this this is an equivalent constraint to the one in the paper
            # print(race, bg_df[race].sum(), 'total', bg_df['student_count'].sum())

            race_min = int(((bg_df[race].sum() / bg_df['total'].sum()) - 0.15) * SCALING_FACTOR)
            race_max = int(((bg_df[race].sum() / bg_df['total'].sum()) + 0.15) * SCALING_FACTOR)
            rcoef = (bg_df[race] * SCALING_FACTOR).round().astype(int).tolist()
            tcoef = (bg_df['total']).round().astype(int).tolist()
            block_values = list(vm[zone].values())
            race_block_sum = cp_model.LinearExpr.WeightedSum(block_values, rcoef)
            total_block_sum = cp_model.LinearExpr.WeightedSum(block_values, tcoef)
            # r/t > rmin = r> rmin * t
            # rmin = (R/T - 0.15)
            # r > (R/T - 0.15) * t
            # r * scaler > (R - 0.15) * scaler * t

            model.Add(race_block_sum > total_block_sum * race_min)
            model.Add(race_block_sum < total_block_sum * race_max)


def add_contiguity_constraints(model, vm, school_df, bg_df, centroids):
    # create dictionary mapping block group number to list of neighbor block
    # file = os.path.expanduser(
    #     "~/Dropbox/SFUSD/Optimization/b_adjacency_matrix_20.csv"
    # )
    # with open(file, "r") as f:
    #     reader = csv.reader(f)
    #     adjacency_matrix = list(reader)
    # file = os.path.expanduser(
    #     '~/Dropbox/SFUSD/Optimization/distances_b2b_20.csv'
    # )
    # with open(file, 'r') as f:
    #     # ignore first row
    #     reader = csv.reader(f)
    #     travel_matrix = list(reader)
    #     # travel_matrix = travel_matrix[1:]
    # #
    # all_blocks = set(bg_df['Block'].values)
    # # # only take blocks in bg_df
    # # # create 2d dictionary mapping block group number to block group number to travel time
    # school_bgs = set(school_df['Block'].astype(int).values)
    # travels = defaultdict(dict)
    # for i in range(len(travel_matrix)):
    #     start = int(travel_matrix[i][0])
    #     dest = int(travel_matrix[i][1])
    #     if start in all_blocks and dest in all_blocks:
    #         travels[start][dest] = float(travel_matrix[i][2])
    #         travels[dest][start] = float(travel_matrix[i][2])
    # # save travels to file
    # with open('travels.json', 'w') as f:
    #     json.dump(travels, f)
    #
    #
    # # create dictionary mapping attendance area school id to list of neighbor
    # # attendance area ids (similarly, block group number)
    # neighbors = {}
    # for row in adjacency_matrix:
    #     # cast all vals in row to int
    #     if int(row[0]) not in all_blocks:
    #         continue
    #     neighbors[int(row[0])] = list(set([int(x) for x in row[1:] if x != '' and int(x) in all_blocks]))
    #     if len(neighbors[int(row[0])]) == 0:
    #         print('No neighbors for ', row[0])

    # with open('neighbors.json', 'w') as f:
    #     json.dump(neighbors, f)
    # print('yippie yi oh')
    with open('neighbors.json', 'r') as f:
        neighbors = json.load(f)
    with open('travels.json', 'r') as f:
        travels = json.load(f)
    # all_blocks_to_maybe_add = set()
    weird_blocks = set()
    centroid_bgs = []
    for zone in centroids:
        centroid_bgs.append(str(school_df[school_df['school_id'] == zone]['Block'].iloc[0]))

    for zone in centroids:
        zone_bg = school_df[school_df['school_id'] == zone]['Block'].iloc[0]
        for bg in vm:
            bg = str(int(bg))
            zone_bg = str(int(zone_bg))
            if bg in centroid_bgs:
                continue
            bg_distance_to_zone = travels[bg][zone_bg]
            neighbors_closer = set()
            # neighbors_not_in_blocks = set()
            all_neighbors = set()
            for neighbor in neighbors[bg]:
                neighbor = str(int(neighbor))
                if int(neighbor) not in vm:
                    # neighbors_not_in_blocks.add(neighbor)
                    continue
                # if neighbor is a centroid, this must be assigned to that centroid
                if neighbor in centroid_bgs:
                    if int(neighbor) == 60750607012013:
                        model.Add(vm[int(bg)] == 1)
                    else:
                        model.Add(vm[int(bg)] == 0)
                all_neighbors.add(neighbor)
                neighbor_distance_to_zone = travels[neighbor][zone_bg]
                if neighbor_distance_to_zone <= bg_distance_to_zone:
                    neighbors_closer.add(float(neighbor))
            if len(neighbors_closer) == 0:
                print('No closer neighbors ', bg)
                # print(neighbors_not_in_blocks)
                weird_blocks.add(bg)
                print(len(all_neighbors))
            closer_neigbor_vars = [vm[int(n)] for n in neighbors_closer]
            if zone == 999:
                model.Add(sum(closer_neigbor_vars) >= 1).OnlyEnforceIf(vm[int(bg)])
            else:
                model.Add(sum(closer_neigbor_vars) < len(neighbors_closer)).OnlyEnforceIf(vm[int(bg)].Not())
            # model.Add(sum(vm[zone][int(n)] for n in neighbors_closer) >= 1).OnlyEnforceIf(vm[zone][int(bg)])
    print(weird_blocks)
    # print(all_blocks_to_maybe_add)
    # print(f'Block groups not in travels: {bgs_not_in_travels}')
    # print(len(bgs_not_in_travels))


def add_school_number_constraints(model, vm, school_df, bg_df, centroids):
    schools_per_zone = len(school_df.index) // len(centroids)
    #     The number of schools in the zone must be equal across zones
    for zone in centroids:
        schools_in_zone = 0
        for bg in vm[zone]:
            if bg in school_df['Block'].values:
                schools_in_zone += vm[zone][bg]
        model.Add(schools_in_zone == schools_per_zone)


def add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids):
    # The total capacity should be within 15% of the total number of students
    for zone in centroids:
        block_values = list(vm[zone].values())
        zone_capacity_coefs = pd.Series([get_bg_for_school(school_df, bg) for bg in vm[zone]])
        zone_capacity_coefs_max = (zone_capacity_coefs * SCALING_FACTOR * 1.15).round().astype(int).tolist()
        zone_capacity_coefs_min = (zone_capacity_coefs * SCALING_FACTOR * 0.85).round().astype(int).tolist()

        bg_counts = (bg_df['total'] * SCALING_FACTOR).round().astype(int).tolist()

        print(sum(zone_capacity_coefs_max) / SCALING_FACTOR, sum(zone_capacity_coefs_min) / SCALING_FACTOR,
              sum(bg_counts) / SCALING_FACTOR)
        zone_capacity_min = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_min)
        zone_capacity_max = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_max)
        zone_students = cp_model.LinearExpr.WeightedSum(block_values, bg_counts)
        # the number of students cannot be more than 15% greater than the capacity
        model.Add(zone_students <= zone_capacity_max)
        # the number of students cannot be less than 15% less than the capacity
        model.Add(zone_students >= zone_capacity_min)


def add_zone_duplicates_constraints(model, vm, school_df, bdf, centroids):
    #  Every blockgroup should be assigned to exactly one zone
    for bg in bdf['Block']:
        model.Add(sum([vm[zone][bg] for zone in centroids]) == 1)


def get_bg_for_school(school_df, bg):
    # Find the school in that blockgroup
    if bg in school_df['Block'].values:
        return school_df[school_df['Block'] == int(bg)].iloc[0]['capacity']
    else:
        return 0
