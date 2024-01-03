import csv
import itertools
import os
from collections import defaultdict
from math import ceil

import pandas as pd
from ortools.sat.python import cp_model

from Zone_Generation.Optimization_CP.constants import SCALING_FACTOR, RACES


def add_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping):
    # Every centroid must have the block the school is in assigned to it
    for zone in centroids:
        bg = get_bg_of_school(school_df, zone)
        model.Add(vm[bg] == centroid_mapping[zone])
    add_school_number_constraints(model, vm, school_df, bg_df, centroids,
                                  centroid_mapping)  # hard to convert to integer
    # add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids) #not needed since true by definition
    add_contiguity_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping)  # easy to convert to integer
    # All of these are essentially the exact same problem
    # add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids) #hard to convert to integer
    # add_diversity_constraints(model, vm, school_df, bg_df, centroids) #hard to convert to integer
    # add_frl_constraints(model, vm, school_df, bg_df, centroids)#hard to convert to integer


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
    #    All zones must have more than 15% less of the average number of any group (FRL, White, Asian, Latino)

    for zone in centroids:
        for race in RACES:
            # TODO: Check that this this is an equivalent constraint to the one in the paper
            # print(race, bg_df[race].sum(), 'total', bg_df['student_count'].sum())

            race_min = int(((bg_df[race].sum() / bg_df['student_count'].sum()) - 0.15) * SCALING_FACTOR)
            race_max = int(((bg_df[race].sum() / bg_df['student_count'].sum()) + 0.15) * SCALING_FACTOR)
            rcoef = (bg_df[race] * SCALING_FACTOR).round().astype(int).tolist()
            tcoef = (bg_df['student_count']).round().astype(int).tolist()
            block_values = list(vm[zone].values())
            race_block_sum = cp_model.LinearExpr.WeightedSum(block_values, rcoef)
            total_block_sum = cp_model.LinearExpr.WeightedSum(block_values, tcoef)
            # r/t > rmin = r> rmin * t
            # rmin = (R/T - 0.15)
            # r > (R/T - 0.15) * t
            # r * scaler > (R - 0.15) * scaler * t

            model.Add(race_block_sum > total_block_sum * race_min)
            model.Add(race_block_sum < total_block_sum * race_max)


def add_contiguity_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping):
    #     There exists some blockgroup in the zone that is closer to the centroid than the current blockgroup for all blockgroups

    # create dictionary mapping block group number to list of neighbor block
    file = os.path.expanduser(
        "~/Dropbox/SFUSD/Optimization/block_group_adjacency_matrix.csv"
    )
    with open(file, "r") as f:
        reader = csv.reader(f)
        adjacency_matrix = list(reader)
    file = os.path.expanduser(
        '~/Dropbox/SFUSD/Optimization/bg2bg_distances.csv'
    )
    with open(file, 'r') as f:
        reader = csv.reader(f)
        travel_matrix = list(reader)
    travels = defaultdict(dict)
    # create 2d dictionary mapping block group number to block group number to travel time
    for i in range(1, len(travel_matrix)):
        for j in range(1, len(travel_matrix[i])):
            if (travel_matrix[i][j] == '' or travel_matrix[i][0] == '' or travel_matrix[0][j] == ''):
                continue

            travels[int(travel_matrix[i][0])][int(travel_matrix[0][j])] = float(travel_matrix[i][j])

    # create dictionary mapping attendance area school id to list of neighbor
    # attendance area ids (similarly, block group number)
    neighbors = {}
    for row in adjacency_matrix:
        neighbors[row[0]] = set(row[1:])
    centroid_bgs = []
    for zone in centroids:
        centroid_bgs.append(school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0])
    for bg in vm.keys():
        if bg in centroid_bgs:
            continue
        #     cant be smart and avoid creating extra variables, because each variable has a different travel time to each zone
        all_neighbors = neighbors[str(int(bg))]
        neighbor_assigned_to_same = {}
        for neighbor in all_neighbors:
            if neighbor == '':
                continue
            neighbor = int(neighbor)
            if float(neighbor) not in vm.keys():
                continue
            neighbor_assigned_to_same[neighbor] = model.NewBoolVar(f'neighbor_assigned_to_same_{bg}_{neighbor}')
            model.Add(vm[bg] == vm[neighbor]).OnlyEnforceIf(neighbor_assigned_to_same[neighbor])
            model.Add(vm[bg] != vm[neighbor]).OnlyEnforceIf(neighbor_assigned_to_same[neighbor].Not())
        for zone in centroids:
            zone_bg = school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0]
            closer_neighbors = set()
            for neighbor in all_neighbors:
                if neighbor == '':
                    continue
                neighbor = int(neighbor)
                if float(neighbor) not in vm.keys():
                    continue
                neighbor_distance_to_zone = travels[neighbor][zone_bg]
                bg_distance_to_zone = travels[int(bg)][int(zone_bg)]
                if neighbor_distance_to_zone < bg_distance_to_zone:
                    closer_neighbors.add(neighbor_assigned_to_same[neighbor])
            #         if this block group is assigned to this zone, then at least one closer neighbor must be equal to bg
            assigned_to_zone = model.NewBoolVar(f'assigned_to_zone_{bg}_{zone}')
            model.Add(vm[bg] == centroid_mapping[zone]).OnlyEnforceIf(assigned_to_zone)
            model.Add(vm[bg] != centroid_mapping[zone]).OnlyEnforceIf(assigned_to_zone.Not())
            model.AddAtLeastOne(closer_neighbors).OnlyEnforceIf(assigned_to_zone)

    # for zone in centroids:
    #     zone_bg = school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0]
    #     for bg in vm[zone]:
    #         if bg in centroid_bgs:
    #             continue
    #
    #         bg_distance_to_zone = travels[int(bg)][int(zone_bg)]
    #         neighbors_closer = set()
    #         all_neighbors = set()
    #         for neighbor in neighbors[str(int(bg))]:
    #             if neighbor == '':
    #                 continue
    #             neighbor = int(neighbor)
    #             if float(neighbor) not in vm[zone]:
    #                 continue
    #             all_neighbors.add(float(neighbor))
    #             neighbor_distance_to_zone = travels[neighbor][zone_bg]
    #             if neighbor_distance_to_zone < bg_distance_to_zone:
    #                 neighbors_closer.add(float(neighbor))
    #         model.Add(sum(vm[zone][n] for n in neighbors_closer) >= vm[zone][bg])


def add_school_number_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping):
    # if m is the number of blocks, and n is the number of zones, then we can iterate over ever variable in the model
    # such that variable i is equal

    school_in_bg = []
    for bg in vm.keys():
        if bg in school_df['BlockGroup'].values:
            school_in_bg.append(vm[bg])

    num_school_blocks = len(school_in_bg)
    num_zones = len(centroids)
    schools_per_zone = ceil(num_school_blocks / num_zones)
    alt_schools_per_zone = schools_per_zone - 1
    schools_in_zone = {}
    for bg in school_in_bg:
        for i in range(num_zones):
            equal_var = model.NewBoolVar(f'equal_{bg}_{i}')
            model.Add(bg == i).OnlyEnforceIf(equal_var)
            model.Add(bg != i).OnlyEnforceIf(equal_var.Not())
            if i not in schools_in_zone:
                schools_in_zone[i] = []
            schools_in_zone[i].append(equal_var)
    for zone in schools_in_zone:
        model.Add(sum(schools_in_zone[zone]) >= schools_per_zone - 1)
        model.Add(sum(schools_in_zone[zone]) <= schools_per_zone)

    # mapped_vm = {}
    # for bg in school_in_bg:
    #     mapped_vm[str(bg)] = model.NewIntVar(1, num_school_blocks ** (num_zones - 1), f'mapped_{bg}')
    #     for i in range(num_zones):
    #         mapped_val = int((num_school_blocks+1) ** i)
    #         mapped_var = mapped_vm[str(bg)]
    #         intermediate_var = model.NewBoolVar(f'intermediate_{bg}_{i}_1')
    #         intermediate_var_2 = model.NewBoolVar(f'intermediate_{bg}_{i}_2')
    #         model.Add((bg == i)).OnlyEnforceIf(intermediate_var)
    #         model.Add((bg != i)).OnlyEnforceIf(intermediate_var.Not())
    #         model.Add((mapped_var == mapped_val)).OnlyEnforceIf(intermediate_var_2)
    #         model.Add((mapped_var != mapped_val)).OnlyEnforceIf(intermediate_var_2.Not())
    #         model.AddImplication(intermediate_var, intermediate_var_2)
    #         model.AddImplication(intermediate_var_2, intermediate_var)
    # model.Add((mapped_var == mapped_val)).OnlyEnforceIf(intermediate_var)
    # model.Add((mapped_var != mapped_val)).OnlyEnforceIf(intermediate_var.Not())
    #
    # mapped_sum = cp_model.LinearExpr.Sum(list(mapped_vm.values()))
    # think of the sum as a number in base num_school_blocks
    # check if the ith digit is schools_per_zone or schools_per_zone - 1

    # for i in range(num_zones):
    #     # check if ith digits is school per zone or school per zone - 1
    #     lb = 1
    #     ub = num_school_blocks ** (num_zones - 1 - i)
    #     division_thing = model.NewIntVar(lb, ub, f'division_{i}')
    #     divisor = num_school_blocks ** i
    #     model.AddDivisionEquality(division_thing, mapped_sum, divisor)
    #     ith_digit = model.NewIntVar(0, num_school_blocks, f'ith_digit_{i}')
    #     model.AddModuloEquality(ith_digit, division_thing, num_school_blocks)
    #     model.Add(ith_digit >= alt_schools_per_zone)
    #     model.Add(ith_digit <= schools_per_zone)


def add_school_number_constraints_alt(model, vm, school_df, bg_df, centroids, centroid_mapping):
    # if m is the number of blocks, and n is the number of zones, then we can iterate over ever variable in the model
    # such that variable i is equal

    school_in_bg = []
    for bg in vm.keys():
        if bg in school_df['BlockGroup'].values:
            school_in_bg.append(vm[bg])

    num_school_blocks = len(school_in_bg)
    num_zones = len(centroids)
    schools_per_zone = ceil(num_school_blocks / num_zones)
    alt_schools_per_zone = schools_per_zone - 1

    lb = mapper_function(0, num_school_blocks)
    ub = mapper_function(num_zones - 1, num_school_blocks)
    mapped_vm = {}
    for bg in school_in_bg:
        mapped_vm[str(bg)] = model.NewIntVar(lb, ub, f'mapped_{bg}')
        for i in range(num_zones):
            mapped_val = mapper_function(i, num_school_blocks)
            mapped_var = mapped_vm[str(bg)]
            intermediate_var = model.NewBoolVar(f'intermediate_{bg}_{i}_1')
            model.Add((bg == i)).OnlyEnforceIf(intermediate_var)
            model.Add((bg != i)).OnlyEnforceIf(intermediate_var.Not())

            # intermediate_var_2 = model.NewBoolVar(f'intermediate_{bg}_{i}_2')
            # model.Add((mapped_var == mapped_val)).OnlyEnforceIf(intermediate_var_2)
            # model.Add((mapped_var != mapped_val)).OnlyEnforceIf(intermediate_var_2.Not())
            # model.AddImplication(intermediate_var, intermediate_var_2)
            # model.AddImplication(intermediate_var_2, intermediate_var)

            # you can also do this without using the second intermediate variable
            model.Add((mapped_var == mapped_val)).OnlyEnforceIf(intermediate_var)
            model.Add((mapped_var != mapped_val)).OnlyEnforceIf(intermediate_var.Not())
    #
    mapped_sum = cp_model.LinearExpr.Sum(list(mapped_vm.values()))
    all_possible_vals = get_sum_combinations(schools_per_zone, num_school_blocks)
    #   mapped_sum must be equal to exactly one of the possible values
    equals = []
    for i in range(len(all_possible_vals)):
        equal_to_i = model.NewBoolVar(f'equal_to_{i}')
        model.Add(mapped_sum == all_possible_vals[i]).OnlyEnforceIf(equal_to_i)
        model.Add(mapped_sum != all_possible_vals[i]).OnlyEnforceIf(equal_to_i.Not())
        equals.append(equal_to_i)
    model.AddExactlyOne(equals)


def mapper_function(num, n):
    val = (n + 1) ** num
    #  if the upper bound gets too large, add the base 64 integer min to double possible values
    return val


def get_all_possible_b_and_b_minus_1(b, n):
    all_possible_b_and_b_minus_1 = []
    bound = n // b + 1
    for i in range(bound):
        for j in range(bound):
            if i * b + j * (b - 1) == n:
                all_possible_b_and_b_minus_1.append((i, j))
    return all_possible_b_and_b_minus_1


def get_sum_combinations(b, n):
    # b and b-1 are our target values
    # n is the number of blocks
    all_possible_b_and_b_minus_1 = get_all_possible_b_and_b_minus_1(b, n)

    all_sums = []
    for configuration in all_possible_b_and_b_minus_1:
        values = [b] * configuration[0] + [b - 1] * configuration[1]
        permutations = set(list(itertools.permutations(values)))
        for permutation in permutations:
            sum = 0
            for i in range(len(permutation)):
                sum += permutation[i] * mapper_function(i, n)
            all_sums.append(sum)

    a = len(all_sums)
    b = len(set(all_sums))
    print(a)
    if a != b:
        print('THIS SHOULD NEVER HAPPEN')
        raise Exception('Sums are not unique')
    return tuple(all_sums)
    # terms = []
    # for i in range(m):
    #     terms.append([-1] * 2)
    #     terms[i][0] = b * mapper_function(i, n)
    #     terms[i][1] = (b - 1) * mapper_function(i, n)
    # # add all possible combinations of terms
    # for i in range(2 ** m):
    #     sum = 0
    #     for j in range(m):
    #         if i & (1 << j):
    #             sum += terms[j][0]
    #         else:
    #             sum += terms[j][1]
    #     all_sums.append(sum)
    # return tuple(all_sums)


def add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids):
    # The total capacity should be within 15% of the total number of students
    for zone in centroids:
        block_values = list(vm[zone].values())
        zone_capacity_coefs = pd.Series([get_bg_for_school(school_df, bg) for bg in vm[zone]])
        zone_capacity_coefs_max = (zone_capacity_coefs * SCALING_FACTOR * 1.15).round().astype(int).tolist()
        zone_capacity_coefs_min = (zone_capacity_coefs * SCALING_FACTOR * 0.85).round().astype(int).tolist()

        bg_counts = (bg_df['student_count'] * SCALING_FACTOR).round().astype(int).tolist()

        print(sum(zone_capacity_coefs_max) / SCALING_FACTOR, sum(zone_capacity_coefs_min) / SCALING_FACTOR,
              sum(bg_counts) / SCALING_FACTOR)
        zone_capacity_min = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_min)
        zone_capacity_max = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_max)
        zone_students = cp_model.LinearExpr.WeightedSum(block_values, bg_counts)
        # the number of students cannot be more than 15% greater than the capacity
        model.Add(zone_students <= zone_capacity_max)
        # the number of students cannot be less than 15% less than the capacity
        model.Add(zone_students >= zone_capacity_min)


# Inherently handled by the fact that the assignment is to an integer
# def add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids):
#     #  Every blockgroup should be assigned to exactly one zone
#     for bg in bg_df['census_blockgroup']:
#         model.Add(sum([vm[zone][bg] for zone in centroids]) == 1)


def get_bg_for_school(school_df, bg):
    # Find the school in that blockgroup
    if bg in school_df['BlockGroup'].values:
        return school_df[school_df['BlockGroup'] == int(bg)].iloc[0]['capacity']
    else:
        return None


def get_bg_of_school(school_df, school_id):
    # Find the school in that blockgroup
    if school_id in school_df['school_id'].values:
        return school_df[school_df['school_id'] == int(school_id)].iloc[0]['BlockGroup']
    else:
        return None
