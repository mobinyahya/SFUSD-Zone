import itertools
from math import ceil

import pandas as pd
from ortools.sat.python import cp_model

from Zone_Generation.Optimization_CP.constants import SCALING_FACTOR, RACES


def add_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping, neighbors, travels):
    # Every centroid must have the block the school is in assigned to it
    for zone in centroids:
        bg = get_bg_of_school(school_df, zone)
        model.Add(vm[bg] == centroid_mapping[zone])
    blocks_assigned_to_zone = create_indicator_variables(model, vm, school_df, bg_df, centroids, centroid_mapping)
    neighbor_pairs = create_neighbor_pair_indicators(model, vm, school_df, bg_df, centroids, centroid_mapping,
                                                     neighbors, travels)
    add_school_number_constraints(model, vm, school_df, bg_df, centroids,
                                  centroid_mapping, blocks_assigned_to_zone)  # hard to convert to integer
    add_contiguity_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping,
                               blocks_assigned_to_zone, neighbors, travels,
                               neighbor_pairs)  # easy to convert to integer
    # All of these are essentially the exact same problem
    # add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids,
    #                               centroid_mapping, blocks_assigned_to_zone)  # hard to convert to integer
    # add_diversity_constraints(model, vm, school_df, bg_df, centroids,
    #                           centroid_mapping, blocks_assigned_to_zone)  # hard to convert to integer
    # add_frl_constraints(model, vm, school_df, bg_df, centroids,
    #                     centroid_mapping, blocks_assigned_to_zone)  # hard to convert to integer

    # add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids) #not needed since true by definition
    return blocks_assigned_to_zone, neighbor_pairs


def create_indicator_variables(model, vm, school_df, bg_df, centroids, centroid_mapping):
    blocks_assigned_to_zone = {}
    for bg in vm:
        if bg not in blocks_assigned_to_zone:
            blocks_assigned_to_zone[bg] = []
        for i in range(len(centroids)):
            equal_var = model.NewBoolVar(f'equal_{bg}_{i}')
            model.Add(vm[bg] == i).OnlyEnforceIf(equal_var)
            model.Add(vm[bg] != i).OnlyEnforceIf(equal_var.Not())
            blocks_assigned_to_zone[bg].append(equal_var)
    return blocks_assigned_to_zone


def create_neighbor_pair_indicators(model, vm, school_df, bg_df, centroids, centroid_mapping, neighbors, travels):
    neighbor_pairs = {}
    for bg in vm.keys():
        neighbor_pairs[bg] = {}
        all_neighbors = neighbors[str(int(bg))]
        #     cant be smart and avoid creating extra variables, because each variable has a different travel time to each zone
        for neighbor in all_neighbors:
            if neighbor == '':
                continue

            neighbor = int(neighbor)
            if float(neighbor) not in vm.keys():
                print('amogus')
                continue
            indicator = model.NewBoolVar(f'neighbor_assigned_to_same_{bg}_{neighbor}')
            neighbor_pairs[bg][neighbor] = indicator

            model.Add(vm[bg] != vm[neighbor]).OnlyEnforceIf(indicator)
            model.Add(vm[bg] == vm[neighbor]).OnlyEnforceIf(indicator.Not())
    return neighbor_pairs


def add_contiguity_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping, blocks_assigned_to_zone,
                               neighbors, travels, neighbor_pairs):
    #     There exists some blockgroup in the zone that is closer to the centroid than the current blockgroup for all blockgroups

    centroid_bgs = []
    for zone in centroids:
        centroid_bgs.append(school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0])
    for bg in vm.keys():
        if bg in centroid_bgs:
            continue
        for zone in centroids:
            zone_bg = school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0]
            closer_neighbors = set()
            all_neighbors = neighbors[str(int(bg))]
            for neighbor in all_neighbors:
                if neighbor == '':
                    continue
                neighbor = int(neighbor)
                if float(neighbor) not in vm.keys():
                    print('why would this happen')
                    continue
                neighbor_distance_to_zone = travels[neighbor][zone_bg]
                bg_distance_to_zone = travels[int(bg)][int(zone_bg)]
                if neighbor_distance_to_zone < bg_distance_to_zone:
                    closer_neighbors.add(neighbor_pairs[bg][neighbor])
            # if len(closer_neighbors) == 0:
            #     print(zone, bg)
            #     # this should be fine to skip???
            #     continue
            # if this block group is assigned to this zone, then at least one closer neighbor must be equal to bg,
            # that is that one of them is 0
            model.Add(cp_model.LinearExpr.Sum(list(closer_neighbors)) < len(closer_neighbors)).OnlyEnforceIf(
                blocks_assigned_to_zone[bg][centroid_mapping[zone]])

            # alternate method, may or may not be faster?

            # at_least_one_indicator = model.NewBoolVar(f'neighbor_assigned_to_same_{bg}_{neighbor}')
            # model.AddMinEquality(at_least_one_indicator, closer_neighbors)
            # model.Add(at_least_one_indicator == 0).OnlyEnforceIf(
            #     blocks_assigned_to_zone[bg][centroid_mapping[zone]])



def add_school_number_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping, blocks_assigned_to_zone):
    # if m is the number of blocks, and n is the number of zones, then we can iterate over ever variable in the model
    # such that variable i is equal

    school_in_bg = []
    for bg in vm.keys():
        if bg in school_df['BlockGroup'].values:
            school_in_bg.append(bg)

    num_school_blocks = len(school_in_bg)
    num_zones = len(centroids)
    schools_per_zone = ceil(num_school_blocks / num_zones)
    alt_schools_per_zone = schools_per_zone - 1
    schools_in_zone = {}

    for i in range(num_zones):
        if i not in schools_in_zone:
            schools_in_zone[i] = []
        for bg in school_in_bg:
            schools_in_zone[i].append(blocks_assigned_to_zone[bg][i])
    for zone in schools_in_zone:
        model.Add(cp_model.LinearExpr.Sum(schools_in_zone[zone]) >= alt_schools_per_zone)
        model.Add(cp_model.LinearExpr.Sum(schools_in_zone[zone]) <= schools_per_zone)

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


def add_school_number_constraints_alt(model, vm, school_df, bg_df, centroids,
                                      centroid_mapping, blocks_assigned_to_zone):
    # if m is the number of blocks, and n is the number of zones, then we can iterate over ever variable in the model
    # such that variable i is equal

    school_in_bg = []
    for bg in vm.keys():
        if bg in school_df['BlockGroup'].values:
            school_in_bg.append(bg)

    num_school_blocks = len(school_in_bg)
    num_zones = len(centroids)
    schools_per_zone = ceil(num_school_blocks / num_zones)
    alt_schools_per_zone = schools_per_zone - 1

    lb = mapper_function(0, num_school_blocks)
    ub = mapper_function(num_zones - 1, num_school_blocks)
    mapped_vm = {}
    for bg in school_in_bg:
        mapped_vm[bg] = model.NewIntVar(lb, ub, f'mapped_{bg}')
        for i in range(num_zones):
            mapped_val = mapper_function(i, num_school_blocks)
            mapped_var = mapped_vm[bg]

            # intermediate_var_2 = model.NewBoolVar(f'intermediate_{bg}_{i}_2')
            # model.Add((mapped_var == mapped_val)).OnlyEnforceIf(intermediate_var_2)
            # model.Add((mapped_var != mapped_val)).OnlyEnforceIf(intermediate_var_2.Not())
            # model.AddImplication(intermediate_var, intermediate_var_2)
            # model.AddImplication(intermediate_var_2, intermediate_var)

            # you can also do this without using the second intermediate variable
            model.Add((mapped_var == mapped_val)).OnlyEnforceIf(blocks_assigned_to_zone[bg][i])
            model.Add((mapped_var != mapped_val)).OnlyEnforceIf(blocks_assigned_to_zone[bg][i].Not())
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
    model.AddBoolOr(equals)


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


def add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids,
                                  centroid_mapping, blocks_assigned_to_zone):
    # The total capacity should be within 15% of the total number of students

    bgs = vm.keys()

    zone_capacity_coefs = pd.Series([get_school_capacity_of_bg(school_df, bg) for bg in bgs])

    bg_counts = (bg_df['student_count'] * SCALING_FACTOR).round().astype(int).tolist()
    zone_capacity_coefs_max = (zone_capacity_coefs * SCALING_FACTOR * 1.15).round().astype(int).tolist()
    zone_capacity_coefs_min = (zone_capacity_coefs * SCALING_FACTOR * 0.85).round().astype(int).tolist()

    for zone in centroids:
        block_values = [blocks_assigned_to_zone[bg][centroid_mapping[zone]] for bg in bgs]
        zone_capacity_min = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_min)
        zone_capacity_max = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_max)
        zone_students = cp_model.LinearExpr.WeightedSum(block_values, bg_counts)
        # the number of students cannot be more than 15% greater than the capacity
        model.Add(zone_students <= zone_capacity_max)
        # the number of students cannot be less than 15% less than the capacity
        model.Add(zone_students >= zone_capacity_min)


def add_frl_constraints(model, vm, school_df, bg_df, centroids,
                        centroid_mapping, blocks_assigned_to_zone):
    #     The FRL percentage of the zone must be within 15% of the average FRL percentage

    frl_min = int(
        ((bg_df['FRL'].sum() / bg_df['student_count'].sum()) - 0.15) * SCALING_FACTOR)
    frl_max = int(
        ((bg_df['FRL'].sum() / bg_df['student_count'].sum()) + 0.15) * SCALING_FACTOR)

    frl_coef = (bg_df['FRL'] * SCALING_FACTOR).round().astype(int).tolist()
    tcoef = (bg_df['student_count']).round().astype(int).tolist()
    bgs = vm.keys()
    for zone in centroids:
        block_values = [blocks_assigned_to_zone[bg][centroid_mapping[zone]] for bg in bgs]
        frl_block_sum = cp_model.LinearExpr.WeightedSum(block_values, frl_coef)
        total_block_sum = cp_model.LinearExpr.WeightedSum(block_values, tcoef)
        model.Add(frl_block_sum >= total_block_sum * frl_min)
        model.Add(frl_block_sum <= total_block_sum * frl_max)


def add_diversity_constraints(model, vm, school_df, bg_df, centroids,
                              centroid_mapping, blocks_assigned_to_zone):
    #    All zones must have more than 15% less of the average number of any group (FRL, White, Asian, Latino)

    bgs = vm.keys()
    for race in RACES:

        race_min = int(((bg_df[race].sum() / bg_df['student_count'].sum()) - 0.15) * SCALING_FACTOR)
        race_max = int(((bg_df[race].sum() / bg_df['student_count'].sum()) + 0.15) * SCALING_FACTOR)
        rcoef = (bg_df[race] * SCALING_FACTOR).round().astype(int).tolist()
        tcoef = (bg_df['student_count']).round().astype(int).tolist()
        for zone in centroids:
            # TODO: Check that this this is an equivalent constraint to the one in the paper
            # print(race, bg_df[race].sum(), 'total', bg_df['student_count'].sum())

            block_values = [blocks_assigned_to_zone[bg][centroid_mapping[zone]] for bg in bgs]
            race_block_sum = cp_model.LinearExpr.WeightedSum(block_values, rcoef)
            total_block_sum = cp_model.LinearExpr.WeightedSum(block_values, tcoef)
            # r/t > rmin = r> rmin * t
            # rmin = (R/T - 0.15)
            # r > (R/T - 0.15) * t
            # r * scaler > (R - 0.15) * scaler * t

            model.Add(race_block_sum > total_block_sum * race_min)
            model.Add(race_block_sum < total_block_sum * race_max)


# Inherently handled by the fact that the assignment is to an integer
# def add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids):
#     #  Every blockgroup should be assigned to exactly one zone
#     for bg in bg_df['census_blockgroup']:
#         model.Add(sum([vm[zone][bg] for zone in centroids]) == 1)


def get_school_capacity_of_bg(school_df, bg):
    # Find the school in that blockgroup
    if bg in school_df['BlockGroup'].values:
        return school_df[school_df['BlockGroup'] == int(bg)].iloc[0]['capacity']
    else:
        return 0


def get_bg_of_school(school_df, school_id):
    # Find the school in that blockgroup
    if school_id in school_df['school_id'].values:
        return school_df[school_df['school_id'] == int(school_id)].iloc[0]['BlockGroup']
    else:
        return None
