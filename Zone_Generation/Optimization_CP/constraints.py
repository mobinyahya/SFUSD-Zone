import csv
import os
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

from Zone_Generation.Optimization_CP.constants import SCALING_FACTOR, RACES


def add_constraints(model, vm, school_df, bg_df, centroids):
    # Every centroid must have the block the school is in assigned to it
    for zone in centroids:
        model.Add(vm[zone][school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0]] == True)
    add_school_number_constraints(model, vm, school_df, bg_df, centroids)
    add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids)
    add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids)
    add_contiguity_constraints(model, vm, school_df, bg_df, centroids)
    add_diversity_constraints(model, vm, school_df, bg_df, centroids)
    add_frl_constraints(model, vm, school_df, bg_df, centroids)


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


def add_contiguity_constraints(model, vm, school_df, bg_df, centroids):
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
    for zone in centroids:
        zone_bg = school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0]
        for bg in vm[zone]:
            if bg in centroid_bgs:
                continue

            bg_distance_to_zone = travels[int(bg)][int(zone_bg)]
            neighbors_closer = set()
            all_neighbors = set()
            for neighbor in neighbors[str(int(bg))]:
                if neighbor == '':
                    continue
                neighbor = int(neighbor)
                if float(neighbor) not in vm[zone]:
                    continue
                all_neighbors.add(float(neighbor))
                neighbor_distance_to_zone = travels[neighbor][zone_bg]
                if neighbor_distance_to_zone < bg_distance_to_zone:
                    neighbors_closer.add(float(neighbor))
            model.Add(sum(vm[zone][n] for n in neighbors_closer) >= vm[zone][bg])


def add_school_number_constraints(model, vm, school_df, bg_df, centroids):
    schools_per_zone = len(school_df.index) // len(centroids)
    #     The number of schools in the zone must be equal across zones
    for zone in centroids:
        schools_in_zone = 0
        for bg in vm[zone]:
            if bg in school_df['BlockGroup'].values:
                schools_in_zone += vm[zone][bg]
        model.Add(
            schools_in_zone >= schools_per_zone - 1)
        # for some reason this -1 is necessary, otherwise the model is infeasible.
        model.Add(schools_in_zone <= schools_per_zone)


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


def add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids):
    #  Every blockgroup should be assigned to exactly one zone
    for bg in bg_df['census_blockgroup']:
        model.Add(sum([vm[zone][bg] for zone in centroids]) == 1)


def get_bg_for_school(school_df, bg):
    # Find the school in that blockgroup
    if bg in school_df['BlockGroup'].values:
        return school_df[school_df['BlockGroup'] == int(bg)].iloc[0]['capacity']
    else:
        return 0
