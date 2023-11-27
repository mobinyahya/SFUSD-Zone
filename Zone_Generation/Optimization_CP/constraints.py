import csv
import os
from collections import defaultdict

from ortools.sat.python import cp_model


def add_constraints(model, vm, school_df, bg_df, centroids):
    # Every centroid must have the block the school is in assigned to it
    for zone in centroids:
        model.Add(vm[zone][school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0]] == True)
    add_school_number_constraints(model, vm, school_df, bg_df, centroids)
    add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids)
    add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids)
    add_contiguity_constraints(model, vm, school_df, bg_df, centroids)
    # add_diversity_constraints(model, vm, school_df, bg_df, centroids)


def add_diversity_constraints(model, vm, school_df, bg_df, centroids):
    #    All zones must have more than 15% less of the average number of any group (FRL, White, Asian, Latino)
    races = ['Asian', 'White', 'Hispanic/Latino']
    for zone in centroids:
        for race in races:
            # TODO: Check that this this is an equivalent constraint to the one in the paper
            print(race, bg_df[race].sum(), 'total', bg_df['student_count'].sum())
            race_min = int(bg_df[race].sum() - (0.15 * bg_df['student_count'].sum()))
            # ^^ this is equivalent to (race/total - 0.15) * total
            rounded_race = (bg_df[race]).round().tolist()
            block_values = list(vm[zone].values())
            race_block_sum = cp_model.LinearExpr.WeightedSum(block_values, rounded_race)
            model.Add(race_block_sum > race_min)


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
    for zone in centroids:
        zone_bg = school_df[school_df['school_id'] == zone]['BlockGroup'].iloc[0]
        for bg in vm[zone]:
            if bg in school_df['BlockGroup'].values:
                continue

            bg_distance_to_zone = travels[int(bg)][int(zone_bg)]
            neighbors_closer = []
            for neighbor in neighbors[str(int(bg))]:
                if neighbor == '':
                    continue
                neighbor = int(neighbor)
                if float(neighbor) not in vm[zone]:
                    continue
                neighbor_distance_to_zone = travels[neighbor][zone_bg]
                if neighbor_distance_to_zone < bg_distance_to_zone:
                    neighbors_closer.append(float(neighbor))
            model.Add(
                sum(vm[zone][n] for n in neighbors_closer) > 0
            ).OnlyEnforceIf(vm[zone][bg])


def add_school_number_constraints(model, vm, school_df, bg_df, centroids):
    schools_per_zone = len(school_df.index) // len(centroids)
    #     The number of schools in the zone must be equal across zones
    for zone in centroids:
        schools_in_zone = 0
        for bg in vm[zone]:
            if bg in school_df['BlockGroup'].values:
                schools_in_zone += vm[zone][bg]
        model.Add(
            schools_in_zone >= schools_per_zone - 2)
        # for some reason this -1 is necessary, otherwise the model is infeasible.
        model.Add(schools_in_zone <= schools_per_zone + 2)


def add_zone_capacity_constraints(model, vm, school_df, bg_df, centroids):
    # The total capacity should be within 15% of the total number of students
    for zone in centroids:
        school_capacities = []
        for bg in vm[zone]:
            if bg in school_df['BlockGroup'].values:
                school_capacities.append(vm[zone][bg] * get_bg_for_school(school_df, bg)['capacity'])
        bg_students = []
        for bg in vm[zone]:
            bg_students.append(vm[zone][bg] * bg_df[bg_df['census_blockgroup'] == bg]['student_count'].values[0])
        zone_capacity = sum(school_capacities)
        zone_students = sum(bg_students)
        # the number of students cannot be more than 15% greater than the capacity
        model.Add(100 * zone_students <= zone_capacity * 115)


def add_zone_duplicates_constraints(model, vm, school_df, bg_df, centroids):
    #  Every blockgroup should be assigned to exactly one zone
    for bg in bg_df['census_blockgroup']:
        model.Add(sum([vm[zone][bg] for zone in centroids]) == 1)


def get_bg_for_school(school_df, bg):
    # Find the school in that blockgroup
    if bg in school_df['BlockGroup'].values:
        return school_df[school_df['BlockGroup'] == int(bg)].iloc[0]
    else:
        return None
