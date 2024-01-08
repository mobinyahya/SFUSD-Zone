import copy
import csv
import datetime
import json
import os
from collections import defaultdict

import pandas as pd
import yaml
from ortools.sat.python import cp_model

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimization_CP.constants import NUM_SOLVER_THREADS, YEAR, RACES, MAX_SOLVER_TIME, CENTROIDS
from Zone_Generation.Optimization_CP.constraints import add_constraints
from Zone_Generation.Optimization_CP.optimization import add_optimization


def get_neighbors_mapping():
    # create dictionary mapping block group number to list of neighbor block
    file = os.path.expanduser(
        "~/Dropbox/SFUSD/Optimization/block_group_adjacency_matrix.csv"
    )
    with open(file, "r") as f:
        reader = csv.reader(f)
        adjacency_matrix = list(reader)
    # create dictionary mapping attendance area school id to list of neighbor
    # attendance area ids (similarly, block group number)
    neighbors = {}
    for row in adjacency_matrix:
        neighbors[row[0]] = set(row[1:])
    return neighbors


def get_travel_matrix():
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
    return travels


def prep_model(time, zones):
    model = cp_model.CpModel()
    print(zones)
    print(time // 60)
    print('Creating variables')
    vm, school_df, bg_df, centroids, centroid_mapping = add_variables(model, zones)
    print('Adding constraints')
    neighbors = get_neighbors_mapping()
    travels = get_travel_matrix()
    blocks_assigned_to_zone, neighbor_pairs = add_constraints(model, vm, school_df, bg_df, centroids, centroid_mapping,
                                                              neighbors, travels)
    print('Adding optimization')
    add_optimization(model, vm, school_df, bg_df, centroids, centroid_mapping, neighbors, travels, neighbor_pairs)
    print('Solving')
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = time

    # Adding parallelism
    solver.parameters.num_search_workers = NUM_SOLVER_THREADS
    # solver.parameters.log_search_progress = True
    print(model.Validate())
    status = solver.Solve(model)

    print(f"Status = {solver.StatusName(status)}")
    if status == cp_model.INFEASIBLE or status == cp_model.MODEL_INVALID or status == cp_model.UNKNOWN:
        return (None, None, None, None, None)

    return solver, vm, school_df, bg_df, centroids


def add_variables(model, zones):
    bg_df = pd.read_csv(f'~/Dropbox/SFUSD/Data/final_area_data/area_data_no_k8.csv')
    school_df = pd.read_csv(f'~/SFUSD/Data/Cleaned/schools_rehauled_{YEAR}.csv')
    # program_df = pd.read_csv(f'~/SFUSD/Data/Cleaned/programs_{YEAR}.csv')
    # program_df = pd.read_csv(F'~/SFUSD/Data/Cleaned/programs_{YEAR}.csv')
    # student_df = pd.read_csv(f'~/SFUSD/Data/Cleaned/enrolled_{YEAR}.csv')

    bg_df['enrolled_and_ge_applied'] = bg_df['enrolled_and_ge_applied'] / 6
    for race in RACES:
        bg_df[race] = bg_df['enrolled_and_ge_applied'] * (
                bg_df[f'resolved_ethnicity_{race}'] / bg_df['num_with_ethnicity'])
        bg_df[race] = bg_df[race].fillna(0)
    bg_df['FRL'] = bg_df['enrolled_and_ge_applied'] * (bg_df['frl_count'] / bg_df['frl_total_count'])
    bg_df['FRL'] = bg_df['FRL'].fillna(0)
    bg_df['student_count'] = bg_df['enrolled_and_ge_applied']
    bg_df['student_count'] = bg_df['student_count'].fillna(0)

    # student_df[race] = student_df['resolved_ethnicity'].apply(
    #     lambda resolved: 1 if resolved == race else 0)
    #
    # bg_df = student_df.groupby('census_blockgroup')
    # bg_df = bg_df.agg(
    #     {'studentno': 'count', 'FRL Score': 'first', 'census_blockgroup': 'first', 'idschoolattendance': 'first',
    #      'Asian': 'sum', 'White': 'sum', 'Hispanic/Latino': 'sum'})
    # bg_df = bg_df.rename(columns={'studentno': 'student_count'}).reset_index(drop=True)

    # Check that the last part of the program_id is "KG"

    # program_df = program_df[program_df['program_id'].str[-2:] == 'KG']

    def program_capacity_for_school(row):
        #     get blockgroup in bg_df
        bg = bg_df[bg_df['census_blockgroup'] == row['BlockGroup']]
        if bg.empty:
            return 0
        bg = bg.iloc[0]
        if bg['ge_schools'] != 1:
            return 0
        return bg['ge_capacity']

    school_df['capacity'] = school_df.apply(program_capacity_for_school, axis=1)
    # print(school_df['capacity'].sum())
    # print(bg_df['student_count'].sum())
    # school_df = pd.merge(school_df, program_df, on='school_id', how='inner')
    # remove cap_lb, as it is not neccessarily the same capacity for the GE-KG program
    # school_df = school_df.drop(columns=['cap_lb'])
    # print(sum(school_df['capacity']))

    centroids = None
    with open("Zone_Generation/Config/centroids.yaml", "r") as stream:
        centroids = yaml.safe_load(stream)[zones]
    vals = []
    # number of blocks
    n = len(bg_df['census_blockgroup'])
    # number of zones
    m = len(centroids)
    centroid_mapping = {}
    for i, c in enumerate(centroids):
        centroid_mapping[c] = i

    # Create a mapping of blockgroups to zones
    vm = {}
    for bg in bg_df['census_blockgroup']:
        vm[bg] = model.NewIntVar(0, m - 1, f'x_{bg}')
        # vm[bg] = model.NewIntVar(0, len(centroids) - 1, f'x_{bg}')

    return vm, school_df, bg_df, centroids, centroid_mapping


def visualize(solver, vm, school_df, bg_df, centroids):
    if solver is None:
        return
    # Print solution.
    print(f"Objective value = {solver.ObjectiveValue()}")
    int_map = {}
    zone_dict = {}
    for i, z in enumerate(centroids):
        int_map[z] = i
    centroid_locations = pd.DataFrame()
    centroid_locations['lat'] = 0
    centroid_locations['lon'] = 0
    for zone in centroids:
        centroid_locations.loc[zone, 'lat'] = school_df[school_df['school_id'] == zone]['lat'].iloc[0]
        centroid_locations.loc[zone, 'lon'] = school_df[school_df['school_id'] == zone]['lon'].iloc[0]
    for bg in bg_df['census_blockgroup']:
        zone_dict[bg] = solver.Value(vm[bg])
        # for zone in centroids:
        #     if solver.BooleanValue(vm[zone][bg]) == 1:
        #         zone_dict[bg] = int_map[zone]
        #         break
    path = os.path.expanduser(f'~/Dropbox/SFUSD/Optimization/Zones/all-ge-students/CP/{len(centroids)}/')
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = f'{solver.parameters.max_time_in_seconds}_{datetime.datetime.now().strftime("%d-%m %H:%M")}_{solver.ObjectiveValue()}.json'
    with open(path + file_name, 'w') as f:
        copy_map = copy.deepcopy(zone_dict)
        copy_map['int_map'] = int_map
        json.dump(copy_map, f)
    zv = ZoneVisualizer('BlockGroup')
    # bad_boys = pd.DataFrame()
    # bad_boys['lat'] = 0
    # bad_boys['lon'] = 0
    # for n in BAD_NEIGHBORS:
    #     bad_boys.loc[n, 'lat'] = bg_df[bg_df['census_blockgroup'] == n]['lat'].iloc[0]
    #     bad_boys.loc[n, 'lon'] = bg_df[bg_df['census_blockgroup'] == n]['lon'].iloc[0]
    zv.visualize_zones_from_dict(zone_dict, centroid_location=centroid_locations,
                                 title=f'SFUSD Zoning with {len(centroids)} zones, with obj val {solver.ObjectiveValue()}, in {solver.parameters.max_time_in_seconds} seconds, with {solver.parameters.num_search_workers} threads',
                                 save_name=str(datetime.datetime.now()))


def main(time=MAX_SOLVER_TIME, zones=CENTROIDS):
    visualize(*prep_model(time, zones))


if __name__ == '__main__':
    main()
