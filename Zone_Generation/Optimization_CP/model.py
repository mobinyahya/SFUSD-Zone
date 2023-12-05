import copy
import datetime
import json
import os
import sys

import pandas as pd
import yaml
from ortools.sat.python import cp_model

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimization_CP.constants import MAX_SOLVER_TIME, NUM_SOLVER_THREADS, CENTROIDS, YEAR, RACES
from Zone_Generation.Optimization_CP.constraints import add_constraints
from Zone_Generation.Optimization_CP.optimization import add_optimization


def prep_model():
    model = cp_model.CpModel()
    print(CENTROIDS)
    print(MAX_SOLVER_TIME)
    print('Creating variables')
    vm, school_df, bg_df, centroids = add_variables(model)
    print('Adding constraints')
    add_constraints(model, vm, school_df, bg_df, centroids)
    print('Adding optimization')
    add_optimization(model, vm, school_df, bg_df, centroids)
    print('Solving')
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = MAX_SOLVER_TIME

    # Adding parallelism
    solver.parameters.num_search_workers = NUM_SOLVER_THREADS
    # solver.parameters.log_search_progress = True
    print(model.Validate())
    status = solver.Solve(model)

    print(f"Status = {solver.StatusName(status)}")
    if status == cp_model.INFEASIBLE:
        sys.exit(1)
    return solver, vm, school_df, bg_df, centroids


def add_variables(model):
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
            print('...')
            return 0
        return bg['ge_capacity']

    school_df['capacity'] = school_df.apply(program_capacity_for_school, axis=1)
    print(school_df['capacity'].sum())
    print(bg_df['student_count'].sum())
    # school_df = pd.merge(school_df, program_df, on='school_id', how='inner')
    # remove cap_lb, as it is not neccessarily the same capacity for the GE-KG program
    # school_df = school_df.drop(columns=['cap_lb'])
    # print(sum(school_df['capacity']))

    centroids = None
    with open("Zone_Generation/Config/centroids.yaml", "r") as stream:
        centroids = yaml.safe_load(stream)[CENTROIDS]

    # Create a 2d binary variable matrix for each school and each blockgroup
    # Each cell in the matrix represents whether a student from that blockgroup is assigned to that school
    vm = {}
    for zone in centroids:
        vm[zone] = {}
        for bg in bg_df['census_blockgroup']:
            vm[zone][bg] = model.NewBoolVar(f'x_{zone}_{bg}')

    return vm, school_df, bg_df, centroids


def visualize(solver, vm, school_df, bg_df, centroids):
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
        for zone in centroids:
            if solver.BooleanValue(vm[zone][bg]) == 1:
                zone_dict[bg] = int_map[zone]
                break
    path = os.path.expanduser(f'~/Dropbox/SFUSD/Optimization/Zones/all-ge-students/CP/{CENTROIDS}/')
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = f'{MAX_SOLVER_TIME}_{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}_{solver.ObjectiveValue()}.json'
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
                                 title=f'SFUSD Zoning with {CENTROIDS[0]} zones',
                                 save_name=str(datetime.datetime.now()))


def main():
    visualize(*prep_model())


if __name__ == '__main__':
    main()
