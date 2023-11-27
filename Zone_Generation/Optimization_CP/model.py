import pandas as pd
import yaml
from ortools.sat.python import cp_model

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimization_CP.constants import MAX_SOLVER_TIME, NUM_SOLVER_THREADS
from Zone_Generation.Optimization_CP.constraints import add_constraints
from Zone_Generation.Optimization_CP.optimization import add_optimization


def prep_model():
    model = cp_model.CpModel()
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
    return solver, vm, school_df, bg_df, centroids


def add_variables(model):
    student_df = pd.read_csv('~/SFUSD/Data/Cleaned/enrolled_1819.csv')
    school_df = pd.read_csv('~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv')
    program_df = pd.read_csv('~/SFUSD/Data/Cleaned/programs_1819.csv')

    student_df = student_df[student_df['grade'] == 'KG']
    print(len(student_df.index))

    races = ['Asian', 'White', 'Hispanic/Latino']
    for race in races:
        student_df[race] = student_df['resolved_ethnicity'].apply(
            lambda resolved: 1 if resolved == race else 0)

    bg_df = student_df.groupby('census_blockgroup')
    bg_df = bg_df.agg(
        {'studentno': 'count', 'FRL Score': 'first', 'census_blockgroup': 'first', 'idschoolattendance': 'first',
         'Asian': 'sum', 'White': 'sum', 'Hispanic/Latino': 'sum'})
    bg_df = bg_df.rename(columns={'studentno': 'student_count'}).reset_index(drop=True)

    # Check that the last part of the program_id is "KG"

    program_df = program_df[program_df['program_id'].str[-2:] == 'KG']

    def program_capacity_for_school(row):
        return program_df[program_df['school_id'] == row['school_id']]['capacity'].sum()
        # row['capacity'] = program_df[program_df['school_id'] == row['school_id']]['capacity'].sum()

    school_df['capacity'] = school_df.apply(program_capacity_for_school, axis=1)
    # school_df = pd.merge(school_df, program_df, on='school_id', how='inner')
    # remove cap_lb, as it is not neccessarily the same capacity for the GE-KG program
    # school_df = school_df.drop(columns=['cap_lb'])
    print(sum(school_df['capacity']))

    centroids = None
    with open("Zone_Generation/Config/centroids.yaml", "r") as stream:
        centroids = yaml.safe_load(stream)['3-zone-0']

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
    for i, z in enumerate(centroids):
        int_map[z] = i
    zone_dict = {}
    for z in centroids:
        zone_dict[z] = []
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
    zv = ZoneVisualizer('BlockGroup')
    zv.visualize_zones_from_dict(zone_dict, centroid_location=centroid_locations, title='No Diversity test',
                                 save_name='test')


def main():
    visualize(*prep_model())


if __name__ == '__main__':
    main()
