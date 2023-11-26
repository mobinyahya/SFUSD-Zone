import pandas as pd
import yaml
from ortools.sat.python import cp_model

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
    print(model.Validate())
    status = solver.Solve(model)
    print(f"Status = {solver.StatusName(status)}")


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


def main():
    prep_model()


if __name__ == '__main__':
    main()
