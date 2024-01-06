import json

import pandas as pd
from matplotlib import pyplot as plt

from Zone_Generation.Optimization_CP.constants import YEAR, RACES


def main():
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

    with open(
            '/Users/kumar/Dropbox/SFUSD/Optimization/Zones/all-ge-students/CP/6-zone-1/14400_05-12-2023 10:31:24_504.json',
            'r') as f:
        zonings = json.load(f)

    int_map = zonings['int_map']
    del zonings['int_map']
    # reverse map
    rev_map = {}
    for k, v in int_map.items():
        rev_map[v] = k
    zoning_df = pd.DataFrame()
    zoning_df['census_blockgroup'] = zonings.keys()
    zoning_df['census_blockgroup'] = zoning_df['census_blockgroup'].astype(int).astype(float)
    zoning_df['assigned_school_id'] = zoning_df['census_blockgroup'].apply(lambda x: rev_map[zonings[str(int(x))]])
    zoning_df = pd.merge(zoning_df, bg_df, on='census_blockgroup', how='left')
    zschool_df = zoning_df.groupby('assigned_school_id')
    percentage_df = pd.DataFrame()
    percentage_df['school_id'] = ''
    percentage_df['assigned_students'] = 0
    for race in RACES:
        percentage_df[f'assigned_{race}_percentage'] = 0
    for assigned, df in zschool_df:
        total = df['student_count'].sum()
        add_value = {'school_id': assigned, 'assigned_students': total}
        for race in RACES:
            add_value[f'assigned_{race}_percentage'] = df[race].sum() / total

        percentage_df.loc[len(percentage_df)] = add_value
        #     add new row to bottom of percentage_df
        # percentage_df = percentage_df.append({'school_id': assigned, 'assigned_students': total}, ignore_index=True)

    # create a seprate graph for every race with a bar plot of the percentage of students assigned to each school.
    # then add a horizonatal line at 15% for each graph

    for race in RACES:
        global_race = bg_df[race].sum() / bg_df['student_count'].sum()

        percentage_df.plot.bar(x='school_id', y=f'assigned_{race}_percentage',
                               title=f'Percentage of {race} students assigned to each zone').axhline(y=global_race,
                                                                                                     color='r',
                                                                                                     linestyle='-',
                                                                                                     label=f'District Wide {race} percentage')

    plt.show()


if __name__ == '__main__':
    main()
