import json
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
from ortools.sat.python import cp_model

from Graphic_Visualization.zone_viz import ZoneVisualizer
# from Zone_Generation.Config.Constants import zone_colors
from Zone_Generation.Optimization_CP.constants import MAX_SOLVER_TIME, NUM_SOLVER_THREADS, config_to_str, set_config
from Zone_Generation.Optimization_CP.constraints import add_constraints
from Zone_Generation.Optimization_CP.hints import add_hints
from Zone_Generation.Optimization_CP.optimization import add_optimization


def prep_model():
    model = cp_model.CpModel()
    # print(CENTROIDS)
    # print(MAX_SOLVER_TIME)
    print('Creating variables')
    vm, school_df, bg_df, centroids = add_variables(model)
    print('Adding constraints')
    add_constraints(model, vm, school_df, bg_df, centroids)
    # add_hints(model, vm, school_df, bg_df, centroids)
    print('Adding optimization')
    add_optimization(model, vm, school_df, bg_df, centroids)
    print('Solving')
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = MAX_SOLVER_TIME
    # Adding parallelism
    solver.parameters.num_search_workers = NUM_SOLVER_THREADS
    print(model.Validate())
    # status = solver.Solve(model)

    # print(f"Status = {solver.StatusName(status)}")
    # if status == cp_model.INFEASIBLE:
    #     sys.exit(1)
    return solver, vm, school_df, bg_df, centroids


def add_variables(model):
    bdf = pd.read_csv(f'~/SFUSD/Data/Webster-MissionBay/Max and Avg ES Students by Block.xlsx - ES Avg by Race.csv')
    blocks_to_use = pd.read_csv(
        f'~/SFUSD/Data/Webster-MissionBay/Max and Avg ES Students by Block.xlsx - CORRECTED Webster ESAA blocks.csv')
    mbes = {'school_id': 999, 'school_name': 'Mission Bay', 'school_name_long': 'Mission Bay ES',
            'lat': 37.76982, 'lon': -122.394519, 'Block': 60750607012013, 'capacity': 66 * 15}
    # doesnt have all columns, just fill in the rest with nan
    # add the new school to the school_df
    web = {'school_id': 497, 'school_name': 'Webster', 'school_name_long': 'Webster ES',
           'lat': 37.760521, 'lon': -122.39584, 'Block': 60750227022008, 'capacity': 22 * 15}
    school_df = pd.DataFrame([mbes, web])
    bdf = bdf.rename(columns={'400 - Filipino': 'filipino', '500 - Hispanic/Latino': 'hispanic',
                              '600 - Black/African American': 'black', '700 - White': 'white',
                              '720 - Middle Eastern/Arabic': 'arab', 'All other': 'other', 'Asian': 'asian',
                              'Pacific Islander': 'pacific_islander', 'Geoid20': 'Block'})
    frl_data = pd.read_csv(
        f'~/Downloads/TK-5 FRL Count for Webster Census Blocks 23-25 - TK-5 FRL Count, Webster Blocks.csv')
    frl_data = frl_data.rename(columns={'Geoid20': 'Block'})
    frl_data['frl_percent'] = frl_data['23-25 Avg FRL Count'] / frl_data['23-25 Avg Students']
    frl_data = frl_data[['Block', 'frl_percent']]
    frl_data = frl_data.fillna(0)
    bdf = pd.merge(bdf, frl_data, on='Block', how='left')

    # cast all columns to numeric
    bdf = bdf.apply(pd.to_numeric, errors='coerce')

    # replace all non-numeric values with 0 based on dtype
    blocks_to_use = blocks_to_use.rename(columns={'Geoid20': 'Block'})
    blocks_to_use = blocks_to_use.drop_duplicates(subset='Block')
    # filter blocks that are in blocks_to_use. If a block in blocks_to_use is not in bdf, add it with all other columns as 0
    bdf = bdf[bdf['Block'].isin(blocks_to_use['Block'])]
    bdf = pd.merge(blocks_to_use, bdf, on='Block', how='left')
    bdf = bdf.fillna(0)
    bdf = bdf.drop_duplicates(subset='Block')
    bdf['total'] = bdf['asian'] + bdf['white'] + bdf['hispanic'] \
                   + bdf['black'] + bdf['filipino'] + bdf[
                       'pacific_islander'] + bdf['arab'] + bdf['other']

    bdf['frl'] = bdf['total'] * bdf['frl_percent']
    bdf['other'] = bdf['total'] - bdf['asian'] - bdf['white'] - bdf['hispanic']
    centroids = [999, 497]
    school_df = school_df[school_df['school_id'].isin(centroids)]

    # print(bdf['total'].sum())

    # Create a 2d binary variable matrix for each school and each blockgroup
    # Each cell in the matrix represents whether a student from that blockgroup is assigned to that school
    # 1 == MBES, 0 == Webster
    vm = {}
    for b in bdf['Block']:
        vm[b] = model.NewBoolVar(f'x_{b}')

    return vm, school_df, bdf, centroids


def analyze_demographics(zone_dict, school_df, bg_df, config):
    # Get the demographics of each zone
    bg_df['zone'] = bg_df['Block'].apply(lambda x: zone_dict[x])

    with open('travels.json', 'r') as f:
        travels = json.load(f)

    centroid_bgs = {}
    for zone in [999, 497]:
        centroid_bgs[zone] = school_df[school_df['school_id'] == zone]['Block'].iloc[0]
    bg_df['d2web'] = bg_df.apply(lambda x: travels[str(int(x['Block']))][str(int(centroid_bgs[497]))], axis=1)
    bg_df['d2mbes'] = bg_df.apply(lambda x: travels[str(int(x['Block']))][str(int(centroid_bgs[999]))], axis=1)
    bg_df['unweighted_distance'] = bg_df.apply(lambda x: x['d2web'] if x['zone'] == 497 else x['d2mbes'], axis=1)
    bg_df['d2web'] = bg_df['d2web'] * bg_df['total']
    bg_df['d2mbes'] = bg_df['d2mbes'] * bg_df['total']
    # true_distance = web if zone == 497 else mbes
    bg_df['true_distance'] = bg_df.apply(lambda x: x['d2web'] if x['zone'] == 497 else x['d2mbes'], axis=1)

    # create bar graph of travel time distribution using matplotlib with 1 bar for each school
    # duplicate values based on the number of students in each blockgroup
    kumar_mapping = {497: 'crimson', 999: 'midnightblue', 0: 'white'}
    mbes_dists = bg_df[bg_df['zone'] == 999][['unweighted_distance', 'total']]
    mbes_dists = mbes_dists['unweighted_distance'].repeat(mbes_dists['total'])
    webster_dists = bg_df[bg_df['zone'] == 497][['unweighted_distance', 'total']]
    webster_dists = webster_dists['unweighted_distance'].repeat(webster_dists['total'])
    # create ax
    fig, ax = plt.subplots()
    # add offset for each bar
    ax.hist([mbes_dists, webster_dists], bins=20, label=['MBES', 'Webster'],
            color=[kumar_mapping[999], kumar_mapping[497]])
    # add x line for average distance to each school
    ax.axvline(bg_df[bg_df['zone'] == 999]['true_distance'].sum() / bg_df[bg_df['zone'] == 999]['total'].sum(),
               color=kumar_mapping[999], linestyle='dashed', linewidth=1, label='MBES Avg')
    ax.axvline(bg_df[bg_df['zone'] == 497]['true_distance'].sum() / bg_df[bg_df['zone'] == 497]['total'].sum(),
               color=kumar_mapping[497], linestyle='dashed', linewidth=1, label='Webster Avg')
    ax.legend(loc='upper right')
    ax.set_title('Zoned School Distance Distribution')
    ax.set_xlabel('Distance (Miles)')
    ax.set_ylabel('Number of Students')
    plt.savefig(f'results/graphs/{config}_hist.png')
    plt.clf()

    overall_avg_distance = bg_df['true_distance'].sum() / bg_df['total'].sum()

    zone_demographics = bg_df.groupby('zone').sum()
    # change each of the above numbers to be perentages
    zone_demographics['white_percent'] = (100 * zone_demographics['white'] / zone_demographics['total']).round(1)
    zone_demographics['asian_percent'] = (100 * zone_demographics['asian'] / zone_demographics['total']).round(1)
    zone_demographics['hispanic_percent'] = (100 * zone_demographics['hispanic'] / zone_demographics['total']).round(1)
    zone_demographics['black_percent'] = (100 * zone_demographics['black'] / zone_demographics['total']).round(1)
    zone_demographics['filipino_percent'] = (100 * zone_demographics['filipino'] / zone_demographics['total']).round(1)
    zone_demographics['pacific_islander_percent'] = (
            100 * zone_demographics['pacific_islander'] / zone_demographics['total']).round(1)
    zone_demographics['arab_percent'] = (100 * zone_demographics['arab'] / zone_demographics['total']).round(1)
    zone_demographics['other_percent'] = (100 * zone_demographics['other'] / zone_demographics['total']).round(1)
    zone_demographics['frl_percent'] = (100 * zone_demographics['frl'] / zone_demographics['total']).round(1)
    zone_demographics['total_percent'] = (100 * zone_demographics['total'] / zone_demographics['total'].sum()).round(1)
    zone_demographics['avg_distance'] = zone_demographics['true_distance'] / zone_demographics['total']

    # calculate mean absolute deviation
    total_deviation = 0
    MAD = 0
    for demo in ['white', 'asian', 'hispanic', 'black', 'frl']:
        prop = zone_demographics[f'{demo}_percent'].sum() / 2
        total_deviation += abs((zone_demographics.loc[999][f'{demo}_percent'] - prop) / prop)
        total_deviation += abs((zone_demographics.loc[497][f'{demo}_percent'] - prop) / prop)
        MAD += abs((zone_demographics.loc[999][f'{demo}_percent'] - zone_demographics.loc[497][f'{demo}_percent']))
    MAD /= 5
    print(f'Mean Absolute Deviation: {MAD}')

    print(f'Overall Average Distance: {overall_avg_distance}')

    print(f'Mean Absolute Percent Deviation: {total_deviation / 5 * 100}')
    for zone in [999, 497]:
        print(f'Zone {"MBES" if zone == 999 else "Webster"}')
        print(f'White: {zone_demographics.loc[zone]["white_percent"]}')
        print(f'Asian: {zone_demographics.loc[zone]["asian_percent"]}')
        print(f'Hispanic: {zone_demographics.loc[zone]["hispanic_percent"]}')
        print(f'Black: {zone_demographics.loc[zone]["black_percent"]}')
        print(f'Other: {zone_demographics.loc[zone]["other_percent"]}')
        print(f'FRL: {zone_demographics.loc[zone]["frl_percent"]}')
        print(f'Total: {zone_demographics.loc[zone]["total_percent"]}')

        print(f'Average Distance: {zone_demographics.loc[zone]["avg_distance"]}')
    zone_demographics['avg_distance'] = zone_demographics['avg_distance'].round(3)
    zone_demographics['MAPD'] = total_deviation / 5 * 100
    zone_demographics['MAD'] = MAD
    zone_demographics['overall_avg_distance'] = overall_avg_distance
    zone_demographics['School Name'] = zone_demographics.index.map(
        lambda x: school_df[school_df['school_id'] == x]['school_name'].iloc[0])
    zone_demographics[
        ['School Name', 'white_percent', 'asian_percent', 'hispanic_percent', 'black_percent', 'other_percent',
         'frl_percent',
         'total_percent', 'avg_distance', 'MAPD', 'MAD', 'overall_avg_distance']].to_csv(
        f'results/demos/{config}.csv')


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
    for x in vm:
        if solver.BooleanValue(vm[x]) == 1:
            zone_dict[x] = 999
        else:
            zone_dict[x] = 497
    path = os.path.expanduser('results/mapping/')
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = f'{config_to_str()}.csv'
    pd.DataFrame.from_dict(zone_dict, orient='index').to_csv(path + file_name, header=['school_id'],
                                                             index_label='Block')
    # with open(path + file_name, 'w') as f:
    #     copy_map = copy.deepcopy(zone_dict)
    #     # convert to csv
    #     copy_map = {int(k): v for k, v in copy_map.items()}
    #
    #     for k, v in copy_map.items():
    #         f.write(f'{k},{v}\n')

    # copy_map['int_map'] = int_map
    # json.dump(copy_map, f)
    zv = ZoneVisualizer('Block')
    zv.zones_from_dict(zone_dict, centroid_location=centroid_locations,
                       title=f'Mission Bay and Webster ES Attendance Area Zoning',
                       save_path=f'results/graphs/{config_to_str()}', label=True)
    plt.clf()
    analyze_demographics(zone_dict, school_df, bg_df)


def visualize_solution(zoned_path, school_df, bg_df, config):
    zone_dict = pd.read_csv(zoned_path)
    zone_dict = zone_dict.rename(columns={'Geoid20': 'Block', 'Geoid20 (group)': 'zone'})
    zone_dict = zone_dict.dropna(subset=['Block'])
    zone_dict['Block'] = zone_dict['Block'].astype(int)
    zone_dict['zone'] = zone_dict['zone'].apply(lambda x: 999 if x == 'MBES' else 497)
    zone_dict = dict(zip(zone_dict['Block'], zone_dict['zone']))
    centroid_locations = pd.DataFrame()
    centroid_locations['lat'] = 0
    centroid_locations['lon'] = 0
    for zone in [999, 497]:
        centroid_locations.loc[zone, 'lat'] = school_df[school_df['school_id'] == zone]['lat'].iloc[0]
        centroid_locations.loc[zone, 'lon'] = school_df[school_df['school_id'] == zone]['lon'].iloc[0]

    zv = ZoneVisualizer('Block')
    zv.zones_from_dict(zone_dict, centroid_location=centroid_locations,
                       title=f'Mission Bay and Webster ES Attendance Area Zoning',
                       save_path=f'results/graphs/smoothed_{config}', label=True)
    plt.clf()
    analyze_demographics(zone_dict, school_df, bg_df, f'smoothed_{config}')

def visualize_solutions():
    solver, vm, school_df, bg_df, centroids = prep_model()

    for file in os.listdir('smoothed/'):
        if '.DS_Store' in file:
            continue
        file_name = file.split('.')[0]
        seperated = file_name.split(' ')
        file_name = f'{seperated[0]}_{seperated[-1]}'
        visualize_solution(f'smoothed/{file}', school_df, bg_df, file_name)

def param_search():
    d_weights = [3, 6]
    f_weights = [3, 6]
    dist_weights = [5, 10, 20]
    max_pop_ratios = [2, 3]
    for d in d_weights:
        for f in f_weights:
            for dist in dist_weights:
                for max_pop in max_pop_ratios:
                    set_config(d, f, dist, max_pop)
                    print(f'Running {config_to_str()}')
                    visualize(*prep_model())


def plot_tradeoff():
    MAPDs = []
    avg_distances = []
    labels = []
    for file in os.listdir('results/demos/'):
        if '.DS_Store' in file:
            continue
        if '1_2_40_3' in file:
            continue
        df = pd.read_csv(f'results/demos/{file}')

        MAPDs.append(df.iloc[0]['MAD'])
        avg_distances.append(df.iloc[0]['overall_avg_distance'])
        labels.append(file.split('.')[0])
    fig, ax = plt.subplots()
    ax.scatter(avg_distances, MAPDs)
    ax.set_xlabel('Average Distance')
    ax.set_ylabel('Relative Segregation')
    ax.set_title('Tradeoff between Average Distance and Relative Segregation')

    # use adjust text library to prevent overlap
    import adjustText
    texts = [plt.text(avg_distances[i], MAPDs[i], labels[i]) for i in range(len(labels))]
    adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'), max_move=(20, 20))

    plt.savefig('results/graphs/tradeoff.png')
    plt.clf()


def main():
    # set_config(1, 2, 40, 3)
    # visualize(*prep_model())

    # param_search()
    plot_tradeoff()
    # visualize_solutions()


if __name__ == '__main__':
    main()
