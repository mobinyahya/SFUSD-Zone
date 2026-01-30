import copy
import datetime
import json
import os
import re

import pandas as pd
import yaml
import seaborn as sns

from Zone_Generation.Optimization.optimizer import Optimizer
from Zone_Generation.Optimization.recursive_zoning import recursive_zoning


def run_configs():
    solve_time_limits = [10 * 60]  # in seconds
    random_seeds = [42, 14]

    centroids_types = [
        '4-zone-rec-4',
        '4-zone-rec-3',
        '5-zone-AF',
        '5-zone-AF-relocated',
        '6-zone-2',
        '6-zone-3',
        '7-zone-14',
        '7-zone-19',
        '8-zone-25',
        '8-zone-22',
        '10-zone-11',
        '10-zone-3',
        '13-zone-6',
        '13-zone-5'
    ]

    levels = ['Block_0']
    frl_devs = [0.15, 0.2, 0.25, 0.3]
    racial_devs = [0.15, 0.2, 0.25, 0.3]
    optimizers = ['cp_int']

    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # navigate through all combinations, changing the parameters in the config

    for level in levels:
        for time_limit in solve_time_limits:
            for seed in random_seeds:
                for centroids_type in centroids_types:
                    for frl_dev in frl_devs:
                        for racial_dev in racial_devs:
                            for optimizer_name in optimizers:
                                config['solve_time_limit'] = time_limit
                                config['random_seed'] = seed
                                config['centroids_type'] = centroids_type
                                config['level'] = level
                                config['frl_dev'] = frl_dev
                                config['racial_dev'] = racial_dev
                                config['optimizer'] = optimizer_name

                                print(f"Testing config: time_limit={time_limit}, seed={seed}, "
                                      f"centroids_type={centroids_type}, level={level}, "
                                      f"frl_dev={frl_dev}, racial_dev={racial_dev}, "
                                      f"optimizer={optimizer_name}")
                                print(datetime.datetime.now())

                                optimizer = Optimizer.get_optimizer(config)
                                optimizer.add_constraints()
                                optimizer.add_objective()

                                folder_name = (f"time{time_limit}_seed{seed}_centroids{centroids_type}_"
                                               f"level{level}_frl{frl_dev}_racial{racial_dev}_opt{optimizer_name}")
                                output_folder = os.path.expanduser(
                                    f"~/sfusd-local-data/zones/SFUSD/local_runs/1-1-26-runs/{folder_name}")

                                # make the folder if it does not exist
                                os.makedirs(output_folder, exist_ok=True)
                                config['log_folder'] = output_folder

                                try:
                                    solution_output = optimizer.solve()
                                    solution_output.save_output(output_folder)
                                except Exception as e:
                                    print(f"Error solving with config: {e}")
                                    # save a file at the folder with the error message
                                    with open(os.path.expanduser(f"{output_folder}/error.txt"), "w") as f:
                                        f.write(str(e))
                                    continue


def run_recursive_configs():
    # similar to run_configs but for recursive zoning
    computations = [
        # [['BlockGroup_0', 1]],
        [['BlockGroup_1', 0.5], ['BlockGroup_0', 0.5]],

        # [['Block_1', 3 / 4], ['Block_0', 1 / 4]],
        # [['Block_1', 1 / 2], ['Block_0', 1 / 2]],
        #
        # [['Block_2', 2 / 3], ['Block_0', 1 / 3]],
        # [['Block_2', 1 / 2], ['Block_0', 1 / 2]],
        #
        # [['Block_2', 1 / 2], ['Block_1', 1 / 4], ['Block_0', 1 / 4]],
        # [['Block_2', 1 / 3], ['Block_1', 1 / 3], ['Block_0', 1 / 3]]
    ]

    random_seeds = [42]

    centroids_types = [
        '4-zone-rec-4',
        '4-zone-rec-3',
        '5-zone-AF',
        '5-zone-AF-relocated',
        '6-zone-2',
        '6-zone-3',
        '7-zone-14',
        '7-zone-19',
        '8-zone-25',
        '8-zone-22',
        '10-zone-11',
        '10-zone-3',
        '13-zone-6',
        '13-zone-5'
    ]

    frl_devs = [0.12, 0.15, 0.2, 0.25]
    racial_devs = [0.12, 0.15, 0.2, 0.25]

    overages = [0.7, 0.8, 0.9]
    shortages = [0.15, 0.2, 0.25]

    total_times = [4 * 60]  # total time for all levels

    with open("../Config/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    for frl_dev in frl_devs:
        for racial_dev in racial_devs:
            for overage in overages:
                for shortage in shortages:
                    for total_time in total_times:
                        for seed in random_seeds:
                            for centroids_type in centroids_types:
                                for computation in computations:
                                    config = copy.deepcopy(base_config)
                                    config['frl_dev'] = frl_dev
                                    config['racial_dev'] = racial_dev
                                    config['overage'] = overage
                                    config['shortage'] = shortage
                                    config['random_seed'] = seed
                                    config['centroids_type'] = centroids_type
                                    config['recursive_levels'] = [level for level, _ in computation]
                                    config['relative_gap_limits'] = [0.05 for _ in computation]
                                    config['solve_time_limits'] = [int(total_time * proportion) for _, proportion in
                                                                   computation]

                                    print(f"Testing recursive config: seed={seed}, centroids_type={centroids_type}, "
                                          f"levels={config['recursive_levels']}, "
                                          f"time_limits={config['solve_time_limits']}")
                                    print(datetime.datetime.now())

                                    folder_name = (
                                        f"{centroids_type}/{seed}/{total_time}/{frl_dev}/{racial_dev}/{overage}/{shortage}/"
                                        f"{'-'.join(config['recursive_levels'])}"
                                        f"_tl_{'-'.join([str(tl) for tl in config['solve_time_limits']])}")

                                    output_folder = os.path.expanduser(
                                        f"~/sfusd-local-data/zones/SFUSD/local_runs/llm_bg_runs/{folder_name}")

                                    # make the folder if it does not exist
                                    os.makedirs(output_folder, exist_ok=True)
                                    config['log_folder'] = output_folder

                                    try:
                                        solutions = recursive_zoning(config)
                                        # save the last solution output
                                        for solution_output in solutions:
                                            solution_output.save_output(output_folder)
                                    except Exception as e:
                                        print(f"Error solving with recursive config: {e}")
                                        # save a file at the folder with the error message
                                        with open(os.path.expanduser(f"{output_folder}/error.txt"), "w") as f:
                                            f.write(str(e))
                                        continue


def aggregate_recursive_results():
    # average the results across seeds and between centroid types, but group by centroid number
    # so that we can compare 4-zone, 5-zone, etc.
    import re
    root_folder = '~/sfusd-local-data/zones/SFUSD/local_runs/recursive-runs/'
    expanded_root = os.path.expanduser(root_folder)
    centroid_folders = [f for f in os.listdir(expanded_root) if
                        os.path.isdir(os.path.join(expanded_root, f))]
    results = []
    pattern = re.compile(
        r"^(?P<centroid_num>\d+)-zone-[^/]+/(?P<seed>[^/]+)/(?P<time_limit>[^/]+)/"
        r"(?P<levels>.+)_tl_(?P<time_limits>.+)$"
    )

    # only take the Block_0 level results for comparison
    for centroid_folder in centroid_folders:
        centroid_folder_path = os.path.join(expanded_root, centroid_folder)
        seed_folders = [f for f in os.listdir(centroid_folder_path) if
                        os.path.isdir(os.path.join(centroid_folder_path, f))]
        for seed_folder in seed_folders:
            seed_folder_path = os.path.join(centroid_folder_path, seed_folder)
            time_limit_folders = [f for f in os.listdir(seed_folder_path) if
                                  os.path.isdir(os.path.join(seed_folder_path, f))]
            for time_limit_folder in time_limit_folders:
                time_limit_folder_path = os.path.join(seed_folder_path, time_limit_folder)
                computation_folders = [f for f in os.listdir(time_limit_folder_path) if
                                       os.path.isdir(os.path.join(time_limit_folder_path, f))]
                for computation_folder in computation_folders:
                    computation_folder_path = os.path.join(time_limit_folder_path, computation_folder)
                    # load all the solution outputs in the computations for this level, and sum the wall times
                    # use the solution output for Block_0 level for the objective value and status
                    try:
                        filename = os.path.join(computation_folder_path, "solution_info_Block_0.json")
                        with open(filename, "r") as f:
                            output_info = json.load(f)
                        status = output_info.get('status')
                        wall_time = output_info.get('wall_time')
                        objective_value = output_info.get('boundary_cost')
                        for level in computation_folder.split('_tl_')[0].split('-'):
                            if level == 'Block_0':
                                continue
                            filename = os.path.join(computation_folder_path, f"solution_info_{level}.json")
                            with open(filename, "r") as f:
                                output_info = json.load(f)
                            wall_time += output_info.get('wall_time')
                    except Exception:
                        status = 'ERROR'
                        wall_time = None
                        objective_value = None
                    m = pattern.match(f"{centroid_folder}/{seed_folder}/{time_limit_folder}/{computation_folder}")
                    if m:
                        param_dict = m.groupdict()
                    else:
                        param_dict = {
                            'centroid_num': None,
                            'seed': None,
                            'time_limit': None,
                            'levels': None,
                            'time_limits': None
                        }

                    result_entry = {
                        'centroid_num': param_dict.get('centroid_num'),
                        'centroid_type': centroid_folder,
                        'seed': param_dict.get('seed'),
                        'time_limit': param_dict.get('time_limit'),
                        'levels': param_dict.get('levels'),
                        'time_limits': param_dict.get('time_limits'),
                        'status': status,
                        'wall_time': wall_time,
                        'objective_value': objective_value
                    }
                    results.append(result_entry)

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.expanduser(f"{root_folder}/comparative_recursive_results.csv"), index=False)


def analyze_and_plot(filename):
    # 1. Load the dataframe
    df = pd.read_csv(filename)

    # 2. Filter for valid runs
    df_clean = df.dropna(subset=['wall_time', 'objective_value'])
    df_clean['time_limits'] = df_clean['time_limits'].apply(
        lambda x: '-'.join([str(round(int(tl) / 60, 1)) for tl in x.split('-')])
    )

    # ignore Block_0 level results
    df_clean = df_clean[df_clean['levels'] != 'Block_0']

    # 3. Group and calculate mean, min, max
    grouped_df = df_clean.groupby(['centroid_num', 'levels', 'time_limits']).agg({
        'wall_time': ['mean', 'min', 'max'],
        'objective_value': ['mean', 'min', 'max']
    }).reset_index()

    # Flatten column names
    grouped_df.columns = ['centroid_num', 'levels', 'time_limits',
                          'wall_time_mean', 'wall_time_min', 'wall_time_max',
                          'objective_value_mean', 'objective_value_min', 'objective_value_max']

    # Calculate error bar distances (distance from mean to min/max)
    grouped_df['wall_time_yerr_lower'] = grouped_df['wall_time_mean'] - grouped_df['wall_time_min']
    grouped_df['wall_time_yerr_upper'] = grouped_df['wall_time_max'] - grouped_df['wall_time_mean']
    grouped_df['objective_yerr_lower'] = grouped_df['objective_value_mean'] - grouped_df['objective_value_min']
    grouped_df['objective_yerr_upper'] = grouped_df['objective_value_max'] - grouped_df['objective_value_mean']

    # Rename centroid_num to Num Zones
    grouped_df = grouped_df.rename(columns={'centroid_num': 'Num Zones'})

    # 4. Create Plots with error bars showing min/max
    sns.set_style("whitegrid")

    # --- Plot 1: Objective Value ---
    g1 = sns.catplot(
        data=grouped_df,
        kind='bar',
        x='levels',
        y='objective_value_mean',
        hue='time_limits',
        col='Num Zones',
        col_wrap=4,
        height=4,
        aspect=1.2,
        sharey=False,
        errorbar=None
    )

    # Add custom error bars
    for ax, (zone_num, zone_data) in zip(g1.axes.flat, grouped_df.groupby('Num Zones')):
        for i, (idx, row) in enumerate(zone_data.iterrows()):
            # Find the correct bar position
            level_idx = list(zone_data['levels'].unique()).index(row['levels'])
            hue_idx = list(zone_data['time_limits'].unique()).index(row['time_limits'])
            n_hues = len(zone_data['time_limits'].unique())
            width = ax.patches[0].get_width()
            x = level_idx + (hue_idx - n_hues / 2 + 0.5) * width

            yerr = [[row['objective_yerr_lower']], [row['objective_yerr_upper']]]
            ax.errorbar(x, row['objective_value_mean'], yerr=yerr,
                        fmt='none', color='black', capsize=3, linewidth=1)

    g1.figure.subplots_adjust(top=0.9)
    g1.figure.suptitle('Average Objective Value by Strategy (Levels + Time Limits)\nError bars show min/max range')
    for ax in g1.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    g1.savefig('objective_value_comparison.png')
    print("Saved objective_value_comparison.png")

    # --- Plot 2: Wall Time ---
    g2 = sns.catplot(
        data=grouped_df,
        kind='bar',
        x='levels',
        y='wall_time_mean',
        hue='time_limits',
        col='Num Zones',
        col_wrap=4,
        height=4,
        aspect=1.2,
        sharey=True,
        errorbar=None
    )

    # Add custom error bars
    for ax, (zone_num, zone_data) in zip(g2.axes.flat, grouped_df.groupby('Num Zones')):
        for i, (idx, row) in enumerate(zone_data.iterrows()):
            level_idx = list(zone_data['levels'].unique()).index(row['levels'])
            hue_idx = list(zone_data['time_limits'].unique()).index(row['time_limits'])
            n_hues = len(zone_data['time_limits'].unique())
            width = ax.patches[0].get_width()
            x = level_idx + (hue_idx - n_hues / 2 + 0.5) * width

            yerr = [[row['wall_time_yerr_lower']], [row['wall_time_yerr_upper']]]
            ax.errorbar(x, row['wall_time_mean'], yerr=yerr,
                        fmt='none', color='black', capsize=3, linewidth=1)

    g2.figure.subplots_adjust(top=0.9)
    g2.figure.suptitle('Average Wall Time by Strategy (Levels + Time Limits)\nError bars show min/max range')
    for ax in g2.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    g2.savefig('wall_time_comparison.png')
    print("Saved wall_time_comparison.png")


def compare_across_configs():

    # iterate through the saved outputs and compare the results as dataframe, save to csv
    root_folder = '~/sfusd-local-data/zones/SFUSD/local_runs/1-1-26-runs/'

    expanded_root = os.path.expanduser(root_folder)
    subfolders = [f for f in os.listdir(expanded_root) if
                  os.path.isdir(os.path.join(expanded_root, f))]
    results = []

    pattern = re.compile(
        r"^time(?P<time>[^_]+)_seed(?P<seed>[^_]+)_centroids(?P<centroids>[^_]+)"
        r"_level(?P<level>[^_]+)_frl(?P<frl>[^_]+)_racial(?P<racial>[^_]+)_opt(?P<opt>.+)$"
    )

    for folder in subfolders:
        output_folder = os.path.join(expanded_root, folder)
        # load the solution output
        try:
            filename = os.path.join(output_folder, "solution_info.json")
            with open(filename, "r") as f:
                output_info = json.load(f)
            status = output_info.get('status')
            wall_time = output_info.get('wall_time')
            objective_value = output_info.get('boundary_cost')
        except Exception:
            status = 'ERROR'
            wall_time = None
            objective_value = None

        m = pattern.match(folder)
        if m:
            param_dict = m.groupdict()
        else:
            param_dict = {
                'time': None,
                'seed': None,
                'centroids': None,
                'level': None,
                'frl': None,
                'racial': None,
                'opt': None
            }

        result_entry = {
            'time_limit': param_dict.get('time'),
            'seed': param_dict.get('seed'),
            'centroids_type': param_dict.get('centroids'),
            'level': param_dict.get('level'),
            'frl_dev': param_dict.get('frl'),
            'racial_dev': param_dict.get('racial'),
            'optimizer': param_dict.get('opt'),
            'status': status,
            'wall_time': wall_time,
            'objective_value': objective_value
        }
        results.append(result_entry)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.expanduser(f"{root_folder}/comparative_results.csv"), index=False)
    return results_df


def aggregate_recursive_metrics():
    import pickle
    from Zone_Generation.Config.Constants import get_dropbox_path
    from Zone_Generation.Running_Analysis.zoning_metrics import ZoneMetrics

    # Load Graph
    is_local = False
    dropbox_path = get_dropbox_path(is_local)
    graph_path = f'{dropbox_path}/Optimization/Zones/Graphs/BlockGroup_0.pickle'

    print(f"Loading graph from {graph_path}")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    root_folder = os.path.expanduser('~/sfusd-local-data/zones/SFUSD/local_runs/llm_bg_runs/')
    print(f"Scanning {root_folder} for zone_dict_BlockGroup_0.json files...")

    results = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(root_folder):
        if 'zone_dict_BlockGroup_0.json' in files:
            # Parse parameters from path relative to root_folder
            rel_path = os.path.relpath(root, root_folder)
            parts = rel_path.split('/')

            if len(parts) >= 8:
                # centroids_type/seed/total_time/frl_dev/racial_dev/overage/shortage/config_str
                centroids_type = parts[0]
                seed = parts[1]
                total_time = parts[2]
                frl_dev = parts[3]
                racial_dev = parts[4]
                overage = parts[5]
                shortage = parts[6]
                config_str = parts[7]

                # Load zone dict
                try:
                    with open(os.path.join(root, 'zone_dict_BlockGroup_0.json'), 'r') as f:
                        zone_dict = json.load(f)

                    # Compute Metrics
                    zm = ZoneMetrics(zone_dict.copy(), G)
                    metrics = zm.get_metrics()

                    # Add metadata
                    metrics['centroids_type'] = centroids_type
                    metrics['seed'] = seed
                    metrics['total_time'] = total_time
                    metrics['frl_dev'] = frl_dev
                    metrics['racial_dev'] = racial_dev
                    metrics['overage'] = overage
                    metrics['shortage'] = shortage
                    metrics['config_str'] = config_str

                    # include the path for reference
                    metrics['path'] = root

                    # Also try to retrieve solution_info_BlockGroup_0.json
                    if 'solution_info_BlockGroup_0.json' in files:
                        try:
                            with open(os.path.join(root, 'solution_info_BlockGroup_0.json'), 'r') as f:
                                sol_info = json.load(f)
                                metrics['wall_time'] = sol_info.get('wall_time')
                                metrics['boundary_cost'] = sol_info.get('boundary_cost')
                                metrics['status'] = sol_info.get('status')
                        except Exception as e:
                            print(f"Error reading solution_info in {root}: {e}")

                    results.append(metrics)
                except Exception as e:
                    print(f"Error processing {root}: {e}")

    if results:
        df = pd.DataFrame(results)
        output_csv = os.path.join(root_folder, 'recursive_metrics_flattened.csv')
        df.to_csv(output_csv, index=False)
        print(f"Saved metrics to {output_csv}")
    else:
        print("No results found.")


if __name__ == "__main__":
    aggregate_recursive_metrics()
    # compare_across_configs()
    # run_configs()
    # run_recursive_configs()
    # aggregate_recursive_results()
    # test_across_configs()
    # file = '~/sfusd-local-data/zones/SFUSD/local_runs/recursive-runs/comparative_recursive_results.csv'
    # expanded_file = os.path.expanduser(file)
    # analyze_and_plot(expanded_file)
