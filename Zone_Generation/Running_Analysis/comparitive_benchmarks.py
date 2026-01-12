import copy
import datetime
import json
import os

import pandas as pd
import yaml
import seaborn as sns

from Zone_Generation.Optimization.optimizer import Optimizer
from Zone_Generation.Optimization.recursive_zoning import recursive_zoning


def run_configs():
    solve_time_limits = [10 * 60]  # in seconds
    random_seeds = [42, 14]

    centroids_types = [
        '6-zone-2',
        '6-zone-3',
        '7-zone-14',
        '7-zone-19',
        '8-zone-25',
        '8-zone-22',
        '10-zone-11',
        '10-zone-3',
        '13-zone-6',
        '13-zone-5',
        '18-zone-7',
        '18-zone-10'
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
        [['Block_0', 1]],

        [['Block_1', 3 / 4], ['Block_0', 1 / 4]],
        [['Block_1', 1 / 2], ['Block_0', 1 / 2]],

        [['Block_2', 2 / 3], ['Block_0', 1 / 3]],
        [['Block_2', 1 / 2], ['Block_0', 1 / 2]],

        [['Block_2', 1 / 2], ['Block_1', 1 / 4], ['Block_0', 1 / 4]],
        [['Block_2', 1 / 3], ['Block_1', 1 / 3], ['Block_0', 1 / 3]]
    ]

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

    total_times = [10 * 60, 5 * 60]  # total time for all levels

    with open("../Config/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
    for total_time in total_times:
        for seed in random_seeds:
            for centroids_type in centroids_types:
                for computation in computations:
                    config = copy.deepcopy(base_config)
                    config['random_seed'] = seed
                    config['centroids_type'] = centroids_type
                    config['recursive_levels'] = [level for level, _ in computation]
                    config['relative_gap_limits'] = [0 for _ in computation]
                    config['solve_time_limits'] = [int(total_time * proportion) for _, proportion in computation]

                    print(f"Testing recursive config: seed={seed}, centroids_type={centroids_type}, "
                          f"levels={config['recursive_levels']}, "
                          f"time_limits={config['solve_time_limits']}")
                    print(datetime.datetime.now())

                    folder_name = (f"{centroids_type}/{seed}/{total_time}/"
                                   f"{'-'.join(config['recursive_levels'])}"
                                   f"_tl_{'-'.join([str(tl) for tl in config['solve_time_limits']])}")

                    output_folder = os.path.expanduser(
                        f"~/sfusd-local-data/zones/SFUSD/local_runs/recursive-runs/{folder_name}")

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
    # We drop rows where wall_time or objective_value is NaN (e.g., ERROR status)
    # to ensure the averages are calculated on valid data only.
    df_clean = df.dropna(subset=['wall_time', 'objective_value'])
    # also divide time limits by 60 to convert to minutes
    df_clean['time_limits'] = df_clean['time_limits'].apply(
        lambda x: '-'.join([str(round(int(tl) / 60, 2)) for tl in x.split('-')])
    )

    # 3. Group and Average
    # We group by:
    #  - centroid_num: to separate results by problem size/type
    #  - levels & time_limits: these combined define the "strategy"
    # We aggregate by averaging over the 'seed' entries.
    grouped_df = df_clean.groupby(['centroid_num', 'levels', 'time_limits'])[
        ['wall_time', 'objective_value']
    ].mean().reset_index()

    # 4. Create Plots
    # Use a consistent style
    sns.set_style("whitegrid")

    # --- Plot 1: Objective Value ---
    # We use 'catplot' to create a grid of plots (facets) based on centroid_num
    g1 = sns.catplot(
        data=grouped_df,
        kind='bar',
        x='levels',
        y='objective_value',
        hue='time_limits',  # Separation by time limits using color
        col='centroid_num',  # Separation by centroid number using facets
        col_wrap=3,  # Adjust layout (e.g., 3 plots per row)
        height=4,
        aspect=1.2,
        sharey=False  # Allow different y-scales for different centroid numbers
    )
    g1.figure.subplots_adjust(top=0.9)
    g1.figure.suptitle('Average Objective Value by Strategy (Levels + Time Limits)')

    # Rotate x-axis labels for readability
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
        y='wall_time',
        hue='time_limits',
        col='centroid_num',
        col_wrap=3,
        height=4,
        aspect=1.2,
        sharey=True  # Wall times share the same scale (limit ~600s)
    )
    g2.figure.subplots_adjust(top=0.9)
    g2.figure.suptitle('Average Wall Time by Strategy (Levels + Time Limits)')

    for ax in g2.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    g2.savefig('wall_time_comparison.png')
    print("Saved wall_time_comparison.png")


def compare_across_configs():
    import re

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


if __name__ == "__main__":
    # compare_across_configs()
    # run_configs()
    # run_recursive_configs()
    # aggregate_recursive_results()
    # test_across_configs()
    file = '~/sfusd-local-data/zones/SFUSD/local_runs/recursive-runs/comparative_recursive_results.csv'
    expanded_file = os.path.expanduser(file)
    analyze_and_plot(expanded_file)
