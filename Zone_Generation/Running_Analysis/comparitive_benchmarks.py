import json
import os

import pandas as pd
import yaml

from Zone_Generation.Optimization.optimizer import DesignZones, Optimizer, SolutionOutput


def run_configs():
    solve_time_limits = [5 * 60 * 60]  # in seconds
    random_seeds = [42, 2025, 1014, 7]

    centroids_types = [
        '6-zone-2',
        '8-zone-25',
        '10-zone-3',
        '13-zone-6',
    ]

    levels = ['Block_0']
    frl_devs = [0.15, 0.25]
    racial_devs = [0.2]
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

                                optimizer = Optimizer.get_optimizer(config)
                                optimizer.add_constraints()
                                optimizer.add_objective()

                                folder_name = (f"time{time_limit}_seed{seed}_centroids{centroids_type}_"
                                               f"level{level}_frl{frl_dev}_racial{racial_dev}_opt{optimizer_name}")
                                # output_folder = f"~/sfusd-local-data/zones/SFUSD/local_runs/comparisons/{folder_name}"
                                output_folder = f"~/sfusd-local-data/zones/SFUSD/local_runs/comparisons/{folder_name}"

                                # make the folder if it does not exist
                                os.makedirs(os.path.expanduser(output_folder), exist_ok=True)
                                try:
                                    solution_output = optimizer.solve()
                                except Exception as e:
                                    print(f"Error solving with config: {e}")
                                    # save a file at the folder with the error message
                                    with open(os.path.expanduser(f"{output_folder}/error.txt"), "w") as f:
                                        f.write(str(e))
                                    continue

                                solution_output.save_output(output_folder)


def compare_across_configs():
    import re

    # iterate through the saved outputs and compare the results as dataframe, save to csv
    root_folder = '~/sfusd-local-data/zones/SFUSD/local_runs/comparisons/'

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
    compare_across_configs()
    # test_across_configs()
