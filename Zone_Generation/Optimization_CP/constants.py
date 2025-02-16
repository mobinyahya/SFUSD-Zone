MAX_SOLVER_TIME =60 * 5

NUM_SOLVER_THREADS = 5
RACES = ['asian', 'white', 'hispanic', 'black']
YEAR = 2223
# CENTROIDS = '3-zone-0'
# CENTROIDS = '6-zone-1'
# BAD_NEIGHBORS = {60750107004.0, 60750611003.0, 60750127002.0, 60750165004.0, 60750478011.0, 60750479014.0,
#                  60750119011.0, 60750128002.0, 60750305002.0}
SCALING_FACTOR = 100
CONFIG = {
    'frl_weight': 3,
    'diversity_weight': 1,
    'distance_weight': 20,
    'max_pop_ratio': 3
}
def config_to_str():
    return f'{CONFIG["frl_weight"]}_{CONFIG["diversity_weight"]}_{CONFIG["distance_weight"]}_{CONFIG["max_pop_ratio"]}'

def set_config(frl_weight, diversity_weight, distance_weight, max_pop_ratio):
    CONFIG['frl_weight'] = frl_weight
    CONFIG['diversity_weight'] = diversity_weight
    CONFIG['distance_weight'] = distance_weight
    CONFIG['max_pop_ratio'] = max_pop_ratio