import json
import pickle

import networkx as nx

from Helper_Functions.util import compute_zone_demographics, compute_zone_deviations, convert_to_block_zone_dict
from Zone_Generation.Config.Constants import get_dropbox_path, AREA_ETHNICITIES


class ZoneMetrics:
    def __init__(self, zone_dict: dict[int, int], G: nx.Graph):
        # convert zone_dict keys to ints
        for k in list(zone_dict.keys()):
            zone_dict[int(k)] = zone_dict.pop(k)

        # zone_dict = convert_to_block_zone_dict(zone_dict,G)
        self.zone_dict = zone_dict
        # map zone to the blocks in that zone
        self.zone_blocks = {}
        for block, zone in zone_dict.items():
            if zone not in self.zone_blocks:
                self.zone_blocks[zone] = []
            self.zone_blocks[zone].append(block)
        self.G = G

    def get_demo_deviation(self) -> dict[str, float]:
        # Get area-wide averages from graph
        area_frl_pct = self.G.graph['F']
        area_ethnicities = {eth: self.G.graph['R'][eth] for eth in AREA_ETHNICITIES}

        # Compute deviation for each zone
        deviations = {}
        for zone in self.zone_blocks:
            zone_demo = {'ge_students': 0, 'FRL': 0}
            for ethnicity in AREA_ETHNICITIES:
                zone_demo[ethnicity] = 0

            for block in self.zone_blocks[zone]:
                zone_demo['ge_students'] += self.G.nodes[block]['ge_students']
                zone_demo['FRL'] += self.G.nodes[block]['FRL']
                for ethnicity in AREA_ETHNICITIES:
                    zone_demo[ethnicity] += self.G.nodes[block][ethnicity]

            # Compute % deviation from area averages
            if zone_demo['ge_students'] > 0:
                if 'FRL' not in deviations:
                    deviations['FRL'] = []
                zone_frl_pct = zone_demo['FRL'] / zone_demo['ge_students']
                frl_deviation = abs(zone_frl_pct - area_frl_pct)
                deviations['FRL'].append(frl_deviation)

                for ethnicity in AREA_ETHNICITIES:
                    if ethnicity not in deviations:
                        deviations[ethnicity] = []
                    zone_eth_pct = zone_demo[ethnicity] / zone_demo['ge_students']
                    eth_deviation = abs(zone_eth_pct - area_ethnicities[ethnicity])
                    deviations[ethnicity].append(eth_deviation)

        # Average deviations across zones
        avg_deviations = {}
        for key in deviations:
            avg_deviations[key] = sum(deviations[key]) / len(deviations[key])

        return avg_deviations

    def get_seat_disparity(self) -> float:
        # compute average % shortage / overage across zones
        average_diff = 0
        for zone in self.zone_blocks:
            seats = 0
            students = 0
            for block in self.zone_blocks[zone]:
                seats += self.G.nodes[block]['ge_capacity']
                students += self.G.nodes[block]['ge_students']
            if students == 0:
                continue
            diff = abs(seats - students) / students
            average_diff += diff
        average_diff = average_diff / len(self.zone_blocks)
        return average_diff

    def closest_school_distances(self) -> float:
        # compute average distance to closest school across zones
        school_nodes = set()
        for node in self.G.nodes:
            if len(self.G.nodes[node]['school_ids']) > 0:
                school_nodes.add(node)
        total_distance = 0
        for node in self.G.nodes:
            closest_dist = float('inf')
            for school_node in school_nodes:
                dist = self.G.graph['distance_dict'][school_node][node]
                if dist < closest_dist:
                    closest_dist = dist
            total_distance += closest_dist
        return float(total_distance / self.G.number_of_nodes())

    def get_metrics(self) -> dict:
        metrics = self.get_demo_deviation()
        metrics['seat_disparity'] = self.get_seat_disparity()
        metrics['closest_school_distances'] = self.closest_school_distances()
        return metrics


if __name__ == "__main__":
    is_local = False
    output_folder = f'{get_dropbox_path(is_local)}/Optimization/Zones/Graphs'
    with open(f'{output_folder}/BlockGroup_0.pickle', 'rb') as f:
        G = pickle.load(f)

    # example zone dict
    zd_file = ('/home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/1-1-26-runs/'
               'time600_seed42_centroids13-zone-6_levelBlockGroup_0_frl0.3_racial0.3_optcp_int/zone_dict.json')
    with open(zd_file, 'r') as f:
        zone_dict = json.load(f)

    zm = ZoneMetrics(zone_dict, G)
    metrics = zm.get_metrics()
    print(metrics)
