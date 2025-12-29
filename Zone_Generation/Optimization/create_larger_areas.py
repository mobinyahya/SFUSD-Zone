import pickle
import time

import os

import networkx as nx
import pandas as pd
import yaml

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import load_census_shapefile, calculate_euc_distance, convert_to_block_zone_dict
from Zone_Generation.Config.Constants import AREA_ETHNICITIES, get_dropbox_path
from Zone_Generation.Optimization.graph_utils import partition_graph_metis_partial_constraint, \
    partition_graph_metis_constrained


def create_graph(dz, config) -> nx.Graph:
    """
    Create a graph from DesignZones where nodes represent areas and edges represent adjacency.

    Node attributes include demographics, capacity, and location information.
    Edges are created based on the neighbors dictionary.
    """
    G = nx.Graph()

    census_sf = load_census_shapefile('Block', False)
    df = census_sf.dissolve(by="Block", as_index=False)
    df["centroid"] = df.centroid
    df["Lat"] = df["centroid"].apply(lambda x: x.y)
    df["Lon"] = df["centroid"].apply(lambda x: x.x)
    df = df[["Block", "Lat", "Lon"]]
    df.loc[:, "key"] = 0
    df = df.merge(df, how="outer", on="key")

    df.rename(
        columns={
            "Lat_x": "Lat",
            "Lon_x": "Lon",
            "Lat_y": "st_lat",
            "Lon_y": "st_lon",
            "Block_x": "Block",
        },
        inplace=True,
    )

    # df["distance"] = df.apply(get_distance, axis=1)
    df['Block'] = df['Block'].astype('Int64')
    df.set_index('Block', inplace=True)

    school_path = f"{get_dropbox_path(config['is_local'])}/Data/Cleaned/schools_table_for_zone_development_updated.csv"
    school_df = pd.read_csv(school_path)
    distance_path = f"{get_dropbox_path(config['is_local'])}/Optimization/distances_b2b_schools.csv"

    distances = pd.read_csv(distance_path, index_col=config['level'])
    distances.columns = [int(float(x)) for x in distances.columns]

    distance_dict = {}
    rows = distances.index.tolist()
    cols = list(distances.columns)

    # Change the csv file into a double dictionary, so the distances can be accessed easier
    for area_i in rows:
        inner_dict = {}
        for area_j in cols:
            inner_dict[dz.area2idx[area_j]] = distances.loc[area_i, area_j]

        # school_id = school_df.loc[school_df[config['level']] == area_i, 'school_id'].iloc[0]
        # area_i is the school area id
        distance_dict[dz.area2idx[area_i]] = inner_dict

    # add as a graph attribute
    G.graph['distance_dict'] = distance_dict

    # 'english_score': float(area_row['english_score']),
    # 'math_score': float(area_row['math_score']),
    # 'greatschools_rating': float(area_row['greatschools_rating']),

    school_data = {}
    for _, row in school_df.iterrows():
        school_info = row.to_dict()
        school_id = school_info['school_id']
        school_info.pop('school_id', None)
        school_data[school_id] = school_info
    G.graph['school_data'] = school_data
    # Add nodes with attributes from area_data
    for idx in range(dz.A):
        area_id = dz.idx2area[idx]
        area_row = dz.area_data.iloc[idx]

        lat = float(df.loc[area_id, 'Lat'].iloc[0])
        lon = float(df.loc[area_id, 'Lon'].iloc[0])

        schools_in_area = school_df[school_df[dz.level] == area_id]['school_id'].tolist()

        # Create node with attributes
        node_attrs = {
            'area_id': area_id,
            'ge_students': float(area_row['ge_students']),
            'ge_capacity': float(area_row['ge_capacity']),
            'all_prog_students': float(area_row['all_prog_students']),
            'all_prog_capacity': float(area_row['all_prog_capacity']),
            'num_schools': int(area_row['num_schools']),
            'FRL': float(area_row['FRL']),
            'school_ids': schools_in_area,
            'lat': lat,
            'lon': lon
        }

        # Add ethnicity attributes
        for ethnicity in AREA_ETHNICITIES:
            node_attrs[ethnicity] = float(area_row[ethnicity])

        G.add_node(idx, **node_attrs)

    total_students = 0
    total_f = 0
    total_r = {}
    for node in G.nodes(data=True):
        total_students += node[1]["ge_students"]
        total_f += node[1]['FRL']
        for ethnicity in AREA_ETHNICITIES:
            if ethnicity not in total_r:
                total_r[ethnicity] = 0
            total_r[ethnicity] += node[1][ethnicity]
    G.graph['F'] = total_f / total_students if total_students > 0 else 0
    r_props = {}
    for ethnicity in AREA_ETHNICITIES:
        r_props[ethnicity] = total_r[ethnicity] / total_students if total_students > 0 else 0
    G.graph['R'] = r_props

    # Add edges based on neighbors
    for idx in range(dz.A):
        for neighbor_idx in dz.neighbors[idx]:
            if not G.has_edge(idx, neighbor_idx):
                # Add edge with euclidean distance as weight
                G.add_edge(idx, neighbor_idx)

    return G


def partition_to_subgraphs(G, partition):
    """
    G: A networkx Graph
    partition: A list of lists/sets of nodes, e.g., [{1, 2}, {3, 4, 5}]
    """
    # Use G.subgraph(nodes).copy() if you need independent graphs
    # Use G.subgraph(nodes) for a read-only view (faster, saves memory)
    return [G.subgraph(nodes).copy() for nodes in partition]


def recursively_split_with_zones(G, cur_size, depth, zone_offset=0):
    """
    Returns (zone_dict, next_zone_id) where zone_dict maps node -> zone_id
    and next_zone_id is the next available zone ID.
    """
    if depth == 0:
        # Base case: assign all nodes in this subgraph to zone_offset
        # Then increment for the next partition
        return (
            {node: zone_offset for node in G.nodes()},
            zone_offset + 1  # Return next available zone ID
        )
    if cur_size <= 4:
        return (
            {node: zone_offset for node in G.nodes()},
            zone_offset + 1  # Return next available zone ID
        )

    # Partition current graph
    super_nodes = partition_graph_metis_partial_constraint(G, cur_size)

    zone_dict = {}
    current_zone_id = zone_offset

    for partition_nodes in super_nodes.values():
        sub_g = G.subgraph(partition_nodes).copy()

        # Recursively partition deeper
        sub_zones, next_zone_id = recursively_split_with_zones(
            sub_g,
            cur_size // 3,
            depth - 1,
            zone_offset=current_zone_id
        )

        zone_dict.update(sub_zones)
        current_zone_id = next_zone_id  # Use returned next_zone_id

    return zone_dict, current_zone_id


def aggregate_zone_dict(partition, G):
    # make new graph with partitioned nodes
    # partition: dict of node_id to partition_id
    # use partition_id as new area_id
    # determine adjaceny based on superset of neighbors from original graph
    new_G = nx.Graph()
    for node, part_id in partition.items():
        if part_id not in new_G:
            new_G.add_node(part_id, ge_students=0, ge_capacity=0,
                           all_prog_students=0, all_prog_capacity=0, num_schools=0,
                           FRL=0, english_score=0, math_score=0, greatschools_rating=0,
                           lat=0, lon=0, count=0)
        # aggregate attributes
        new_G.nodes[part_id]['ge_students'] += G.nodes[node]['ge_students']
        new_G.nodes[part_id]['ge_capacity'] += G.nodes[node]['ge_capacity']
        new_G.nodes[part_id]['all_prog_students'] += G.nodes[node]['all_prog_students']
        new_G.nodes[part_id]['all_prog_capacity'] += G.nodes[node]['all_prog_capacity']
        new_G.nodes[part_id]['num_schools'] += G.nodes[node]['num_schools']
        new_G.nodes[part_id]['FRL'] += G.nodes[node]['FRL']
        new_G.nodes[part_id].setdefault('school_ids', []).extend(G.nodes[node]['school_ids'])
        for ethnicity in AREA_ETHNICITIES:
            if ethnicity not in new_G.nodes[part_id]:
                new_G.nodes[part_id][ethnicity] = 0
            new_G.nodes[part_id][ethnicity] += G.nodes[node][ethnicity]

        # create new attribute of block_ids, which is a list of original area_ids
        new_G.nodes[part_id].setdefault('block_ids', []).append(G.nodes[node]['area_id'])

    # load census shapefile
    census_df = load_census_shapefile('Block', False)
    census_df = census_df[['Block', 'geometry']]

    # add column indicating which partition each block belongs to
    block_to_part = {}
    for node, part_id in partition.items():
        area_id = G.nodes[node]['area_id']
        block_to_part[area_id] = part_id
    census_df['part_id'] = census_df['Block'].map(block_to_part)
    # dissolve by part_id to get new geometries
    dissolved_df = census_df.dissolve(by='part_id', as_index=False)
    dissolved_df = dissolved_df[['part_id', 'geometry']]
    dissolved_df.set_index('part_id', inplace=True)

    # compute centroids for each new partition
    dissolved_df['centroid'] = dissolved_df.centroid
    dissolved_df['lat'] = dissolved_df['centroid'].apply(lambda x: x.y)
    dissolved_df['lon'] = dissolved_df['centroid'].apply(lambda x: x.x)
    for part_id in new_G.nodes():
        if part_id in dissolved_df.index:
            new_G.nodes[part_id]['lat'] = dissolved_df.loc[part_id, 'lat']
            new_G.nodes[part_id]['lon'] = dissolved_df.loc[part_id, 'lon']
    # create edges based on dissolved geometries adjacency
    dissolved_df['geometry'] = dissolved_df['geometry'].buffer(0)
    dissolved_df['neighbors'] = dissolved_df.geometry.apply(
        lambda x: dissolved_df[dissolved_df.geometry.touches(x)].index.tolist()
    )
    for part_id, row in dissolved_df.iterrows():
        if len(row['neighbors']) == 0:
            print(f"Warning: Partition {part_id} has no neighbors!")
        for neighbor_part_id in row['neighbors']:
            new_G.add_edge(part_id, neighbor_part_id)

    # TODO: Figure out how to handle this dumb thing

    # normalize school quality metrics by number of schools
    # num_schools = new_G.nodes[part_id]['num_schools']
    # if num_schools > 0:
    #     new_G.nodes[part_id]['english_score'] /= num_schools
    #     new_G.nodes[part_id]['math_score'] /= num_schools
    #     new_G.nodes[part_id]['greatschools_rating'] /= num_schools

    # recompute the distant_dict by calculating distances between new centroids
    # keep in mind that the distance dict is idx of a school to all other blocks
    distance_dict = {}
    # first get all the schools
    for node_i in new_G.nodes():
        distance_dict[node_i] = {}
        lat_i = new_G.nodes[node_i]['lat']
        lon_i = new_G.nodes[node_i]['lon']
        for node_j in new_G.nodes():
            if node_i == node_j:
                distance = 0
            else:
                lat_j = new_G.nodes[node_j]['lat']
                lon_j = new_G.nodes[node_j]['lon']
                distance = calculate_euc_distance(lat_i, lon_i, lat_j, lon_j)
            distance_dict[node_i][node_j] = distance
    new_G.graph['distance_dict'] = distance_dict
    new_G.graph['F'] = G.graph['F']
    new_G.graph['R'] = G.graph['R']
    new_G.graph['school_data'] = G.graph['school_data']
    new_G.graph['partition'] = partition
    return new_G


def create_base_graph(save_folder):
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config['level'] = 'Block'
    start_time = time.time()
    dz = DesignZones(config=config)
    end_time = time.time()
    print(f"DesignZones created in {end_time - start_time:.2f} seconds")
    G = create_graph(dz, config)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Sample node attributes: {list(G.nodes(data=True))[0]}")
    file_name = f"{save_folder}/Block_0.pickle"
    # if path does not exist, create it
    os.makedirs(save_folder, exist_ok=True)

    # save to file
    with open(file_name, 'wb') as f:
        pickle.dump(G, f)


def recursively_split_and_save(output_folder):
    with open(f'{output_folder}/Block_0.pickle', 'rb') as f:
        G = pickle.load(f)

    zv = ZoneVisualizer('Block', is_local=False)
    # Unpack both return values
    for depth in range(1, 4):
        zone_dict, _ = recursively_split_with_zones(G, 4 ** 3, depth=depth)

        print(f"Depth {4 - depth}:")
        print(f" Total nodes assigned: {len(zone_dict)} / {G.number_of_nodes()}")
        print(f" Number of zones: {len(set(zone_dict.values()))}")
        print(f" Original number of nodes: {G.number_of_nodes()}")

        block_zone_dict = convert_to_block_zone_dict(zone_dict, G)
        zv.zones_from_dict(block_zone_dict, show_plot=True)

        # Save zone_dict to file
        # with open(f'block_zones_depth_{4 - depth}.pickle', 'wb') as f:
        #     pickle.dump(zone_dict, f)
        file_name = f"{output_folder}/block_zones_depth_{4-depth}.pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(zone_dict, f)


def create_intermediate_graphs(output_folder):
    with open(f'{output_folder}/Block_0.pickle', 'rb') as f:
        G = pickle.load(f)

    for level in range(1, 4):
        with open(f'{output_folder}/block_zones_depth_{level}.pickle', 'rb') as f:
            zone_dict = pickle.load(f)

        aggregated_G = aggregate_zone_dict(zone_dict, G)
        print(f'Number of connected components at depth {level}: {nx.number_connected_components(aggregated_G)}')
        print(f"Aggregated graph at depth {level} has {aggregated_G.number_of_nodes()} nodes.")

        with open(f'{output_folder}/Block_{level}.pickle', 'wb') as f:
            pickle.dump(aggregated_G, f)


if __name__ == "__main__":
    is_local = False
    output_folder = f'{get_dropbox_path(is_local)}/Optimization/Zones/Graphs'

    # create_base_graph(output_folder)
    # recursively_split_and_save(output_folder)
    # create_intermediate_graphs(output_folder)
    with open('../Config/config.yaml', "r") as f:
        config = yaml.safe_load(f)
    with open(f'{output_folder}/Block_0.pickle', 'rb') as f:
        G = pickle.load(f)

    # open centroids file
    with open("../Config/centroids.yaml", "r") as f:
        centroid_configs = yaml.safe_load(f)
    if config['centroids_type'] not in centroid_configs:
        raise ValueError("The centroids type specified is not defined in centroids.yaml.")

    centroid_schools = centroid_configs[config['centroids_type']]
    # search graph for centroid_school in node['school_ids']
    centroids = []
    for centroid_school in centroid_schools:
        for node in G.nodes(data=True):
            if centroid_school in node[1]['school_ids']:
                centroids.append(node[0])
                break

    super_nodes = partition_graph_metis_constrained(G, len(centroids), centroids)
    zone_dict = {}
    for zone_id, nodes in super_nodes.items():
        for node in nodes:
            zone_dict[node] = zone_id

    block_zone_dict = convert_to_block_zone_dict(zone_dict, G)
    zv = ZoneVisualizer('Block', is_local)
    zv.zones_from_dict(block_zone_dict, show_plot=True)
