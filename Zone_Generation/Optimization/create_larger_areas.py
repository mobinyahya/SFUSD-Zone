import pickle
import time
from collections import defaultdict

import kahip

import networkx as nx
import numpy as np
import pymetis

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import load_census_shapefile
from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.Optimization.optimizer import DesignZones


def create_graph(dz: DesignZones) -> nx.Graph:
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

    # Add nodes with attributes from area_data
    for idx in range(dz.A):
        area_id = dz.idx2area[idx]
        area_row = dz.area_data.iloc[idx]

        lat = float(df.loc[area_id, 'Lat'].iloc[0])
        lon = float(df.loc[area_id, 'Lon'].iloc[0])

        # Create node with attributes
        node_attrs = {
            'area_id': area_id,
            'ge_students': float(area_row['ge_students']),
            'ge_capacity': float(area_row['ge_capacity']),
            'all_prog_students': float(area_row['all_prog_students']),
            'all_prog_capacity': float(area_row['all_prog_capacity']),
            'num_schools': int(area_row['num_schools']),
            'FRL': float(area_row['FRL']),
            'lat': lat,
            'lon': lon
        }

        # Add ethnicity attributes
        for ethnicity in AREA_ETHNICITIES:
            node_attrs[ethnicity] = float(area_row.get(ethnicity, 0))

        # Add school quality metrics if available
        if 'english_score' in area_row:
            node_attrs['english_score'] = float(area_row.get('english_score', 0))
        if 'math_score' in area_row:
            node_attrs['math_score'] = float(area_row.get('math_score', 0))
        if 'greatschools_rating' in area_row:
            node_attrs['greatschools_rating'] = float(area_row.get('greatschools_rating', 0))

        G.add_node(idx, **node_attrs)

    # Add edges based on neighbors
    for idx in range(dz.A):
        for neighbor_idx in dz.neighbors[idx]:
            if not G.has_edge(idx, neighbor_idx):
                # Add edge with euclidean distance as weight
                G.add_edge(idx, neighbor_idx)

    return G


def partition_graph_metis(G, k):
    """
    Partitions a NetworkX graph into super-nodes of approximate size k.

    Args:
        G: networkx.Graph
        k: The target size for each super-node.
    """
    # 1. Calculate number of partitions needed
    n_partitions = max(1, len(G.nodes) // k)
    # print(f"Partitioning into {n_partitions} super-nodes...")

    # 2. METIS requires nodes to be 0...N-1. Create a mapping.
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 3. Build Adjacency List for PyMetis
    # adjncy[i] contains the neighbors of node i
    adj_list = [
        [node_to_idx[neighbor] for neighbor in G.neighbors(node)]
        for node in nodes
    ]

    # 4. Perform Partitioning
    # cuts is the number of edges between super-nodes
    # membership is a list where membership[i] is the partition ID of node i
    cuts, membership = pymetis.part_graph(n_partitions, adjacency=adj_list)

    # 5. Group original nodes into their super-node sets
    super_nodes = {}
    for node_idx, partition_id in enumerate(membership):
        if partition_id not in super_nodes:
            super_nodes[partition_id] = []
        super_nodes[partition_id].append(nodes[node_idx])

    return super_nodes


def partition_graph_metis_partial_constraint(G, k):
    """
        Partitions a NetworkX graph into super-nodes of approximate size k.

        Args:
            G: networkx.Graph
            k: The target size for each super-node.
        """
    # 1. Calculate number of partitions needed
    n_partitions = max(1, len(G.nodes) // k)
    # print(f"Partitioning into {n_partitions} super-nodes...")

    # 2. METIS requires nodes to be 0...N-1. Create a mapping.
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 3. Build Adjacency List for PyMetis
    # adjncy[i] contains the neighbors of node i
    adj_list = [
        [node_to_idx[neighbor] for neighbor in G.neighbors(node)]
        for node in nodes
    ]

    vweights = []
    for node in nodes:
        # Add a base weight of 1 to schools so no partition is 'allowed' to be empty
        schools = int(G.nodes[node]['num_schools'] * 100)+ 1
        students = int(G.nodes[node]['ge_students'] * 10) + 2

        vweights.extend([schools, students])

    # 4. Perform Partitioning
    # cuts is the number of edges between super-nodes
    # membership is a list where membership[i] is the partition ID of node i
    options = pymetis.Options()
    options.ufactor = 25  # Set imbalance constraints
    options.niter = 30
    options.ncuts = 10
    # options.ubvec = ubvec
    options.contig = True

    cuts, membership = pymetis.part_graph(
        n_partitions,
        adjacency=adj_list,
        vweights=vweights,
        options=options,
    )

    # 5. Group original nodes into their super-node sets
    super_nodes = {}
    for node_idx, partition_id in enumerate(membership):
        if partition_id not in super_nodes:
            super_nodes[partition_id] = []
        super_nodes[partition_id].append(nodes[node_idx])

    partition_stats = defaultdict(lambda: {'schools': 0, 'students': 0})
    for ethnicity in AREA_ETHNICITIES:
        for part_id in range(n_partitions):
            partition_stats[part_id][ethnicity] = 0
    for i, part_id in enumerate(membership):
        node = nodes[i]
        partition_stats[part_id]['schools'] += G.nodes[node]['num_schools']
        partition_stats[part_id]['students'] += G.nodes[node]['ge_students']

    # Print Report
    # print("Partitioning Report:")
    # for part_id, stats in partition_stats.items():
    #     # percent difference between capacity and students
    #
    #     print(f" Partition {part_id}: "
    #           f"Schools={stats['schools']}, "
    #           f"Students=({int(stats['students'])}) ")
    return super_nodes


def partition_graph_metis_constrained(G, k):
    """
    Partitions a NetworkX graph into super-nodes of approximate size k.

    Args:
        G: networkx.Graph
        k: The target size for each super-node.
    """
    # 1. Calculate number of partitions needed
    n_partitions = max(1, len(G.nodes) // k)
    # print(f"Partitioning into {n_partitions} super-nodes...")

    # 2. METIS requires nodes to be 0...N-1. Create a mapping.
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 3. Build Adjacency List for PyMetis
    # adjncy[i] contains the neighbors of node i
    adj_list = [
        [node_to_idx[neighbor] for neighbor in G.neighbors(node)]
        for node in nodes
    ]

    total_frl = sum(G.nodes[node]['FRL'] for node in nodes)
    total_students = sum(G.nodes[node]['ge_students'] for node in nodes)
    target_frl_prop = total_frl / total_students
    target_eth_props = {}
    for ethnicity in AREA_ETHNICITIES:
        node_eth = sum(G.nodes[node][ethnicity] for node in nodes)
        target_eth_props[ethnicity] = node_eth / total_students

    print(f"Target FRL proportion: {target_frl_prop:.2%}")

    min_cap_diff = min(
        G.nodes[node]['ge_capacity'] - G.nodes[node]['ge_students']
        for node in nodes
    )
    vweights = []
    for node in nodes:
        # Add a base weight of 1 to schools so no partition is 'allowed' to be empty
        schools = int(G.nodes[node]['num_schools']) + 1

        # Make capacity difference always positive
        cap_diff = G.nodes[node]['ge_capacity'] - G.nodes[node]['ge_students']
        cap_coef = int(100 * (cap_diff - min_cap_diff))

        node_frl = G.nodes[node]['FRL']
        node_students = G.nodes[node]['ge_students']
        node_frl_prop = node_frl / node_students if node_students > 0 else 0

        # Weight by deviation from target (scaled by student count)
        # Higher weight = further from target
        frl_deviation = abs(node_frl_prop - target_frl_prop) * node_students
        frl_weight = int(1000 * frl_deviation) + 1

        r_weights = []
        for ethnicity in AREA_ETHNICITIES:
            node_ethnicity = G.nodes[node][ethnicity]
            node_eth_prop = node_ethnicity / node_students if node_students > 0 else 0
            eth_deviation = abs(node_eth_prop - target_eth_props[ethnicity]) * node_students
            eth_weight = int(1000 * eth_deviation) + 1
            r_weights.append(eth_weight)

        vweights.extend([cap_coef, schools, frl_weight] + r_weights)

    # 4. Perform Partitioning
    # cuts is the number of edges between super-nodes
    # membership is a list where membership[i] is the partition ID of node i
    options = pymetis.Options()
    options.ufactor = 30  # Set imbalance constraints
    options.niter = 30
    options.ncuts = 10
    # options.ubvec = ubvec

    cuts, membership = pymetis.part_graph(
        n_partitions,
        adjacency=adj_list,
        vweights=vweights,
        options=options,
        contiguous=True
    )

    # 5. Group original nodes into their super-node sets
    super_nodes = {}
    for node_idx, partition_id in enumerate(membership):
        if partition_id not in super_nodes:
            super_nodes[partition_id] = []
        super_nodes[partition_id].append(nodes[node_idx])

    # Aggregate results
    partition_stats = defaultdict(lambda: {'schools': 0, 'students': 0, 'capacity': 0, 'frl': 0})
    for ethnicity in AREA_ETHNICITIES:
        for part_id in range(n_partitions):
            partition_stats[part_id][ethnicity] = 0
    for i, part_id in enumerate(membership):
        node = nodes[i]
        partition_stats[part_id]['schools'] += G.nodes[node]['num_schools']
        partition_stats[part_id]['students'] += G.nodes[node]['ge_students']
        partition_stats[part_id]['capacity'] += G.nodes[node]['ge_capacity']
        partition_stats[part_id]['frl'] += G.nodes[node]['FRL']
        for ethnicity in AREA_ETHNICITIES:
            partition_stats[part_id][ethnicity] += G.nodes[node][ethnicity]

    # Print Report
    print("Partitioning Report:")
    for part_id, stats in partition_stats.items():
        # Add frl prop and race prop by dividing by students
        frl_prop = stats['frl'] / stats['students'] if stats['students'] > 0 else 0
        r_props = {}
        for ethnicity in AREA_ETHNICITIES:
            r_props[ethnicity] = stats[ethnicity] / stats['students'] if stats['students'] > 0 else 0

        # percent difference between capacity and students
        capacity_diff = (stats['capacity'] - stats['students']) / stats['students'] if stats['students'] > 0 else 0

        print(f" Partition {part_id}: "
              f"Schools={stats['schools']}, "
              f"Cap Diff=({capacity_diff:.2%}), "
              f"FRL=({frl_prop:.2%}), "
              + ", ".join([f"{ethnicity}= ({r_props[ethnicity]:.2%})" for ethnicity in AREA_ETHNICITIES]))
    return super_nodes


def partition_graph_kahip(G, k):
    num_nodes = G.number_of_nodes()
    n_partitions = max(1, num_nodes // k)

    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    xadj = [0]
    adjncy = []
    adjwgt = []

    for node in node_list:
        for neighbor in G.neighbors(node):
            adjncy.append(node_to_idx[neighbor])

            # --- COMPACTNESS TRICK ---
            # If nodes have 'pos' (x, y) coordinates:
            if 'pos' in G.nodes[node] and 'pos' in G.nodes[neighbor]:
                p1 = np.array([G.nodes[node]['lat'], G.nodes[node]['lon']])
                p2 = np.array([G.nodes[neighbor]['lat'], G.nodes[neighbor]['lon']])
                dist = np.linalg.norm(p1 - p2)
                # Penalize cutting short edges heavily
                weight = int(10000 / (dist + 0.1) ** 2)
            else:
                weight = 1

            adjwgt.append(weight)
        xadj.append(len(adjncy))

    # Vertex weights (uniform)
    vwgt = np.ones(num_nodes, dtype=np.int32).tolist()

    # Using the Strongest available mode
    edge_cuts, membership = kahip.kaffpa(
        vwgt,
        xadj,
        adjncy,
        adjwgt,
        int(n_partitions),
        0.01,  # Lower imbalance = tighter, more METIS-like balance
        False,
        1,
        int(kahip.STRONGSOCIAL)
    )

    super_nodes = {}
    for idx, partition_id in enumerate(membership):
        if partition_id not in super_nodes:
            super_nodes[partition_id] = []
        super_nodes[partition_id].append(node_list[idx])

    return super_nodes


def partition_to_subgraphs(G, partition):
    """
    G: A networkx Graph
    partition: A list of lists/sets of nodes, e.g., [{1, 2}, {3, 4, 5}]
    """
    # Use G.subgraph(nodes).copy() if you need independent graphs
    # Use G.subgraph(nodes) for a read-only view (faster, saves memory)
    return [G.subgraph(nodes).copy() for nodes in partition]


def recursively_split(G, cur_size, depth):
    """
    Recursively partition a graph into smaller subgraphs.

    Args:
        G: NetworkX graph to partition
        cur_size: Target size for partitioning at current depth
        depth: Number of recursive splits remaining
        collect_all_depths: If True, return graphs from all depths

    Returns:
        If collect_all_depths is False: list of final subgraphs
        If collect_all_depths is True: dict mapping depth -> list of subgraphs at that depth
    """

    # New behavior: collect graphs at all depths
    depth_graphs = {depth: [G]}

    if depth == 0 or cur_size <= 3:
        return depth_graphs

    if cur_size > 30:
        super_nodes = partition_graph_metis_partial_constraint(G, cur_size)
    else:
        super_nodes = partition_graph_metis(G, cur_size)
    sub_graphs = partition_to_subgraphs(G, super_nodes.values())

    # Recursively collect from subgraphs
    for sub_g in sub_graphs:
        sub_depth_graphs = recursively_split(sub_g, cur_size // 3, depth - 1)
        for d, graphs in sub_depth_graphs.items():
            if d not in depth_graphs:
                depth_graphs[d] = []
            depth_graphs[d].extend(graphs)

    return depth_graphs


if __name__ == "__main__":
    # with open("../Config/config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    #
    # config['level'] = 'Block'
    # start_time = time.time()
    # dz = DesignZones(config=config)
    # end_time = time.time()
    # print(f"DesignZones created in {end_time - start_time:.2f} seconds")
    # G = create_graph(dz)
    #
    # print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    # print(f"Sample node attributes: {list(G.nodes(data=True))[0]}")
    #
    #
    # # save to file
    # with open('Block_0.pickle', 'wb') as f:
    #     pickle.dump(G, f)

    # load from file
    start_time = time.time()
    with open('Block_0.pickle', 'rb') as f:
        G = pickle.load(f)
    end_time = time.time()
    print(f"Graph loaded from file in {end_time - start_time:.2f} seconds")

    # super_nodes = partition_graph_metis_partial_constraint(G, len(G.nodes()) // 13)
    #
    # # create zone_dict
    # zone_dict = {}
    # for zone_idx, nodes in super_nodes.items():
    #     for node in nodes:
    #         area_id = G.nodes[node]['area_id']
    #         zone_dict[area_id] = zone_idx
    # print(f"Created {len(super_nodes)} zones.")
    # zv = ZoneVisualizer('Block', is_local=False)
    # zv.zones_from_dict(zone_dict, show_plot=True)

    depth_subgraphs = recursively_split(G, 4 ** 3, depth=4)

    for depth, subgraphs in depth_subgraphs.items():
        if depth == 4:
            continue
        print(f"Depth {depth}: {len(subgraphs)} subgraphs")
        # Create zone_dict for visualization
        zone_dict = {}
        for zone_idx, sub_g in enumerate(subgraphs):
            for node in sub_g.nodes():
                area_id = G.nodes[node]['area_id']
                zone_dict[area_id] = zone_idx
        print(f"Created {len(subgraphs)} zones at depth {depth}.")
        zv = ZoneVisualizer('Block', is_local=False)
        zv.zones_from_dict(zone_dict, show_plot=True)

        # save zone_dict to pickle
        with open(f'block_zones_depth_{depth}.pickle', 'wb') as f:
            pickle.dump(zone_dict, f)
