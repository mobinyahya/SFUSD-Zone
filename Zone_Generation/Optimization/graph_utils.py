from collections import defaultdict

import kahip

import numpy as np
import pymetis
from matplotlib import pyplot as plt

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import  convert_to_block_zone_dict
from Zone_Generation.Config.Constants import AREA_ETHNICITIES


def estimate_boundary_cost(G, m):
    """
    Parameters:
    m (int): Number of zones.

    Returns:
    int: Estimated boundary cost.
    """
    total_nodes = G.number_of_nodes()
    average_per_partition = total_nodes // m

    super_nodes = partition_graph_metis_partial_constraint(G, average_per_partition)
    # super nodes maps partition id to list of nodes
    # create a mapping from node to partition id
    node_to_partition = {}
    for partition_id, nodes in super_nodes.items():
        for node in nodes:
            node_to_partition[node] = partition_id

    boundary_cost = 0
    for u, v in G.edges():
        if node_to_partition[u] != node_to_partition[v]:
            boundary_cost += 1

    fully_isolated = {}
    neighbors_assigned_to_same_partition = []
    total_nodes = {}
    for partition_id in super_nodes.keys():
        fully_isolated[partition_id] = 0
        total_nodes[partition_id] = len(super_nodes[partition_id])
        for node in super_nodes[partition_id]:
            is_isolated = True
            total_neighbors = 0
            same_neighbors = 0
            for neighbor in G.neighbors(node):
                if node_to_partition[neighbor] != partition_id:
                    is_isolated = False
                else:
                    same_neighbors += 1
                total_neighbors += 1
            if is_isolated:
                fully_isolated[partition_id] += 1
            neighbors_assigned_to_same_partition.append(same_neighbors / total_neighbors)
    percentages = [fully_isolated[pid] / total_nodes[pid] for pid in super_nodes.keys()]
    # print('Percent fully isolated per partition: ', percentages)

    zv = ZoneVisualizer('Block', is_local=False)
    block_zone_dict = convert_to_block_zone_dict(node_to_partition, G)
    zv.zones_from_dict(block_zone_dict, show_plot=True)

    # use matplotlib to plot distribution of neighbors assigned to same partition
    plt.hist(neighbors_assigned_to_same_partition, bins=20, edgecolor='black')
    plt.title('Distribution of Neighbors Assigned to Same Partition')
    plt.xlabel('Proportion of Neighbors in Same Partition')
    plt.ylabel('Frequency')
    plt.show()

    print(f"Partitioning resulted in boundary_cost of {boundary_cost} for {m} zones.")
    print('Minimum percent_fully_isolated: ', min(percentages))

    return boundary_cost


def partition_graph_metis(G, k):
    """
    Partitions a NetworkX graph into super-nodes of approximate size k.

    Args:
        G: networkx.Graph
        k: The number of partitions
    """

    # 2. METIS requires nodes to be 0...N-1. Create a mapping.
    nodes = list(G.nodes())

    # handle case when n_partitions > number of nodes
    k = min(k, len(nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 3. Build Adjacency List for PyMetis
    # adjncy[i] contains the neighbors of node i
    adj_list = [
        [node_to_idx[neighbor] for neighbor in G.neighbors(node)]
        for node in nodes
    ]
    options = pymetis.Options()
    options.contig = True
    options.niter = 30
    options.ncuts = 10
    options.seed = 42
    # 4. Perform Partitioning
    # cuts is the number of edges between super-nodes
    # membership is a list where membership[i] is the partition ID of node i
    cuts, membership = pymetis.part_graph(
        k,
        adjacency=adj_list,
        options=options,
    )

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
            k: The number of partitions
        """

    # 2. METIS requires nodes to be 0...N-1. Create a mapping.
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}
    k = min(k, len(nodes))

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
        schools = int(G.nodes[node]['num_schools'] * 100) + 1
        students = int(G.nodes[node]['ge_students'] * 10) + 2

        vweights.extend([schools, students])

    # 4. Perform Partitioning
    # cuts is the number of edges between super-nodes
    # membership is a list where membership[i] is the partition ID of node i
    options = pymetis.Options()
    # options.ufactor = 25  # Set imbalance constraints
    options.niter = 30
    options.ncuts = 10
    # options.ubvec = ubvec
    options.contig = True

    cuts, membership = pymetis.part_graph(
        k,
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
        for part_id in range(k):
            partition_stats[part_id][ethnicity] = 0
    for i, part_id in enumerate(membership):
        node = nodes[i]
        partition_stats[part_id]['schools'] += G.nodes[node]['num_schools']
        partition_stats[part_id]['students'] += G.nodes[node]['ge_students']

    return super_nodes


def partition_graph_metis_constrained(G, k, centroids):
    """
        Partitions a NetworkX graph into super-nodes of approximate size k.

        Args:
            G: networkx.Graph
            k: The number of partitions
            centroids: List of nodes that must be in separate partitions
        """

    # 2. METIS requires nodes to be 0...N-1. Create a mapping.
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}
    k = min(k, len(nodes))

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
        schools = int(G.nodes[node]['num_schools'] * 100) + 1
        students = int(G.nodes[node]['ge_students'] * 10) + 2
        centroid_coef = 1000 if node in centroids else 1

        vweights.extend([schools, students, centroid_coef])

    # 4. Perform Partitioning
    # cuts is the number of edges between super-nodes
    # membership is a list where membership[i] is the partition ID of node i
    options = pymetis.Options()
    # options.ufactor = 25  # Set imbalance constraints
    options.niter = 30
    options.ncuts = 10
    # options.ubvec = ubvec
    options.contig = True

    cuts, membership = pymetis.part_graph(
        k,
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

    # partition_stats = defaultdict(lambda: {'schools': 0, 'students': 0, 'centroids': 0})
    # for ethnicity in AREA_ETHNICITIES:
    #     for part_id in range(k):
    #         partition_stats[part_id][ethnicity] = 0
    # for i, part_id in enumerate(membership):
    #     node = nodes[i]
    #     partition_stats[part_id]['schools'] += G.nodes[node]['num_schools']
    #     partition_stats[part_id]['students'] += G.nodes[node]['ge_students']
    #     if node in centroids:
    #         partition_stats[part_id]['centroids'] += 1
    #
    # print("Partition stats with centroid constraints:")
    # for part_id, stats in partition_stats.items():
    #     print(f"Partition {part_id}: {stats}")


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
