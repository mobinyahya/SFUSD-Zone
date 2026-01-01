import os
import pickle
import subprocess
import tempfile
from collections import defaultdict
from typing import Union, Iterable

import gerrychain
import kahip

import pymetis
import yaml
from matplotlib import pyplot as plt

from Graphic_Visualization.zone_viz import ZoneVisualizer
from Helper_Functions.util import convert_to_block_zone_dict, compute_zone_deviations
from Zone_Generation.Config.Constants import AREA_ETHNICITIES, get_dropbox_path
from redistricting.redistricting import Redistricting


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


def partition_graph_kahip(G, k):
    kahip_graph = kahip.kahip_graph()
    kahip_graph.set_num_nodes(G.number_of_nodes())
    for i, (edge_start, edge_end) in enumerate(G.edges()):
        kahip_graph.add_undirected_edge(edge_start, edge_end)

    # set node weights based on number of students
    for node in G.nodes():
        num_students = int(10 * G.nodes[node]['ge_students'])
        num_frl = int(10 * G.nodes[node]['FRL'])

        kahip_graph.set_weight(node, num_students)

    # Get CSR arrays

    vwgt, xadj, adjcwgt, adjncy = kahip_graph.get_csr_arrays()
    edgecut, membership = kahip.kaffpa(vwgt,
                                       xadj,
                                       adjcwgt,
                                       adjncy,
                                       k,
                                       0.2,
                                       False,
                                       42,
                                       int(kahip.STRONGSOCIAL))
    print(kahip.ilp_improve())

    super_nodes = {}
    for node_idx, partition_id in enumerate(membership):
        if partition_id not in super_nodes:
            super_nodes[partition_id] = []
        super_nodes[partition_id].append(node_idx)

    return super_nodes


def write_graph_to_metis_file(G, filename, node_weight_attr=None, edge_weight_attr=None, node_size_attr=None):
    """
    Writes a NetworkX graph to a file in METIS format.

    Args:
        G: networkx.Graph
        filename: str, path to the output file
    """
    if G.is_directed():
        raise ValueError("METIS format is only for undirected graphs.")

        # METIS requires vertices to be 1 to n.
        # We create a mapping to ensure the file is consistent regardless of input labels.
    nodes = list(G.nodes())
    node_map = {node: i + 1 for i, node in enumerate(nodes)}

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Determine fmt (3-bit binary converted to decimal)
    # bit 1: edge weights, bit 2: node weights, bit 3: node sizes
    has_edge_w = 1 if edge_weight_attr else 0
    has_node_w = 1 if node_weight_attr else 0
    has_node_s = 1 if node_size_attr else 0

    fmt_val = (has_node_s * 100) + (has_node_w * 10) + (has_edge_w * 1)
    # Convert binary-lookalike integer to string and pad with zeros
    fmt = f"{fmt_val:03}" if fmt_val > 0 else None

    # Handle multiple constraints (ncon)
    ncon = None
    if isinstance(node_weight_attr, list):
        ncon = len(node_weight_attr)
    elif node_weight_attr and not isinstance(node_weight_attr, list):
        ncon = 1

    with open(filename, 'w') as f:
        # Write Header: n m [fmt] [ncon]
        header = [str(n), str(m)]
        if fmt:
            header.append(fmt)
        if ncon and ncon > 1:
            header.append(str(ncon))
        f.write(" ".join(header) + "\n")

        # Write Node Lines
        for u in nodes:
            line = []

            # 1. Vertex Size (s)
            if has_node_s:
                line.append(str(int(G.nodes[u].get(node_size_attr, 1))))

            # 2. Vertex Weights (w1, w2... wncon)
            if has_node_w:
                if isinstance(node_weight_attr, list):
                    for attr in node_weight_attr:
                        line.append(str(int(G.nodes[u].get(attr, 1))))
                else:
                    line.append(str(int(G.nodes[u].get(node_weight_attr, 1))))

            # 3. Adjacency Info (v1 e1 v2 e2...)
            for v in G.neighbors(u):
                # Target vertex (1-indexed)
                line.append(str(node_map[v]))
                # Edge weight (if applicable)
                if has_edge_w:
                    weight = G[u][v].get(edge_weight_attr, 1)
                    line.append(str(int(weight)))

            f.write(" ".join(line) + "\n")


def parse_metis_output(filename):
    """
    Parses a METIS partitioning output file.

    Args:
        filename: str, path to the METIS output file

    Returns:
        dict: Mapping from partition ID to list of original node IDs
    """
    partition_map = {}
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            partition_id = int(line.strip())
            if partition_id not in partition_map:
                partition_map[partition_id] = []
            # Original node IDs are 1-indexed in METIS output
            partition_map[partition_id].append(idx + 1)
    return partition_map


def partitions_to_zone_dict(partition_map, node_list):
    """
    Converts a partition mapping to a zone dictionary.

    Args:
        partition_map: dict, mapping from partition ID to list of original node IDs
        node_list: list, original list of node IDs in the graph

    Returns:
        dict: Mapping from original node ID to partition ID
    """
    zone_dict = {}
    for partition_id, nodes in partition_map.items():
        for node_idx in nodes:
            original_node = node_list[node_idx - 1]  # Convert back to 0-indexed
            zone_dict[original_node] = partition_id
    return zone_dict


def gp_metis(G,
             n_parts=2,
             node_weight_attr: Union[None, str, Iterable[str]] = None,
             edge_weight_attr=None,
             ptype=None,  # rb or kway
             ctype=None,  # rm or shem
             iptype=None,  # grow or random
             objtype=None,  # cut or vol
             no2hop=False,  # disable 2-hop matching
             contig=False,  # force contiguous partitions
             minconn=False,  # minimize subdomain connectivity
             tpwgts=None,  # path to target partition weights file
             ufactor=None,  # max allowed load imbalance
             ubvec: list[float] = None,  # per-constraint load imbalance
             niter=None,  # refinement iterations
             ncuts=None,  # number of different partitionings
             seed=None,  # random seed
             dbglvl=None,  # debugging information level
             path='~/local/bin/gpmetis'
             ):
    # Initialize files as None for the finally block
    metis_file_path = None
    output_file = None

    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(delete=False) as metis_file:
            metis_file_path = metis_file.name
            write_graph_to_metis_file(G, metis_file_path, node_weight_attr, edge_weight_attr)

        output_file = f"{metis_file_path}.part.{n_parts}"

        # Construct command-line arguments [cite: 361]
        cmd = [os.path.expanduser(path)]

        # Add optional parameters if provided
        if ptype: cmd.append(f"-ptype={ptype}")
        if ctype: cmd.append(f"-ctype={ctype}")
        if iptype: cmd.append(f"-iptype={iptype}")
        if objtype: cmd.append(f"-objtype={objtype}")
        if no2hop: cmd.append("-no2hop")
        if contig: cmd.append("-contig")
        if minconn: cmd.append("-minconn")
        if tpwgts: cmd.append(f"-tpwgts={tpwgts}")
        if ufactor is not None: cmd.append(f"-ufactor={ufactor}")
        if ubvec:
            # convert ubvec to string if it's a list
            ubvec = " ".join([str(v) for v in ubvec])
            cmd.append(f"-ubvec={ubvec}")
        if niter is not None: cmd.append(f"-niter={niter}")
        if ncuts is not None: cmd.append(f"-ncuts={ncuts}")
        if seed is not None: cmd.append(f"-seed={seed}")
        if dbglvl is not None: cmd.append(f"-dbglvl={dbglvl}")

        # Final required arguments
        cmd.extend([metis_file_path, str(n_parts)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"gpmetis failed: {result.stderr}")

        partition_map = parse_metis_output(output_file)
        return partition_map

    finally:
        # Clean up temporary files
        if metis_file_path and os.path.exists(metis_file_path):
            os.remove(metis_file_path)
        if output_file and os.path.exists(output_file):
            os.remove(output_file)


def spec_recom(G, centroids):
    k = len(centroids)
    # assigne nodes using voronoi partitioning
    initial_assignment = {}
    for i in range(G.number_of_nodes()):
        closest_centroid = None
        closest_distance = float('inf')
        for z in range(k):
            centroid_z = centroids[z]
            dist = G.graph['distance_dict'][centroid_z][i]
            if dist < closest_distance:
                closest_distance = dist
                closest_centroid = z
        initial_assignment[i] = closest_centroid
    for node in G.nodes():
        G.nodes[node]['population'] = len(G.nodes[node]['school_ids'])

    # turn networkx graph into gerrychain graph
    G = gerrychain.Graph.from_networkx(G)

    r = Redistricting(
        graph=G,  # graph to redistrict
        k=k,  # number of districts
        assignment=initial_assignment,  # initial district assignment
        proposal="recom",  # proposal function for individual redistricting steps
        steps=5,  # number of steps to run the chain for
        step_updaters=["cut edge count"],  # statistics to collect after each step
        single_updaters=["population deviation"],  # statistics to collect at the start and at the end
        population_key="population",  # key under which each node's population is stored
        graph_name="56x56 grid graph",  # human-readable name of the graph
        assignment_name="horizontal stripes"  # human-readable name of the assignment
    )

    r.run(
        plot_interval=500,  # how often to plot the graph (in intervals of steps)
        interactive_level="progress",  # how much information should be printed to the console
        output_level="minimal",  # how much information should be included in the output file
        output_parent="./output/",  # the directory to write output files and images to
        description="grid graph run",  # human-readable description of this run
        checkpoint_interval=0,  # how often to checkpoint the run (in intervals of steps)
        # checkpoint_dest=checkpoint_file,  # the file to save checkpoints to
        keep_final_step=False  # maintain the final step's partition data in the checkpoint file
    )

    #load the assignment
    return r.assignment


def partition_graph_metis_constrained(G, k):
    """
        Partitions a NetworkX graph into super-nodes with centroid constraints.

        Args:
            G: networkx.Graph
        """

    # add constraint vector to all nodes
    for node in G.nodes():
        G.nodes[node]['num_frl'] = int(10 * G.nodes[node]['FRL'])
        G.nodes[node]['num_students'] = int(10 * G.nodes[node]['ge_students'])
        G.nodes[node]['num_schools'] = len(G.nodes[node]['school_ids'])
        G.nodes[node]['area_weight'] = 1
        # for ethnicity in AREA_ETHNICITIES:
        #     G.nodes[node][f'num_{ethnicity}'] = int(100 * G.nodes[node][ethnicity]) + 1

    avg_number_of_schools = sum(len(G.nodes[node]['school_ids']) for node in G.nodes()) / k
    print(f"Average number of schools: {avg_number_of_schools}")
    school_tol = 1 / avg_number_of_schools
    print(f"School tolerance set to: {school_tol}")
    print(f'School UB', ((1 + school_tol) * avg_number_of_schools))
    print(f'School LB', ((1 - school_tol) * avg_number_of_schools))

    ubvec = [4, 1.4, 1.4, 1.15]
    node_weight_attr = ['area_weight', 'num_frl', 'num_students', 'num_schools']
    # for ethnicity in AREA_ETHNICITIES:
    #     ubvec.append(1.2)
    #     node_weight_attr.append(f'num_{ethnicity}')
    partition_map = gp_metis(
        G,
        n_parts=k,
        node_weight_attr=node_weight_attr,
        contig=True,
        seed=42,
        niter=1000,
        ncuts=10,
        minconn=True,
        objtype='cut',
        ubvec=ubvec
    )

    return partition_map


if __name__ == "__main__":
    is_local = False
    output_folder = f'{get_dropbox_path(is_local)}/Optimization/Zones/Graphs'

    with open(f'{output_folder}/Block_0.pickle', 'rb') as f:
        G = pickle.load(f)

    with open('../Config/config.yaml', "r") as f:
        config = yaml.safe_load(f)

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

    zone_dict = spec_recom(G, centroids)
    zv = ZoneVisualizer('Block', is_local)
    zv.zones_from_dict(convert_to_block_zone_dict(zone_dict, G), show_plot=True)

    # find the number of centroids and school per partition
    partitions = {}
    for node, partition_id in zone_dict.items():
        if partition_id not in partitions:
            partitions[partition_id] = []
        partitions[partition_id].append(node)

    partition_stats = defaultdict(lambda: {'schools': 0})
    for partition_id in partitions.keys():
        partition_stats[partition_id]['schools'] = 0
    for node, partition_id in zone_dict.items():
        partition_stats[partition_id]['schools'] += len(G.nodes[node]['school_ids'])

    print("Partition stats with centroid constraints:")
    for part_id, stats in partition_stats.items():
        print(f"Partition {part_id}: {stats}")

    print(compute_zone_deviations(G, zone_dict))
