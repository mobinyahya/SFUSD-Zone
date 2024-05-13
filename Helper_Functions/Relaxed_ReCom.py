import random
from Helper_Functions.Graph import Graph
from Helper_Functions.Spanning_Tree import *


class Relaxed_ReCom(object):

    def __init__(self, initial_zones, dz):
        self.dz = dz
        self.area2idx = dz.area2idx
        self.idx2area = dz.idx2area
        self.neighbors = dz.neighbors

        self.base_graph = self.load_base_graph()
        # List of Spanning Trees. Each Spanning Tree represents a zone
        self.ST = self.load_st(initial_zones)


    # load/initializes a graph object with SFUSD map
    def load_base_graph(self):
        base_graph = Graph()
        for idx in self.idx2area:
            base_graph.add_vertex(idx)

            for j in self.neighbors[idx]:
                base_graph.add_edge({idx, j})
        return base_graph


    # load/initializes a set of spanning tree, based on the initial SFUSD zoning map
    def load_st(self, initial_zones):
        ST = []
        # for each zone, find a spanning tree on nodes of that zone, and edges in the base graph
        # and this spanning tree to ST list
        for z in range(len(initial_zones)):
            zone_base_graph = Graph()
            zone_areas = initial_zones[z]
            for area in zone_areas:
                idx = self.area2idx[area]
                zone_base_graph.add_vertex(idx)

                for j in self.neighbors[idx]:
                    if self.idx2area[j] in zone_areas:
                        zone_base_graph.add_edge({idx, j})
            zone_st = Spanning_Tree(zone_base_graph)
            ST.append(zone_st)
        return ST



    # mix two adjacent zones
    def ReCom_step(self, zones):
        # [A,B]: index of the two randomly selected zones that share a boundary
        # e: random edge connecting zones A,B
        [A,B], e = self.adjacent_zones(zones)

        # M: Initialize a new spanning tree, merging A and B
        M = merge_spanning_tree(self.ST[A], self.ST[B], e)

        # Divide spanning tree M, using the up_down_walk
        cutting_edge = self.up_down_walk_cut(M)

        st_a,st_b = cut_spanning_tree(M, cutting_edge)

        # Update spanning trees of the zones A,B from ST
        self.ST[A] = st_a
        self.ST[B] = st_b

        # Update spanning trees of the zones A,B from ST
        zones[A] = [self.idx2area[v] for v in st_a.all_vertices()]
        zones[B] = [self.idx2area[v] for v in st_b.all_vertices()]

        return zones

    # Cut a spanning tree using the up_down_walk probabilistic approach
    # Evaluate the probability of each zonings after dropping each possible cutting edge,
    # and select a cutting edge according to the given distribution.
    def up_down_walk_cut(self, M):
        # input M: a Merged Spanning Tree

        # For each edge e, what is the probability of the resulting
        # zoning if we cut the spanning tree from edge e
        edge_prob = {}
        for edge in M.all_edges():
            M.remove_edge(edge)
            new_zones = M.connected_components()
            if len(new_zones) != 2:
                raise Exception("After cutting the spanning tree, we received more/less than 2 connected components")
            # zone_A = [self.idx2area[v] for v in new_zones[0]]
            # zone_B = [self.idx2area[v] for v in new_zones[1]]
            zone_A = new_zones[0]
            zone_B = new_zones[1]
            balance_score = self.evaluate_zones(zone_A, zone_B)
            edge_prob[frozenset(edge)] = balance_score

            M.add_edge(edge)

        cutting_edge = self.random_selection(edge_prob)
        return cutting_edge

    # Randomly select an edge from edge_prob dictionary, such that
    # each edge is selected with its corresponding probability
    def random_selection(self, edge_prob):

        # Extract edges and their corresponding probabilities
        edges = list(edge_prob.keys())
        probs = list(edge_prob.values())

        # Normalize the probabilities (just to ensure they sum to 1)
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]

        # Sample an edge based on the probabilities
        selected_edge = set(random.choices(edges, weights=probs, k=1)[0])

        return selected_edge


    def evaluate_zones(self, zone_A, zone_B):
        # metrics: for each zone A,B, compute [Number of tree nodes,
        # Total FRL count, Total students, etc and evaluate the zone by
        #  multiplying these metrics, (using fixed exp power on each)

        weight = {}
        weight["nodes"] = 1
        weight["frl"] = 3
        weight["students"] = 1
        weight["seats"] = 1
        weight["shortage%"] = 10
        weight["sch_count"] = 45

        zone_A_metric = self.compute_metrics(zone_A)
        zone_B_metric = self.compute_metrics(zone_B)

        evaluation_score = 1
        for metric in weight:
            # if metric == "shortage%":
            #     print("metric:   " + str(metric))
            #     print(zone_A_metric[metric] * zone_B_metric[metric])
            evaluation_score *= (zone_A_metric[metric] * zone_B_metric[metric]) ** weight[metric]

        return evaluation_score


    def compute_metrics(self, zone):
        zone_metric = {}
        zone_metric["nodes"] = len(zone)
        zone_metric["frl"] = sum([self.dz.area_data["FRL"][j] for j in zone])
        zone_metric["students"] = sum([self.dz.studentsInArea[j] for j in zone])
        zone_metric["seats"] = sum([self.dz.seats[j] for j in zone])
        zone_metric["shortage%"] = max((zone_metric["students"] - zone_metric["seats"]), 1)
        zone_metric["sch_count"] = sum([self.dz.schools[j] for j in zone])

        return zone_metric


    def adjacent_zones(self, zones):
        # randomly select two zones with a shared boundary more than a fixed amount
        while (True):
            # randomly select two zones
            shuffle_zones = random.sample(range(len(zones)), 2)
            # shuffle_zones = [0, 2]  # TODO

            # count total shared boundary cost
            # if selected zones have less than a fixed shared boundary, reselect two zones
            boundary_edges = []
            # iterate over every area (blockgroup) assigned to
            # the two selected zones( shuffle_zones[0] and shuffle_zones[1])
            for i in zones[shuffle_zones[0]]:
                for j in zones[shuffle_zones[1]]:
                    # if i and j are neighbors, it is considered as a boundary
                    idx_i = self.area2idx[i]
                    idx_j = self.area2idx[j]
                    if idx_j in self.base_graph._graph_dict[idx_i]:
                        boundary_edges.append({idx_i, idx_j})


            if len(boundary_edges) > 10:
                print("shuffle_zones are: " + str(shuffle_zones) + " and shared boundary is: " + str(boundary_edges))
                # connecting_edge = random.choice(boundary_edges)
                connecting_edge = boundary_edges[0] #TODO
                break
            # otherwise go back and reselect two new zones.

        return shuffle_zones, connecting_edge







