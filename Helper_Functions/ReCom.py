import random
from Zone_Generation.Optimzation_Heuristics.zone_eval import * # evaluate_assignment_score, Tuning_param, boundary_trimming, evaluate_contiguity
# from Helper_Functions import *


def random_spanning_tree_with_root(root, geo_graph):
    # Sample a random spanning tree on the union of the two zones.
    spanning_tree = Graph()

    spanning_tree.add_vertex(root)

    Next = {}
    for vertex in geo_graph.all_vertices():
        # print("initial vertex number: " + str(vertex))
        covered_vertices = spanning_tree.all_vertices()

        pointer = vertex
        while pointer not in covered_vertices:
            random_walk = random.choice(geo_graph.edges(pointer))
            Next[pointer] = random_walk
            pointer = Next[pointer]

        pointer = vertex
        while pointer not in covered_vertices:
            spanning_tree.add_vertex(pointer)
            spanning_tree.add_edge({pointer, Next[pointer]})
            pointer = Next[pointer]
    return spanning_tree

def random_spanning_tree(geo_graph):
    root = random.choice(list(geo_graph.all_vertices()))
    print("root " + str(root))

    spanning_tree = random_spanning_tree_with_root(root, geo_graph)

    if len(spanning_tree.all_vertices()) - len(spanning_tree.all_edges()) != 1:
        print("Error, our spanning tree does not have a tree fomation")
        print("number of spanning tree vertices: " + str(len(spanning_tree.all_vertices())))
        print("number of spanning tree edges: " + str(len(spanning_tree.all_edges())))


    # print("set of spanning tree edges are " + str(spanning_tree.all_edges()))
    spanning_tree.degree_stats()

    return spanning_tree

def adjacent_shuffle_zones(zones, area2idx, neighbors):
    # randomly select two zones with a shared boundary more than a fixed amount
    while (True):
        # randomly select two zones
        shuffle_zones = random.sample(range(len(zones)), 2)
        # shuffle_zones = [1, 2]  # TODO

        # count total shared boundary cost
        # if selected zones have less than a fixed shared boundary, reselect two zones
        shared_boundary = 0
        # iterate over every area (blockgroup) assigned to
        # the two selected zones( shuffle_zones[0] and shuffle_zones[1])
        for i in zones[shuffle_zones[0]]:
            for j in zones[shuffle_zones[1]]:
                # if i and j are neighbors, it is considered as a boundary
                if area2idx[j] in neighbors[area2idx[i]]:
                    shared_boundary += 1

        print("shuffle_zones are: " + str(shuffle_zones) + " and shared boundary is: " + str(shared_boundary))

        if shared_boundary > 10:
            break
        # otherwise go back and reselect two new zones.


    return shuffle_zones

def construct_shuffle_geograph(zones, area2idx, idx2area, neighbors):
    shuffle_zones = adjacent_shuffle_zones(zones, area2idx, neighbors)

    # now that the two candidate zones are finalized, start redrawing boundaries.
    # construct a graph with adjacency matrix dictionary on the two zones.
    geo_graph = Graph()

    shuffle_areas = zones[shuffle_zones[0]] + zones[shuffle_zones[1]]
    for area in shuffle_areas:
        idx = area2idx[area]
        geo_graph.add_vertex(idx)

        for j in neighbors[idx]:
            if idx2area[j] in shuffle_areas:
                geo_graph.add_edge({idx, j})

    return geo_graph, shuffle_zones


def random_cut(unacceptable_edge_cuts, spanning_tree):
    valid_edge_cuts = [edge for edge in spanning_tree.all_edges() if edge not in unacceptable_edge_cuts]

    # randomly select an edge from the spanning tree
    cut_edge = random.choice(valid_edge_cuts)
    spanning_tree.remove_edge(cut_edge)

    connected_components = spanning_tree.connected_components()

    # print("the random cut_edge is: " + str(cut_edge))
    # print("spanning_tree.connected_components " + str(connected_components))
    # print("spanning_tree 0 connected_components" + str(connected_components[0]))
    # print("spanning_tree 1 connected_components" + str(connected_components[1]))
    # print("len of 0 connected component " + str(len(connected_components[0])))
    # print("len of 1 connected component " + str(len(connected_components[1])))

    spanning_tree.add_edge(cut_edge)

    return connected_components, cut_edge



def random_cut_evaluation(dz, spanning_tree, shuffle_zones, param=None):
    # set of all cutting edges that we tried and
    # the resulting graph cut was not balanced
    unacceptable_edge_cuts = []

    zone_dict_backup = dict(dz.zone_dict).copy()

    while (len(unacceptable_edge_cuts) < len(spanning_tree.all_edges())):
        connected_components, cut_edge = random_cut(unacceptable_edge_cuts, spanning_tree)

        # Evaluate the two connected components;
        # If they satisfy the balance constraints, break
        # else, add the edge back and pick another random cutting edge

        for j in range(len(connected_components)):
            for idx in connected_components[j]:
                dz.zone_dict[dz.idx2area[idx]] = shuffle_zones[j]

        if param == None:
            population_ratio = len(connected_components[0]) / (len(connected_components[0]) + len(connected_components[1]))
            if (population_ratio > 0.55) | (population_ratio < 0.45):
                evaluation_cost = 1000000000
            else:
                evaluation_cost = 0

        else:
            evaluation_cost = evaluate_assignment_score(param, dz, dz.zone_dict)

        if evaluation_cost < 1000000000:
            print("evaluation_cost " + str(evaluation_cost))
            return evaluation_cost

        unacceptable_edge_cuts.append(cut_edge)

    dz.zone_dict = zone_dict_backup
    return evaluation_cost
    #########################