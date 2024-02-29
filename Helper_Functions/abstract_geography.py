from matplotlib import pyplot
import matplotlib as colors

from Helper_Functions.ReCom import *

class Geo(object):
    n = 0
    M = 0
    area2idx = {}
    idx2area = {}
    neighbors = {}
    zd = {}
    z = []
    # initialize an abstract geography that looks like a rectangle
    def __init__(self, n=0, m=0):
        self.n = n
        self.M = m
        self.z = [[] for i in range(self.M)]

        for i in range(self.n * self.n):
            self.area2idx[i] = i
            self.idx2area[i] = i

            self.neighbors[i] = []
            if (i % self.n) > 0:
                self.neighbors[i].append(i - 1)
            if (i % self.n) < (self.n - 1):
                self.neighbors[i].append(i + 1)
            if (i - self.n) >= 0:
                self.neighbors[i].append(i - self.n)
            if (i + self.n) < (self.n * self.n):
                self.neighbors[i].append(i + self.n)

            zone_number = int(i / (self.n * self.n / self.M))
            self.zd[i] = zone_number

            self.z[zone_number].append(i)

        print("statistics on the abstract geography")
        print("zones:      " + str(self.z))
        print("zone_dict:  " + str(self.zd))
        print("neighbors: " + str(self.neighbors))

        # visualize the constructed abstract geography
        self.visualize_array_map(-1)


    # visualize the array geography
    def visualize_array_map(self, round):
        vis_array = [[0] * self.n for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                vis_array[i][j] = self.zd[j * self.n + i]

        print("vis_array in round: " + str(round))
        for row in vis_array:
            print(row)
        pyplot.figure(figsize=(5, 5))
        pyplot.imshow(vis_array, cmap='tab10')
        # pyplot.imshow(vis_array, cmap= colors.ListedColormap(['blue', 'black', 'red']))
        pyplot.colorbar()

        pyplot.show()


def cut_evaluation(G, spanning_tree, shuffle_zones):
    # set of all cutting edges that we tried and
    # the resulting graph cut was not balanced
    unacceptable_edge_cuts = []

    while (len(unacceptable_edge_cuts) < len(spanning_tree.all_edges())):
        connected_components, cut_edge = random_cut(unacceptable_edge_cuts, spanning_tree)


        # Evaluate the two connected components;
        # If they satisfy the balance constraints, break
        # else, add the edge back and pick another random cutting edge

        population_ratio = len(connected_components[0]) / (len(connected_components[0]) + len(connected_components[1]))
        if (population_ratio > 0.55) | (population_ratio < 0.45):
            unacceptable_edge_cuts.append(cut_edge)
            print("* Evaluation_cost is high and " + str(len(unacceptable_edge_cuts)) + " unacceptable_edge_cuts selected")

        else:
            for j in range(len(connected_components)):
                for idx in connected_components[j]:
                    G.zone_dict[G.idx2area[idx]] = shuffle_zones[j]

            return 0, G
    return 100000, G
    #########################

def update_z(G, zone_dict):
    G.zone_lists = []
    for z in range(G.M):
        G.zone_lists.append([])
    for b in zone_dict:
        G.zone_lists[zone_dict[b]].append(b)
    return G

# iteratively update the zoning using spanning tree graph cut.
# in each iteration:
def spanning_tree_cut(G):
    geograph, shuffle_zones = construct_shuffle_geograph(zones=G.zone_lists, area2idx=G.area2idx, idx2area=G.idx2area, neighbors=G.neighbors)
    for j in range(10):
        print('**** Redrawing spanning trees, Round ' + str(j))

        spanning_tree = random_spanning_tree(geograph)
        print("spanning tree is drawn")

        evaluation_cost, G = cut_evaluation(G, spanning_tree, shuffle_zones)

        if evaluation_cost < 10000:
            return G

    return G

if __name__ == "__main__":

    G = Geo(n=21, m=7)
    for i in range(40):
        print('********** Mixing neighboring zones, Round ' + str(i))
        G = spanning_tree_cut(G)
        G = update_z(G, G.zd)
        G.visualize_array_map(i)

