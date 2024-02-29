from Helper_Functions.Graph import Graph
import random



class Spanning_Tree(Graph):

    # initializes a graph object
    def __init__(self, base_graph):
        self._graph_dict = {}
        self.random_spanning_tree(base_graph)


    def random_spanning_tree(self, base_graph):
        if len(base_graph.all_vertices()) == 0:
            return
        root = random.choice(list(base_graph.all_vertices()))

        self.add_vertex(root)

        Next = {}
        for vertex in base_graph.all_vertices():
            # print("initial vertex number: " + str(vertex))
            covered_vertices = self.all_vertices()

            pointer = vertex
            while pointer not in covered_vertices:
                random_walk = random.choice(base_graph.edges(pointer))
                Next[pointer] = random_walk
                pointer = Next[pointer]

            pointer = vertex
            while pointer not in covered_vertices:
                self.add_vertex(pointer)
                self.add_edge({pointer, Next[pointer]})
                pointer = Next[pointer]


# Input:
# st1: Spanning Tree 1
# st2: Spanning Tree 2
# connecting_edge: edge connecting st1 to st2
def merge_spanning_tree(st1, st2, connecting_edge):
    merged_graph = Graph()

    merged_graph.add_edge(connecting_edge)
    for edge in st1.all_edges() + st2.all_edges():
        merged_graph.add_edge(edge)

    # merged_st: a spanning tree resulting from merging and connecting st1 to st2 through e
    merged_st = Spanning_Tree(merged_graph)
    return merged_st


# cut Spanning Tree st from edge e into two spanning trees, st1 and st2
def cut_spanning_tree(st, edge):
    edge = set(edge)
    vertex1, vertex2 = tuple(edge)


def cut_spanning_tree(st, edge):
    # Convert edge to set and unpack its vertices
    edge = set(edge)
    vertex1, vertex2 = tuple(edge)

    # Remove the edge from the spanning tree
    st.remove_edge(edge)

    # Find connected components of the tree after the cut
    connected_components = st.connected_components()

    # Create two new spanning trees for the two components
    st1 = Spanning_Tree(Graph())  # Initialize with empty Graph
    st2 = Spanning_Tree(Graph())  # Initialize with empty Graph

    # Identify which component each vertex of the edge belongs to
    for component in connected_components:
        if vertex1 in component:
            for vertex in component:
                st1.add_vertex(vertex)
            for v1 in component:
                for v2 in st._graph_dict[v1]:  # Using original st for adjacency information
                    st1.add_edge({v1, v2})
            if len(st1.all_vertices()) <= len(st1.all_edges()):
                raise Exception("The Spanning Tree is Not a Tree.")


        if vertex2 in component:
            for vertex in component:
                st2.add_vertex(vertex)
            for v1 in component:
                for v2 in st._graph_dict[v1]:  # Using original st for adjacency information
                    st2.add_edge({v1, v2})
            if len(st2.all_vertices()) <= len(st2.all_edges()):
                raise Exception("The Spanning Tree is Not a Tree.")

    return st1, st2





