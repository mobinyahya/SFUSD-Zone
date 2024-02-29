import random

class Graph(object):
    def __init__(self):
        # initializes a graph object
        # an empty dictionary will be used

        self._graph_dict = {}

    def edges(self, vertice):
        # returns a list of all the edges of a vertice
        return self._graph_dict[vertice]

    def all_vertices(self):
        # returns the vertices of a graph as a set
        return set(self._graph_dict.keys())

    def all_edges(self):
        # returns the edges of a graph
        return self.__generate_edges()

    def degree_stats(self):
        degree_one_vertices = 0
        for vertex in self._graph_dict:
            if len(self._graph_dict[vertex]) == 1:
                degree_one_vertices += 1
        print("Number of degree-one vertices is: " + str(degree_one_vertices))
        print("Number total vertices is: " + str(len(self.all_vertices())))

    def __generate_edges(self):
        # A static method generating the edges of the
        # graph "graph". Edges are represented as sets
        # with one (a loop back to the vertex) or two
        # vertices

        edges = []
        for vertex in self._graph_dict:
            for neighbour in self._graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def add_vertex(self, vertex):
        # If the vertex "vertex" is not in
        # self._graph_dict, a key "vertex" with an empty
        # list as a value is added to the dictionary.
        # Otherwise nothing has to be done.

        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        # assumes that edge is of type set, tuple or list;
        #  between two vertices can be multiple edges!

        # add the new edge to graph dict structure
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
            if x in self._graph_dict:
                if y not in self._graph_dict[x]:
                    self._graph_dict[x].append(y)
            else:
                self._graph_dict[x] = [y]

    def remove_node(self, node):
        if node in self._graph_dict:
            # Remove any references to this node in other nodes' adjacency lists
            for key in self._graph_dict[node]:
                self._graph_dict[key].remove(node)

            # Remove the node itself
            del self._graph_dict[node]
        else:
            print("Error: we are trying to delete a node that does not exist")

    def remove_edge(self, edge):
        # assumes that edge is of type set, tuple or list;
        #  between two vertices can be multiple edges!

        # delete the given edge from graph dict structure
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
            if x in self._graph_dict:
                self._graph_dict[x].remove(y)
            else:
                print("Error: we are trying to delete an edge, that one of its endpoint do not exist")

    def BFS(self, s, visited):
        # list of elements that will be visited from
        # starting vertex s, in the BFS order
        component = []

        # Create a queue for BFS
        queue = []

        # Add the source node to visited list and enqueue it
        queue.append(s)
        visited.append(s)

        while queue:
            # Dequeue a vertex from queue and add it to the component list
            s = queue.pop(0)
            component.append(s)

            # Get all adjacent vertices of the dequeued vertex s. If a adjacent
            # has not been visited, then mark it visited and enqueue it
            for i in self._graph_dict[s]:
                if i not in visited:
                    queue.append(i)
                    visited.append(i)
        return visited, component

    def connected_components(self):
        cc = []
        visited = []
        for vertex in self.all_vertices():
            if vertex not in visited:
                visited, component = self.BFS(vertex, visited)
                cc.append(component)
        return cc

