# # Python Program for Floyd Warshall Algorithm
#
#
# # Define infinity as the large
# # enough value. This value will be
# # used for vertices not connected to each other
# # Solves all pair shortest path
# # via Floyd Warshall Algorithm
# def floydWarshall(graph, centroids):
#     INF = 1000000
#     # dist[][] will be the output matrix that will finally have the shortest distances between every pair of vertices
#     #  initializing the solution matrix  same as input graph matrix OR we can say that the initial values of shortest distances
#     # are based on shortest paths considering no intermediate vertices
#     V = len(graph)
#     dist = [[INF] * V for _ in range(V)]
#
#     for i in range(V):
#         for j in range(V):
#             if i in graph[j]:
#                 dist[i][j] = 1
#                 dist[j][i] = 1
#             else:
#                 dist[i][j] = INF
#                 dist[j][i] = INF
#         dist[i][i] = 0
#
#     for k in range(V):
#         # pick all vertices as source one by one
#         for i in range(V):
#             # Pick all vertices as destination for the
#             # above picked source
#             # for j in range(V):
#             for j in centroids:
#                 # If vertex k is on the shortest path from
#                 # i to j, then update the value of dist[i][j]
#                 dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
#     return dist
#
#
# inputGraph = {}
# inputGraph[0] = []
# inputGraph[1] = []
# inputGraph[2] = []
# inputGraph[3] = []
# inputGraph[4] = []
# inputGraph[0].append(1)
# inputGraph[0].append(2)
# inputGraph[1].append(4)
#
# # area2idx = {}
# # area2idx[10] = 1
# # area2idx[20] = 2
# # area2idx[30] = 3
# # area2idx[40] = 4
# # area2idx[0] = 0
#
# floydWarshall(inputGraph, {4})
#

# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
import numpy as np

maxint = 1000000
def Shortest_Path(inputGraph, centroids):
	V = len(inputGraph)
	graph = [[0 for column in range(V)] for row in range(V)]

	for i in range(V):
		for j in range(V):
			if i in inputGraph[j]:
				graph[i][j] = 1
				graph[j][i] = 1
			else:
				graph[i][j] = 0
				graph[j][i] = 0
		graph[i][i] = 0
	# print(graph)

	shorestpaths = {}

	for c in centroids:
		dist = dijkstra(c, graph, V)
		for n in range(V):
			shorestpaths[n,c] = int(dist[n])
	# print("shorestpath")
	# print(shorestpaths)

	return shorestpaths


# A utility function to find the vertex with
# minimum distance value, from the set of vertices
# not yet included in shortest path tree
def minDistance(dist, sptSet, V):
	# Initialize minimum distance for next node
	min = maxint

	# Search not nearest vertex not in the
	# shortest path tree
	for u in range(V):
		if dist[u] < min and sptSet[u] == False:
			min = dist[u]
			min_index = u
	return min_index

# Function that implements Dijkstra's single source
# shortest path algorithm for a graph represented
# using adjacency matrix representation
def dijkstra(src, graph, V):

	dist = [maxint] * V
	dist[src] = 0
	sptSet = [False] * V

	for cout in range(V):

		# Pick the minimum distance vertex from
		# the set of vertices not yet processed.
		# x is always equal to src in first iteration
		x = minDistance(dist, sptSet, V)

		# Put the minimum distance vertex in the
		# shortest path tree
		sptSet[x] = True

		# Update dist value of the adjacent vertices
		# of the picked vertex only if the current
		# distance is greater than new distance and
		# the vertex in not in the shortest path tree
		for y in range(V):
			if graph[x][y] > 0 and sptSet[y] == False and \
			dist[y] > dist[x] + graph[x][y]:
					dist[y] = dist[x] + graph[x][y]

	return dist

# inputGraph = {}
# inputGraph[0] = []
# inputGraph[1] = []
# inputGraph[2] = []
# inputGraph[3] = []
# inputGraph[4] = []
# inputGraph[0].append(1)
# inputGraph[1].append(0)
#
# inputGraph[0].append(2)
# inputGraph[2].append(0)
#
# inputGraph[1].append(3)
# inputGraph[3].append(1)
#
# inputGraph[1].append(4)
# inputGraph[4].append(1)
#
# g = Shortest_Path(inputGraph, [0,1])
