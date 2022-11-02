import math
from collections import defaultdict
from zone_opt.Tuning_param import Tuning_param


def evaluate_school_quality_stats(dz, zone_dict):
    scores = dz.area_data['AvgColorIndex'].fillna(value=0)
    sch_qlty_stats = [0] * dz.M

    for z in range(dz.M):
        zone_sum = sum([scores[dz.area2idx[b]] * dz.schools[dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)&(b in dz.area2idx)])
        # the following district_average is a weighted average. If a zone has more schools, their sum of qualities will of course be more
        zone_sch = sum([dz.schools[dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)&(b in dz.area2idx)])
        sch_qlty_stats[z] =  zone_sum / zone_sch
    return sch_qlty_stats


def evaluate_metric_stats(dz, zone_dict, metric):
    metric_stats = [0] * dz.M
    # metric_ratio = sum(dz.area_data[metric].fillna(0)) / float(dz.N)
    for z in range(dz.M):
        zone_stud = sum([dz.studentsInArea[dz.area2idx[b]] for b in zone_dict
                               if (zone_dict[b] == z)&(b in dz.area2idx)])
        metric_stats[z] = sum([dz.area_data[metric][dz.area2idx[b]] for b in zone_dict
                               if (zone_dict[b] == z)&(b in dz.area2idx)]) / zone_stud
    return metric_stats

def evaluate_proximity_stats(dz, zone_dict):
    max_sch_dist = [0] * dz.M
    proximate_choices = [0] * dz.M
    sch_list = {}

    for z in range(dz.M):
        sch_list[z] = [b for b in zone_dict
                       if ((zone_dict[b] == z)
                           & (b in dz.area2idx)
                           & (dz.schools[dz.area2idx[b]] == 1))]

        print(sch_list[z])
        proximate_sch_count = 0

        for b in zone_dict:
            if (zone_dict[b] == z)&(b in dz.area2idx):
                # For blockgroup b, find closest GE school in the same zone
                min_sch_dist = 100
                for school in sch_list[z]:
                    dist = dz.euc_distances.loc[school, str(b)]
                    min_sch_dist = min(min_sch_dist, dist)

                    # For each student in blockgroup b, we have found a school within 2mile radius.
                    if dist <= 2:
                        proximate_sch_count += dz.studentsInArea[dz.area2idx[b]]

                max_sch_dist[z] = max(max_sch_dist[z], min_sch_dist)

        zone_stud = sum([dz.studentsInArea[dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)&(b in dz.area2idx)])
        proximate_choices[z] = proximate_sch_count / zone_stud

    return max_sch_dist, proximate_choices


def stats_evaluation(dz, zone_dict):
    stats = {}
    stats["K8 School Count"] = [0] * dz.M
    # print(dz.area_data.columns)
    for z in range(dz.M):
        for b in zone_dict:
            if (zone_dict[b] == z):
                if (b in dz.area2idx):
                    for idx, row in dz.sc_df.iterrows():
                        if (row["K-8"]) == 1:
                            if row[dz.level] == b:
                                stats["K8 School Count"][z] += 1
    # report diversity
    # evaluation_metrics = ['frl', 'sped_count', 'ell_count'] + dz.ethnicity_cols
    # for metric in evaluation_metrics:
    #     stats[metric] = evaluate_metric_stats(dz, zone_dict, metric)
    # # report Access
    # stats['AvgColorIndex'] = evaluate_school_quality_stats(dz, zone_dict)
    # # report Proximity
    # # stats['Max Distance to GE'], stats['Proximate Choices'] = evaluate_proximity_stats(dz, zone_dict)
    print(stats)

    print(sum(dz.sc_df["K-8"]))


    return stats

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------




# This class represents a directed graph
# using adjacency list representation
class Graph:
    # Constructor
    def __init__(self, graph_size):
        # default dictionary to store graph
        self.graph = defaultdict(list)
        self.size = graph_size

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # Function to print a BFS of graph
    def BFS(self, s):
        # Mark all the vertices as not visited
        # visited = [False] * (max(self.graph) + 1)
        visited = [False] * (self.size)
        visited_count = 0
        # Create a queue for BFS
        queue = []

        # Mark the source node as
        # visited and enqueue it
        queue.append(s)
        visited[s] = True

        while queue:
            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            visited_count += 1

            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        del queue
        return visited

def evaluate_discontiguity_cost(dz, zone_dict):
    # initialize discontiguity cost to zero
    distance_weight = {}
    dc_cost = 0
    for z in range(dz.M):
        # form a graph g on areas assigned to zone z
        # there is an edge from area i to area j, if i is a neighbor of j that is closer to the centroid
        # and assigned to the same zone (~ i is the connecting point of contact for j)
        g = Graph(dz.A)
        zone_areas = [x for x in zone_dict if zone_dict[x] == z]
        c_idx = dz.centroids[z]
        for area in zone_areas:
            closer_neighbors = dz.closer_euc_neighbors[dz.area2idx[area], c_idx]
            for neighbor_idx in closer_neighbors:
                if zone_dict[dz.idx2area[neighbor_idx]] == z:
                    g.addEdge(neighbor_idx, dz.area2idx[area])
        # bfs on G starting from zone centroid
        # visited_count = g.BFS(c_idx)
        visited = g.BFS(c_idx)
        visited_count = len([x for x in visited if x == True])
        counter = 0
        for area in zone_areas:
            if visited[dz.area2idx[area]] == False:
                distance_weight[area] = 100000
            else:
                counter+=1
                distance_weight[area] = 1
        # compute size of visited nodes
        dc_cost += (len(zone_areas) - visited_count)
        del visited
        del g

    return dc_cost, distance_weight

def evaluate_school_quality(dz, zone_dict, min_pct):
    y_school_balance = 1
    scores = dz.guardrails['AvgColorIndex'].fillna(value=0)

    if min_pct > 0:
        for z in range(dz.M):
            zone_sum = sum([scores[dz.area2idx[b]] * dz.schools[dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
            # the following district_average is a weighted average. If a zone has more schools, their sum of qualities will of course be more
            district_average = sum(scores * dz.schools) / sum(dz.schools) * sum([dz.schools[dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
            y_school_balance = min(y_school_balance, zone_sum / district_average)
            if y_school_balance < min_pct:
            # if zone_sum < min_pct * district_average:
                y_school_balance = math.inf
    return y_school_balance

def evaluate_student_balance(dz, zone_dict, balance_limit):
    y_balance = 0
    #  the maximum distortion from average number of students (across zones)
    for z in range(dz.M):
        zone_stud = sum([dz.studentsInArea[dz.area2idx[b]] for b in zone_dict if zone_dict[b] == z])
        zone_imbalance = abs(zone_stud - dz.N / dz.M)
        if zone_imbalance > y_balance:
            y_balance = zone_imbalance

    if y_balance > balance_limit:
        y_balance = math.inf
    return y_balance



def evaluate_frl(dz, zone_dict, lbfrl_limit, ubfrl_limit):
    y_frl = 0
    for z in range(dz.M):
        zone_frl = sum([dz.cleaned_frl[dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
        zone_stud = sum([dz.studentsInArea[dz.area2idx[b]] for b in zone_dict if zone_dict[b] == z])
        avg_frl = (float(dz.F) / float(dz.N)) * zone_stud

        # district_average = (float(self.global_F) / float(self.global_N)) * gp.quicksum([self.studentsInArea[j] * self.x[j, z] for j in range(self.A)])
        # self.m.addConstr(zone_sum >= min_pct * district_average)
        if abs(avg_frl - zone_frl) > y_frl:
            y_frl = abs(avg_frl - zone_frl)

    if zone_frl < lbfrl_limit * avg_frl:
        y_frl = math.inf
    if zone_frl > ubfrl_limit * avg_frl:
        y_frl = math.inf
    return y_frl

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def drop_boundary(dz, zone_dict):
    sketchy_boundary = []
    for j in range(dz.A):
        level_j = dz.idx2area[j]
        if level_j in zone_dict:
            neighbors = dz.neighbors[j]
            if len(neighbors) >= 1:
                for neighbor_idx in neighbors:
                    if dz.idx2area[neighbor_idx] in zone_dict:
                        if zone_dict[dz.idx2area[neighbor_idx]] != zone_dict[level_j]:
                            sketchy_boundary.append(level_j)
                            break
                    else:
                        sketchy_boundary.append(level_j)
                        break

    for level_j in sketchy_boundary:
        zone_dict.pop(level_j, None)

    return zone_dict

# we trim neighbors on the boundary that do not satisfy strong contiguity
def trim_boundary(dz, zone_dict):
    while True:
        old_zone_dict_size = len(zone_dict)
        isContiguous, zone_dict = strong_contiguity_analysis(dz, zone_dict, mode="trimming")
        if old_zone_dict_size == len(zone_dict):
            break
        # old_zone_dict_size = len(zone_dict)
    return zone_dict

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def evaluate_contiguity(dz, zone_dict):
    isContiguous, dz = strong_contiguity_analysis(dz, zone_dict, mode="evaluation")
    return isContiguous

def strong_contiguity_analysis(dz, zone_dict, mode = "evaluation"):
    # every centroid belongs to its own zone
    for z in range(dz.M):
        level_z = dz.idx2area[dz.centroids[z]]
        if level_z in zone_dict:
            if zone_dict[level_z] != z:
                # print("for z : " +str(z) + "  centroid location is: " +str(level_z) +
                #       " which is assigned to zone :  " + str(zone_dict[level_z]))
                zone_dict.pop(level_z)
                return False, zone_dict
        # in evaluation mode, all centroid blockgroups must be assigned to a blocl.
        # but in trimming mode, these blockgroups can be missing from our zone_dict
        elif mode == "evaluation":
            print("z.idx2area[dz.centroids[" + str(z) + "]] should already be in zone_dict, Error")
            print(level_z)
            return False, None



    for j in range(dz.A):
        level_j = dz.idx2area[j]
        if j in dz.centroids:
            continue
        count = 0
        if level_j in zone_dict:
            c = dz.centroids[zone_dict[level_j]]
            closer_neighbors = dz.closer_euc_neighbors[j, c]
            # closer_neighbors = dz.closer_geodesic_neighbors[j, c]

            if len(closer_neighbors) >= 1:
                for neighbor_idx in closer_neighbors:
                    if dz.idx2area[neighbor_idx] in zone_dict:
                        if zone_dict[dz.idx2area[neighbor_idx]] == zone_dict[level_j]:
                            count += 1

            if count == 0:
                if mode == "trimming":
                    # print("dropping blockgroup " + str(level_j) + " that was matched to zone " +str(zone_dict[level_j]))
                    zone_dict.pop(level_j)
                elif mode == "evaluation":
                    if len(closer_neighbors) >= 1:
                        # print("moving to an undesired direction for blockgroup " + str(level_j))
                        return False, zone_dict

        elif mode == "evaluation":
            print("???")
            print(level_j)
            return False, zone_dict
    return True, zone_dict

def check_assignment_completeness(dz, zone_dict):
    for j in range(dz.A):
        if dz.idx2area[j] not in zone_dict:
            print("There is an unassigned area, Error.")
            print(dz.idx2area[j])


# def evaluate_distance(dz, zone_dict, distance_weight):
def evaluate_distance(dz, zone_dict):
    y_distance = 0
    for z in range(dz.M):
        zone_area = str(dz.idx2area[dz.centroids[z]])
        # y_distance += sum([((dz.drive_distances.loc[b, zone_area]) ** 2) * dz.studentsInArea[dz.area2idx[b]] for b in zone_dict if zone_dict[b] == z])
        y_distance += sum([((dz.euc_distances.loc[b, zone_area]) ** 2) * dz.studentsInArea[dz.area2idx[b]] for b in zone_dict if zone_dict[b] == z])
        # y_distance += sum([((dz.euc_distances.loc[b, zone_area]) ** 2) * dz.studentsInArea[dz.area2idx[b]] * distance_weight[b] for b in zone_dict if zone_dict[b] == z])

    y_distance = y_distance/dz.N
    return y_distance


def evaluate_shortage(dz, zone_dict, shortage_limit):
    y_shortage = 0

    # for z in range(dz.M):
    #     zone_stud = sum([dz.studentsInArea[dz.area2idx[b]] for b in zone_dict if zone_dict[b] == z])
    #     zone_seats = sum([dz.seats[dz.area2idx[b]] for b in zone_dict if zone_dict[b] == z])
    #     zone_shortage = zone_stud - zone_seats
    #     if zone_shortage > zone_seats * shortage_limit:
    #         print(zone_shortage/zone_seats)
    #         y_shortage = math.inf

    for z in range(dz.M):
        zone_shortage = sum(
                [(dz.studentsInArea[j] - dz.seats[j])
                 for j in range(dz.A)
                 if zone_dict[dz.idx2area[j]] == z]
            )
        zone_stud = sum(
                [dz.studentsInArea[j]
                 for j in range(dz.A)
                 if zone_dict[dz.idx2area[j]] == z]
            )

        if (zone_shortage) > zone_stud * shortage_limit:
            # print("Optimized shortage:                     " + str(zone_shortage/zone_stud))
            y_shortage = math.inf
            return y_shortage

    return y_shortage



def evaluate_diversity(dz, zone_dict, frl_dev, race_dev):
    y_diversity = 0

    for z in range(dz.M):
        if dz.use_loaded_data:
            zone_frl = sum([dz.area_data["frl_count"][dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
            zone_total_frl = sum([dz.area_data["frl_total_count"][dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])

            if zone_frl < zone_total_frl * (dz.FRL_ratio - frl_dev):
                y_diversity = math.inf
            if zone_frl > zone_total_frl * (dz.FRL_ratio + frl_dev):
                y_diversity = math.inf

            for race in dz.ethnicity_cols:
                race_ratio = sum(dz.area_data[race]) / sum(dz.area_data["num_with_ethnicity"])
                zone_sum = sum([dz.area_data[race][dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
                zone_total = sum([dz.area_data["num_with_ethnicity"][dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
                if (zone_sum < (race_ratio - race_dev) * zone_total):
                    y_diversity = math.inf
                if (zone_sum > (race_ratio + race_dev) * zone_total):
                    y_diversity = math.inf

        else:
            district_students = sum([dz.studentsInArea[dz.area2idx[b]] for b in zone_dict if zone_dict[b] == z])
            zone_frl = sum([dz.area_data["frl"][dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
            if (zone_frl < (float(dz.F) / (dz.N) - frl_dev) * district_students) | \
                    (zone_frl > (float(dz.F) / (dz.N) + frl_dev) * district_students):
                y_diversity = math.inf

            for race in \
                    dz.ethnicity_cols:
                    # ['resolved_ethnicity_White',
                    #      'resolved_ethnicity_Hispanic/Latinx',
                    #      'resolved_ethnicity_Asian']:
                race_ratio = sum(dz.area_data[race]) / float(dz.N)

                zone_sum = sum([dz.area_data[race][dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])

                if (zone_sum < (race_ratio - race_dev) * district_students):
                    y_diversity = math.inf
                if (zone_sum > (race_ratio + race_dev) * district_students):
                    y_diversity = math.inf



    return y_diversity

def evaluate_boundary(dz, zone_dict):
    y_boundary = 0
    # count total boundary cost
    for i in range(dz.A):
        for j in range(i + 1, dz.A):
            # if i and j are neighbors, check if they are boundaries of different zones
            if j in dz.neighbors[i]:
                if (dz.idx2area[i] in zone_dict) & (dz.idx2area[j] in zone_dict):
                    if zone_dict[dz.idx2area[i]] != zone_dict[dz.idx2area[j]]:
                        y_boundary += 1
                else:
                    print("*** Anomaly ***: " + str(dz.idx2area[i]) + " " + str(dz.idx2area[j]))
    return y_boundary



def evaluate_sch_count_balance(dz, zone_dict):
    y_sch_count_balance = 0

    zone_school_count = {}
    avg_school_count = sum([dz.schools[j] for j in range(dz.A)]) / dz.M + 0.0001
    for z in range(dz.M):
        zone_school_count[z] = sum([dz.schools[dz.area2idx[b]] for b in zone_dict if (zone_dict[b] == z)])
        if (zone_school_count[z] > avg_school_count + 1) | (zone_school_count[z] < avg_school_count - 1):
            y_sch_count_balance = math.inf


    # if K8 schools are included,
    # make sure no zone has more than one K8 schools
    if dz.include_k8:
        zone_k8_count = {}
        print("max value")
        print(dz.sc_df.index.max())
        for z in range(dz.M):
            zone_k8_count[z] = sum([
                dz.sc_df["K-8"][j]
                # for j in range(len(dz.sc_df.index))
                for j in range((dz.sc_df.index.max()))
                if (j in dz.sc_df[dz.level])
                if (zone_dict[dz.sc_df[dz.level][j]] == z)
            ])
            if zone_k8_count[z] > 1:
                y_sch_count_balance = math.inf


    return y_sch_count_balance

def evaluate_school_buffer_boundary(dz, zone_dict, boundary_threshold):
    # make sure areas that are closer than boundary_threshold distance
    # to a school, are matched to the same zone as that school.
    y_sch_buffer_boundary = 0

    for i, row in dz.sc_df.iterrows():
        if (row["K-8"] == 1) & (dz.include_k8 == False):
            continue
        s = row["school_id"]

        for b in zone_dict:
            if zone_dict[b] != zone_dict[dz.sch2block[s]]:
                # if dz.euc_distances.loc[dz.sch2block[s], str(b)] < boundary_threshold:
                if dz.area2idx[dz.sch2block[s]] in dz.neighbors[dz.area2idx[b]]:
                    y_sch_buffer_boundary += 1

    return y_sch_buffer_boundary

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def evaluate_assignment_score(param, dz, zone_dict, boundary_cost_fraction = 1):
    # check_assignment_completeness(dz, zone_dict)

    # boundary_cost_fraction is the fraction of boundary cost that should be included in the evaluated objective value.
    # over different iterations of local search, we increase boundary_cost_fraction from 0% to 100% (=1).

    # NOTE: In order to do the evaulation, we need to fix the order of centroids. When initializing "choices", we randomly select the schools
    # And this changes the initialization, and the assignment it represents in each round.
    # discontiguity_cost, distance_weight = evaluate_discontiguity_cost(dz, zone_dict)
    # ------------------------------------------------------------
    # param = Tuning_param()

    if evaluate_contiguity(dz, zone_dict) == False:
        # print("contiguity not satisfied")
        return math.inf
    # else:
    #     discontiguity_cost = 0
    #     distance_weight = {}
    #     for area in zone_dict:
    #         distance_weight[area] = 1
    # print("Optimized discontiguity  " + str(discontiguity_cost))

    # this y_distance is the sum, and not the max over the zones
    # y_distance = evaluate_distance(dz, zone_dict, distance_weight)
    y_distance = evaluate_distance(dz, zone_dict)
    if y_distance > 100000000000:
        print("Optimized distance:                     " + str(y_distance))
        return math.inf

    y_sch_count_balance = evaluate_sch_count_balance(dz, zone_dict)
    if y_sch_count_balance > 100000000000:
        print("Optimized school count balance:          " + str(y_sch_count_balance))
        return math.inf

    y_shortage = evaluate_shortage(dz, zone_dict, param.shortage)
    if y_shortage > 100000000000:
        print("Optimized shortage:                     " + str(y_shortage))
        return math.inf

    y_diversity = evaluate_diversity(dz, zone_dict, param.frl_dev, param.racial_dev)
    if y_diversity > 100000000000:
        print("Optimized diversity:                           " + str(y_diversity))
        return math.inf

    y_boundary = evaluate_boundary(dz, zone_dict)
    print("Optimized boundary:                          " + str(y_boundary))

    y_sch_buffer_boundary = evaluate_school_buffer_boundary(dz, zone_dict, param.boundary_threshold)
    print("Optimized buffer boundary:                   " + str(y_sch_buffer_boundary))

    # y_balance = evaluate_student_balance(dz, zone_dict, param.balance)
    # if y_balance > 100000000000:
    #     print("Optimized balance:                      " + str(y_balance))
    #     return math.inf

    # y_school_balance = evaluate_school_quality(dz, zone_dict, param.lbscqlty)
    # if y_school_balance > 100000000000:
    #     print("Optimized school quality imbalance:      " + str(y_school_balance))
    #     return math.inf

    # discontiguity_coef = 1000000000
    distance_coef = 1
    # balance_coef = 0
    # shortage_coef = 0
    # frl_coef = 0
    boundary_coef = 30
    school_buffer_coef = 1000
    # sch_quality_coef = -10
    # objective = discontiguity_coef * discontiguity_cost + distance_coef * y_distance + balance_coef * y_balance + shortage_coef * y_shortage \
    #                                            + frl_coef * y_frl + boundary_cost_fraction * boundary_coef * y_boundary +  sch_quality_coef * y_school_balance

    objective =  distance_coef * y_distance + school_buffer_coef * y_sch_buffer_boundary\
             + boundary_cost_fraction * boundary_coef * y_boundary
    # print("Objective value is")
    # print(objective)


    return objective