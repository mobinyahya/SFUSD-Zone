import csv
import os

import ortools.sat.python.cp_model


def add_optimization(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    boundary_vars = []
    file = os.path.expanduser(
        "~/Dropbox/SFUSD/Optimization/block_group_adjacency_matrix.csv"
    )
    with open(file, "r") as f:
        reader = csv.reader(f)
        adjacency_matrix = list(reader)

    neighbors = {}
    for row in adjacency_matrix:
        neighbors[row[0]] = set(row[1:])
    for zone in centroids:
        for bg in vm[zone]:
            for neighbor in neighbors[str(int(bg))]:
                if neighbor == '':
                    continue
                neighbor = int(neighbor)
                if float(neighbor) not in vm[zone]:
                    continue
                b = model.NewBoolVar(f"boundary_{bg}_{neighbor}")
    #             minimize the number of neighbors with different zoning
                model.AddAbsEquality(b, vm[zone][bg] - vm[zone][neighbor])
                boundary_vars.append(b)

    model.Minimize(sum(boundary_vars))