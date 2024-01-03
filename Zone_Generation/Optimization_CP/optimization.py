import csv
import os

import ortools.sat.python.cp_model
from ortools.sat.python import cp_model


def add_optimization(model, vm, school_df, bg_df, centroids, centroid_mapping, neighbors, travels, neighbor_pairs):
    boundary_vars = []
    for thing in neighbor_pairs.values():
        boundary_vars.extend(thing.values())

    model.Minimize(cp_model.LinearExpr.Sum(boundary_vars))
