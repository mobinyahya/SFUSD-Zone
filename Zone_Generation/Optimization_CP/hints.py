import ortools.sat.python.cp_model
import pandas as pd


def add_hints(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    #  use results/mapping csv to get the school_id for each block group
    mapping = pd.read_csv('results/mapping/6_6_5_2.csv')

    for bg in vm:
        school_id = mapping[mapping['Block'] == bg]['school_id'].iloc[0]
        var = 1 if school_id == 999 else 0
        model.AddHint(vm[bg], var)