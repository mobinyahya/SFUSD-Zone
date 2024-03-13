import sys
import yaml
import gurobipy as gp
import pandas as pd

sys.path.append("../..")
from Zone_Generation.Config.Constants import *



def load_data_old():
    if include_k8:
        area_data = pd.read_csv(f"~/Dropbox/SFUSD/Data/Cleaned/final_area_data/area_data_k8.csv", low_memory=False)
    else:
        area_data = pd.read_csv(f"~/Dropbox/SFUSD/Data/Cleaned/final_area_data/area_data_no_k8.csv", low_memory=False)

    #####################  rename columns of input data so it matches with current format ###################
    area_data.drop(["all_nonsped_cap", "ge_schools", "ge_capacity"], axis='columns', inplace=True)
    area_data.rename(columns={"enrolled_and_ge_applied": "ge_students",
                                   "enrolled_students": "all_prog_students",
                                   "census_blockgroup": "BlockGroup"}, inplace=True)

    area_data.dropna(subset=['BlockGroup'], inplace=True)

    area_data["ge_students"] = area_data["ge_students"] / 6
    area_data["all_prog_students"] = area_data["all_prog_students"] / 6

    sch_stats = _load_school_data()
    area_data = area_data.merge(sch_stats, how='left', on="BlockGroup")

    for metric in ['frl_count', 'sped_count', 'ell_count',
                   'ge_students', 'all_prog_students', "all_prog_capacity", "ge_capacity",
                   'num_with_ethnicity', 'K-8'] + ETHNICITY_COLS:
        area_data[metric].fillna(value=0, inplace=True)

    # bg2att has thr following structural issue:
    # School S, might be in attendance area A, and in Blockgroup B.
    # But bg2att[B] != A. This is due to city structure, and that Blockgroups
    # are not always a perfect subset of only 1 attendance area.
    bg2att = load_bg2att(level)
    # We fix the structural issue in bg2aa, manually only for school locations.
    # Mapping of BGs to AA, for the location of schools
    bg2att_schools = dict(zip(school_df["BlockGroup"], school_df["attendance_area"]))
    bg2att.update(bg2att_schools)

    if level == 'attendance_area':
        # TODO 497 --> 0
        area_data['attendance_area'] = area_data['BlockGroup'].apply(lambda x: bg2att[int(x)] if int(x) in bg2att else 497)
        area_data = area_data.groupby(level, as_index=False).sum()
        # both schools 603 and 723 are in the same bg 60750228031. while they are in different aa (603 and 723)
        # Increase 'ge_capacity' by 44 and (capacity of school 603) for rows where 'attendance_area' equals 603
        area_data.loc[area_data['attendance_area'] == 603, 'ge_capacity'] += 44
        area_data.loc[area_data['attendance_area'] == 603, 'all_prog_capacity'] += 92
        area_data.loc[area_data['attendance_area'] == 603, 'num_schools'] += 1
        area_data.loc[area_data['attendance_area'] == 723, 'ge_capacity'] -= 44
        area_data.loc[area_data['attendance_area'] == 723, 'all_prog_capacity'] -= 92
        area_data.loc[area_data['attendance_area'] == 723, 'num_schools'] -= 1

        area_data.reset_index(inplace=True)

    return area_data