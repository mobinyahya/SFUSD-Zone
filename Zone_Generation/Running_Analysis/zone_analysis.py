'''
Putting  stuff from generate_zones that belongs elsewhere here to start
'''
import csv
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from generate_zones import DesignZones, _get_zone_dict, _get_constraints_from_filename


def get_zone_of_school(design_zones: DesignZones, school_id):
    tmp = design_zones.sch_df.loc[
        design_zones.sch_df["idschoolattendance"] == school_id
        ].reset_index()
    return tmp[design_zones.level][0]


def distance_zone_to_school(design_zones: DesignZones, zone_id, school_id):
    school_zone = get_zone_of_school(design_zones, school_id)
    if design_zones.level == "idschoolattendance":
        return design_zones.distances.loc[zone_id, str(float(school_zone))]
    return design_zones.distances.loc[zone_id, str(school_zone)]


def check_assignment(design_zones: DesignZones, assignmentname):
    zone_dict = _get_zone_dict(assignmentname)
    constrs = _get_constraints_from_filename(assignmentname)
    print(constrs)

    df = design_zones.area_data
    df["zone_id"] = df[design_zones.level].replace(zone_dict)
    df["num_students"] = df["num_students"].fillna(0)

    for k, v in constrs.items():
        if k == "distance" or k == "M":
            continue
        elif k == "frl":
            df2 = df.groupby("zone_id").sum()
            print(df2["frl"] / df2["num_students"])
            print("district avg:", sum(df[k]) / sum(df["num_students"]))
            if type(v) is tuple:
                print(
                    str(v[0] * 100) + "%:",
                    v[0] * sum(df[k]) / sum(df["num_students"]),
                )
                print(
                    str(v[1] * 100) + "%:",
                    v[1] * sum(df[k]) / sum(df["num_students"]),
                )
            else:
                print(str(v * 100) + "%:", v * sum(df[k]) / sum(df["num_students"]))
            continue

        df[k] = df[k].fillna(0)
        df["tmp"] = df["num_students"] * df[k]
        df2 = df.groupby("zone_id").sum()
        df2["tmp"] = df2["tmp"] / df2["num_students"]
        print(df2["tmp"])
        print(
            "district avg:",
            sum(df["num_students"] * df[k]) / sum(df["num_students"]),
        )
        if type(v) is tuple:
            print(
                str(v[0] * 100) + "%:",
                v[0] * sum(df["num_students"] * df[k]) / sum(df["num_students"]),
            )
            print(
                str(v[1] * 100) + "%:",
                v[1] * sum(df["num_students"] * df[k]) / sum(df["num_students"]),
            )
        else:
            print(
                str(v * 100) + "%:",
                v * sum(df["num_students"] * df[k]) / sum(df["num_students"]),
            )


def check_feasiblity(self):
    for j in range(0, self.A):
        for k in self.farAwayLists[j]:
            for z in range(0, self.M):
                if self.x[j, z] == 1 and self.x[k, z] == 1:
                    for z in range(0, self.M):
                        self.m.addConstr(self.x[j, z] + self.x[k, z] <= 1)


# def get_zone_stats(design_zones: DesignZones, assignmentname=""):
#     if assignmentname == "":
#         zone_dict = design_zones.zone_dict
#     else:
#         zone_dict = _get_zone_dict(assignmentname)
#     M = len(set(zone_dict.values()))
#
#     design_zones.studentsPerZone = np.zeros([M])
#     design_zones.seatsPerZone = np.zeros([M])
#     design_zones.deficitPerZone = np.zeros([M])
#     design_zones.maxDistanceInZone = np.zeros([M])
#     design_zones.imbalance = 0
#     design_zones.quailityEng = np.zeros([M])
#     design_zones.qualityMath = np.zeros([M])
#     design_zones.greatSchools = np.zeros([M])
#     design_zones.qualityMet = np.zeros([M])
#     design_zones.schoolsPerZone = np.zeros([M])
#     design_zones.qualityColor = np.zeros([M])
#
#     scoresEng = design_zones.area_data["eng_scores_1819"].fillna(value=0)
#     scoresMath = design_zones.area_data["math_scores_1819"].fillna(value=0)
#     scoresMet = design_zones.area_data["MetStandards"].fillna(value=0)
#     scoresColor = design_zones.area_data["AvgColorIndex"].fillna(value=0)
#     schools = design_zones.area_data["num_schools"].fillna(value=0)
#     scoresGreatSchools = design_zones.area_data["greatschools_rating"].fillna(value=0)
#
#     design_zones.M = M
#
#     for j in zone_dict:
#         if j == 0:
#             continue
#         design_zones.schoolsPerZone[zone_dict[j]] += 1
#         if (
#                 design_zones.schno2aa[j] != j
#         ):  # add only once, citywide already in aggregated data
#             continue
#         design_zones.studentsPerZone[zone_dict[j]] += design_zones.studentsInArea[design_zones.area2idx[j]]
#         design_zones.seatsPerZone[zone_dict[j]] += design_zones.seats[design_zones.area2idx[j]]
#         design_zones.quailityEng[zone_dict[j]] += (
#                 scoresEng[design_zones.area2idx[j]] * schools[design_zones.area2idx[j]]
#         )
#         design_zones.qualityMath[zone_dict[j]] += (
#                 scoresMath[design_zones.area2idx[j]] * schools[design_zones.area2idx[j]]
#         )
#         design_zones.qualityMet[zone_dict[j]] += (
#                 scoresMet[design_zones.area2idx[j]] * schools[design_zones.area2idx[j]]
#         )
#         design_zones.greatSchools += (
#                 scoresGreatSchools[design_zones.area2idx[j]] * schools[design_zones.area2idx[j]]
#         )
#         design_zones.qualityColor[zone_dict[j]] += (
#                 scoresColor[design_zones.area2idx[j]] * schools[design_zones.area2idx[j]]
#         )
#
#         for k in zone_dict:
#             if k == 0 or k != design_zones.schno2aa[k]:
#                 continue
#             if zone_dict[j] == zone_dict[k]:
#                 if (
#                         design_zones.distances.loc[j, str(k)]
#                         > design_zones.maxDistanceInZone[zone_dict[int(j)]]
#                 ):
#                     design_zones.maxDistanceInZone[zone_dict[j]] = design_zones.distances.loc[
#                         j, str(k)
#                     ]
#
#     for z in range(0, M):
#         design_zones.deficitPerZone[z] = design_zones.studentsPerZone[z] - design_zones.seatsPerZone[z]
#         design_zones.quailityEng[z] = design_zones.quailityEng[z] / design_zones.schoolsPerZone[z]
#         design_zones.qualityMet[z] = design_zones.qualityMet[z] / design_zones.schoolsPerZone[z]
#         design_zones.qualityMath[z] = design_zones.qualityMath[z] / design_zones.schoolsPerZone[z]
#         design_zones.qualityColor[z] = design_zones.qualityColor[z] / design_zones.schoolsPerZone[z]
#
#     design_zones.totalstudents = sum(design_zones.studentsInArea)
#     design_zones.totalseats = sum(design_zones.seats)
#
#     for z in range(0, M):
#         for q in range(0, M):
#             if design_zones.studentsPerZone[z] - design_zones.studentsPerZone[q] > design_zones.imbalance:
#                 design_zones.imbalance = design_zones.studentsPerZone[z] - design_zones.studentsPerZone[q]


def _find_prog_at_school(zones, prog_list):
    prog_zones = []
    program_codes = ["GE", "NC", "CE", "NS", "SN", "SB", "CB"]
    for z in zones:
        pz = []
        for aa in z:
            if aa == "":
                continue
            i = 0
            while "{}-{}-KG".format(aa, program_codes[i]) not in prog_list:
                i += 1
                if i >= len(program_codes):
                    print("ERROR: couldn't find program for {}".format(aa))
                    exit()
            pz.append("{}-{}-KG".format(aa, program_codes[i]))
        prog_zones.append(pz)
    return prog_zones


# def oct14_zone_stats(
#         design_zones: DesignZones, assignmentname
# ):
#     # load distance data if not yet loaded
#     if not hasattr(design_zones, "student_distance"):
#         dist_file = os.path.expanduser(
#             "~/SFUSD/Data/Precomputed/student_program_distances_dropoptout_1819.csv"
#         )
#         design_zones.student_distance = pd.read_csv(dist_file, index_col="studentno")
#         travel_file = os.path.expanduser(
#             "~/SFUSD/Data/Cleaned/student-all-KG-travel_data-filled.csv"
#         )
#         design_zones.travel_times = pd.read_csv(travel_file, index_col="studentno")
#
#     # format zone data
#     zone_dict, zone_list = _get_zone_dict(assignmentname, return_list=True)
#     prog_zones = _find_prog_at_school(zone_list, design_zones.student_distance.columns)
#     M = len(zone_list)
#
#     # predictability
#     schoolsPerZone = np.zeros([M])
#     studentsPerZone = np.zeros([M])
#     seatsPerZone = np.zeros([M])
#
#     # proximity
#     maxDistanceInZone = np.zeros([M])
#     maxTravelTimeInZone = np.zeros([M])
#     blockgroupsPerZone = np.zeros(M)
#
#     # get diversity metrics
#     df = design_zones.all_student_data
#     df = pd.get_dummies(df, columns=["resolved_ethnicity"])
#     df["frl"] = df["reducedlunch_prob"] + df["freelunch_prob"]
#     df["studentno"] = df["studentno"].astype("int64")
#     df["zone_id"] = df["idschoolattendance"].replace(zone_dict)
#     cols = [
#         "frl",
#         "HOCidx1",
#         "resolved_ethnicity_American Indian or Alaskan Native",
#         "resolved_ethnicity_Asian",
#         "resolved_ethnicity_Black or African American",
#         "resolved_ethnicity_Decline to State",
#         "resolved_ethnicity_Filipino",
#         "resolved_ethnicity_Hispanic/Latino",
#         "resolved_ethnicity_Pacific Islander",
#         "resolved_ethnicity_Two or More Races",
#         "resolved_ethnicity_White",
#     ]
#     diversity_metrics = df.groupby("zone_id").mean()[
#         cols
#     ]  # HOCidx1, frl, % race each zone
#
#     # get max travel time, max distance, and block group count for each zone
#     studentno2idx = dict(
#         zip(
#             design_zones.all_student_data["studentno"],
#             range(len(design_zones.all_student_data.index)),
#         )
#     )
#     for z, students in df.groupby("zone_id"):
#         studentnos = [int(studentno2idx[x]) for x in students["studentno"]]
#         dist = design_zones.student_distance.loc[studentnos, prog_zones[int(z)]].max()
#         maxDistanceInZone[int(z)] = max(dist)
#         tt_idxs = [
#             True if x in students["studentno"] else False
#             for x in design_zones.travel_times.index
#         ]
#         tt = design_zones.travel_times.loc[tt_idxs, zone_list[int(z)]].max()
#         maxTravelTimeInZone[int(z)] = max(tt)
#         blockgroupsPerZone[int(z)] = len(students["BlockGroup"].unique())
#
#     # get student and seat counts for deficit
#     for j in zone_dict:
#         if j == 0:
#             continue
#         schoolsPerZone[zone_dict[j]] += 1
#         if (
#                 design_zones.schno2aa[j] != j
#         ):  # add only once, citywide already in aggregated data
#             continue
#         studentsPerZone[zone_dict[j]] += design_zones.studentsInArea[design_zones.area2idx[j]]
#         seatsPerZone[zone_dict[j]] += design_zones.seats[design_zones.area2idx[j]]
#
#     deficitPerZone = studentsPerZone - seatsPerZone
#     result = pd.DataFrame(
#         np.array(
#             [
#                 maxDistanceInZone,
#                 maxTravelTimeInZone,
#                 deficitPerZone,
#                 schoolsPerZone,
#                 blockgroupsPerZone,
#             ]
#         ).T,
#         columns=[
#             "maxDistanceInZone",
#             "maxTravelTimeInZone",
#             "deficitPerZone",
#             "schoolsPerZone",
#             "blockgroupsPerZone",
#         ],
#     )
#     result = result.join(diversity_metrics)
#     return result


# def zone_summary_statistics(design_zones: DesignZones, assignmentname="", zoneStats=False):
#     if assignmentname == "":
#         zone_dict = design_zones.zone_dict
#         get_zone_stats(design_zones)
#     else:
#         zone_dict = _get_zone_dict(assignmentname)
#         if zoneStats:
#             get_zone_stats(design_zones,assignmentname)
#
#     df = design_zones.area_data
#     df.loc[:, "zone_id"] = df[design_zones.level].replace(zone_dict)
#     df[design_zones.level] = df[design_zones.level].astype("int64")
#
#     # get citywide capacity
#     sc_df = (
#         design_zones.sc_df[[design_zones.level, "r3_capacity", "all_program_cap"]]
#             .groupby(design_zones.level, as_index=False)
#             .sum()
#     )
#     sc_df.rename(
#         columns={
#             "r3_capacity": "cap_with_citywide",
#             "all_program_cap": "all_prog_cap_w_citywide",
#         },
#         inplace=True,
#     )
#     df = df.merge(sc_df, how="outer", on=design_zones.level)
#
#     # get all student count (not just enrolled)
#     df.rename(columns={"num_students": "num_enrolled"}, inplace=True)
#     st = design_zones.all_student_data[[design_zones.level]]
#     st.loc[:, "num_students"] = 1
#     st = st.groupby(design_zones.level, as_index=False).sum()
#     df = df.merge(st, how="left", on=design_zones.level)
#
#     intcols = [
#         "num_enrolled",
#         "r3_capacity",
#         "cap_with_citywide",
#         "all_program_cap",
#         "all_prog_cap_w_citywide",
#         "num_schools",
#         "frl",
#     ]
#     for col in intcols:
#         df.loc[:, col] = df[col].fillna(0).astype("int64")
#
#     cols = [
#         "HOCidx1",
#         "HOCidx2",
#         "HOCidx3",
#         "AALPI Score",
#         "Academic Score",
#         "N'hood SES Score",
#         "FRL Score",
#         "eng_scores_1819",
#         "math_scores_1819",
#         "MetStandards",
#         "AvgColorIndex",
#         "greatschools_rating",
#         "resolved_ethnicity_Asian",
#         "resolved_ethnicity_Black or African American",
#         "resolved_ethnicity_Decline to State",
#         "resolved_ethnicity_Hispanic/Latino",
#         "resolved_ethnicity_White",
#     ]
#     for col in cols:
#         df.loc[:, col] = df["num_enrolled"] * df[col]
#
#     df2 = df.groupby("zone_id").sum()
#     for col in cols:
#         df2.loc[:, col] = df2[col] / df2["num_enrolled"]
#     df2.loc[:, "M"] = len(np.unique(zone_dict.values()))
#     return df2[["M"] + intcols + cols]


def zone_metric_viz(design_zones: DesignZones, metric, assignmentname="", by_zone=True, centroids=False, title=""):
    if design_zones.level == "BlockGroup":
        df = design_zones.area_data[[design_zones.level, metric, "num_students"]]
        df = design_zones.census_sf.merge(df, how="left", on=design_zones.level)
    else:
        df = design_zones.area_data[[design_zones.level, metric, "num_students", "index_right"]]
        df = design_zones.aa_sf.merge(
            df, how="left", left_index=True, right_on="index_right"
        )

    # plot metrics for calculated zones
    if by_zone:
        if assignmentname == "":
            zone_dict = design_zones.zone_dict
        else:
            zone_dict = _get_zone_dict(assignmentname)
        df["zone_id"] = df[design_zones.level].replace(zone_dict)
        geo = df[["zone_id", "geometry"]].dissolve(by="zone_id")

        df.loc[:, metric] = df["num_students"] * df[metric]

        df2 = df.groupby("zone_id").sum()
        df2.loc[:, metric] = df2[metric] / df2["num_students"]
        df2 = gpd.GeoDataFrame(df2.join(geo))

    else:  # plot metrics at the block group or attendance area level
        df2 = gpd.GeoDataFrame(df)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    if design_zones.level == "BlockGroup":
        design_zones.census_sf.boundary.plot(ax=plt.gca(), alpha=0.4, color="grey")
    elif design_zones.level == "idschoolattendance":
        design_zones.aa_sf.boundary.plot(ax=plt.gca(), alpha=0.4, color="grey")

    # plot zones
    df2.plot(ax=ax, column=metric, cmap="OrRd", legend=True)

    # plot centroids
    if hasattr(design_zones, "centroids") and centroids == True:
        c = design_zones.area_data.iloc[design_zones.centroids]
        plt.scatter(c["lon"], c["lat"], s=20, c="blue", marker="^")

    # plot school locations
    # aa = self.sc_merged.loc[self.sc_merged['category']=='Attendance']
    # citywide = self.sc_merged.loc[self.sc_merged['category']=='Citywide']
    # plt.scatter(aa['lon'],aa['lat'],s=10, c='red',marker='s')
    # plt.scatter(citywide['lon'],citywide['lat'],s=10, c='black',marker='^')
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    if title != "":
        plt.title(title + ": " + metric)
    else:
        plt.title(metric)
    ax.set_xlim(-122.525, -122.350)
    ax.set_ylim(37.70, 37.84)
    # plt.show()


def read_language_zone(zone_file):
    with open(os.path.expanduser(zone_file), "r") as f:
        s = f.readline()
    aa2prog = eval(s)
    aa2prog = {k: str(v) for k, v in aa2prog.items()}
    data = np.array([list(aa2prog.keys()), list(aa2prog.values())], dtype=object).T
    df = pd.DataFrame(data=data, columns=["aa", "programs"])
    prog2zone = dict(
        zip(df["programs"].unique(), range(len(df["programs"].unique())))
    )
    df["zone_id"] = df["programs"].replace(prog2zone)
    aa2zone = dict(zip(df["aa"], df["zone_id"]))
    return aa2zone


def language_viz(design_zones: DesignZones, zone_file="", title="", metric="shortage"):
    if metric == "shortage":
        df = design_zones.area_data[
            [design_zones.level, "r3_capacity", "num_students", "index_right"]
        ]
    else:
        df = design_zones.area_data[[design_zones.level, metric, "num_students", "index_right"]]
    df = design_zones.aa_sf.merge(df, how="left", left_index=True, right_on="index_right")

    # plot metrics for calculated zones
    zone_dict = read_language_zone(zone_file)
    df["zone_id"] = df[design_zones.level].replace(zone_dict)
    geo = df[["zone_id", "geometry"]].dissolve(by="zone_id")

    if metric != "shortage":
        df.loc[:, metric] = df["num_students"] * df[metric]

        df2 = df.groupby("zone_id").sum()
        df2.loc[:, metric] = df2[metric] / df2["num_students"]
        # df2 =gpd.GeoDataFrame(df2.join(geo))
    else:
        df2 = df.groupby("zone_id").sum()
        df2.loc[:, "shortage"] = df2["num_students"] / df2["r3_capacity"]
    df2 = gpd.GeoDataFrame(df2.join(geo))

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    design_zones.aa_sf.boundary.plot(ax=ax, alpha=0.4, color="grey")

    # plot zones
    df2.plot(ax=ax, column=metric, cmap="OrRd", legend=True, vmin=40, vmax=52)

    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    if title != "":
        plt.title(title + ": " + metric)
    else:
        plt.title(metric)
    ax.set_xlim(-122.525, -122.350)
    ax.set_ylim(37.70, 37.84)
    plt.savefig(
        os.path.expanduser(
            "~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept18/"
        )
        + title
        + "_"
        + metric
    )
    # plt.show()


# def make_oct14_zone_tables():
#     dz = DesignZones(
#         M=6, level="idschoolattendance", include_citywide=True
#     )  # ,program_type=program_type)
#     # assignmentname = '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Itai/selected_zones/optGE246024.csv'
#     # df18 = pd.read_csv('~/Dropbox/SFUSD/Optimization/Zones/best_zones_sept23_nodup.csv')
#     df18 = pd.read_csv("~/Dropbox/SFUSD/Optimization/Zones/best_zones_may27.csv")
#     files1 = [
#         "~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv".format(x)
#         for x in df18["zone_file"]
#         if x[:5] == "optGE"
#     ]
#     files2 = [
#         "~/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/{}.csv".format(x)
#         for x in df18["zone_file"]
#         if x[:7] == "zoneopt"
#     ]
#     files3 = [
#         "~/Dropbox/SFUSD/Optimization/Zones/Zones_Sept9/{}.csv".format(x)
#         for x in df18["zone_file"]
#         if x[:7] == "optzone"
#     ]
#     files = [os.path.expanduser(x) for x in files1 + files2 + files3]
#     for assignmentname in files:
#         print(assignmentname)
#         result = oct14_zone_stats(dz, assignmentname)
#         name = assignmentname.split("/")[-1][:-4]
#         savefile = os.path.expanduser(
#             "~/Dropbox/SFUSD/Data/Computed/Kaleigh/{}_zone_stats.csv".format(name)
#         )
#         result.to_csv(savefile)


# def make_oct14_zone_tables_selected_zones():
#     dz = DesignZones(
#         M=6, level="idschoolattendance", include_citywide=True
#     )  # ,program_type=program_type)
#     zone_path = os.path.expanduser(
#         "~/Dropbox/SFUSD/Data/Computed/Itai/selected_zones/optGE*.csv"
#     )
#     zone_path = os.path.expanduser(
#         "~/Dropbox/SFUSD/Data/Computed/Itai/selected_zones/concept*zones.csv"
#     )
#     files = glob.glob(zone_path)
#     for assignmentname in [
#         "/Users/katherinementzer/Dropbox/SFUSD/Zones/concept1zones.csv"
#     ]:  # files:
#         result = oct14_zone_stats(dz, assignmentname)
#         name = assignmentname.split("/")[-1][:-4]
#         print(name)
#         savefile = os.path.expanduser(
#             "~/Dropbox/SFUSD/Data/Computed/Kaleigh/{}_zone_stats.csv".format(name)
#         )
#         result.to_csv(savefile)


def get_zone_dict(zone_file):
    with open(zone_file, "r") as f:
        reader = csv.reader(f)
        zones = list(reader)
    zone_dict = {}
    for idx, schools in enumerate(zones):
        zone_dict = {
            **zone_dict,
            **{int(float(s)): idx for s in schools if s != ""},
        }
    return zone_dict


def translate_colors_to_numerics(schools):
    cols = ['ela_color', 'math_color', 'chronic_color', 'suspension_color']
    translator = {'Red':1,'Orange': 2, 'Yellow': 3, 'Green': 4, 'Blue': 5, 'None': np.nan}
    for col in cols:
        schools[col] = schools[col].replace(translator)
    return schools


def zone_metrics_with_eligibile_students(zone_file, level='idschoolattendance'):
    dz = DesignZones(level=level)

    zone_dict = get_zone_dict(zone_file)
    dz.area_data['zone_id'] = dz.area_data[level].replace(zone_dict)

    mean_metrics = ['HOCidx1', 'HOCidx2', 'HOCidx3', 'AALPI Score',
       'Academic Score', 'N\'hood SES Score', 'FRL Score', 'frl', 'eng_scores_1819',
       'math_scores_1819', 'greatschools_rating', 'MetStandards',
       'AvgColorIndex']
    sum_metrics = ['resolved_ethnicity_American Indian or Alaskan Native',
       'resolved_ethnicity_Asian',
       'resolved_ethnicity_Black or African American',
       'resolved_ethnicity_Decline to State', 'resolved_ethnicity_Filipino',
       'resolved_ethnicity_Hispanic/Latino',
       'resolved_ethnicity_Pacific Islander',
       'resolved_ethnicity_Two or More Races', 'resolved_ethnicity_White',
       'num_students', 'sped_count', 'ell_count', 'all_program_cap', 'r3_capacity', 'num_schools']

    mean_metrics_data = dz.area_data[['zone_id'] + mean_metrics].groupby('zone_id').mean()
    sum_metrics_data = dz.area_data[['zone_id'] + sum_metrics].groupby('zone_id').sum()
    return mean_metrics_data.join(sum_metrics_data)


def zone_metrics_with_all_students(zone_file, level='idschoolattendance', student_data=None):
    if student_data is None:
        student_data = pd.read_csv('~/SFUSD/Data/Cleaned/student_1819.csv')

    zone_dict = get_zone_dict(zone_file)
    student_data['zone_id'] = student_data[level].replace(zone_dict)
    student_data["num_students"] = 1
    student_data["frl"] = (
        student_data["reducedlunch_prob"] + student_data["freelunch_prob"]
    )
    student_data["sped"] = np.where(student_data["speced"] == "Yes", 1, 0)
    student_data["ell"] = student_data["englprof_desc"].apply(
        lambda x: 1 if (x == "N-Non English" or x == "L-Limited English") else 0
    )
    student_data = pd.get_dummies(student_data, columns=["resolved_ethnicity"])

    mean_metrics = ['HOCidx1', 'HOCidx2', 'HOCidx3', 'AALPI Score',
       'Academic Score', 'N\'hood SES Score', 'FRL Score', 'frl']
    sum_metrics = ['resolved_ethnicity_American Indian or Alaskan Native',
       'resolved_ethnicity_Asian',
       'resolved_ethnicity_Black or African American',
       'resolved_ethnicity_Decline to State', 'resolved_ethnicity_Filipino',
       'resolved_ethnicity_Hispanic/Latino',
       'resolved_ethnicity_Pacific Islander',
       'resolved_ethnicity_Two or More Races', 'resolved_ethnicity_White',
       'num_students', 'sped', 'ell']

    mean_metrics_data = student_data[['zone_id'] + mean_metrics].groupby('zone_id').mean()
    sum_metrics_data = student_data[['zone_id'] + sum_metrics].groupby('zone_id').sum()
    return mean_metrics_data.join(sum_metrics_data)


def color_histogram_by_zone(zone_file):
    zone_dict = get_zone_dict(zone_file)
    schools = pd.read_csv('~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv')
    schools = schools.loc[schools['category'] == 'Attendance']
    schools['zone_id'] = schools['school_id'].replace(zone_dict)
    schools = translate_colors_to_numerics(schools)
    # fig, axs = plt.subplots(nrows=len(zone_dict.keys()), ncols=4, figsize=(10,27))
    for zone, sch_df in schools.groupby('zone_id'):
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10,3), sharey='all')
        for i, col in enumerate(['ela_color', 'math_color', 'chronic_color', 'suspension_color']):
            # print(zone,i)
            # plt.figure()
            sch_df[col].hist(ax=axs[i], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
            axs[i].set_xticks([1,2,3,4,5])
            axs[i].set_xticklabels(['Red', "Orange", 'Yellow', 'Green', 'Blue'], rotation=90)
            axs[i].set_title(f'{col}')
        plt.suptitle(f'Zone {zone}')
        plt.tight_layout()
        plt.savefig(os.path.expanduser(f'~/Desktop/zone{zone}_color_hist.png'))
            # sys.exit()
            # sch_df[col].hist(ax=axs[zone,i])
    # plt.show()


if __name__ == '__main__':
    # dz = DesignZones(level='BlockGroup', include_citywide=True)
    # print(dz.area_data[['BlockGroup','studentno', 'num_schools']])
    # print(dz.area_data.columns)
    medium_zones = os.path.expanduser('~/Documents/sfusd/local_runs/Data/optGE862432.csv')
    # from zone_viz import ZoneVisualizer
    # zv = ZoneVisualizer('aa')
    # print(len(zv.sc_merged))
    # zv.visualize_zones(medium_zones, label=False, show_schools=False)
    # zv.sc_merged = translate_colors_to_numerics(zv.sc_merged)
    # for col, name in zip(['AvgColorIndex', 'ela_color', 'math_color', 'chronic_color', 'suspension_color'], ['', 'ELA', 'Math', 'Chronic Absenteeism', 'Suspension']):
    #     zv.visualize_zones_with_column(medium_zones, col, title=f'Average {name} Color Index for Medium Zones', show=False, vmin=1,vmax=5)
    #     plt.savefig(os.path.expanduser(f'~/Desktop/zone_{col}.png'))

    # sns.set()
    # color_histogram_by_zone(medium_zones)
    df1 = zone_metrics_with_eligibile_students(medium_zones)
    print(df1)
    df2 = zone_metrics_with_all_students(medium_zones)
    print(df2)
