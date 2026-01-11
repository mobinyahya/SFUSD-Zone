import csv
import math
import os
import pickle
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from Zone_Generation.Config.Constants import get_dropbox_path, get_sfusd_path, AREA_ETHNICITIES


def get_school_latlon():
    # get school latitude and longitude DataFrame
    sch_latlong = os.path.expanduser('~/SFUSD/Data/2018-19 SAS Schools Master List.xlsx')
    sc_ll = pd.read_excel(sch_latlong)
    sc_ll = sc_ll.drop_duplicates()
    sc_ll = sc_ll.drop(labels=[11, 85, 52])
    return sc_ll


def make_student_geodataframe(kg_only=False, passin=False, st_df=None, return_sf=False):
    if not passin:
        # get student data table
        cleanstudentpath = '~/SFUSD/Data/Cleaned/student_1819.csv'
        st_df = pd.read_csv(cleanstudentpath, low_memory=False)

    if kg_only:
        # only take kindergarteners
        st_df = st_df.loc[st_df['grade'] == 'KG']

    # make geo data frame
    st_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    geometry = [Point(xy) for xy in zip(st_df['longitude'], st_df['latitude'])]
    student_geo_df = gpd.GeoDataFrame(st_df, crs='epsg:4326', geometry=geometry)

    # read shape file of attendance areas
    path = os.path.expanduser('~/Downloads/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
    sf = gpd.read_file(path)
    sf = sf.to_crs('epsg:4326')

    # student data and shape file merged
    st_merged = gpd.sjoin(student_geo_df, sf, how="inner", op='intersects')
    if return_sf:
        return st_merged, sf
    return st_merged


def make_simplified_student_geodataframe(kg_only=False, dropnull=False, year=18):
    # get student data table
    cleanstudentpath = '~/SFUSD/Data/Cleaned/student_{}{}.csv'.format(year, year + 1)
    st_df = pd.read_csv(cleanstudentpath, low_memory=False)

    if kg_only:
        # only take kindergarteners
        st_df = st_df.loc[st_df['grade'] == 'KG']

    # if studentt lat lon is 0, 0, treat that as a missing value
    st_df['latitude'] = st_df['latitude'].replace({0: np.nan})
    st_df['longitude'] = st_df['longitude'].replace({0: np.nan})

    if dropnull:
        st_df.dropna(subset=['latitude', 'longitude'], inplace=True)

    # make geo data frame
    geometry = [Point(xy) for xy in zip(st_df['longitude'], st_df['latitude'])]
    crs = {'init': 'epsg:4326'}
    student_geo_df = gpd.GeoDataFrame(st_df, crs='epsg:4326', geometry=geometry)

    return student_geo_df


def make_school_geodataframe(school_path='~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv'):
    ''' make sc_merged '''
    # get school data table
    # cleanschoolpath = '~/SFUSD/Data/Cleaned/school_1819.csv'

    sch_df = pd.read_csv(school_path)

    # make GeoDataFrame
    geometry = [Point(xy) for xy in zip(sch_df['lon'], sch_df['lat'])]
    school_geo_df = gpd.GeoDataFrame(sch_df, crs='epsg:4326', geometry=geometry)

    # read shape file of attendance areas
    path = os.path.expanduser('~/Downloads/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
    sf = gpd.read_file(path)
    sf = sf.to_crs('epsg:4326')

    # school data and shape file merged
    sc_merged = gpd.sjoin(school_geo_df, sf, how="inner", op='intersects')

    return sc_merged


def load_census_shapefile(level, is_local):
    # get census block shapefile
    if is_local:
        path = os.path.expanduser(
            f'{get_sfusd_path(is_local)}/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
    else:
        path = '/share/data/school_choice/shapefiles/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp'
    census_sf = gpd.read_file(path)
    census_sf["Block"] = (
        census_sf["geoid10"].fillna(value=0).astype("int64", copy=False)
    )

    df = pd.read_csv(f"{get_dropbox_path(is_local)}/Optimization/block_blockgroup_tract.csv")
    df.loc[:, "Block"] = df["Block"].fillna(0)
    df["Block"] = df["Block"].astype("int64")

    census_sf = census_sf.merge(df, how="left", on="Block")

    census_sf.dropna(subset=['BlockGroup', 'Block'], inplace=True)
    census_sf[level] = census_sf[level].astype('int64')

    return census_sf


def load_euc_distance_data(level, area2idx, is_local, complete_bg=False, ):
    if level == "attendance_area":
        save_path = f"{get_dropbox_path(is_local)}/Optimization/distances_aa2aa.csv"
    elif level == "BlockGroup":
        save_path = f"{get_dropbox_path(is_local)}/Optimization/distances_bg2bg.csv"
    elif (level == "Block") & (complete_bg == False):
        save_path = f"{get_dropbox_path(is_local)}/Optimization/distances_b2b_schools.csv"
    elif (level == "Block") & (complete_bg == True):
        save_path = f"{get_dropbox_path(is_local)}/Optimization/distances_b2b.csv"

    if os.path.exists(os.path.expanduser(save_path)):
        distances = pd.read_csv(save_path, index_col=level)
        distances.columns = [int(float(x)) for x in distances.columns]

        distance_dict = {}
        rows = distances.index.tolist()
        cols = list(distances.columns)

        # Change the csv file into a double dictionary, so the distances can be accessed easier
        for area_i in rows:
            inner_dict = {}
            for area_j in cols:
                inner_dict[area2idx[area_j]] = distances.loc[area_i, area_j]
            distance_dict[area2idx[area_i]] = inner_dict

        return distance_dict

    if level == "Block":
        census_sf = load_census_shapefile(level, is_local)
        df = census_sf.dissolve(by="Block", as_index=False)
        df["centroid"] = df.centroid
        df["Lat"] = df["centroid"].apply(lambda x: x.y)
        df["Lon"] = df["centroid"].apply(lambda x: x.x)
        df = df[["Block", "Lat", "Lon"]]
        df.loc[:, "key"] = 0
        df = df.merge(df, how="outer", on="key")

        df.rename(
            columns={
                "Lat_x": "Lat",
                "Lon_x": "Lon",
                "Lat_y": "st_lat",
                "Lon_y": "st_lon",
                "Block_x": "Block",
            },
            inplace=True,
        )
    elif level == "BlockGroup":
        census_sf = load_census_shapefile(level, is_local)
        df = census_sf.dissolve(by="BlockGroup", as_index=False)
        df["centroid"] = df.centroid
        df["Lat"] = df["centroid"].apply(lambda x: x.y)
        df["Lon"] = df["centroid"].apply(lambda x: x.x)
        df = df[["BlockGroup", "Lat", "Lon"]]
        df.loc[:, "key"] = 0
        df = df.merge(df, how="outer", on="key")
        df.rename(
            columns={
                "Lat_x": "Lat",
                "Lon_x": "Lon",
                "Lat_y": "st_lat",
                "Lon_y": "st_lon",
                "BlockGroup_x": "BlockGroup",
            },
            inplace=True,
        )
    elif level == "attendance_area":
        df = area_data[["attendance_area", "lat", "lon"]]
        df.loc[:, "key"] = 0
        df = df.merge(df, how="outer", on="key")
        df.rename(
            columns={
                "lat_x": "Lat",
                "lon_x": "Lon",
                "lat_y": "st_lat",
                "lon_y": "st_lon",
                "attendance_area_x": "attendance_area",
            },
            inplace=True,
        )

    df["distance"] = df.apply(get_distance, axis=1)
    df[level] = df[level].astype('Int64')

    table = pd.pivot_table(
        df,
        values="distance",
        index=[level],
        columns=[level + "_y"],
        aggfunc=np.sum,
    )
    table.to_csv(save_path)
    return table


def load_driving_distance_data(level, choices, sch2level, area_data=None, destinations=None):
    if level == 'BlockGroup':
        savename = '~/Dropbox/SFUSD/Optimization/OD_drive_time_cut60.csv'

    if os.path.exists(os.path.expanduser(savename)):
        drive_time = pd.read_csv(savename)
    if destinations == None:
        destinations = choices

    drive_time_distance = pd.DataFrame(index=sorted(list(area_data['BlockGroup'])))
    for school_id in destinations:
        # make sure this school_id was found by the system
        if len(drive_time.loc[drive_time['Name_1'] == school_id]) != 0:
            distance_array = []
            for bg in sorted(list(area_data['BlockGroup'])):
                # make sure this bg id was found by the system, and the lack of distance info
                # is not just due to the cut-off
                if len(drive_time.loc[drive_time['Name_12'] == bg]) != 0:
                    dist = drive_time.loc[drive_time['Name_1'] == school_id].loc[drive_time['Name_12'] == bg][
                        'Total_Trav']
                    if len(dist) == 1:
                        dist = float(dist)
                    elif len(dist) == 0:
                        # dist = 100
                        dist = math.inf
                    else:
                        print("duplicate distance data, error")
                        print(dist)

                else:
                    # value -1 represents an unassigned value for now, we later
                    # construct value for these indices, using their neighbor avg
                    dist = -1
                    # print("missing bg init: " + str(bg))

                distance_array.append(dist)
            print("school_id  " + str(school_id))
            print(" this is the sch2block  " + str(sch2level[school_id]))
            drive_time_distance[str(sch2level[school_id])] = distance_array


def load_b2bg(is_local):
    df = pd.read_csv(f'{get_dropbox_path(is_local)}/Optimization/block_blockgroup_tract.csv')
    b2bg = {}
    # Iterate over each row in the DataFrame to populate the dictionary
    for _, row in df.iterrows():
        bg = row['BlockGroup']
        b = row['Block']

        # If the blockgroup is not yet in the dictionary, add it with the current block in a list
        if b not in b2bg:
            b2bg[b] = bg
        # else:
        #     print("Duplicate exists in the map")
    return b2bg


def load_bg2att(is_local, census_sf=None):
    savename = f'{get_dropbox_path(is_local)}/Optimization/bg2aa_mapping.pkl'

    # # This mapping is based on polygon shapefile information (not the students info)
    # if self.level=='Block':
    #     savename ='/Users/mobin/Dropbox/SFUSD/Optimization/b2aa_mapping.pkl'
    # if self.level=='BlockGroup':
    #     savename ='/Users/mobin/Dropbox/SFUSD/Optimization/bg2aa_mapping.pkl'
    # elif self.level=='attendance_area':
    #     # We don't need a mapping in this case
    #     return

    if os.path.exists(os.path.expanduser(savename)):
        file = open(savename, "rb")
        bg2att = pickle.load(file)
        print("bg to aa map was loaded from file")
        return bg2att

    level = "Blockgroup"
    # load attendance area geometry + its id in a single dataframe
    # path = os.path.expanduser('~/Downloads/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
    # moving out of downloads for obvious reasons
    path = os.path.expanduser(f'{get_sfusd_path(is_local)}/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
    sf = gpd.read_file(path)
    sf = sf.to_crs('epsg:4326')
    sc_merged = make_school_geodataframe()
    translator = sc_merged.loc[sc_merged['category'] == 'Attendance'][['school_id', 'index_right']]
    translator['school_id'] = translator['school_id'].fillna(value=0).astype('int64', copy=False)
    sf = sf.merge(translator, how='left', left_index=True, right_on='index_right')

    # load blockgroup/block  geometry + its id in a single dataframe
    df = census_sf.dissolve(level, as_index=False)

    bg2att = {}
    for i in range(len(df.index)):
        area_c = df['geometry'][i].centroid
        for z, row in sf.iterrows():
            aa_poly = row['geometry']
            # if aa_poly.contains(area_c) | aa_poly.touches(area_c):
            if aa_poly.contains(area_c):
                bg2att[df[level][i]] = row['school_id']

    file = open(savename, "wb")
    pickle.dump(bg2att, file)
    file.close()

    return bg2att


def make_simplified_school_geodataframe():
    ''' make sc_merged '''
    # get school data table
    # cleanschoolpath = '~/SFUSD/Data/Cleaned/school_1819.csv'
    cleanschoolpath = '~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv'
    sch_df = pd.read_csv(cleanschoolpath)

    # Make GeoDataFrame
    geometry = [Point(xy) for xy in zip(sch_df['lon'], sch_df['lat'])]
    school_geo_df = gpd.GeoDataFrame(sch_df, crs='epsg:4326', geometry=geometry)

    return school_geo_df


# helper functions copied from characteristics_table.py
def str_to_list(x):
    ''' helper function for reading in preference lists '''
    l = x[1:-1].split(',')
    return [y.strip() for y in l]


def get_first_choice(df):
    ''' make column in student data table containing school first choice'''
    df['firstchoice'] = np.nan
    for round in range(1, 6):
        col = 'r{}_ranked_idschool'.format(round)
        if col in df.columns:
            # format column names
            df[col] = df[col].fillna('[]')
            df[col] = df[col].apply(str_to_list)
            tmp = 'r{}_tmp'.format(round)
            # get first choice from this round, if exists
            df[tmp] = [df.at[i, col][0] if df.at[i, col][0] != '' else np.nan for i in df.index]
            df['firstchoice'] = df['firstchoice'].fillna(df[tmp])
            df = df.drop(columns=[tmp])

    return df


def get_first_choice_program(df):
    ''' make column in student data table containing program first choice'''
    df['firstchoice_prog'] = np.nan
    if 'program_types' not in df.columns:
        df = make_program_type_lists(df)
    for round in range(1, 6):
        col = 'r{}_ranked_idschool'.format(round)
        pcol = 'r{}_programs'.format(round)
        if col in df.columns:
            # format column names
            df[col] = df[col].fillna('[]')
            df[col] = df[col].apply(str_to_list)
            tmp = 'r{}_tmp'.format(round)
            # get first choice from this round, if exists
            df[tmp] = ['{}_{}'.format(df.at[i, col][0], df.at[i, pcol][0]) if \
                           df.at[i, col][0] != '' else np.nan for i in df.index]
            df['firstchoice_prog'] = df['firstchoice_prog'].fillna(df[tmp])
            df = df.drop(columns=[tmp])

    return df


def get_second_choice(df):
    ''' make column in student data table containing school first choice'''
    df['secondchoice'] = np.nan
    for round in range(1, 6):
        col = 'r{}_ranked_idschool'.format(round)
        if col in df.columns:
            # format column names
            # df[col] = df[col].fillna('[]')
            # df[col] = df[col].apply(str_to_list)
            tmp = 'r{}_tmp'.format(round)
            # get first choice from this round, if exists
            df[tmp] = [df.at[i, col][1] if len(df.at[i, col]) > 1 else np.nan for i in df.index]
            df['secondchoice'] = df['secondchoice'].fillna(df[tmp])
            df = df.drop(columns=[tmp])

    return df


def get_designation_status(df):
    ''' make column in student data table indicating if the student was
    designated in the last round they participated in '''
    df['designated'] = np.nan
    for round in reversed(range(1, 6)):
        col = 'r{}_isdesignation'.format(round)
        if col in df.columns:
            df['designated'].fillna(df[col], inplace=True)
    return df


# helper function from student summary statistics to see if student applied to school
def make_cols_lists(df):
    for round in range(1, 6):
        # format column name
        col = 'r{}_ranked_idschool'.format(round)
        if col in df.columns:
            # format round rankings
            df[col] = df[col].fillna('[]')
            df[col] = df[col].apply(str_to_list)
            df[col] = df[col].apply(lambda x: [int(l) for l in x if '' not in x])
    return df


def make_all_schools_list(df):
    ''' create column with each school applied to across all rounds'''
    for round in range(1, 6):
        # format column name
        col = 'r{}_ranked_idschool'.format(round)
        if col in df.columns:
            if round == 1:
                df['all_schools'] = df[col]
            else:
                df['all_schools'] += df[col]
    df['all_schools'] = df['all_schools'].apply(lambda x: np.unique(x))
    return df


def applied_to_school(df, schoolid):
    ''' helper function making indicator column representing whether or not the
    student applied to this particular school (adapted from
    student_summary_statistics.py) '''

    df['applied'] = 0
    for round in range(1, 6):
        # format column name
        col = 'r{}_ranked_idschool'.format(round)
        if col in df.columns:
            new = 'r{}_applied'.format(round)

            # make indicator
            df[new] = [schoolid in df.at[x, col] for x in df.index]
            df[new] = np.where(df[new], 1, 0)

    for round in range(1, 6):
        # format column name
        new = 'r{}_applied'.format(round)
        if new in df.columns:
            df['applied'] = df['applied'] + df[new]
            # df = df.drop(new)
    df['applied'] = np.where(df['applied'] > 0, 1, 0)
    return df


def convert_to_block_zone_dict(zone_dict, G):
    if not zone_dict:
        return {}
    block_zone_dict = {}
    for node, zone in zone_dict.items():
        if zone is None:
            raise ValueError("Zone dict contains None zone assignment.")
        if np.isnan(zone):
            raise ValueError("Zone dict contains NaN zone assignment.")
        # if area_id is an attribute, just use it
        if 'area_id' in G.nodes[node]:
            area_id = G.nodes[node]['area_id']
            block_zone_dict[area_id] = zone
        else:
            # otherwise, assume node is a list of block ids
            block_ids = G.nodes[node]['block_ids']
            for block_id in block_ids:
                block_zone_dict[block_id] = zone
    return block_zone_dict

def convert_block_zone_to_zone_dict(block_zone_dict, G):
    zone_dict = {}
    for node in G.nodes():
        if 'area_id' in G.nodes[node]:
            zone_dict[node] = block_zone_dict[G.nodes[node]['area_id']]
            continue
        block_ids = G.nodes[node]['block_ids']
        # check that all block_ids have the same zone assignment
        zones = {block_zone_dict[block_id] for block_id in block_ids}
        if len(zones) > 1:
            raise ValueError(f"Block IDs {block_ids} in node {node} have conflicting zone assignments: {zones}")

        # assign the zone of the first block_id in the list
        zone_dict[node] = block_zone_dict[block_ids[0]]
    return zone_dict

def calculate_euc_distance(lat1, lon1, lat2, lon2):
    # print(str(lat1) + "  " + str(lon1) + "  " + str(lat2) + "  " + str(lon2) + "  " )
    return 6371.01 * np.arccos(np.sin(lat1 * np.pi / 180) * np.sin(lat2 * np.pi / 180) + \
                               np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) \
                               * np.cos((lon1 - lon2) * np.pi / 180)) * 0.621371  # return distance in miles


def get_distance(row):
    ''' helper function for calculating distance from student to school, get
    distance in miles between two lat-lon pairs'''
    lat1 = row['Lat']
    lon1 = row['Lon']
    lat2 = row['st_lat']
    lon2 = row['st_lon']
    return calculate_euc_distance(lat1, lon1, lat2, lon2)


def fix_studentno(df):
    # take the leading 'S' off of student id number to make it an int
    df["studentno"] = df['studentno'].str.slice(1, None)
    df["studentno"] = pd.to_numeric(df["studentno"])
    return df


def make_program_type_lists(df):
    ''' create column with each type of program applied to '''
    # df['program_types'] =
    for round in range(1, 6):
        # format column name
        col = 'r{}_programs'.format(round)
        if col in df.columns:
            # format round rankings
            df[col] = df[col].fillna('[]')
            df[col] = df[col].apply(str_to_list)
            df[col] = df[col].apply(lambda x: [l[1:-1] for l in x if '' not in x])

            if round == 1:
                df['program_types'] = df[col]
            else:
                df['program_types'] = df['program_types'] + df[col]
    df['program_types'] = df['program_types'].apply(lambda x: np.unique(x))
    return df


def combine_asian_ethnicities(df):
    asian = ['Asian Indian', 'Cambodian', 'Chinese', 'Japanese', 'Korean', \
             'Other Asian', 'Vietnamese', 'Hmong']
    eth_dict = dict(zip(asian, ['Asian'] * len(asian)))
    df.resolved_ethnicity.replace(eth_dict, inplace=True)
    return df


def make_request_table(year):
    scpath = '~/SFUSD/Data/School choice data/school choice data/'
    for round in range(1, 4):
        filename = "out_SFUSD20191220_1_1_sas_{}{}r{}_prerun_request.dta.csv".format(year, year + 1, round)
        if not os.path.exists(os.path.expanduser(scpath + filename)):
            print('file not found:', filename)
            continue
        if round == 1:
            requests = pd.read_csv(scpath + filename, low_memory=False)
        else:
            df = pd.read_csv(scpath + filename, low_memory=False)
            requests = requests.append(df)
    requests = requests.loc[requests['grade'] == 'KG']
    return requests


def get_diversity_categories(st_df):
    thresh33, thresh66 = np.percentile(st_df['HOCidx1'].dropna(), [33, 66])
    st_df['diversity_category'] = st_df['HOCidx1'].apply(lambda x: 1 if x < thresh33 else (2 if x < thresh66 else 3))
    return st_df


def get_SES_score(st_df):
    st_df['SES_score'] = .25 * st_df["N'hood SES Score"] + .25 * st_df['FRL Score']
    thresh33, thresh66 = np.percentile(st_df['SES_score'].dropna(), [33, 66])
    st_df['SES_category'] = st_df['SES_score'].apply(lambda x: 1 if x < thresh33 else (2 if x < thresh66 else 3))
    return st_df


def get_CTIP_categories(st_df):
    # get CTIP category
    blks = '~/Dropbox/SFUSD/Data/block_to_ctip_category.csv'
    blks = pd.read_csv(blks)
    st_df = st_df.merge(blks, how='left', left_on='census_block', right_on='Block')
    return st_df


def get_student_zone(zone_file, st_df, return_dict=False):
    with open(os.path.expanduser(zone_file), 'r') as f:
        reader = csv.reader(f)
        zones = list(reader)
    zone_dict = {}
    for idx, schools in enumerate(zones):
        zone_dict = {**zone_dict, **{int(float(s)): idx for s in schools if s != ''}}
    st_df['zone_id'] = st_df['attendance_area'].astype('Int64').replace(zone_dict)
    if return_dict:
        return st_df, zone_dict
    return st_df


def shorten_ethnicity_names(st_df):
    shortened = {'Black or African American': 'Black', 'Hispanic/Latino': 'Latinx'}
    st_df['resolved_ethnicity'].replace(shortened, inplace=True)
    return st_df


def make_aalpi_column(st_df):
    st_df['aalpi'] = st_df['resolved_ethnicity'].apply(lambda x: 1 if x in \
                                                                      ['Black or African American', 'Hispanic/Latino',
                                                                       'Pacific Islander'] else 0)
    return st_df


def simplify_ethnicities(df):
    ethnicity_dict = {
        'Chinese': 'Asian',
        'Two or More': 'Two or More Races',
        'Middle Eastern/Arab': 'White',
        'Decline To State': 'Decline to state',
        'American Indian or Alaska Native': 'American Indian',
        'Korean': 'Asian',
        'Hispanic': 'Hispanic/Latinx',
        'Cambodian': 'Asian',
        'Japanese': 'Asian',
        'Other Pacific Islander': 'Pacific Islander',
        'Hawaiian': 'Pacific Islander',
        'Black or African American': 'Black or African American',
        'Vietnamese': 'Asian',
        'Samoan': 'Pacific Islander',
        'Tahitian': 'Pacific Islander',
        'Laotian': 'Asian',
        'Asian Indian': 'Asian',
        'Not Specified': 'Not specified',
        'Other Asian': 'Asian',
        'Hmong': 'Asian',
        'Middle Eastern/Arabic': 'White',
        'American Indian or Alaskan Native': 'American Indian',
        'Hispanic/Latino': 'Hispanic/Latinx',
        'Two or more races': 'Two or More Races'
    }
    df['resolved_ethnicity'].replace(ethnicity_dict, inplace=True)
    return df


def get_zone_dict(zonefile):
    with open(zonefile, 'r') as f:
        reader = csv.reader(f)
        zones = list(reader)

    zone_dict = {}
    for idx, schools in enumerate(zones):
        zone_dict = {**zone_dict, **{int(float(s)): idx for s in schools if s != ''}}
    return zone_dict

def compute_zone_demographics(G, zone_dict):
    """
            Calculates aggregated and proportional demographics (total students, FRL proportion,
            and racial/ethnic proportions) for each zone.
            """

    # 1. Define the columns to aggregate and the corresponding keys for the output dictionary
    # The keys will be like 'White', 'Black', etc.
    ETHNICITY_KEYS = [col[len('Ethnicity_'):] for col in AREA_ETHNICITIES]

    # 2. Use a defaultdict to simplify accumulation and avoid checking for key existence
    # We initialize with a lambda that returns a dict structure for aggregation
    zone_aggregates = defaultdict(lambda: {
        "total_students": 0.0,
        "FRL": 0.0,
        **{key: 0.0 for key in ETHNICITY_KEYS}
    })
    # 3. Iterate over each area in the graph and accumulate counts into the appropriate zone

    for node in G.nodes(data=True):
        area_idx = node[0]
        zone_id = zone_dict[area_idx]
        area_data = node[1]

        # Reference the aggregate dictionary for the current zone
        agg = zone_aggregates[zone_id]

        # Accumulate total students and FRL counts
        agg["total_students"] += float(area_data["ge_students"])
        agg["FRL"] += float(area_data["FRL"])
        for ethnicity_col, key in zip(AREA_ETHNICITIES, ETHNICITY_KEYS):
            agg[key] += float(area_data[ethnicity_col])

    # 4. Convert aggregated counts to proportions
    final_zone_demographics = {}
    for zone_id, data in zone_aggregates.items():
        total_students = data["total_students"]

        # Create a new dictionary for the final, proportional results
        final_demographics = {"total_students": round(total_students, 2)}

        if total_students > 0:
            # Calculate FRL proportion
            final_demographics["FRL"] = round(data["FRL"] / total_students, 2)

            # Calculate ethnicity proportions
            for key in ETHNICITY_KEYS:
                final_demographics[key] = round(data[key] / total_students, 2)
        else:
            # Handle zones with zero students (FRL and ethnic proportions are 0)
            final_demographics["FRL"] = 0.0
            for key in ETHNICITY_KEYS:
                final_demographics[key] = 0.0

        final_zone_demographics[zone_id] = final_demographics

    return final_zone_demographics

def compute_zone_deviations(G, zone_dict):
    zone_demographics = compute_zone_demographics(G, zone_dict)
    # get deviations in FRL and racial composition from average across zones
    if not zone_demographics:
        return {"max_frl_deviation": 0.0, "max_racial_deviation": 0.0, "per_ethnicity_deviation": {}}

    zones = list(zone_demographics.keys())
    num_zones = len(zones)

    if num_zones == 0:
        return {"max_frl_deviation": 0.0, "max_racial_deviation": 0.0, "per_ethnicity_deviation": {}}

    # infer ethnicity keys as everything except 'total_students' and 'FRL'
    sample = next(iter(zone_demographics.values()))
    ethnicity_keys = [k for k in sample.keys() if k not in ('total_students', 'FRL')]

    # compute average FRL and ethnicity proportions across zones
    avg_frl = sum(zone_demographics[z].get('FRL', 0) for z in zones) / num_zones
    avg_ethnicity = {eth: sum(zone_demographics[z].get(eth, 0) for z in zones) / num_zones
                     for eth in ethnicity_keys}

    # compute deviations
    frl_deviations = [abs(zone_demographics[z].get('FRL', 0) - avg_frl) for z in zones]
    frl_deviation = max(frl_deviations) if frl_deviations else 0.0

    per_ethnicity_deviation = {}
    for eth in ethnicity_keys:
        eth_deviations = [abs(zone_demographics[z].get(eth, 0) - avg_ethnicity[eth]) for z in zones]
        per_ethnicity_deviation[eth] = round(max(eth_deviations), 2) if eth_deviations else 0.0

    max_racial_deviation = max(per_ethnicity_deviation.values()) if per_ethnicity_deviation else 0.0

    return {
        "max_frl_deviation": round(frl_deviation, 2),
        "max_racial_deviation": max_racial_deviation,
        "per_ethnicity_deviation": per_ethnicity_deviation
    }


def Compute_Name(config):
    name = str(config["centroids_type"])
    # # add frl deviation
    # name += "_frl_" + str(config["frl_dev"])
    # # add shortage
    # # name += "_shortage_" + str(config["shortage"])

    return name


def load_zones_from_file(file_path):
    zone_lists = []
    with open(file_path, 'r', newline='') as file:
        print("file_path ", file_path)
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            # Convert each element in the row to an integer and store it in the list
            zone_row = []
            for cell in row:
                # Split the cell content by commas, convert to integers, and append to the row
                cell_values = [int(val.strip()) for val in cell.split(',') if val.strip()]  #
                zone_row.extend(cell_values)
            zone_lists.append(zone_row)

    # build a zone dictionary based on zone_list
    zone_dict = {}
    for index, sublist in enumerate(zone_lists):
        for item in sublist:
            zone_dict[item] = index

    return zone_lists, zone_dict
