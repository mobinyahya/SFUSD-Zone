import numpy as np
import pandas as pd
import os
import csv
import geopandas as gpd
from shapely.geometry import Point, Polygon


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


def make_school_geodataframe(school_path ='~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv'):
    ''' make sc_merged '''
    # get school data table
    # cleanschoolpath = '~/SFUSD/Data/Cleaned/school_1819.csv'

    sc_df = pd.read_csv(school_path)

    # make GeoDataFrame
    geometry = [Point(xy) for xy in zip(sc_df['lon'], sc_df['lat'])]
    school_geo_df = gpd.GeoDataFrame(sc_df, crs='epsg:4326', geometry=geometry)

    # read shape file of attendance areas
    path = os.path.expanduser('~/Downloads/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
    sf = gpd.read_file(path)
    sf = sf.to_crs('epsg:4326')

    # school data and shape file merged
    sc_merged = gpd.sjoin(school_geo_df, sf, how="inner", op='intersects')

    return sc_merged


def make_simplified_school_geodataframe():
    ''' make sc_merged '''
    # get school data table
    # cleanschoolpath = '~/SFUSD/Data/Cleaned/school_1819.csv'
    cleanschoolpath = '~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv'
    sc_df = pd.read_csv(cleanschoolpath)

    # Make GeoDataFrame
    geometry = [Point(xy) for xy in zip(sc_df['lon'], sc_df['lat'])]
    school_geo_df = gpd.GeoDataFrame(sc_df, crs='epsg:4326', geometry=geometry)

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
    st_df['zone_id'] = st_df['idschoolattendance'].astype('Int64').replace(zone_dict)
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
    with open(zonefile,'r') as f:
        reader = csv.reader(f)
        zones = list(reader)

    zone_dict = {}
    for idx, schools in enumerate(zones):
        zone_dict = {**zone_dict,**{int(float(s)):idx for s in schools if s != ''}}
    return zone_dict