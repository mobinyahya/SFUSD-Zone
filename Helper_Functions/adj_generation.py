import csv
import os
import sys

import geopandas as gpd
import pandas as pd

sys.path.insert(1, '..')
from util import make_school_geodataframe

from haversine import haversine, Unit


def make_block_distance_matrix():
    #     use 2020 census block shapefile
    path = os.path.expanduser(
        '~/Downloads/Census 2020_ Blocks for San Francisco_20241204[33]/geo_export_7924b18f-b5ea-4c14-b7cc-819b95caaf08.shp')
    census_sf = gpd.read_file(path)
    census_sf['geoid20'] = census_sf['geoid20'].fillna(value=0).astype('int64', copy=False)

    print(census_sf.columns)
    savefile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/distances_b2b_20.csv')
    # TODO: the following "dissolve" should not have any effect. But once I comment this line, we get an error in line 29
    census_sf = census_sf.dissolve(by='geoid20', as_index=False)

    with open(savefile, 'w') as f:
        writer = csv.writer(f)
        all_rows = len(census_sf)
        for i in range(all_rows):
            row = census_sf.iloc[i]
            centroid1 = row['geometry'].centroid
            for j in range(i , all_rows):
                row2 = census_sf.iloc[j]
                centroid2 = row2['geometry'].centroid
                # use haversine package
                miles = haversine((centroid1.y, centroid1.x), (centroid2.y, centroid2.x), unit=Unit.MILES)
                writer.writerow([int(row['geoid20']), int(row2['geoid20']), float(miles)])

def make_block_adjacency_list():
    # get census block shapefile
    # path = os.path.expanduser('~/SFUSD/Census 2010_ Blocks for San Francisco/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp')
    path = os.path.expanduser(
        '/Users/kumar/Downloads/Census 2020_ Blocks for San Francisco_20241204[33]/geo_export_7924b18f-b5ea-4c14-b7cc-819b95caaf08.shp')
    census_sf = gpd.read_file(path)
    census_sf['geoid20'] = census_sf['geoid20'].fillna(value=0).astype('int64', copy=False)

    print(census_sf.columns)
    savefile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/b_adjacency_matrix_20.csv')
    # TODO: the following "dissolve" should not have any effect. But once I comment this line, we get an error in line 29
    census_sf = census_sf.dissolve(by='geoid20', as_index=False)

    with open(savefile, 'w') as f:
        writer = csv.writer(f)
        for index, row in census_sf.iterrows():
            neighbors = census_sf[census_sf.geometry.touches(row['geometry'])]['geoid20'].tolist()
            neighbors = [str(int(row['geoid20']))] + [str(int(x)) for x in neighbors]
            writer.writerow(neighbors)


def make_block_group_adjacency_list():
    # get census block shapefile
    path = os.path.expanduser(
        '~/SFUSD/Census 2010_ Blocks for San Francisco/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp')
    census_sf = gpd.read_file(path)
    census_sf['geoid10'] = census_sf['geoid10'].fillna(value=0).astype('int64', copy=False)

    df = pd.read_csv('~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv')
    df['Block'] = df['Block'].fillna(value=0).astype('int64', copy=False)
    census_sf = census_sf.merge(df, how='left', left_on='geoid20', right_on='Block')

    census_sf = census_sf.dissolve(by='BlockGroup', as_index=False)

    savefile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/bg_adjacency_matrix_20.csv')
    with open(savefile, 'w') as f:
        writer = csv.writer(f)
        for index, row in census_sf.iterrows():
            neighbors = census_sf[census_sf.geometry.touches(row['geometry'])]['BlockGroup'].tolist()
            neighbors = [str(int(row['BlockGroup']))] + [str(int(x)) for x in neighbors]
            writer.writerow(neighbors)


def make_attendance_area_adjacency_list():
    # read shape file of attendance areas
    path = os.path.expanduser('~/Downloads/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
    sf = gpd.read_file(path)
    sf = sf.to_crs('epsg:4326')

    sc_merged = make_school_geodataframe()
    translator = sc_merged.loc[sc_merged['category'] == 'Attendance'][['school_id', 'index_right']]
    translator['school_id'] = translator['school_id'].fillna(value=0).astype('int64', copy=False)
    sf = sf.merge(translator, how='left', left_index=True, right_on='index_right')

    savefile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/attendance_area_adjacency_matrix.csv')
    with open(savefile, 'w') as f:
        writer = csv.writer(f)
        for index, row in sf.iterrows():
            neighbors = sf[sf.geometry.touches(row['geometry'])]['school_id'].tolist()
            neighbors = [str(int(row['school_id']))] + [str(int(x)) for x in neighbors]
            writer.writerow(neighbors)


if __name__ == "__main__":
    make_block_distance_matrix()
