import os

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

ox.config(use_cache=True, log_console=True)

SLOW_STREET_PATH = "/Users/katherinementzer/Downloads/Slow_Streets.csv"
CENSUS_SHAPEFILE_PATH = "~/SFUSD/Census 2010_ Blocks for San Francisco/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp"
CENSUS_TRANSLATOR_PATH = "~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv"
SCHOOL_PATH = "/Users/katherinementzer/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv"


def load_slow_streets_data():
    slow_streets = gpd.read_file(SLOW_STREET_PATH, crs="epsg:4326")
    slow_streets["geometry"] = slow_streets["shape"].apply(wkt.loads)
    slow_streets = slow_streets.set_crs("epsg:4326")
    return slow_streets.to_crs("epsg:3857")


def load_census_block_group_shapefile():
    path = os.path.expanduser(CENSUS_SHAPEFILE_PATH)
    census = gpd.read_file(path)
    census["geoid10"] = census["geoid10"].fillna(value=0).astype("int64", copy=False)
    df = pd.read_csv(CENSUS_TRANSLATOR_PATH)
    df["Block"] = df["Block"].fillna(value=0).astype("int64", copy=False)
    census = census.merge(df, how="left", left_on="geoid10", right_on="Block")
    census = census.dissolve(by="BlockGroup", as_index=False)
    return census.to_crs("epsg:3857")


def load_school_data():
    sc_df = pd.read_csv(SCHOOL_PATH)
    geometry = [Point(xy) for xy in zip(sc_df["lon"], sc_df["lat"])]
    sc_df = gpd.GeoDataFrame(sc_df, crs="epsg:4326", geometry=geometry)
    return sc_df.to_crs("epsg:3857")


def get_walk_relevant_streets(streets, buffered_sch, school_id):
    school = buffered_sch.loc[buffered_sch.school_id == school_id]
    return gpd.sjoin(streets, school).drop(columns=["index_right"])


def get_walk_relevant_blocks(streets, buffered_sch, buffered_census, school_id):
    relevant_streets = get_walk_relevant_streets(streets, buffered_sch, school_id)
    return gpd.sjoin(buffered_census, relevant_streets).rename(
        columns={"BlockGroup_left": "BlockGroup"}
    )


def graph_path_cost(G, path):
    return sum([G[path[i]][path[i + 1]][0]["length"] for i in range(len(path) - 1)])


def get_close_blocks(G, end_node, relevant_blocks):
    close_blocks = []
    for idx, row in relevant_blocks.iterrows():
        start_node = ox.get_nearest_node(
            G, (float(row["block_lon"]), float(row["block_lat"]))
        )
        route = nx.shortest_path(G, start_node, end_node, weight="distance")
        if graph_path_cost(G, route) < 1609:
            close_blocks.append(row["BlockGroup"])
    return [int(x) for x in close_blocks]


def get_walking_distance_blocks(
    relevant_schools, streets, buffered_sch, buffered_census, G
):
    walk_distance_blocks = {}
    for idx, row in relevant_schools.iterrows():
        school_relevant_blocks = get_walk_relevant_blocks(
            streets, buffered_sch, buffered_census, row.school_id
        )
        end_node = ox.get_nearest_node(G, (float(row["lon"]), float(row["lat"])))
        close_blocks = get_close_blocks(G, end_node, school_relevant_blocks)
        walk_distance_blocks[row["school_id"]] = close_blocks
