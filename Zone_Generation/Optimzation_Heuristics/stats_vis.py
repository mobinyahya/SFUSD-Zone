import os
import yaml
import pandas as pd
from Zone_Generation.Optimization_IP.generate_zones import *
from Graphic_Visualization.zone_viz import ZoneVisualizer
from Zone_Generation.Optimzation_Heuristics.stats_report import Stat_Class

with open("../Config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


input_folder = "/Users/mobin/Documents/sfusd/local_runs/Zones/Zones02-16"
zoning_file = "/Users/mobin/Documents/sfusd/local_runs/Zones/Final_Zones/6-zone-9_2_Old_BG"
assignment_file = zoning_file + "_Assignment"


# zoning_file = zoning_file.replace(".csv", "")
# assignments = pd.read_csv(os.path.join(input_folder, assignment_file + ".csv"), low_memory=False)

if "AA" in zoning_file:
    input_level = "attendance_area"
else:
    input_level = 'BlockGroup'

dz = DesignZones(
    config=config,
    level=input_level,
)

zv = ZoneVisualizer(input_level)

area_population = dz.area_data[["BlockGroup", "ge_students"]]
school_df = dz.school_df[["lat", "lon", "ge_capacity", "ge_popularity"]].loc[dz.school_df['K-8'] == 0]


# # Visualize the population heatmap accross SF and
# # schools with their corresponding capacities or popularities
# zv.population_heatmap(area_population, school_df, metric="ge_capacity", save_path="Capacity_Population")
# zv.population_heatmap(area_population, school_df, metric="ge_popularity", save_path="Popularity_Population")

Stat = Stat_Class(dz, input_level)


# Load zone lists and dictionaries from saved zoning file
zone_lists, zone_dict = load_zones_from_file(file_path = zoning_file + ".csv")
Stat.dz.zone_lists = zone_lists
Stat.dz.zone_dict = zone_dict
stat_df, acceptable_zone = Stat.compute_metrics()
# stat_df, acceptable_zone = Stat.compute_metrics(assignments)


# save the stats data into a csv file
output_file_path = zoning_file + "_Stats.csv"
stat_df.to_csv(output_file_path, index=False)

# Keep only the rows where the "zone ID" column contains the string "Zone"
stat_df = stat_df[stat_df['zone ID'].str.contains("Zone", na=False)]
stat_df["zone ID"] = stat_df['zone ID'].str.split().str.get(1).astype(int)

# zv.zone_stats_heatmap("shortage%", zone_dict, stat_df, metric_min=-0.55, metric_max=0.55, save_name=zoning_file + "_Shortage")
# zv.zone_stats_heatmap("real_rank1_unassigned%", zone_dict, stat_df, metric_min=0, metric_max=100, save_path=zoning_file + "_Unassigned_Real_Rank1")
# zv.zone_stats_heatmap("rank1_shortage%", zone_dict, stat_df, metric_min=0, metric_max=0.6, save_path=zoning_file + "_Rank1_Shortage_"+ str(dz.years[0]))
# zv.zone_stats_heatmap("assigned_rank_1%", zone_dict, stat_df, metric_min=5, metric_max=80, save_path=zoning_file + "_Assigned_Rank1")
zv.zone_stats_heatmap("listed_rank_1%", zone_dict, stat_df, metric_min=10, metric_max=85, save_path=zoning_file + "_Listed_Rank1")
