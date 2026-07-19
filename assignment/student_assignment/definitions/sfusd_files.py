import os

import pandas as pd

# FILES RELATIVE TO SFUSD/
CLEANED_PATH = "Data/Cleaned/"
PRECOMPUTED_PATH = "Data/Precomputed/"
# Files in the root Data directory
PROGRAM_CODES_FILE = "Data/program_codes.csv"
BLOCK_DATA_FILE = "Data/SF 2010 blks 022119 with field descriptions (1).xlsx"
ASSIGNMENT_FILE_FORMAT = "Assignment_CTIP{}_round_merged{}_policy{}_ties{}_prefExtend{}_iteration{}.csv"

# Files in the Cleaned directory
SCHOOL_DATA_FILE = CLEANED_PATH + "schools_rehauled_{}{}{}.csv"
PROGRAM_DATA_FILE = CLEANED_PATH + "programs_{}{}{}.csv"
# PROGRAM_DATA_FILE_NOTKG = CLEANED_PATH + 'programs_{}_{}{}.csv'
# STUDENT_DATA_FILE = CLEANED_PATH + "drop_optout_{}{}.csv"
STUDENT_DATA_FILE = CLEANED_PATH + "student_{}{}.csv"
MET_DATA_FILE = CLEANED_PATH + "schools_rehauled_{}{}.csv"
# Files in the Precomputed directory
DISTANCE_DATA_FILE = PRECOMPUTED_PATH + "student_program_distances_{}{}.csv"
CBEDS_FILE = "Data/Student Location Data/out_SFUSD20191220_1_1_cbeds20{}.dta"
CENSUS_BLOCK_FILE = "Census 2010_ Blocks for San Francisco/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp"


# FILES NOT RELATIVE TO SFUSD/
ZONE_FILE = "~/Dropbox/SFUSD/Optimization/Zones/citywideGE_optGE862432.txt"
BLOCK_TRANSLATOR_FILE = (
    "~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv"
)


# DYNAMICALLY CALCULATED PATHS
def zone_paths(set_number):
    if set_number == 0:
        paths = [
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE646249.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1430116.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1240737.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE340634.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1018246.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1630917.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1364825.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1423449.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE922908.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE341062.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1904536.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1738241.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE657169.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE927435.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE780873.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE927380.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE767164.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1553780.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE103341.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1974316.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1739835.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1492775.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1004696.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1704622.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE932414.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE691953.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1126877.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1607021.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE916949.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1465277.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1559160.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1860225.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE247894.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1987612.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE206230.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1798449.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE269243.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1377364.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1274684.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE287353.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE272594.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1675053.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE169502.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE713813.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE658904.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1234089.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE688135.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1524473.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1729507.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1944091.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1465011.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE601328.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1266888.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE774417.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1319096.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE4506.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE130080.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE348417.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE956602.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE587507.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE116961.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1990365.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE333452.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1724377.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1671404.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1323765.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1390047.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1730029.csv",
        ]

    if set_number == 1:
        paths = [
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1389929.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1630917.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE253769.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1156692.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE922908.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE341062.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1904536.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1738241.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE657169.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE927435.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE780873.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE927380.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE767164.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1553780.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE103341.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1974316.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1739835.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1492775.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1004696.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1704622.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE932414.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE691953.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1126877.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE1607021.csv",
            "~/Dropbox/SFUSD/Data/Computed/Itai/optGE916949.csv",
        ]

    if set_number == 2:
        df18 = pd.read_csv(
            "~/Dropbox/SFUSD/Optimization/Zones/best_zones_sept23_nodup.csv"
        )
        files1 = [
            f"~/Dropbox/SFUSD/Data/Computed/Itai/{x}.csv"
            for x in df18["zone_file"]
            if x[:5] == "optGE"
        ]
        files2 = [
            f"~/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/{x}.csv"
            for x in df18["zone_file"]
            if x[:7] == "zoneopt"
        ]
        files3 = [
            f"~/Dropbox/SFUSD/Optimization/Zones/Zones_Sept9/{x}.csv"
            for x in df18["zone_file"]
            if x[:7] == "optzone"
        ]
        paths = [os.path.expanduser(x) for x in files1 + files2 + files3]

    if set_number == 3:
        paths = [
            "optGE35532",
            "optGE246024",
            "optGE678810",
            "optGE890088",
            "optGE1323765",
            "optGE2402784",
            "optGE3530403",
        ]
        for i in range(len(paths)):
            paths[i] = (
                "~/Dropbox/SFUSD/Data/Computed/Itai/selected_zones/"
                + paths[i]
                + ".csv"
            )

    return paths
