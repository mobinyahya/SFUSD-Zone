import os

ETHNICITY_DICT = {
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

ETHNICITY_COLS = [
    "resolved_ethnicity_American Indian",
    "resolved_ethnicity_Asian",
    "resolved_ethnicity_Black or African American",
    "resolved_ethnicity_Filipino",
    "resolved_ethnicity_Hispanic/Latinx",
    "resolved_ethnicity_Pacific Islander",
    "resolved_ethnicity_Two or More Races",
    "resolved_ethnicity_White"]

AREA_COLS = [
    'AALPI Score',
    'FRL',
    'ge_students',
    'all_prog_students'
]
OLD_COLS = [
    "Unnamed: 0",
    "studentno",
    "randomnumber",
    "requestprogramdesignation",
    "latitude",
    "longitude",
    "r1_idschool",
    "r1_rank",
    "r1_isdesignation",
    "r1_distance",
    "ctip1",
    "r3_idschool",
    "r3_rank",
    "r3_isdesignation",
    "r3_distance",
    "zipcode",
    "disability",
    "enrolled_idschool",
    "math_scalescore",
    "ela_scalescore",
    "r2_idschool",
    "final_school",
    "num_ranked",
    "census_block",
    "filter",
    "sped",
    "ell",
    "r2_rank",
    "r2_isdesignation",
    "r2_distance",
    "trans_sped"
]

BUILDING_BLOCKS = [
    "Block",
    "BlockGroup",
    "attendance_area",
]

# columns that are important, and we would
# save them for the filtered student data
IMPORTANT_COLS = [
    # "r1_idschool",
    # "r1_rank",
    "studentno",
    "AALPI Score",
    "FRL",
    "ge_students",
    "all_prog_students",
    "r1_ranked_idschool",
    "r1_rank"
    # "",
    # "",
    # "",
] + BUILDING_BLOCKS

AUX_BG = [60759804011000, 60759804011001, 60759804011002, 60759804011003]
K8_SCHOOLS = [618, 485, 676, 449, 479, 760, 796, 493]



SF_Montessori = 814

Mission_Bay = [999, 909]
CBEDS_SBAC_PATH = os.path.expanduser("~/SFUSD/Data/SFUSD_Demographics_SBAC_byblock.xlsx")
# I have a set of zones. These zones are stored in two formats:
# 1- a dicrtionary zone_dict that maps each area to a zone number.
# 2 a list of lists called zone_lists. Such that zone_lists[i] is a list of areas that are assigned to zone number i.

SUFFIX = {
    "attendance_area": "AA",
    "BlockGroup": "BG",
    "Block": "B"
}