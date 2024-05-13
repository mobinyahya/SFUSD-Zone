import pandas as pd
import ast
import random, gc, os, csv
import numpy as np

from Zone_Generation.Config.Constants import *

class Students(object):
    def __init__( self, config):
        self.drop_optout = config["drop_optout"]
        self.years = config["years"]
        self.population_type = config["population_type"]


    def load_student_data(self):

        cleaned_student_path = ("/Users/mobin/SFUSD/Data/Cleaned/Cleaned_Students_" +
                                '_'.join([str(year) for year in self.years]) + ".csv")
        if os.path.exists(cleaned_student_path):
            student_df = pd.read_csv(cleaned_student_path, low_memory=False)
            return student_df

        student_data_years = [0] * len(self.years)
        for i in range(len(self.years)):
            # load student data of year i
            student_data_years[i] = self._load_student_data(year=self.years[i])

        student_df = pd.concat(student_data_years, ignore_index=True)

        student_df.to_csv(cleaned_student_path, index=False)

        return student_df


    def _load_student_data(self, year):
        # get student data
        if self.drop_optout:
            if year == 19:
                print("WARNING: Due to limited data, switching to using student_1920.csv instead of "
                      "drop_optout_1920.csv and including all students instead of just enrolled.")
                student_data = pd.read_csv(
                    f"~/SFUSD/Data/Cleaned/student_1920.csv", low_memory=False)
                student_data = student_data.dropna(subset=["enrolled_idschool"])

            else:
                student_data = pd.read_csv(
                    # f"~/SFUSD/Data/Cleaned/drop_optout_{year}{year + 1}.csv", low_memory=False
                    f"~/SFUSD/Data/Cleaned/enrolled_{year}{year + 1}.csv", low_memory = False)
        else:
            student_data = pd.read_csv(
                f"~/SFUSD/Data/Cleaned/student_{year}{year + 1}.csv", low_memory=False)

        student_data = student_data.loc[student_data["grade"] == "KG"]
        student_data['resolved_ethnicity'] = student_data['resolved_ethnicity'].replace(ETHNICITY_DICT)
        student_data.rename(columns={"census_block": "Block", "census_blockgroup": "BlockGroup", "idschoolattendance": "attendance_area", "FRL Score":"FRL"}, inplace=True)
        student_data.dropna(subset=BUILDING_BLOCKS, inplace=True)


        if year in [21,22]:
            student_data["ell_count"] = student_data["englprof"].apply(lambda x: 1 if x in ["N", "L"] else 0)
        else:
            student_data["ell_count"] = student_data["englprof_desc"].apply(
                lambda x: 1 if (x == "N-Non English" or x == "L-Limited English") else 0
            )
        student_data["enrolled_students"] = np.where(student_data["enrolled_idschool"].isna(), 0, 1)



        student_data['r1_programs'] = student_data['r1_programs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        for rnd in range(2, 6):
            col_name = f"r{rnd}_programs"

            # Check if the column exists
            if col_name in student_data.columns:
                # Identify rows where r1_programs is empty and the current round column contains a list
                condition = student_data['r1_programs'].apply(lambda x: len(x) <= 0) & student_data[col_name].apply(lambda x: isinstance(x, list))

                # For rows meeting the condition, update r1_programs with the value from the current round column
                student_data.loc[condition, 'r1_programs'] = student_data.loc[condition, col_name]
        #
        student_data = student_data.loc[student_data['r1_programs'].apply(lambda x: x != [])]

        # Each student counts as a full all_prog_student
        # and a partial (between [0,1]) ge_student
        student_data["all_prog_students"] = 1
        student_data["ge_students"] = student_data.apply(
            lambda x: sum([i == "GE" for i in x["r1_programs"]]) / len(x["r1_programs"])
            if x["enrolled_students"] == 1 and len(x["r1_programs"]) > 0
            else 0,
            axis=1
        )

        student_data = pd.get_dummies(
            student_data, columns=["resolved_ethnicity"]
        )

        #  Make sure we only count the students statistics
        #  (i.e. frl, racial) if the student is GE
        for col in ETHNICITY_COLS + ['FRL']:
            student_data[col] = student_data.apply(
                lambda x: x[col] * x["ge_students"],
                axis=1
            )

        # # Load FRL
        # # TODO check if I need this new approach to compute FRL
        # cbeds = pd.read_excel(CBEDS_SBAC_PATH, sheet_name="CBEDS2122")
        # cbeds.rename(columns={'Geoid10': 'census_block'}, inplace=True)
        # cbeds['frl'] = cbeds["FRPM by block"]/cbeds["Distinct count of Student No"]
        # print("student_data.columns ", student_data.columns)
        # student_data = student_data.merge(cbeds[['census_block', 'frl']], how='left', on='census_block')

        student_data = self._make_program_types(student_data)
        student_data = self._filter_program_types(student_data, year)

        student_data = student_data[IMPORTANT_COLS + ETHNICITY_COLS]


        # Fill NaN values in columns with the mean value
        for col in ["FRL", "AALPI Score"]:
            mean_value = student_data[col].mean()
            student_data['FRL'].fillna(value=mean_value, inplace=True)

        # Fill NaN values in the remaining columns with 0
        student_data.fillna(value=0, inplace=True)

        student_data.rename(columns={"resolved_ethnicity_American Indian": "Ethnicity_American_Indian", "resolved_ethnicity_Asian": "Ethnicity_Asian",
                                     "resolved_ethnicity_Black or African American": "Ethnicity_Black_or_African_American", "resolved_ethnicity_Filipino": "Ethnicity_Filipino",
                                     "resolved_ethnicity_Pacific Islander": "Ethnicity_PacificIslander", "resolved_ethnicity_Hispanic/Latinx": "Ethnicity_Hispanic/Latinx",
                                     "resolved_ethnicity_Two or More Races": "Ethnicity_Two_or_More_Races", "resolved_ethnicity_White": "Ethnicity_White",
                                     }, inplace=True)

        return student_data

    def _make_program_types(self, df):
        # look at rounds 1-4, and all the programs listed in those rounds
        # make a new column program_types, which is a list of all such program types over different rounds
        """ create column with each type of program applied to """
        for round in range(1, 4):
            col = "r{}_programs".format(round)
            if col in df.columns:
                if round == 1:
                    df["program_types"] = df[col]
                else:
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
                    df["program_types"] = df["program_types"] + df[col]
        df["program_types"] = df["program_types"].apply(lambda x: np.unique(x))
        return df

    def _filter_program_types(self, df, year):
        """Identify students relevant to the program of interest"""
        if self.population_type == "SB":
            df["filter"] = df.apply(
                lambda row: 1
                if row["homelang_desc"] == "SP-Spanish" or "SB" in row["program_types"]
                else 0,
                axis=1,
            )
        elif self.population_type == "CB":
            df["filter"] = df.apply(
                lambda row: 1
                if row["homelang_desc"] == "CC-Chinese Cantonese"
                   or "CB" in row["program_types"]
                else 0,
                axis=1,
            )
        elif self.population_type == "GE":
            if year != 18:
                df['filter'] = df.apply(
                    lambda row: 1
                    if "GE" in row["program_types"]
                    else 0,
                    axis=1
                )
            else:
                def filter_students(row):
                    if (
                            row["r1_idschool"] == row["enrolled_idschool"]
                            and row["r1_programcode"] == "GE"
                    ):
                        return 1
                    elif (
                            row["r3_idschool"] == row["enrolled_idschool"]
                            and row["r3_programcode"] == "GE"
                    ):
                        return 1
                    elif (
                            row["r1_idschool"] != row["enrolled_idschool"]
                            and row["r3_idschool"] != row["enrolled_idschool"]
                    ):
                        return 1
                    else:
                        return 0

                df["filter"] = df.apply(filter_students, axis=1)
        # if program type is "All"
        elif self.population_type == "All":
            return df
        return df.loc[df["filter"] == 1]

