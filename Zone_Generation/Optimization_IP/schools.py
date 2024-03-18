import pandas as pd
import ast
from Zone_Generation.Config.Constants import *

class Schools(object):
    def __init__(self, config):
        self.new_schools = config["new_schools"]
        self.capacity_scenario = config["capacity_scenario"]
        self.level = config["level"]
        self.include_k8 = config["include_k8"]

        if self.capacity_scenario not in ["Old", "A", "B", "C", "D"]:
            raise ValueError(
                f"Unrecognized capacity scenario {self.capacity_scenario}. Please use one of allowed capacity scenarios.")

    def load_school_data(self):
        # Load School Dataframe. Map School Name to AA number
        if self.new_schools:
            # school_df = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development.csv")
            school_df = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/schools_table_for_zone_development_updated.csv")
        else:
            school_df = pd.read_csv(f"~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv")

        school_df.rename(columns={"attendance_area": "attendance_area"}, inplace=True)

        print("self.level ", self.level)
        school_df[self.level] = school_df[self.level].astype('Int64')

        school_df = school_df.loc[school_df["category"] != "Citywide"]

        # Load School Capacities according to capacity policy
        school_df["K-8"] = school_df["school_id"].apply(lambda x: 1 if ((x in K8_SCHOOLS)) else 0)

        school_df = self._load_capacity(school_df)

        # school_df = self._compute_school_popularity(school_df)
        # school_df = self._inflate_capacity(school_df)
        school_df["num_schools"] = 1

        school_df.rename(
            columns={
                "eng_scores_1819": "english_score",
                "math_scores_1819": "math_score",
            }, inplace=True
        )
        return school_df

    def _load_capacity(self, school_df):
        # add on capacity
        programs = pd.read_csv("~/Dropbox/SFUSD/Data/Cleaned/stanford_capacities_12.23.21.csv")
        programs.rename(
            columns={
                "SchNum": "school_id",
                "PathwayCode": "program_type",
                f"Scenario_{self.capacity_scenario}_Capacity": "Capacity",
            }, inplace=True
        )

        prog_ge = programs.loc[programs["program_type"] == "GE"][["school_id", "Capacity"]]
        prog_ge.rename(columns={"Capacity": "ge_capacity"}, inplace=True)

        prog_all = programs[["school_id", "Capacity"]].rename(columns={"Capacity": "all_prog_capacity"})
        prog_all = prog_all.groupby("school_id", as_index=False).sum()

        school_df = school_df.merge(prog_all, how="inner", on="school_id")
        school_df = school_df.merge(prog_ge, how="inner", on="school_id")

        school_df = school_df.loc[school_df['ge_capacity'] > 0]

        if self.include_k8:
            school_df = school_df.loc[:, :]
        else:
            school_df = school_df.loc[school_df["K-8"] == 0]

        return school_df

    # Compute the popularity of each school, and add it as an additional column, ge_popularity
    # Popularity computation for each school s:
    # (Number of GE students that selected school s as top choice) / (real capacity of school s)
    def _compute_school_popularity(self, school_df):
        # Convert "r1_ranked_idschool" from string to list
        self.student_df['r1_ranked_idschool'] = self.student_df['r1_ranked_idschool'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

        # Extract top 1 school for each student and map it to its zone
        self.student_df['top1_school'] = self.student_df['r1_ranked_idschool'].apply(lambda x: x[0] if x else None)

        # Sum the ge_students values for each school ranked as top choice
        top1_school_weights = self.student_df.groupby('top1_school')['ge_students'].sum().reset_index()
        # rename the ge_demand column, and normalize the demand by the number of years.
        top1_school_weights["ge_student_demand"] = top1_school_weights["ge_students"].astype(float) / len(self.years)

        top1_school_weights["school_id"] = top1_school_weights["top1_school"].astype(int)
        top1_school_weights = top1_school_weights[['school_id', 'ge_student_demand']]

        # Merge this weighted sum with the school_df DataFrame
        school_df = school_df.merge(top1_school_weights, on='school_id', how='left')

        # For schools in Mission_Bay, there is no historical data, so manually inflate popularity
        school_df.loc[school_df['school_id'].isin(Mission_Bay), 'ge_student_demand'] = \
            school_df.loc[school_df['school_id'].isin(Mission_Bay), 'ge_capacity']

        # Calculate the weighted popularity of each school
        school_df['ge_popularity'] = school_df['ge_student_demand'] / school_df['ge_capacity']

        return school_df

    def _inflate_capacity(self, school_df):
        # Apply the function to calculate inflated_ge_capacity row by row, for each school
        school_df['inflated_ge_capacity'] = school_df.apply(self.calculate_inflated_capacity, axis=1)
        return school_df

    def calculate_inflated_capacity(self, row):
        decay_power = 1.2
        if row['ge_popularity'] > 1:
            return row['ge_capacity']
        elif 0.5 < row['ge_popularity'] <= 1:
            return row['ge_capacity'] *  (1/row['ge_popularity']) ** decay_power
        elif row['ge_popularity'] <= 0.5:
            return 2**decay_power * row['ge_capacity']
