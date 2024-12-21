import os
import csv
import yaml
import pandas as pd
from Zone_Generation.Optimization_IP.design_zones import *
from Zone_Generation.Config.Constants import *


class Stat_Class(object):
    def __init__(self, dz, config):
        self.level = config["level"]
        self.years = config["years"]

        self.dz = dz
        self.area2idx = dz.area2idx
        self.idx2area = dz.idx2area
        self.neighbors = dz.neighbors
        self.schools = self.dz.school_df
        self.students = self.dz.student_df

        self.students23 = pd.read_csv("/Users/mobin/SFUSD/Data/Cleaned/Cleaned_Students_22.csv", low_memory=False)


    def compute_zone_metrics(self, zone):
        # compute metrics for each individual zone
        zone_metrics = {}

        if self.level == "attendance_area":
            zone = [u for u in zone if u in self.dz.sch2area]
        if self.dz.capacity_scenario == "Closure":
            student_type = "all_students"
            seat_type = "all_seats"
        else:
            student_type = "ge_students"
            seat_type = "ge_seats"


        zone_metrics[student_type] = sum([self.dz.studentsInArea[self.area2idx[j]] for j in zone])

        for ethnicity in AREA_ETHNICITIES:
            zone_metrics[ethnicity] = sum([self.dz.area_data[ethnicity][self.area2idx[j]] for j in zone]) / zone_metrics[student_type]

        zone_metrics["frl%"] = sum([self.dz.area_data["FRL"][self.area2idx[j]] for j in zone]) / zone_metrics[student_type]

        zone_metrics[seat_type] = sum([self.dz.seats[self.area2idx[j]] for j in zone])
        zone_metrics["seats shortage/overage"] = abs(zone_metrics[student_type] - zone_metrics[seat_type])
        zone_metrics["shortage%"] = (zone_metrics[student_type] - zone_metrics[seat_type])/zone_metrics[student_type]
        zone_metrics["school_count"] = sum([self.dz.schools[self.area2idx[j]] for j in zone])

        schools = self.dz.school_df
        sch_dict = schools.set_index('school_id')["school_name"].to_dict()
        zone_metrics["schools_names"] = []
        for sch_id in sch_dict:
            if self.level == "attendance_area":
                sch_area = self.dz.sch2area[sch_id]
            elif self.level == "BlockGroup":
                sch_area = self.dz.sch2area[sch_id]
            elif self.level == "Block":
                sch_area = self.dz.sch2area[sch_id]
            if sch_area in zone:
                zone_metrics["schools_names"].append(sch_dict[sch_id])


        return zone_metrics

    # Step 3: Find the top-ranked school in the same zone for each student
    def find_top_school_in_zone(self, ranked_schools, student_zone):
        for school_id in ranked_schools:
            if self.school_zone.get(school_id) == student_zone:
                return school_id
        return None  # Return None or an appropriate value if no school matches

    # For each student, consider the top school, from student x's
    # round 1 ranking, in which that school is in the same zone as the student x.
    # Compute the total demand for each school, using withing-zone-top-choices across all students.
    # In each zone, compute the total student demand that is not fulfilled (from students top 1 within zone choices).
    # topchoice_shortage for each zone: Unfulfilled demand / total students.
    def compute_topchoice_shortage(self):

        # Apply the function to each student row
        self.students['top_school_in_zone'] = self.students.apply(
            lambda row: self.find_top_school_in_zone(row['r1_ranked_idschool'], row['zone ID']), axis=1
        )

        valid_students = self.students[["ge_students", "top_school_in_zone", "zone ID"]].dropna()

        merged_df = pd.merge(valid_students, self.schools, left_on='top_school_in_zone', right_on='school_id')

        # Proceed with grouping by schoo_id and summing 'ge_students'
        school_capacity_usage = merged_df.groupby('school_id').size().reset_index(name='ranked1_students')
        # noramlize the requests by number of year
        school_capacity_usage['ranked1_students'] = school_capacity_usage['ranked1_students'] / len(self.years)

        # Merging the calculated sum back with the original schools dataframe to have the capacities
        schools_df = pd.merge(self.schools, school_capacity_usage, left_on='school_id', right_on='school_id', how='left')

        schools_df['ranked1_students'].fillna(0, inplace=True)
        schools_df['rank1_shortage'] = schools_df.apply(lambda row: max(row['ranked1_students'] - row['ge_capacity'], 0), axis=1)
        # shortage for each zone is the number of students within that zone that didn't get matched to their top school.
        zone_rank1_shortage = schools_df.groupby("zone ID").sum()[["rank1_shortage"]].reset_index()
        zone_rank1_shortage["zone ID"] = zone_rank1_shortage["zone ID"].apply(lambda x: "Zone " + str(int(x)))

        return zone_rank1_shortage


    # Compute percentage of students in each zone that ranked the
    # school within that zone as their top school in r1_ranked_idschool
    def compute_listed_rank1(self):
        self.schools["zone ID"] = self.schools[self.level].replace(self.dz.zone_dict)
        self.students["zone ID"] = self.students[self.level].replace(self.dz.zone_dict)

        self.school_zone = self.schools.set_index('school_id')["zone ID"].to_dict()

        # Extract top 1 school for each student and map it to its zone
        self.students['top1_school_zone'] = self.students['top1_school'].apply(lambda x: self.school_zone[x] if x in self.school_zone else None)

        # Non-GE schools are not inclduded in school_zone dictionary.
        # Hence, for every student with top1 school that is non-GE, top1_school_zone is NaN
        self.students.dropna(subset=['top1_school_zone'], inplace=True)


        # Check if student's zone matches top 1 school's zone
        self.students['zone_match'] = self.students['zone ID'] == self.students['top1_school_zone']


        # Group by zone and calculate weighted sums based on 'zone_match' and total 'ge_students'
        zone_match_sum = self.students[self.students['zone_match']].groupby('zone ID')['ge_students'].sum()
        total_ge_students = self.students.groupby('zone ID')['ge_students'].sum()

        # Calculate the percentage for each zone
        percentage_zone_match = (zone_match_sum / total_ge_students) * 100

        # Convert to DataFrame for better readability
        percentage_zone_match = percentage_zone_match.reset_index().rename(columns={'ge_students': 'listed_rank_1%'})
        percentage_zone_match["zone ID"] = percentage_zone_match["zone ID"].apply(lambda x: "Zone " + str(int(x)))

        return percentage_zone_match

    # Given historical choice, some subset of students rank their attendance area school first.
    # What is the percentage of such students that do not get assigned to their attendance area school.
    def compute_real_rank1_assignment(self):
        # students in which their top school is in their zone
        zone_preferring_students = self.students[self.students['zone_match']==1]

        total_students_per_zone = zone_preferring_students.groupby('zone ID')['studentno'].count().reset_index(name='Total_Students')
        rank1_assigned_students_per_zone = zone_preferring_students[zone_preferring_students['r1_rank'] == 1].groupby('zone ID')['studentno'].count().reset_index(name='rank1_assigned')

        zone_stats = pd.merge(total_students_per_zone, rank1_assigned_students_per_zone, on='zone ID')

        # Calculate the percentage of students in each zone who got their top choice
        zone_stats['real_rank1_unassigned%'] = (1 - zone_stats['rank1_assigned'] / zone_stats['Total_Students']) * 100
        zone_stats["zone ID"] = zone_stats["zone ID"].apply(lambda x: "Zone " + str(int(x)))

        return zone_stats

    def concat_summaries(self, df, summary):
        # Create an empty row with the same columns as the DataFrame
        empty_row = pd.Series(dtype='object')
        empty_row = empty_row.reindex(df.columns)

        # Append the empty row to the DataFrame
        df = df.append(empty_row, ignore_index=True)

        # Convert the dictionary to a DataFrame
        df_summary = pd.DataFrame(summary.items(), columns=['zone ID', 'frl%'])

        # Concatenate the two DataFrames vertically
        result_df = pd.concat([df, df_summary], ignore_index=True)
        return result_df

    def compute_assignment_metrics(self, assignments):
        # TODO only count GE students proportion and not all students
        student_stats = self.students23.merge(assignments, how='inner', on="studentno")
        student_stats['zone ID'] = student_stats[self.dz.level].replace(self.dz.zone_dict)

        # Filter rows where 'rank' is 1
        rank_1_matches = student_stats[student_stats['rank'] == 1]

        # Group by 'zone ID' and count rank 1 matches
        rank_1_counts = rank_1_matches.groupby('zone ID').size().rename('assigned_rank_1_count')

        # Group student_stats by 'zone ID' and count total students in each zone
        total_counts = student_stats.groupby('zone ID').size().rename('total_count')

        # Step 4: Merge the two Series into a DataFrame on 'zone ID'
        merge_df = pd.merge(rank_1_counts, total_counts, left_index=True, right_index=True)

        # Calculate the percentage of students matched to their rank 1 school within each zone
        merge_df['assigned_rank_1%'] = (merge_df['assigned_rank_1_count'] / merge_df['total_count']) * 100

        # Reset the index to turn 'zone ID' back into a column
        assignment_metrics = merge_df.reset_index()[['zone ID', 'assigned_rank_1%']]

        # rename the zone ID column values in assignment_metrics to match the entries of zone_metrics
        assignment_metrics["zone ID"] = assignment_metrics["zone ID"].apply(lambda x: "Zone " + str(int(x)))

        return assignment_metrics

    def compute_metrics(self, assignments = None):
        zone_metrics_df = pd.DataFrame(columns = ["zone ID"])
        for zone in self.dz.zone_lists:

            zone_metrics = self.compute_zone_metrics(zone)
            zone_metrics["zone ID"] = "Zone " + str(self.dz.zone_lists.index(zone))
            zone_metrics_df = zone_metrics_df.append(zone_metrics, ignore_index=True)

        # student_metrics = self.compute_listed_rank1()
        # zone_metrics_df = zone_metrics_df.merge(student_metrics, how='inner', on="zone ID")


        # student_shortage = self.compute_topchoice_shortage()
        # zone_metrics_df = zone_metrics_df.merge(student_shortage, how='inner', on="zone ID")
        # zone_metrics_df["rank1_shortage%"] = zone_metrics_df["rank1_shortage"] / zone_metrics_df["ge_students"]
        # zone_metrics_df.drop(["rank1_shortage"], axis='columns', inplace=True)

        # real_rank1_assignments = self.compute_real_rank1_assignment()
        # zone_metrics_df = zone_metrics_df.merge(real_rank1_assignments, how='inner', on="zone ID")

        if assignments is not None:
            assignment_metrics = self.compute_assignment_metrics(assignments)
            zone_metrics_df = zone_metrics_df.merge(assignment_metrics, how='inner', on="zone ID")

        print("zone_metrics_df ", zone_metrics_df)

        # Calculate the average values for each column
        avg_values = zone_metrics_df.mean()

        #change the definition of average frl_percentage
        avg_values["frl%"] = self.dz.F

        for ethnicity in AREA_ETHNICITIES:
            avg_values[ethnicity] = sum([self.dz.area_data[ethnicity][j] for j in range(self.dz.A)]) / self.dz.N

        # Calculate the maximum deviation for each column from the average
        max_deviations = zone_metrics_df.subtract(avg_values).abs().max()


        # Rename the zone ID name for average row
        max_deviations["zone ID"] = "Max Deviation"
        # Rename the zone ID name for average row
        avg_values["zone ID"] = "Average Value"

        summary={}
        summary["Max seats shortage/overage"] = zone_metrics_df["seats shortage/overage"].max()
        summary["Max shortage %"] = zone_metrics_df["shortage%"].max()
        summary["Max frl% deviation"] = max_deviations["frl%"]

        # Add the average values as a new row to the DataFrame
        zone_metrics_df = zone_metrics_df.append(avg_values, ignore_index=True)

        # Add a new row with the maximum deviations to the DataFrame
        zone_metrics_df = zone_metrics_df.append(max_deviations, ignore_index=True)


        result_df = self.concat_summaries(zone_metrics_df, summary)

        acceptable_zone = True
        if (summary["Max shortage %"] > 0.32) | (summary["Max frl% deviation"] > 0.35) | (max_deviations["school_count"] > 3):
            acceptable_zone = False

        return result_df, acceptable_zone

    def load_zones_from_pkl(self, dz, file_path):
        zone_lists = []
        with open(file_path, 'rb') as file:
            zone_dict = pickle.load(file)
            # zone_dict = {key: all_schools.index(value) for key, value in zd.items()}

        # for z in range(len(all_schools)):
        zone_ids =  sorted(set(zone_dict.values()))
        M = len(zone_ids)
        for z in range(M):
            zone_z = []
            for area in zone_dict:
                if zone_dict[area] == zone_ids[z]:
                    zone_z.append(area)

            zone_lists.append(zone_z)

        print("zone_lists = ", zone_lists)
        print("zone_dict = ", zone_dict)
        # filename = file_path.replace(".pkl", ".csv")
        # print("filename", filename)
        # # save zones themselves
        # if "5-zone-2x_B" in filename:
        #     with open(filename, "w") as outFile:
        #         writer = csv.writer(outFile, lineterminator="\n")
        #         for z in zone_lists:
        #             writer.writerow(z)
        #         print("savingxx")

        # print("zone_dict ", zone_dict)

        # zv = ZoneVisualizer(config["level"])
        # zv.zones_from_dict(zone_dict)

        return zone_lists, zone_dict


if __name__ == "__main__":
    with open("../Config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mcmc_filter_mode = False

    dz = DesignZones(
        config=config,
    )

    input_folder = "/Users/mobin/Documents/sfusd/local_runs/Zones/School_Closure Candidates -all programs/13,14-zones/FRL < 15%"
    # input_folder = "/Users/mobin/Documents/sfusd/local_runs/Zones/Zones03-15"


    # files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]

    if config["level"] == 'Block':
        zoning_files = [csv_file for csv_file in files if "B" in csv_file and "BG" not in csv_file]
    elif config["level"] == 'BlockGroup':
        zoning_files = [csv_file for csv_file in files]
        zoning_files = [csv_file for csv_file in files if "BlockGroup" in csv_file]
    elif config["level"] == 'attendance_area':
        zoning_files = [csv_file for csv_file in files if "AA" in csv_file]

    zoning_files = [csv_file for csv_file in zoning_files if "Stats" not in csv_file
                    and "Assignment" not in csv_file]
    print("zoning_files   ", zoning_files)

    for zoning_file in zoning_files:
        Stat = Stat_Class(dz, config)
        print("file: " + str(zoning_file))

        # zone_lists, zone_dict = load_zones_from_file(file_path=os.path.join(input_folder, zoning_file))
        zone_lists, zone_dict = Stat.load_zones_from_pkl(dz, file_path=os.path.join(input_folder, zoning_file))

        Stat.dz.zone_lists = zone_lists
        Stat.dz.zone_dict = zone_dict
        # assignments = pd.read_csv(os.path.join(input_folder, zoning_file.replace(".csv", "")  + "_Assignment.csv"), low_memory=False)

        # print("zone_list  " + str(zone_lists))
        df = Stat.dz.school_df
        df['new_name'] = df['school_id'].astype(str) + " " + df['school_name']

        # Create 'Zone_ID' column using the map method
        df['Zone_ID'] = df['BlockGroup'].map(zone_dict)
        # df = df[["new_name", "Zone_ID", ]]
        # print(df)
        # base_name, _ = os.path.splitext(zoning_file)
        # output_file_path = os.path.join(input_folder, base_name + "_schools.csv")
        # # save the stats data into file
        # df.to_csv(output_file_path, index=False)
        # continue

        if len(zone_lists)<=1:
            continue

        # df, acceptable_zone = Stat.compute_metrics(assignments)
        df, acceptable_zone = Stat.compute_metrics()

        # Save the DataFrame as a CSV file in the same folder with "_stats" appended
        base_name, _ = os.path.splitext(zoning_file)
        if mcmc_filter_mode:
            if acceptable_zone:
                output_file_path = os.path.join(input_folder + "/filter", base_name + "_Stats.csv")
                #save the stats data into file
                df.to_csv(output_file_path, index=False)
        else:
            output_file_path = os.path.join(input_folder, base_name + "_Stats.csv")
            #save the stats data into file
            df.to_csv(output_file_path, index=False)
