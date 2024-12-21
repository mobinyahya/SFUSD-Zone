import sys
import pandas as pd
sys.path.append('../..')
sys.path.append('../../summary_statistics')
# import pickle
# from Zone_Generation.Optimization_IP.design_zones import DesignZones,  Compute_Name
# from Zone_Generation.Config.Constants import *

# student_programs_dict1 = {
#     # student: [f"{school}-{program}" for school, program in zip(str(schools)[1:-1].split(', '), str(programs)[1:-1].split(', '))]
#     student: [f"{school}-{program}-KG" for school, program in zip(str(schools)[1:-1].split(', '), str(programs)[1:-1].split(', '))]
#     for student, schools, programs in zip(students['studentno'], students['r1_ranked_idschool'], students['r1_programs'])
# }
#
# student_programs_dict = {}
# for key, value in student_programs_dict1.items():
#     programs_list = []
#     for p in value:
#         x = p.replace("'", "")
#         programs_list.append(x)
#     student_programs_dict[key] = programs_list







student_file_path = '/Users/mobin/SFUSD/Data/Cleaned/student_2324.csv'
program_file_path = '/Users/mobin/SFUSD/Data/Cleaned/programs_2223_facility_capicity-1.csv'
estimate_file_path = '/Users/mobin/Dropbox/SFUSD/Choice_Model/estimates_for_2324_withMissionBay/estimates.csv'


# Load the CSV file into a DataFrame
students_all = pd.read_csv(student_file_path)
programs = pd.read_csv(program_file_path)
estimates = pd.read_csv(estimate_file_path)


special_programs = ["AF", "DA", "DT", "ED", "MM", "MS", "SA", "TC", "AO"]
students_all["is_special"] = students_all["r1_programs"].apply(
    lambda x: (
        False
        if str(x) == "nan"
        else len(set(eval(x)).intersection(special_programs)) > 0
    )
)
students_all = students_all[students_all["is_special"] == 0].drop(
    columns=["is_special"]
)

students_all = students_all[
     ~students_all["r1_ranked_idschool"].isna()
]  # Filter round 1 students only
students_all["Assigned"] = False


programs = programs.dropna(subset=['capacity'])
prog2capacity = pd.Series(programs.capacity.astype(int).values, index=programs.program_id).to_dict()

print("len(students_all)", len(students_all))

students = students_all[students_all["grade"] == "KG"]
students = students.dropna(subset=['idschoolattendance'])
student2idschoolattendance = pd.Series(students.idschoolattendance.astype(int).values, index=students.studentno).to_dict()
student2AAprog = {}

for student in student2idschoolattendance:
    student2AAprog[student] = str(int(student2idschoolattendance[student])) + "-GE-KG"

# print(student2prog)
# exit()


# Remove the prefix '2223-' from the 'studentno' column
estimates['studentno'] = estimates['studentno'].astype(str).str.replace('2324-', '')
estimates = estimates.apply(pd.to_numeric, errors='coerce')

# Create an empty dictionary to store the results
student_prog_dict = {}
# Iterate over each row (student)
for index, row in estimates.iterrows():
    student = int(row['studentno'])  # Use the 'studentno' column value
    # Sort the values for the current student in descending order
    sorted_values = row.drop('studentno').sort_values(ascending=False, na_position='last')
    # Get the top 3 programs for this student
    top_programs = sorted_values.index.tolist()[:1]
    # Store the top programs in the dictionary
    student_prog_dict[student] = top_programs
# print(student_prog_dict)
counter = 0
matched_students = []
for student in student_prog_dict:
    if student not in student2AAprog:
        # print("student ", student)
        counter+=1
        continue
    if student2AAprog[student] in student_prog_dict[student]:

        prog = student2AAprog[student]
        prog2capacity[prog] = prog2capacity[prog] - 1
        students.loc[students['studentno'] == student, 'Assigned'] = True
        student_row = {
            'studentno': student,
            'programno': 0,
            'programcodes': student2AAprog[student],
            'rank': student_prog_dict[student].index(student2AAprog[student]) + 1,
            'designation': 0,
            'assigned_utility': estimates.loc[estimates['studentno'] == student, student2AAprog[student]].values[0],
            'In-Zone Rank': 1
        }
        matched_students.append(student_row)
# Converting the list of dictionaries to a DataFrame
matched_students_df = pd.DataFrame(matched_students)
print("matched_students_df", matched_students_df)

print("counter ", counter)

print(students[["studentno", "Assigned"]])
students2assigned = pd.Series(students.Assigned.astype(int).values, index=students.studentno).to_dict()

students_all['Assigned'] = students_all['studentno'].map(students2assigned).combine_first(students_all['Assigned'])
matched_students = students_all[students_all['Assigned'] == True]
print("len(students2assigned) ", len(students2assigned))
print("len(matched_students)", len(matched_students))
students_all = students_all[students_all['Assigned'] == False]
print("len(students_all)", len(students_all))

programs['capacity'] = programs['program_id'].map(prog2capacity)


students_all.to_csv('updated_students.csv', index=False)
programs.to_csv('updated_programs.csv', index=False)


assignment_file_path = '/Users/mobin/Documents/sfusd/local_runs/assignments/optional_choice/simulation2_2324_optional_choice_choice_model_real_match_afterRemoveTop1AA.csv'
assignments = pd.read_csv(assignment_file_path)

# print(assignments.columns)
# exit()
# Adding new rows to the existing DataFrame
assignments = pd.concat([assignments, matched_students_df], ignore_index=True)

assignments.to_csv('updated_assignments.csv', index=False)
