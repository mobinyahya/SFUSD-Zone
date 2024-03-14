area_data_columns = {
  "json description":
          "In our area_data dataframe, each row is a different area.\nYou can access information of column X for area with index j, as self.area_data[X][j]\nHere is an example:\nNumber of ge students in area j is: self.area_data[\"ge_students\"][j]. Here are the set of columns in self.area_data:",
  "studentno": {
    "AALPI Score": "int",
    "Values": "8 or 9 digit integer",
    "Description": "student number, the leading ‘S’ truncated and changed to an integer",
    "Notes": "Changed to int for faster indexing"
  },
  "FRL": {
    "DataType": "int",
    "Values": "",
    "Description": "",
    "Notes": ""
  },
  "ge_students": {
    "DataType": "int",
    "Values": "Number of GE (General Education) students in each area",
    "Description": "List of ints, the corresponding rank of each ranked school in ‘r1_ranked_idschool’"
  },
  "all_prog_students": {
    "DataType": "int",
    "Values": "Total number of students (across all programs including GE, Special Education, etc) in each area",
    "Description": "",
    "Notes": ""
  },
  "Block": {
    "DataType": "int",
    "Values": "Census Block number each area",
    "Description": "",
    "Notes": ""
  },
  "BlockGroup": {
    "DataType": "int",
    "Values": "Census BlockGroup number for each area",
    "Description": "",
    "Notes": ""
  },
  "attendance_area": {
    "DataType": "int",
    "Values": "Attendance Area number for each area",
    "Description": "",
    "Notes": ""
  },
  "eng_scores_1819": {
    "DataType": "int",
    "Values": "",
    "Description": "",
    "Notes": ""
  },
  "math_scores_1819": {
    "DataType": "int",
    "Values": "",
    "Description": "",
    "Notes": ""
  },
  "greatschools_rating": {
    "DataType": "int",
    "Values": "",
    "Description": "",
    "Notes": ""
  },
  "MetStandards": {
    "DataType": "int",
    "Values": "",
    "Description": "",
    "Notes": ""
  },
  "AvgColorIndex": {
    "DataType": "int",
    "Values": "",
    "Description": "",
    "Notes": ""
  },
  "all_prog_capacity": {
    "DataType": "int",
    "Values": "Total Number of seats in each area. (We precomputed total number schools in each area, the summed the total number seats across all those schools, for different program types)",
    "Description": "",
    "Notes": ""
  },
  "ge_capacity": {
    "DataType": "int",
    "Values": "",
    "Description": "Total Number of GE seats in each area. (We precomputed total number schools in each area, the summed the total number of GE seats across all those schools)",
    "Notes": ""
  },
  "num_schools": {
    "DataType": "int",
    "Values": "",
    "Description": "Total Number of Schools in the area",
    "Notes": ""
  }
}
