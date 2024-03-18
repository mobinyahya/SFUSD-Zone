import requests
import os
from config import ANTHROPIC_API_KEY



# Set up the API endpoint and headers
api_endpoint = "https://api.anthropic.com/v1/complete"
api_key = ANTHROPIC_API_KEY
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Set up the file paths
file_paths = [
    "/Users/mobin/PycharmProjects/SFUSD-Zone/Zone_Generation/Optimization_IP/integer_program.py",
    "/Users/mobin/PycharmProjects/SFUSD-Zone/Zone_Generation/Data_Description/area_data.txt"
]

# Read the contents of the files
file_contents = []
for file_path in file_paths:
    with open(file_path, "r") as file:
        file_contents.append(file.read())




# Prepare the API request payload
prompt = f"""
Here are the files:

{file_contents[0]}

Project Description: 
In partnership with SFUSD (San Francisco Unified School District), I've developed a solution,
to find a zoning system for schools. Where we divide the city of San Francisco, into a number of zones (i.e. a number between 5 to 10)

Census areas: Building blocks of the city. Each census area should be assigned to exactly one zone.
              Variable self.x[i,z]: is a binary variable. It indicates whether area with index i is assigned to zone z or not.
Number of zones: self.M. TNumber of zones that we are trying to divide the city into.

Zone Description: 
- Each zone, consists of a set of census areas.
- To compute a zone metric we aggregate the corresponding metric for all census areas, assigned to that zone. 
    Example: Total number of students in zone z = sum of students among all census areas assigned to zone z. 
- Zones should have limited shortage: Shortage is the percentage of extra number of GE students in each zone compared to total number of GE seats within that zone. 
    Zone should not have shortage more than max_shortage percentage, where max_shortage is a given input. 
- Zones should be contiguous, and not to be consisting of multiple islands. 
- Zones should be nicely shaped and compact looking.

We use Gurobi for linear programming to impose the desired constraints. I attached the code that generates zone constraints.
I want you to fully read and understand the code and the data structures used within the code.
Then, help me add new set of constraints to the existing model as I ask later.


Instructions:
1. Please thoroughly analyze the code in integer_program.py, 
    pay close attention to the data structures and code organization, by reading the comments.
2. Aim to deeply understand integer_program.py and its functionality, 
    and how I implemented the integer programming constraints.
4. self.area_data dataframe: 
    - Each row of self.area_data is a different area. Row i, has information about area with index i.
    - Top 5 rows of self.area_data dataframe are saved in top_5_area_data.csv. 
    - Meaning of columns in self.area_data dataframe is saved in area_data.txt. 
    - Make sure you understand the area_data dataframe by reading the content of area_data.txt and looking at area_data.csv.
5. Data structures in integer_program.py are explained in the code comments in integer_program.py file.
5. Very important: Ensure your suggested code can be appended to the existing 
    code file without causing any conflicts or errors.

Request: How can I add a the following function to the code, while ensuring compatibility with the existing data structures and code base?
Function: Write a function in python to make sure: 
average quality of schools across zones is balanced. 
Use average color index to measure school quality. 
"""

payload = {
    "prompt": prompt,
    "max_tokens_to_sample": 500,  # Adjust the desired response length
    "model": "claude-v1.3"
}

# Send the API request
response = requests.post(api_endpoint, json=payload, headers=headers)

# Check the response status
if response.status_code == 200:
    # Parse the API response
    api_response = response.json()
    assistant_reply = api_response["completion"]
    print("Claude's response:")
    print(assistant_reply)
else:
    print("Error occurred during API call:", response.status_code)