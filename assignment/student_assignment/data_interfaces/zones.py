"""Class to handle zone data for market_generator.

Created 7/25/20
@author Itai Ashlagi
Class creating zones
"""

import copy
import csv
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data_interfaces import Programs, Students
from ..definitions import CW2AA


class Zones:
    """Object to handle zone data for market_generator."""

    def __init__(
        self,
        config: dict,
        attendance_area_schools: pd.DataFrame,
        programs: Programs,
        students: Students,
    ):
        """Create object to handle zone data.

        Args:
            config (Configerator.config): market generator configuration file
            attendance_area_schools (pd.DataFrame): schools indexed by
                attendance area.
            programs (Programs): program data interface.
            students (Students): student data interface.
        """
        self.programs = programs
        self.aa_schools = attendance_area_schools
        self.aa_schools.index.name = "school_id"
        self.aa_schools.reset_index(inplace=True)
        self.students = students

        self.config = config
        self.zones_set = False
        self._zone_eligibility_matrix = None
        self._zone_priority_matrix = None

    def _get_area2school_id(self) -> tuple[dict, dict]:
        """Generate dictionaries mapping areas to schools and schools to areas.

        Returns:
            area2school_id (dict): dictionary from an area to a list of schools inside that area
            school_id2area (dict): dictionary from a school_id to the area it's inside
                (attendance area, block, or block group)
        """
        if self.config["zone-building-blocks"] == "attendance_area":
            # TODO: figure out how to handle citywide schools in zones
            return (
                {
                    x: [x] for x in self.aa_schools["attendance_area"]
                },  # doesn't include citywide schools
                {
                    **{x: x for x in self.aa_schools["attendance_area"]},
                    **CW2AA,
                },  # adding citywide schools (is this needed? Possibly for zone files that include citywide schools)
            )
        elif self.config["zone-building-blocks"] == "block_group":
            area2school_id = defaultdict(
                list,
                self.aa_schools.groupby("BlockGroup")["school_id"]
                .apply(list)
                .to_dict(),
            )  # does not include citywide schools right now
            school_id2area = dict(
                zip(self.aa_schools.school_id, self.aa_schools.BlockGroup)
            )
            return area2school_id, school_id2area
        elif self.config["zone-building-blocks"] == "block":
            area2school_id = defaultdict(
                list,
                self.aa_schools.groupby("Block")["school_id"]
                .apply(list)
                .to_dict(),
            )  # does not include citywide schools right now
            school_id2area = dict(
                zip(self.aa_schools.school_id, self.aa_schools.Block)
            )
            return area2school_id, school_id2area
        elif self.config["zone-building-blocks"] == "home_based":
            return {}, {}
        else:
            raise ValueError(
                f"Unrecognized zone building block '{self.config['zone-building-blocks']}'. Please use 'attendance_area', "
                f"'block_group', 'block', or 'home_based. "
            )

    def _create_zone(
        self, zone_file: str, concept: int = None
    ) -> tuple[dict, list]:
        """Create list of areas in each zone and dictionary of area to zone id.

        Args:
            zone_file (str): path to zone csv file
            concept (int): 0 if concept 0 (which has a small error to correct, otherwise not used

        Returns:
            area_id2zone_id (dict): dictionary mapping an area_id to the zone_id that the area is a part of
            zone_lists (list): a list of sets containing the area_id's in each zone
        """
        if (
            self.config["zone-building-blocks"] == "home_based"
            and concept is None
        ):
            with open(zone_file) as f:
                student2programs = json.load(f)
            return student2programs, []

        zone_lists = []
        reader = csv.reader(open(zone_file))
        for row in reader:
            if concept == 0:
                zone_lists.append(
                    {int(x) + 1 for x in row if str(x) != "" and int(x) > 0}
                )  # Concept 0 Attendance area codes off by 1
            else:
                zone_lists.append(
                    {int(x) for x in row if str(x) != "" and int(x) > 0}
                )
        area_id2zone_id = self._create_zone_dictionary(zone_lists)
        return area_id2zone_id, zone_lists

    def set_area_id2prog_list_dict(
        self,
        lp_zone_path_list: list[str] = None,
        lp_zone_dict_list: dict = None,
        remaining_programs_citywide: bool = False,
        lp_same_as_ge: bool = False,
    ):
        """Create a dictionary mapping an area to the list of accessible programs.

        Args:
            lp_zone_path_list (list): list of file paths for language or citywide program zones
            lp_zone_dict_list (dict): dictionary mapping areas to list of programs that should be made accessible
            remaining_programs_citywide (bool): whether or not to distribute all programs not mentioned in zones to all
                zones
            lp_same_as_ge (bool): whether language programs share the same zone
                layout as general-education programs.
        """
        if self.config["zone-building-blocks"] == "home_based":
            self.area_id2prog_list = self.area2zone
            self._update_programs_for_area()
            return

        self.get_area_id2ge_program_id_dict()
        lp_area_id2prog_list = {}
        if lp_zone_path_list is not None:
            for zone_path in lp_zone_path_list:
                print("Loading additional schools for zones from:", zone_path)
                with open(os.path.expanduser(zone_path)) as f:
                    s = f.read().rstrip("\n")
                aaDict_new = eval(s)
                for k, v in aaDict_new.items():
                    lp_area_id2prog_list[k] = (
                        lp_area_id2prog_list.get(k, []) + v
                    )

        if lp_zone_dict_list is not None:
            for aaDict_new in lp_zone_dict_list:
                for k, v in aaDict_new.items():
                    lp_area_id2prog_list[k] = (
                        lp_area_id2prog_list.get(k, []) + v
                    )

        # Convert language program zones (attendance area) to match the zone
        # building blocks, by changing the keys of lp_area_id2prog_list to
        # corresponding block or block group.
        lp_area_id2prog_list = self.transform_aa_keys_to_building_blocks(
            lp_area_id2prog_list
        )

        if lp_same_as_ge:
            sch2pr_idx = self.programs.school_to_indices
            for area_id, prog_list in self.area_id2ge_program_id.items():
                program_idxs = [
                    y for x in prog_list for y in sch2pr_idx[int(x[:3])]
                ]
                lp_area_id2prog_list[area_id] = [
                    self.programs.codes[x] for x in program_idxs
                ]
            self.area_id2prog_list = lp_area_id2prog_list
        self.lp_area_id2prog_list = lp_area_id2prog_list

        # combine GE and LP zones
        area_id2prog_list = copy.deepcopy(self.area_id2ge_program_id)
        for area_id, program_list in self.lp_area_id2prog_list.items():
            area_id2prog_list[area_id] = (
                area_id2prog_list.get(area_id, []) + program_list
            )
        self.area_id2prog_list = area_id2prog_list

        if remaining_programs_citywide:
            self.add_remaining_schools_to_all_zones()

        # print("THIS IS STILL IN HERE MAKE SURE TO REFACATOR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # citywide_programs_df = pd.read_csv("../transformed_data.csv")
        # self.add_citywide_programs_to_zones(citywide_programs_df)

        self._update_programs_for_area()

    def transform_aa_keys_to_building_blocks(self, dict_to_update: int):
        """Convert the key in the input dictionary from attendance area to the type
        of zone-building-blocks. Currently only support changing to blocks and
        block groups.
        Inputs:
        - dict_to_update: the dictionary to update keys.
        """
        if self.config["zone-building-blocks"] == "attendance_area" or not len(
            dict_to_update
        ):
            return dict_to_update
        if self.config["zone-building-blocks"] not in ["block_group", "block"]:
            print(
                "Warning: Using AA keys for LP or citiwide schools with"
                + self.config["zone-building-blocks"]
                + "zone-building-blocks might cause errors."
            )
            return dict_to_update

        if list(dict_to_update.keys())[0] > 1000:
            # Attendance areas are 3 digit numbers. If the number > 1000, then
            # We assume that it is already the BG or Block.
            return dict_to_update

        # The dict to match the attendance area to block or block group.
        # The matching columns in aa_schools is "Block" and "Block Group", while
        # the zone-building-blocks is block or block_group, so we need to
        # reformat it to the respective column name.
        building_block = (
            self.config["zone-building-blocks"]
            .replace("_", " ")
            .title()
            .replace(" ", "")
        )
        if len(dict_to_update):
            print(
                "Warning: Transforming AA keys for LP or citiwide schools to keys in"
                + self.config["zone-building-blocks"]
                + "zone-building-blocks. This might raise issues as we do not have 1-to-1 mapping between AA and BGs/blocks."
            )
        dict_to_match = {
            x: y
            for [x, y] in self.aa_schools[
                ["attendance_area", building_block]
            ].to_numpy()
        }
        dict_updated = {}
        for k, v in dict_to_update.items():
            k = dict_to_match[k]
            dict_updated[k] = dict_updated.get(k, []) + v
        # Remove duplicate values.
        for k, v in dict_updated.items():
            dict_updated[k] = list(set(v))

        return dict_updated

    def _update_programs_for_area(self):
        """Create a num_programs length indicator vector for each area indicating program eligibility."""
        area2progs_vec = {}
        for k, v in self.area_id2prog_list.items():
            prog_idxs = [x - 1 for x in self.programs.index_list(v)]
            vec = np.zeros(len(self.programs.program_df.index))
            vec[prog_idxs] = 1
            area2progs_vec[k] = np.array(vec, dtype=int)
        self.area2progs_vec = area2progs_vec

    def programs_for_area_id(
        self, attendance_area: int, block_group: int, block: int, studentno: int
    ):
        """Get a num_programs length indicator vector for an area indicating program eligibility.

        Note that for market_generator interface simplicity, we always pass attendance_area and block_group, then allow
        the zones object determine which is appropriate for the zones in use.

        Args:
            attendance_area (int): an attendance area school_id
            block_group (int): a census block group id
            block (int): a census block id
            studentno (int): the student number the eligibility is for

        Returns:
            a binary np.ndarray with program eligibility for the area
        """
        # set up dictionary if you haven't before
        if not hasattr(self, "area2progs_vec"):
            self._update_programs_for_area()
        if self.config["zone-building-blocks"] == "attendance_area":
            return (
                self.area2progs_vec[attendance_area]
                if attendance_area in self.area2progs_vec
                else np.zeros(len(self.programs.program_df.index))
            )
        if self.config["zone-building-blocks"] == "home_based":
            return self.area2progs_vec[str(studentno)]

        area = (
            block_group
            if self.config["zone-building-blocks"] == "block_group"
            else block
        )
        area_name = (
            "block group"
            if self.config["zone-building-blocks"] == "block_group"
            else "census block"
        )
        try:
            return self.area2progs_vec[int(area)]
        except KeyError:
            print(f"Could not find any programs for {area_name} {area}.")
            return np.zeros(len(self.programs.program_df.index), dtype=int)
        except ValueError:
            # print(f"Missing {area_name} for student.")
            return np.zeros(len(self.programs.program_df.index), dtype=int)

    def get_area_id2ge_program_id_dict(self):
        """Create dictionary mapping zone_ids to list of area_id's and dictionary of area ids to eligible programs."""
        self.area2school_id, self.school_id2area = self._get_area2school_id()
        zone2area_list = defaultdict(
            list
        )  # dictionary mapping zone_id to list of area_ids in zone
        for area_id, zone_id in self.area2zone.items():
            zone2area_list[zone_id].append(area_id)

        area_id2program_id = {}
        for area_id, zone_id in self.area2zone.items():
            school_list_of_lists = []
            for area in list(zone2area_list[zone_id]):
                if area in self.area2school_id:
                    school_list_of_lists.append(self.area2school_id[area])
            school_list = [
                x for sublist in school_list_of_lists for x in sublist
            ]
            area_id2program_id[area_id] = [
                str(school_id) + "-GE-KG" for school_id in school_list
            ]

        self.zone2area_list = zone2area_list
        self.area_id2ge_program_id = area_id2program_id

    def add_remaining_schools_to_all_zones(self):
        """Add any program not in any zone (area_id2prog_list) to all zones."""
        programs_in_zones = {
            prog
            for prog_list in self.area_id2prog_list.values()
            for prog in prog_list
        }  # set of all programs
        for program in list(set(self.programs.indices) - programs_in_zones):
            for area_id in self.area_id2prog_list:
                self.area_id2prog_list[area_id].append(program)

    def add_citywide_programs_to_zones(self, citywide_programs: pd.DataFrame):
        """Add citywide programs from a provided DataFrame to all zones.

        Args:
            citywide_programs (pd.DataFrame): DataFrame with a column 'program_id' and 'Type'.
                                              Programs with 'Type' == 'Citywide' will be added
                                              to all zones.
        """
        # import pdb

        # pdb.set_trace()
        citywide_programs_df = citywide_programs.copy()
        # Filter the DataFrame to only include Citywide programs
        citywide_programs = citywide_programs_df[
            citywide_programs_df["Type"] == "Citywide"
        ]["program_id"].tolist()

        # Add each Citywide program to all zones
        for program in citywide_programs:
            for area_id in self.area_id2prog_list:
                if program not in self.area_id2prog_list[area_id]:
                    self.area_id2prog_list[area_id].append(program)

        print(f"Added {len(citywide_programs)} Citywide programs to all zones.")

    def set_zone(self, concept: int | str):
        """Set the current zones to a preexisting concept (0, 1, 2, 3) or the zone file passed.

        Args:
            concept (int or str): Either integers 0, 1, 2, or 3, or a file path to a zone csv file
        """
        if concept == "real_match":
            self.area2zone, self.zone2area_list = self._create_zone(
                os.path.expanduser(self.config["paths"]["zone-files"]["Con1"]),
                concept=1,
            )
        else:
            self.area2zone, self.zone2area_list = self._create_zone(
                os.path.expanduser(concept)
            )
        if self.config["zone-building-blocks"] in [
            "attendance_area",
            "block_group",
            "block",
        ]:
            self.zones_set = True

    # def set_zone_from_dict(self, zone_list: List, zone_dict: dict):
    #     """
    #     Set up zone object from already calculated zone list and zone dict.

    #     Args:
    #         zone_list (List): list of lists, inner lists containing areas for a single zone
    #         zone_dict (dict): dictionary mapping area_ids to zone_ids
    #     """
    #     self.zone2area_list = zone_list
    #     self.area2zone = zone_dict

    @staticmethod
    def _create_zone_dictionary(zone_list: list):
        """Create a dictionary mapping area_ids to zone_ids.

        Args:
            zone_list (List): list of lists, each inner list containing the areas in a zone

        Returns:
            dict mapping area_ids to zone_ids
        """
        area2zone_id = {}
        for i, zone in enumerate(zone_list):
            area2zone_id = {**area2zone_id, **{area: i for area in zone}}
        return area2zone_id

    def get_studentno_to_zone_dict(self, student_data):
        if self.config["zone-building-blocks"] == "attendance_area":
            return {
                studentno: self.area2zone[x["idschoolattendance"]]
                for studentno, x in student_data.iterrows()
                if not np.isnan(x["idschoolattendance"])
            }
        elif self.config["zone-building-blocks"] == "block_group":
            return {
                studentno: self.area2zone[x["census_blockgroup"]]
                for studentno, x in student_data.iterrows()
                if not np.isnan(x["census_blockgroup"])
            }
        elif self.config["zone-building-blocks"] == "block":
            return {
                studentno: self.area2zone[x["census_block"]]
                for studentno, x in student_data.iterrows()
                if not np.isnan(x["census_block"])
            }

    def get_program_idx_to_zone_dict(self) -> dict:
        """Create a dictionary mapping program indices to zone_ids.

        Returns:
            dict mapping program indices to zone_ids
        """
        prog2zone = {}
        for area, prog_list in self.area_id2prog_list.items():
            prog_idxs = [x - 1 for x in self.programs.index_list(prog_list)]
            prog2zone = {
                **prog2zone,
                **{idx: self.area2zone[area] for idx in prog_idxs},
            }
        return prog2zone

    @property
    def zone_priority_matrix(self) -> np.ndarray:
        """Create a 0-1 matrix indicating where students get in-zone priority.

        Note that this does not include sibling or CTIP zone exceptions.

        Returns:
            np.ndarray: 0-1 (num students) by (num programs) matrix where 1 indicates zone priority
        """
        if self._zone_priority_matrix is not None:
            return self._zone_priority_matrix

        self._zone_priority_matrix = np.zeros(
            (self.students.n, self.programs.num_programs), dtype=int
        )

        for studentno, row in self.students.student_data.iterrows():
            self._zone_priority_matrix[
                self.students.studentno2idx[studentno], :
            ] = self.programs_for_area_id(
                row.idschoolattendance,
                row.census_blockgroup,
                row.census_block,
                studentno,
            )
        return self._zone_priority_matrix

    @property
    def zone_eligibility_matrix(self) -> np.ndarray:
        """Create a 0-1 matrix indicating zone eligibility for each student-program pair.

        This differs from zone priority in terms of sibling and CTIP zone eligibility exeptions.

        Returns:
            np.ndarray: 0-1 (num students) by (num programs) matrix where 1 indicates zone eligibility
        """
        if self._zone_eligibility_matrix is not None:
            return self._zone_eligibility_matrix

        in_zone = self.zone_priority_matrix
        ctip = np.outer(
            self.students.student_data.ctip1.to_numpy(dtype=int),
            np.ones(self.programs.num_programs, dtype=int),
        )
        sibling_array = self.students.sibling(self.programs)
        in_zone += int(self.config["sibling-access"]) * sibling_array
        in_zone += int(not self.config["restrict-zone"]) * 1
        in_zone += int(self.config["restrict-zone"] == "CTIP_access") * ctip
        in_zone = np.clip(in_zone, 0, 1)
        self._zone_eligibility_matrix = in_zone
        return self.zone_eligibility_matrix
