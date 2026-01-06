from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import Domain

from Zone_Generation.Optimization.constraint_program_boolean import BooleanConstraintProgram


class IntegerConstraintProgram(BooleanConstraintProgram):
    def __init__(self, config):
        super().__init__(config)
        self.y = self.add_integer_variables()
        if self.config.get('closest_school_max_distance'):
            self.school_id_2_area, self.school_access_vars = self.add_school_access_variables()

    def add_integer_variables(self):
        y = []
        for i in range(self.A):
            if len(self.valid_zone_per_area[i]) == 0:
                continue
            var = self.m.NewIntVarFromDomain(Domain.FromValues(self.valid_zone_per_area[i]), f"area_{i}_zone_idx")
            for z in self.valid_zone_per_area[i]:
                self.m.Add(var == z).OnlyEnforceIf(self.x[z][i])
                self.m.Add(var != z).OnlyEnforceIf(self.x[z][i].Not())

            y.append(var)
        return y

    def _feasibility_const(self):
        # each centroid belong to its own zone
        for z in range(self.Z):
            centroid_z = self.centroids[z]
            self.m.Add(self.y[centroid_z] == z)
            for neighbor in self.G.neighbors(centroid_z):
                if neighbor in self.valid_zone_per_area[neighbor]:
                    self.m.Add(self.y[neighbor] == z)

    def add_school_access_variables(self):
        # we define area i as having access to a school if it is assigned to the same zone as the school
        school_id_2_area = {}
        for node in self.G.nodes(data=True):
            area_idx = node[0]
            school_ids = node[1]['school_ids']
            for school_id in school_ids:
                school_id_2_area[school_id] = area_idx
        school_access_vars = {}  # define whether area i has access to school s
        for area_id in range(self.A):
            for school_id in school_id_2_area.keys():
                access_var = self.m.NewBoolVar(f"area_{area_id}_access_school_{school_id}")
                school_area = school_id_2_area[school_id]
                school_access_vars[(area_id, school_area)] = access_var

                # if area and school_area are assigned to the same zone, then access_var = 1
                self.m.Add(self.y[area_id] == self.y[school_area]).OnlyEnforceIf(access_var)
                self.m.Add(self.y[area_id] != self.y[school_area]).OnlyEnforceIf(access_var.Not())
        return school_id_2_area, school_access_vars

    # def _add_hints(self):
    #     super()._add_hints()
    #     # add hint that each area will be assigned to the closest centroid
    #     for i in range(self.A):
    #         closest_centroid = None
    #         closest_distance = float('inf')
    #         for z in range(self.Z):
    #             centroid_z = self.centroids[z]
    #             dist = self.G.graph['distance_dict'][centroid_z][i]
    #             if dist < closest_distance:
    #                 closest_distance = dist
    #                 closest_centroid = z
    #         if closest_centroid in self.valid_zone_per_area[i]:
    #             self.m.AddHint(self.y[i], closest_centroid)

    def _closest_school_const(self):
        # ensure that the closet school to each area is within a certain distance
        max_distance = self.config.get('closest_school_max_distance')
        if not max_distance:
            return
        for i in range(self.A):
            # iterate through all schools to find the closest one
            # get distance from area i to each school

            # use the fact that the distance dict in the graph is from school to area
            close_enough_schools = []
            for school_area in self.G.graph['distance_dict'].keys():
                dist = self.G.graph['distance_dict'][school_area][i]
                if dist < max_distance:
                    close_enough_schools.append(school_area)

            if not close_enough_schools:
                print(f"Area {i} has no schools within the max distance of {max_distance}.")
                self.m.Add(False)
            # at least one of the close enough schools must be assigned to the same zone as area i
            access_vars = []
            for school_area in close_enough_schools:
                access_var = self.school_access_vars.get((i, school_area))
                if access_var is not None:
                    access_vars.append(access_var)
            self.m.AddBoolOr(access_vars)

    def add_objective(self):
        boundary_vars = []
        for area in range(self.A):
            for neighbor in self.G.neighbors(area):
                if area < neighbor:
                    boundary_var = self.m.NewBoolVar(f"boundary_area_{area}_neighbor_{neighbor}")
                    boundary_vars.append(boundary_var)
                    # if area and neighbor are assigned to different zones, then boundary_var = 1
                    self.m.Add(self.y[area] != self.y[neighbor]).OnlyEnforceIf(boundary_var)
                    self.m.Add(self.y[area] == self.y[neighbor]).OnlyEnforceIf(boundary_var.Not())

        boundary_sum = cp_model.LinearExpr.Sum(boundary_vars)
        self.m.Minimize(boundary_sum)

    def fix_areas(self, fixed_zone_dict):
        super().fix_areas(fixed_zone_dict)
        if fixed_zone_dict is None:
            return
        for area, zone in fixed_zone_dict.items():
            centroid_idx = self.centroid_schools.index(zone)
            if centroid_idx in self.valid_zone_per_area[area]:
                self.m.Add(self.y[area] == centroid_idx)
