import time
from dataclasses import dataclass
from typing import Optional

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import Domain
from shapely.constructive import boundary

from Zone_Generation.Config.Constants import AREA_ETHNICITIES, SCALING_CONST
from Zone_Generation.Optimization.constraint_program_boolean import BooleanConstraintProgram
from Zone_Generation.Optimization.optimizer import Optimizer, DesignZones, SolutionOutput


class IntegerConstraintProgram(BooleanConstraintProgram):
    def __init__(self, dz: DesignZones, config):
        super().__init__(dz, config)
        self.y = self.add_integer_variables()

    def add_integer_variables(self):
        y = []
        for i in range(self.dz.A):
            var = self.m.NewIntVarFromDomain(Domain.FromValues(self.valid_zone_per_area[i]),
                                             f"area_{self.dz.idx2area[i]}_zone_idx")
            for z in self.valid_zone_per_area[i]:
                self.m.Add(var == z).OnlyEnforceIf(self.x[z][i])
                self.m.Add(var != z).OnlyEnforceIf(self.x[z][i].Not())

            y.append(var)
        return y

    def _feasibility_const(self):
        # each centroid belong to its own zone
        for z in range(self.dz.Z):
            centroid_z = self.dz.centroids[z]
            self.m.Add(self.y[centroid_z] == z)

    def _add_hints(self):
        # add hint that each area will be assigned to the closest centroid
        for i in range(self.dz.A):
            closest_centroid = None
            closest_distance = float('inf')
            for z in range(self.dz.Z):
                centroid_z = self.dz.centroids[z]
                dist = self.dz.euc_distances[centroid_z][i]
                if dist < closest_distance:
                    closest_distance = dist
                    closest_centroid = z
            if closest_centroid in self.valid_zone_per_area[i]:
                self.m.AddHint(self.y[i], closest_centroid)

    def add_objective(self):
        self._add_hints()
        boundary_vars = []
        for area in range(self.dz.A):
            for neighbor in self.dz.neighbors[area]:
                if area < neighbor:
                    boundary_var = self.m.NewBoolVar(
                        f"boundary_area_{self.dz.idx2area[area]}_neighbor_{self.dz.idx2area[neighbor]}")
                    boundary_vars.append(boundary_var)
                    # if area and neighbor are assigned to different zones, then boundary_var = 1
                    self.m.Add(self.y[area] != self.y[neighbor]).OnlyEnforceIf(boundary_var)
                    self.m.Add(self.y[area] == self.y[neighbor]).OnlyEnforceIf(boundary_var.Not())

        boundary_sum = cp_model.LinearExpr.Sum(boundary_vars)
        self.m.Minimize(boundary_sum)

    def fix_areas(self, fixed_zone_dict):
        if fixed_zone_dict is None:
            return
        fixed_areas = 0
        for area, zone in fixed_zone_dict.items():
            area_idx = self.dz.area2idx[area]
            self.m.Add(self.y[area_idx] == zone)
            fixed_areas += 1
        print(f"Fixed areas: {fixed_areas}")

    def _generate_zone_dict(self, solver):
        zone_dict = {}
        for i in range(self.dz.A):
            assigned_zone = solver.Value(self.y[i])
            zone_dict[self.dz.idx2area[i]] = assigned_zone
        return zone_dict
