"""Created 8/19/20.

@author Itai Ashlagi


"""

import numpy as np
import pandas as pd

from .da_with_guardrails import DAwithGuards


class GuardrailSetup:
    def __init__(self, market, prefs):
        self.schools = market.schools
        self.students = market.students
        self.programs = market.programs
        self.n = market.n
        self.market = market
        self.num_programs = market.num_programs
        self.prefs = prefs
        self.capacity = self.programs.capacity
        self._setup_cache_key = None
        if market.zones.zones_set:
            self.student2zone = market.zones.get_studentno_to_zone_dict(
                self.students.student_data
            )
            self.program2zone = market.zones.get_program_idx_to_zone_dict()

    def run(self, priorities, reserve_settings, strictGuards=0):
        setup_cache_key = (
            repr(reserve_settings),
            getattr(self.market, "_active_policy_cache_context", None),
        )
        if setup_cache_key != self._setup_cache_key:
            self._create_categories(reserve_settings)
            self._set_up_program_reserves(reserve_settings)
            self._setup_cache_key = setup_cache_key

        da = DAwithGuards(
            self.capacity,
            priorities,
            self.prefs,
            self.classOfStudent,
            strictGuards,
        )

        da.setguards(self.program_reserve_frac, numOfClasses=self.num_classes)
        self.match, self.studentRank = da.run()
        return self.match, self.studentRank

    def _create_categories(self, reserve_settings: dict):
        """Set up diversity categories for reserves.

        Args:
            reserve_settings: dictionary with keys "column" (column or dictionary of columns to corresponding weights
                to create underlying index) and either "num_categories" (number of diversity categories to create) or
                "thresholds" (explicit values of the column to use to separate classes) and "lower_disadvantaged"
                (whether a lower value of the column indicates more disadvantage)
        """
        data = self.students.student_data
        self._set_category_index_column(reserve_settings["column"], data)
        index_col = data["category_index"].to_numpy()
        index_col = np.nan_to_num(index_col, nan=np.nanmean(index_col))

        if "num_categories" in reserve_settings:
            percentages = (
                100
                * np.array(range(1, reserve_settings["num_categories"]))
                / reserve_settings["num_categories"]
            )
            thresholds = np.percentile(index_col, percentages)
        elif "thresholds" in reserve_settings:
            thresholds = reserve_settings[
                "thresholds"
            ]  # expect lower index => more disadvantaged

            if isinstance(thresholds, str) and thresholds.startswith(
                "percentile:"
            ):
                try:
                    p_val = float(thresholds.split(":")[1])
                    # 2. On calcule la vraie valeur numérique dans les données
                    thresholds = np.percentile(index_col, p_val)
                    print(
                        f"Computed dynamic threshold for {thresholds} (percentile {p_val})"
                    )
                    thresholds = np.atleast_1d(thresholds)
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Invalid format for thresholds: {thresholds}. Expected format 'percentile:50'"
                    )
            print(
                "Using explicit thresholds for reserve categories:", thresholds
            )
        else:
            raise ValueError(
                "Neither 'num_categories' nor 'thresholds' defined for reserve settings."
            )

        classes = np.searchsorted(thresholds, index_col)
        if reserve_settings["lower_disadvantaged"]:
            classes = (
                len(thresholds) - classes
            )  # switch to higher value => greater priority

        data.loc[:, "diversity_category"] = classes
        self.classOfStudent = classes
        self.num_classes = len(thresholds) + 1

    @staticmethod
    def _calculate_thresholds(data, num_categories):
        percentiles = [
            int(100 * i / num_categories) for i in range(1, num_categories)
        ]
        thresholds = np.percentile(data["category_index"].dropna(), percentiles)
        return thresholds

    @staticmethod
    def _set_category_index_column(columns, data):
        if isinstance(columns, dict):
            data.loc[:, "category_index"] = 0
            for col, weight in columns.items():
                data.loc[:, "category_index"] += weight * data[col]
        else:
            data["category_index"] = data[columns]

    def _calculate_zone_fractions(self):
        data = self.students.student_data
        data.loc[:, "count"] = 1
        # print(set(data.index) - set(self.student2zone.keys())) # missing for 2 students
        data.loc[:, "zone_id"] = [
            self.student2zone[x] if x in self.student2zone else np.nan
            for x in data.index
        ]
        count_per_zone = (
            data[["zone_id", "diversity_category", "count"]]
            .groupby(["zone_id", "diversity_category"], as_index=False)
            .sum()
        )
        zone_total = (
            data[["zone_id", "count"]].groupby("zone_id", as_index=False).sum()
        )
        count_per_zone = count_per_zone.merge(
            zone_total, how="left", on="zone_id", suffixes=("", "_tot")
        )
        count_per_zone.loc[:, "count"] /= count_per_zone.count_tot
        zone_frac = pd.pivot_table(
            count_per_zone,
            index="zone_id",
            columns="diversity_category",
            values="count",
            fill_value=0,
        )
        return zone_frac

    def _set_up_program_reserves(self, reserve_settings):
        reserve_frac = reserve_settings.get("reserve_fraction", -1)
        citywide_only = reserve_settings.get("citywide_only", False)
        if reserve_frac != -1:
            error_msg = (
                "Reserve fractions length does not match the number of classes"
            )
            assert len(reserve_frac) == self.num_classes, error_msg
        if hasattr(self, "student2zone"):
            zone_frac = self._calculate_zone_fractions()
        else:
            zone_frac = np.bincount(self.classOfStudent) / self.n
        self.program_reserve_frac = np.zeros(
            (self.num_programs, self.num_classes)
        )

        all_programs_ids = range(self.num_programs)
        if citywide_only:
            # TODO: Change hand-coded citywide schools KG grade to work with different data.
            print("Reserve only for citywide schools.")
            citywide_schools = [
                618,
                449,
                476,
                509,
                796,
                485,
                537,
                724,
                760,
                676,
                479,
                714,
                493,
                814,
            ]
            citywide_GE = [f"{x}-GE-KG" for x in citywide_schools]
            program_indices = self.programs.indices
            # - 1 for all as program indices are 1-indexed.
            all_programs_ids = [
                program_indices[x] - 1
                for x in citywide_GE
                if x in program_indices
            ]

        # Check if we should use separate reserves for citywide schools (defaults to True)
        use_citywide_separate_reserves = self.market.config.get(
            "citywide-separate-reserves", True
        )

        if use_citywide_separate_reserves:
            print("Using separate reserves for citywide schools.")
            citywide_schools = [
                618,
                449,
                476,
                509,
                796,
                485,
                537,
                724,
                760,
                676,
                479,
                714,
                493,
                814,
            ]
            citywide_GE = [f"{x}-GE-KG" for x in citywide_schools]
            program_indices = self.programs.indices
            # - 1 for all as program indices are 1-indexed.
            all_city_wide_programs_ids = [
                program_indices[x] - 1
                for x in citywide_GE
                if x in program_indices
            ]

        prog_type = self.programs.program_type

        for j in all_programs_ids:
            if (
                prog_type.iloc[j] == "GE"
                # TODO: add option in config to toggle on/off
                # or prog_type.iloc[j].endswith("N")
                # or prog_type.iloc[j].endswith("E")
            ):
                if reserve_frac != -1:
                    self.program_reserve_frac[j, :] = reserve_frac
                elif hasattr(self, "student2zone"):
                    if j in self.program2zone:
                        self.program_reserve_frac[j, :] = zone_frac.iloc[
                            self.program2zone[j]
                        ]
                else:
                    self.program_reserve_frac[j, :] = zone_frac

        # Only apply separate reserves for citywide schools if enabled
        if use_citywide_separate_reserves:
            # Get citywide reserve ratios from config, defaulting to [0.57, 0.43]
            citywide_ratios = self.market.config.get(
                "citywide-reserve-ratios", [0.57, 0.43]
            )

            for j in all_city_wide_programs_ids:
                self.program_reserve_frac[j, :] = citywide_ratios
