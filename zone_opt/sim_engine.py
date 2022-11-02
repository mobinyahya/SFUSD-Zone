import os
from typing import List

import numpy as np
import pandas as pd

from simulator_engine.src.market_generator.match_evaluator import MatchEvaluator
from simulator_engine.src.market_generator.school_choice_market_generator import (
    MarketGenerator,
)

DEFAULT_SUMMARY_SAVE_PATH = "~/Documents/sfusd/summary.csv"


class SimulateZones:
    """Run assignments for a large list of zones"""

    def __init__(self, evaluator="eval_assignment_manual"):
        """
        Generate student assignment object for zones

        :param evaluator: which evaluator function in MatchEvaluator to use in assignment
        """
        self.market = MarketGenerator()
        self.evaluator = evaluator

        # ensure that assignments are returned instead of saved
        self.market.config["save-assignment"] = False
        self.market.config["policies"] = ["dynamic"]

    @staticmethod
    def _get_zone_stats(zone: str) -> pd.Series:
        """
        Get statistics about zone to add to summary row.

        TODO: add more zone metrics

        :param zone: path to zone file
        :return: Series containing zone information
        """
        name = zone.split("/")[-1][:-4]
        return pd.Series({"zone": name})

    def _process_assignment(self, assignment: pd.DataFrame, iteration: int) -> pd.Series:
        """
        Calculate statistics on a given assignmnet using match evaluator.

        :param assignment: pd.DataFrame containing student assignment
        :param iteration: which iteration of the current policy is this assignment
        :return: pd.Series containing assignment metrics
        """
        assignment["programcodes"].replace({"": np.nan}, inplace=True)
        me = MatchEvaluator(
            self.market.students, assignment, self.market.students.distance_data
        )
        stats = getattr(me, self.evaluator)()
        return pd.concat([stats, pd.Series({'iteration': iteration})])

    def run_market_on_zone_list(self, zone_file_list: List[str], summary_file: str = "", append: bool = False):
        """
        Compute assignment and summary statistics on a large number of zone policies without saving assignments.

        :param zone_file_list: List of paths to zones to test
        :param summary_file: Path to save metrics computed here
        :param append: whether or not to append new metrics to a preexisting summary file
        """
        if summary_file == "":
            summary_file = DEFAULT_SUMMARY_SAVE_PATH

        if append:
            summary = pd.read_csv(summary_file)
        else:
            summary = pd.DataFrame()

        for zone in zone_file_list:
            self.market.config["paths"]["zone-files"]["dynamic"] = zone
            zone_stats = self._get_zone_stats(zone)
            outer_generator = self.market.simulate()

            for i, middle_generator in enumerate(outer_generator):
                for inner_generator in middle_generator:
                    for assignment in inner_generator:
                        result = self._process_assignment(assignment, i)
                        summary = summary.append(
                            pd.concat([result, zone_stats]), ignore_index=True
                        )
                        summary.to_csv(summary_file, index=False)

    def get_optimization_params(self, file_name):
        params = {
            "year": self.market.config["generator"]["year"],
            "zone_file": file_name.split("/")[-1][:-4],
            "M": len(self.market.zones.zone2area_list),
            "LP Zones": self.market.config["lp-zones"],
        }
        if os.path.exists(file_name[:-4] + "_params.txt"):
            with open(file_name[:-4] + "_params.txt", "r") as f:
                for line in f:
                    s = line.split()
                    if s[0] == "contiguity":
                        params["Contiguity"] = s[1]
                    elif s[0] == "citywide":
                        params["Citywide"] = eval(s[1])

        return params


if __name__ == "__main__":
    zone_list = [
        "/Users/katherinementzer/Dropbox/SFUSD/Zones/concept2zones.csv",
        "/Users/katherinementzer/Dropbox/SFUSD/Zones/concept0zones.csv",
    ]
    sz = SimulateZones()
    sz.run_market_on_zone_list(zone_list, summary_file="~/Desktop/test.csv")
