import pathlib

import numpy as np
import pandas as pd

from ..market_generator.school_choice_market_generator import (
    MarketGenerator,
)

from .match_evaluator import MatchEvaluator


class EvaluateAssignments:
    def __init__(self, market, iterations=25):
        self.market = market
        self.distances = market.students.distance_data
        self.iterations = iterations

    def evaluate_results(
        self,
        assignment_path: pathlib.Path,
        assignment_names: list[str],
        table_path: pathlib.Path,
    ):
        assignment1 = assignment_names[0]
        if assignment1 != "Assignment_real_match":
            filename = assignment_path / (assignment1 + "_iteration0.csv")
        else:
            filename = assignment_path / (assignment1 + ".csv")
        info = pd.Series({"Assignment": "Placeholder (ignore)"})
        sim_num = 0
        me = MatchEvaluator(
            self.market.students,
            pd.read_csv(filename, index_col=0),
            self.distances,
        )
        result = me.eval_assignment_basic()
        result["sim_number"] = sim_num
        result = pd.concat([info, result])
        results = pd.DataFrame(result).T

        """need to loop here on assignments and on filters"""
        for assignment_name in assignment_names:
            sim_num += 1
            for i in range(0, self.iterations):
                self.iteration = i
                if assignment_name != "Assignment_real_match":
                    filename = assignment_path / (
                        assignment_name + f"_iteration{self.iteration}.csv"
                    )
                else:
                    filename = assignment_path / (assignment_name + ".csv")
                info = pd.Series({"Assignment": assignment_name})
                # assignment_df = pd.read_csv(filename, index_col=0)
                me = MatchEvaluator(
                    self.market.students,
                    pd.read_csv(filename, index_col=0),
                    self.distances,
                )
                self.me = me
                result = me.eval_assignment_basic()
                result["sim_number"] = sim_num
                result = pd.concat([info, result])
                results = pd.concat(
                    [results, pd.DataFrame(result).T], ignore_index=True
                )

        label_cols = ["Assignment"]
        labels = results[label_cols + ["sim_number"]]
        labels = labels.groupby("sim_number").first()

        metric_cols = [x for x in results.columns if x not in label_cols]
        metrics = results[metric_cols]
        metrics = metrics.apply(pd.to_numeric, errors="coerce")
        metrics = metrics.groupby("sim_number").mean(numeric_only=True)

        results.loc[:, "Iterations"] = 1
        count = results[["sim_number", "Iterations"]]
        count = count.groupby("sim_number").sum()

        table = labels.join(metrics)
        table = table.join(count)
        table.to_csv(table_path, index=False)

    @staticmethod
    def reformat_paper_metrics(
        table_path: pathlib.Path, save_name: str = "clean_summary.csv"
    ):
        metrics = pd.read_csv(table_path)
        metrics.drop(index=0, inplace=True)
        metrics.set_index("Assignment", inplace=True)

        proximity = ["Distance Av", "Distance < 0.5", "Distance > 3"]
        diversity = [
            "Schools above 10% district FRL",
            "Schools above 15% district FRL",
            "AALPI in school with +10% FRL",
            "AALPI in school with +15% FRL",
            "Dissimilarity AALPI",
            "Dissimilarity SES3",
            "Programs with 1-4 AA",
        ]
        choice = [
            "Unassigned",
            "Designated",
            "Top 3 choice",
            "Top 1 choice",
            "Top 3 in-zone choice",
            "Top 1 in-zone choice",
            "Dist >= 3, Rank >= 5",
            "Avg utility",
        ]
        community_cohesion = ["BG Cohesion (3)"]

        groups = [
            "Black or African American",
            "Asian",
            "Hispanic/Latino",
            "White",
            "Low FRL",
            "High FRL",
        ]
        equity_of_access = []
        for g in groups:
            equity_of_access.append(f"Top 3 choice {g}")
            equity_of_access.append(f"Distance Av {g}")
            equity_of_access.append(f"{g} in school with +15% FRL")
            equity_of_access.append(f"{g} Dist >= 3, Rank >= 5")
            equity_of_access.append(" ")

        metrics.loc[:, " "] = np.nan
        table = metrics.T.loc[
            proximity
            + [" "]
            + diversity
            + [" "]
            + choice
            + [" "]
            + community_cohesion
            + [" "]
            + equity_of_access
        ]
        table.to_csv(table_path.parent / save_name)


if __name__ == "__main__":
    ##########
    # Inputs #
    ##########

    iterations = 25

    path = pathlib.Path("~/Documents/sfusd/local_runs").expanduser()
    # assignment_path = path / 'assignments' / 'Assignments_June_28' / 'Sarah'
    # summary_path = path / 'Summary'

    assignment_path = path / "assignments"  # / 'peng_menus'
    summary_path = path / "Table_June_28_Sarah_Fixed.csv"

    assignment_names = []
    # assignment_names.append('Assignment_real_match')

    # assignment_names.append("Assignment_CTIP0_round_merged123_policypeng-menushome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.65_alpha1_card8home_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.9_alpha1_card8_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.9_alpha1_card0_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.65_alpha1_card8_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.65_alpha1_card0_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl1_alpha1_card0_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged0_policydist4_maxfrl1_alpha1_card0_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl1_alpha1_card0_umodelavghome_based_peng_tiesSTB_rankall")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl1_alpha0.5_card0_umodelavg_bigpenaltyhome_based_peng_tiesSTB0_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl1_alpha0.5_card0_umodelavg_bigpenaltyhome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist10_maxfrl1_alpha0.5_card0_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist10_maxfrl1_alpha0.5_card0_umodelavg_less-scalehome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist10_maxfrl1_alpha0.5_card0_umodelavg_less-scale-2home_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl1_alpha0.5_card15_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl1_alpha0.5_card0_umodelavghome_based_peng_tiesSTB_noumodelnoise")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl1_alpha1_card0_umodelavghome_based_peng_tiesSTB_noumodelnoise")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.9_alpha0.5_card10_umodelavghome_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.65_alpha1_card0home_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP0_round_merged123_policydist4_maxfrl0.65_alpha0.5_card0home_based_peng_tiesSTB")
    # assignment_names.append("Assignment_CTIP3D_round_merged0_policyMedium1zones_tiesSTB")

    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_2_2_2_2_centroids 7-zone-14_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_1_2_2_1_centroids 7-zone-15_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_2_1_1_2_1_centroids 6-zone-9_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_2_1_1_2_2_centroids 6-zone-9_BG_lszones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_2_1_1_2_2_centroids 6-zone-10_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_1_1_2_2_centroids 6-zone-9_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_2_1_2_2_centroids 6-zone-7_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_1_2_2_1_centroids 7-zone-14_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_2_2_2_2_centroids 7-zone-17_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_1_1_2_2_centroids 6-zone-3_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_2_1_2_2_centroids 6-zone-3_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_2_1_2_2_centroids 6-zone-10_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_2_1_1_2_2_centroids 6-zone-2_BG_lszones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_2_1_2_2_centroids 6-zone-3_BG_lszones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_1_1_2_1_centroids 6-zone-9_BGzones+reserves_bgs_tiesSTB"
    )
    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyGE_Zoning_1_2_1_2_2_centroids 6-zone-9_BGzones+reserves_bgs_tiesSTB"
    )

    assignment_names.append(
        "Assignment_CTIP3D_round_merged0_policyMedium1zones+reserves_tiesSTB"
    )
    # assignment_names.append("Assignment_CTIP3D_round_merged0_policyCon1priorities_tiesSTB")
    assignment_names.append("Assignment_real_match")
    # assignment_names.append('Assignment_CTIP1_round_merged123_policyCon1choice_model_real_match_tiesMTB')
    # assignment_names.append('Assignment_CTIP1_round_merged0_policyCon1choice_model_real_match_tiesMTB')

    """
    assignment_names.append('Assignment_CTIP1_round_merged123_policyCon1-simulatedReal-prefLength07_tiesSTB_prefExtend0') 
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyMedium1-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyMedium1-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyMedium1-priorities-prefLength07_tiesSTB_prefExtend0')
    """

    # for i in range(1, 16):
    #     assignment_names.append('Assignment_CTIP3D_round_merged123_policyJosephZones{}zones_tiesSTB'.format(i))
    # for i in range(1, 16):
    #     assignment_names.append('Assignment_CTIP0_round_merged123_policyJosephZones{}zones_tiesSTB'.format(i))
    # for i in range(1, 16):
    #     assignment_names.append('Assignment_CTIP3D_round_merged123_policyJosephZones{}zones+reserves_tiesSTB'.format(i))

    """
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyr4-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5q2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyq4-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1q2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')

    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyr4-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5q2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyq4-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1q2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyr4-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5q2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyq4-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1q2-GuardRails0-prefLength07_tiesSTB_prefExtend0')

    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyr4-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk5q2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyk4r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyq4-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP3D_round_merged123_policyd1q2-prefLength07_tiesSTB_prefExtend0')
    """

    """
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyr4-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5q2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyq4-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1q2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd2-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')

    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyr4-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5q2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyq4-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1q2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd2-GuardRails0-no_zone_restrict-prefLength07_tiesSTB_prefExtend0')
    
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyr4-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5q2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyq4-GuardRails0-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1q2-GuardRails0-prefLength07_tiesSTB_prefExtend0')

    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyr4-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk5q2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyk4r2-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyq4-prefLength07_tiesSTB_prefExtend0')
    assignment_names.append('Assignment_CTIP0_round_merged123_policyd1q2-prefLength07_tiesSTB_prefExtend0')
    """
    #################
    # End of inputs #
    #################

    market = MarketGenerator()
    # market.students.get5CTIPTypes()
    market.students.get_diversity_categories()
    s = EvaluateAssignments(market)
    s.evaluate_results(assignment_path, assignment_names, table_path=summary_path)
    s.reformat_paper_metrics(summary_path)
