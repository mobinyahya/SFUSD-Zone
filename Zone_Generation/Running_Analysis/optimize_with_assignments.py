import glob
import os
import random
import sys
from itertools import product

import numpy as np
import pandas as pd
import yaml

from generate_zones import DesignZones
from simulator_engine.match_evaluator import MatchEvaluator
from simulator_engine.src.data_interfaces.programs import Programs
from simulator_engine.src.data_interfaces.schools import Schools
from simulator_engine.src.market_generator.school_choice_market_generator import (
    SchoolChoiceMarket,
)
from simulator_engine.src.market_generator.utility_model_v2 import UtilityModel
from zone_analysis import zone_summary_statistics

STUDENT_PATH = "~/SFUSD/Data/Cleaned/drop_optout_{}{}.csv"
SCHOOLS_PATH = "~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv"
PROGRAM_PATH = "~/SFUSD/Data/Cleaned/programs_1819.csv"
DEFAULT_SUMMARY_SAVE_PATH = "~/Dropbox/SFUSD/Data/Computed/Itai/summary.csv"
MNL_COEFF_PATH = "~/Dropbox/SFUSD/Data/MNL/MNL Coefficients 0912 (No SpEd).xlsx"  # Path to the file that contains the parameters of the MNL model
MNL_FEATURES_PATH = "~/Dropbox/SFUSD/Data/MNL/Program_MNL_0912.csv"  # Path to the file that contains the features of the schools used in the MNL model


class OptimizePostChoice:
    def __init__(self):
        pass

    def generate_opt(
        self,
        M=13,
        shortage=120,
        distance=4,
        balance=180,
        coverdistance=3.8,
        cont=0,
        lowfrl=0.3,
        maxfrl=2,
        lbHOCidx1=-1,
        ubHOCidx1=5,
        lbaalpi=-1,
        ubaapli=5,
        lowMet=-1,
        highMet=10,
        topMet=0,
        citywide=False,
    ):

        if cont == 1:
            self.opt.set_optimization_parameters(
                M,
                level="idschoolattendance",
                centroids_type=random.randint(0, 1),
                include_k8=citywide,
            )
            self.opt._set_objective_model(shortage, balance)
            self.opt._contiguity_const(
                max_distance=-1,
                real_distance=False,
                cover_distance=coverdistance,
                contiguity=True,
                neighbor=False,
            )

        else:
            self.opt.set_optimization_parameters(
                M=M,
                level="idschoolattendance",
                centroids_type=-1,
                include_k8=citywide,
            )
            self.opt._set_objective_model(shortage, balance)
            self.opt._contiguity_const(
                max_distance=distance,
                real_distance=True,
                cover_distance=-1,
                contiguity=False,
                neighbor=False,
            )

        self.opt._diversity_const(lowfrl, maxfrl, lbHOCidx1, ubHOCidx1, lbaalpi, ubaapli)
        self.opt._color_quality_const(lowMet, highMet, topMet)

    def _set_up_programs_and_schools(self):
        if not hasattr(self, "schools"):
            school_data_file = SCHOOLS_PATH
            program_data_file = PROGRAM_PATH
            self.programs = Programs(program_data_file)
            self.schools = Schools(school_data_file, self.programs)

    def _generate_assignment(self, market, priority_weights, tiebreaking):
        market.guaranteed = np.ones([market.n, market.num_programs])
        market.inaccessible = np.ones([market.n, market.num_programs])
        market.inaccessible_in_zone = np.ones([market.n, market.num_programs])
        market.in_zone = np.ones([market.n, market.num_programs])
        market.unpredictable = np.zeros([market.n, market.num_programs])
        market.setTieBreaker(tiebreaking)
        market.setPolicyZones()
        market.setPriorities(priority_weights)
        assignment_df = self.runDA(market)
        market.guaranteed = market.guaranteed[
            :, market.programs.program_df["program_type"] == "GE"
        ]
        market.inaccessible = market.inaccessible[
            :, market.programs.program_df["program_type"] == "GE"
        ]
        market.inaccessible_in_zone = market.inaccessible_in_zone[
            :, market.programs.program_df["program_type"] == "GE"
        ]
        market.in_zone = market.in_zone[
            :, market.programs.program_df["program_type"] == "GE"
        ]
        market.unpredictable = market.unpredictable[
            :, market.programs.program_df["program_type"] == "GE"
        ]
        return assignment_df

    @staticmethod
    def _set_ge_and_language_zones(
        LP_zone_names, LP_zone_path_list, file_name, i, market
    ):
        market.zone_lists.set_zone(file_name)
        if i == 0:
            market.zone_lists.set_area_id2prog_list_dict()
            lp_zones = 0
        else:
            market.zone_lists.set_area_id2prog_list_dict(
                LP_zone_path_list=LP_zone_path_list[i - 1]
            )
            lp_zones = LP_zone_names[i - 1]
        return lp_zones

    def _set_up_market(
        self,
        policy="a",
        priority_weights=None,
        rounds_merged=0,
        year=18,
        restrictzone=True,
    ):
        student_data_file = STUDENT_PATH.format(year, year + 1)
        market = SchoolChoiceMarket(year=year, student_data_file=student_data_file)
        market.set_new_capacities()
        market.priority_weights = priority_weights or {
            "sibling": 4,
            "zone": 2,
            "ctip": 1,
        }
        market.policy = policy
        market.setRoundsMerged(rounds_merged)
        market.restrict_zone = restrictzone

        if self.useMNL:
            self.umodel = UtilityModel(
                market.programs, MNL_COEFF_PATH, MNL_FEATURES_PATH
            )
            self.umodel.students23 = market.students.students23
            self.umodel.draw_utility_model_randomness(market, option=None)
            # market.OriginalPreferences = umodel.OriginalPreferences
            self.umodel.get_restricted_preferences(market)
        return market

    def runDA(self, market=None):
        if market is None:
            market = self._set_up_market()

        if self.useMNL:
            self.umodel.policy_zones = market.policy_zones
            market.setPreferencesFromModel(
                self.umodel, truncate_option=6, designation_option="designate"
            )

        market.generateAssignment()
        assignment_df = self._generate_assignment_df(market)
        return assignment_df

    def _generate_assignment_df(self, market):
        assignment_df = pd.DataFrame()
        assignment_df["studentno"] = market.students.students23.index
        assignment_df["programno"] = market.match
        assignment_df["programcodes"] = [
            x if x != "" else np.nan for x in market.assignmentCodes
        ]
        assignment_df["rank"] = market.studentRank
        assignment_df["designation"] = market.designatedAssignment
        if hasattr(market, "policy_zones") == True:
            inZoneRank = [
                int(x) if x != "" else 0
                for x in market._get_in_zone_rank(market.prefs)
            ]
            assignment_df["In-Zone Rank"] = inZoneRank
        else:
            assignment_df["In-Zone Rank"] = assignment_df["rank"]  # Just repeats the r
        assignment_df["program_cutoff"] = assignment_df["programno"].apply(
            lambda x: market.cutoffs[int(x) - 1]
        )
        if self.useMNL == True:
            market.setMatchUtilities(
                self.umodel
            )  # Calculate match utility of each student (0 if unmatched)
            if hasattr(market, "match_utilities"):
                assignment_df["assigned_utility"] = market.match_utilities
        return assignment_df

    def _append_result(self, results, result):
        if results.empty:
            return result
        else:
            return results.append(result, ignore_index=True)

    def _config_opt(self, config_file="default"):
        """ read optimization parameters from file or return default values.
        Expected file format: each line is a setting name, followed by a value
        or list of values (without any spaces!!). For example, the first 3 lines
        might look like:
            Moptions [13]
            distanceOptions [3,5]
            shortage 120
        """
        if config_file == "default":
            with open("config_zone_grid_search_default.yaml", "r") as f:
                config = yaml.safe_load(f)
        elif config_file[-5:] == ".yaml":
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}
            with open(os.path.expanduser(config_file), "r") as f:
                for line in f:
                    s = line.split()
                    config[s[0]] = eval(s[1])
        return config

    def _stats_are_good(self, stats):
        """ function to determine if zone stats are good enough to save """
        return stats["Thiel"] < 0.26 and stats["Max School FRL"] < 0.73

    def _stats_are_good2(self, stats):
        """ function to determine if zone stats are good enough to save """
        pt1 = stats["Thiel"] < 0.265 and stats["Max School FRL"] < 0.73
        pt2 = stats["BG isolation (2)"] < 0.38 and stats["Unassigned"] < 0.05
        pt3 = stats["Dist > 3"] < 0.15
        return pt1 and pt2 and pt3

    def _stats_are_good_easy(self, stats):
        """ function to determine if zone stats are good enough to save """
        return stats["Max School FRL"] < 0.73

    def optimize_assignment_stats(
        self,
        year=18,
        year2=-1,
        opt_config_file="",
        good_stats_fn="_stats_are_good",
        LP_zone_path_list=None,
        summaryfile="",
    ):
        """ input: year - year of data to design zones on
                   year2 - second year of data to check zones on. Optional, to
                            turn this option off put -1. """
        self._set_up_programs_and_schools()

        # set up market
        market = self._set_up_market(year=year)
        if year2 > 0:
            market2 = self._set_up_market(year=year2)  # second year to verify results
        priority_weights = market.priority_weights
        summary = pd.DataFrame()
        good_stats = getattr(self, good_stats_fn)

        config = self._config_opt(opt_config_file)

        self.opt = DesignZones(
            level="idschoolattendance", include_k8=config["citywide"]
        )

        if summaryfile == "":
            summaryfile = DEFAULT_SUMMARY_SAVE_PATH

        market.restrict_zone = True
        for (
            M,
            distance,
            lowfrl,
            maxfrl,
            lbHOCidx1,
            lbaalpi,
            cont,
            coverdistance,
            lowMet,
            topMet,
        ) in product(
            config["Moptions"],
            config["distanceOptions"],
            config["lowfrlOptions"],
            config["maxfrlOptions"],
            config["lbHOCidx1Options"],
            config["lbaalpiOptions"],
            config["contigOptions"],
            config["coverdistanceOptions"],
            config["lowMetOptions"],
            config["topMetOptions"],
        ):

            balance = (20 - M) * 31
            shortage = (23 - M) * 12

            if cont == 1:
                iterations = 100
            else:
                iterations = 6
                self.generate_opt(
                    M,
                    shortage,
                    distance,
                    balance,
                    coverdistance,
                    cont,
                    lowfrl,
                    maxfrl,
                    lbHOCidx1,
                    config["ubHOCidx1"],
                    lbaalpi,
                    config["ubaapli"],
                    lowMet,
                    config["highMet"],
                    topMet,
                )

            infeasible = 0
            for i in range(0, iterations):
                if cont == 1:
                    self.generate_opt(
                        M,
                        shortage,
                        distance,
                        balance,
                        coverdistance,
                        cont,
                        lowfrl,
                        maxfrl,
                        lbHOCidx1,
                        config["ubHOCidx1"],
                        lbaalpi,
                        config["ubaapli"],
                        lowMet,
                        config["highMet"],
                        topMet,
                    )

                rc = self.opt.solve(write=False)  # need this to get zones
                if rc == -1 and cont == 0:
                    break
                if infeasible >= 15:
                    break
                if rc == -1:
                    infeasible += 1
                    continue
                infeasible = 0
                zone_list = self.opt.zone_lists
                zone_dict = self.opt.zone_dict

                if i % 2 == 0:
                    ties = "STB"
                else:
                    ties = "MTB"

                stats = self.runMarket(
                    ties,
                    market,
                    priority_weights,
                    zone_list,
                    zone_dict,
                    LP_zone_path_list,
                )
                if good_stats(stats):
                    if year2 > 0:
                        # check if zones are good for another year of data, don't save if they're not
                        stats2 = self.runMarket(
                            ties,
                            market2,
                            priority_weights,
                            zone_list,
                            zone_dict,
                            LP_zone_path_list,
                        )
                        if not good_stats(stats2):
                            continue
                    # save zone stats and optimization parameters
                    k = random.randint(0, 400000)
                    self.opt.save(k)
                    pZone = self.getZoneStats()
                    params = pd.Series(self.opt.constraints)
                    params_and_stats = pd.concat([params, stats, pZone])
                    p = pd.DataFrame(params_and_stats).T
                    summary = self._append_result(summary, p)
                    summary.to_csv(summaryfile, index=False, header=True)

        return summary

    def runMarket(
        self, ties, market, priority_weights, zone_list, zone_dict, LP_zone_path_list
    ):

        self.opt.constraints["Ties"] = ties
        market.setTieBreaker(ties)
        market.zone_lists.set_zone_from_dict(zone_list, zone_dict)
        market.zone_lists.set_area_id2prog_list_dict(
            LP_zone_path_list=LP_zone_path_list
        )
        market.setPolicyZones()
        market.setPriorities(priority_weights)
        assignment_df = self.runDA(market)
        me = MatchEvaluator(
            market.students, assignment_df, market.students.distance_data
        )
        stats = me.lightweight_eval_assignment(
            self.schools, self.programs, market, priority_weights
        )
        if self.opt.constraints["Ties"] == "MTB":
            stats["Min Quality Access"] = 0
        return stats

    def getZoneStats(self, concept=-1):

        if concept == -1:
            df = zone_summary_statistics(self.opt)
        else:
            asssignment_file = "~/SFUSD/Zones/concept" + str(concept) + "zones.csv"
            af = os.path.expanduser(asssignment_file)
            df = zone_summary_statistics(self.opt, af, True)

        zStats = {
            "Zaalpi Score": min(df["AALPI Score"]),
            "Zfrl": min(df["FRL Score"]),
            "Zfrlmax": max(df["FRL Score"]),
            "Zdeficit": max(self.opt.deficitPerZone),
            "Zimbalance": self.opt.imbalance,
            "Zmet": min(self.opt.qualityMet),
            "Zcolor": min(self.opt.qualityColor),
        }
        pZoneStats = pd.Series(zStats)
        return pZoneStats


if __name__ == "__main__":

    if len(sys.argv) > 1:
        # THIS PART IS FOR RUNNING THE BASH SCRIPT
        year = int(sys.argv[1])
        year2 = int(sys.argv[2])  # to turn second year comparison off: -1
        opt_config_file = sys.argv[3]  # to use defaults: 'default'
        good_stats = sys.argv[4]  # to use defaults: 'default'

        sim = OptimizePostChoice()
        sim.optimize_assignment_stats(
            year=year,
            year2=year2,
            opt_config_file=opt_config_file,
            good_stats=good_stats,
        )
    else:

        # LANGUAGE PROGRAM ZONES
        # save_path = '~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/'
        # save_path = '/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/language_zones_sept19-e/'
        # sim = SimulateZones()
        # sim.run_language_zone_opt(programs=[],save_path=save_path)

        # GENERAL ZONES

        sim = OptimizePostChoice()
        # lp_sept17,lp_sept18 = sim.language_paths_sept17_sept18()
        # sim.runMarketFromFile(zone_path_list)

        """
        year = 18
        year2 = -1
        config_file ='~/Dropbox/SFUSD/Optimization/Zones/sept17_config.txt'
        good_stats = 'easy'
        summaryfile = '~/Dropbox/SFUSD/Data/Computed/Itai/summaryM13citywideFalse.csv'
        sim = SimulateZones()
        sim.optimize_assignment_stats(year=year,year2=year2,opt_config_file=config_file,\
                good_stats=good_stats,LP_zone_path_list=zone_path_list,summaryfile=summaryfile)

        config_file ='~/Dropbox/SFUSD/Optimization/Zones/sept17_config2.txt'
        summaryfile = '~/Dropbox/SFUSD/Data/Computed/Itai/summaryM13citywideTrue.csv'
        sim.optimize_assignment_stats(year=year,year2=year2,opt_config_file=config_file,\
                good_stats=good_stats,LP_zone_path_list=zone_path_list,summaryfile=summaryfile)
        """

        # to_test = '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/*'
        # to_test_list = glob.glob(to_test)
        to_test2 = os.path.expanduser(
            "~/Dropbox/SFUSD/Optimization/Zones/Zones_Sept9/*"
        )
        # to_test_list = to_test_list + glob.glob(to_test2)
        to_test_list = glob.glob(to_test2)
        #
        # summaryfile = to_test[:-1]+'summary_sept17LPzones_2018.csv'
        # sim.runMarketFromFileList(to_test_list,zone_path_list,summaryfile=summaryfile)

        # summaryfile = to_test[:-1]+'summary_sept17LPzones_2017.csv'
        # sim.runMarketFromFileList(to_test_list,zone_path_list,summaryfile=summaryfile,year=17)

        # to_test3 = '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Itai/optGE*.csv'
        # to_test_list = glob.glob(to_test3)
        # # summaryfile = to_test[:-1]+'summary_sept17LPzones_2018b.csv'
        # # sim.runMarketFromFileList(to_test_list,zone_path_list,summaryfile=summaryfile,citywide=True,append=True)
        #
        # summaryfile = to_test[:-1]+'summary_sept17LPzones_2017b.csv'
        # sim.runMarketFromFileList(to_test_list,zone_path_list,summaryfile=summaryfile,year=17,append=True)

        # LP_zone_path_list,LP_zone_names = sim.read_language_paths()
        # LP_zone_path_list,LP_zone_names = [LP_zone_path_list[-3]],[LP_zone_names[-3]]
        # print(LP_zone_names)
        # print(LP_zone_path_list)
        #
        # df18 = pd.read_csv('~/Dropbox/SFUSD/Optimization/Zones/best_zones_sept23_nodup.csv')
        # files1 = ['~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(x) for x in df18['zone_file'] if x[:5]=='optGE']
        # files2 = ['~/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/{}.csv'.format(x) for x in df18['zone_file'] if x[:7]=='zoneopt']
        # files3 = ['~/Dropbox/SFUSD/Optimization/Zones/Zones_Sept9/{}.csv'.format(x) for x in df18['zone_file'] if x[:7]=='optzone']
        # files = [os.path.expanduser(x) for x in files1+files2+files3]
        # summaryfile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/Zones/paper_zone_appendix_may3.csv')
        # sim.runMarketFromFileList(files,LP_zone_path_list=[],summaryfile=summaryfile,\
        #     LP_zone_names=[],new_evaluator=True,append=False,useMNL=False)

        # df18 = pd.read_csv('/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/summary_sept17LPzones_2018b.csv')
        # df18 = df18.loc[df18['LP Zones']==0]
        # # files1 = ['~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(x) for x in df18['zone_file'] if x[:5]=='optGE']
        # files1 = ['~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(x) for x in df18['zone_file'] if x[:5]=='optGE']
        # # files2 = ['~/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/{}.csv'.format(x) for x in df18['zone_file'] if x[:7]=='zoneopt']
        # # files3 = ['~/Dropbox/SFUSD/Optimization/Zones/Zones_Sept9/{}.csv'.format(x) for x in df18['zone_file'] if x[:7]=='optzone']
        # files = [os.path.expanduser(x) for x in files1]#+files2+files3]
        # summaryfile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/Zones/paper_zone_appendix_may3_allzones.csv')
        # sim.runMarketFromFileList(files,LP_zone_path_list=[],summaryfile=summaryfile,\
        #     LP_zone_names=[],new_evaluator=True,append=True,useMNL=False)

        # add benchmarks to nodup file
        # benchmarks = [
        # '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Irene/Default5/Benchmark2/Assignment_real_match.csv',
        # '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Irene/Default5/Benchmark2/Assignment_CTIP1_round_merged0_policyCon1_tiesSTB_iteration1.csv',
        # '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Irene/Default5/Benchmark2/Assignment_CTIP1_round_merged0_policyCon1_tiesMTB_iteration1.csv',
        # '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Irene/Default5/Benchmark2/Assignment_CTIP0_round_merged0_policyCon1_tiesSTB_iteration1.csv',
        # '/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Irene/Default5/Benchmark2/Assignment_CTIP0_round_merged0_policyCon1_tiesMTB_iteration1.csv'
        # ]
        # benchmark_names = ['real_match','Con1 ctip1 STB','Con1 ctip1 MTB','Con1 ctip0 STB','Con1 ctip0 MTB']
        # sim.useMNL=False
        # # summaryfile = '/users/katherinementzer/SFUSD/updated_eval_assignment_benchmarks.csv'
        # sim.add_benchmarks_to_summary(summaryfile,benchmarks,benchmark_names)

        zones = ["optGE678810", "optGE862432", "optGE35532"]
        files = [
            "~/Dropbox/SFUSD/Data/Computed/Itai/selected_zones/{}.csv".format(x)
            for x in zones
        ]
        files = [os.path.expanduser(x) for x in files]
        summaryfile = os.path.expanduser(
            "~/Dropbox/SFUSD/Optimization/Zones/resegregation_viz_metrics.csv"
        )
        sim.runMarketFromFileList(
            files,
            LP_zone_path_list=[],
            summaryfile=summaryfile,
            LP_zone_names=[],
            new_evaluator=False,
            append=False,
            useMNL=False,
        )
