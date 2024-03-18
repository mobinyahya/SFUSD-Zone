import math
import os, sys, yaml
from collections import Counter

sys.path.append('../..')
sys.path.append('../../summary_statistics')
from Graphic_Visualization.zone_viz import ZoneVisualizer
# from IP_Zoning import DesignZones
from Zone_Generation.Optimization_IP.generate_zones import DesignZones, load_zones_from_file, Compute_Name
from Zone_Generation.Optimzation_Heuristics.zone_eval import * # evaluate_assignment_score, Tuning_param, boundary_trimming, evaluate_contiguity
from Helper_Functions.ReCom import *
from Helper_Functions.abstract_geography import *
from Helper_Functions.Relaxed_ReCom import Relaxed_ReCom
from Zone_Generation.Optimzation_Heuristics.local_search_zoning import *

C = B = [453, 820, 521,  513, 664,
            625, 644, 478, 435, 722,
            691, 838, 842, 723, 603,
            680, 729,  782,456,
            848,  481, 413, 569, 549,
            544, 589,  830, 867, 614,
            750, 539,  575, 507, 746, 670,
            862, 876, 488, 593, 656]
A =\
[453, 625, 830, 507, 513,
521, 867, 614, 481, 729,
593, 456, 838, 680, 691,
664, 435, 848, 413, 876,
544, 644, 549, 862, 569,
750, 782, 539, 670]

unique_A = list(set(A))

unique_A = list(set(A))

print("len(A) ", len(A))
print("'en(unique_A) ", len(unique_A))
element_counts = Counter(A)
# Extract elements that appeared more than once
non_unique_elements = [element for element, count in element_counts.items() if count > 1]

