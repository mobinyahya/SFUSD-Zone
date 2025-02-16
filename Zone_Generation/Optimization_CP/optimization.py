import json

import ortools.sat.python.cp_model
from ortools.sat.python import cp_model

from Zone_Generation.Optimization_CP import constants
from Zone_Generation.Optimization_CP.constants import RACES


def add_optimization(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    # add_capacity_ratio_optimization(model, vm, school_df, bg_df, centroids)
    # add_boundary_optimization(model, vm, school_df, bg_df, centroids)
    frl_weight = constants.CONFIG['frl_weight']
    diversity_weight = constants.CONFIG['diversity_weight']
    distance_weight = constants.CONFIG['distance_weight']
    distance_aspect = add_distance_optimization(model, vm, school_df, bg_df, centroids)
    r_aspect = add_diversity_optimization(model, vm, school_df, bg_df, centroids)
    frl_aspect = add_frl_optimization(model, vm, school_df, bg_df, centroids)
    model.Minimize(diversity_weight * r_aspect +  frl_weight * frl_aspect + distance_weight * distance_aspect)


def add_distance_optimization(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    with open('travels.json', 'r') as f:
        travels = json.load(f)
    centroid_bgs = {}

    for zone in centroids:
        centroid_bgs[zone] = school_df[school_df['school_id'] == zone]['Block'].iloc[0]

    d_mbes_travels = []
    d_webster_travels = []
    # student_totals = []
    for bg in vm:
        student_total = int(bg_df[bg_df['Block'] == bg]['total'].iloc[0])
        d_mbes = int(travels[str(bg)][str(centroid_bgs[999])]* 100)
        d_mbes_travels.append(d_mbes * student_total)
        d_webster = int(travels[str(bg)][str(centroid_bgs[497])] * 100)
        d_webster_travels.append(d_webster * student_total)
    d_vars = cp_model.LinearExpr.WeightedSum([vm[bg] for bg in vm], d_mbes_travels) + cp_model.LinearExpr.WeightedSum(
        [vm[bg].Not() for bg in vm], d_webster_travels)


    print('d_coef', (sum(d_mbes_travels) + sum(d_webster_travels))/2)
    return d_vars


def add_frl_optimization(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    # optimize for beign as close to the district average as possible for each
    frl_total = bg_df['frl'].sum()
    total_total = bg_df['total'].sum()

    frl_coef = (bg_df['frl'] * total_total).round().astype(int).tolist()
    total_coef = (bg_df['total'] * frl_total).round().astype(int).tolist()

    block_values = list(vm.values())

    frl_block_sum = cp_model.LinearExpr.WeightedSum(block_values, frl_coef)
    total_block_sum = cp_model.LinearExpr.WeightedSum(block_values, total_coef)
    print('frl_coef:', sum(frl_coef))
    print('total_coef:', sum(total_coef))
    diff = model.NewIntVar(0, 3000000, 'diff_frl')

    model.AddAbsEquality(diff, frl_block_sum - total_block_sum)

    return diff


def add_diversity_optimization(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    # optimize for beign as close to the district average as possible for each race
    race_totals = {}
    for race in RACES:
        race_totals[race] = bg_df[race].sum()
    race_totals['total'] = bg_df['total'].sum()

    race_vars = []

    block_values = list(vm.values())

    for race in RACES:
        # TODO: Check that this this is an equivalent constraint to the one in the paper
        # print(race, bg_df[race].sum(), 'total', bg_df['student_count'].sum())
        rcoef = (bg_df[race] * race_totals['total']).round().astype(int).tolist()

        tcoef = (bg_df['total'] * race_totals[race]).round().astype(int).tolist()
        print(f'{race} coef: ', sum(rcoef))
        print('total coef', sum(tcoef))
        total_block_sum = cp_model.LinearExpr.WeightedSum(block_values, tcoef)
        race_block_sum = cp_model.LinearExpr.WeightedSum(block_values, rcoef)
        # Minimize abs(r/t - R/T)
        # where rp is a fraction
        # abs(r/t - R/T) proportional to abs(rT - Rt)

        diff = model.NewIntVar(0, 3000000, f'diff_{race}')
        model.AddAbsEquality(diff, race_block_sum - total_block_sum)
        race_vars.append(diff)
    return sum(race_vars)


def add_capacity_ratio_optimization(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    # the number of students that attend  mission bay should be 3 x the number of students that attend the other school
    # minimize the difference between the number of students that attend mission bay and the other school
    mbes_students = None
    webster_students = None
    for zone in centroids:
        block_values = list(vm[zone].values())
        # zone_capacity_coefs = pd.Series([get_bg_for_school(school_df, bg) for bg in vm[zone]])
        # zone_capacity_coefs_max = (zone_capacity_coefs  * 115).round().astype(int).tolist()
        # zone_capacity_coefs_min = (zone_capacity_coefs  * 85).round().astype(int).tolist()

        bg_counts = (bg_df['total']).round().astype(int).tolist()

        print(sum(bg_counts))
        # zone_capacity_min = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_min)
        # zone_capacity_max = cp_model.LinearExpr.WeightedSum(block_values, zone_capacity_coefs_max)
        zone_students = cp_model.LinearExpr.WeightedSum(block_values, bg_counts)
        if zone == 999:
            mbes_students = zone_students
        else:
            webster_students = zone_students

    #     need to create auxillary variable to represent absolute value, then minimize that variable
    diff = model.NewIntVar(0, 3000000, 'diff')
    model.AddAbsEquality(diff, mbes_students - 5 * webster_students)
    model.Minimize(diff)


def add_boundary_optimization(model: ortools.sat.python.cp_model.CpModel, vm, school_df, bg_df, centroids):
    boundary_vars = []

    with open('neighbors.json', 'r') as f:
        neighbors = json.load(f)
    for zone in centroids:
        for bg in vm[zone]:
            bg = int(bg)
            if str(bg) not in neighbors:
                print(bg)
                continue
            for neighbor in neighbors[str(int(bg))]:
                if neighbor == '':
                    continue
                # if neighbor not in bg_df['Block'].values:
                #     continue
                neighbor = float(neighbor)
                if float(neighbor) not in vm[zone]:
                    continue
                b = model.NewBoolVar(f"boundary_{bg}_{neighbor}")
                #             minimize the number of neighbors with different zoning
                model.AddAbsEquality(b, vm[zone][bg] - vm[zone][neighbor])
                boundary_vars.append(b)

    model.Minimize(sum(boundary_vars))
