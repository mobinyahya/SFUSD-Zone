import matplotlib.pyplot as plt
import pandas as pd
from sim_engine import SimulateZones
import os

def run_feasibility_simulation(label=''):
    sim = SimulateZones()
    # find feasibility region for frl versus cover distance
    for size in ['medium']:#['small','medium','large']:
        config = '~/Dropbox/SFUSD/Optimization/Zones/frl_vs_dist_feasibility/config_frl_v_dist_{}{}.txt'.format(size,label)
        summary = '~/Dropbox/SFUSD/Optimization/Zones/frl_vs_dist_feasibility/summary_{}{}.csv'.format(size,label)
        sim.zone_size_frl_feasiblity(opt_config_file=config,summaryfile=summary)

def make_figs(label=''):
    for size in ['small','medium','large']:
        summary = '~/Dropbox/SFUSD/Optimization/Zones/frl_vs_dist_feasibility/summary_{}{}.csv'.format(size,label)
        # sim.zone_size_frl_feasiblity(opt_config_file=config,summaryfile=summary)
        df = pd.read_csv(summary)
        grid = pd.pivot_table(df,values='feasible',index='cover_distance',columns='frl_deviation')
        for col in grid.columns:
            for i in grid.index:
                if grid.loc[i,col]>0:
                    grid.loc[i:,col] = 1
                    break
        grid = grid.iloc[::-1]
        print(grid)
        fig = plt.figure()
        img = plt.imshow(grid)
        plt.gca().set_xticks(range(len(grid.columns)))
        plt.gca().set_yticks(range(len(grid.index)))
        plt.gca().set_xticklabels(grid.columns)
        plt.gca().set_yticklabels(grid.index)
        plt.xlabel('FRL % Deviation')
        plt.ylabel('Cover Distance (mi)')
        plt.title('{} Zone FRL vs Dist Tradeoff'.format(size.capitalize()))
        fig.colorbar(img)
        filename = os.path.expanduser('~/Dropbox/SFUSD/Paper/frl_v_dist_tradeoff_{}{}.png'.format(size,label))
        plt.savefig(filename)


if __name__ == '__main__':
    # run_feasibility_simulation(label='_balance_shortage')
    make_figs(label='_balance_shortage')
