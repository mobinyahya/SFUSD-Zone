import pandas as pd
import numpy as np
import sys, os


zones2018_path ='/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/summary_sept17LPzones_2018b.csv'
zones2018 = pd.read_csv(zones2018_path)
# remove stats using language zones
zones2018 = zones2018.loc[zones2018['LP Zones']==0]
print('Started with {} zones'.format(len(zones2018.index)))

def make_thresholds(pct,zones2018):
    thresholds = {}
    thresholds['Avg Dist'] = (1,np.percentile(zones2018['Avg Dist'],pct)) # smaller better
    thresholds['Dist > 3'] = (1,np.percentile(zones2018['Dist > 3'],pct)) # smaller better
    thresholds['BG isolation (2)'] = (1,np.percentile(zones2018['BG isolation (2)'],pct)) # smaller better
    thresholds['In-Zone Rank <= 1'] = (-1,np.percentile(-1*zones2018['In-Zone Rank <= 1'],pct)) # bigger better
    thresholds['In-Zone Rank <= 3'] = (-1,np.percentile(-1*zones2018['In-Zone Rank <= 3'],pct)) # bigger better
    thresholds['Max School FRL'] = (1,np.percentile(zones2018['Max School FRL'],pct)) # smaller better
    return thresholds

def make_thresholds_may2021(pct,zones2018):
    thresholds = {}
    thresholds['Distance Av'] = (1,np.percentile(zones2018['Distance Av'],pct)) # smaller better
    thresholds['Distance < 0.5'] = (-1,np.percentile(-1*zones2018['Distance < 0.5'],pct)) # bigger better
    thresholds['Rank Top 3'] = (-1,np.percentile(-1*zones2018['Rank Top 3'],pct)) # bigger better
    thresholds['Schools w/in 15% district FRL'] = (-1,np.percentile(-1*zones2018['Schools w/in 15% district FRL'],pct)) # bigger better
    return thresholds


def filter_by_thresholds(thresholds,filtered):
    for col,th in thresholds.items():
        filtered = filtered.loc[filtered[col]*th[0] <= th[1]]
    return filtered

def remove_duplicates(tmp):
    tmp['is_duplicate']=0
    for i,row in tmp.iterrows():
        file1 = '~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(row['zone_file'])
        if not os.path.exists(os.path.expanduser(file1)):
            continue
        df1 = pd.read_csv(file1,sep='delimiter',header=None)
        for j, row2 in tmp.loc[:i-1,:].iterrows():
            file2 = '~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(row2['zone_file'])
            if not os.path.exists(os.path.expanduser(file2)):
                continue
            df2 = pd.read_csv(file2,sep='delimiter',header=None)
            if df1.equals(df2):
                tmp.loc[i,'is_duplicate'] =1
                break
    return tmp.loc[tmp['is_duplicate']==0]

def find_unique_zones(unfiltered):
    unique = []
    for i,zone1 in enumerate(unfiltered):
        print('i =',i)
        file1 = '~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(zone1)
        if not os.path.exists(os.path.expanduser(file1)):
            print('could not find',zone1)
            continue
        df1 = pd.read_csv(file1,sep='delimiter',header=None)
        duplicate = False
        for zone2 in unique:
            file2 = '~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(zone2)
            if not os.path.exists(os.path.expanduser(file2)):
                print('could not find',zone2,'in unique')
                continue
            df2 = pd.read_csv(file2,sep='delimiter',header=None)
            if df1.equals(df2):
                duplicate = True
                break
        if not duplicate:
            print(zone1)
            unique += [zone1]
    zone_list = pd.DataFrame(unique,columns=['zone_file'])
    zone_list.to_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/unique_zones.csv')
    print(len(unique),'unique zones out of',len(unfiltered))
    return zone_list

def get_best_zones_each_type(zones2018):
    filtered = pd.DataFrame()
    cutoffs = ''
    for M in [6,9,13]:
        df = zones2018.loc[zones2018.M==M]
        for cont in [0,1]:
            df3 = df.loc[df.Contiguity==cont]
            # for cw in [True,False]:
            #     df3 = df2.loc[df2.Citywide==cw]
            cw='N/A'
            s = '\nzone type M={}, contiguity={}, citywide={}\n'.format(M,cont,cw)
            s += 'Number of zones of this type:{}\n'.format(len(df3.index))
            print('\nzone type M={}, contiguity={}, citywide={}'.format(M,cont,cw))
            print('Number of zones of this type:',len(df3.index))
            if len(df3.index)<8:
                print('Not enough zones, taking all of them')
                filtered.append(df3)
                continue
            zs = pd.DataFrame()
            pct = 25
            while (len(zs.index) > 25 or len(zs.index) < 15) and pct<100:
                if len(zs.index) > 25: pct-= 1
                if len(zs.index) < 15: pct+= 1
                thresholds = make_thresholds_may2021(pct,df3)
                zs = filter_by_thresholds(thresholds,df3)
                # zs = remove_duplicates(zs)
            s += str(thresholds)+'\n'
            cutoffs += s
            print('found {} zones using percentile {}'.format(len(zs.index),pct))
            filtered= filtered.append(zs)

    print(len(filtered.index))
    filtered.to_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_may27.csv',index=False)
    with open('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_may27_cutoffs.txt','w') as f:
        f.write(cutoffs)
    # filtered.to_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_sept22_nodup.csv',index=False)
    # with open('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_sept22_nodup_cutoffs.txt','w') as f:
    #     f.write(cutoffs)

unique2018 = pd.read_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/paper_zone_appendix_may3_allzones_nodup.csv')
# zones2018 = unique2018.merge(zones2018,how='left',on='zone_file')
get_best_zones_each_type(unique2018)

# df17 = pd.read_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_sept22_2017.csv')
# df18 = pd.read_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_sept22.csv')
# df_both = df17.merge(df18,how='inner',on='zone_file')
#
# print('2017 Zones:',len(df17.index))
# print('2018 Zones:',len(df18.index))
# print('Both:', len(df_both.index))
# print(df_both[['zone_file','M_x','Citywide_x','Contiguity_x']])
# # print(list(df_both.loc['zone_file']))
#
# tmp = df18.loc[df18.M==6]
# tmp.loc[:,'is_duplicate']=0
# for name in df18.loc[df18.M==6]['zone_file']:
#     zonefile = '~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(name)
#     print(zonefile)

# df = pd.read_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_sept23.csv')
# df = remove_duplicates(df)
# df.to_csv('/Users/katherinementzer/Dropbox/SFUSD/Optimization/Zones/best_zones_sept23_nodup.csv',index=False)

# get unique zones from the large zones list
# df18 = pd.read_csv('/Users/katherinementzer/Dropbox/SFUSD/Data/Computed/Itai/ZonesToTest/summary_sept17LPzones_2018b.csv')
# df18 = df18.loc[df18['LP Zones']==0]
# unfiltered = [x for x in df18['zone_file'] if x[:5]=='optGE']
# find_unique_zones(unfiltered)
