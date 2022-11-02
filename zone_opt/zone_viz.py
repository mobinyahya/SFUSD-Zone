import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import descartes
from shapely.geometry import Point, Polygon
import os
import csv
import glob

# sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
# sns.color_palette("colorblind")

class ZoneVisualizer:
    def __init__(self,level):
        self.level=level
        self._read_data(level)



    def _get_zone_dict(self,zonefile):
        with open(zonefile,'r') as f:
            reader = csv.reader(f)
            zones = list(reader)

        zone_dict = {}
        for idx, schools in enumerate(zones):
            #print(schools)
            zone_dict = {**zone_dict,**{int(float(s)):idx for s in schools if s != ''}}
        return zone_dict

    def _read_data(self,level):
        # get school latitude and longitude
        sc_ll = pd.read_csv('~/SFUSD/Data/Cleaned/schools_rehauled_1819.csv')
        geometry = [Point(xy) for xy in zip(sc_ll['lon'], sc_ll['lat'])]
        school_geo_df = gpd.GeoDataFrame(sc_ll, crs='epsg:4326',geometry=geometry)

        # read shape file of attendance areas
        if level == 'aa':
            path = os.path.expanduser('~/Downloads/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
            self.sf = gpd.read_file(path)


        elif level == 'BlockGroup':
            path = os.path.expanduser('~/SFUSD/Census 2010_ Blocks for San Francisco/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp')
            self.sf = gpd.read_file(path)
            self.block_sf = self.sf
            self.sf['geoid10'] = self.sf['geoid10'].fillna(value=0).astype('int64', copy=False)
            df = pd.read_csv('~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv')
            df['Block'] = df['Block'].fillna(value=0).astype('int64', copy=False)
            self.sf = self.sf.merge(df,how='left',left_on='geoid10',right_on='Block')
            self.sf = self.sf.dissolve(by="BlockGroup", as_index=False)

        self.sf = self.sf.to_crs(epsg=4326)
        # school data and shape file merged
        sc_merged = gpd.sjoin(school_geo_df, self.sf,how="inner", op='intersects')
        self.labels = sc_merged.loc[sc_ll['category']=='Citywide'][['school_id','index_right','geometry']]

        # # make zone to attendance area id translator
        # translator = sc_merged.loc[sc_ll['category']=='Attendance'][['school_id','index_right','geometry']]
        # self.translator = translator.rename(columns={'school_id':'aa_zone'})

        # self.sc_merged = sc_merged.merge(self.translator, how ='left',on='index_right')
        self.sc_merged = sc_merged.rename(columns={"attendance_area": "aa_zone"})

    def visualize_zones(self,zonefile, label=False, show_schools=False, show=True, title='', centroids: list = None):
        zone_dict = self._get_zone_dict(zonefile)
        df = self.set_up_shapefile(zone_dict)

        # plot zones and attendance area school numbers
        plt.figure()#figsize=(10,10))
        ax = plt.gca()
        # if self.level=='aa':
        # print(self.sf)
        linewidth = 1 if self.level == "aa" else 0.7
        self.sf.boundary.plot(ax=plt.gca(),alpha=.4, color='white', linewidth=linewidth)
        # self.block_sf.boundary.plot(ax=plt.gca(), alpha=.1, color="gray", linewidth=1)

        if label:
            self.sc_merged.apply(lambda x: ax.annotate(s=x.aa_zone, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
            self.labels.apply(lambda x: ax.annotate(s=x.school_id, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);

        # plot zones
        # df = df.dropna()
        # print(df)
        df.plot(ax=ax,column='zone_id', cmap='tab20',legend=False)

        # plot school locations
        if show_schools:
            # aa = self.sc_merged.loc[self.sc_merged['category']=='Attendance']
            citywide = self.sc_merged.loc[self.sc_merged['category']=='Citywide']
            citywide['filter'] = citywide['school_id'].apply(lambda x: 1 if x in [449,676,479,760,796,493,814] else 0)
            citywide = gpd.GeoDataFrame(citywide.loc[citywide['filter']==1])
            print(citywide)
            citywide.apply(lambda x: ax.annotate(s=x.school_id, xy=x.geometry_x.centroid.coords[0], ha='center'),axis=1)
            # plt.scatter(aa['lon'],aa['lat'],s=10, c='red',marker='s')
            plt.scatter(citywide['lon'],citywide['lat'],s=10, c='black',marker='^')

        if centroids is not None:
            centroid_df = self.sc_merged.loc[self.sc_merged['school_id'].isin(centroids)]
            plt.scatter(centroid_df['lon'], centroid_df['lat'], s=10, c='black', marker='^', zorder=50)# s=50, c='black', alpha=.6, zorder=50)

        if title=='':
            # plt.title(zonefile.split('/')[-1][:-4])
            plt.title('')
        else:
            plt.title(title)
        plt.gca().set_yticks([])
        plt.gca().set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        ax.set_xlim(-122.525,-122.350)
        ax.set_ylim(37.70,37.84)
        if show:
            plt.show()

    def set_up_shapefile(self, zone_dict):
        if self.level == 'aa':
            self.sc_merged['zone_id'] = self.sc_merged['aa_zone'].replace(zone_dict)

            df = self.sf.merge(self.sc_merged, how='left', right_on='index_right', left_index=True)
        elif self.level == 'BlockGroup':
            self.sf['zone_id'] = self.sf['BlockGroup'].replace(zone_dict)
            self.sf.dropna(subset=['zone_id'], inplace=True)
            self.sf['filter'] = self.sf['zone_id'].apply(lambda x: 1 if int(x) in range(20) else 0)
            df = self.sf.loc[self.sf['filter'] == 1]
        return df

    def visualize_zones_with_column(self, zonefile, column, label=False, show_schools=False, show=True,title='',vmax=None, vmin=None):
        zone_dict = self._get_zone_dict(zonefile)
        df = self.set_up_shapefile(zone_dict)

        zone_avg = df.groupby('zone_id').mean()
        replace_with_average = dict(zip(zone_avg.index, zone_avg[column]))
        df[column] = df['zone_id'].replace(replace_with_average)

        # plot zones and attendance area school numbers
        plt.figure()#figsize=(10,10))
        ax = plt.gca()
        self.sf.boundary.plot(ax=plt.gca(),alpha=.4, color='white', linewidth=1)

        if label:
            self.translator.apply(lambda x: ax.annotate(s=x.aa_zone, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
            self.labels.apply(lambda x: ax.annotate(s=x.school_id, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);

        # plot zones
        # print(len(df))
        df = df.loc[df['category']=='Attendance']
        print(len(df))
        df.plot(ax=ax,column=column, legend=True, vmax=vmax, vmin=vmin)#cmap='tab20',legend=False)

        # plot school locations
        if show_schools:
            # aa = self.sc_merged.loc[self.sc_merged['category']=='Attendance']
            citywide = self.sc_merged.loc[self.sc_merged['category']=='Citywide']
            citywide['filter'] = citywide['school_id'].apply(lambda x: 1 if x in [449,676,479,760,796,493,814] else 0)
            citywide = gpd.GeoDataFrame(citywide.loc[citywide['filter']==1])
            print(citywide)
            citywide.apply(lambda x: ax.annotate(s=x.school_id, xy=x.geometry_x.centroid.coords[0], ha='center'),axis=1)
            # plt.scatter(aa['lon'],aa['lat'],s=10, c='red',marker='s')
            plt.scatter(citywide['lon'],citywide['lat'],s=10, c='black',marker='^')

        if title=='':
            # plt.title(zonefile.split('/')[-1][:-4])
            plt.title('')
        else:
            plt.title(title)
        plt.gca().set_yticks([])
        plt.gca().set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()
        # ax.set_xlim(-122.525,-122.350)
        # ax.set_ylim(37.70,37.84)
        if show:
            plt.show()


    def visualize_zones_from_dict(self,zone_dict,label=True,show=True,col='zone_id',title=''):
        #zone_dict = self._get_zone_dict(zonefile)
        self.sc_merged['zone_id'] = self.sc_merged['aa_zone'].replace(zone_dict)
        df = self.sf.merge(self.sc_merged,how='left',right_on='index_right',left_index=True)


        # plot zones and attendance area school numbers
        plt.figure(figsize=(15,15))
        ax =self.sf.boundary.plot(ax=plt.gca(),alpha=0.4,color='grey')
        #ax =self.sf.boundary.plot(alpha=0.4,color='grey')
        if label:
            self.translator.apply(lambda x: ax.annotate(s=x.aa_zone, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);
            self.labels.apply(lambda x: ax.annotate(s=x.school_id, xy=x.geometry.centroid.coords[0], ha='center'),axis=1);

        # plot zones
        df.plot(ax=ax,column='zone_id', cmap='tab20',legend=True)
        plt.title(title)

        # plot school locations
        aa = self.sc_merged.loc[self.sc_merged['category']=='Attendance']
        citywide = self.sc_merged.loc[self.sc_merged['category']=='Citywide']
        plt.scatter(aa['lon'],aa['lat'],s=10, c='red',marker='s')
        plt.scatter(citywide['lon'],citywide['lat'],s=10, c='black',marker='^')
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.gca().set_xlim(-122.525,-122.350)
        plt.gca().set_ylim(37.70,37.84)

        if show:
           plt.show()
        return plt


    def get_sept9_zones(self):
        zone_path = '~/Dropbox/SFUSD/Optimization/Zones/tmp'

        zone_files = {}
        zone_files['M6a'] = zone_path+'/Large/optzone_M_6_Shortage_100_cover_3.6210470430275024_contiguity_1_frl_0.7_HOCidx1_0.7-5_MetStandards_0.7.csv'
        zone_files['M6b'] = zone_path+'/Large/optzone_M_6_Shortage_100_cover_4.621047043027502_contiguity_1_frl_0.7_HOCidx1_0.7-5_MetStandards_0.7.csv'
        zone_files['M6c'] = zone_path+'/Large/optzone_M_6_Shortage_100_cover_4.621047043027502_contiguity_1_frl_0.9_HOCidx1_0.7-5_MetStandards_0.7.csv'
        zone_files['M6d'] = zone_path+'/Large/optzone_M_6_Shortage_100_frl_0.7_HOCidx1_0.7-5_MetStandards_0.7.csv'
        zone_files['M6e'] = zone_path+'/Large/optzone_M_6_Shortage_100_frl_0.9_HOCidx1_0.7-5_MetStandards_0.7.csv'
        zone_files['M8a'] = zone_path+'/Large/optzone_M_8_Shortage_100_cover_3.6210470430275024_contiguity_1_frl_0.7_HOCidx1_0.7-5_MetStandards_0.7.csv'
        zone_files['M8b'] = zone_path+'/Large/optzone_M_8_Shortage_100_frl_0.85_HOCidx1_0.6-5_MetStandards_0.7.csv'
        zone_files['M10a'] = zone_path+'/Medium/optzone_M_10_Shortage_150_frl_0.6_HOCidx1_0.5-5_MetStandards_0.7.csv'
        zone_files['M12a'] = zone_path+'/Medium/optzone_M_12_Shortage_100_frl_0.7_HOCidx1_0.75-5.csv'
        zone_files['M12b'] = zone_path+'/Medium/optzone_M_12_Shortage_150_frl_0.6_HOCidx1_0.4-5_MetStandards_0.7.csv'
        zone_files['M12c'] = zone_path+'/Medium/optzone_M_12_Shortage_150_frl_0.6_HOCidx1_0.6-5_MetStandards_0.7.csv'
        zone_files['M12d'] = zone_path+'/Medium/optzone_M_12_Shortage_150_frl_0.75_HOCidx1_0.3-5.csv'
        zone_files['M13a'] = zone_path+'/new_9_8_2020/optzone_M_13_Shortage_120_cover_4.5_contiguity_1_frl_0.45.csv'
        zone_files['M13b'] = zone_path+'/new_9_8_2020/optzone_M_13_Shortage_120_cover_4.5_contiguity_1_frl_0.45_HOCidx1_0.3-5_MetStandards_0.3.csv'
        zone_files['M13c'] = zone_path+'/new_9_8_2020/optzone_M_13_Shortage_120_cover_5_contiguity_1_frl_0.45_HOCidx1_0.3-5_MetStandards_0.3.csv'
        zone_files['M13d'] = '~/Dropbox/SFUSD/Optimization/Zones/selected_zones/Medium zones/optzone_M_13_Balance_61.406153846153856_frl_0.65_HOCidx1_0.7-1.3_AALPIScore_0.6-1.5_engscores1819_0.75-1.25.csv'
        zone_files['M13e'] = '~/Dropbox/SFUSD/Optimization/Zones/tmp/new_9_8_2020/optzone_M_13_Shortage_100_frl_0.7_HOCidx1_0.2-5.csv'
        zone_files['M13f'] = '~/Dropbox/SFUSD/Optimization/Zones/tmp/new_9_8_2020/optzone_M_13_Shortage_100_frl_0.7_HOCidx1_0.2-5_MetStandards_0.3.csv'
        zone_files['M13g'] = '~/Dropbox/SFUSD/Optimization/Zones/tmp/new_9_8_2020/optzone_M_13_Shortage_100_frl_0.65_HOCidx1_0.3-5_MetStandards_0.4.csv'

        zone_path2 = '~/Dropbox/SFUSD/Optimization/Zones/selected_zones/Large zones/'

        zone_files['M6a-2'] = zone_path2+'optzone_M_6_Balance_118.79166666666667_neighbor_1_frl_0.7_HOCidx1_0.8-1.2_AALPIScore_0.7-1.3_engscores1819_0.75-1.25.csv'
        zone_files['M6b-2'] = zone_path2+'optzone_M_6_Balance_133.04666666666668_cover_3.4_contiguity_1_frl_0.8_HOCidx1_0.75-1.25_AALPIScore_0.72-1.3_engscores1819_0.8-1.25.csv'
        zone_files['M6c-2'] = zone_path2+'optzone_M_6_Balance_133.04666666666668_cover_3.4_contiguity_1_frl_0.95_HOCidx1_0.75-1.25_AALPIScore_0.72-1.3_engscores1819_0.8-1.25.csv'
        zone_files['M6d-2'] = zone_path2+'optzone_M_6_Balance_133.04666666666668_frl_0.8_HOCidx1_0.8-1.2_AALPIScore_0.7-1.3_engscores1819_0.75-1.25.csv'
        zone_files['M6e-2'] = zone_path2+'optzone_M_6_Balance_133.04666666666668_frl_0.8_HOCidx1_0.8-1.2_AALPIScore_0.7-1.3_engscores1819_0.83-1.25.csv'
        zone_files['M6f-2'] = zone_path2+'optzone_M_6_Balance_133.04666666666668_neighbor_1_frl_0.9_HOCidx1_0.7-1.3_AALPIScore_0.7-1.3_engscores1819_0.8-1.25.csv'
        zone_files['M6g-2'] = zone_path2+'optzone_M_6_Balance_133.04666666666668_neighbor_1_frl_0.9_HOCidx1_0.8-1.2_AALPIScore_0.7-1.3_engscores1819_0.75-1.25.csv'
        return zone_files


    def save_sept9_visualizations(self):
        zone_files = self.get_sept9_zones()
        save_path = os.path.expanduser('~/Dropbox/SFUSD/Zones/Sept9_zone_viz/')
        for k,v in zone_files.items():
            self.visualize_zones(os.path.expanduser(v),label=False,show=False,title=k)
            plt.savefig(save_path+k)


    def read_language_zone(self,zone_file):
        with open(os.path.expanduser(zone_file),'r') as f:
            s = f.readline()
        aa2prog = eval(s)
        aa2prog = {k:str(v) for k,v in aa2prog.items()}
        data =np.array([list(aa2prog.keys()),list(aa2prog.values())],dtype=object).T
        df = pd.DataFrame(data=data,columns=['aa','programs'])
        prog2zone = dict(zip(df['programs'].unique(),range(len(df['programs'].unique()))))
        df['zone_id'] = df['programs'].replace(prog2zone)
        aa2zone = dict(zip(df['aa'],df['zone_id']))
        return aa2zone


    def save_sept18_visualizations(self):
        zone_path_list = {}
        zone_path_list['CB'] = '~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/optCB6184.txt'
        zone_path_list['CE'] ='~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/optCE6184.txt'
        zone_path_list['FB'] ='~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/optFB6184.txt'
        zone_path_list['JE'] ='~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/optJE6184.txt'
        zone_path_list['ME'] ='~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/optME6184.txt'
        zone_path_list['SB'] = '~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/optSB6184.txt'
        zone_path_list['SE'] = '~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept17/optSE40487.txt'
        save_path = os.path.expanduser('~/Dropbox/SFUSD/Zones/sept18_zone_viz/')
        for k,v in zone_path_list.items():
            zone_dict = self.read_language_zone(v)
            self.visualize_zones_from_dict(zone_dict,label=False,show=False,title=k)
            plt.savefig(save_path+k)


    def save_directory_languge_visualizations(self,directory):
        zone_path_list = glob.glob(os.path.expanduser(directory)+'*.txt')
        lp_zone_dict = {}
        for path in zone_path_list:
            if path.split('/')[-1].find('_') == -1:
                name = path.split('/')[-1][3:-4]
                if not os.path.exists(os.path.expanduser(directory)+name+'.png')\
                    and path.split('/')[-1][:-4]!='readme':
                    lp_zone_dict[name] = path

        save_path = os.path.expanduser(directory)
        for k,v in lp_zone_dict.items():
            zone_dict = self.read_language_zone(v)
            self.visualize_zones_from_dict(zone_dict,label=False,show=False,title=k)
            plt.savefig(save_path+k)

if __name__ =='__main__':
    # zonefile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/Zones/optzone_distance_4.5_frl_0.7_HOCidx1_1.csv')
    # zv = ZoneVisualizer('aa')

    # zonefile = os.path.expanduser('~/Dropbox/SFUSD/Optimization/Zones/optblockgroup_distance_4.5_frl_0.7_HOCidx1_0.9-1.6.csv')
    zonefile = os.path.expanduser('/Users/katherinementzer/Downloads/GE_Zoning_1_1_3_2_1_centroids 8-zone-28_BG.csv')
    zv = ZoneVisualizer('BlockGroup')
    #
    zv.visualize_zones(zonefile,show=True, show_schools=False, label=False)

    # zv.save_sept9_visualizations()
    # zv.save_sept18_visualizations()
    # zv.save_directory_languge_visualizations('~/Dropbox/SFUSD/Optimization/Zones/language_zones_sept19-e/')


    # top_zones = ['optGE601328', 'optGE1541671', 'optGE1903962', 'optGE116961', \
    # 'optGE1361398', 'optGE1724377', 'optGE1323765', 'optGE541557', 'optGE316443', \
    # 'optGE1845090', 'optGE1098922', 'optGE1730029']
    # top_zones =['optGE922908', 'optGE341062', 'optGE264644', 'optGE59369', 'optGE106243', 'optGE302509', 'optGE158250', 'optGE1013353', 'optGE1904536']
    # for_max = ['optGE59369','optGE922908','optGE1013353','optGE1904536']
    # for_max = ['optGE862432']
    # for_presentation = ['optGE1730029', 'optGE1098922', 'optGE424194', 'optGE114846', 'optGE134936', 'optGE1105753']
    # for_presentation =[
    #     '/Users/katherinementzer/Downloads/aa_9_noconcentration_zoning.csv',
    #     '/Users/katherinementzer/Downloads/aa_9_representetiveness_zoning.csv',
    #     '/Users/katherinementzer/Downloads/aa_6_noconcentration_zoning.csv',
    #     '/Users/katherinementzer/Downloads/aa_6_representetiveness_zoning.csv'
    # ]
    # for zone in for_presentation:
    #     zonefile = os.path.expanduser(zone)
    #     name = zone.split('/')[-1][:-4]
    #     # zonefile = os.path.expanduser('~/Dropbox/SFUSD/Data/Computed/Itai/{}.csv'.format(name))
    #     print(zonefile)
    #     zv.visualize_zones(zonefile,show=False, show_schools=False, label=False)
    #     plt.savefig('/Users/katherinementzer/Downloads/'+name, transparent=True)
