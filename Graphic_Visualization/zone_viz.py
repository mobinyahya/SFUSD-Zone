import csv
import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point



class ZoneVisualizer:
    def __init__(self, level, year='1819'):
        self.level = level
        self.year = year
        self._read_data()

    def _get_zone_dict(self, zonefile):
        with open(zonefile, 'r') as f:
            reader = csv.reader(f)
            zones = list(reader)

        zone_dict = {}
        for idx, schools in enumerate(zones):
            # print(schools)
            zone_dict = {**zone_dict, **{int(float(s)): idx for s in schools if s != ''}}
        return zone_dict

    def _read_data(self):
        # pd.set_option("display.max_rows", None, "display.max_columns", None)

        # get school latitude and longitude
        sc_ll = pd.read_csv(f'~/SFUSD/Data/Cleaned/schools_rehauled_{self.year}.csv')
        geometry = [Point(xy) for xy in zip(sc_ll['lon'], sc_ll['lat'])]
        # almost the sc_ll file (school data) + Points(lon/lat) of them, in a geo-data-frame
        school_geo_df = gpd.GeoDataFrame(sc_ll, crs='epsg:4326', geometry=geometry)
        # read shape file of attendance areas
        if self.level == 'idschoolattendance':
            # Used to be in downloads folder, but I moved it to SFUSD folder
            path = os.path.expanduser('~/SFUSD/drive-download-20200216T210200Z-001/2013 ESAAs SFUSD.shp')
            self.sf = gpd.read_file(path)

        elif (self.level == 'BlockGroup') | (self.level == 'Block'):
            # Changed to shape files folder from 2010 census
            path = os.path.expanduser('~/SFUSD/shapefiles/geo_export_d4e9e90c-ff77-4dc9-a766-6a1a7f7d9f9c.shp')
            self.sf = gpd.read_file(path)
            self.sf['geoid10'] = self.sf['geoid10'].fillna(value=0).astype('int64', copy=False)

            df = pd.read_csv('~/Dropbox/SFUSD/Optimization/block_blockgroup_tract.csv')
            df['Block'] = df['Block'].fillna(value=0).astype('int64', copy=False)
            self.sf = self.sf.merge(df, how='left', left_on='geoid10', right_on='Block')
        self.sf = self.sf.to_crs(epsg=4326)

        if self.level == 'idschoolattendance':
            # school data and shape file merged
            # sc_merged includes all data in school_geo_df, and also showing in which attendance area(?) each Point is.
            # This info is available in an extra columnt, 'index_right'
            sc_merged = gpd.sjoin(school_geo_df, self.sf, how="inner", op='intersects')
            self.labels = sc_merged.loc[sc_ll['category'] == 'Citywide'][['school_id', 'index_right', 'geometry']]

            # make zone to attendance area id translator
            translator = sc_merged.loc[sc_ll['category'] == 'Attendance'][['school_id', 'index_right', 'geometry']]
            self.translator = translator.rename(columns={'school_id': 'aa_zone'})
            self.sc_merged = sc_merged.merge(self.translator, how='left', on='index_right')

    # def visualize_zones_from_dict(self, zone_dict,label=True,show=True, col='zone_id',title=''):
    def visualize_zones_from_dict(self, zone_dict, label=False, show=True, col='zone_id', title='',
                                  centroid_location=-1, save_name="don't save",):

        # for each aa_zone (former school_id), change it with whichever zone index this gets
        # matched to based on the LP solution in zone_dict
        if self.level == 'idschoolattendance':
            self.sc_merged['zone_id'] = self.sc_merged['aa_zone'].replace(zone_dict)
            df = self.sf.merge(self.sc_merged, how='left', right_on='index_right', left_index=True)
            # df['zone_id'] = df['aa_zone'].replace(zone_dict)

            df['filter'] = df['zone_id'].apply(lambda x: 1 if int(x) in range(20) else 0)
            df = df.loc[df['filter'] == 1]

            plt.figure(figsize=(15, 15))
            ax = self.sf.boundary.plot(ax=plt.gca(), alpha=0.4, color='grey')

            if label:
                self.translator.apply(
                    lambda x: ax.annotate(fontsize=12, s=x.aa_zone, xy=x.geometry.centroid.coords[0], ha='center'),
                    axis=1);
                self.labels.apply(
                    lambda x: ax.annotate(fontsize=12, s=x.school_id, xy=x.geometry.centroid.coords[0], ha='center'),
                    axis=1);

        elif (self.level == 'BlockGroup') | (self.level == 'Block'):
            plt.figure(figsize=(20, 20))
            ax = self.sf.boundary.plot(ax=plt.gca(), alpha=0.4, color='grey')

            self.sf.dropna(subset=[self.level], inplace=True)
            # drop rows that have NaN for zone_id
            if label:
                # self.sf.apply(lambda x: ax.annotate(fontsize=8, s= int(x.BlockGroup), xy=x.geometry.centroid.coords[0], ha='center'), axis=1);
                self.sf.apply(lambda x: ax.annotate(fontsize=8,
                                                    text=int(x.BlockGroup),
                                                    xy=x.geometry.centroid.coords[0], ha='center'), axis=1);
                # self.sf.apply(lambda x: ax.annotate(fontsize=15, s= int(x.Block) if int(x.BlockGroup) == 60750179021 else ".", xy=x.geometry.centroid.coords[0], ha='center'), axis=1);
                # self.sf.apply(lambda x: ax.annotate(fontsize=15, s= int(x.Block) if int(x.Block) == 60750255002023 else ".", xy=x.geometry.centroid.coords[0], ha='center'), axis=1);

            self.sf['zone_id'] = self.sf[self.level].replace(zone_dict)
            self.sf['filter'] = self.sf['zone_id'].apply(lambda x: 1 if int(x) in range(20) else 0)
            df = self.sf.loc[self.sf['filter'] == 1]

        # plot zones
        df.plot(ax=ax, column='zone_id', cmap='tab20', legend=True, aspect=1)
        plt.title(title)

        # plot centroid locations
        plt.scatter(centroid_location['lon'], centroid_location['lat'], s=20, c='black', marker='s')
        # plt.scatter(bb['lon'], bb['lat'], s=20, c='red', marker='s')

        # # plot school locations
        # aa = self.sc_merged.loc[self.sc_merged['category']=='Attendance']
        # citywide = self.sc_merged.loc[self.sc_merged['category']=='Citywide']
        # print("this is the number of all schools")
        # print(len(aa)+len(citywide))
        # plt.scatter(aa['lon'],aa['lat'],s=10, c='red',marker='s')
        # plt.scatter(citywide['lon'],citywide['lat'],s=10, c='black',marker='^')

        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.gca().set_xlim(-122.525, -122.350)
        plt.gca().set_ylim(37.70, 37.84)

        if show:
            if save_name != "don't save":
                path = os.path.expanduser("~/SFUSD/Visualization_Tool_Data/")
                print(path + save_name + '.png')
                plt.savefig(path + save_name + '.png')
            # plt.show()
            plt.close()
        print("Finished plotting")
        return plt

    def visualize_SpEd(self, sped_df, sped_types, label=False, centroid_location=-1):
        # Note TOD: This visualization assumes that each school has at most 1 program to be visualized.
        # if there is a school with multiple programs to be visualized,
        # we mix them here and assume it's one big program (with it's code being the latter one)
        plt.figure(figsize=(40, 40))
        self.sf.dropna(subset=[self.level], inplace=True)
        df = self.sf

        # plot zones
        df.plot()

        # plot centroid locations
        colors = ['w', 'b', 'y', 'r', 'c', 'm', 'k', 'g']
        colors_meaning = {'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow',
                          'k': 'black', 'w': 'white'}

        sped_df['color_new'] = 0
        sped_df['color_old'] = 0
        for idx, row in sped_df.iterrows():
            for t in range(1, len(sped_types)):
                if row[str(sped_types[t] + "_classes_new")] > 0:
                    sped_df['color_new'][idx] = colors[t]
                if row[str(sped_types[t] + "_classes")] > 0:
                    sped_df['color_old'][idx] = colors[t]

        # size of the point will be number of sped classes the school has.
        # we count for t starting 1, because sped_types[0] == 'GE'
        sped_df['size_new'] = sum(sped_df[str(t + "_classes_new")] for t in sped_types[1:len(sped_types)])
        sped_df['size_old'] = sum(sped_df[str(t + "_classes")] for t in sped_types[1:len(sped_types)])

        sped_df_new = sped_df.loc[sped_df['size_new'] >= 1]
        sped_df_old = sped_df.loc[sped_df['size_old'] >= 1]

        plt.scatter(sped_df_new['lon'], sped_df_new['lat'], s=10 * sped_df_new['size_new'], c=sped_df_new['color_new'],
                    marker='s')
        plt.scatter(sped_df_old['lon'], sped_df_old['lat'], s=10 * sped_df_old['size_old'], c=sped_df_old['color_old'],
                    marker='*')

        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.gca().set_xlim(-122.525, -122.350)
        plt.gca().set_ylim(37.70, 37.84)
        plt.show()
        for t in range(1, len(sped_types)):
            print("program: " + str(sped_types[t]) + "   is colored: " + str(colors_meaning[colors[t]]))
        print("Finished plotting")

        return plt

    def read_language_zone(self, zone_file):
        with open(os.path.expanduser(zone_file), 'r') as f:
            s = f.readline()
        aa2prog = eval(s)
        aa2prog = {k: str(v) for k, v in aa2prog.items()}
        data = np.array([list(aa2prog.keys()), list(aa2prog.values())], dtype=object).T
        df = pd.DataFrame(data=data, columns=['aa', 'programs'])
        prog2zone = dict(zip(df['programs'].unique(), range(len(df['programs'].unique()))))
        df['zone_id'] = df['programs'].replace(prog2zone)
        aa2zone = dict(zip(df['aa'], df['zone_id']))
        return aa2zone

    def save_directory_languge_visualizations(self, directory):
        zone_path_list = glob.glob(os.path.expanduser(directory) + '*.txt')
        lp_zone_dict = {}
        for path in zone_path_list:
            if path.split('/')[-1].find('_') == -1:
                name = path.split('/')[-1][3:-4]
                if not os.path.exists(os.path.expanduser(directory) + name + '.png') \
                        and path.split('/')[-1][:-4] != 'readme':
                    lp_zone_dict[name] = path

        save_path = os.path.expanduser(directory)
        for k, v in lp_zone_dict.items():
            zone_dict = self.read_language_zone(v)
            self.visualize_zones_from_dict(zone_dict, label=False, show=False, title=k)
            plt.savefig(save_path + k)
