import pandas as pd
import os
import sys

import logging
import numpy as np
import random
from PCA import *

import read_data_sirea as sirea



##########################################################################################
#################################### DATA PREPARATION ####################################
##########################################################################################

class data_reader():
    def __init__(self, station, type):
        self.station = station
        self.type = type
        path = 'neureco/' + self.type + '/' + self.station + '/full_data.pkl'

        if 'Sirea' in type:
            if 'prod' in type:
                self.output = 'prod_kW'
            else:
                self.output = 'load_kW'
            self.meteo_features = ['10 metre U wind component', '10 metre V wind component', 'Total cloud cover',
                                   '2 metre temperature', 'Surface net solar radiation',
                                   'Surface net thermal radiation']

            if os.path.exists(path):
                self.full_data = pd.read_pickle(path)
            else:
                data_reader = sirea.main(station)
                self.full_data = data_reader.full_data

        elif 'Terega' in type:
            self.output = 'VOLUME'
            self.meteo_features = ['Humidity', 'year', 'Rain', 'Temperature', 'Wind']
            if os.path.exists(path):
                self.full_data = pd.read_pickle(path)
            else:
                data_reader = terega.main(station)
                self.full_data = data_reader.full_data

        elif 'meteoswift' in type:
            self.output = 'ff10'
            meteo = ['temp10', 'rh10', 'u10', 'v10', 'u100', 'v100']
            liste = ['_p' + str(i) for i in range(255)]
            self.meteo_features = [n + m for m in liste for n in meteo]
            if os.path.exists(path):
                self.full_data = pd.read_pickle(path)
            else:
                data_reader = meteoswift.main(station)
                self.full_data = data_reader.full_data


##########################################################################################
#################################### NEURECO FUNCTIONS ####################################
##########################################################################################

class build_neureco():
    def __init__(self, d_read, inputs, model_name=0, norm_case=3):
        self.station = d_read.station
        self.type = d_read.type
        self.full_data = d_read.full_data
        self.model_inputs = inputs
        self.model_output = d_read.output
        self.model_name = model_name
        self.folder = 'neureco/' + self.type + '/' + self.station + '/' + 'model_' + self.model_name + '/'
        self.model_address = self.folder + 'model.neureco'
        self.norm_case = norm_case
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)



    def split_data(self):
        '''split full_data into learning set, testing set and validation set'''
        df = self.full_data
        stat = df.Station_name.unique()
        df.Station = df.Station_name.replace(stat, [i for i in range(len(stat))])

        indices = df[self.model_inputs + [self.model_output]].reset_index().dropna().index
        data = df.iloc[indices]
        learn = pd.DataFrame()
        test1 = pd.DataFrame()
        test2 = pd.DataFrame()
        stations = data.Station.unique()

        for stat in stations:
            df = data[data.Station == stat]
            year = df.index.year.max()
            df1 = df[(df.index.month == 9) & (df.index.year == year - 1)]
            df2 = df[(df.index.month == 1) & (df.index.year == year)]
            df3 = pd.concat([df1, df2, df]).drop_duplicates(keep=False)
            ''' number of weeks in train dataset'''
            n = len(df3.week.unique()) // 4
            random_weeks = random.sample(list(df3.week.unique()), n)
            df3['valid'] = df3.week.apply(lambda x: x in random_weeks)
            test1 = test1.append(df1)
            test2 = test2.append(df2)
            learn = learn.append(df3)

        valid = learn.reset_index()
        valid_indices = valid[valid.valid].index
        self.model_valid = valid_indices
        self.dataset = [learn, test1, test2]



    def normalize(self, data, meteo=False):
        '''data normalization'''
        norm_case = self.norm_case
        output = self.model_output
        y = self.dataset[0][output].values
        y = y.reshape(len(y), 1)
        self.max_out = np.max(y)

        output_cols = [col for col in data.columns if output in col]
        data_norm = data.copy()
        if norm_case == 1:
            '''no normalization'''
            self.max_out = 1
        if norm_case == 2:
            '''norm on load only'''
            data_norm[output_cols] = data.loc[:, output_cols] / self.max_out
        if norm_case == 3:
            '''normalization on output only depending on station'''
            max_stat = data_norm.Station_max.unique()
            stat = data_norm.Station.unique()
            for i in range(len(stat)):
                data_norm.loc[data_norm.Station == stat[i], output_cols] = data.loc[data_norm.Station == stat[
                    i], output_cols] / max_stat[i]
            if meteo:
                cols = self.meteo_features
                data_norm[cols] = data_norm[cols] / abs(data_norm[cols]).max()
        return data_norm



    def PCA(self, dynamic_features, static_features):
        '''PCA on dynamic features'''
        learn = self.model_learn
        pca_features = []
        pca_data = []
        for feature in dynamic_features.copy():
            pca_result = PCA(learn, self.model_output, feature)
            if len(pca_result) == 0:
                '''not enough data for PCA on this feature => take cos and sin instead'''
                dynamic_features.remove(feature)
                static_features.append('cos_' + feature)
                static_features.append('sin_' + feature)
                print(
                    ' >>> alert : ' + feature + ' removed from dynamic features, cos() added in static_features instead')
            else:
                pca_features += pca_result.columns.tolist()
                pca_data.append(pca_result)
        self.dynamic_features = dynamic_features
        self.static_features = static_features
        self.pca_data = pca_data
        self.pca_features = pca_features

