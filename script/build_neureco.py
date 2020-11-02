import pandas as pd
from NeurEcoWrapper import NeurEco;
import os
import sys

import logging
import numpy as np
from NeurEcoWrapperDerivatives import NeurEcoDerivatives
import neurEcoInputs as neur_in
from pythonGN import *
import neurecoWeights as neur_w
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
        path = '../neureco/' + self.type + '/' + self.station + '/full_data.pkl'

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
        self.folder = '../neureco/' + self.type + '/' + self.station + '/' + 'model_' + self.model_name + '/'
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



    def build_neureco(self, x0, y0, valid_i):
        '''build Neureco without normalization : x0 and y0 must be normalized'''
        model = NeurEco()
        if os.path.exists(self.model_address):
            model.load(self.model_address)
        else:
            os.chdir(self.folder)
            x0.to_pickle('neureco_inputs.pkl')
            x0.to_csv('neureco_inputs.csv', index=False)
            np.save('valid_indices.npy', valid_i)
            np.save('neureco_outputs.npy', y0)
            model.build(
                input_data=x0,
                output_data=y0,
                write_model_to="model.neureco",
                apply_normalization=-1,
                validation_indices=valid_i)
            # write_model_output_to_directory= 'NeurEco_Build_Results')
            self.LOG_insert()
        self.model = model



    def prepare_test(self, test):
        '''normalize and update features of a testing set'''
        test = self.normalize(test)
        pca_data = self.pca_data
        for i in range(len(pca_data)):
            col = self.dynamic_features[i]
            new_col = self.pca_features[i]
            test = test.merge(pca_data[i].reset_index(), how='left', on=col)
        return test



    def evaluate(self, test, model_name=None, test_prepared=False):
        '''evaluate testing set for a given model'''
        if model_name == None:
            model = self.model
            forecast = 'neureco'
        else:
            model = NeurEco()
            model.load(self.folder + model_name + '.neureco')
            forecast = 'neureco_' + model_name
        if not test_prepared:
            dataset = self.prepare_test(test)
        else:
            dataset = test
        x0 = dataset[self.model_inputs]
        y0 = dataset[self.model_output].values

        pred_norm = model.evaluate(x0)

        '''add forecast to the dataset'''
        test.loc[:, forecast + '_norm'] = pred_norm.reshape(len(y0))
        test.loc[:, forecast] = test[forecast + '_norm'] * test.Station_max

        '''print error fro'''
        true = test[self.model_output].values.reshape(pred_norm.shape)
        pred = test[forecast].values.reshape(pred_norm.shape)
        print('error fro : ', round(la.norm(pred - true, 'fro') / la.norm(true, 'fro') * 100, 2))



    def update_inputs(self, learn, features):
        '''lberate and update unique values of features from learn '''
        x_norm = learn[self.model_inputs]
        y_norm = learn[self.model_output].values

        current_inputs = []
        for col in features:
            current_inputs = np.concatenate((current_inputs, x_norm[col].unique()))

        self.current_inputs = current_inputs

        neureco_builder = NeurEcoDerivatives()
        neureco_builder.load(self.model_address)
        num_params = neureco_builder.update_size_vec()
        original_weights = neureco_builder.get_vec()

        ''' pgn functions '''
        pgn_residu = partial(neur_in.residu, features, original_weights, x_norm, y_norm, neureco_builder)
        pgn_direct = partial(neur_in.direct, features, original_weights, x_norm, y_norm, neureco_builder)
        pgn_inverse = partial(neur_in.inverse, features, original_weights, x_norm, y_norm, neureco_builder)

        ''' pgn object '''

        pgn = pythonGN()
        pgn.setFunction(pgn_residu, pgn_direct, pgn_inverse)
        pgn.verifyDerivatives = 1
        pgn.lim_it = 5

        ''' Solver '''
        path = self.folder + 'updated_inputs.npy'
        if os.path.exists(path):
            updated_inputs = np.load(path)
        else:
            current_inputs = np.reshape(current_inputs, (-1, 1), "F")
            updated_inputs = pgn.solve(np.copy(current_inputs))
            current_inputs = current_inputs.reshape(len(current_inputs))
            updated_inputs = updated_inputs.reshape(len(updated_inputs))

        self.updated_inputs = updated_inputs
        self.current_inputs = current_inputs




    def update_weights(self, learn_norm, model_name):
        '''liberate and update weight of a given model'''
        x_norm = learn_norm[self.model_inputs]
        y_norm = learn_norm[self.model_output].values
        y_norm = y_norm.reshape(len(y_norm), 1)

        path = self.folder + model_name + ".neureco"
        neureco_builder = NeurEco()

        if os.path.exists(path):
            neureco_builder.load(path)
        else:
            neureco_builder = NeurEcoDerivatives()
            neureco_builder.load(self.model_address)
            num_params = neureco_builder.update_size_vec()
            original_weights = neureco_builder.get_vec()

            x_norm = np.asfortranarray(x_norm.T)
            y_norm = np.asfortranarray(y_norm.T)

            ''' pgn functions '''
            pgn_residu = partial(neur_w.residu, x_norm, y_norm, neureco_builder)
            pgn_direct = partial(neur_w.direct, x_norm, y_norm, neureco_builder)
            pgn_inverse = partial(neur_w.inverse, x_norm, y_norm, neureco_builder)

            ''' pgn object '''
            pgn = pythonGN()
            pgn.setFunction(pgn_residu, pgn_direct, pgn_inverse)
            pgn.verifyDerivatives = 1
            pgn.lim_it = 5
            pgn.setFunction(pgn_residu, pgn_direct, pgn_inverse)

            ''' Solver '''

            updated_weights = pgn.solve(np.copy(original_weights))
            neureco_builder.set_vec(updated_weights)
            neureco_builder.save(self.folder + model_name + ".neureco")
        self.model_modified = neureco_builder



    def LOG_insert(self):
        '''add information to logfile when building neureco model '''
        logfile = 'NeurEco_Build_Results/build/neureco_build_config.log'
        f = open(logfile, 'a')

        text = '\n\n' + 'Station : ' + self.station + '\n' \
               + 'Model Address : ' + self.model_address + '\n' \
               + 'Normalisation : ' + str(self.norm_case) + '\n' \
               + '\n Input Features : ' + str(self.model_inputs) + '\n'
        f.write(text)
        f.close()

        return
