import pandas as pd
import numpy as np
import os
from datetime import datetime
import netCDF4
from datetime_features import add_temp_features


def f(x, sigma, mu, inte):
    # return gaussian(x) depending on sigma, mu and the value of the integral
    a = (x - mu) / sigma
    b = inte / (sigma * np.sqrt(2 * np.pi))
    return b * np.exp(-1 / 2 * a * a)



def test_func(args):
    sigma = 3
    mu = 14
    return f(args[3], sigma, mu, args[4])



class data_reader():
    def __init__(self, station):

        # extract the csv file from https://energie.microserver.fr/ and update the following path
        if 'Almaric' in station:
            self.station = 'Marceau Almaric'
            folder = '../projects/2. SIREA/MARCEAU ALMARIC/'
            meteo = folder + '/meteo/'
            raw_data = folder + '/raw_data/'
            self.folder = folder

        if 'SYD TGBT' in station:
            folder = '../projects/2. SIREA/SYD TGBT/'
            meteo = folder + '/meteo/'
            self.folder = folder

            if '1' in station:
                self.station = 'SYD TGBT1'
                self.station_num = 1
                raw_data = folder + '/raw_data_1/'
            if '2' in station:
                self.station = 'SYD TGBT2'
                self.station_num = 2
                raw_data = folder + '/raw_data_2/'
            if '3' in station:
                self.station = 'SYD TGBT3'
                self.station_num = 3
                raw_data = folder + '/raw_data_3/'

        self.path_meteo = [meteo + i for i in os.listdir(meteo)]
        self.path_data = [raw_data + i for i in os.listdir(raw_data)]

    def emcwf_to_pandas(self, path):
        #read emcwf netCDF data and return panda dataframe
        nc = netCDF4.Dataset(path)
        times = nc.variables['time']
        time_str = netCDF4.num2date(times[:],
                                    times.units).astype('datetime64[ns]')

        df = pd.DataFrame(data={'time': time_str})

        variables = nc.variables.keys()
        for var in variables:
            if var in ['time', 'expver']:
                pass
            elif var == 'longitude':
                long = nc.variables['longitude'][:].data[0]
            elif var == 'latitude':
                lat = nc.variables['latitude'][:].data[0]
            else:
                col = nc.variables[var][:, 0, 0].data
                name = nc.variables[var].long_name
                df[name] = col
        df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H')
        coord = (lat, long)
        self.coor = coord
        return df

    def upload_data_almaric(self):
        #read csv file from sirea website and return a panda dataframe
        L = self.path_data
        df = pd.read_csv(L[0], sep=';')
        for i in range(1,len(L)):
            df = df.append(pd.read_csv(L[i], sep=';'))

        df.loc[:, 'Valeur'] = df.Valeur.str.replace(',', '.').astype(float)
        df['date'] = pd.to_datetime(df.Date, format='%d/%m/%Y %H:%M:%S')
        df = df.set_index('date')
        df['time'] = (df.index + pd.DateOffset(hours=1)).strftime('%Y-%m-%d %H')


        ## load
        newdf = pd.DataFrame(index=pd.DatetimeIndex(pd.date_range(df.time.min(), df.time.max(), freq='1H')))
        df_load = df[df.Variable == 'Puissance totale consommée'][[
            'time', 'Valeur'
        ]]
        df_load = df_load.groupby(['time']).mean().rename(columns={'Valeur': 'load_kW'})
        df_load.load_kW = abs(df_load.load_kW / 1000)
        newdf = newdf.merge(df_load, how='left', right_index=True, left_index=True)

        ## prod
        df_prod = df[df.Variable == 'Puissance totale des onduleurs revente'][[
            'time', 'Valeur'
        ]]
        df_prod = df_prod.groupby(['time']).mean().rename(columns={'Valeur': 'prod_kW'})
        df_prod = abs(df_prod.prod_kW / 1000)

        newdf = newdf.merge(df_prod, how='left', right_index=True, left_index=True)
        newdf.prod_kW = newdf.loc[newdf.index >= '2019-09-03', 'prod_kW'].fillna(0)
        self.df = newdf

    def upload_data_solaredge(self):
        newdf = pd.DataFrame(index=pd.DatetimeIndex(
            pd.date_range('2019-09-17 00:00:00', datetime.today().strftime('%Y-%m-%d %H:%M'), freq='15min')))

        L = self.path_data
        df = pd.read_csv(L[0])

        for i in range(1, len(L)):
            df = df.append(pd.read_csv(L[i]))
        df.Time = pd.to_datetime(df.Time, format='%d/%m/%Y %H:%M')
        df = df.set_index('Time')
        newdf = newdf.merge(df, how='left', left_index=True, right_index=True)

        newdf['time'] = (newdf.index + pd.DateOffset(hours=1)).strftime('%Y-%m-%d %H')
        newdf = newdf.groupby(['time']).mean()
        newdf.index = pd.DatetimeIndex(newdf.index)
        newdf['load_kW'] = (newdf['Autoconsommation (W)'] + newdf['Importer (W)']) / 1000
        newdf['prod_kW'] = newdf['Production solaire (W)'] / 1000
        self.df = newdf[['load_kW', 'prod_kW']]


    def upload_forecast_almaric(self):
        path = '../projects/2. SIREA/MARCEAU ALMARIC/prevision prod/'
        # read daily solar panel forecast from sirea, fit a gaussian to get hourly data and return a panda dataframe
        df = pd.read_csv(path + 'avril_2019.csv', sep=';')
        month = ['mai_2019', 'juin_2019', 'juillet_2019', 'aout_2019', 'sept_2019', 'oct_2019',
                 'nov_2019', 'dec_2019', 'jan_2020', 'fev_2020', 'mars_2020', 'avril_2020']
        for file in month:
            df = df.append(pd.read_csv(path + file + '.csv', sep=';'))

        df['time'] = pd.to_datetime(df.Date, format="%d/%m/%Y %H:%M:%S")
        df['year'] = df.time.dt.year
        df['day'] = df.time.dt.dayofyear
        df = df.drop(['Format', 'Date', 'time'], axis=1)

        newdf = pd.DataFrame({
            'time':
            pd.date_range(start="2019-04-01 00:00:00",
                          end="2020-05-01 00:00:00",
                          freq='1H')
        })
        newdf['year'] = newdf.time.dt.year
        newdf['day'] = newdf.time.dt.dayofyear
        newdf['hour'] = newdf.time.dt.hour
        newdf = newdf.merge(df, on=['year', 'day'], how='left')
        newdf['forecast_kW'] = newdf.apply(test_func, axis=1)
        newdf = newdf.drop('Valeur', axis=1)
        newdf = newdf.loc[:, ['time', 'forecast_kW']]
        newdf = newdf.set_index(pd.DatetimeIndex(newdf.time))
        self.forecast = newdf.drop('time', axis=1)

    def process_data(self, type):
        # save and merge all data
        if 'Almaric' in type:
            self.upload_data_almaric()
            self.upload_forecast_almaric()
            self.df = self.df.merge(self.forecast, how='left',
                                         left_index=True, right_index=True)
        if 'SYD TGBT' in type:
            self.upload_data_solaredge()

        L = self.path_meteo
        meteo = self.emcwf_to_pandas(L[0])
        for i in range(1, len(L)):
            meteo = meteo.append(self.emcwf_to_pandas(L[i]))
        meteo = meteo.set_index(pd.DatetimeIndex(meteo.time)).drop('time', axis=1)

        '''meteo is not accurate 2 months prior to today so we have to get rid of it'''
        meteo = meteo[meteo.index + pd.DateOffset(months=3) <= datetime.today()]
        self.meteo = meteo

        data = self.df.merge(self.meteo, how='outer',
                                         left_index=True, right_index=True)

        self.merged_data = data
        data = add_temp_features(data)
        data['load_kW-1'] = data.load_kW.shift(1)
        data['load_kW-2'] = data.load_kW.shift(2)
        data['load_kW-3'] = data.load_kW.shift(3)
        data['prod_kW-1'] = data.prod_kW.shift(1)
        data['prod_kW-2'] = data.prod_kW.shift(2)
        data['prod_kW-3'] = data.prod_kW.shift(3)
        data['Station'] = self.station
        data['Station_name'] = self.station
        data['Station_max'] = data.load_kW.max()
        self.full_data = data
        self.full_data.to_pickle(self.folder + self.station + '.pkl')
        print(type, self.coor)


def main(station):
    if station == 'SYD TGBT':
        data = pd.DataFrame()
        for stat in ['SYD TGBT1', 'SYD TGBT2', 'SYD TGBT3']:
            d_read = data_reader(stat)
            d_read.process_data(stat)
            df = d_read.full_data
            data = data.append(df)
        d_read.station = 'SYD TGBT'
        d_read.full_data = data
    else:
        d_read = data_reader(station)
        d_read.process_data(station)
    folder = 'neureco/' + str(d_read.station) + '/'

    if not os.path.exists(folder):
        os.mkdir(folder)
    d_read.full_data.to_pickle(folder + 'full_data.pkl')
    return d_read

# main('SYD TGBT')