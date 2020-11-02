import pandas as pd
import numpy as np
import holidays


class stockage():
    def __init__(self, capacite, station):
        if 'SYD TGBT' in station:
            if station == 'SYD TGBT1':
                p_armoire = 17
            if station == 'SYD TGBT2':
                p_armoire = 15
            if station == 'SYD TGBT3':
                p_armoire = 12.5
            prix_achat = 0.15
            prix_revente = 0.06
        elif 'Almaric' in station:
            p_armoire = 30
            prix_achat = 0.1
            prix_revente = 0

        self.capacite = capacite
        self.seuil_max = 0.98 * capacite
        self.seuil_min = 0.2 * capacite
        self.p_charge = p_armoire
        self.p_decharge = 0.1 * capacite

        self.prix_batterie = 120 * capacite
        self.prix_achat = prix_achat
        self.prix_revente = prix_revente

        self.etat_ini_batterie = 10

    def add_storage(self, d_read):
        df = d_read.full_data[['load_kW', 'prod_kW']].dropna()
        df = df.assign(import_kW = (df.load_kW - df.prod_kW).apply(lambda x: max(x, 0)))
        df = df.assign(export_kW= (df.prod_kW - df.load_kW).apply(lambda x: max(x, 0)))

        batterie = self.etat_ini_batterie
        col_stock = []
        col_import = []
        col_export = []
        col_batterie = []

        for i in range(len(df)):
            prod = df.prod_kW.values[i]
            conso = df.load_kW.values[i]
            delta = prod - conso
            stock = 0
            vente = 0
            achat = 0

            if delta > 0:
                # on stocke et on vend le surplus
                stock = min(delta, self.p_charge, self.seuil_max - batterie)
                vente = delta - stock

            if delta <= 0:
                # on destocke et on achète ce qu'il manque
                stock = - min(abs(delta), self.p_decharge, max(batterie - self.seuil_min, 0))
                achat = abs(delta) + stock

            batterie += stock
            col_stock.append(stock)
            col_import.append(achat)
            col_export.append(vente)
            col_batterie.append(batterie)

        df = df.assign(storage_kW = col_stock)
        df = df.assign(battery_kWh = col_batterie)
        df = df.assign(export_storage_kW = col_export)
        df = df.assign(import_storage_kW = col_import)
        self.df = df

    def interet_stockage(self, d_read):
        self.add_storage(d_read)
        df = self.df
        duration = (df.index.to_series().max() - df.index.to_series().min()).days / 365

        newdf = pd.DataFrame()
        newdf['capacite'] = [self.capacite]
        newdf['%_autoconsomation'] = [(df.load_kW.sum() - df.import_storage_kW.sum()) / df.load_kW.sum() * 100]
        newdf['%_pertes_prod'] = [df.export_storage_kW.sum() / df.prod_kW.sum() * 100]
        newdf['gain_E_par_an'] = [((df.import_kW.sum() - df.import_storage_kW.sum()) * self.prix_achat - (
                    df.export_kW.sum() - df.export_storage_kW.sum()) * self.prix_revente) / duration]
        newdf['annees_amortissement'] = self.prix_batterie / (newdf.gain_E_par_an)
        return newdf

class opt_load():
    def __init__(self, df):
        df.insert(1, 'day', df.index.dayofyear)
        self.df = df
    
    def shift_data(self, day):
        df = self.df
        df_day = df[df.day == day]
        self.max_prod = df_day.loss_storage.max()
        self.max_load = df_day.EDF_storage.max()
        self.i_prod = df_day.loss_storage.idxmax()
        self.i_load = df_day.EDF_storage.idxmax()
        loss = [df_day.loss_storage]
        edf=[df_day.EDF_storage]
        if df_day.weekday.max() == 0:
            while (self.max_prod > 0) & (self.max_load > 0):
                to_shift = min(self.max_load, self.max_prod)
                df_day.loc[self.i_prod, 'loss_storage'] -= to_shift
                df_day.loc[self.i_load, 'EDF_storage'] -= to_shift

                self.max_prod = df_day.loss_storage.max()
                self.max_load = df_day.EDF_storage.max()
                self.i_prod = df_day.loss_storage.idxmax()
                self.i_load = df_day.EDF_storage.idxmax()
                
                loss.append(df_day.loss_storage)
                edf.append(df_day.EDF_storage)
            self.df.loc[df_day.index, :] = df_day 
        return loss, edf




