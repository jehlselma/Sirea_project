import pandas as pd
import holidays
import numpy as np

def dates_feries(annee):
    # renvoie les dates des jours fériés d'une année
    feries = holidays.FRA(years=annee)
    dates = []
    for i in feries:
        dates.append(str(i))
    return dates


def dates_ponts(annee):
    # renvoie les dates des ponts d'une année : il y a pont le lundi si un jour férié tombe un mardi et le vendredi si ça tombe un jeudi
    dates = []
    for d in holidays.FRA(years=annee):
        if (d.month <= 9) & (d.month >= 4):
            if d.isocalendar()[2] == 2:
                day = d - pd.DateOffset(days=1)
                day = day.strftime('%Y-%m-%d')
                dates.append(day)
            if d.isocalendar()[2] == 4:
                day = d + pd.DateOffset(days=1)
                day = day.strftime('%Y-%m-%d')
                dates.append(day)
    return dates


def add_ferie(df):
    # ajoute les colonnes ponts et ferie qui renvoie 1 si c'est un jour ferie/pont et 0 sinon
    feries = []
    ponts = []

    year = df.index.year.unique()
    for i in year:
        annee = int(i)
        feries += dates_feries(annee)
        ponts += dates_ponts(annee)

    df = df.assign(ferie=df.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d")).isin(feries))
    df = df.assign(ponts=df.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d")).isin(ponts))

    return df


def vacances():
    # renvoie les dates des vacances de la zone C
    ZoneC = {
        'Noël': [['2014-12-20', '2015-01-05'], ['2015-12-20', '2016-01-03'],
                 ['2016-12-17', '2017-01-03'], ['2017-12-23', '2018-01-08'],
                 ['2018-12-22', '2019-01-07'], ['2019-12-21', '2020-01-06']],
        'Printemps':
            [['2015-04-18', '2015-05-04'], ['2016-04-23', '2016-05-09'],
             ['2017-04-08', '2017-04-24'], ['2018-04-14', '2018-04-30'],
             ['2019-04-20', '2019-05-06'], ['2020-04-04', '2020-04-20']],
        'Ete': [['2015-08-01', '2015-08-31'], ['2016-08-01', '2016-08-31'],
                ['2017-08-01', '2017-09-04'], ['2018-07-20', '2018-09-03'],
                ['2019-07-20', '2019-09-02']]
    }

    return ZoneC


def add_holiday(df):
    # ajoute une colonne vacances a df qui renvoie 1 si c'est les vacances 0 sinon
    vac = vacances()
    df['vacances'] = 0
    c = 0
    for i in vac:
        liste = vac[i]
        c += 1
        for dates in liste:
            d1 = dates[0]
            d2 = dates[1]
            df.loc[(df.index >= d1) & (df.index <= d2), 'vacances'] = 1
    return df


def add_temp_features(df):
    # return df with relevant temporal variables
    df = df.assign(hour=df.index.hour)
    df = df.assign(week=df.index.week)
    df = df.assign(day=df.index.dayofyear)
    df = df.assign(month=df.index.month)
    df = df.assign(weekday=df.index.weekday)
    df = df.assign(weekday_name=df.index.day_name())
    df = df.assign(cos_hour=df.hour.apply(lambda x: np.cos(2 * np.pi * (x) / 24)))
    df = df.assign(sin_hour=df.hour.apply(lambda x: np.sin(2 * np.pi * (x) / 24)))
    df = df.assign(cos_week=df.week.apply(lambda x: np.cos(2 * np.pi * (x - 1) / (52))))
    df = df.assign(sin_week=df.week.apply(lambda x: np.sin(2 * np.pi * (x - 1) / 52)))
    df.insert(1, 'year', df.index.year)
    df = add_ferie(df)
    df['working_day'] = 1
    df.loc[(df.weekday == 5) | (df.weekday == 6), 'working_day'] = 0
    return df
