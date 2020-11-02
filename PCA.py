import pandas as pd
import numpy as np
import numpy.linalg as la

def grouped_pca_matrix(df, output, feature, group_by):
    data = df[group_by + [feature, output]]
    data = data.assign(groupby=0)
    for i in range(len(group_by)):
        power = pow(2, len(group_by) - (i + 1))
        if power == 1:
            power = 0
        data.loc[:,'groupby'] += data[group_by[i]] * pow(10, power)
    df = data.groupby(by=[feature] + group_by).mean().reset_index().set_index([feature, 'groupby'])
    df = df[[output]].unstack().transpose()
    return df

def PCA(data, output, feature):
    if feature == 'hour':
        group_by = ['day']
    elif feature == 'day':
        group_by = ['hour']
    elif feature == 'week':
        group_by = ['weekday', 'hour']
    elif feature == 'weekday':
        group_by = ['week', 'hour']
    else:
        group_by = ['day', 'hour']

    df = grouped_pca_matrix(data, output, feature, group_by).dropna()
    if len(df) > 0:
        cov = df.cov()
        eig_val, eig_vec = la.eig(cov)
        variance = abs(eig_val)
        components = abs(eig_vec.T)
        results = pd.DataFrame(index=df.columns, data=components)

        Variance = pd.DataFrame(data={'%_Variance': variance / variance.sum() * 100})
        Variance['cumul_sum'] = Variance['%_Variance'].cumsum()
        n_inputs = Variance[Variance.cumul_sum >= 80].index.min()
        pca_values = pd.DataFrame(index = results.index)
        for i in range(n_inputs + 1):
            col_name = feature + '_pca_' + str(i)
            data[col_name] = data[feature].replace(results.index, results.loc[:, i])
            pca_values[col_name] = results.loc[:,i]
        return pca_values
    else:
        print(' >>> alert : not enough data for PCA on '+feature)
        return []

