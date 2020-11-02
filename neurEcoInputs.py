
from pythonGN import *
import pandas as pd
import os



def residu(dynamic_features, vec_neureco, x_norm, y_norm, ne_model, vec):
    '''yref to fortran'''
    yref = np.asfortranarray(y_norm.reshape(len(y_norm), 1).T)

    out = np.ones_like(yref)
    out = np.asfortranarray(out)

    '''update x0 and apply neureco'''
    x0 = x_norm.copy()
    i = 0
    for col in dynamic_features:
        old_value = x_norm[col].unique()
        new_value = vec[i: i + len(old_value)]
        x0[col] = x0[col].replace(old_value, new_value)
        i += len(old_value)

    x0 = np.asfortranarray(x0.T)
    ne_model.apply(out, vec_neureco, x0)

    F = out - yref
    F = np.asfortranarray(F)
    F = np.reshape(F, (-1, 1), order="F")
    e = 0.5 * np.linalg.norm(F) ** 2
    return F, out, e


def direct(dynamic_features, vec_neureco, x_norm, y_norm, ne_model, vec, dvec, xu):
    yref = np.asfortranarray(y_norm.reshape(len(y_norm), 1).T)
    dvec = np.array(dvec)
    dout = np.ones(shape=(yref.shape[0] * yref.shape[1], 1), dtype=float, order='F')
    dout = np.asfortranarray(dout)

    '''update x0 and dx0'''
    x0 = x_norm.copy()
    dx0 = x_norm.copy()
    dx0.loc[:] = 0

    i = 0
    df = pd.DataFrame()
    for col in dynamic_features:
        old_value = x_norm[col].unique()
        df_vec = pd.DataFrame(index=old_value)
        x0[col] = x0[col].replace(old_value, vec[i: i + len(old_value)])
        dx0.loc[:, col] = x_norm[col]
        dx0[col] = dx0[col].replace(dx0[col].unique(), dvec[i: i + len(old_value)])
        i += len(old_value)
        df = df.append(df_vec)

    df['vec'] = vec
    df['dvec'] = dvec

    x0 = np.asfortranarray(x0.T)
    dx0 = np.asfortranarray(dx0.T)
    dvec_neureco = np.zeros_like(vec_neureco)
    ne_model.direct(dout, vec_neureco, x0, dvec_neureco, dx0)
    return dout


def inverse(dynamic_features, vec_neureco, x_norm, y_norm, ne_model, vec, pF, xu):
    x0 = x_norm.copy()
    x0 = np.asfortranarray(x0.T)

    px0 = np.asfortranarray(np.ones(shape=(x0.shape[1], x0.shape[0]), dtype=float, order='F'))
    pvec_neureco = np.asfortranarray(
        np.ones(shape=(vec_neureco.shape[1] * vec_neureco.shape[0], 0), dtype=float, order='F'))

    ne_model.inverse(pvec_neureco, px0, vec_neureco, x0, pF)

    '''update pvec with px0'''
    px0 = np.reshape(px0, x0.shape, order="F").T
    px0 = pd.DataFrame(columns=x_norm.columns, data=px0)

    df = pd.DataFrame()
    for col in dynamic_features:
        old_value = x_norm[col].unique()
        df_vec = pd.DataFrame(index=old_value)
        new_value = pd.DataFrame()
        new_value['original_input'] = x_norm[col].values
        new_value['pvec'] = px0[col].values
        new_value = new_value.groupby(by='original_input').sum()
        df_vec = df_vec.merge(new_value, how='left', left_index=True, right_index=True)
        df = df.append(df_vec)

    df['vec'] = vec

    pvec = np.asfortranarray(np.reshape(df.pvec.values, (-1, 1), "F"))
    return pvec
