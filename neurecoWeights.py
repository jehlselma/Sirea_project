from NeurEcoWrapperDerivatives import NeurEcoDerivatives
from pythonGN import *
from functools import partial


def residu(x0, yref, ne_model, vec):
    out = np.ones_like(yref)
    out = np.asfortranarray(out)
    x0 = np.asfortranarray(x0)
    ne_model.apply(out, vec, x0)
    F = out - yref
    F = np.asfortranarray(F)
    F = np.reshape(F, (-1, 1), order="F")
    e = 0.5 * np.linalg.norm(F) ** 2
    return F, out, e


def direct(x0, yref, ne_model, vec, dvec, xu):
    dout = np.ones(shape=(yref.shape[0] * yref.shape[1], 1), dtype=float, order='F')
    dout = np.asfortranarray(dout)
    ne_model.direct(dout, vec, x0, dvec, None)
    return dout


def inverse(x0, yref, ne_model, vec, pF, xu):
    pvec = np.ones_like(vec)
    px = np.ones_like(x0)
    pvec = np.asfortranarray(pvec)
    px = np.asfortranarray(px)
    ne_model.inverse(pvec, px, vec, x0, pF)
    return pvec
