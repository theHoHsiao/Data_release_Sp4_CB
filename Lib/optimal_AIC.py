import numpy as np
import pandas as pd
import csv
import itertools
from multiprocessing import Pool
import sys

sys.path.insert(1, "./Lib")
import bootstrap
import fitting


def beta_to_w0(b):
    if b == 7.62:
        return bootstrap.b2w_QB1
    elif b == 7.7:
        return bootstrap.b2w_QB2
    elif b == 7.85:
        return bootstrap.b2w_QB3
    elif b == 8.0:
        return bootstrap.b2w_QB4
    elif b == 8.2:
        return bootstrap.b2w_QB5
    else:
        return bootstrap.gauB_DIS(0, 0.0001, bootstrap.num_sample)


def ch_lb(ch):
    return {
        "Lambda": 0,
        "Sigma": 1,
        "SigmaS": 2,
    }.get(ch, ch)


def FIT_FUNC(i):
    return {
        0: fitting.baryon_M2,
        1: fitting.baryon_M3,
        2: fitting.baryon_MF4,
        3: fitting.baryon_MA4,
        4: fitting.baryon_MC4,
    }.get(i, i)


def get_pars(ch, ans, PS_cut, ps_cut):
    PS_cut, ps_cut
    func = FIT_FUNC(ans)

    select_data = lat_cutoff & (w0mPS[:, -1] < PS_cut) & (w0mps[:, -1] < ps_cut)

    m_PS = w0mPS[select_data]
    m_ps = w0mps[select_data]
    lat_a = 1.0 / w0s[select_data, :]

    m_cb = MASS_CB[ch_lb(ch), select_data, :] * w0s[select_data, :]
    fit_val, chi2 = func(m_PS, m_ps, lat_a, m_cb)

    return fit_val


def Gvar():
    global lat_cutoff, w0mPS, w0mps, w0s, MASS_CB

    CB_mass = pd.read_csv("CSVs/CB_mass.csv")
    F_meson = pd.read_csv("CSVs/F_meson.csv")
    AS_meson = pd.read_csv("CSVs/AS_meson.csv")

    MASS_CB = np.load("tmp_data/MASS_chimera.npy")
    MASS_PS = np.load("tmp_data/MASS_PS_F.npy")
    MASS_ps = np.load("tmp_data/MASS_ps_AS.npy")

    betas = CB_mass.beta.values
    f_bare_mass = CB_mass.f_bare_mass.values
    as_bare_mass = CB_mass.as_bare_mass.values

    Ndata = len(betas)
    mPS = np.zeros(shape=(Ndata, bootstrap.num_sample + 1))
    mps = np.zeros(shape=(Ndata, bootstrap.num_sample + 1))
    w0s = np.zeros(shape=(Ndata, bootstrap.num_sample + 1))

    for i in range(Ndata):
        mPS[i] = MASS_PS[
            np.round(MASS_PS[:, -1], 12)
            == np.round(
                F_meson[
                    (F_meson.beta == betas[i]) & (F_meson.f_bare_mass == f_bare_mass[i])
                ].m_PS.values[0],
                12,
            )
        ]
        mps[i] = MASS_ps[
            np.round(MASS_ps[:, -1], 12)
            == np.round(
                AS_meson[
                    (AS_meson.beta == betas[i])
                    & (AS_meson.as_bare_mass == as_bare_mass[i])
                ].m_ps.values[0],
                12,
            )
        ]
        w0s[i] = beta_to_w0(betas[i])

    w0mPS = w0s * mPS
    w0mps = w0s * mps

    lat_cutoff = (mPS[:, -1] < 1) & (mps[:, -1] < 1)


Gvar()
