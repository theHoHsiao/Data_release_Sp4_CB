import numpy as np
import pandas as pd
import csv
import itertools
from multiprocessing import Pool
import sys


sys.path.insert(1, "./Lib")
import fitting
import bootstrap


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
        0: "Lambda",
        1: "Sigma",
        2: "SigmaS",
    }.get(ch, ch)


def global_fit(CUT):
    PS_cut, ps_cut = CUT

    CSV_data = []

    select_data = lat_cutoff & (w0mPS[:, -1] < PS_cut) & (w0mps[:, -1] < ps_cut)

    Ndata = sum(select_data)

    print(
        "---------- CUT ( m_PS , m_ps ) = (",
        PS_cut,
        ",",
        ps_cut,
        ") Ndata =",
        Ndata,
        "----------\n",
    )

    m_PS = w0mPS[select_data]
    m_ps = w0mps[select_data]
    lat_a = 1.0 / w0s[select_data, :]

    for ch in range(3):
        # print('fitting----------',ch_lb(ch), '\n')
        m_cb = MASS_CB[ch, select_data, :] * w0s[select_data, :]

        ###################### Ansatz ------ M2
        ans = "M2"
        num_par = 4

        fit_val, chi2 = fitting.baryon_M2(m_PS, m_ps, lat_a, m_cb)

        AIC = chi2 + 2 * num_par + 2 * (144 - Ndata)

        csv_tmp = []
        csv_tmp.extend((ch_lb(ch), ans, PS_cut, ps_cut, chi2, AIC))

        for i in range(num_par):
            err_tmp = bootstrap.bootstrap_error(fit_val[i, 0:-1], fit_val[i, -1])

            csv_tmp.extend((fit_val[i, -1], err_tmp))

        CSV_data.append(csv_tmp)

        ###################### Ansatz ------ M3
        ans = "M3"
        num_par = 8

        fit_val, chi2 = fitting.baryon_M3(m_PS, m_ps, lat_a, m_cb)

        AIC = chi2 + 2 * num_par + 2 * (144 - Ndata)

        csv_tmp = []
        csv_tmp.extend((ch_lb(ch), ans, PS_cut, ps_cut, chi2, AIC))

        for i in range(num_par):
            err_tmp = bootstrap.bootstrap_error(fit_val[i, 0:-1], fit_val[i, -1])

            csv_tmp.extend((fit_val[i, -1], err_tmp))

        CSV_data.append(csv_tmp)

        ###################### Ansatz ------ MF4
        ans = "MF4"
        num_par = 9

        fit_val, chi2 = fitting.baryon_MF4(m_PS, m_ps, lat_a, m_cb)

        AIC = chi2 + 2 * num_par + 2 * (144 - Ndata)

        csv_tmp = []
        csv_tmp.extend((ch_lb(ch), ans, PS_cut, ps_cut, chi2, AIC))

        for i in range(num_par):
            err_tmp = bootstrap.bootstrap_error(fit_val[i, 0:-1], fit_val[i, -1])

            csv_tmp.extend((fit_val[i, -1], err_tmp))

        CSV_data.append(csv_tmp)

        ###################### Ansatz ------ MA4
        ans = "MA4"
        num_par = 9

        fit_val, chi2 = fitting.baryon_MA4(m_PS, m_ps, lat_a, m_cb)

        AIC = chi2 + 2 * num_par + 2 * (144 - Ndata)

        csv_tmp = []
        csv_tmp.extend((ch_lb(ch), ans, PS_cut, ps_cut, chi2, AIC))

        for i in range(num_par):
            err_tmp = bootstrap.bootstrap_error(fit_val[i, 0:-1], fit_val[i, -1])

            if i == 8:
                csv_tmp.extend((0, 0, fit_val[i, -1], err_tmp))
            else:
                csv_tmp.extend((fit_val[i, -1], err_tmp))

        CSV_data.append(csv_tmp)

        ###################### Ansatz ------ MC4
        ans = "MC4"
        num_par = 9

        fit_val, chi2 = fitting.baryon_MC4(m_PS, m_ps, lat_a, m_cb)

        AIC = chi2 + 2 * num_par + 2 * (144 - Ndata)

        csv_tmp = []
        csv_tmp.extend((ch_lb(ch), ans, PS_cut, ps_cut, chi2, AIC))

        for i in range(num_par):
            err_tmp = bootstrap.bootstrap_error(fit_val[i, 0:-1], fit_val[i, -1])

            if i == 8:
                csv_tmp.extend((0, 0, 0, 0, fit_val[i, -1], err_tmp))
            else:
                csv_tmp.extend((fit_val[i, -1], err_tmp))

        CSV_data.append(csv_tmp)

    return CSV_data


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


def main():
    print(">>>>>>>>> This analysis takes days to finish <<<<<<<<<<<<<<<< :-.< ")
    ##############################################

    PS_cut = np.round(np.arange(0.52, 1.08, 0.05), 2)
    ps_cut = np.round(np.arange(0.52, 1.88, 0.05), 2)

    CUTs = itertools.product(PS_cut, ps_cut)
    pool = Pool(initializer=Gvar)
    pool_outputs = pool.map(global_fit, list(CUTs))

    ##################################

    with open("CSVs/FIT_mass.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Baryon",
            "Ansatz",
            "mPS_cut",
            "mps_cut",
            "chisqr",
            "AIC",
            "M_CB",
            "M_CB_err",
            "F2",
            "F2_err",
            "A2",
            "A2_err",
            "L1",
            "L1_err",
            "F3",
            "F3_err",
            "A3",
            "A3_err",
            "L2F",
            "L2F_err",
            "L2A",
            "L2A_err",
            "F4",
            "F4_err",
            "A4",
            "A4_err",
            "C4",
            "C4_err",
        ]

        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for i in range(len(pool_outputs)):
            writer.writerows(pool_outputs[i])

if __name__ == "__main__":
    main()
