import numpy as np
import pandas as pd
import csv
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
        0: "Lambda",
        1: "Sigma",
        2: "SigmaS",
    }.get(ch, ch)


def cross_fit_fixF():
    CSV_data = []

    for a in LAT_uni:
        mPS_tmp = w0mPS[w0s[:, -1] == a]

        m_uni = np.unique(mPS_tmp[:, -1])
        for m in m_uni:
            select_data = lat_cutoff & (w0s[:, -1] == a) & (w0mPS[:, -1] == m)
            print(betas[select_data][0], a, "m_PS_fix =", m / a)

            if sum(select_data) > 4:
                m_ps = w0mps[select_data]

                for ch in range(3):
                    csv_tmp = []
                    csv_tmp.extend(
                        (
                            ch_lb(ch),
                            betas[select_data][0],
                            f_bare_mass[select_data][0],
                            m,
                        )
                    )

                    m_cb = MASS_CB[ch, select_data, :] * w0s[select_data, :]
                    fit_val, chi2 = fitting.cross_check_fit(m_ps, m_cb)

                    for i in range(3):
                        err_tmp = bootstrap.bootstrap_error(
                            fit_val[i, 0:-1], fit_val[i, -1]
                        )
                        csv_tmp.extend((fit_val[i, -1], err_tmp))

                    CSV_data.append(csv_tmp)

    return CSV_data


def cross_fit_fixAS():
    CSV_data = []

    for a in LAT_uni:
        mps_tmp = w0mps[w0s[:, -1] == a]

        m_uni = np.unique(mps_tmp[:, -1])
        for m in m_uni:
            select_data = lat_cutoff & (w0s[:, -1] == a) & (w0mps[:, -1] == m)
            print(a, "m_ps_fix =", m / a)

            if sum(select_data) > 4:
                m_PS = w0mPS[select_data]

                for ch in range(3):
                    csv_tmp = []
                    csv_tmp.extend(
                        (
                            ch_lb(ch),
                            betas[select_data][0],
                            as_bare_mass[select_data][0],
                            m,
                        )
                    )

                    m_cb = MASS_CB[ch, select_data, :] * w0s[select_data, :]

                    fit_val, chi2 = fitting.cross_check_fit(m_PS, m_cb)
                    for i in range(3):
                        err_tmp = bootstrap.bootstrap_error(
                            fit_val[i, 0:-1], fit_val[i, -1]
                        )
                        csv_tmp.extend((fit_val[i, -1], err_tmp))

                    CSV_data.append(csv_tmp)

    return CSV_data


def Gvar():
    global \
        lat_cutoff, \
        w0mPS, \
        w0mps, \
        w0s, \
        MASS_CB, \
        LAT_uni, \
        betas, \
        f_bare_mass, \
        as_bare_mass

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
    LAT_uni = np.unique(w0s[:, -1])


def main():
    Gvar()
    ##############################################

    CSV_AS = cross_fit_fixAS()

    with open("CSVs/FIT_cross_fixAS.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Baryon",
            "beta",
            "mps_fix",
            "w0mps",
            "M_CB",
            "M_CB_err",
            "A2",
            "A2_err",
            "A3",
            "A3_err",
        ]

        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        writer.writerows(CSV_AS)

    ##############################################

    CSV_F = cross_fit_fixF()

    with open("CSVs/FIT_cross_fixF.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Baryon",
            "beta",
            "mPS_fix",
            "w0mPS",
            "M_CB",
            "M_CB_err",
            "F2",
            "F2_err",
            "F3",
            "F3_err",
        ]

        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        writer.writerows(CSV_F)

if __name__ == "__main__":
    main()
