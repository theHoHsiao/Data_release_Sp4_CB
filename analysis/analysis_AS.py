import numpy as np
import h5py
import csv
import sys

sys.path.insert(1, "./Lib")

import extract
import bootstrap


def beta_tag(b):
    if b == "7.62":
        return "QB1"
    elif b == "7.7":
        return "QB2"
    elif b == "7.85":
        return "QB3"
    elif b == "8.0":
        return "QB4"
    elif b == "8.2":
        return "QB5"
    else:
        return "QBX"


def main():
    DATA = h5py.File("tmp_data/data.h5")

    CSV_data_AS = []

    MASS = np.zeros(shape=(42, bootstrap.num_sample + 1))

    n = 0

    with open("metadata/AS_meson_meta.csv", newline="") as csvfile:
        meta = csv.DictReader(csvfile)
        for row in meta:
            print(
                "loading... beta = ",
                row.get("beta"),
                "mas_0 = ",
                row.get("as_bare mass"),
            )

            h5_path = (
                "meson_anti/"
                + row.get("Nt")
                + "x"
                + row.get("Ns")
                + "x"
                + row.get("Ns")
                + "x"
                + row.get("Ns")
                + "b"
                + row.get("beta")
                + "/"
                + row.get("as_bare mass")
            )

            smear_group = DATA[h5_path]

            epsilon = smear_group[
                "source_N"
                + row.get("ps_N_source")
                + "_sink_N"
                + row.get("ps_N_sink")
                + "_anti/"
            ].attrs["epsilon"]

            corr_PS = smear_group[
                "source_N"
                + row.get("ps_N_source")
                + "_sink_N"
                + row.get("ps_N_sink")
                + "_anti/g5/correlators"
            ][()]

            # if corr_PS.shape[0] != 200:
            #    continue
            print("g5 --> ", end="")
            m_PS, m_PS_err, m_PS_chi2 = extract.meson_mass(
                corr_PS, int(row.get("ps_ti")), int(row.get("ps_tf"))
            )

            MASS[n] = m_PS

            corr_V1 = smear_group[
                "source_N"
                + row.get("v_N_source")
                + "_sink_N"
                + row.get("v_N_sink")
                + "_anti/g1/correlators"
            ][()]
            corr_V2 = smear_group[
                "source_N"
                + row.get("v_N_source")
                + "_sink_N"
                + row.get("v_N_sink")
                + "_anti/g2/correlators"
            ][()]
            corr_V3 = smear_group[
                "source_N"
                + row.get("v_N_source")
                + "_sink_N"
                + row.get("v_N_sink")
                + "_anti/g3/correlators"
            ][()]

            corr_V = np.mean((corr_V1, corr_V2, corr_V3), axis=0)
            print("gi --> ", end="")
            m_V, m_V_err, m_V_chi2 = extract.meson_mass(
                corr_V, int(row.get("v_ti")), int(row.get("v_tf"))
            )

            m_R = m_PS / m_V
            m_R_err = bootstrap.bootstrap_error(m_R[0:-1], m_R[-1])

            CSV_data_AS.append(
                [
                    beta_tag(row.get("beta")),
                    row.get("Nt"),
                    row.get("Ns"),
                    row.get("beta"),
                    row.get("as_bare mass"),
                    epsilon,
                    row.get("ps_N_source"),
                    row.get("ps_N_sink"),
                    row.get("ps_ti"),
                    row.get("ps_tf"),
                    m_PS_chi2,
                    m_PS[-1],
                    m_PS_err,
                    row.get("v_N_source"),
                    row.get("v_N_sink"),
                    row.get("v_ti"),
                    row.get("v_tf"),
                    m_V_chi2,
                    m_V[-1],
                    m_V_err,
                    m_R[-1],
                    m_R_err,
                ]
            )

            n += 1
            print("\n")

    np.save("tmp_data/MASS_ps_AS.npy", MASS)

    with open("CSVs/AS_meson.csv", "w", newline="") as csvfile:
        fieldnames = [
            "ENS",
            "Nt",
            "Ns",
            "beta",
            "as_bare_mass",
            "as_epsilon",
            "ps_N_source",
            "ps_N_sink",
            "ps_ti",
            "ps_tf",
            "ps_chisquare/dof",
            "m_ps",
            "m_ps_error",
            "v_N_source",
            "v_N_sink",
            "v_ti",
            "v_tf",
            "v_chisquare/dof",
            "m_v",
            "m_v_error",
            "r_ps_v",
            "r_ps_v_error",
        ]

        writer = csv.writer(csvfile)

        writer.writerow(fieldnames)
        writer.writerows(CSV_data_AS)

if __name__ == "__main__":
    main()
