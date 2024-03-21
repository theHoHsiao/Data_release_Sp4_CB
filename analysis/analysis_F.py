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

    CSV_data_F = []

    MASS = np.zeros(shape=(28, bootstrap.num_sample + 1))

    n = 0

    with open("metadata/F_meson_meta.csv", newline="") as csvfile:
        meta = csv.DictReader(csvfile)
        for row in meta:
            print(
                "loading... beta = ", row.get("beta"), "mf_0 = ", row.get("f_bare mass")
            )
            print("g5 --> ", end="")

            h5_path = (
                "meson_fund/"
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
                + row.get("f_bare mass")
            )

            smear_group = DATA[h5_path]

            epsilon = smear_group[
                "source_N"
                + row.get("PS_N_source")
                + "_sink_N"
                + row.get("PS_N_sink")
                + "_fund/"
            ].attrs["epsilon"]

            corr_PS = smear_group[
                "source_N"
                + row.get("PS_N_source")
                + "_sink_N"
                + row.get("PS_N_sink")
                + "_fund/g5/correlators"
            ][()]

            # if corr_PS.shape[0] != 200:
            #    continue

            m_PS, m_PS_err, m_PS_chi2 = extract.meson_mass(
                corr_PS, int(row.get("PS_ti")), int(row.get("PS_tf"))
            )

            MASS[n] = m_PS

            corr_V1 = smear_group[
                "source_N"
                + row.get("V_N_source")
                + "_sink_N"
                + row.get("V_N_sink")
                + "_fund/g1/correlators"
            ][()]
            corr_V2 = smear_group[
                "source_N"
                + row.get("V_N_source")
                + "_sink_N"
                + row.get("V_N_sink")
                + "_fund/g2/correlators"
            ][()]
            corr_V3 = smear_group[
                "source_N"
                + row.get("V_N_source")
                + "_sink_N"
                + row.get("V_N_sink")
                + "_fund/g3/correlators"
            ][()]

            corr_V = np.mean((corr_V1, corr_V2, corr_V3), axis=0)
            print("gi --> ", end="")
            m_V, m_V_err, m_V_chi2 = extract.meson_mass(
                corr_V, int(row.get("V_ti")), int(row.get("V_tf"))
            )

            m_R = m_PS / m_V
            m_R_err = bootstrap.bootstrap_error(m_R[0:-1], m_R[-1])

            CSV_data_F.append(
                [
                    beta_tag(row.get("beta")),
                    row.get("Nt"),
                    row.get("Ns"),
                    row.get("beta"),
                    row.get("f_bare mass"),
                    epsilon,
                    row.get("PS_N_source"),
                    row.get("PS_N_sink"),
                    row.get("PS_ti"),
                    row.get("PS_tf"),
                    m_PS_chi2,
                    m_PS[-1],
                    m_PS_err,
                    row.get("V_N_source"),
                    row.get("V_N_sink"),
                    row.get("V_ti"),
                    row.get("V_tf"),
                    m_V_chi2,
                    m_V[-1],
                    m_V_err,
                    m_R[-1],
                    m_R_err,
                ]
            )

            n += 1
            print("\n")

    np.save("tmp_data/MASS_PS_F.npy", MASS)

    with open("CSVs/F_meson.csv", "w", newline="") as csvfile:
        fieldnames = [
            "ENS",
            "Nt",
            "Ns",
            "beta",
            "f_bare_mass",
            "f_epsilon",
            "PS_N_source",
            "PS_N_sink",
            "PS_ti",
            "PS_tf",
            "PS_chisquare/dof",
            "m_PS",
            "m_PS_error",
            "V_N_source",
            "V_N_sink",
            "V_ti",
            "V_tf",
            "V_chisquare/dof",
            "m_V",
            "m_V_error",
            "r_PS_V",
            "r_PS_V_error",
        ]

        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(CSV_data_F)


if __name__ == "__main__":
    main()
