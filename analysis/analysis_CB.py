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


def LB_chimera(b):
    if b == "Lambda":
        return "Chimera_OC"
    elif b == "Sigma":
        return "Chimera_OV12"
    elif b == "SigmaS":
        return "Chimera_OV32"
    else:
        return "Chimera"


def main():
    DATA = h5py.File("tmp_data/data.h5")

    CSV_data = []
    CHs = ["Lambda", "Sigma", "SigmaS"]

    MASS = np.zeros(shape=(3, 144, bootstrap.num_sample + 1))

    n = 0

    with open("metadata/CB_mass_meta.csv", newline="") as csvfile:
        meta = csv.DictReader(csvfile)
        for row in meta:
            print(
                "mf_0 =", row.get("f_bare mass"), ", mas_0 =", row.get("as_bare mass")
            )

            h5_path = (
                "chimera/"
                + row.get("Nt")
                + "x"
                + row.get("Ns")
                + "x"
                + row.get("Ns")
                + "x"
                + row.get("Ns")
                + "b"
                + row.get("beta")
                + "/mf"
                + row.get("f_bare mass")
                + "/mas"
                + row.get("as_bare mass")
            )

            smear_group = DATA[h5_path]

            epsilon_f = smear_group[
                "source_N"
                + row.get("Lambda_N_source")
                + "_sink_N"
                + row.get("Lambda_N_sink")
            ].attrs["epsilon_f"]
            epsilon_as = smear_group[
                "source_N"
                + row.get("Lambda_N_source")
                + "_sink_N"
                + row.get("Lambda_N_sink")
            ].attrs["epsilon_as"]

            data_tmp = []
            data_tmp.extend(
                (
                    beta_tag(row.get("beta")),
                    row.get("Nt"),
                    row.get("Ns"),
                    row.get("beta"),
                    row.get("f_bare mass"),
                    row.get("as_bare mass"),
                    epsilon_f,
                    epsilon_as,
                )
            )

            m = np.zeros(shape=(3, bootstrap.num_sample + 1))
            for i in range(3):
                ch = CHs[i]

                print(ch, row.get(ch + "_N_source"), row.get(ch + "_N_sink"))

                corr_e = smear_group[
                    "source_N"
                    + row.get(ch + "_N_source")
                    + "_sink_N"
                    + row.get(ch + "_N_sink")
                    + "/"
                    + LB_chimera(ch)
                    + "_even_re/correlators"
                ][()]
                corr_o = smear_group[
                    "source_N"
                    + row.get(ch + "_N_source")
                    + "_sink_N"
                    + row.get(ch + "_N_sink")
                    + "/"
                    + LB_chimera(ch)
                    + "_odd_re/correlators"
                ][()]

                CORR_e, CORR_0 = extract.bin_proj_Baryon(corr_e, corr_o)

                m[i], m_err, chi2 = extract.baryon_mass(
                    CORR_e, int(row.get(ch + "_ti")), int(row.get(ch + "_tf"))
                )

                data_tmp.extend(
                    (
                        row.get(ch + "_N_source"),
                        row.get(ch + "_N_sink"),
                        row.get(ch + "_ti"),
                        row.get(ch + "_tf"),
                        chi2,
                        m[i, -1],
                        m_err,
                    )
                )

                MASS[i, n] = m[i]

            R_LvS = m[0] / m[1]
            R_LvS_err = bootstrap.bootstrap_error(R_LvS[0:-1], R_LvS[-1])

            R_SvSs = m[1] / m[2]
            R_SvSs_err = bootstrap.bootstrap_error(R_SvSs[0:-1], R_SvSs[-1])

            data_tmp.extend((R_LvS[-1], R_LvS_err, R_SvSs[-1], R_SvSs_err))

            CSV_data.append(data_tmp)

            n += 1

            print("\n")

    np.save("tmp_data/MASS_chimera.npy", MASS)

    with open("CSVs/CB_mass.csv", "w", newline="") as csvfile:
        fieldnames = [
            "ENS",
            "Nt",
            "Ns",
            "beta",
            "f_bare_mass",
            "as_bare_mass",
            "f_epsilon",
            "as_epsilon",
            "Lambda_N_source",
            "Lambda_N_sink",
            "Lambda_ti",
            "Lambda_tf",
            "Lambda_chisquare/dof",
            "m_Lambda",
            "m_Lambda_error",
            "Sigma_N_source",
            "Sigma_N_sink",
            "Sigma_ti",
            "Sigma_tf",
            "Sigma_chisquare/dof",
            "m_Sigma",
            "m_Sigma_error",
            "SigmaS_N_source",
            "SigmaS_N_sink",
            "SigmaS_ti",
            "SigmaS_tf",
            "SigmaS_chisquare/dof",
            "m_SigmaS",
            "m_SigmaS_error",
            "R_Lambda_Sigma",
            "R_Lambda_Sigma_error",
            "R_Sigma_SigmaS",
            "R_Sigma_SigmaS_error",
        ]

        writer = csv.writer(csvfile)

        writer.writerow(fieldnames)
        writer.writerows(CSV_data)


if __name__ == "__main__":
    main()
