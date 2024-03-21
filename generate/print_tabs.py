import numpy as np
import pandas as pd
import glob
import sys
import os

sys.path.insert(1, "./Lib")
from plot_package import *
import bootstrap


def find_non_zero(x):
    if x < 1:
        A = "%e" % x
        return int(A.partition("-")[2]) - 1
    else:
        # print('error > 1')
        return 0


def print_non_zero(v, err):
    dig = find_non_zero(err) + 2
    return str(v)[0 : dig + 2] + "(" + str(err)[dig : dig + 2] + ")"


def LB_channel(ch):
    return {
        "Lambda": r"$\Lambda_{\rm CB}$",
        "Sigma": r"$\Sigma_{\rm CB}$",
        "SigmaS": r"$\Sigma^\ast_{\rm CB}$",
    }.get(ch, ch)


def print_abv(v):
    if v > 0.005:
        return str(round(v, 2))

    else:
        tmp = "%e" % v
        dig = tmp.split("e")[1]
        return r"$\sim 10^{" + dig + "}$"


def get_AIC(ch):
    string = []

    AICs = np.zeros(shape=(5, 28, 12))
    m02D = np.zeros(shape=(5, 28, 12))
    x22D = np.zeros(shape=(5, 28, 12))

    ansatz = ["M2", "M3", "MF4", "MA4", "MC4"]
    par_ans = [4, 8, 9, 9, 9]
    for ans in range(5):
        for i in range(12):  # len of m_PS_cut
            m02D[ans, :, i] = FIT_mass[
                (FIT_mass.Baryon == ch)
                & (FIT_mass.mPS_cut == m_PS_cut[i])
                & (FIT_mass.Ansatz == ansatz[ans])
            ].M_CB.values
            x22D[ans, :, i] = FIT_mass[
                (FIT_mass.Baryon == ch)
                & (FIT_mass.mPS_cut == m_PS_cut[i])
                & (FIT_mass.Ansatz == ansatz[ans])
            ].chisqr.values

        AICs[ans] = x22D[ans] + 2 * par_ans[ans] + 2 * (Ndata - n_2D)

    w = 0.5 * np.exp(-AICs) / np.sum(np.unique(0.5 * np.exp(-AICs)))
    ind = np.unravel_index(w.argmax(), w.shape)

    for i in range(5):
        ind = np.unravel_index(w[i].argmax(), w[i].shape)

        string.append(
            ansatz[i]
            + " & "
            + str(m_PS_cut[ind[1]])
            + " & "
            + str(m_ps_cut[ind[0]])
            + " & "
            + str(round(x22D[i, ind[0], ind[1]], 2))
            + " & "
            + str(round(AICs[i, ind[0], ind[1]], 2))
            + " & "
            + print_abv(w[i, ind[0], ind[1]])
        )

    return string, w


################################################################################################


def tab_1():
    ENS = [
        "48x24x24x24b7.62",
        "60x48x48x48b7.7",
        "60x48x48x48b7.85",
        "60x48x48x48b8.0",
        "60x48x48x48b8.2",
    ]
    plaq_str = []
    for ens in ENS:
        path = "raw_data/" + ens + "/plaq_b*"

        filename = glob.glob(path)

        # print(filenames)

        f = open(filename[0], "r")

        a_tmp = []
        for line in f:
            a_tmp.append(float(line.split("=")[1][:-1]))

            # a_tmp.append( float(line.split(' ')[1]))

        a_tmp = np.array(a_tmp)
        plaq = np.zeros(shape=(200, 1))

        # plaq = np.zeros(shape=(750,1))

        plaq[:, 0] = a_tmp

        plaq_boot = bootstrap.bootstrap_main(plaq)

        err = bootstrap.bootstrap_error(plaq_boot[:, 0], np.mean(a_tmp))

        # print( np.mean(plaq_boot) - np.mean(a_tmp), err  )
        str_out = print_non_zero(np.mean(plaq_boot), err)
        plaq_str.append(str_out)

    line = [
        "Ensemble & $\\beta$   & $N_t\\times N^3_s$ & $\left < P \\right >$  & $w_0/a$  \\\\ \\hline \n",
        "QB1	    & $7.62$    & $48\\times24^3$   & "
        + plaq_str[0]
        + "	& 1.448(3)      \\\\ \n",
        "QB2	    & $7.7$     & $60\\times48^3$    & "
        + plaq_str[1]
        + "	& 1.6070(19)    \\\\ \n",
        "QB3	    & $7.85$    & $60\\times48^3$   & "
        + plaq_str[2]
        + "	& 1.944(3)      \\\\ \n",
        "QB4	    & $8.0$     & $60\\times48^3$    & "
        + plaq_str[3]
        + "	& 2.3149(12)    \\\\ \n",
        "QB5	    & $8.2$     & $60\\times48^3$    & "
        + plaq_str[4]
        + "	& 2.8812(21)    \\\\ \n",
    ]

    f = open("tabs/table_2.tex", "w")
    for i in range(len(line)):
        f.write(line[i])

    f.close()


def tab_4_7():
    f = open("tabs/table_4.tex", "w")

    f.write("& \\multicolumn{5}{c|}{$\\Lambda_{\\rm CB}$}   \\\\ \n")
    f.write(
        "Ansatz&  $\\hat{m}_{\\rm PS,cut}$ &  $\\hat{m}_{\\rm ps,cut}$ & $\\chi^2/N_{\\rm d.o.f.}$ &  AIC & $W$ \\\\ \hline \n"
    )
    string_tmp, w_lam = get_AIC("Lambda")
    for i in range(len(string_tmp)):
        f.write(string_tmp[i] + "  \\\\ \n")
    f.close()

    f = open("tabs/table_5.tex", "w")

    f.write(" & \\multicolumn{5}{c|}{$\\Sigma_{\\rm CB}$}  \\\\ \n")
    f.write(
        "Ansatz & $\\hat{m}_{\\rm PS,cut}$ &  $\\hat{m}_{\\rm ps,cut}$  & $\\chi^2/N_{\\rm d.o.f.}$ &  AIC & $W$ \\\\ \n\hline \n"
    )

    string_tmp, w_sig = get_AIC("Sigma")
    for i in range(len(string_tmp)):
        f.write(string_tmp[i] + "  \\\\ \n")
    f.close()

    f = open("tabs/table_6.tex", "w")

    f.write(" & \\multicolumn{5}{c|}{$\\Sigma^\\ast_{\\rm CB}$}  \\\\ \n")
    f.write(
        "Ansatz & $\\hat{m}_{\\rm PS,cut}$ &  $\\hat{m}_{\\rm ps,cut}$  & $\\chi^2/N_{\\rm d.o.f.}$ &  AIC & $W$ \\\\ \n\hline \n"
    )

    string_tmp, w_sigs = get_AIC("SigmaS")
    for i in range(len(string_tmp)):
        f.write(string_tmp[i] + "  \\\\ \n")

    f.close()

    ind_lam = np.unravel_index(w_lam.argmax(), w_lam.shape)
    ind_sig = np.unravel_index(w_sig.argmax(), w_sig.shape)
    ind_sigs = np.unravel_index(w_sigs.argmax(), w_sigs.shape)

    ind = [ind_lam, ind_sig, ind_sigs]
    chs = ["Lambda", "Sigma", "SigmaS"]

    tab_7 = []
    f = open("tabs/table_7.tex", "w")
    f.write(
        "CB & Ansatz &$\\hat{m}_{\\rm CB}^\chi $ & $ F_2  $ 	&  $A_2 $   &  $L_1 $  & $F_3 $  &   $A_3 $  & $L_{2F} $  &   $L_{2A}$ & $C_4$ \\\\ \n \hline \n"
    )

    for ch in range(3):
        tmp_tab = LB_channel(ch_ind(ch)) + " & " + ansatz[ind[ch][0]]
        opt = (
            (FIT_mass.Baryon == ch_ind(ch))
            & (FIT_mass.mPS_cut == m_PS_cut[ind[ch][2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind[ch][1]])
            & (FIT_mass.Ansatz == ansatz[ind[ch][0]])
        )
        par_tmp = FIT_mass[opt].values[0][6:]
        for p in range(8):
            tmp_tab += " & " + print_non_zero(par_tmp[2 * p], par_tmp[2 * p + 1])

        if ind[ch][0] == 1:
            tmp_tab += " & -"
        else:
            tmp_tab += " & " + print_non_zero(par_tmp[20], par_tmp[21])

        f.write(tmp_tab + "\n")
        tab_7.append(tmp_tab)


def get_F_mesons(ENS, bare_mass):
    select = (F_meson.f_bare_mass == bare_mass) & (F_meson.ENS == ENS)

    mps = print_non_zero(
        F_meson[select].m_PS.values[0], F_meson[select].m_PS_error.values[0]
    )
    r_ps_v = print_non_zero(
        F_meson[select].r_PS_V.values[0], F_meson[select].r_PS_V_error.values[0]
    )
    return " & " + mps + " & " + r_ps_v


def get_AS_mesons(ENS, bare_mass):
    select = (AS_meson.as_bare_mass == bare_mass) & (AS_meson.ENS == ENS)

    mps = print_non_zero(
        AS_meson[select].m_ps.values[0], AS_meson[select].m_ps_error.values[0]
    )
    r_ps_v = print_non_zero(
        AS_meson[select].r_ps_v.values[0], AS_meson[select].r_ps_v_error.values[0]
    )
    return mps, r_ps_v


def get_CB_mass(data, f_bare_mass, as_bare_mass):
    select = (data.as_bare_mass == as_bare_mass) & (data.f_bare_mass == f_bare_mass)

    m_lam = print_non_zero(
        data[select].m_Lambda.values[0], data[select].m_Lambda_error.values[0]
    )
    m_sig = print_non_zero(
        data[select].m_Sigma.values[0], data[select].m_Sigma_error.values[0]
    )
    m_sigS = print_non_zero(
        data[select].m_SigmaS.values[0], data[select].m_SigmaS_error.values[0]
    )

    return m_lam + " & " + m_sig + " & " + m_sigS


def print_amass(ENS, f):
    data_tmp = CB_mass[CB_mass.ENS == ENS]
    as_masses = np.unique(data_tmp.as_bare_mass.values)
    as_masses[::-1].sort()

    for i in range(as_masses.size):
        data_subset = data_tmp[data_tmp.as_bare_mass == as_masses[i]]

        f_masses = np.unique(data_subset.f_bare_mass.values)
        f_masses[::-1].sort()
        for j in range(f_masses.size):
            string_tmp = ""
            if j == 0:
                mps, r = get_AS_mesons(ENS, as_masses[i])
                string_tmp += (
                    str(f_masses[j])
                    + " & "
                    + str(as_masses[i])
                    + get_F_mesons(ENS, f_masses[j])
                    + "& \multirow{"
                    + str(f_masses.size)
                    + "}{*}{"
                    + mps
                    + "} & "
                    + "\multirow{"
                    + str(f_masses.size)
                    + "}{*}{"
                    + r
                    + "} & "
                    + get_CB_mass(data_tmp, f_masses[j], as_masses[i])
                )
            else:
                string_tmp += (
                    str(f_masses[j])
                    + " & "
                    + str(as_masses[i])
                    + get_F_mesons(ENS, f_masses[j])
                    + " & & & "
                    + get_CB_mass(data_tmp, f_masses[j], as_masses[i])
                )

            print(string_tmp + "\\\\ \n", file=f)
        print("\\hline \n", file=f)

    return data_tmp


def print_meta_F():
    f = open("tabs/table_13.tex", "w")
    f.write(
        "& 	& &     \\multicolumn{4}{c|}{PS}   &   \\multicolumn{4}{c|}{V}  \\\\ \n"
    )
    f.write(
        " Ensemble & $am_0^{(f)}$ & 	$\\epsilon^{(f)}$   "
        + "  &  $N_{\\rm W}^{\\rm source}$  	 &  $N_{\\rm W}^{\\rm sink}$ 	&  I 	   &  $\\chi^2/N_{\\rm d.o.f.}$ "
        + "&   $N_{\\rm W}^{\\rm source}$ 	 &  $N_{\\rm W}^{\\rm sink}$   &   I 	   &  $\\chi^2/N_{\\rm d.o.f.}$  \\\\ \n \\hline \n"
    )

    for N in range(5):
        ens = "QB" + str(N + 1)
        data_ens = F_meson[F_meson.ENS == ens].sort_values(
            by="f_bare_mass", ascending=False
        )

        f.write("\\multirow{" + str(data_ens.shape[0]) + "}{*}{" + ens + "} \n")
        for v in range(data_ens.shape[0]):
            string_tmp = ""
            for a in range(4):
                string_tmp += " & " + str(data_ens.values[v, a + 4])
            string_tmp += (
                " & ["
                + str(data_ens.values[v, a + 5])
                + " "
                + str(data_ens.values[v, a + 6])
                + "]"
            )
            string_tmp += " & " + str(data_ens.values[v, a + 7])

            for b in range(2):
                string_tmp += " & " + str(data_ens.values[v, b + 13])
            string_tmp += (
                " & ["
                + str(data_ens.values[v, b + 14])
                + " "
                + str(data_ens.values[v, b + 15])
                + "]"
            )
            string_tmp += " & " + str(data_ens.values[v, b + 16])

            f.write(string_tmp + " \\\\ \n")
        f.write("\\hline")

    f.close()


def print_meta_AS():
    f = open("tabs/table_14.tex", "w")
    f.write(
        "& 	& &     \\multicolumn{4}{c|}{ps}   &   \\multicolumn{4}{c|}{v}  \\\\ \n"
    )
    f.write(
        " Ensemble & $am_0^{(as)}$ & 	$\\epsilon^{(as)}$   "
        + "  &  $N_{\\rm W}^{\\rm source}$  	 &  $N_{\\rm W}^{\\rm sink}$ 	&  I 	   &  $\\chi^2/N_{\\rm d.o.f.}$ "
        + "&   $N_{\\rm W}^{\\rm source}$ 	 &  $N_{\\rm W}^{\\rm sink}$   &   I 	   &  $\\chi^2/N_{\\rm d.o.f.}$  \\\\ \n \\hline \n"
    )

    for N in range(5):
        ens = "QB" + str(N + 1)
        data_ens = AS_meson[AS_meson.ENS == ens].sort_values(
            by="as_bare_mass", ascending=False
        )
        f.write("\multirow{" + str(data_ens.shape[0]) + "}{*}{" + ens + "}")
        for v in range(data_ens.shape[0]):
            string_tmp = ""
            for a in range(4):
                string_tmp += " & " + str(data_ens.values[v, a + 4])

            string_tmp += (
                " & ["
                + str(data_ens.values[v, a + 5])
                + " "
                + str(data_ens.values[v, a + 6])
                + "]"
            )
            string_tmp += " & " + str(data_ens.values[v, a + 7])

            for b in range(2):
                string_tmp += " & " + str(data_ens.values[v, b + 13])
            string_tmp += (
                " & ["
                + str(data_ens.values[v, b + 14])
                + " "
                + str(data_ens.values[v, b + 15])
                + "]"
            )
            string_tmp += " & " + str(data_ens.values[v, b + 16])

            f.write(string_tmp + " \\\\ \n")
        f.write("\\hline")

    f.close()


def print_meta_CB(ENS, f):
    data_ens = CB_mass[CB_mass.ENS == ENS].sort_values(
        by=["as_bare_mass", "f_bare_mass"], ascending=False
    )

    gap = 0
    for v in range(data_ens.shape[0]):
        if (gap != 0) & (gap != data_ens.values[v, 5]):
            print("\\hline", file=f)

        string_tmp = ""
        for a in range(6):
            string_tmp += str(data_ens.values[v, a + 4]) + " & "

        string_tmp += (
            " ["
            + str(data_ens.values[v, a + 5])
            + " "
            + str(data_ens.values[v, a + 6])
            + "] "
        )
        string_tmp += " & " + str(data_ens.values[v, a + 7])

        for b in range(2):
            string_tmp += " & " + str(data_ens.values[v, b + 15])
        string_tmp += (
            " & ["
            + str(data_ens.values[v, b + 16])
            + " "
            + str(data_ens.values[v, b + 17])
            + "]"
        )
        string_tmp += " & " + str(data_ens.values[v, b + 18])

        for c in range(2):
            string_tmp += " & " + str(data_ens.values[v, c + 22])
        string_tmp += (
            " & ["
            + str(data_ens.values[v, c + 23])
            + " "
            + str(data_ens.values[v, c + 24])
            + "]"
        )
        string_tmp += " & " + str(data_ens.values[v, c + 25])

        print(string_tmp + " \\\\", file=f)

        gap = data_ens.values[v, 5]

    print("\\hline", file=f)


################################################################################


def main():
    if os.path.isdir("tabs") == False:
        os.mkdir("tabs")

    ################################# Global var. and loading files ################################
    global FIT_mass, m_PS_cut, m_ps_cut, Ndata, n_2D, ansatz, CB_mass, AS_meson, F_meson

    CB_mass = pd.read_csv("CSVs/CB_mass.csv")
    F_meson = pd.read_csv("CSVs/F_meson.csv")
    AS_meson = pd.read_csv("CSVs/AS_meson.csv")
    FIT_mass = pd.read_csv("CSVs/FIT_mass.csv")

    ansatz = ["M2", "M3", "MF4", "MA4", "MC4"]

    betas = CB_mass.beta.values
    f_bare_mass = CB_mass.f_bare_mass.values
    as_bare_mass = CB_mass.as_bare_mass.values

    Ndata = len(betas)
    mPS = np.zeros(Ndata)
    mps = np.zeros(Ndata)
    w0s = np.zeros(Ndata)

    for i in range(Ndata):
        mPS[i] = F_meson[
            (F_meson.beta == betas[i]) & (F_meson.f_bare_mass == f_bare_mass[i])
        ].m_PS.values[0]
        mps[i] = AS_meson[
            (AS_meson.beta == betas[i]) & (AS_meson.as_bare_mass == as_bare_mass[i])
        ].m_ps.values[0]
        w0s[i] = beta_to_w0(betas[i])

    m_PS_cut = np.round(np.arange(0.52, 1.08, 0.05), 2)
    m_ps_cut = np.round(np.arange(0.52, 1.88, 0.05), 2)
    n_2D = np.zeros(shape=(28, 12))
    lat_cutoff = (mPS < 1) & (mps < 1)

    for i in range(28):
        for j in range(12):
            select_data = (
                lat_cutoff & (mPS * w0s < m_PS_cut[j]) & (mps * w0s < m_ps_cut[i])
            )

            n_2D[i, j] = sum(select_data)

    #########################################################

    tab_1()

    tab_4_7()

    for i in range(5):
        f = open(f"tabs/table_{8+i}.tex", "w")

        f.write(
            "$am_0^{(f)}$ &  $am_0^{(as)}$ & $am_{\\rm PS}$	& $m_{\\rm PS} /  m_{\\rm V}$ & $am_{\\rm ps}$"
            + "& $m_{\\rm ps} / m_{\\rm v}$ & $am_{\\Lambda_{\\rm CB}}$      & $am_{\\Sigma_{\\rm CB}}$     & $am_{\\Sigma^\\star_{\\rm CB}}$\\\\ \n \\hline \n"
        )

        print_amass("QB" + str(i + 1), f)

        f.close()

    print_meta_F()

    print_meta_AS()

    for i in range(5):
        f = open(f"tabs/table_{15+i}.tex", "w")

        f.write(
            "& 	&  &  & \\multicolumn{4}{c|}{$\\Lambda_{\\rm CB}$}  &   "
            + "\\multicolumn{4}{c|}{$\\Sigma_{\\rm CB}$} &   \\multicolumn{4}{c|}{$\\Sigma^\\ast_{\\rm CB}$}  \\\\ \n"
        )

        f.write(
            " $am_0^{(f)}$ & $am_0^{(as)}$ & $\\epsilon^{(f)}$  & $\\epsilon^{(as)}$   &  $N_{\\rm W}^{\\rm source}$ 	"
            + " &  $N_{\\rm W}^{\\rm sink}$ 	&  I 	   &  $\\chi^2/N_{\\rm d.o.f.}$"
            + " & $N_{\\rm W}^{\\rm source}$ 	 &  $N_{\\rm W}^{\\rm sink}$   &   I 	   &  $\\chi^2/N_{\\rm d.o.f.}$"
            + "& $N_{\\rm W}^{\\rm source}$ 	 &  $N_{\\rm W}^{\\rm sink}$   &   I 	   &  $\\chi^2/N_{\\rm d.o.f.}$ \\\\ \n \hline"
        )

        print_meta_CB("QB" + str(i + 1), f)

        f.close()


if __name__ == "__main__":
    main()
