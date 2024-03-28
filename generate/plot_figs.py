import numpy as np
import pandas as pd
import h5py
import sys
import os


sys.path.insert(1, "./Lib")
from plot_package import *
import extract
import bootstrap
import optimal_AIC


markers = {
    "OV12": "o",
    "OV32": "p",
    "OC": "*",
}


def get_corr_CB(ens, mf, mas, smear, channel):
    corr_e = raw_data[
        "chimera/"
        + ens
        + "/mf"
        + mf
        + "/mas"
        + mas
        + "/"
        + smear
        + "/Chimera_"
        + channel
        + "_even_re/correlators"
    ][()]
    corr_o = raw_data[
        "chimera/"
        + ens
        + "/mf"
        + mf
        + "/mas"
        + mas
        + "/"
        + smear
        + "/Chimera_"
        + channel
        + "_odd_re/correlators"
    ][()]
    C_Ebin, C_Obin = extract.bin_proj_Baryon(corr_e, corr_o)
    return bootstrap.Correlator_resample(C_Ebin), bootstrap.Correlator_resample(C_Obin)


def get_AIC(ch):
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

        x22D[ans] = x22D[ans] / (n_2D - par_ans[ans] - 1)

    w = 0.5 * np.exp(-AICs) / np.sum(np.unique(0.5 * np.exp(-AICs)))
    ind = np.unravel_index(w.argmax(), w.shape)

    return m02D, x22D, w


def main():
    if os.path.isdir("figs") == False:
        os.mkdir("figs")

    ################################# Global var. and loading files ################################
    global raw_data, FIT_mass, m_PS_cut, m_ps_cut, n_2D, Ndata

    raw_data = h5py.File("tmp_data/data.h5")
    CB_mass = pd.read_csv("CSVs/CB_mass.csv")
    F_meson = pd.read_csv("CSVs/F_meson.csv")
    AS_meson = pd.read_csv("CSVs/AS_meson.csv")
    FIT_mass = pd.read_csv("CSVs/FIT_mass.csv")
    cross_fixF = pd.read_csv("CSVs/FIT_cross_fixF.csv")
    cross_fixAS = pd.read_csv("CSVs/FIT_cross_fixAS.csv")

    ansatz = ["M2", "M3", "MF4", "MA4", "MC4"]

    betas = CB_mass.beta.values

    def export_legend(legend, filename="legend.pdf", expand=[-5,-10,5,5]):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)

    def beta_legend(fig, ax):
        legend_handles = []
        for beta in sorted(set(betas)):
            legend_handles.append(
                ax.errorbar(
                    [np.nan],
                    [np.nan],
                    yerr=[np.nan],
                    marker=beta_MRK(beta),
                    linestyle="",
                    color="k",
                    alpha=0.7,
                )[0]
            )
            legend_handles[-1].set_label(f"${beta}$")
        fig.legend(
            handles=legend_handles,
            loc="outside upper center",
            ncols=5,
            title=r"$\beta$",
        )        

    f_bare_mass = CB_mass.f_bare_mass.values
    as_bare_mass = CB_mass.as_bare_mass.values

    Ndata = len(betas)
    mPS = np.zeros(Ndata)
    mps = np.zeros(Ndata)
    MKS = np.zeros(Ndata, dtype=str)
    w0s = np.zeros(Ndata)

    for i in range(Ndata):
        mPS[i] = F_meson[
            (F_meson.beta == betas[i]) & (F_meson.f_bare_mass == f_bare_mass[i])
        ].m_PS.values[0]
        mps[i] = AS_meson[
            (AS_meson.beta == betas[i]) & (AS_meson.as_bare_mass == as_bare_mass[i])
        ].m_ps.values[0]
        MKS[i] = beta_MRK(betas[i])
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

    label_ind_f = [0, 6, -1]
    label_ind_as = [0, 9, 18, -1]

    norm_as = colors.Normalize(
        vmin=(mps * w0s).min() ** 2, vmax=(mps * w0s).max() ** 2, clip=True
    )
    mapper_as = cm.ScalarMappable(norm=norm_as, cmap=cm.viridis_r)
    mapper_as.set_array([])

    norm_f = colors.Normalize(
        vmin=(mPS * w0s).min() ** 2, vmax=(mPS * w0s).max() ** 2, clip=True
    )
    mapper_f = cm.ScalarMappable(norm=norm_f, cmap=cm.viridis_r)
    mapper_f.set_array([])

    norm_0 = colors.Normalize(vmin=-1.05, vmax=-0.8, clip=True)
    mapper_0 = cm.ScalarMappable(norm=norm_0, cmap=cm.viridis_r)
    mapper_0.set_array([])

    mPSxw0_sqr = (mPS * w0s) ** 2
    mpsxw0_sqr = (mps * w0s) ** 2

    ################################# Figure 1a ################################
    fig = plt.figure("Figure_1", figsize=(2.25, 2.25), layout="constrained")

    corr_e = -raw_data[
        "chimera/48x24x24x24b7.62/mf-0.77/mas-1.1/source_N100_sink_N0/Chimera_OC_even_re/correlators"
    ][()]
    corr_o = raw_data[
        "chimera/48x24x24x24b7.62/mf-0.77/mas-1.1/source_N100_sink_N0/Chimera_OC_odd_re/correlators"
    ][()]

    C_even = bootstrap.Correlator_resample(corr_e)
    C_odd = bootstrap.Correlator_resample(corr_o)
    lnc = extract.Analysis_lnC(C_even, 0, 48, r"even", marker="8")
    lnc = extract.Analysis_lnC(C_odd, 0, 48, r"odd", marker="v")

    plt.ylabel(r"$\rm{log}[C_{\Lambda_{\rm CB}}(t)]$")
    plt.xlabel(r"$t/a$")
    plt.xticks([0, 12, 24, 36, 48], ["0", "12", "24", "36", "48"])
    plt.legend()
    fig.savefig("figs/MCB_eolnc.pdf")
    plt.close(fig)

    next(extract.markers)
    next(extract.markers)
    ################################# Figure 1b ################################
    fig = plt.figure("Figure_1b", figsize=(2.25, 2.25), layout="constrained")
    extract.sperater.reset()

    C_Ebin, C_Obin = extract.bin_proj_Baryon(corr_e, -corr_o)
    C_Ebin = bootstrap.Correlator_resample(C_Ebin)
    C_Obin = bootstrap.Correlator_resample(C_Obin)
    C_og = bootstrap.Correlator_resample(corr_e - corr_o)

    M_tmp = extract.Analysis_Mass_eff_simple(
        C_Ebin,
        0,
        24,
        1,
        r"$am^{+}_{\rm eff,\Lambda_{\rm CB}}$",
        marker="+",
    )
    M_tmp = extract.Analysis_Mass_eff_simple(
        C_Obin,
        0,
        17,
        1,
        r"$am^{-}_{\rm eff,\Lambda_{\rm CB}}$",
        marker="x",
    )
    M_tmp = extract.Analysis_Mass_eff_simple(
        C_og,
        0,
        24,
        1,
        r"$am_{\rm eff,\Lambda_{\rm CB}}$",
        marker=markers["OC"],
    )

    plt.ylim(0.6, 2)
    plt.xlim(0, 24)
    plt.xlabel(r"$t/a$")
    plt.ylabel(r"effective mass")
    plt.legend()
    fig.savefig("figs/MCB_eo.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 1b ################################
    fig = plt.figure("Figure_1c", figsize=(2.25, 2.25), layout="constrained")
    extract.sperater.reset()

    C_Ebin, C_Obin = get_corr_CB(
        "48x24x24x24b7.62", "-0.77", "-1.1", "source_N100_sink_N0", "OV12"
    )
    M_tmp = extract.Analysis_Mass_eff_simple(
        C_Ebin,
        0,
        24,
        1,
        r"$am^{+}_{\rm eff,\Sigma_{\rm CB}}$",
        marker=">",
    )

    C_Ebin, C_Obin = get_corr_CB(
        "48x24x24x24b7.62", "-0.77", "-1.1", "source_N100_sink_N0", "OV32"
    )
    M_tmp = extract.Analysis_Mass_eff_simple(
        C_Ebin,
        0,
        24,
        1,
        r"$am^{+}_{\rm eff,\Sigma^\ast_{\rm CB}}$",
        marker="s",
    )

    C_non_proj = np.loadtxt("raw_data/C_non_proj.txt")
    # C_non_proj = np.load('output_file/C_48x24x24x24b7.62mf0.77mas1.1_Chimera_OV_gi_gi_re_APE0.4N50_sm0.1_N100_N0_s1.npy')
    M_tmp = extract.Analysis_Mass_eff_simple(
        C_non_proj,
        0,
        24,
        1,
        r"$am_{{\rm eff,CB},\mu\nu}$",
        marker="h",
    )

    plt.ylim(0.7, 1.2)
    plt.xlim(6, 24)
    plt.ylabel(r"effective mass")
    plt.xlabel(r"$t/a$")
    plt.xticks([6, 12, 18, 24], ["6", "12", "18", "24"])
    plt.legend(loc="upper right")
    fig.savefig("figs/MCB_spin.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 2a ################################
    fig = plt.figure("Figure_2a", figsize=(3.5, 2.5), layout="constrained")

    for ch in ["OC", "OV12", "OV32"]:
        C_Ebin, C_Obin = get_corr_CB(
            "60x48x48x48b8.0", "-0.6", "-0.81", "source_N100_sink_N0", ch
        )
        M_tmp = extract.Analysis_Mass_eff_simple(
            C_Ebin, 0, 35, 1, LB_channel(ch), marker=markers[ch]
        )

    plt.ylabel(r"$am_{\rm{eff,CB}}(t)$")
    plt.xlabel(r"$t/a$")
    plt.ylim(0.92, 1.04)
    plt.xlim(10, 35)
    plt.legend(loc="upper left")
    fig.savefig("figs/mh_h.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 2b ################################
    fig = plt.figure("Figure_2b", figsize=(3.5, 2.5), layout="constrained")

    for ch in ["OC", "OV12", "OV32"]:
        C_Ebin, C_Obin = get_corr_CB(
            "60x48x48x48b8.0", "-0.69", "-0.81", "source_N50_sink_N0", ch
        )
        M_tmp = extract.Analysis_Mass_eff_simple(
            C_Ebin, 0, 35, 1, LB_channel(ch), marker=markers[ch]
        )

    plt.ylabel(r"$am_{\rm{eff,CB}}(t)$")
    plt.xlabel(r"$t/a$")
    plt.ylim(0.7, 1)
    plt.xlim(10, 35)
    plt.legend(loc="upper left")
    fig.savefig("figs/mh_l.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 3 ################################

    fig, axes = plt.subplots( num="Figure_3a", figsize=(2.25, 2.7), layout="constrained" )
    for x, y, err, mk, co in zip(
        mPSxw0_sqr,
        CB_mass.m_Lambda.values * w0s,
        CB_mass.m_Lambda_error.values,
        MKS,
        mapper_as.to_rgba(mpsxw0_sqr),
    ):
        graph = axes.errorbar(
            x, y, err, marker=mk, linestyle="", color=co, alpha=0.7
        )
    axes.set_ylabel(r"$\hat{m}_{\Lambda_{\rm CB}}$")
    axes.set_xlabel(r"$\hat{m}_{\rm PS}^2$")
    fig.colorbar(mapper_as, ax=axes, label=r"$\hat{m}_{\rm ps}^2$")
    axes.set_ylim(0.9, 2.25)
    fig.savefig("figs/m_lam_mf.pdf", transparent=True)
    plt.close(fig)
    #
    #
    fig, axes = plt.subplots( num="Figure_3d", figsize=(2.25, 2.7), layout="constrained" )
    for x, y, err, mk, co in zip(
        mpsxw0_sqr,
        CB_mass.m_Lambda.values * w0s,
        CB_mass.m_Lambda_error.values,
        MKS,
        mapper_f.to_rgba(mPSxw0_sqr),
    ):
        graph = axes.errorbar(
            x, y, err, marker=mk, linestyle="", color=co, alpha=0.7
        )
        
    axes.set_ylabel(r"$\hat{m}_{\Lambda_{\rm CB}}$")
    axes.set_xlabel(r"$\hat{m}_{\rm ps}^2$")
    fig.colorbar(mapper_f, ax=axes, label=r"$\hat{m}_{\rm PS}^2$")
    axes.set_ylim(0.9, 2.25)
    fig.savefig("figs/m_lam_mas.pdf", transparent=True)
    plt.close(fig)
    #
    #
    #
    fig, axes = plt.subplots( num="Figure_3b", figsize=(2.25, 2.7), layout="constrained" )
    for x, y, err, mk, co in zip(
        mPSxw0_sqr,
        CB_mass.m_Sigma.values * w0s,
        CB_mass.m_Sigma_error.values,
        MKS,
        mapper_as.to_rgba(mpsxw0_sqr),
    ):
        graph = axes.errorbar(
            x, y, err, marker=mk, linestyle="", color=co, alpha=0.7
        )
        
    axes.set_ylabel(r"$\hat{m}_{\Sigma_{\rm CB}}$")
    axes.set_xlabel(r"$\hat{m}_{\rm PS}^2$")
    fig.colorbar(mapper_as, ax=axes, label=r"$\hat{m}_{\rm ps}^2$")
    axes.set_ylim(0.9, 2.25)
    fig.savefig("figs/m_sig_mf.pdf", transparent=True)
    plt.close(fig)
    #
    #
    fig, axes = plt.subplots( num="Figure_3e", figsize=(2.25, 2.7), layout="constrained" )
    for x, y, err, mk, co in zip(
        mpsxw0_sqr,
        CB_mass.m_Sigma.values * w0s,
        CB_mass.m_Sigma_error.values,
        MKS,
        mapper_f.to_rgba(mPSxw0_sqr),
    ):
        graph = axes.errorbar(
            x, y, err, marker=mk, linestyle="", color=co, alpha=0.7
        )
    axes.set_ylabel(r"$\hat{m}_{\Sigma_{\rm CB}}$")
    axes.set_xlabel(r"$\hat{m}_{\rm ps}^2$")
    fig.colorbar(mapper_f, ax=axes, label=r"$\hat{m}_{\rm PS}^2$")
    axes.set_ylim(0.9, 2.25)
    fig.savefig("figs/m_sig_mas.pdf", transparent=True)
    plt.close(fig)
    #
    #
    #
    fig, axes = plt.subplots( num="Figure_3c", figsize=(2.25, 2.7), layout="constrained" )
    for x, y, err, mk, co in zip(
        mPSxw0_sqr,
        CB_mass.m_SigmaS.values * w0s,
        CB_mass.m_SigmaS_error.values,
        MKS,
        mapper_as.to_rgba(mpsxw0_sqr),
    ):
        graph = axes.errorbar(
            x, y, err, marker=mk, linestyle="", color=co, alpha=0.7
        )
    axes.set_ylabel(r"$\hat{m}_{\Sigma^\ast_{\rm CB}}$")
    axes.set_xlabel(r"$\hat{m}_{\rm PS}^2$")
    fig.colorbar(mapper_as, ax=axes, label=r"$\hat{m}_{\rm ps}^2$")
    axes.set_ylim(0.9, 2.25)
    fig.savefig("figs/m_sigs_mf.pdf", transparent=True)
    plt.close(fig)
    #
    #
    fig, axes = plt.subplots( num="Figure_3f", figsize=(2.25, 2.7), layout="constrained" )
    for x, y, err, mk, co in zip(
        mpsxw0_sqr,
        CB_mass.m_SigmaS.values * w0s,
        CB_mass.m_SigmaS_error.values,
        MKS,
        mapper_f.to_rgba(mPSxw0_sqr),
    ):
        graph = axes.errorbar(
            x, y, err, marker=mk, linestyle="", color=co, alpha=0.7
        )
    axes.set_ylabel(r"$\hat{m}_{\Sigma^\ast_{\rm CB}}$")
    axes.set_xlabel(r"$\hat{m}_{\rm ps}^2$")
    fig.colorbar(mapper_f, ax=axes, label=r"$\hat{m}_{\rm PS}^2$")
    axes.set_ylim(0.9, 2.25)
    fig.savefig("figs/m_sigs_mas.pdf", transparent=True)
    plt.close(fig)
    #
    #
    #
    fig, ax = plt.subplots()
    legend_handles = []
    for beta in sorted(set(betas)):
        legend_handles.append(
            ax.errorbar(
                [np.nan],
                [np.nan],
                yerr=[np.nan],
                marker=beta_MRK(beta),
                linestyle="",
                color="k",
                alpha=0.7,
                )[0]
            )
        legend_handles[-1].set_label(f"${beta}$")
    legend = fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncols=5,
        title=r"$\beta$",
    )
    export_legend(legend, filename="figs/legend_black.pdf")
    plt.close(fig)

    ################################# Figure 4 ################################
    plt.rcParams["font.size"] = 14
    fig = plt.figure("Figure_4", figsize=(7, 5))

    ax = fig.add_subplot(111, projection="3d")

    for mps_f, mps_as, mCB, mCB_err, beta in zip(
        mPSxw0_sqr,
        mpsxw0_sqr,
        CB_mass.m_Lambda.values * w0s,
        CB_mass.m_Lambda_error.values * w0s,
        betas,
    ):
        ax.plot(
            [mps_as, mps_as],
            [mps_f, mps_f],
            [mCB + mCB_err, mCB - mCB_err],
            marker="_",
            color=beta_color(beta),
            alpha=0.6,
        )

    for beta in sorted(set(betas)):
        ax.errorbar(
            [np.nan],
            [np.nan],
            [np.nan],
            zerr=[np.nan],
            marker="none",
            ls="none",
            color=beta_color(beta),
            label=f"${beta}$",
            capsize=5,
        )

    ax.view_init(12, -45)
    ax.set_xlabel(r"$\hat{m}_{\rm ps}^2$", labelpad=10, fontsize=14)
    ax.set_ylabel(r"$\hat{m}_{\rm PS}^2$", labelpad=10, fontsize=14)
    ax.set_zlabel(r"$\hat{m}_{\Lambda_{\rm CB}}$", labelpad=5, fontsize=14)
    ax.set_zlim(0.5, 2.5)
    ax.legend(loc="upper center", title=r"$\beta$", ncols=5, fontsize=12)
    fig.tight_layout()
    fig.savefig("figs/m_lam_3d.pdf", transparent=True)
    plt.rcParams["font.size"] = 8

    ################################# Figure 5 ################################

    fig, ax = plt.subplots(1, 2, num="Figure_5", figsize=(7, 3.2), layout="constrained")

    for x, y, err, mk, co in zip(
        mPSxw0_sqr,
        CB_mass.R_Lambda_Sigma.values,
        CB_mass.R_Lambda_Sigma_error.values,
        MKS,
        mapper_as.to_rgba(mpsxw0_sqr),
    ):
        graph = ax[1].errorbar(x, y, err, marker=mk, linestyle="", color=co, alpha=0.7)

    for x, y, err, mk, co in zip(
        mpsxw0_sqr,
        CB_mass.R_Lambda_Sigma.values,
        CB_mass.R_Lambda_Sigma_error.values,
        MKS,
        mapper_f.to_rgba(mPSxw0_sqr),
    ):
        graph = ax[0].errorbar(x, y, err, marker=mk, linestyle="", color=co, alpha=0.7)

    ax[1].set_ylabel(r"$m_{\Lambda_{\rm CB}} / m_{\Sigma_{\rm CB}}$")
    ax[0].set_ylabel(r"$m_{\Lambda_{\rm CB}} / m_{\Sigma_{\rm CB}}$")
    ax[1].set_xlabel(r"$\hat{m}_{\rm PS}^2$")
    ax[0].set_xlabel(r"$\hat{m}_{\rm ps}^2$")
    ax[0].plot([0,5],[1,1], alpha=0.8, color='k')
    ax[1].plot([0,5],[1,1], alpha=0.8, color='k')
    ax[0].set_xlim(0,4)
    ax[1].set_xlim(0,1.2)
    
    fig.colorbar(mapper_as , ax=ax[1], label=r"$\hat{m}_{\rm ps}^2$")
    fig.colorbar(mapper_f, ax=ax[0], label=r"$\hat{m}_{\rm PS}^2$")
    beta_legend(fig, ax[0])
    fig.savefig("figs/mh_mps2.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 6 ################################

    fig, ax = plt.subplots(1, 2, num="Figure_6", figsize=(7, 3.2), layout="constrained")

    for x, y, err, mk, co in zip(
        mPSxw0_sqr,
        CB_mass.R_Sigma_SigmaS.values,
        CB_mass.R_Sigma_SigmaS_error.values,
        MKS,
        mapper_as.to_rgba(mpsxw0_sqr),
    ):
        graph = ax[1].errorbar(x, y, err, marker=mk, linestyle="", color=co, alpha=0.7)

    for x, y, err, mk, co in zip(
        mpsxw0_sqr,
        CB_mass.R_Sigma_SigmaS.values,
        CB_mass.R_Sigma_SigmaS_error.values,
        MKS,
        mapper_f.to_rgba(mPSxw0_sqr),
    ):
        graph = ax[0].errorbar(x, y, err, marker=mk, linestyle="", color=co, alpha=0.7)

    ax[1].set_ylabel(r"$m_{ \Sigma_{\rm CB} } / m_{\Sigma^\ast_{\rm CB}}$")
    ax[0].set_ylabel(r"$m_{ \Sigma_{\rm CB} } / m_{\Sigma^\ast_{\rm CB}}$")
    ax[1].set_xlabel(r"$\hat{m}_{\rm PS}^2$")
    ax[0].set_xlabel(r"$\hat{m}_{\rm ps}^2$")
    ax[0].plot([0,5],[1,1], alpha=0.8, color='k')
    ax[1].plot([0,5],[1,1], alpha=0.8, color='k')
    ax[0].set_xlim(0,4)
    ax[1].set_xlim(0,1.2)
    
    fig.colorbar(mapper_as, ax=ax[1], label=r"$\hat{m}_{\rm ps}^2$")
    fig.colorbar(mapper_f, ax=ax[0], label=r"$\hat{m}_{\rm PS}^2$")
    beta_legend(fig, ax[0])
    fig.savefig("figs/ss_mps2.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 7 ################################
    plt.rcParams["font.size"] = 10 
    fig, ax = plt.subplots(3, 5, num="Figure_7", figsize=(7, 6), layout="constrained")

    m0_lam, x2_lam, w_lam = get_AIC("Lambda")

    for i in range(5):
        imm = ax[2][i].imshow(
            m0_lam[i], origin="lower", vmin=0.6, vmax=1.2, cmap=colormaps["viridis"]
        )
        imw = ax[1][i].imshow(
            w_lam[i],
            origin="lower",
            vmin=0,
            vmax=w_lam.max(),
            cmap=colormaps["plasma"],
        )
        imx = ax[0][i].imshow(
            x2_lam[i], origin="lower", vmin=0, vmax=2, cmap=colormaps["twilight"]
        )

        labels_f = [r"$" + str(x) + " $" for x in m_PS_cut[label_ind_f]]
        labels_as = [r"$" + str(x) + " $" for x in m_ps_cut[label_ind_as]]

        ax[0][i].set_yticks(np.arange(28)[label_ind_as])
        ax[0][i].set_yticklabels(labels_as)
        ax[0][i].set_xticks(np.arange(12)[label_ind_f])
        ax[0][i].set_xticklabels(labels_f)

        ax[1][i].set_yticks(np.arange(28)[label_ind_as])
        ax[1][i].set_yticklabels(labels_as)
        ax[1][i].set_xticks(np.arange(12)[label_ind_f])
        ax[1][i].set_xticklabels(labels_f)

        ax[2][i].set_yticks(np.arange(28)[label_ind_as])
        ax[2][i].set_yticklabels(labels_as)
        ax[2][i].set_xticks(np.arange(12)[label_ind_f])
        ax[2][i].set_xticklabels(labels_f)

        ax[0][i].set_title(r"$\rm{" + ansatz[i] + "}$")

    cbtick = np.linspace(0.6, 1.2, 3)

    cbm = fig.colorbar(
        imm,
        ax=ax[2][4],
        label=r"$\hat{m}_{\Lambda_{\rm CB}}^\chi$",
        ticks=cbtick,
    )
    fig.colorbar(imw, ax=ax[1][4], label=r"$W$")
    cbm_x = fig.colorbar(
        imx, ax=ax[0][4], label=r"$\chi^2/N_{\rm d.o.f.}$", ticks=[0, 0.5, 1, 1.5, 2]
    )
    cbm_x.ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$", r"$1.5$", r"$\geq 2$"])

    fig.supxlabel(r"$\hat{m}_{\rm PS,cut}$")
    fig.supylabel(r"$\hat{m}_{\rm ps,cut}$")
    fig.savefig("figs/M_OC_AIC.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 8 ################################
    fig, ax = plt.subplots(3, 5, num="Figure_8", figsize=(7, 6), layout="constrained")

    for i in range(28):
        for j in range(12):
            select_data = (
                lat_cutoff & (mPS * w0s < m_PS_cut[j]) & (mps * w0s < m_ps_cut[i])
            )

            n_2D[i, j] = sum(select_data)

    m0_sig, x2_sig, w_sig = get_AIC("Sigma")

    for i in range(5):
        imm = ax[2][i].imshow(
            m0_sig[i], origin="lower", vmin=0.6, vmax=1.2, cmap=colormaps["viridis"]
        )
        imw = ax[1][i].imshow(
            w_sig[i],
            origin="lower",
            vmin=0,
            vmax=w_sig.max(),
            cmap=colormaps["plasma"],
        )
        imx = ax[0][i].imshow(
            x2_sig[i], origin="lower", vmin=0, vmax=2, cmap=colormaps["twilight"]
        )

        labels_f = [r"$" + str(x) + " $" for x in m_PS_cut[label_ind_f]]
        labels_as = [r"$" + str(x) + " $" for x in m_ps_cut[label_ind_as]]

        ax[0][i].set_yticks(np.arange(28)[label_ind_as])
        ax[0][i].set_yticklabels(labels_as)
        ax[0][i].set_xticks(np.arange(12)[label_ind_f])
        ax[0][i].set_xticklabels(labels_f)

        ax[1][i].set_yticks(np.arange(28)[label_ind_as])
        ax[1][i].set_yticklabels(labels_as)
        ax[1][i].set_xticks(np.arange(12)[label_ind_f])
        ax[1][i].set_xticklabels(labels_f)

        ax[2][i].set_yticks(np.arange(28)[label_ind_as])
        ax[2][i].set_yticklabels(labels_as)
        ax[2][i].set_xticks(np.arange(12)[label_ind_f])
        ax[2][i].set_xticklabels(labels_f)

        ax[0][i].set_title(r"$\rm{" + ansatz[i] + "}$")

    cbtick = np.linspace(0.6, 1.2, 3)

    cbm = fig.colorbar(
        imm,
        ax=ax[2][4],
        label=r"$\hat{m}_{\Sigma_{\rm CB}}^\chi$",
        ticks=cbtick,
    )
    fig.colorbar(imw, ax=ax[1][4], label=r"$W$")
    cbm_x = fig.colorbar(
        imx, ax=ax[0][4], label=r"$\chi^2/N_{\rm d.o.f.}$", ticks=[0, 0.5, 1, 1.5, 2]
    )
    cbm_x.ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$", r"$1.5$", r"$\geq 2$"])

    fig.supxlabel(r"$\hat{m}_{\rm PS,cut}$")
    fig.supylabel(r"$\hat{m}_{\rm ps,cut}$")
    fig.savefig("figs/M_OV12_AIC.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 9 ################################
    fig, ax = plt.subplots(3, 5, num="Figure_9", figsize=(7, 6), layout="constrained")

    for i in range(28):
        for j in range(12):
            select_data = (
                lat_cutoff & (mPS * w0s < m_PS_cut[j]) & (mps * w0s < m_ps_cut[i])
            )

            n_2D[i, j] = sum(select_data)

    m0_sigs, x2_sigs, w_sigs = get_AIC("SigmaS")

    for i in range(5):
        imm = ax[2][i].imshow(
            m0_sigs[i], origin="lower", vmin=0.4, vmax=1.6, cmap=colormaps["viridis"]
        )
        imw = ax[1][i].imshow(
            w_sigs[i],
            origin="lower",
            vmin=0,
            vmax=w_sigs.max(),
            cmap=colormaps["plasma"],
        )
        imx = ax[0][i].imshow(
            x2_sigs[i], origin="lower", vmin=0, vmax=2, cmap=colormaps["twilight"]
        )

        labels_f = [r"$" + str(x) + " $" for x in m_PS_cut[label_ind_f]]
        labels_as = [r"$" + str(x) + " $" for x in m_ps_cut[label_ind_as]]

        ax[0][i].set_yticks(np.arange(28)[label_ind_as])
        ax[0][i].set_yticklabels(labels_as)
        ax[0][i].set_xticks(np.arange(12)[label_ind_f])
        ax[0][i].set_xticklabels(labels_f)

        ax[1][i].set_yticks(np.arange(28)[label_ind_as])
        ax[1][i].set_yticklabels(labels_as)
        ax[1][i].set_xticks(np.arange(12)[label_ind_f])
        ax[1][i].set_xticklabels(labels_f)

        ax[2][i].set_yticks(np.arange(28)[label_ind_as])
        ax[2][i].set_yticklabels(labels_as)
        ax[2][i].set_xticks(np.arange(12)[label_ind_f])
        ax[2][i].set_xticklabels(labels_f)

        ax[0][i].set_title(r"$\rm{" + ansatz[i] + "}$")

    cbtick = np.linspace(0.64, 1.6, 3)

    cbm = fig.colorbar(
        imm,
        ax=ax[2][4],
        label=r"$\hat{m}_{\Sigma^\ast_{\rm CB}}^\chi$",
        ticks=cbtick,
    )
    fig.colorbar(imw, ax=ax[1][4], label=r"$W$")
    cbm_x = fig.colorbar(
        imx, ax=ax[0][4], label=r"$\chi^2/N_{\rm d.o.f.}$", ticks=[0, 0.5, 1, 1.5, 2]
    )
    cbm_x.ax.set_yticklabels([r"$0$", r"$0.5$", r"$1$", r"$1.5$", r"$\geq 2$"])

    fig.supxlabel(r"$\hat{m}_{\rm PS,cut}$")
    fig.supylabel(r"$\hat{m}_{\rm ps,cut}$")
    fig.savefig("figs/M_OV32_AIC.pdf", transparent=True)
    plt.close(fig)

    ind_lam = np.unravel_index(w_lam.argmax(), w_lam.shape)
    ind_sig = np.unravel_index(w_sig.argmax(), w_sig.shape)
    ind_sigs = np.unravel_index(w_sigs.argmax(), w_sigs.shape)

    ################################# Figure 10 ################################
    plt.rcParams["font.size"] = 8
    fig, ax = plt.subplots(3, 3, num="Figure_10", figsize=(7, 5), layout="constrained")

    ind = [ind_lam, ind_sig, ind_sigs]
    hatches = ["/", "x", "\\", "|", "O"]

    parlb = [r"$\tilde{m}^\chi_{\rm CB}$", r"$\tilde{F}_2$", r"$\tilde{F}_3$"]

    for ch in range(3):
        for x, y, err, beta in zip(
            (cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].w0mps.values) ** 2,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].M_CB.values,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].M_CB_err.values,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].beta.values,
        ):
            graph = ax[0][ch].errorbar(
                x,
                y,
                err,
                marker=beta_MRK(beta),
                linestyle="",
                color="r",
                alpha=0.7,
                label=str(beta),
            )

        for x, y, err, beta in zip(
            (cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].w0mps.values) ** 2,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].A2.values,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].A2_err.values,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].beta.values,
        ):
            graph = ax[1][ch].errorbar(
                x,
                y,
                err,
                marker=beta_MRK(beta),
                linestyle="",
                color="g",
                alpha=0.7,
                label=str(beta),
            )

        for x, y, err, beta in zip(
            (cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].w0mps.values) ** 2,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].A3.values,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].A3_err.values,
            cross_fixAS[cross_fixAS.Baryon == ch_ind(ch)].beta.values,
        ):
            graph = ax[2][ch].errorbar(
                x,
                y,
                err,
                marker=beta_MRK(beta),
                linestyle="",
                color="b",
                alpha=0.7,
                label=str(beta),
            )

        ax[0][ch].set_ylim(0, 2)
        ax[1][ch].set_ylim(-0.5, 1.5)
        ax[2][ch].set_ylim(-0.54, 1.04)

        cut_m = (m_ps_cut[ind[ch][1]]) ** 2
        opt = (
            (FIT_mass.Baryon == ch_ind(ch))
            & (FIT_mass.mPS_cut == m_PS_cut[ind[ch][2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind[ch][1]])
            & (FIT_mass.Ansatz == ansatz[ind[ch][0]])
        )

        par_tmp = FIT_mass[opt].M_CB.values[0]
        err_tmp = FIT_mass[opt].M_CB_err.values[0]
        ax[0][ch].fill_between(
            [0, cut_m],
            par_tmp + err_tmp,
            par_tmp - err_tmp,
            alpha=0.5,
            facecolor="lightgrey",
            edgecolor="r",
            hatch=hatches[ind[ch][0]],
        )

        par_tmp = FIT_mass[opt].F2.values[0]
        err_tmp = FIT_mass[opt].F2_err.values[0]
        ax[1][ch].fill_between(
            [0, cut_m],
            par_tmp + err_tmp,
            par_tmp - err_tmp,
            alpha=0.5,
            facecolor="lightgrey",
            edgecolor="g",
            hatch=hatches[ind[ch][0]],
        )

        par_tmp = FIT_mass[opt].F3.values[0]
        err_tmp = FIT_mass[opt].F3_err.values[0]
        ax[2][ch].fill_between(
            [0, cut_m],
            par_tmp + err_tmp,
            par_tmp - err_tmp,
            alpha=0.5,
            facecolor="lightgrey",
            edgecolor="b",
            hatch=hatches[ind[ch][0]],
            label=ansatz[ind[ch][0]],
        )

        ax[0][ch].set_title(r"$" + LB_ind(ch) + "$")
        for i in range(3):
            ax[i][0].set_ylabel(parlb[i])
            ax[i][ch].set_xlim(0, 4)
            ax[2][ch].set_xlabel(r"$\hat{m}_{\rm ps}^2$")

    ax[2][ch].fill_between(
        [-0, -1],
        par_tmp + err_tmp,
        par_tmp - err_tmp,
        alpha=0.5,
        facecolor="lightgrey",
        edgecolor="b",
        hatch=hatches[ind[0][0]],
        label=ansatz[ind[0][0]],
    )
    handles, labels = fig.gca().get_legend_handles_labels()

    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="outside lower center", ncol=6)
    fig.savefig("figs/fix_AS_check.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 11 ################################

    fig, ax = plt.subplots(3, 3, num="Figure_11", figsize=(7, 5), layout="constrained")

    parlb = [r"$\tilde{m}^\chi_{\rm CB}$", r"$\tilde{A}_2$", r"$\tilde{A}_3$"]

    for ch in range(3):
        for x, y, err, beta in zip(
            (cross_fixF[cross_fixF.Baryon == ch_ind(ch)].w0mPS.values) ** 2,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].M_CB.values,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].M_CB_err.values,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].beta.values,
        ):
            graph = ax[0][ch].errorbar(
                x,
                y,
                err,
                marker=beta_MRK(beta),
                linestyle="",
                color="r",
                alpha=0.7,
                label=str(beta),
            )

        for x, y, err, beta in zip(
            (cross_fixF[cross_fixF.Baryon == ch_ind(ch)].w0mPS.values) ** 2,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].F2.values,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].F2_err.values,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].beta.values,
        ):
            graph = ax[1][ch].errorbar(
                x,
                y,
                err,
                marker=beta_MRK(beta),
                linestyle="",
                color="g",
                alpha=0.7,
                label=str(beta),
            )

        for x, y, err, beta in zip(
            (cross_fixF[cross_fixF.Baryon == ch_ind(ch)].w0mPS.values) ** 2,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].F3.values,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].F3_err.values,
            cross_fixF[cross_fixF.Baryon == ch_ind(ch)].beta.values,
        ):
            graph = ax[2][ch].errorbar(
                x,
                y,
                err,
                marker=beta_MRK(beta),
                linestyle="",
                color="b",
                alpha=0.7,
                label=str(beta),
            )

        cut_m = (m_PS_cut[ind[ch][2]]) ** 2
        opt = (
            (FIT_mass.Baryon == ch_ind(ch))
            & (FIT_mass.mPS_cut == m_PS_cut[ind[ch][2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind[ch][1]])
            & (FIT_mass.Ansatz == ansatz[ind[ch][0]])
        )

        par_tmp = FIT_mass[opt].M_CB.values[0]
        err_tmp = FIT_mass[opt].M_CB_err.values[0]
        ax[0][ch].fill_between(
            [0, cut_m],
            par_tmp + err_tmp,
            par_tmp - err_tmp,
            alpha=0.5,
            facecolor="lightgrey",
            edgecolor="r",
            hatch=hatches[ind[ch][0]],
        )

        par_tmp = FIT_mass[opt].A2.values[0]
        err_tmp = FIT_mass[opt].A2_err.values[0]
        ax[1][ch].fill_between(
            [0, cut_m],
            par_tmp + err_tmp,
            par_tmp - err_tmp,
            alpha=0.5,
            facecolor="lightgrey",
            edgecolor="g",
            hatch=hatches[ind[ch][0]],
        )

        par_tmp = FIT_mass[opt].A3.values[0]
        err_tmp = FIT_mass[opt].A3_err.values[0]
        ax[2][ch].fill_between(
            [0, cut_m],
            par_tmp + err_tmp,
            par_tmp - err_tmp,
            alpha=0.5,
            facecolor="lightgrey",
            edgecolor="b",
            hatch=hatches[ind[ch][0]],
            label=ansatz[ind[ch][0]],
        )

        ax[0][ch].set_title(r"$" + LB_ind(ch) + "$")
        ax[0][ch].set_ylim(0, 1.6)
        ax[1][ch].set_ylim(-0.2, 1)
        ax[2][ch].set_ylim(-0.4, 0.4)

        for i in range(3):
            ax[i][0].set_ylabel(parlb[i])
            ax[i][ch].set_xlim(0, 1.2)
            ax[2][ch].set_xlabel(r"$\hat{m}_{\rm PS}^2$")

    ax[2][ch].fill_between(
        [-0, -1],
        par_tmp + err_tmp,
        par_tmp - err_tmp,
        alpha=0.5,
        facecolor="lightgrey",
        edgecolor="b",
        hatch=hatches[ind[0][0]],
        label=ansatz[ind[0][0]],
    )

    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="outside lower center", ncol=6)
    fig.savefig("figs/fix_F_check.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 12 ################################
    fig, ax = plt.subplots(1, 2, num="Figure_12", figsize=(7, 4), layout="constrained")

    ##########
    print("It may take a while...", end="")
    par_lam = optimal_AIC.get_pars(
        "Lambda", ind_lam[0], m_PS_cut[ind_lam[2]], m_ps_cut[ind_lam[1]]
    )

    plot_line_fixAS(
        ax[0], par_lam, r"$\Lambda_{\rm CB}$", m_PS_cut[ind_lam[2]], ind_lam[0]
    )
    plot_line_fixF(
        ax[1], par_lam, r"$\Lambda_{\rm CB}$", m_ps_cut[ind_lam[1]], ind_lam[0]
    )
    ##########
    print("almost there...", end="")
    par_sig = optimal_AIC.get_pars(
        "Sigma", ind_sig[0], m_PS_cut[ind_sig[2]], m_ps_cut[ind_sig[1]]
    )

    plot_line_fixAS(
        ax[0], par_sig, r"$\Sigma_{\rm CB}$", m_PS_cut[ind_sig[2]], ind_sig[0]
    )
    plot_line_fixF(
        ax[1], par_sig, r"$\Sigma_{\rm CB}$", m_ps_cut[ind_sig[1]], ind_sig[0]
    )
    ##########

    par_sigs = optimal_AIC.get_pars(
        "SigmaS", ind_sigs[0], m_PS_cut[ind_sigs[2]], m_ps_cut[ind_sigs[1]]
    )

    plot_line_fixAS(
        ax[0], par_sigs, r"$\Sigma^\ast_{\rm CB}$", m_PS_cut[ind_sigs[2]], ind_sigs[0]
    )
    plot_line_fixF(
        ax[1], par_sigs, r"$\Sigma^\ast_{\rm CB}$", m_ps_cut[ind_sigs[1]], ind_sigs[0]
    )
    print("DONE!")

    ax[0].set_xlabel(r"$\hat{m}_{\textrm{PS}}^2$")
    ax[0].set_ylabel(r"$\hat{m}_{\rm{CB}}$ at $\hat{m}_{\textrm{ps}}^2 = 0$")
    ax[0].set_ylim(0.75, 1.8)
    ax[0].set_xlim(0, 1.2)

    ax[1].set_xlabel(r"$\hat{m}_{\textrm{ps}}^2$")
    ax[1].set_ylabel(r"$\hat{m}_{\rm{CB}}$ at $\hat{m}_{\textrm{PS}}^2 = 0$")
    ax[1].set_ylim(0.75, 1.8)
    ax[1].set_xlim(0, 3.4)

    fig.legend(loc="outside upper center", ncol=3)

    fig.savefig("figs/plot_FIT_massless.pdf", transparent=True)
    plt.close(fig)

    ################################# Figure 13 ################################
    plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(
        1, 1, num="Figure_13", layout="constrained", figsize=(7, 5.2)
    )

    plot_line(ax, r"PS", 0, 0.04, 0.0, 0.0)
    plot_line(ax, r"V", -0.05, 0.04, 0.451, 0.013)
    plot_line(ax, r"T", 0.05, 0.04, 0.455, 0.020)
    plot_line(ax, r"AV", -0.08, 0, 1.14, 0.10)
    plot_line(ax, r"AT", 0.08, 0, 1.36, 0.09)
    plot_line(ax, r"S", 0, 0.05, 1.52, 0.09)

    plot_line_as(ax, r"ps", 0, 0.04, 0.0, 0.0)
    plot_line_as(ax, r"v", -0.05, 0.04, 0.657, 0.021)
    plot_line_as(ax, r"t", 0.05, 0.04, 0.675, 0.029)
    plot_line_as(ax, r"av", 0, 0, 2.01, 0.11)
    plot_line_as(ax, r"at", 0, 0, 2.5, 0.18)
    plot_line_as(ax, r"s", 0, 0, 1.8, 0.09)

    plot_line_cb(
        ax,
        r"$\Lambda_{\rm CB}$",
        0,
        0.032,
        FIT_mass[
            (FIT_mass.Baryon == "Lambda")
            & (FIT_mass.Ansatz == ansatz[ind_lam[0]])
            & (FIT_mass.mPS_cut == m_PS_cut[ind_lam[2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind_lam[1]])
        ].M_CB.values[0],
        FIT_mass[
            (FIT_mass.Baryon == "Lambda")
            & (FIT_mass.Ansatz == ansatz[ind_lam[0]])
            & (FIT_mass.mPS_cut == m_PS_cut[ind_lam[2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind_lam[1]])
        ].M_CB_err.values[0],
    )

    plot_line_cb(
        ax,
        r"$\Sigma_{\rm CB}$",
        0,
        0.02,
        FIT_mass[
            (FIT_mass.Baryon == "Sigma")
            & (FIT_mass.Ansatz == ansatz[ind_sig[0]])
            & (FIT_mass.mPS_cut == m_PS_cut[ind_sig[2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind_sig[1]])
        ].M_CB.values[0],
        FIT_mass[
            (FIT_mass.Baryon == "Sigma")
            & (FIT_mass.Ansatz == ansatz[ind_sig[0]])
            & (FIT_mass.mPS_cut == m_PS_cut[ind_sig[2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind_sig[1]])
        ].M_CB_err.values[0],
    )

    plot_line_cb(
        ax,
        r"$\Sigma^*_{\rm CB}$",
        0,
        0.045,
        FIT_mass[
            (FIT_mass.Baryon == "SigmaS")
            & (FIT_mass.Ansatz == ansatz[ind_sigs[0]])
            & (FIT_mass.mPS_cut == m_PS_cut[ind_sigs[2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind_sigs[1]])
        ].M_CB.values[0],
        FIT_mass[
            (FIT_mass.Baryon == "SigmaS")
            & (FIT_mass.Ansatz == ansatz[ind_sigs[0]])
            & (FIT_mass.mPS_cut == m_PS_cut[ind_sigs[2]])
            & (FIT_mass.mps_cut == m_ps_cut[ind_sigs[1]])
        ].M_CB_err.values[0],
    )

    plot_line_gb(ax, r"$0^+$", 0, -0.05, 1.398, 0.073, 3.3, 4.3)
    plot_line_gb(ax, r"$0^-$", 0, -0.050, 2.10, 0.13, 3.3, 4.3)

    plot_line_gb(ax, r"$1^+$", 0, -0.05, 3.24, 0.29, 3.3, 3.475)
    plot_line_gb(ax, r"$1^-$", 0, -0.05, 3.36, 0.29, 3.575, 3.75)

    plot_line_gb(ax, r"$2^+$", 0, -0.05, 1.75, 0.10, 3.3, 4.3)
    plot_line_gb(ax, r"$2^-$", 0, -0.03, 2.60, 0.14, 3.3, 3.75)

    plot_line_gb(ax, r"$3^+$", 0, -0.04, 2.98, 0.23, 3.85, 4.025)
    plot_line_gb(ax, r"$3^-$", 0, -0.05, 3.03, 0.56, 4.125, 4.3)

    # ax.set_xlabel(r'$(\hat{m}_{\textrm{PS}})^2$')
    ax.set_xticks([0.5, 1.6, 2.7, 3.8])
    ax.set_xticklabels(
        [r"$(f)$ meson", r"$(as)$ meson", r"Chimera Baryon", r"Glueball"]
    )
    ax.set_xticklabels([r"", r"", r"", r""])
    ax.set_ylabel(r"$\hat{m}$")

    ax.set_ylim(-0.05, 4)

    ax2 = ax.twinx()
    ax2.set_ylim(-0.05 / 0.08746427842267951, 4 / 0.08746427842267951)
    ax2.set_ylabel(r"$\hat{m}/\hat{f}_{\rm PS}$")

    y_up = -0.1
    y_dn = -0.2
    ax.fill_between([0, 1], y_up, y_dn, alpha=0.4, label=r"$(f)$ meson", color="b")
    ax.fill_between([0, 1], y_up, y_dn, alpha=0.4, label=r"$(as)$ meson", color="r")
    ax.fill_between([0, 1], y_up, y_dn, alpha=0.4, label=r"Chimera Baryon", color="k")
    ax.fill_between([0, 1], y_up, y_dn, alpha=0.4, label=r"Glueball", color="y")

    ax.set_xticks([])
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95))

    fig.savefig("figs/plot_quench_all_m0.pdf", transparent=True)
    plt.close(fig)

    if "show_3d" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
