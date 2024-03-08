import fitting
import bootstrap
import numpy as np
import itertools
import matplotlib.pyplot as plt

marker = itertools.cycle(("o", "v", "*", "s", "p", "x"))
sperater = np.nditer(np.linspace(0, 0.5, 12))


def meson_mass(C_tmp, TI, TF):
    C_fold = bootstrap.fold_correlators(C_tmp)
    C_boot = bootstrap.Correlator_resample(C_fold)

    E_fit, E_fit_err, X2 = fitting.X2_single_state_fit(C_boot, TI, TF)
    print(
        fitting.print_non_zero(E_fit[-1], E_fit_err)
        + "--- Interval=["
        + str(TI)
        + ","
        + str(TF)
        + "] Xsqr= "
        + str(round(X2 / (TF - TI - 2), 2))
    )

    return E_fit, E_fit_err, round(X2 / (TF - TI - 2), 2)


def bin_proj_Baryon(corr_e, corr_o):
    def flip_boundary(C, t):
        if t == 2:
            return C
        else:
            C[:, -(t - 2) :] = -C[:, -(t - 2) :]
            return C

    size = np.shape(corr_e)

    C_e = flip_boundary(corr_e, 0)  # source location at t=0
    C_o = flip_boundary(corr_o, 0)

    C_Ebin = np.zeros(shape=(size[0], size[1] + 1))
    C_Obin = np.zeros(shape=(size[0], size[1] + 1))

    C_etmp = np.zeros(shape=(size[0], size[1] + 1))
    C_etmp[:, 0 : size[1]] = C_e
    C_etmp[:, size[1]] = C_e[:, 0]

    C_otmp = np.zeros(shape=(size[0], size[1] + 1))
    C_otmp[:, 0 : size[1]] = C_o
    C_otmp[:, size[1]] = C_o[:, 0]

    C_Ebin += (C_etmp - np.flip(C_otmp, axis=1)) / 2
    C_Obin += (C_otmp - np.flip(C_etmp, axis=1)) / 2

    return C_Ebin, -C_Obin


def baryon_mass(C_tmp, TI, TF):
    C_boot = bootstrap.Correlator_resample(C_tmp)

    E_fit, E_fit_err, X2 = fitting.X2_single_exp_fit(C_boot, TI, TF)
    print(
        fitting.print_non_zero(E_fit[-1], E_fit_err)
        + "--- Interval=["
        + str(TI)
        + ","
        + str(TF)
        + "] Xsqr= "
        + str(round(X2 / (TF - TI - 2), 2))
    )

    return E_fit, E_fit_err, round(X2 / (TF - TI - 2), 2)


def Analysis_lnC(C_resample, ti, tf, measurement):
    num_sample = np.shape(C_resample)[0]
    GLB_T = np.shape(C_resample)[1]

    Mass_channel_tmp = np.zeros(shape=(num_sample, GLB_T))
    T_dot = np.arange(ti, tf, 1, dtype=int)

    mass_tmp = []
    err_tmp = []
    for t in T_dot:
        for N in range(num_sample):
            c_t = C_resample[N, t]
            Mass_channel_tmp[N, t] = np.log(c_t)

        Mass_err = bootstrap.bootstrap_error(
            Mass_channel_tmp[0:-1, t], Mass_channel_tmp[-1, t]
        )

        mass_tmp.append(Mass_channel_tmp[-1, t])
        err_tmp.append(Mass_err)

    sprnext = next(sperater)
    plt.errorbar(
        T_dot + sprnext,
        mass_tmp,
        err_tmp,
        linestyle="",
        marker=next(marker),
        alpha=0.6,
        label=measurement,
    )

    return Mass_channel_tmp


def Analysis_Mass_eff_simple(C_resample, ti, tf, dt, measurement):
    num_sample = np.shape(C_resample)[0]
    GLB_T = np.shape(C_resample)[1]

    Mass_channel_tmp = np.zeros(shape=(num_sample, GLB_T))
    T_dot = np.arange(ti, tf, 1, dtype=int)

    mass_tmp = []
    err_tmp = []
    for t in T_dot:
        for N in range(num_sample):
            c_t1 = C_resample[N, t + dt]
            c_t = C_resample[N, t]

            Mass_channel_tmp[N, t] = -np.log(c_t1 / c_t) / dt

        Mass_err = bootstrap.bootstrap_error(
            Mass_channel_tmp[0:-1, t], Mass_channel_tmp[-1, t]
        )
        mass_tmp.append(Mass_channel_tmp[-1, t])
        err_tmp.append(Mass_err)

    sprnext = next(sperater)
    plt.errorbar(
        T_dot + sprnext,
        mass_tmp,
        err_tmp,
        linestyle="",
        marker=next(marker),
        alpha=0.6,
        label=measurement,
    )

    return Mass_channel_tmp
