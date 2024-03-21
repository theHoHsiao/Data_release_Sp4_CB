import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

from matplotlib import rc, rcParams, cm, colors, colormaps
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm, colors
import numpy as np
import bootstrap

import os

os.environ["PATH"] = (
    "/Library/Frameworks/Python.framework/Versions/3.11/bin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin"
)


scale = 0.5
W = 12 * scale
r = 0.8

rcParams.update(
    {
        "figure.figsize": (W, W * r),  # 4:3 aspect ratio
        "font.size": 16 * scale,  # Set font size to 11pt
        "axes.labelsize": 16 * scale,  # -> axis labels
        "legend.fontsize": 16 * scale,  # -> legends
        "lines.markersize": 10 * scale,
        "font.family": "lmodern",
        "text.usetex": True,
        "text.latex.preamble": (  # LaTeX preamble
            r"\usepackage{lmodern}"
            # ... more packages if needed
        ),
    }
)


def beta_MRK(b):
    if b == "7.62":
        return "o"
    elif b == "7.7":
        return "v"
    elif b == "7.85":
        return "*"
    elif b == "8.0":
        return "s"
    else:
        return "p"


def beta_color(b):
    if b == 7.62:
        return "r"
    elif b == 7.7:
        return "g"
    elif b == 7.85:
        return "b"
    elif b == 8.0:
        return "k"
    elif b == 8.2:
        return "y"
    else:
        return "c"


def ch_lb(b):
    if b == "Lambda":
        return "\Lambda"
    elif b == "S":
        return "\Sigma"
    elif b == "S*":
        return "\Sigma*"
    else:
        return "p"


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


def beta_MRK(b):
    if b == 7.62:
        return "o"
    elif b == 7.7:
        return "v"
    elif b == 7.85:
        return "*"
    elif b == 8.0:
        return "s"
    elif b == 8.2:
        return "h"
    else:
        return "p"


def LB_chimera(b):
    if b == "Lambda":
        return "Chimera_OC"
    elif b == "Sigma":
        return "Chimera_OV12"
    elif b == "SigmaS":
        return "Chimera_OV32"
    else:
        return "Chimera"


def LB_channel(ch):
    return {
        "OC": r"$\Lambda_{\rm CB}$",
        "OV12": r"$\Sigma_{\rm CB}$",
        "OV32": r"$\Sigma^\ast_{\rm CB}$",
    }.get(ch, ch)


def beta_to_w0(b):
    if b == 7.62:
        return 1.448
    elif b == 7.7:
        return 1.6070
    elif b == 7.85:
        return 1.944
    elif b == 8.0:
        return 2.3149
    elif b == 8.2:
        return 2.8812
    else:
        return 0


def ch_ind(ch):
    return {
        0: "Lambda",
        1: "Sigma",
        2: "SigmaS",
    }.get(ch, ch)


def LB_ind(ch):
    return {
        0: r"$\Lambda_{\rm CB}$",
        1: r"$\Sigma_{\rm CB}$",
        2: r"$\Sigma^\ast_{\rm CB}$",
    }.get(ch, ch)


def plot_line_fixAS(ax, fit_tmp, ch, cutv, offset):
    pars = np.zeros(shape=(11, fit_tmp.shape[1]))
    for i in range(fit_tmp.shape[0]):
        if i == 8:
            pars[i + offset - 2] = fit_tmp[i]

        else:
            pars[i] = fit_tmp[i]

    n_fit = 1000
    Yfit = np.zeros(shape=(fit_tmp.shape[1], n_fit))

    x1 = np.linspace(0, cutv, n_fit)
    x2 = 0  # mps could be non-zero values

    y_up = np.zeros(n_fit)
    y_dn = np.zeros(n_fit)
    # print(pars[:,-1])
    for n in range(fit_tmp.shape[1]):
        Yfit[n] = (
            pars[0, n]
            + pars[1, n] * x1**2
            + pars[2, n] * x2**2
            + pars[4, n] * x1**3
            + pars[5, n] * x2**3
            + pars[8, n] * x1**4
            + pars[9, n] * x2**4
            + pars[10, n] * x1**2 * x2**2
        )  # +\
        # pars[3,n] * lat_a  + pars[6,n] * x1**2 * lat_a + pars[7,n]* x2**2  * lat_a

    for i in range(n_fit):
        y_err = bootstrap.bootstrap_error(Yfit[0:-1, i], Yfit[-1, i])
        y_up[i] = Yfit[-1, i] + y_err
        y_dn[i] = Yfit[-1, i] - y_err

    ax.plot(x1**2, Yfit[-1], "--", linewidth=0.75, alpha=0.6)
    ax.fill_between(
        x1**2, y_up, y_dn, alpha=0.4
    )  # , color=plt.gca().lines[-1].get_color(),  label=ch_lb(ch))


def plot_line_fixF(ax, fit_tmp, ch, cutv, offset):
    pars = np.zeros(shape=(11, fit_tmp.shape[1]))
    for i in range(fit_tmp.shape[0]):
        if i == 8:
            pars[i + offset - 2] = fit_tmp[i]

        else:
            pars[i] = fit_tmp[i]

    n_fit = 1000
    Yfit = np.zeros(shape=(fit_tmp.shape[1], n_fit))

    x2 = np.linspace(0, cutv, n_fit)
    x1 = 0  # mPS could be non-zero values

    y_up = np.zeros(n_fit)
    y_dn = np.zeros(n_fit)

    for n in range(fit_tmp.shape[1]):
        Yfit[n] = (
            pars[0, n]
            + pars[1, n] * x1**2
            + pars[2, n] * x2**2
            + pars[4, n] * x1**3
            + pars[5, n] * x2**3
            + pars[8, n] * x1**4
            + pars[9, n] * x2**4
            + pars[10, n] * x1**2 * x2**2
        )  # +\
        # pars[3,n] * lat_a  + pars[6,n] * x1**2 * lat_a + pars[7,n]* x2**2  * lat_a

    for i in range(n_fit):
        y_err = bootstrap.bootstrap_error(Yfit[0:-1, i], Yfit[-1, i])
        y_up[i] = Yfit[-1, i] + y_err
        y_dn[i] = Yfit[-1, i] - y_err

    ax.plot(x2**2, Yfit[-1], "--", linewidth=0.75, alpha=0.6)
    ax.fill_between(
        x2**2, y_up, y_dn, alpha=0.4, label=ch
    )  # color=plt.gca().lines[-1].get_color()


def plot_line(ax, lb, off_set, y_offset, a, err_a):
    y_up = np.sqrt(a + err_a)
    y_dn = np.sqrt(a - err_a)
    y = np.sqrt(a)

    ax.fill_between(
        [0, 1], y_up, y_dn, alpha=0.4, color="b"
    )  # color=plt.gca().lines[-1].get_color(),  label=ch_lb(ch)
    ax.text(
        0.5 + off_set,
        y + y_offset,
        lb,
        horizontalalignment="center",
        verticalalignment="center",
    )


def plot_line_as(ax, lb, off_set, y_offset, a, err_a):
    y_up = np.sqrt(a + err_a)
    y_dn = np.sqrt(a - err_a)
    y = np.sqrt(a)

    ax.fill_between(
        [1.1, 2.1], y_up, y_dn, alpha=0.4, color="r"
    )  # color=plt.gca().lines[-1].get_color(),  label=ch_lb(ch)
    ax.text(
        1.6 + off_set,
        y + y_offset,
        lb,
        horizontalalignment="center",
        verticalalignment="center",
    )


def plot_line_cb(ax, lb, off_set, y_offset, a, err_a):
    y_up = a + err_a
    y_dn = a - err_a
    y = a

    ax.fill_between(
        [2.2, 3.2], y_up, y_dn, alpha=0.4, color="k"
    )  # color=plt.gca().lines[-1].get_color(),  label=ch_lb(ch)
    ax.text(
        2.7 + off_set,
        y + y_offset,
        lb,
        horizontalalignment="center",
        verticalalignment="bottom",
    )


def plot_line_gb(ax, lb, off_set, y_offset, a, err_a, l_end, r_end):
    y_up = a + err_a
    y_dn = a - err_a
    y = a

    ax.fill_between(
        [l_end, r_end], y_up, y_dn, alpha=0.4, color="y"
    )  # color=plt.gca().lines[-1].get_color(),  label=ch_lb(ch)
    ax.text(
        (l_end + r_end) / 2 + off_set,
        y + y_offset,
        lb,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
