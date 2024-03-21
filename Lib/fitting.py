import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize

from bootstrap import bootstrap_error

######## print ########


def find_non_zero(x):
    if x < 1:
        A = "%e" % x
        return int(A.partition("-")[2]) - 1
    else:
        print("error > 1")
        return 0


def print_non_zero(v, err):
    dig = find_non_zero(err) + 2
    buff = 0

    if dig == 0:
        if v < 0:
            buff = 1
        return str(v)[0 : dig + 3 + buff] + "(" + str(err)[dig : dig + 3] + ")"
    else:
        if v < 0:
            buff = 1
        return str(v)[0 : dig + 2 + buff] + "(" + str(err)[dig : dig + 2] + ")"


######## mass extraction with exp. function ########


def X2_single_state_fit(C, ti, tf):
    num_sample = np.shape(C)[0]
    T = np.shape(C)[1] - 1

    def func(t, a, M):
        return a * a * M * (np.exp(-M * t) + np.exp(-M * (T - t))) / 2

    def Cov(ti, tj):
        return np.mean((C[:, ti] - np.mean(C[:, ti])) * (C[:, tj] - np.mean(C[:, tj])))

    size = tf - ti
    M = np.mat(np.zeros(shape=(size, size)))
    for a in np.arange(ti, tf, 1):
        for b in np.arange(ti, tf, 1):
            M[a - ti, b - ti] = Cov(a, b)

    M_I = M.I

    def X2_boot_const(X):
        def Cf_vector(ti, tf):
            return C[N, ti:tf] - func(np.arange(ti, tf), a, E)

        (a, E) = X

        V = np.mat(Cf_vector(ti, tf))

        return V * M_I * V.T

    def nor_X2(a, E1):
        def Cf_vector_nor(ti, tf):
            return C[N, ti:tf] - func(np.arange(ti, tf), a, E1)

        V = np.mat(Cf_vector_nor(ti, tf))

        return (V * M_I * V.T)[0, 0]

    E1 = []
    a1 = []

    t_slice = np.arange(ti, tf)

    x0, pcov = curve_fit(func, t_slice, C[-1, ti:tf])
    print("initial: ", x0)

    print("one-state fitting time region ", ti, "to", tf - 1, ": ", end="")

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-8)

        E1.append(res.x[1])
        a1.append(res.x[0])

    X2 = nor_X2(a1[-1], E1[-1])
    E_err = bootstrap_error(E1[0:-1], E1[-1])

    return np.array(E1), E_err, X2


def X2_single_exp_fit(C, ti, tf):
    num_sample = np.shape(C)[0]

    def func(t, a, M):
        return a * M * np.exp(-M * t)

    def Cov(ti, tj):
        return np.mean((C[:, ti] - np.mean(C[:, ti])) * (C[:, tj] - np.mean(C[:, tj])))
        # return np.mean( ( C[:, ti]  *  C[:, tj]) - np.mean(C[:, ti])*np.mean(C[:, tj])  )

    size = tf - ti
    M = np.mat(np.zeros(shape=(size, size)))
    for a in np.arange(ti, tf, 1):
        for b in np.arange(ti, tf, 1):
            M[a - ti, b - ti] = Cov(a, b)

    M_I = M.I

    def X2_boot_const(X):
        def Cf_vector(ti, tf):
            return C[N, ti:tf] - func(np.arange(ti, tf), a, E)

        (a, E) = X

        V = np.mat(Cf_vector(ti, tf))

        return V * M_I * V.T

    def nor_X2(a, E1):
        def Cf_vector_nor(ti, tf):
            return C[N, ti:tf] - func(np.arange(ti, tf), a, E1)

        V = np.mat(Cf_vector_nor(ti, tf))

        return (V * M_I * V.T)[0, 0]

    E1 = []
    a1 = []

    t_slice = np.arange(ti, tf)

    x0, pcov = curve_fit(func, t_slice, C[-1, ti:tf])
    print("x0=", x0)

    print("A*m*exp(-mt) fitting time region ", ti, "to", tf - 1, ": ", end="")

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-8)

        E1.append(res.x[1])
        a1.append(res.x[0])

    X2 = nor_X2(a1[-1], E1[-1])
    E_err = bootstrap_error(E1[0:-1], E1[-1])

    return np.array(E1), E_err, X2


############################ mass extrapolation with repect to single fermion mass #################


def cross_check_fit(X, Y):
    num_sample = np.shape(X)[1]
    num_pars = 3

    # print('m = a + bm^2 + cm^3 ',)

    def func(V, a, b, c):
        m = V
        return a + b * m**2 + c * m**3

    def Cov(i, j):
        return np.mean((Y[i, :] - np.mean(Y[i, :])) * (Y[j, :] - np.mean(Y[j, :])))

    x0, pcov = curve_fit(func, X[:, -1], Y[:, -1])
    size = np.shape(X)[0]

    M = np.mat(np.zeros(shape=(size, size)))
    """
    for a in range(size):
        for b in range(size):
            M[a , b] = Cov(a, b)
    """
    for a in range(size):
        M[a, a] = Cov(a, a)

    M_I = M.I

    def X2_boot_const(pars):
        def Cf_vector():
            return Y[:, N] - func(X[:, N], a, b, c)

        (a, b, c) = pars

        V = np.mat(Cf_vector())

        return V * M_I * V.T

    def nor_X2(a, b, c):
        def Cf_vector_nor():
            return Y[:, -1] - func(X[:, -1], a, b, c)

        V = np.mat(Cf_vector_nor())

        chisqr = (V * M_I * V.T)[0, 0]
        # print(r'Xsqr/d.o.f.='+ str( chisqr / (size-num_pars-1)) )
        return chisqr / (size - num_pars - 1)

    fit_val = np.zeros(shape=(num_pars, num_sample))

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-10)

        for n_pars in range(num_pars):
            fit_val[n_pars, N] = res.x[n_pars]

    X2 = nor_X2(res.x[0], res.x[1], res.x[2])

    return fit_val, X2


##################### baryon chPT fitting ######################


def baryon_M4(X1, X2, LAT_A, Y):
    num_sample = np.shape(X1)[0]
    num_pars = 11
    # print('baryon M4 fitting... M = a + b*mf**2 + c*mas**2 + d*mf**3 + e*mas**3 + f*lat_a + g*lat_a*mf**2 + h*lat_a*mas**2 + i*mf**4 + j*mas**4 + k* (mf**2) * (mas**2) ')

    def func(V, M_CB, F2, A2, L1, F3, A3, L2F, L2A, F4, A4, C4):
        mf, mas, lat_a = V
        return (
            M_CB
            + F2 * mf**2
            + A2 * mas**2
            + F3 * mf**3
            + A3 * mas**3
            + L1 * lat_a
            + L2F * lat_a * mf**2
            + L2A * lat_a * mas**2
            + F4 * mf**4
            + A4 * mas**4
            + C4 * (mf**2) * (mas**2)
        )

    def Cov(i, j):
        return np.mean((Y[i, :] - np.mean(Y[i, :])) * (Y[j, :] - np.mean(Y[j, :])))

    x0, pcov = curve_fit(func, (X1[:, -1], X2[:, -1], LAT_A[:, -1]), Y[:, -1])

    print("initial values:\n", x0)

    size = np.shape(X1)[0]

    M = np.mat(np.zeros(shape=(size, size)))

    for a in range(size):
        M[a, a] = Cov(a, a)

    M_I = M.I

    def X2_boot_const(pars):
        def Cf_vector():
            return Y[:, N] - func(
                (X1[:, N], X2[:, N], LAT_A[:, N]),
                M_CB,
                F2,
                A2,
                L1,
                F3,
                A3,
                L2F,
                L2A,
                F4,
                A4,
                C4,
            )

        (M_CB, F2, A2, L1, F3, A3, L2F, L2A, F4, A4, C4) = pars

        V = np.mat(Cf_vector())

        return V * M_I * V.T

    def nor_X2(M_CB, F2, A2, L1, F3, A3, L2F, L2A, F4, A4, C4):
        V = Y[:, -1] - func(
            (X1[:, -1], X2[:, -1], LAT_A[:, -1]),
            M_CB,
            F2,
            A2,
            L1,
            F3,
            A3,
            L2F,
            L2A,
            F4,
            A4,
            C4,
        )
        V = np.mat(V)

        chisqr = (V * M_I * V.T)[0, 0]
        # print(r'Xsqr/d.o.f.='+ str( chisqr / (size-num_pars-1)))
        return chisqr

    fit_val = np.zeros(shape=(num_pars, num_sample))

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-10)

        for n_pars in range(num_pars):
            fit_val[n_pars, N] = res.x[n_pars]

    X2 = nor_X2(
        res.x[0],
        res.x[1],
        res.x[2],
        res.x[3],
        res.x[4],
        res.x[5],
        res.x[6],
        res.x[7],
        res.x[8],
        res.x[9],
        res.x[10],
    )

    # print('Result:\n',fit_val[:,-1],'\n')

    return fit_val, X2


def baryon_MC4(X1, X2, LAT_A, Y):
    num_sample = np.shape(X1)[1]
    num_pars = 9

    # print('baryon MC4 fitting: M = a + b*mf**2 + c*mas**2 + d*mf**3 + e*mas**3 + f*lat_a + g*lat_a*mf**2 + h*lat_a*mas**2  + k* (mf**2) * (mas**2)')

    def func(V, M_CB, F2, A2, L1, F3, A3, L2F, L2A, C4):
        mf, mas, lat_a = V
        return (
            M_CB
            + F2 * mf**2
            + A2 * mas**2
            + F3 * mf**3
            + A3 * mas**3
            + L1 * lat_a
            + L2F * lat_a * mf**2
            + L2A * lat_a * mas**2
            + C4 * (mf**2) * (mas**2)
        )

    def Cov(i, j):
        return np.mean((Y[i, :] - np.mean(Y[i, :])) * (Y[j, :] - np.mean(Y[j, :])))
        # return np.mean( ( C[:, ti]  *  C[:, tj]) - np.mean(C[:, ti])*np.mean(C[:, tj])  )

    x0, pcov = curve_fit(func, (X1[:, -1], X2[:, -1], LAT_A[:, -1]), Y[:, -1])
    # print('initial values:\n', x0)

    size = np.shape(X1)[0]

    M = np.mat(np.zeros(shape=(size, size)))

    for a in range(size):
        M[a, a] = Cov(a, a)

    M_I = M.I

    def X2_boot_const(pars):
        (M_CB, F2, A2, L1, F3, A3, L2F, L2A, C4) = pars

        V = Y[:, N] - func(
            (X1[:, N], X2[:, N], LAT_A[:, N]),
            M_CB,
            F2,
            A2,
            L1,
            F3,
            A3,
            L2F,
            L2A,
            C4,
        )
        V = np.mat(V)

        return V * M_I * V.T

    def nor_X2(M_CB, F2, A2, L1, F3, A3, L2F, L2A, C4):
        V = Y[:, -1] - func(
            (X1[:, -1], X2[:, -1], LAT_A[:, -1]),
            M_CB,
            F2,
            A2,
            L1,
            F3,
            A3,
            L2F,
            L2A,
            C4,
        )
        V = np.mat(V)

        chisqr = (V * M_I * V.T)[0, 0]
        # print(r'Xsqr/d.o.f.='+ str( chisqr / (size-num_pars-1) ) )
        return chisqr

    fit_val = np.zeros(shape=(num_pars, num_sample))

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-10)

        for n_pars in range(num_pars):
            fit_val[n_pars, N] = res.x[n_pars]

    X2 = nor_X2(
        res.x[0],
        res.x[1],
        res.x[2],
        res.x[3],
        res.x[4],
        res.x[5],
        res.x[6],
        res.x[7],
        res.x[8],
    )
    # print('Result:\n',fit_val[:,-1],'\n')

    return fit_val, X2


def baryon_MA4(X1, X2, LAT_A, Y):
    num_sample = np.shape(X1)[1]
    num_pars = 9

    # print('baryon MA4 fitting... M = a + b*mf**2 + c*mas**2 + d*mf**3 + e*mas**3 + f*lat_a + g*lat_a*mf**2 + h*lat_a*mas**2  + j * (mas**4)')

    def func(V, M_CB, F2, A2, L1, F3, A3, L2F, L2A, A4):
        mf, mas, lat_a = V
        return (
            M_CB
            + F2 * mf**2
            + A2 * mas**2
            + F3 * mf**3
            + A3 * mas**3
            + L1 * lat_a
            + L2F * lat_a * mf**2
            + L2A * lat_a * mas**2
            + A4 * mas**4
        )

    def Cov(i, j):
        return np.mean((Y[i, :] - np.mean(Y[i, :])) * (Y[j, :] - np.mean(Y[j, :])))

    x0, pcov = curve_fit(func, (X1[:, -1], X2[:, -1], LAT_A[:, -1]), Y[:, -1])
    # print('initial values:\n', x0)

    size = np.shape(X1)[0]

    M = np.mat(np.zeros(shape=(size, size)))

    for a in range(size):
        M[a, a] = Cov(a, a)

    M_I = M.I

    def X2_boot_const(pars):
        (M_CB, F2, A2, L1, F3, A3, L2F, L2A, A4) = pars

        V = Y[:, N] - func(
            (X1[:, N], X2[:, N], LAT_A[:, N]),
            M_CB,
            F2,
            A2,
            L1,
            F3,
            A3,
            L2F,
            L2A,
            A4,
        )
        V = np.mat(V)

        return V * M_I * V.T

    def nor_X2(M_CB, F2, A2, L1, F3, A3, L2F, L2A, A4):
        V = Y[:, -1] - func(
            (X1[:, -1], X2[:, -1], LAT_A[:, -1]),
            M_CB,
            F2,
            A2,
            L1,
            F3,
            A3,
            L2F,
            L2A,
            A4,
        )
        V = np.mat(V)

        chisqr = (V * M_I * V.T)[0, 0]
        # print(r'Xsqr/d.o.f.='+ str( chisqr / (size-num_pars-1) ))
        return chisqr

    fit_val = np.zeros(shape=(num_pars, num_sample))

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-10)

        for n_pars in range(num_pars):
            fit_val[n_pars, N] = res.x[n_pars]

    X2 = nor_X2(
        res.x[0],
        res.x[1],
        res.x[2],
        res.x[3],
        res.x[4],
        res.x[5],
        res.x[6],
        res.x[7],
        res.x[8],
    )

    # print('Result:\n',fit_val[:,-1],'\n')

    return fit_val, X2


def baryon_MF4(X1, X2, LAT_A, Y):
    num_sample = np.shape(X1)[1]
    num_pars = 9

    # print('baryon MF4 fitting... M = a + b*mf**2 + c*mas**2 + d*mf**3 + e*mas**3 + f*lat_a + g*lat_a*mf**2 + h*lat_a*mas**2  + j * (mf**4)')

    def func(V, M_CB, F2, A2, L1, F3, A3, L2F, L2A, F4):
        mf, mas, lat_a = V
        return (
            M_CB
            + F2 * mf**2
            + A2 * mas**2
            + F3 * mf**3
            + A3 * mas**3
            + L1 * lat_a
            + L2F * lat_a * mf**2
            + L2A * lat_a * mas**2
            + F4 * mf**4
        )

    def Cov(i, j):
        return np.mean((Y[i, :] - np.mean(Y[i, :])) * (Y[j, :] - np.mean(Y[j, :])))

    x0, pcov = curve_fit(func, (X1[:, -1], X2[:, -1], LAT_A[:, -1]), Y[:, -1])
    # print('initial values:\n', x0)

    size = np.shape(X1)[0]

    M = np.mat(np.zeros(shape=(size, size)))
    for a in range(size):
        M[a, a] = Cov(a, a)

    M_I = M.I

    def X2_boot_const(pars):
        (M_CB, F2, A2, L1, F3, A3, L2F, L2A, F4) = pars

        V = Y[:, N] - func(
                (X1[:, N], X2[:, N], LAT_A[:, N]),
                M_CB,
                F2,
                A2,
                L1,
                F3,
                A3,
                L2F,
                L2A,
                F4,
            )
        V = np.mat(V)

        return V * M_I * V.T

    def nor_X2(M_CB, F2, A2, L1, F3, A3, L2F, L2A, F4):
        V = Y[:, -1] - func(
            (X1[:, -1], X2[:, -1], LAT_A[:, -1]),
            M_CB,
            F2,
            A2,
            L1,
            F3,
            A3,
            L2F,
            L2A,
            F4,
        )
        V = np.mat(V)

        chisqr = (V * M_I * V.T)[0, 0]
        # print((r'Xsqr/d.o.f.='+ str( chisqr / (size-num_pars-1) )))
        return chisqr

    fit_val = np.zeros(shape=(num_pars, num_sample))

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-10)

        for n_pars in range(num_pars):
            fit_val[n_pars, N] = res.x[n_pars]

    X2 = nor_X2(
        res.x[0],
        res.x[1],
        res.x[2],
        res.x[3],
        res.x[4],
        res.x[5],
        res.x[6],
        res.x[7],
        res.x[8],
    )
    # print('Result:\n',fit_val[:,-1],'\n')

    return fit_val, X2


def baryon_M3(X1, X2, LAT_A, Y):
    num_sample = np.shape(X1)[1]
    num_pars = 8

    # print('baryon M3 fitting... M = a + b*mf**2 + c*mas**2 + d*mf**3 + e*mas**3 + f*lat_a + g*lat_a*mf**2 + h*lat_a*mas**2 ')

    def func(V, M_CB, F2, A2, L1, F3, A3, L2F, L2A):
        mf, mas, lat_a = V
        return (
            M_CB
            + F2 * mf**2
            + A2 * mas**2
            + F3 * mf**3
            + A3 * mas**3
            + L1 * lat_a
            + L2F * lat_a * mf**2
            + L2A * lat_a * mas**2
        )

    def Cov(i, j):
        return np.mean((Y[i, :] - np.mean(Y[i, :])) * (Y[j, :] - np.mean(Y[j, :])))

    x0, pcov = curve_fit(func, (X1[:, -1], X2[:, -1], LAT_A[:, -1]), Y[:, -1])
    # print('initial values:\n', x0)

    size = np.shape(X1)[0]

    M = np.mat(np.zeros(shape=(size, size)))

    for a in range(size):
        M[a, a] = Cov(a, a)

    M_I = M.I

    def X2_boot_const(pars):
        (M_CB, F2, A2, L1, F3, A3, L2F, L2A) = pars

        V = Y[:, N] - func(
            (X1[:, N], X2[:, N], LAT_A[:, N]), M_CB, F2, A2, L1, F3, A3, L2F, L2A
        )
        V = np.mat(V)

        return V * M_I * V.T

    def nor_X2(M_CB, F2, A2, L1, F3, A3, L2F, L2A):
        V = Y[:, -1] - func(
            (X1[:, -1], X2[:, -1], LAT_A[:, -1]), M_CB, F2, A2, L1, F3, A3, L2F, L2A
        )
        V = np.mat(V)

        chisqr = (V * M_I * V.T)[0, 0]
        # print((r'Xsqr/d.o.f.='+ str( chisqr / (size-num_pars-1) )))
        return chisqr

    fit_val = np.zeros(shape=(num_pars, num_sample))

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-10)

        for n_pars in range(num_pars):
            fit_val[n_pars, N] = res.x[n_pars]

    X2 = nor_X2(
        res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6], res.x[7]
    )

    # print('Result:\n',fit_val[:,-1],'\n')

    return fit_val, X2


def baryon_M2(X1, X2, LAT_A, Y):
    num_sample = np.shape(X1)[1]
    num_pars = 4

    # print('baryon M2 fitting... M = a + b*mf**2 + c*mas**2 + d*lat_a ')

    def func(V, a, b, c, d):
        mf, mas, lat_a = V
        return a + b * mf**2 + c * mas**2 + d * lat_a

    def Cov(i, j):
        return np.mean((Y[i, :] - np.mean(Y[i, :])) * (Y[j, :] - np.mean(Y[j, :])))

    x0, pcov = curve_fit(func, (X1[:, -1], X2[:, -1], LAT_A[:, -1]), Y[:, -1])

    # print('initial values:\n', x0)

    size = np.shape(X1)[0]

    M = np.mat(np.zeros(shape=(size, size)))

    for a in range(size):
        M[a, a] = Cov(a, a)

    M_I = M.I

    def X2_boot_const(pars):
        def Cf_vector():
            V = Y[:, N] - func((X1[:, N], X2[:, N], LAT_A[:, N]), a, b, c, d)

            return V

        (a, b, c, d) = pars

        V = np.mat(Cf_vector())

        return V * M_I * V.T

    def nor_X2(a, b, c, d):
        V = Y[:, -1] - func((X1[:, -1], X2[:, -1], LAT_A[:, -1]), a, b, c, d)
        V = np.mat(V)

        chisqr = (V * M_I * V.T)[0, 0]
        # print((r'Xsqr/d.o.f.='+ str( chisqr / (size-num_pars-1) )))
        return chisqr

    fit_val = np.zeros(shape=(num_pars, num_sample))

    for N in range(num_sample):
        res = minimize(X2_boot_const, x0, method="Nelder-Mead", tol=10**-8)

        for n_pars in range(num_pars):
            fit_val[n_pars, N] = res.x[n_pars]

    X2 = nor_X2(res.x[0], res.x[1], res.x[2], res.x[3])
    # print('Result:\n',fit_val[:,-1],'\n')

    return fit_val, X2
