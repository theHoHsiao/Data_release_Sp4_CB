import numpy as np
import random

### hard coded bootstrap parameters
num_confs = 200

bin_size = 1
num_sample = 800
seed = 37831

######


def random_index():
    random.seed(seed)
    random_index = np.zeros(shape=(num_sample, int(num_confs / bin_size)), dtype=int)
    for i in range(num_sample):
        for j in range(int(num_confs / bin_size)):
            random_index[i, j] = random.randint(0, int(num_confs / bin_size) - 1)
    return random_index


index_list = random_index()
np.random.seed(random.randint(0, 2**32 - 1))

def bootstrap_main(C):
    binS = []
    num_confs = len(C[:, 0])
    num_bin = int(num_confs / bin_size)

    for i in range(num_bin):
        b = C[bin_size * i : bin_size * (i + 1), :]
        binS.append(np.mean(b, axis=0))

    binS = np.array(binS)

    bootstrap_set = []
    for i in range(num_sample):
        sample = []
        for j in range(num_bin):
            sample.append(binS[index_list[i, j], :])

        bootstrap_set.append(np.mean(sample, axis=0))

    return np.array(bootstrap_set)


def bootstrap_error(A, centval):
    B = np.copy(A)
    B.sort()
    index_err = int(len(B) * 0.18)

    ERROR = max(abs(B[-index_err] - centval), abs(centval - B[index_err - 1]))

    return ERROR


def fold_correlators(C):
    size = np.shape(C)
    C_tmp = np.zeros(shape=(size[0], size[1] + 1))
    C_tmp[:, 0 : size[1]] = C
    C_tmp[:, size[1]] = C[:, 0]

    return (C_tmp + np.flip(C_tmp, axis=1)) / 2


def fold_correlators_cross(C):
    size = np.shape(C)
    C_tmp = np.zeros(shape=(size[0], size[1] + 1))
    C_tmp[:, 1 : size[1]] = (C[:, 1:] - np.flip(C[:, 1:], axis=1)) / 2
    C_tmp[:, size[1]] = -C[:, 0]
    C_tmp[:, 0] = C[:, 0]

    return C_tmp


def bin_proj_Baryon(ENS, ch, sour_N, sink_N, SSs, SL):
    def flip_boundary(C, t):
        if t == 2:
            return C
        else:
            C[:, -(t - 2) :] = -C[:, -(t - 2) :]
            return C

    C_Ebin = np.zeros(shape=(ENS.num_sample + 1, ENS.GLB_T + 1))
    C_Obin = np.zeros(shape=(ENS.num_sample + 1, ENS.GLB_T + 1))

    for i in range(len(SSs)):
        C_e = boot_correlator(ENS, ch + "_even_re", sour_N, sink_N, SSs[i])
        C_e = flip_boundary(C_e, SL[i])
        C_o = boot_correlator(ENS, ch + "_odd_re", sour_N, sink_N, SSs[i])
        C_o = flip_boundary(C_o, SL[i])

        size = np.shape(C_e)

        C_etmp = np.zeros(shape=(size[0], size[1] + 1))
        C_etmp[:, 0 : size[1]] = C_e
        C_etmp[:, size[1]] = C_e[:, 0]

        C_otmp = np.zeros(shape=(size[0], size[1] + 1))
        C_otmp[:, 0 : size[1]] = C_o
        C_otmp[:, size[1]] = C_o[:, 0]

        C_Ebin += (C_etmp - np.flip(C_otmp, axis=1)) / 2
        C_Obin += (C_otmp - np.flip(C_etmp, axis=1)) / 2

    C_Ebin = C_Ebin / len(SSs)
    C_Obin = -C_Obin / len(SSs)

    return C_Ebin, C_Obin


def gauB_DIS(cen, err, size):
    atmp = np.random.normal(cen, scale=err, size=size)
    return np.append(atmp, cen)


def Correlator_resample(C_ALL):
    C_ALL_resample = np.zeros(shape=(num_sample + 1, np.shape(C_ALL)[1]))
    C_ALL_resample[0:num_sample, :] = bootstrap_main(C_ALL)
    C_ALL_resample[num_sample, :] = np.mean(C_ALL[:, :], axis=0)

    return C_ALL_resample


b2w_QB1 = gauB_DIS(1.448, 0.003, num_sample)
b2w_QB2 = gauB_DIS(1.607, 0.0019, num_sample)
b2w_QB3 = gauB_DIS(1.944, 0.003, num_sample)
b2w_QB4 = gauB_DIS(2.3149, 0.0012, num_sample)
b2w_QB5 = gauB_DIS(2.8812, 0.0021, num_sample)
