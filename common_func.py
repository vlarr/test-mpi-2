import logging

import numpy as np

F_coef_default = 10.0

log = logging.getLogger()


def pow2_dist(v_i, v_j):
    return (v_j[0] - v_i[0]) ** 2 + (v_j[1] - v_i[1]) ** 2 + (v_j[2] - v_i[2]) ** 2


def calc_rv(v_i, v_j):
    dist = pow2_dist(v_i, v_j) ** 0.5
    return np.array([(v_j[axis] - v_i[axis]) / dist for axis in [0, 1, 2]])


def calc_Fs_ij(i, j, mx_r, arr_m, f_coef=F_coef_default):
    return (f_coef * arr_m[i] * arr_m[j]) / pow2_dist(mx_r[i], mx_r[j])


def calc_Fv_ij(i, j, mx_r, arr_m, f_coef=F_coef_default):
    f_scalar = calc_Fs_ij(i, j, mx_r, arr_m, f_coef)
    r_vect = calc_rv(mx_r[i], mx_r[j])
    return np.multiply(f_scalar, r_vect)


def calc_Fv_sum(i, mx_r, arr_m, f_coef=F_coef_default):
    log.debug(f'call calc_Fv_sum: {i}, {mx_r}, {arr_m}, {f_coef}')
    arr_F_ij = np.zeros(3, float)
    for j in range(len(mx_r)):
        if j != i:
            arr_F_ij = np.add(arr_F_ij, calc_Fv_ij(i, j, mx_r, arr_m, f_coef=f_coef))
    return arr_F_ij
