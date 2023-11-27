import csv
import logging
import os.path

import numpy as np
from mpi4py import MPI
from common_func import calc_Fv_sum

log = logging.getLogger(__name__)


def configure_log(rank):
    log.setLevel(logging.DEBUG)
    default_formatter = logging.Formatter(f"%(asctime)s\t%(levelname)s\trank={rank}, %(message)s")

    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f'logs/main-{rank}.log', mode='w')
    file_handler.setFormatter(default_formatter)
    log.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_formatter)
    log.addHandler(console_handler)


def load_matrix_r(file_name):
    log.info(f'call load_matrix_r: \"{file_name}\"')
    with open(file_name, newline='') as data_csv:
        reader = csv.reader(data_csv, delimiter=' ', skipinitialspace=True)
        mx_r = np.array([[float(cell) for cell in row] for row in reader])

    log.debug(f'mx_r=\n{mx_r}')
    return mx_r


def load_array_mass(file_name):
    log.info(f'call load_array_mass: \"{file_name}\"')
    with open(file_name, newline='') as data_csv:
        reader = csv.reader(data_csv, delimiter=' ', skipinitialspace=True)
        arr_m = np.array([float(row[0]) for row in reader])

    log.debug(f'arr_m={arr_m}')
    return arr_m


def process():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    configure_log(rank)

    if rank == 0:
        mx_r = load_matrix_r('data/example/mx_r-data.csv')
    else:
        mx_r = None
    mx_r = comm.bcast(mx_r, root=0)

    if rank == 0:
        arr_m = load_array_mass('data/example/arr_m-data.csv')
    else:
        arr_m = None
    arr_m = comm.bcast(arr_m, root=0)

    data_size = len(mx_r)
    i_min = int((rank * data_size) / size)
    i_max = int(((rank + 1) * data_size) / size)

    log.debug(f'[i_min, i_max]=[{i_min}, {i_max}]')

    mx_f = np.array([[0.0, 0.0, 0.0]] * data_size)
    for i in range(i_min, i_max):
        mx_f[i] = calc_Fv_sum(i, mx_r, arr_m)

    log.debug(f'mx_F=\n{mx_f}')


if __name__ == '__main__':
    process()
