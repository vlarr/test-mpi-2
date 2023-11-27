import argparse
import csv
import datetime
import logging
import os.path

import numpy
import numpy as np
from mpi4py import MPI

from common_func import calc_Fv_sum

log = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='enable debug output', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--logs', help='logs output dir. Default disabled', default=None)
    parser.add_argument('source_r', help='cvs file with coordinates')
    parser.add_argument('source_m', help='csv file with masses')
    parser.add_argument('output', help='output file name')
    return parser.parse_args()


def configure_log(args, rank):
    if args.debug:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    default_formatter = logging.Formatter(f"%(asctime)s\t%(levelname)s\trank = {rank}, %(message)s")

    if args.logs:
        os.makedirs(args.logs, exist_ok=True)
        log_file_prefix = int((datetime.datetime.now() - datetime.datetime(1970, 1, 1)).total_seconds())
        logs_file_name = f'{args.logs}/{log_file_prefix}-main-{rank}.log'
        file_handler = logging.FileHandler(logs_file_name, mode='w')
        file_handler.setFormatter(default_formatter)
        log.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(default_formatter)
    log.addHandler(console_handler)


def load_matrix_r_from_csv(file_name):
    log.info(f'load matrix_r from \"{file_name}\"')
    with open(file_name, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)
        mx_r = np.array([[float(cell) for cell in row] for row in reader])

    log.info(f'read {len(mx_r)} rows')
    log.debug(f'mx_r =\n{mx_r}')
    return mx_r


def load_array_mass_from_csv(file_name):
    log.info(f'load array_mass from \"{file_name}\"')
    with open(file_name, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)
        arr_m = np.array([float(row[0]) for row in reader])

    log.info(f'read {len(arr_m)} rows')
    log.debug(f'arr_m = {arr_m}')
    return arr_m


def read_or_receive_data(comm, rank, mx_r_file_name, arr_m_file_name):
    if rank == 0:
        mx_r = load_matrix_r_from_csv(mx_r_file_name)
    else:
        mx_r = None
    mx_r = comm.bcast(mx_r, root=0)

    if rank == 0:
        arr_m = load_array_mass_from_csv(arr_m_file_name)
    else:
        arr_m = None
    arr_m = comm.bcast(arr_m, root=0)

    return mx_r, arr_m


def process_data_part(rank, size, mx_r, arr_m):
    data_size = len(mx_r)
    data_part_i_min = int((rank * data_size) / size)
    data_part_i_max = int(((rank + 1) * data_size) / size)
    log.info(f'process part: [{data_part_i_min}, {data_part_i_max}], data_size = {data_size}')

    mx_f = np.array([[0.0, 0.0, 0.0]] * data_size)
    for i in range(data_part_i_min, data_part_i_max):
        mx_f[i] = calc_Fv_sum(i, mx_r, arr_m)
    log.debug(f'mx_F =\n{mx_f}')

    return data_part_i_min, data_part_i_max, mx_f


def gather_results(comm, rank, size, mx_f):
    send_buff = mx_f
    recv_buff = None

    if rank == 0:
        recv_buff = np.zeros((size, len(mx_f), 3))

    comm.Gather(send_buff, recv_buff, root=0)
    log.info('completion gather parts')

    if rank == 0:
        log.debug(f'recv_buff =\n{recv_buff}')
        result = np.zeros((len(mx_f), 3))
        for i in range(size):
            result = np.add(result, recv_buff[i])

        return result
    else:
        return None


def save_result(output_file_name, result_mx_f):
    log.info(f'save results to: \"{output_file_name}\"')
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, 'wb') as file:
        numpy.savetxt(file, result_mx_f, delimiter='\t')


def process(comm, rank, size, mx_r_file_name, arr_m_file_name, output_file_name):
    mx_r, arr_m = read_or_receive_data(comm, rank, mx_r_file_name, arr_m_file_name)
    i_min, i_max, mx_f = process_data_part(rank, size, mx_r, arr_m)
    result_mx_f = gather_results(comm, rank, size, mx_f)

    if rank == 0:
        log.debug(f'result_mx_f =\n{result_mx_f}')
        save_result(output_file_name, result_mx_f)


if __name__ == '__main__':
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    configure_log(args, rank)

    if not os.path.isfile(args.source_r):
        log.warning(f"Файл по пути \"{args.source_r}\" не найден")
        exit(1)
    if not os.path.isfile(args.source_m):
        log.warning(f"Файл по пути \"{args.source_m}\" не найден")
        exit(1)

    process(comm, rank, size, args.source_r, args.source_m, args.output)
