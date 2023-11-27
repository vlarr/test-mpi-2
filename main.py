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
        reader = csv.reader(csv_file, delimiter=',', skipinitialspace=True)
        mx_r = np.array([[float(cell) for cell in row] for row in reader])

    log.info(f'read {len(mx_r)} rows')
    log.debug(f'mx_r =\n{mx_r}')
    return mx_r


def load_array_mass_from_csv(file_name):
    log.info(f'load array_mass from \"{file_name}\"')
    with open(file_name, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', skipinitialspace=True)
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


def calc_part_border(rank, rank_size, data_size):
    part_i_min = int((rank * data_size) / rank_size)
    part_i_max = int(((rank + 1) * data_size) / rank_size)
    part_size = part_i_max - part_i_min
    return part_i_min, part_i_max, part_size


def fill_part_mx_f(rank, rank_size, mx_r, arr_m):
    data_size = len(mx_r)
    part_i_min, part_i_max, part_size = calc_part_border(rank, rank_size, data_size)
    log.info(f'fill part: [{part_i_min}, {part_i_max}], part_size = {part_size}, data_size = {data_size}')

    part_mx_f = np.array([[0.0, 0.0, 0.0]] * part_size)
    for i in range(part_size):
        part_mx_f[i] = calc_Fv_sum(part_i_min + i, mx_r, arr_m)
    log.debug(f'part_mx_f =\n{part_mx_f}')

    return part_mx_f


def gather_parts(comm, rank, rank_size, mx_r_size, mx_f):
    if rank == 0:
        #   Количество точек для расчётов может отличаться в разных потоках, но не более чем на 1.
        #   Поэтому подготавливаем буфер с запасом
        recv_buff_max_size = int((mx_r_size / rank_size) + 1)
        recv_buff = np.zeros((rank_size, recv_buff_max_size, 3))
    else:
        recv_buff = None

    log.info('start gather parts')
    comm.Gather(mx_f, recv_buff, root=0)

    if rank == 0:
        log.debug(f'recv_buff =\n{recv_buff}')
        result = np.zeros((0, 3))
        for i in range(rank_size):
            #   Нам нужен актуальный размер части данных. Обрезаем конец матрицы (заполненный мусором)
            i_min, i_max, part_size = calc_part_border(i, rank_size, mx_r_size)
            log.info(
                f'gather for rank = {i}, part = [{i_min}, {i_max}], part_size = {part_size}, data_size = {mx_r_size}')
            mx_f_part = recv_buff[i][:part_size]
            result = np.concatenate((result, mx_f_part), axis=0)
        return result
    else:
        return None


def save_result(output_file_name, mx_f):
    log.info(f'save results to: \"{output_file_name}\"')
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, 'wb') as file:
        numpy.savetxt(file, mx_f, delimiter=',\t')


def process(comm, rank, rank_size, mx_r_file_name, arr_m_file_name, output_file_name):
    mx_r, arr_m = read_or_receive_data(comm, rank, mx_r_file_name, arr_m_file_name)
    part_mx_f = fill_part_mx_f(rank, rank_size, mx_r, arr_m)
    result_mx_f = gather_parts(comm, rank, rank_size, len(mx_r), part_mx_f)
    if rank == 0:
        log.debug(f'result_mx_f =\n{result_mx_f}')
        save_result(output_file_name, result_mx_f)


if __name__ == '__main__':
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_size = comm.Get_size()

    configure_log(args, rank)

    if not os.path.isfile(args.source_r):
        log.warning(f"Файл по пути \"{args.source_r}\" не найден")
        exit(1)
    if not os.path.isfile(args.source_m):
        log.warning(f"Файл по пути \"{args.source_m}\" не найден")
        exit(1)

    process(comm, rank, rank_size, args.source_r, args.source_m, args.output)
