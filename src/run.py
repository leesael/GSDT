import argparse
import multiprocessing
import os
import subprocess
from multiprocessing import Pool

import numpy as np


def parse_args():
    """
    Parse command line arguments for a parallel run.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(4)))
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_known_args()


def run_command(args):
    """
    Run a single command for executing main.py.
    """
    command, gpu_list = args
    gpu_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    gpu = gpu_list[gpu_idx % len(gpu_list)]
    command += ['--device', str(gpu)]
    subprocess.call(command)


def main():
    """
    Main function that trains GSDT in a parallel way.
    """
    args, unknown = parse_args()
    args_list = []
    for seed in args.seeds:
        for data in args.data:
            command = ['python', 'main.py',
                       '--data', data,
                       '--out', args.out,
                       '--seed', str(seed)]
            args_list.append((command + unknown, args.gpus))

    with Pool(len(args.gpus) * args.workers) as pool:
        pool.map(run_command, args_list)

    for data in args.data:
        out_path = '{}/{}'.format(args.out, data)
        values = []
        for seed in args.seeds:
            values.append(np.load(os.path.join(out_path, '{}.npy'.format(seed))))
        values = np.array(values)
        values_mean = values.mean(axis=0)
        values_std = values.std(axis=0)
        print('{}\t{}\t{}\t{}\t{}'.format(
            data, values_mean[0], values_std[0], values_mean[1], values_std[1]))


if __name__ == '__main__':
    main()
