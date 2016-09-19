#!/usr/bin/env python
"""
Dask distributed on kubernetes benchmark script
"""

import argparse
import sys
import logging
import time
import subprocess
import socket

import numpy

import dask
from distributed import Executor

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    "--tasks",
    type=int,
    default=1,
    help="")

parser.add_argument(
    "--task-time-sec",
    type=float,
    default=1.0,
    help="")

parser.add_argument(
    "--task-allocate-mb",
    type=float,
    default=0.0,
    help="")

parser.add_argument(
    "--task-input-mb",
    type=float,
    default=0.0,
    help="")

parser.add_argument(
    "--task-output-mb",
    type=float,
    default=0.0,
    help="")

parser.add_argument(
    "--dask-scheduler",
    metavar="HOST:PORT",
    help="Host and port of dask distributed scheduler")

parser.add_argument(
    "--jobs-range",
    type=int,
    nargs=3,
    default=None,
    help="")

parser.add_argument(
    "--replicas",
    type=int,
    default=1,
    help="")

parser.add_argument(
    "--scale-command",
    default="kubectl scale deployment daskd-worker --replicas=%d",
    help="")

parser.add_argument(
    "--quiet",
    action="store_true",
    default=False,
    help="")

parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="")

parser.add_argument(
    "--out",
    help="")


def make_data(size_mb):
    if not size_mb:
        return None
    return numpy.random.rand(int(size_mb * 2**20 / 8))


def task(task_data, task_time, task_allocate_mb, task_output_mb):
    allocated = make_data(task_allocate_mb)
    time.sleep(task_time)
    return (socket.gethostname(), make_data(task_output_mb))


def go(client, args, cores, out_fds):
    for replica in range(args.replicas):
        tasks = [
            dask.delayed(task)(
                make_data(args.task_input_mb),
                args.task_time_sec,
                args.task_allocate_mb,
                args.task_output_mb)
            for _ in range(args.tasks)
        ]
        start = time.time()
        results = client.compute(tasks, sync=True)
        print(results)
        length = time.time() - start

        assert len(results) == args.tasks

        logging.info("Hosts: %s" % len(set([x[0] for x in results])))

        for fd in out_fds:
            fd.write(", ".join([str(x) for x in [
                "RESULT_ROW",
                cores,
                replica,
                args.tasks,
                args.task_input_mb,
                args.task_time_sec,
                args.task_allocate_mb,
                args.task_output_mb,
                length
            ]]))
            fd.write("\n")
            fd.flush()


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    if not args.quiet:
        logging.basicConfig(level="INFO")
    if args.verbose:
        logging.basicConfig(level="DEBUG")
    out_fds = [sys.stdout]
    if args.out:
        out_fds.append(open(args.out, 'w'))

    client = None
    if args.dask_scheduler:
        client = Executor(args.dask_scheduler)
    else:
        client = Executor()

    print(dir(client))
    logging.info(
        "Running with dask scheduler: %s [%s cores]" % (
            args.dask_scheduler,
            sum(client.ncores().values())))

    if args.jobs_range is not None:
        for i in range(*args.jobs_range):
            command = args.scale_command % i
            logging.info("Running: %s" % command)
            subprocess.check_call(command, shell=True)
            while True:
                cores = sum(client.ncores().values())
                logging.info(
                    "Cores: %d. Waiting for %d cores." % (cores, i))
                if cores == i:
                    break
                time.sleep(1)
            go(client, args, cores, out_fds)
    else:
        cores = sum(client.ncores().values())
        go(client, args, cores, out_fds)
