#!/usr/bin/env python
"""
Dask distributed joblib backend on kubernetes benchmark script
"""

import argparse
import sys
import logging
import time
import subprocess
import socket

import joblib
import numpy

import distributed.joblib  # for side effects

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
    "--joblib-num-jobs",
    type=int,
    default=1,
    help="Set to -1 to use as many jobs as cores")

parser.add_argument(
    "--joblib-pre-dispatch",
    default='2*n_jobs',
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


def go(args, cores, out_fds):
    for replica in range(args.replicas):
        tasks = [
            joblib.delayed(task)(
                make_data(args.task_input_mb),
                args.task_time_sec,
                args.task_allocate_mb,
                args.task_output_mb)
            for _ in range(args.tasks)
        ]
        start = time.time()
        results = joblib.Parallel(
            n_jobs=args.joblib_num_jobs,
            verbose=1 if not args.quiet else 0,
            pre_dispatch=args.joblib_pre_dispatch)(tasks)
        length = time.time() - start

        assert len(results) == args.tasks

        #logging.info("Hosts: %s" % " ".join(set([x[0] for x in results])))
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
    if args.dask_scheduler:
        backend = joblib.parallel_backend(
            'distributed',
            scheduler_host=args.dask_scheduler)
        with backend:
            active_backend = joblib.parallel.get_active_backend()[0]
            logging.info(
                "Running with dask scheduler: %s [%d cores]" % (
                    args.dask_scheduler,
                    active_backend.effective_n_jobs()))

            if args.jobs_range is not None:
                for i in range(*args.jobs_range):
                    command = args.scale_command % i
                    logging.info("Running: %s" % command)
                    subprocess.check_call(command, shell=True)
                    while True:
                        cores = active_backend.effective_n_jobs(n_jobs=args.joblib_num_jobs)
                        logging.info("Cores: %d. Waiting for %d cores." % (cores, i))
                        if cores == i:
                            break
                        time.sleep(1)
                    go(args, cores, out_fds)
            else:
                cores = active_backend.effective_n_jobs(n_jobs=args.joblib_num_jobs)
                go(args, cores, out_fds)

    else:
        active_backend = joblib.parallel.get_active_backend()[0]
        cores = active_backend.effective_n_jobs(n_jobs=args.joblib_num_jobs)
        logging.info(
                "Running with joblib scheduler [%d cores]" % cores)
        go(args, cores, out_fds)