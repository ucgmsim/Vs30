"""Lightweight multiprocessing primitives shared by parallel.py and spatial.py.

Kept in its own module so spatial.py can use ``spawn_context`` without
introducing a circular import via parallel.py.
"""

import contextlib
import multiprocessing as mp

import threadpoolctl

# Use spawn context to avoid GDAL fork issues. GDAL is not fork-safe; using
# spawn starts fresh processes without inheriting the parent's GDAL state,
# which prevents deadlocks.
spawn_context = mp.get_context("spawn")


@contextlib.contextmanager
def limit_blas_threads(threads: int = 1):
    """Restrict BLAS to ``threads`` threads to prevent oversubscription during multiprocessing.

    When the caller spawns N worker processes on an M-core machine, set
    ``threads = max(1, M // N)`` so the workers collectively saturate the
    CPU without oversubscription. Default ``threads=1`` matches the prior
    ``single_threaded_blas`` behaviour for callers that already pin BLAS
    explicitly elsewhere.
    """
    with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
        yield


def resolve_nproc(nproc: int | None) -> int:
    """
    Convert user input to actual process count.

    Parameters
    ----------
    nproc : int or None
        User-specified number of processes.
        None or 1 = single-threaded
        -1 = use all available CPU cores
        > 1 = use that many processes

    Returns
    -------
    int
        Actual number of processes to use (always >= 1)

    Raises
    ------
    ValueError
        If nproc is 0 or less than -1
    """
    if nproc is None or nproc == 1:
        return 1
    if nproc == -1:
        return mp.cpu_count()
    if nproc < -1 or nproc == 0:
        raise ValueError(f"nproc must be -1, 1, or > 1, got {nproc}")
    return min(nproc, mp.cpu_count())
