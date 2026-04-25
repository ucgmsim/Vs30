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
def single_threaded_blas():
    """Restrict BLAS to single-threaded operation to prevent oversubscription during multiprocessing."""
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
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
