import typing as tp
import numpy as np
import time
import numba
import autocompiler


def split_arr(arr: np.array) -> tp.Tuple[np.array, np.array]:
    """Split a array with gaps into start and end indices."""
    if len(arr) == 0:
        raise ValueError("Range is empty")
    start = np.empty(len(arr), dtype=np.int_)
    stop = np.empty(len(arr), dtype=np.int_)
    start[0] = 0
    k = 0
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] != 1:
            stop[k] = i
            k += 1
            start[k] = i
    stop[k] = len(arr)
    return start[:k + 1], stop[:k + 1]


@numba.jit
def numba_split_arr(arr: np.array) -> tp.Tuple[np.array, np.array]:
    """Split a array with gaps into start and end indices."""
    if len(arr) == 0:
        raise ValueError("Range is empty")
    start = np.empty(len(arr), dtype=np.int_)
    stop = np.empty(len(arr), dtype=np.int_)
    start[0] = 0
    k = 0
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] != 1:
            stop[k] = i
            k += 1
            start[k] = i
    stop[k] = len(arr)
    return start[:k + 1], stop[:k + 1]


@autocompiler.jit(cache=True)
def autocompiler_split_arr(arr: np.array) -> tp.Tuple[np.array, np.array]:
    """Split a array with gaps into start and end indices."""
    if len(arr) == 0:
        raise ValueError("Range is empty")
    start = np.empty(len(arr), dtype=np.int_)
    stop = np.empty(len(arr), dtype=np.int_)
    start[0] = 0
    k = 0
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] != 1:
            stop[k] = i
            k += 1
            start[k] = i
    stop[k] = len(arr)
    return start[:k + 1], stop[:k + 1]

if __name__ == '__main__':
    
    arr = np.random.normal(size=(5000000, 1))

    start_time = time.time()
    split_arr(arr)
    print("--- %s seconds ---" % (time.time() - start_time))
    # --- 12.096832990646362 seconds ---

    start_time = time.time()
    numba_split_arr(arr)
    print("--- %s seconds ---" % (time.time() - start_time))
    # --- 0.6376290321350098 seconds ---

    start_time = time.time()
    autocompiler_split_arr(arr)
    print("--- %s seconds ---" % (time.time() - start_time))
    # --- 0.6362500190734863 seconds ---
