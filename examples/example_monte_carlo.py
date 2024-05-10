import random
import time
import numba
import autocompiler


def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


@numba.jit
def numba_monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


@autocompiler.jit(cache=True)
def autocompiler_monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


if __name__ == '__main__':

    nsamples = 10000000

    start_time = time.time()
    monte_carlo_pi(nsamples)
    print("--- %s seconds ---" % (time.time() - start_time))
    # --- 2.1978321075439453 seconds ---

    start_time = time.time()
    numba_monte_carlo_pi(nsamples)
    print("--- %s seconds ---" % (time.time() - start_time))
    # --- 0.13260507583618164 seconds ---

    start_time = time.time()
    autocompiler_monte_carlo_pi(nsamples)
    print("--- %s seconds ---" % (time.time() - start_time))
    # --- 0.13222098350524902 seconds ---
