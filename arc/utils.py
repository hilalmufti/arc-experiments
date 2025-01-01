from contextlib import contextmanager
import random
import time

import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def dedup(xs):
    seen = set()
    ys = []
    for x in xs:
        if x not in seen:
            ys.append(x)
            seen.add(x)
    return ys

@contextmanager
def timer(desc=""):
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    print(f"{desc} {toc - tic:.2f}s")
