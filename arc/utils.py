from contextlib import contextmanager
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


from arc.load_data import Grid

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


def plot_grid(g: Grid):
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(g, cmap=cmap, norm=norm)
    ax.axis("off")
    plt.show()