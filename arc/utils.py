from contextlib import contextmanager
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


from arc.load_data import GridM

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


def plot_grid(g: GridM):
    cmap = ListedColormap([
        '#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ])
    norm = Normalize(vmin=0, vmax=9)
    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(g, cmap=cmap, norm=norm)
    ax.axis("off")
    plt.show()


def print_grid(g):
    colors = {
        0: '\033[40m  \033[0m',         # Black (#000)
        1: '\033[38;5;32m  \033[0m',    # Blue (#0074D9)
        2: '\033[38;5;196m  \033[0m',   # Red (#FF4136)
        3: '\033[38;5;46m  \033[0m',    # Green (#2ECC40)
        4: '\033[38;5;226m  \033[0m',   # Yellow (#FFDC00)
        5: '\033[38;5;248m  \033[0m',   # Gray (#AAAAAA)
        6: '\033[38;5;200m  \033[0m',   # Pink (#F012BE)
        7: '\033[38;5;208m  \033[0m',   # Orange (#FF851B)
        8: '\033[38;5;117m  \033[0m',   # Light Blue (#7FDBFF)
        9: '\033[38;5;88m  \033[0m'     # Dark Red (#870C25)
    }
    colors = {k: color.replace('38', '48') for k, color in colors.items()}
    
    for row in g:
        for cell in row:
            print(colors.get(cell, '  '), end='')
        print()