import matplotlib.pyplot as plt
import numpy as np


def plot(grid, extent):
    xmin = extent[0, 0]
    xmax = extent[0, 1]
    ymin = extent[1, 0]
    ymax = extent[1, 1]
    plt.imshow(grid, origin='lower', extent=(xmin, xmax, ymin, ymax), cmap='RdBu', vmin=-np.max(np.abs(grid)), vmax=np.max(np.abs(grid)))
    plt.colorbar()


def plot_show(grid, extent):
    plot(grid, extent)
    plt.show()


def plot_save(grid, extent, path):
    plot(grid, extent)
    plt.savefig(path)
    plt.close()