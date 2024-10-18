import jax.flatten_util as fu
import jax.numpy as np
import numpy
import numpy.random as random

import matplotlib.pyplot as plt


def gaussian_beam(x, y, k, amplitude, mu, sigma):
    """
    Simple Gaussian beam in 2D.
    :param x: x-coordinate.
    :param y: y-coordinate.
    :param amplitude: Amplitude of the Gaussian. There is no normalisation taken into account!
    :param mu: Centre/mean of the beam.
    :param sigma: Standard deviation of the beam.
    :param k: Wave vector of the beam.
    :return: Value of the Gaussian at position [x, y].
    """
    real_part = np.exp(- 0.5 / sigma**2 * ((x-mu[0])**2 + (y-mu[1])**2))
    imaginary_part = np.exp(1j * (k[0] * (x-mu[0]) + k[1] * (y-mu[1])))
    return amplitude * real_part * imaginary_part


def time_gaussian_function(extent, size_x, size_y, k, amplitude, mu, sigma, omega):
    x_range = np.linspace(*extent[0], size_x)
    y_range = np.linspace(*extent[1], size_y)
    xx, yy = np.meshgrid(x_range, y_range)
    real_part = np.exp(- 0.5 / sigma**2 * ((xx-mu[0])**2 + (yy-mu[1])**2))
    imaginary_part = np.exp(1j * (k[0] * (xx-mu[0]) + k[1] * (yy-mu[1])))

    return amplitude * real_part * imaginary_part


def super_gaussian_function(extent, size_x, size_y, amplitude, mu, sigma, order):
    x_range = np.linspace(*extent[0], size_x)
    y_range = np.linspace(*extent[1], size_y)
    xx, yy = np.meshgrid(x_range, y_range)

    real_part = np.exp(
        - (0.5 * sigma ** 2 * (xx - mu[0]) ** 2) ** order - (0.5 * sigma ** 2 * (yy - mu[1]) ** 2) ** order)

    return amplitude * real_part


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) - 0.5


def zero_function(x, y):
    return np.zeros(x.shape)


def random_noise(x, y):
    grid = random.random(x.shape)
    left = int(0.4 * x.shape[0])
    right = int(0.6 * x.shape[0])
    top = int(0.4 * x.shape[1])
    bottom = int(0.6 * x.shape[1])
    grid_cut = np.array(grid[left:right, top:bottom])
    grid_cut = np.pad(grid_cut, ((left, left+1), (top, top+1)))
    return grid_cut



def _l2_norm(x):
    return np.abs(np.mean(x))


def mask_square(psi, wy, wx):
    mask = numpy.zeros(psi.shape, dtype='?')
    cy = psi.shape[0] // 2
    cx = psi.shape[1] // 2

    mask[int(cy - wy // 2): int(cy + wy // 2), int(cx - wx // 2): int(cx + wx // 2)] = 1
    return np.array(mask)


def mask_circle(psi, r):
    mask = numpy.zeros(psi.shape, dtype='?')
    cx = psi.shape[0] // 2
    cy = psi.shape[1] // 2

    for x in range(mask.shape[0]):
        for y in range(mask.shape[0]):
            if (x-cx)**2 + (y-cy)**2 <= r**2:
                mask[x, y] = 1
    return np.array(mask)


def absorbing_layer(d_x, d_y, size, strength):
    pml_xx = numpy.zeros(size)
    pml_yy = numpy.zeros(size)
    layer_x = numpy.linspace(0, d_x-1, d_x)
    layer_y = numpy.linspace(0, d_y-1, d_y)
    pml_xx[:, :d_x] = strength * np.flip(layer_x[numpy.newaxis, :])**2
    pml_xx[:, -d_x:] = strength * layer_x[numpy.newaxis, :]**2
    pml_yy[:d_y] = strength * numpy.flip(layer_y[:, numpy.newaxis])**2
    pml_yy[-d_y:] = strength * layer_y[:, numpy.newaxis]**2
    return np.array([pml_yy, pml_xx])