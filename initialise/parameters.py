import math
from dataclasses import dataclass
import jax.numpy as np


@dataclass
class Cavity:
    resolution: int
    extent: np.ndarray
    maxtime: float
    timesteps: int

    @property
    def cavsize_x(self):
        return self.extent[0, 1] - self.extent[0, 0]

    @property
    def cavsize_y(self):
        return self.extent[1, 1] - self.extent[1, 0]

    @property
    def size_x(self):
        return self.cavsize_x * self.resolution + 1

    @property
    def size_y(self):
        return self.cavsize_y * self.resolution + 1

    @property
    def dx(self):
        return self.cavsize_x / (self.size_x - 1)

    @property
    def dy(self):
        return self.cavsize_y / (self.size_y - 1)

    @property
    def ts(self):
        return np.linspace(0, self.maxtime, self.timesteps)


@dataclass
class PhysicalParameters:
    hbar: float = 0.6582  # ps * mev
    mc: float = 0.6585  # meV * ps^2 / mum^2
    v: float = 0.
    u: float = 5. * 1e-6  # meV*mum^2
    kappa: float = 4.  # meV
    gamma: float = 0.


@dataclass
class GaussianParameters:
    k: np.ndarray
    amplitude: float
    mu: np.ndarray
    sigma: float


@dataclass
class FlatTop:
    size_x: int
    size_y: int
    percentage: float

    @property
    def right(self):
        return int(self.size_x / 2 + self.size_x * self.percentage)

    @property
    def left(self):
        return int(self.size_x / 2 - self.size_x * self.percentage)

    @property
    def top(self):
        return int(self.size_y / 2 + self.size_y * self.percentage)

    @property
    def bottom(self):
        return int(self.size_y / 2 - self.size_y * self.percentage)


@dataclass
class LocalMax:
    size_x: int
    size_y: int

    @property
    def centre(self):
        return np.array([int(self.size_x / 2), int(self.size_y / 2)])


@dataclass
class PhaseVortex:
    l: int

    @property
    def ang_momentum(self):
        return np.array([self.l, math.factorial((np.abs(self.l)))])
