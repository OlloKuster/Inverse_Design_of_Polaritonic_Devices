import jax.numpy as np


def initialise(extent, size_x, size_y, function, function_parameters):
    """
    Initialise the simulation by adding a starting function to it.
    :param extent: extent of the cavity. Shape [2x2], first dimension represents the x-coordinate, second dimenstion
                   represents the y-coordinate.
    :param resolution: Resolution of the cavity.
    :param function: The function used to initialise the simulation.
    :param function_parameters: Parameters of function.
    :return: The absolute value of the starting cavity.
    """
    x_range = np.linspace(*extent[0], size_x)
    y_range = np.linspace(*extent[1], size_y)
    xx, yy = np.meshgrid(x_range, y_range)
    psi_0 = function(xx, yy, *function_parameters)
    return psi_0


def create_cavity(extent: np.ndarray, resolution: int):
    """
    Creates a coordinate system which represents the cavity.
    :param extent: extent of the cavity. Shape [2x2], first dimension represents the x-coordinate, second dimenstion
                   represents the y-coordinate.
    :param resolution: Resolution of the cavity.
    :return: x and y coordinates of the system.
    """
    x_space = np.linspace(*extent[0], resolution)
    y_space = np.linspace(*extent[1], resolution)
    xx, yy = np.meshgrid(x_space, y_space)
    return xx, yy


