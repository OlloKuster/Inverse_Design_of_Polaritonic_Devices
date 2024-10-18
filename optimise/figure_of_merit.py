import jax.debug
import jax.numpy as np

from solve.solver import solve_gp as solver
from initialise.functions import super_gaussian_function as sgaussian
from initialise.functions import mask_circle, mask_square


def max_function(psi_0, fom_args, ts, function_args):
    def local_max(variable):
        psi = np.array([psi_0.real, psi_0.imag])
        args_full = (*function_args, variable)
        sol = solver(psi, args_full, ts).ys[-1]
        psi = sol[0] ** 2 + sol[1] ** 2
        return - np.sum(psi[fom_args[0]:fom_args[1]]) - np.sum(psi[:, fom_args[0]:fom_args[1]])

    return local_max


def zero_grad_function(psi_0, fom_args, ts, function_args):
    def zero_grad(variable):
        psi = np.array([psi_0.real, psi_0.imag])
        args_full = (*function_args, variable)
        sol = solver(psi, args_full, ts).ys[-1]
        psi = sol[0] ** 2 + sol[1] ** 2
        xdiff = np.abs(np.diff(psi, axis=0))
        ydiff = np.abs(np.diff(psi, axis=1))
        return np.sum(xdiff[fom_args[0]:fom_args[1]]) + np.sum(ydiff[:, fom_args[0]:fom_args[1]])

    return zero_grad

def phase_vortex_function(psi_0, fom_args, ts, function_args):
    # mask = mask_circle(psi_0, psi_0.shape[0]*0.25)
    # mask = mask_square(psi_0, 2*psi_0.shape[0]//3, psi_0.shape[1]//2)
    l = fom_args[0]
    l_fac = fom_args[1]
    w_0 = fom_args[2]
    x_range = np.linspace(*fom_args[3], fom_args[5])
    y_range = np.linspace(*fom_args[4], fom_args[6])
    xx, yy = np.meshgrid(x_range, y_range)
    r = np.sqrt(xx ** 2 + yy ** 2)
    middle = xx.shape[0] // 2
    oam_beam = 1 / w_0 * np.sqrt(1 / (np.pi * np.abs(l_fac))) * (r * np.sqrt(2) / w_0) ** np.abs(l) \
               * np.exp(-r ** 2 / w_0 ** 2) * np.exp(1j * l * np.arctan2(xx, yy))

    def phase_vortex(variable):
        psi = np.array([psi_0.real, psi_0.imag])
        pot = variable
        args_full = (*function_args, pot)
        sol = solver(psi, args_full, ts).ys[-1]
        psi = sol[0] + 1j * sol[1]
        return -np.abs(np.sum(oam_beam * np.conj(psi))) ** 2

    return phase_vortex


def lens_focus(psi_0, fom_args, ts, function_args):
    mask = mask_square(psi_0, psi_0.shape[0], psi_0.shape[1] // 4)

    def lens_focus_function(variable):
        psi = np.array([psi_0.real, psi_0.imag])
        pot = mask * variable
        args_full = (*function_args, pot)
        sol = solver(psi, args_full, ts).ys[-1]
        psi = sol[0] + 1j * sol[1]
        middle = psi.shape[0] // 2
        return -np.abs(psi[middle, int(np.floor(psi.shape[1] * 0.8))])

    return lens_focus_function


def non_linear_function(psi_0, fom_args, ts, function_args):
    mask = mask_square(psi_0, psi_0.shape[0] // 2, psi_0.shape[1] // 2)

    def non_linear(variable):
        pot = mask * variable

        args_full = (*function_args, pot)
        args_zero = (*function_args, np.zeros(pot.shape))

        psi = np.array([psi_0.real, psi_0.imag])
        hp_sol = solver(fom_args[0]*psi, args_full, ts).ys[-1]
        hp_psi = np.sqrt(hp_sol[0] ** 2 + hp_sol[1] ** 2)

        lp_sol = solver(fom_args[1]*psi, args_full, ts).ys[-1]
        lp_psi = np.sqrt(lp_sol[0] ** 2 + lp_sol[1] ** 2)

        middle_y = psi_0.shape[0] // 2
        middle_x = psi_0.shape[1] // 2
        return np.sum(hp_psi - fom_args[2])**2 / np.max(hp_psi)**2 + np.sum(lp_psi - fom_args[3])**2 / np.max(lp_psi)**2

    return non_linear
