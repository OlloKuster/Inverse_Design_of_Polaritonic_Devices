from dataclasses import asdict

import jax
import jax.numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import initialise.functions as f
import initialise.parameters as p
import solve.solver as s
from initialise.initialiser import initialise
import optimise.optimiser as o
import visualise_and_data.plot as plot
import visualise_and_data.save_data as save
import optimise.figure_of_merit as fom

import time

def main(mode):
    device = jax.devices()[1].platform
    jax.config.update("jax_platform_name", device)
    jax.config.update("jax_enable_x64", True)

    if mode == "flat top":
        cavity = p.Cavity(resolution=20,
                          extent=np.array([[-5, 5], [-5, 5]]),
                          maxtime=np.inf,
                          timesteps=50)
        gaussian = p.GaussianParameters(k=np.array([0.0, 0.0]),
                                        amplitude=1.,
                                        mu=np.array([0, 0]),
                                        sigma=2.)
        phys = p.PhysicalParameters(kappa=4.)

        v = np.zeros(
            (cavity.size_y, cavity.size_x))

        flat_top = p.FlatTop(cavity.size_x, cavity.size_y, 0.15)

        fom_args = [flat_top.left, flat_top.right]
        figure_of_merit = fom.zero_grad_function

    if mode == "lens":
        cavity = p.Cavity(resolution=40,
                          extent=np.array([[-5, 20], [-5, 5]]),
                          maxtime=1.5,
                          timesteps=50)
        gaussian = p.GaussianParameters(k=np.array([10.0, 0.0]),
                                        amplitude=1.,
                                        mu=np.array([0, 0]),
                                        sigma=2.)
        phys = p.PhysicalParameters(kappa=0.)

        v = np.zeros(
            (cavity.size_y, cavity.size_x))
        # Add offset for CW pumping w/ finite k at small/zero U
        # otherwise use analytically/numerically calculated offset.
        # - p.PhysicalParameters.hbar ** 2 / p.PhysicalParameters.mc * 10 ** 2 / 2
        fom_args = []
        figure_of_merit = fom.lens_focus

    ts = [0., cavity.maxtime]
    dx = cavity.dx
    dy = cavity.dy
    gaussian_args = asdict(gaussian).values()
    time_gaussian_args = (*gaussian_args, 0.)
    time_gaussian = f.time_gaussian_function(cavity.extent, cavity.size_x, cavity.size_y, *time_gaussian_args)
    sigma = f.absorbing_layer(10, 10, (cavity.size_y, cavity.size_x), 1e-3)
    function_args = (dx, dy, phys.hbar, phys.mc, phys.u, time_gaussian, phys.kappa, phys.gamma, sigma)

    psi_0 = time_gaussian

    opt_pot, loss = o.optimiser(psi_0, figure_of_merit, None, v, function_args, fom_args, ts, 0.1)
    sol = s.solve_gp(np.array([psi_0.real, psi_0.imag]), (*function_args, opt_pot), ts).ys[-1]
    psi = np.sqrt(sol[0] ** 2 + sol[1] ** 2)

    plot.plot_save(psi, cavity.extent, f"./psi_{mode}.png")
    plot.plot_save(opt_pot, cavity.extent, f"./pot_{mode}.png")
    plt.plot(loss)
    plt.savefig(f"./loss_{mode}.png")
    plt.close()
    full_parameters = (cavity.extent, cavity.resolution, ts,
                       cavity.dx, cavity.dy, phys.hbar, phys.mc, phys.u, phys.kappa, phys.gamma,
                       gaussian.k, gaussian.sigma, gaussian.mu)
    save.save(psi, sol, opt_pot, loss, full_parameters, f"/users/tfp/okuster/code/Inverse_design_of_polaritonic_devices/data_{mode}.h5")


if __name__ == "__main__":
    # Both of the examples in the paper can be reproduced by selecting the respective modes.
    mode = ["flat top", "lens"]
    start = time.time()
    with jax.default_device(jax.devices()[1]):
        main(mode[0])
    print(time.time() - start)