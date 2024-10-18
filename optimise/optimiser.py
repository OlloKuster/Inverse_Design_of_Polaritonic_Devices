import h5py
import jax
import jax.numpy as np
import nlopt

import dm_pix as pix
import time
import matplotlib.pyplot as plt


def optimiser(psi_0, fom, constraint, variable, function_args, fom_args, ts, opt):
    print("Starting optimisation")
    loss_hist = []
    v_hist = []

    sigma = np.float64(opt)
    kernel_size = 2 * round(4.0 * sigma) + 1  # Kernel for min. feature size.

    fom_function = fom(psi_0, fom_args, ts, function_args)

    def f(x, grad):
        start = time.time()
        x = x.reshape((*variable.shape, 1))
        # Uncomment when enforcing min. feature size.
        # x = pix.gaussian_blur(x, sigma, kernel_size, padding='SAME')
        x = np.squeeze(x)
        value, gradient = jax.value_and_grad(fom_function)(x)
        v_hist.append(x.copy())

        if grad.size > 0:
            grad[:] = gradient.ravel()
        value = float(value)
        loss_hist.append(value)
        end = time.time()
        print(f"time: {end-start}")
        print(value)
        return value

    opt = nlopt.opt(nlopt.LD_LBFGS, variable.size)
    opt.set_min_objective(f)

    if constraint is not None:
        constraint_function = constraint(psi_0, fom_args, ts, function_args)

        def c(x, grad):
            x = x.reshape(variable.shape)
            value, gradient = jax.value_and_grad(constraint_function)(x)
            if grad.size > 0:
                grad[:] = gradient.ravel()
            value = float(value)
            return value

        opt.add_inequality_constraint(c, 0.1)

    offset = 0.6582**2 / 0.6585 * 10**2/2
    opt.set_maxeval(50)
    opt.set_lower_bounds(-25)
    opt.set_upper_bounds(25)
    opt.set_ftol_abs(1e-6)
    opt.set_ftol_rel(1e-3)
    opt.optimize(variable.ravel()).reshape(variable.shape)

    return v_hist[np.argmin(np.array(loss_hist))], loss_hist

