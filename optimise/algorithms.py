import jax.numpy as np
import nlopt


def gradient_descent(x, grad, gamma=50):
    return x + gamma * grad


def mma_no_constraints(min, f, variable, termination):
    opt = nlopt.opt(nlopt.LD_MMA, variable.size)
    opt.set_min_objective(f) if min else opt.set_max_objective(f)
    if termination['time'] is not None:
        opt.set_maxtime(termination['time'])
    if termination['steps'] is not None:
        opt.set_maxtime(termination['steps'])
    opt.optimize(variable.ravel()).reshape(variable.shape)