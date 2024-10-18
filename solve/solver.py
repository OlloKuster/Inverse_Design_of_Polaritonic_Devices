import diffrax
import jax.numpy as np

import jax
from diffrax import AbstractDiscreteTerminatingEvent
from typing import Callable, Optional



def laplace_2d(f, dx, dy, bc=('wrap', 'wrap')):
    fx = np.pad(f, ((1, 1), (0, 0)), mode=bc[0])
    fy = np.pad(f, ((0, 0), (1, 1)), mode=bc[1])
    ddx = (fx[:-2, :] + fx[2:, :] - 2 * f) / dx**2
    ddy = (fy[:, :-2] + fy[:, 2:] - 2 * f) / dy**2
    return ddx + ddy


def nabla_forward_y(f, dy, bc='constant'):
    fy = np.pad(f, ((1, 0), (0, 0)), mode=bc)
    nablay = (fy[:-1, :] - f) / dy
    return nablay


def nabla_backward_y(f, dy, bc='constant'):
    fy = np.pad(f, ((0, 1), (0, 0)), mode=bc)
    nablay = (f - fy[1:, :]) / dy
    return nablay


def nabla_forward_x(f, dx, bc='constant'):
    fx = np.pad(f, ((0, 0), (1, 0)), mode=bc)
    nablax = (fx[:, :-1] - f) / dx
    return nablax


def nabla_backward_x(f, dx, bc='constant'):
    fx = np.pad(f, ((0, 0), (0, 1)), mode=bc)
    nablax = (f - fx[:, 1:]) / dx
    return nablax


def gross_pitaevskii(t, y, args):
    dx, dy, hbar, mc, u, P_eff, kappa, gamma, sigma, v = args
    psi = y[0] + 1j * y[1]
    rhs = (-0.5 * hbar ** 2 / mc * laplace_2d(psi, dx, dy)
          + v * psi
          + u * np.abs(psi)**2 * psi
          + 1j * (- kappa - gamma * np.abs(psi)**2) * psi)\
          / (1j * hbar)
    return np.array([rhs.real, rhs.imag])


def gross_pitaevskii_pml(t, y, args):
    dx, dy, hbar, mc, u, P_eff, kappa, gamma, sigma, v = args
    psi = y[0] + 1j * y[1]
    c = 1 / (1 + np.exp(1j * np.pi / 4) * sigma)
    laplace_x = (c[1] * nabla_forward_x(psi, dx) - c[1] * nabla_backward_x(psi, dx)) / dx
    laplace_y = (c[0] * nabla_forward_y(psi, dy) - c[0] * nabla_backward_y(psi, dy)) / dy
    rhs = (-0.5 * hbar ** 2 / mc * (laplace_x + laplace_y)
          + v * psi
          + u * np.abs(psi)**2 * psi
          + 1j * (- kappa - gamma * np.abs(psi)**2) * psi + 1j * P_eff)\
          / (1j * hbar) # Adjust/remove P_eff depending on your needs
    return np.array([rhs.real, rhs.imag])


def solve_gp(y0, args, ts, initial_step=0.1, max_steps=16**5):
    term = diffrax.ODETerm(gross_pitaevskii_pml)
    solver = diffrax.Tsit5()
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    stepsize_controller = diffrax.PIDController(pcoeff=0.3, icoeff=0.3, dcoeff=0, rtol=1e-12, atol=1e-14)

    event = diffrax.SteadyStateEvent(atol=0, rtol=1e-6)

    sol = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=initial_step,
        y0=y0,
        args=args,
        adjoint=adjoint,
        # saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        discrete_terminating_event=event,
    )

    return sol


# Print current timestep when debugging.
class CustomEvent(AbstractDiscreteTerminatingEvent):

    rtol: Optional[float] = None
    atol: Optional[float] = None

    def __call__(self, state, *, terms, args, solver, stepsize_controller, **kwargs):
        del kwargs
        _atol = self.atol
        _rtol = self.rtol
        vf = solver.func(terms, state.tprev, state.y, args)
        jax.debug.print("t: {}", state.tprev)
        # jax.debug.print("vf: {}", self.norm(vf))
        # jax.debug.print("y: {}", self.norm(state.y))
        return False