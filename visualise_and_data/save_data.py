import h5py
import jax.numpy as np


def save(psi, sol, potential, losses, parameters, path):
    extent, resolution, ts, dx, dy, hbar, mc, u, kappa, gamma, k, sigma, mu = parameters
    with h5py.File(path, mode='w') as f:
        grp = f.create_group("Flat top")
        grp.create_dataset("potential", data=potential)
        grp.create_dataset("losses", data=losses)
        grp.create_dataset("psi", data=psi)
        grp.create_dataset("psi_complex", data=sol)
        grp.attrs["extent"] = extent
        grp.attrs["resolution"] = resolution
        grp.attrs["t0"] = ts[0]
        grp.attrs["t1"] = np.inf
        grp.attrs["dx"] = dx
        grp.attrs["dy"] = dy
        grp.attrs["hbar"] = hbar
        grp.attrs["mc"] = mc
        grp.attrs["u"] = u
        grp.attrs["kappa"] = kappa
        grp.attrs["gamma"] = gamma
        grp.attrs["v"] = 0
        grp.attrs["k"] = k
        grp.attrs["FoM"] = "Flat top at 5%"
        f.close()

