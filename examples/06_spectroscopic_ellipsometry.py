# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import refjax
from refjax.model import spec_ellips

# %%
# This example script demonstrates how to use refjax to model spectroscopic ellipsometry data, and to fit a film to it.

# The example data is one (real) spectrum of a thick polymer film on a Si wafer with a thin layer of native SiO2 between them,
# measured between 450 and 1000 nm. The thick film puts several interference fringes in the spectrum.

data = pd.read_csv(".example_ellips_data.csv")
print(data)

# The columns are:

# wavelength_nm  (the wavelength, nm)
# psi, delta     (the ellipsometric angles, degrees)
# dpsi, ddelta   (their stated uncertainties, degrees)
# n_si, k_si     (the Si substrate's complex refractive index n + ik at the measurement temperature)
# n_sio2         (the native oxide's refractive index)

# The substrate's indices are given with the data, so the only unknowns here are the film's.

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(data.wavelength_nm, data.psi)
axs[1].plot(data.wavelength_nm, data.delta)
axs[0].set_ylabel(r"$\Psi$ (deg)")
axs[1].set_ylabel(r"$\Delta$ (deg)")
for ax in axs:
    ax.set_xlabel("wavelength (nm)")
plt.show()

# %%

# The model: a stack of media, each with a complex refractive index n + ik at every wavelength, ambient first and substrate last.

#    Ambient (air):    n = 1
#
# ------------------
#
#    Polymer film:     thick = ~250 nm  *
#                      n = A + B / lambda^2 (Cauchy, lambda in um), A = ~1.55 *, B = ~0.01 *
#
# ------------------
#
#    SiO2:             thick = 1.80 nm
#                      n from the data (n_sio2)
#
# ------------------
#
#    Backing (Si):     n + ik from the data (n_si, k_si)

# Parameters marked with * are only known approximately. The angle of incidence is the instrument's, 75.11 degrees.

lam = jnp.array(data.wavelength_nm)
nbar_sio2 = jnp.array(data.n_sio2, dtype=complex)
nbar_si = jnp.array(data.n_si) + 1j * jnp.array(data.k_si)
oxide_nm = 1.7976
aoi_rads = jnp.radians(75.1136)


def predict(x):
    """Psi, Delta for x = (film thickness in nm, Cauchy A, Cauchy B)."""
    thick, A, B = x
    n_film = (A + B / (lam / 1000) ** 2).astype(complex)
    nbar = jnp.stack([jnp.ones_like(nbar_si), n_film, nbar_sio2, nbar_si])
    return spec_ellips({"nbar": nbar, "thick": jnp.array([thick, oxide_nm]), "aoi_rads": aoi_rads}, lam)


# spec_ellips returns Delta in [0, 360), Woollam's convention, so compare it with data modulo 360.


def residuals(x):
    psi, delta = predict(x)
    r_psi = (psi - jnp.array(data.psi)) / jnp.array(data.dpsi)
    r_delta = ((delta - jnp.array(data.delta) + 180) % 360 - 180) / jnp.array(data.ddelta)
    return jnp.concatenate([r_psi, r_delta])


x0 = jnp.array([250.0, 1.55, 0.01])
print(f"Starting guess: reduced chi2 = {jnp.sum(residuals(x0) ** 2) / (2 * lam.shape[0] - 3):.1f}")

# %%

# spec_ellips is jit and grad compatible, so a few Gauss-Newton steps with an exact Jacobian fit the film.

jac = jax.jit(jax.jacfwd(residuals))
res = jax.jit(residuals)
x = x0
for i in range(10):
    r, J = res(x), jac(x)
    x = x - jnp.linalg.solve(J.T @ J, J.T @ r)

r, J = res(x), jac(x)
chi2_red = jnp.sum(r**2) / (r.size - x.size)
sd = jnp.sqrt(jnp.diag(jnp.linalg.inv(J.T @ J)))
for name, v, s in zip(("thickness (nm)", "A", "B (um^2)"), x, sd):
    print(f"{name:15s} {v:.5f} +- {s:.5f}")
print(f"reduced chi2 = {chi2_red:.3f}")

# The fit gives a film of about 269.1 nm with A = 1.559 and B = 0.0097, and a reduced chi2 of about 12.9. The model follows every
# fringe, but a single uniform Cauchy layer doesn't describe a film this thick to within the instrument's small stated errors.

psi, delta = predict(x)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(data.wavelength_nm, data.psi, ".", ms=2, label="Data")
axs[0].plot(lam, psi, label="Model")
axs[1].plot(data.wavelength_nm, data.delta, ".", ms=2)
axs[1].plot(lam, delta)
axs[0].set_ylabel(r"$\Psi$ (deg)")
axs[1].set_ylabel(r"$\Delta$ (deg)")
for ax in axs:
    ax.set_xlabel("wavelength (nm)")
axs[0].legend()
plt.show()

# %%
