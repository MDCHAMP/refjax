from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import refjax as rj


def reference(N, d, lam, aoi):
    # Independent numpy Psi, Delta: Woollam/Fujiwara convention (N = n - ik), Airy recursion
    ncos = np.sqrt(N**2 - (N[0] * np.sin(aoi)) ** 2)
    ncos = np.where(ncos.imag > 0, -ncos, ncos)
    cos = ncos / N
    out = []
    for pol in "sp":
        def fres(i, j):
            if pol == "s":
                return (ncos[i] - ncos[j]) / (ncos[i] + ncos[j])
            return (N[j] * cos[i] - N[i] * cos[j]) / (N[j] * cos[i] + N[i] * cos[j])

        r = fres(-2, -1)
        for j in range(len(d) - 1, -1, -1):
            e = np.exp(-4j * np.pi * d[j] * ncos[j + 1] / lam)
            rij = fres(j, j + 1)
            r = (rij + r * e) / (1 + rij * r * e)
        out.append(r)
    rho = out[1] / out[0]
    return np.degrees(np.arctan(np.abs(rho))), np.degrees(np.angle(rho)) % 360


lam = np.linspace(370, 1000, 50)
aoi = np.radians(75.1)
nSi = 3.9 - 0.02j

cases = {
    "bare Si": ([1.0, nSi], []),
    "SiO2/Si": ([1.0, 1.46, nSi], [2.0]),
    "PS/SiO2/Si": ([1.0, 1.58, 1.46, nSi], [17.0, 2.0]),
    "thick PS/SiO2/Si": ([1.0, 1.58, 1.46, nSi], [600.0, 2.0]),
    "absorbing film": ([1.0, 2.0 - 0.3j, nSi], [30.0]),
}


def params(n, d):
    N = np.array(n, complex)[:, None] * np.ones_like(lam)
    # refjax takes n + ik
    return {"nbar": jnp.array(N.conj()), "thick": jnp.array(d, float), "aoi_rads": aoi}, N


@pytest.mark.parametrize("name", cases)
def test_spec_ellips(name):
    p, N = params(*cases[name])
    psi_ref, delta_ref = reference(N, np.array(cases[name][1], float), lam, aoi)
    psi, delta = rj.model.spec_ellips(p, jnp.array(lam))
    assert np.allclose(psi, psi_ref, atol=1e-9)
    assert np.allclose(delta, delta_ref, atol=1e-9)
    assert np.all((delta >= 0) & (delta < 360))


def test_ellips_can_jit_and_grad():
    p, _ = params(*cases["PS/SiO2/Si"])

    def f(thick):
        return jnp.sum(rj.model.spec_ellips({**p, "thick": thick}, jnp.array(lam))[0])

    g = jax.jit(jax.grad(f))(p["thick"])
    assert np.all(np.isfinite(g))


# %% Measured data: one spectrum of a thick polymer film on Si (native oxide), 450-1000 nm, with the substrate's indices
# at the measurement temperature (Franta's Si, Malitson's SiO2). The parameters are a fit of a Cauchy film to it,
# whose reduced chi2 against the stated errors was 12.860.

data = pd.read_csv(Path(__file__).parent.parent / "examples" / ".example_ellips_data.csv")
film = {"A": 1.5592380152192518, "B": 0.009669302459697682, "thick_nm": 269.12932822155117}
oxide_nm = 1.7976495299127038
aoi_deg = 75.11362116  # the instrument's 75.1034 plus the fitted offset


def test_spec_ellips_vs_data():
    lam = jnp.array(data.wavelength_nm)
    n_film = film["A"] + film["B"] / (lam / 1000) ** 2
    nbar = jnp.stack([jnp.ones_like(lam), n_film, jnp.array(data.n_sio2), jnp.array(data.n_si) + 1j * jnp.array(data.k_si)])
    p = {"nbar": nbar.astype(complex), "thick": jnp.array([film["thick_nm"], oxide_nm]), "aoi_rads": jnp.radians(aoi_deg)}
    psi, delta = rj.model.spec_ellips(p, lam)
    r = np.concatenate([(psi - data.psi) / data.dpsi, ((delta - data.delta + 180) % 360 - 180) / data.ddelta])
    chi2_red = np.sum(r**2) / (r.size - 3)
    assert abs(chi2_red - 12.860) < 1e-2  # the fit's own value
