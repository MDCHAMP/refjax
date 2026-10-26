import jax
import jax.numpy as jnp
import numpy as np
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
