import jax.numpy as jnp
from functools import reduce

# %% Abeles Matrix formalism

# Paramter shapes
# params = {
#     "thick": (n,) layers
#     "rough": (n + 1) surfaces
#     "SLD":  (n + 2) media
#     "scale": (),
#     "bkg": (),
# }


def abeles(params, Q):
    # SLD profile differences
    d_rho = (params["SLD"] - params["SLD"][0]).astype(complex) * 1e-6
    # compute wavevectors for the nth layer
    kn = jnp.sqrt(((Q[:, None] / 2) ** 2) - (4 * jnp.pi * d_rho))
    # Fresnel reflection
    a, b = kn[:, :-1], kn[:, 1:]
    r = (a - b) / (a + b) * jnp.exp(-2 * a * b * params["rough"] ** 2)
    # Compute and multiply characteristic matricies elementwise (n_layers + 1, Q.shape)
    C00 = jnp.exp(1j * a * jnp.array([0, *params["thick"]])).T
    C11 = C00**-1
    C10 = r.T * C00
    C01 = r.T * C11

    # For some reason manually performing the 2x2 matrix multiplications
    # is a huge speedup (~2x) in jit-ed calls compared to matmul (???)
    # Could be something to do with the very wide broadcast size?
    # The batch dim size is often >> 1k and idk if this is not efficient
    # Perhaps there is an einsum that works better? - there isn't 08/04/24
    # Tested it and seemingly not. Bizzare behavior. Absolute rabbithole.
    # Perhaps constructing the large array (n_layers+1, Q.shape, 2,2) is expensive somehow? - Nope 08/04/24.
    # Hard to imagine why this is - something to do with optimisations in jaxprs at a guess.

    M00, M10, M01, M11 = C00[0], C10[0], C01[0], C11[0]
    for c00, c10, c01, c11 in zip(C00[1:], C10[1:], C01[1:], C11[1:]):
        a = M00 * c00 + M10 * c01
        b = M00 * c10 + M10 * c11
        c = M01 * c00 + M11 * c01
        d = M01 * c10 + M11 * c11
        M00, M10, M01, M11 = a, b, c, d
    R = M01 / M00
    return params["scale"] * (R * R.conj()).real + params["bkg"]


# %% Spectroscopic ellipsometry

# Parameter shapes
# params = {
#     "nbar": (n + 2, lam.shape) complex refractive index n + ik (k >= 0), ambient first
#     "thick": (n,) layers, same length unit as lam
#     "aoi_rads": () angle of incidence in the ambient
# }
# Returns psi, delta in degrees: tan(psi) exp(i delta) = rp / rs in Woollam's convention,
# delta in [0, 360). Compare delta to data modulo 360.


def spec_ellips(params, lam):
    nbar = params["nbar"]
    # n cos(theta) in each medium (Snell); principal sqrt gives Im >= 0, the decaying wave for n + ik
    ncos = jnp.sqrt(nbar**2 - (nbar[0] * jnp.sin(params["aoi_rads"])) ** 2)
    cos = ncos / nbar
    ncos22 = ncos[1:]
    ncos11 = ncos[:-1]
    ncos21 = nbar[1:] * cos[:-1]
    ncos12 = nbar[:-1] * cos[1:]
    rs = (ncos11 - ncos22) / (ncos11 + ncos22)
    ts = 2 * ncos11 / (ncos11 + ncos22)
    rp = (ncos21 - ncos12) / (ncos12 + ncos21)
    tp = 2 * ncos11 / (ncos12 + ncos21)
    dz = 1j * params["thick"][:, None] * 2 * jnp.pi * ncos[1:-1] / lam
    one = jnp.ones_like(rs)
    edr = jnp.exp(dz)
    nedr = jnp.exp(-dz)

    def TMM(r, t):
        ms = jnp.array([[nedr, nedr * r[1:]], [edr * r[1:], edr]]) / t[1:]
        ms0 = jnp.array([[one[0], r[0]], [r[0], one[0]]]) / t[0]
        Ms = reduce(jnp.matmul, ms.transpose(2, 3, 0, 1), ms0.T)
        return Ms[:, 1, 0] / Ms[:, 0, 0]

    rho = TMM(rp, tp) / TMM(rs, ts)
    # n + ik is the conjugate of Woollam's n - ik, hence the minus sign on delta
    _psi = jnp.degrees(jnp.arctan(jnp.abs(rho)))
    _delta = jnp.degrees(-jnp.angle(rho)) % 360
    return _psi, _delta
