import pytest

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import refjax as rj
from refnx.reflect.reflect_model import reflectivity


# %% Fixture some data

#data
data = pd.read_csv( r"refjax\examples\.example_data.csv")

#Parameters
struct = jnp.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [254.9, 6.37, 0.0, 6.8],
        [17.0, 3.47, 0.0, 5.2],
        [0.0, 2.07, 0.0, 2.07],
    ]
)  # blame refnx for this

scale = 1.0
bkg = 2e-7


F_refnx = reflectivity(np.array(data.Q), np.array(struct), scale,bkg, np.array(data.FWHM))

params = {
    "thick": jnp.array([254.9, 17.0]),
    "SLD": jnp.array([0.0, 6.37, 3.47, 2.07]),
    "rough": jnp.array([6.8, 5.2, 2.07]),
    "scale": scale,
    "bkg": bkg,
}


# %%

def test_fit():
    kernel = rj.kernel(rj.model.abeles, rj.smear.pointwise_gaussian, jit=0)
    F = kernel(params, jnp.array(data.Q), jnp.array(data.FWHM))
    
    # Vs refnx
    # tolerance because the smearing quadrature methods are not identical
    assert ((F-F_refnx)**2).sum()/F.shape[0] < 1e-6 
    
    # Vs data
    assert ((F-data.R)**2).sum()/F.shape[0] < 1e-1 # low MSE


def test_can_jit():
    kernel = rj.kernel(rj.model.abeles, rj.smear.pointwise_gaussian)
    kernel(params, jnp.array(data.Q), jnp.array(data.FWHM))
    
    
def test_can_grad():
    kernel = jax.jit(rj.kernel(rj.model.abeles, rj.smear.pointwise_gaussian))
    dk_dtheta = jax.jacfwd(kernel)
    dk_dtheta(params, jnp.array(data.Q), jnp.array(data.FWHM))

