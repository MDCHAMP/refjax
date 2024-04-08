import jax
import jax.numpy as jnp
from numpy.polynomial.hermite import hermgauss

_ORDER = 17
herm_root, herm_weight = hermgauss(_ORDER)
herm_root, herm_weight = jnp.array(herm_root), jnp.array(herm_weight)

def no_smear(params, Q, FWHM, model, **kwargs):
    return model(params, Q)

def constant_gaussian(params, Q, cont_smear, model, **kwargs):
    '''
    FWHM = const_smear% * Q
    Gaussian kernel smearing.
    There are a few options here can solve as a convolution or just do gauss-hermite? 
    For now treat as pointwise.
    '''
    return pointwise_gaussian(params, Q, Q*cont_smear/100, model, **kwargs)

def pointwise_gaussian(params, Q, FWHMs, model, **kwargs):
    '''
    Pointwise gaussian smearing kernel. 
    Compute the expectation integral using a 17th order Gauss-Hermite quadrature. 
    Note that this differs from the approch in refnx which uses a finite width interval so numerics may differ.'''
    Q_eval = 0.5 * FWHMs / jnp.sqrt(jnp.log(2.0)) * herm_root[:, None] + Q
    R = jax.vmap(model, in_axes=(None, 0))(params, Q_eval)
    return herm_weight @ R / jnp.sqrt(jnp.pi)

def custom_kernel(params, Q, kernel, model, **kwargs):
    '''TODO
    Manually compute the expectation interval of R(q) w.r.t k(q)
    Can use simpsons or a Gauss-quad maybe?
    '''
    raise NotImplementedError
