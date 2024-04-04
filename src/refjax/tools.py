import jax
from refjax.smear import no_smear


def kernel(model, smear=no_smear, smear_params={}, jit=True):
    """Build a reflection kenrel from a reflection model and a smearing kernel"""
    def kern(params, q, dq=None):
        return smear(params, q, dq, model, **smear_params)    
    if jit:
        kern = jax.jit(kern)
    return kern


def _from_refnx(refnx_model):
    """TODO Pull out required parameters from a refnx model"""
    raise NotImplementedError

def _validata_params(params):
    """TODO Validata params manually because don't want to do this in a jit"""
    raise NotImplementedError


