__version__ = "0.1.1"

from jax import config
from refjax import model, smear
from refjax.tools import kernel

__all__ = ["__version__", "kernel", "model", "smear"]

config.update("jax_enable_x64", True)
