__version__ = "0.0.1"

from jax import config
import refjax.model
import refjax.smear
from refjax.tools import kernel

config.update("jax_enable_x64", True)