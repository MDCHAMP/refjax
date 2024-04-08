# %%
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

import refjax
from refjax.model import abeles
from refjax.smear import pointwise_gaussian

# %%
# This example script demonstrates how to use refjax to make make predictions from models of reflectometry.

# In this example we will look at some (real) example reflectometry data from a sample of polymer film on a Si backing with a thin layer of native Si02 between the two of them.

data = pd.read_csv(".example_data.csv")
print(data)

# Loading in the example data as a pandas array we see that we have four columns:

# Q     (momentum transfer)
# FWHM  (the full-width at half maximum of the Gaussian smearing kernel)
# R     (reflectivity)
# sig_R (std of our uncertainty in R)

plt.loglog(data.Q, data.R)
plt.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.xlabel("$Q$ ($A^{-1}$)")
plt.ylabel("Reflectivity")
plt.legend(["Data", "$\pm 3 \sigma$"])
plt.show()


# Looking now at the plotted data we see the characteristic ringing in the reflectivity as Q increases. We will now use refjax to build a model of this behavoir.

# %%

# It is our intention to model reflectometry data using the slab model below. Note that parameters marked with * are only known approximately.

# Slab model
#
#    Fronting (air):   SLD=0
#
# ------------------   rough=~5.2   *
#
#    Polymer film:     thick=~255 A *
#                      SLD=~6.4     *
#
# ------------------   rough=5.0
#
#        Si02:         thick=17 A
#                      SLD=3.47
#
# ------------------   rough=2.07
#
#     Backing (Si):    SLD=2.07
#

# Misc. parameters
#                      scale=~1.0 *
#                      bkg=2e-7

# We can collect these parameters into a dictionary for use in refjax. Note that for n (here n=2) layers in the above slab model the dimensions of the parameter arrays are:

# thick (n,)   media with thickness i.e n layers (nm)
# SLD   (n+2,) media through which the neutrons travel i.e n layers + fronting + backing
# rough (n+1,) interfaces between media i.e n+1
# scale ()     scaling of the reflectivity
# bkg   ()     offset of the reflectivity due to backround effects

# Assembling these into a pytree (python dict) in a format that refjax expects.

params = {
    "thick": jnp.array([255, 17.0]),
    "SLD": jnp.array([0.0, 6.4, 3.47, 2.07]),
    "rough": jnp.array([5.2, 5.0, 2.07]),
    "scale": 1.0,
    "bkg": 2e-7,
}


# %%

# Getting model predictions in refjax is easy, all we have to do is build our reflectometry kernel (a function that maps Q to R).

kernel = refjax.kernel(
    abeles
)  # i.e Abeles matrix formalism, no smearing in the detector (default)

# Note that the kernel is a jit-compiled function that is compatible with grad, scan, pmap etc.

# We can compute reflectivity by simply calling the kernel with the parameters and the Q values.
R_hat = kernel(params, jnp.array(data.Q))

# From the plot we can see that our model is doing a good job of representing the data.

plt.figure()
plt.loglog(data.Q, data.R)
plt.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.plot(data.Q, R_hat)
plt.xlabel("$Q$ ($A^{-1}$)")
plt.ylabel("Reflectivity")
plt.legend(["Data", "$\pm 3 \sigma$", "Model (no smearing)"])
plt.show()

# However, note that there are significant deviations between the model and the data, particluarly at the bottom of the fringes. This is happening because we have not taken into account the smearing in the detector.

# %%

# Adding smearing to a refjax kernel, we just pass the smearing kernel to the call to refjax.kernel

kernel = refjax.kernel(abeles, pointwise_gaussian)

# How when calling the new kernel with pointwise smearing we must pass the FWHM values (per Q). If we had used a constant Gaussian smear then we would only have to pass a single FWHM value. Arbitrary kernel smearing is not currently supported in refjax but it is something we are working on adding. Raise an issue on the refjax github page if this is something that would be useful to you.

R_hat = kernel(params, jnp.array(data.Q), jnp.array(data.FWHM))

# print and plot
plt.figure()
plt.loglog(data.Q, data.R)
plt.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.plot(data.Q, R_hat)
plt.xlabel("$Q$ ($A^{-1}$)")
plt.ylabel("Reflectivity")
plt.legend(["Data", "$\pm 3 \sigma$", "Model (Gaussian smearing)"])
plt.show()

# Note that there are still some slight differences between our model and the data. This is because the parameters are not known exactly. In the following example scripts we will infer the parameters by making use of the gradient of the error w.r.t to the paramters for efficient learning.

# %%
