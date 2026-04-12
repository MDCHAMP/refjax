import pandas as pd
import matplotlib.pyplot as plt

import refjax
from refjax.model import abeles
from refjax.smear import pointwise_gaussian

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal

# %%

# Once again we will consider the slab model and our example data.

data = pd.read_csv(".example_data.csv")

plt.loglog(data.Q, data.R)
plt.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.xlabel("$Q$ ($A^{-1}$)")
plt.ylabel("Reflectivity")
plt.legend(["Data", "$\\pm 3 \\sigma$"])
plt.show()

# Slab model parameters
#
#    Fronting (air):   SLD=0
#
# ------------------   rough=~5.2   *
#
#     Polymer film:    thick=~255nm *
#                      SLD=~6.4     *
#
# ------------------   rough=5.0
#
#        Si02:         thick=17nm
#                      SLD=3.47
#
# ------------------   rough=2.07
#
#     Backing (Si):    SLD=2.07
#

# Misc. parameters
#                      scale=~1.0   *
#                      bkg=2e-7

# We will learn the same unknown parameters as before but now with Stochastic
# Variational Inference (SVI). SVI fits a variational distribution q(z) to
# approximate the posterior p(z|data) by maximising the ELBO. This is faster
# than MCMC but produces an approximate posterior.

# %%

# As in the previous examples the first step is to define our reflectivity kernel

kernel = refjax.kernel(abeles, pointwise_gaussian)

# The probabilistic graphical model is unchanged from the MCMC example.


def pgm(Q, FWHM, R, sig_R):
    # Prior over unknown parameters
    poly_thick = numpyro.sample("thick", dist.Uniform(210, 275))
    poly_sld = numpyro.sample("sld", dist.Uniform(5.8, 7.0))
    poly_sio2_rough = numpyro.sample("rough", dist.Uniform(0.0, 7.0))
    scale = numpyro.sample("scale", dist.Uniform(0.5, 2.0))

    # Collect our parameters in the form refjax expects
    params = {
        "thick": jnp.array([poly_thick, 17.0]),
        "SLD": jnp.array([0.0, poly_sld, 3.47, 2.07]),
        "rough": jnp.array([poly_sio2_rough, 5.0, 2.07]),
        "scale": scale,
        "bkg": 2e-7,
    }
    # Make prediction using our kernel
    R_hat = numpyro.deterministic("R", kernel(params, Q, FWHM))

    # Observation likelihood
    numpyro.sample("obs", dist.Normal(R_hat, sig_R), obs=R)


# %%
# Define the variational guide
#
# AutoNormal fits a diagonal-normal guide in the unconstrained parameter space,
# with learned mean and standard deviation for each latent variable.

guide = AutoNormal(pgm)

# %%
# Running SVI

Q = jnp.array(data.Q)
FWHM = jnp.array(data.FWHM)
R = jnp.array(data.R)
sig_R = jnp.array(data.sig_R)

optimizer = numpyro.optim.Adam(step_size=1e-3)
svi = SVI(pgm, guide, optimizer, loss=Trace_ELBO())

svi_result = svi.run(
    jax.random.PRNGKey(0),
    num_steps=10_000,
    Q=Q, FWHM=FWHM, R=R, sig_R=sig_R,
)

# %%
# Plot the ELBO loss curve to check convergence

plt.figure()
plt.plot(svi_result.losses)
plt.xlabel("Step")
plt.ylabel("ELBO loss")
plt.show()

# %%
# Draw samples from the fitted guide to approximate the posterior

predictive = Predictive(guide, params=svi_result.params, num_samples=500)
samps = predictive(jax.random.PRNGKey(1), Q, FWHM, R, sig_R)

# Evaluate R_hat for each sample
R_samps = jax.vmap(lambda s: kernel(
    {
        "thick": jnp.array([s["thick"], 17.0]),
        "SLD": jnp.array([0.0, s["sld"], 3.47, 2.07]),
        "rough": jnp.array([s["rough"], 5.0, 2.07]),
        "scale": s["scale"],
        "bkg": 2e-7,
    },
    Q, FWHM,
))(samps)

# %%

# Plot the mean and +/- 3 std results.
plt.figure()
plt.loglog(data.Q, data.R)
plt.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.plot(data.Q, R_samps.mean(0))
plt.fill_between(
    data.Q,
    R_samps.mean(0) - 3 * R_samps.std(0),
    R_samps.mean(0) + 3 * R_samps.std(0),
    alpha=0.2,
    color="C3",
)
plt.legend(["Data", "$\\pm 3 \\sigma$", "Mean prediction", "$\\pm 3 \\sigma$"])
plt.show()

# We now have access to distributional estimates of each of the parameters in our inference
plt.figure()
plt.hist(samps["thick"], density=1, bins=20)
plt.xlabel("Thickness")
plt.ylabel("Density")
plt.show()

