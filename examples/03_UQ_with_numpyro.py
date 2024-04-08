import pandas as pd
import matplotlib.pyplot as plt

import refjax
from refjax.model import abeles
from refjax.smear import pointwise_gaussian

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# %%

# Once again we will consider the slab model and our example data.

data = pd.read_csv(".example_data.csv")

plt.loglog(data.Q, data.R)
plt.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.xlabel("$Q$ ($A^{-1}$)")
plt.ylabel("Reflectivity")
plt.legend(["Data", "$\pm 3 \sigma$"])
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

# Recall that parameters marked with * are not known exactly. Our objective will be to learn these parameters and quantify our uncertainty within a Bayesian sampling framework. Since refjax provides a jax.grad compatible kernel we can use a Hamiltonian Monte-Carlo scheme to efficiently sample from the paramter space. In this example we will use the No U-Turn Sampler (NUTS) implementation in numpyro.

# %%

# As in the previous examples the first step it to define our reflectivity kernel

kernel = refjax.kernel(abeles, pointwise_gaussian)

# Next we define our proabalistic graphical model using numpyro primitives. Here we will place a prior distribution over unknown parameters.


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

    # Finally our observation likelihood - here we will assume a Gaussian (but there are several problems with this choice - for example, R cannot be negative!)
    numpyro.sample("obs", dist.Normal(R_hat, sig_R), obs=R)


# %%
# Running the inference

# Set some parameters for the MCMC - see numpyro docs for details
mcmc_opts = {
    "num_warmup": 500,
    "num_samples": 500,
    "num_chains": 1,
    "progress_bar": True,
}

# Run the NUTS MCMC
mcmc = MCMC(NUTS(pgm), **mcmc_opts)
mcmc.run(
    jax.random.PRNGKey(4583403),
    jnp.array(data.Q),
    jnp.array(data.FWHM),
    jnp.array(data.R),
    jnp.array(data.sig_R),
)

# Print summary of the inference and draw some samples from the chains
mcmc.print_summary()
samps = mcmc.get_samples()

# %%

# Finally plot the mean and +/- 3 std results.
plt.figure()
plt.loglog(data.Q, data.R)
plt.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.plot(data.Q, samps["R"].mean(0))
plt.fill_between(
    data.Q,
    samps["R"].mean(0) - 3 * samps["R"].std(0),
    samps["R"].mean(0) + 3 * samps["R"].std(0),
    alpha=0.2,
    color="C3",
)
plt.legend(["Data", "$\pm 3 \sigma$", "Mean prediction", "$\pm 3 \sigma$"])
plt.show()

# We now have access to distributional estimates of each of the parameters in our inference
plt.figure()
plt.hist(samps["thick"], density=1, bins=20)
plt.xlabel("Thickness")
plt.ylabel("Density")
plt.show()
