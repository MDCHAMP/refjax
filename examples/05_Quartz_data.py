# %%

import pandas as pd
import matplotlib.pyplot as plt

import refjax
from refjax.model import abeles
from refjax.smear import pointwise_gaussian

import jax
import jax.numpy as jnp

import optax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
# %%

def refplot(Q, R, sig_R, R_hat=None, sig_Rhat=None, fig=True, c="C1"):
    x2pp = 0.0
    if fig:
        plt.figure(figsize=(6, 4))
    plt.scatter(Q, R, s=0.2, color='C0')
    plt.fill_between(Q, R - 3 * sig_R, R + 3 * sig_R, alpha=0.2, color="C0")
    if R_hat is not None:
        plt.loglog(Q, R_hat, c=c)
        x2pp = chisqrpp(R, sig_R, R_hat)
    if sig_Rhat is not None:
        plt.fill_between(
            Q, R_hat - 3 * sig_Rhat, R_hat + 3 * sig_Rhat, alpha=0.2, color=c
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$Q$ ($A^{-1}$)")
    plt.ylabel("$R$")
    rh = "$\hat{R}: "
    if sig_Rhat is not None:
        rh = "$E[\hat{R}]: "
    labs = [
        "$R$",
        "$\pm 3 \sigma$",
        rh + f"\chi^2$:{x2pp:.3g}",
        "$\pm 3 \sigma$",
    ]

    if fig:
        plt.legend(
            labs,
            bbox_to_anchor=(0.0, 1, 1, 0),
            loc=3,
            ncol=4,
            mode="expand",
            borderaxespad=0.0,
            frameon=False,
            fontsize="small",
        )

def chisqrpp(R, sig_R, R_hat):
    R, sig_R, R_hat = jnp.array(R), jnp.array(sig_R), jnp.array(R_hat)
    return jnp.sum(((R - R_hat) / sig_R) ** 2, -1) / R.shape[0]



# %%
# Read data

data = pd.read_csv('.Quartz_data.csv')
Q, FWHM, R, sig_R = (
    jnp.array(data.Q),
    jnp.array(data.FWHM),
    jnp.array(data.R),
    jnp.array(data.sig_R),
)

# Slab model
#
#    air:              SLD
#
# ------------------   rough
#
#    quartz:           thick
#                      SLD
#
# ------------------   rough
#
#    Si02:             thick
#                      SLD
#
# ------------------   rough
#
#     Si:              SLD
#

# %%
# fit with parameters from athe original authors

params = {
    "thick": jnp.array([1453.5, 22.84]),
    "SLD": jnp.array([0.0, 4.2392, 3.47, 2.07]),
    "rough": jnp.array([11.061, 5.6208, 14.06]),
    "scale": 1.05,
    "bkg": 3.1584e-7,
}

kernel = refjax.kernel(abeles, pointwise_gaussian)

refplot(Q, R, sig_R, kernel(params, Q, FWHM))
plt.suptitle('Gutfreund et al.')

# %%
# Gradient descent fit


def map(theta):
    theta["bkg"] = jnp.exp(
        theta["bkg"]
    )  # fit log of the bkg parameter as +ve constraint
    theta["SLD"] = jnp.array([0, *theta["SLD"][1:]])  # fix that air SLD = 0.0
    return theta


def error(theta):
    R_hat = kernel(map(theta), Q, FWHM)
    return chisqrpp(R, sig_R, R_hat)


# rough initial guess of parameters
theta0 = {
    "thick": jnp.array([1500.0, 10.0]),
    "SLD": jnp.array([0.0, 5.0, 4.2, 2.07]),
    "rough": jnp.array([5.0, 5.0, 5.0]),
    "scale": 1.00,
    "bkg": jnp.log(1e-7),
}


# Initialise an optax optimiser for gradient descent
optimiser = optax.adam(learning_rate=5e-2)
opt_state = optimiser.init(theta0)


# This function performs one step of the gradient descent and returns the parametrs, optimiser state and the current valaue of the loss funciton.
@jax.jit
def step(params, opt_state):
    loss_value, grads = jax.value_and_grad(error)(params)
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value


# Now we run our gradient descent in a loop and keep the final value of theta
theta = theta0
losses = []  # keep track of loss value to track model convergence.
for i in range(2000+1):
    theta, opt_state, loss_value = step(theta, opt_state)
    losses.append(loss_value)
    # print(theta)
    if i % 100 == 0:
        print(f"step {i}, loss: {loss_value}")

# %%

refplot(Q, R, sig_R, kernel(map(theta0), Q, FWHM))
plt.suptitle('Initial guess')

refplot(Q, R, sig_R, kernel(map(theta), Q, FWHM))
plt.suptitle('ADAM Optimised fit')

# %%
# Fit the Quartz data with numpyro + refjax
# use the GD result (fast) as an initial condition

def pgm(Q, FWHM, R, sig_R):
    # Prior over unknown parameters
    q_thick = numpyro.sample("q_thick", dist.Uniform(1400, 1500)) 
    sio2_thick = numpyro.sample("si02_thick", dist.Uniform(0.0, 50.0))

    q_sld = numpyro.sample("q_sld", dist.Uniform(0.0, 5.0))
    sio2_sld = numpyro.sample("si02_sld", dist.Uniform(0.0, 5.0))
    si_sld = numpyro.sample("si_sld", dist.Uniform(0.0, 5.0))

    rough = numpyro.sample("rough", dist.Uniform(2, 20), sample_shape=(3,))

    scale = numpyro.sample("scale", dist.Uniform(0.9, 1.5))
    logbkg = numpyro.sample("logbkg", dist.Uniform(-20, 0))

    # Collect our parameters in the form refjax expects
    params = {
        "thick": jnp.array([q_thick, sio2_thick]),
        "SLD": jnp.array([0.0, q_sld, sio2_sld, si_sld]),
        "rough": rough,
        "scale": scale,
        "bkg": jnp.exp(logbkg),
    }
    numpyro.deterministic("refjax_params", params)
    R_hat = numpyro.deterministic("R", kernel(params, Q, FWHM))
    numpyro.sample("obs", dist.Normal(R_hat, sig_R), obs=R)


chain_init_values = {
    "q_thick": theta["thick"][0],
    "si02_thick": theta["thick"][1],
    "q_sld": theta["SLD"][1],
    "si02_sld": theta["SLD"][2],
    "si_sld": theta["SLD"][3],
    "rough": theta["rough"],
    "scale": theta["scale"],
    "logbkg": jnp.log(theta["bkg"]),
}


# %% 
# With the chain 

# Set some parameters for the MCMC - see numpyro docs for details
mcmc_opts = {
    "num_warmup": 1_000,  
    "num_samples": 1_000, # Paper uses 2000 for ESS plot
    "num_chains": 1,      # Paper uses 10 for ESS plot
    "progress_bar": True,
}

# Run the NUTS MCMC

init_strat = numpyro.infer.util.init_to_value(values=chain_init_values)

hmc = MCMC(NUTS(pgm, init_strategy=init_strat), **mcmc_opts)
# Inference can start slow because some priors a quite wide and M and dt parameters need time to adapt to the geometry of the typical set. However after a few warmup iteratons (~300) inference usually speeds up dramatically.
hmc.run(  
    jax.random.key(42),
    Q,
    FWHM,
    R,
    sig_R,
)
hmc.print_summary()
hmc_samps = hmc.get_samples()

# %%

plt.figure()
refplot(Q, R, sig_R, hmc_samps["R"].mean(0), hmc_samps["R"].std(0))
plt.suptitle('Posterior prediction with HMC')