import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import refjax
from refjax.model import abeles
from refjax.smear import pointwise_gaussian

import jax
import jax.numpy as jnp
import optax

# %% 

# In this example we will be using gradient descent to learn parameters in the slab model from the previous example

data = pd.read_csv(".example_data.csv")

ax = sns.lineplot(data, x="Q", y="R")
ax.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)
plt.yscale("log")
plt.xscale("log")
plt.legend(['Data', '_', '$\pm 3 \sigma$'])
plt.show()

# Slab model
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
#                      scale=~1.0 *
#                      bkg=2e-7

# Recall that parameters marked with * are not known exactly. Our objective will be to learn these parameters automatically by performing gradient descent w.r.t the unknown parameters on a suitable error function.


# %% 
# Model setup, here we will build several functions to peform the gradient descent

# Define refjax kernel, as before we will use a pointwise Gaussian smearing kernel
kernel = refjax.kernel(abeles, pointwise_gaussian)

# Because we only want to update a subset of the parameters in our model, we need a function that return all the parameters (for refjax) given only the subset we want to 
def params_from_theta(theta):
    thick, sld, rough, scale = theta
    return {
        "thick": jnp.array([thick, 17.0]),
        "SLD": jnp.array([0.0, sld, 3.47, 2.07]),
        "rough": jnp.array([rough, 5.0, 2.07]),
        "scale": scale,
        "bkg": 2e-7,
    }

# define loss function here we are using an L2 loss over the log reflectivity (in order to avoid biasing the model towards low Q values where R has a much larger value).
def L2_error_log(theta):
    R_hat = kernel(params_from_theta(theta), jnp.array(data.Q), jnp.array(data.FWHM))
    return ((jnp.log(jnp.array(data.R)) - jnp.log(R_hat)) ** 2).sum()

# %% 
# Gradient Descent

# Our initial guess of the unknown parameters: thick, sld, rough, scale
theta0 = jnp.array([240.0, 7.0, 4.0, 1.0])

# Initialise an optax optimiser for gradient descent
optimiser=optax.adam(learning_rate=1e-1)
opt_state = optimiser.init(theta0)

# This function performs one step of the gradient descent and returns the parametrs, optimiser state and the current valaue of the loss funciton. 
@jax.jit
def step(params, opt_state):
    loss_value, grads = jax.value_and_grad(L2_error_log)(params) # Gradient of L2 loss w.r.t parameters 
    updates, opt_state = optimiser.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

# Now we will run our gradient descent in a loop and keep the final value of theta
theta = theta0
losses = [] # keep track of loss value to track model convergence.
for i in range(1000):
    theta, opt_state, loss_value = step(theta, opt_state)
    losses.append(loss_value)
    if i % 100 == 0:
        print(f"step {i}, loss: {loss_value}")



# print and plot

# Plotting loss convergence demonstrates the GSD has indeed converged.
plt.figure()
plt.semilogy(losses)
plt.show()

# Comparing loss vlaues from the old and optimised models shows a good improvement
print(L2_error_log(theta0))
print(L2_error_log(theta))

# Print the optimised parameters
print(theta)

# Looking at the gradient of the loss is another good way to test convergence. All the gradients are tiny - indicating we have coinverged.
print(jax.grad(L2_error_log)(theta))


# Plot comparing the old and new models - a great improvemnt.
plt.figure()
ax = sns.lineplot(data, x="Q", y="R")
ax.fill_between(data.Q, data.R - 3 * data.sig_R, data.R + 3 * data.sig_R, alpha=0.2)

plt.plot(
    data.Q, kernel(params_from_theta(theta0), jnp.array(data.Q), jnp.array(data.FWHM))
)
plt.plot(
    data.Q, kernel(params_from_theta(theta), jnp.array(data.Q), jnp.array(data.FWHM))
)

plt.yscale("log")
plt.xscale("log")
plt.legend(['Data', '_', '$\pm 3 \sigma$', 'Old model', 'Optimised model'])
