# Refjax - Lightweight `jit` and `grad` compatible reflectometry in jax

This package provides lightweight, low-level access to neutron reflectometry kernels in jax.

The provided kernels are all `jit` and `grad` compatible and easy to integrate with your favourite optimisation or uncertainty quantification library for fast parameterisation of stratified reflectometry models. Several examples for popular libraries are available in the `examples` folder.

### Design pattern

Reflectometry models are created by creating reflectivity kernels. The call to the `refjax.kernel` function takes three arguments.

```python
kernel = refjax.kernel(model, smearing, jit=True)
```

Currently implemented models:
- `refjax.models.abeles` Abeles matrix formalism for stratified media

Currently implemented smearing kernels
- `refjax.smear.no_smear` No smearing kernel (default)
- `refjax.smear.constant_gaussian` Constant gaussian smearing at every Q value
- `refjax.smear.pointwise_gaussian` Pointwise Gaussian approximation to the true smearing kernel (requires FWHM resolution measurements)

Once a kernel has been created, it is called with the model parameters and Q (and FWHM values for smeared kernels).

```python
R = kernel(params, Q, FWHM) # fast beacuse jit
```

Check out the `examples` folder for more advanced usage and examples of parameter identification schemes. 

Having trouble? Feature request? Raise an issue on the GitHub page.