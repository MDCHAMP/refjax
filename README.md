# Refjax - Lightweight `jit` and `grad` compatible reflectometry in jax

This package provides lightweight, low-level access to neutron reflectometry and spectroscopic ellipsometry kernels in jax.

The provided kernels are all `jit` and `grad` compatible and easy to integrate with your favourite optimisation or uncertainty quantification library for fast parameterisation of stratified reflectometry models. Several examples for popular libraries are available in the `examples` folder.

Current version: 0.1.0.

## Installation

From PyPI:

```bash
pip install refjax
# or
uv add refjax
```

Or from GitHub, pinned to the 0.1.0 release by its tag:

```bash
pip install "refjax @ git+https://github.com/MDCHAMP/refjax@v0.1.0"
# or
uv add git+https://github.com/MDCHAMP/refjax --tag v0.1.0
```

refjax needs Python >= 3.11, `jax` and `jaxlib` >= 0.4.26 and `numpy` >= 2.0. The test suite runs in CI at these minimum versions and at the latest releases.

### 64-bit precision is on by default

`import refjax` switches on jax's 64-bit mode (`jax.config.update("jax_enable_x64", True)`), so refjax computes in float64 and complex128. This setting is global: it applies to all jax code in the same Python process, not only to refjax.

## Neutron reflectometry

### Design pattern

Reflectometry models are created by creating reflectivity kernels with `refjax.kernel`:

```python
kernel = refjax.kernel(model, smear=refjax.smear.no_smear, smear_params=None, jit=True)
```

Currently implemented models:
- `refjax.model.abeles` Abeles matrix formalism for stratified media

Currently implemented smearing kernels:
- `refjax.smear.no_smear` No smearing kernel (default)
- `refjax.smear.constant_gaussian` Constant gaussian smearing at every Q value (the third argument is the FWHM as a percentage of Q)
- `refjax.smear.pointwise_gaussian` Pointwise Gaussian approximation to the true smearing kernel (requires FWHM resolution measurements), by 17th-order Gauss-Hermite quadrature

Once a kernel has been created, it is called with the model parameters and Q (and FWHM values for smeared kernels).

```python
R = kernel(params, Q, FWHM) # fast because jit
```

### Parameters of `abeles`

For a slab model with `n` layers between a fronting and a backing medium, `params` is a dict:

| key     | shape     | meaning                                                                    |
|---------|-----------|----------------------------------------------------------------------------|
| `thick` | `(n,)`    | layer thicknesses, in the length unit of 1/Q (Å for Q in Å⁻¹)                |
| `SLD`   | `(n + 2,)` | scattering length densities in 10⁻⁶ Å⁻², fronting first, backing last     |
| `rough` | `(n + 1,)` | interfacial roughnesses, one per interface                                 |
| `scale` | `()`      | multiplies the reflectivity                                                |
| `bkg`   | `()`      | constant background added to the reflectivity                              |

### Example

```python
import jax.numpy as jnp
import refjax

params = {
    "thick": jnp.array([255.0, 17.0]),
    "SLD": jnp.array([0.0, 6.4, 3.47, 2.07]),
    "rough": jnp.array([5.2, 5.0, 2.07]),
    "scale": 1.0,
    "bkg": 2e-7,
}
kernel = refjax.kernel(refjax.model.abeles, refjax.smear.pointwise_gaussian)
R = kernel(params, Q, FWHM)
```

`examples/01_basic_usage.py` walks through this model against the example data in `examples/.example_data.csv`; the other scripts in `examples` show parameter identification with optax and numpyro.

## Spectroscopic ellipsometry

`refjax.model.spec_ellips(params, lam)` computes the ellipsometric angles Psi and Delta of a stratified medium (transfer-matrix method) at the wavelengths `lam`. It is `jit` and `grad` compatible and is called directly (not through `refjax.kernel`).

For `n` layers between an ambient medium and a substrate, `params` is a dict:

| key        | shape                | meaning                                                                         |
|------------|----------------------|---------------------------------------------------------------------------------|
| `nbar`     | `(n + 2, lam.shape)` | complex refractive index `n + ik` (`k >= 0`) of each medium, ambient first, substrate last |
| `thick`    | `(n,)`               | layer thicknesses, in the same length unit as `lam`                             |
| `aoi_rads` | `()`                 | angle of incidence in the ambient, in radians                                   |

It returns `(psi, delta)` in degrees, each of shape `lam.shape`.

Conventions:
- `tan(psi) exp(i delta) = rp / rs` in Woollam's convention, with `delta` wrapped to `[0, 360)`. Compare `delta` to data modulo 360.
- The complex index is `n + ik` with `k >= 0`. This is the complex conjugate of Woollam's `n - ik`, hence the minus sign on the phase of `rp / rs` in the code. A Woollam-convention index `N = n - ik` is passed as `N.conj()`.
- `n cos(theta)` in each medium follows from Snell's law with the principal square root, which gives `Im >= 0`: the decaying wave for `n + ik`.

```python
import jax.numpy as jnp
import numpy as np
import refjax

lam = jnp.linspace(370.0, 1000.0, 50)  # nm
# ambient, film, SiO2, Si substrate (n + ik), each constant in wavelength here
nbar = jnp.array([1.0, 1.58, 1.46, 3.9 + 0.02j])[:, None] * jnp.ones_like(lam)
params = {"nbar": nbar, "thick": jnp.array([17.0, 2.0]), "aoi_rads": np.radians(75.0)}
psi, delta = refjax.model.spec_ellips(params, lam)
```

`test/test_ellips.py` checks `spec_ellips` against an independent numpy implementation.

### Changes in 0.1.0

`spec_ellips` now follows the conventions above. `delta` is wrapped to `[0, 360)`; before 0.1.0 it was unwrapped along the wavelength axis, so it could differ from the 0.1.0 value by multiples of 360. `n cos(theta)` now comes from the principal square root rather than `arcsin` of Snell's law. Code written against the earlier `spec_ellips` should compare `delta` modulo 360.

## Development

```bash
uv run --extra test pytest
```

## Citing

If you use refjax in published work, please cite:

> Max D. Champneys, Andrew J. Parnell and Stephanie L. Burg, "Towards real-time surrogate-free Bayesian inversion for neutron reflectometry", *Communications Materials* (2026). [doi:10.1038/s43246-026-01341-4](https://doi.org/10.1038/s43246-026-01341-4)

```bibtex
@article{champneys2026refjax,
  author  = {Champneys, Max D. and Parnell, Andrew J. and Burg, Stephanie L.},
  title   = {Towards real-time surrogate-free {B}ayesian inversion for neutron reflectometry},
  journal = {Communications Materials},
  year    = {2026},
  doi     = {10.1038/s43246-026-01341-4},
  url     = {https://doi.org/10.1038/s43246-026-01341-4}
}
```

Please also state the refjax version you used (e.g. 0.1.0). Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button reads it.

Having trouble? Feature request? Raise an issue on the GitHub page.
