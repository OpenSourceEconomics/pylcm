---
title: Shocks
---

# Shocks

Shock grids represent stochastic variables that define their own transition
probabilities (based on the discretization method). Unlike regular grids, they compute
their own grid points and transition matrices — you don't specify them in
`state_transitions`.

## Import Convention

We recommend importing shock modules and using qualified names:

```python
import lcm.shocks.iid
import lcm.shocks.ar1

# Recommended
shock = lcm.shocks.iid.Normal(
    n_points=5, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0
)

# Not recommended — can cause name collisions (e.g., Normal from scipy)
from lcm.shocks.iid import Normal  # noqa
```

Qualified access makes the shock's origin clear in code review and avoids collisions
with common names like `Normal` from other libraries.

## IID Shocks

Shocks that are independent across periods. Import: `import lcm.shocks.iid`

### Normal

Discretized normal distribution $N(\mu, \sigma^2)$.

```python
lcm.shocks.iid.Normal(n_points=7, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0)
```

**Parameters:**

- `n_points`: Number of grid points.
- `gauss_hermite`: If `True`, use Gauss-Hermite quadrature nodes and weights. If
  `False`, use equally spaced points spanning $\mu \pm n_\text{std} \cdot \sigma$.
- `mu`: Mean of the distribution.
- `sigma`: Standard deviation.
- `n_std`: Number of standard deviations for the grid boundary (not used when
  `gauss_hermite=True`).

### LogNormal

Discretized log-normal distribution where $\ln X \sim N(\mu, \sigma^2)$.

```python
lcm.shocks.iid.LogNormal(n_points=7, gauss_hermite=False, mu=0.0, sigma=0.5, n_std=2.0)
```

Same parameters as `Normal`. Grid points are `exp()` of the underlying normal grid.

### Uniform

Discretized uniform distribution $U(\text{start}, \text{stop})$. Both endpoints are
included in the grid.

```python
lcm.shocks.iid.Uniform(n_points=5, start=0.0, stop=1.0)
```

Equally spaced points with uniform probabilities (all `1/n_points`).

### NormalMixture

Two-component normal mixture:
$\varepsilon \sim p_1 \, N(\mu_1, \sigma_1^2) + (1 - p_1) \, N(\mu_2, \sigma_2^2)$.

```python
lcm.shocks.iid.NormalMixture(
    n_points=9,
    n_std=2.0,
    p1=0.9,
    mu1=0.0,
    sigma1=0.1,
    mu2=0.0,
    sigma2=1.0,
)
```

Grid spans the mixture mean $\pm n_\text{std}$ mixture standard deviations.

## AR(1) Shocks

Shocks with serial correlation. Import: `import lcm.shocks.ar1`

The process is $y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$. The innovation
distribution depends on the method:

- **Tauchen** and **Rouwenhorst**: $\varepsilon_t \sim N(0, \sigma^2)$
- **TauchenNormalMixture**:
  $\varepsilon_t \sim p_1 \, N(\mu_1, \sigma_1^2) + (1 - p_1) \, N(\mu_2, \sigma_2^2)$

### Tauchen

Discretization via @tauchen1986. Uses CDF-based transition probabilities.

```python
lcm.shocks.ar1.Tauchen(
    n_points=7,
    gauss_hermite=False,
    rho=0.9,
    sigma=0.1,
    mu=0.0,
    n_std=2.0,
)
```

- `gauss_hermite`: If `True`, use Gauss-Hermite quadrature nodes.
- `n_std`: Number of unconditional standard deviations for the grid boundary (not used
  when `gauss_hermite=True`).

### Rouwenhorst

Discretization via @rouwenhorst1995 / @kopecky2010. Better for highly persistent
processes ($\rho$ close to 1).

```python
lcm.shocks.ar1.Rouwenhorst(n_points=7, rho=0.95, sigma=0.1, mu=0.0)
```

### TauchenNormalMixture

AR(1) with mixture-of-normals innovations, discretized via Tauchen. Following
@fella2019.

```python
lcm.shocks.ar1.TauchenNormalMixture(
    n_points=9,
    rho=0.9,
    mu=0.0,
    n_std=2.0,
    p1=0.9,
    mu1=0.0,
    sigma1=0.1,
    mu2=0.0,
    sigma2=1.0,
)
```

## Using Shocks in a Regime

Shock grids go in `states`. They must **not** appear in state transitions (they manage
their own):

```python
import lcm.shocks.iid
from lcm import LinSpacedGrid, Regime

working = Regime(
    transition=next_regime,
    states={
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=50),
        "income_shock": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0.0,
            sigma=1.0,
            n_std=2.0,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        # income_shock does NOT appear here — it manages its own transitions
    },
    actions={...},
    functions={
        "utility": utility,
        "earnings": lambda wage, income_shock: wage * jnp.exp(income_shock),
    },
)
```

## Key Rules

1. Shock grids go in `states` (they define the values the shock can take).
1. Shock grids must **not** have a `transition` parameter (validation error if they do).
1. Shock parameters can be specified at init or deferred to runtime (set to `None`).
1. Runtime params follow the same hierarchy as other params (see
   [Parameters](parameters.md)).

## Runtime Parameters

Set shock parameters to `None` at grid creation to supply them at runtime:

```python
lcm.shocks.iid.Normal(n_points=5, gauss_hermite=False, mu=None, sigma=None, n_std=None)
```

Then supply the values in the params dict:

```python
params = {
    "regime_name": {
        "mu": 0.0,
        "sigma": 1.0,
        "n_std": 2.0,
    },
}
```

## See Also

- [Approximating Continuous Shocks](../explanations/approximating_continuous_shocks.ipynb)
  — theory behind Tauchen, Rouwenhorst, and quadrature methods
- [Grids](grids.md) — deterministic grid types
- [Parameters](parameters.md) — how to supply runtime shock parameters
