---
title: Continuous stochastic processes
---

# Continuous stochastic processes

A **continuous stochastic process** is a stochastic state variable whose class bundles
both the discretized grid and its transition mechanism. Unlike ordinary grids, a process
computes its own grid points and transition matrix from a distribution and its
parameters — so you place it in `states` and never in `state_transitions`.

Process classes follow the naming convention `<Distribution><Kind>Process` and are
imported directly from `lcm`:

```python
from lcm import NormalIIDProcess, TauchenAR1Process
```

- `*IIDProcess` — independent draws each period.
- `*AR1Process` — an AR(1) process with a chosen discretization scheme.

## IID Processes

Processes whose draws are independent across periods.

### NormalIIDProcess

Discretized normal distribution $N(\mu, \sigma^2)$.

```python
NormalIIDProcess(n_points=7, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0)
```

**Parameters:**

- `n_points`: Number of grid points.
- `gauss_hermite`: If `True`, use Gauss-Hermite quadrature nodes and weights. If
  `False`, use equally spaced points spanning $\mu \pm n_\text{std} \cdot \sigma$.
- `mu`: Mean of the distribution.
- `sigma`: Standard deviation.
- `n_std`: Number of standard deviations for the grid boundary. Mutually exclusive with
  `gauss_hermite=True`.

### LogNormalIIDProcess

Discretized log-normal distribution where $\ln X \sim N(\mu, \sigma^2)$.

```python
LogNormalIIDProcess(n_points=7, gauss_hermite=False, mu=0.0, sigma=0.5, n_std=2.0)
```

Same parameters as `NormalIIDProcess`. Grid points are `exp()` of the underlying normal
grid.

### UniformIIDProcess

Discretized uniform distribution $U(\text{start}, \text{stop})$. Both endpoints are
included in the grid.

```python
UniformIIDProcess(n_points=5, start=0.0, stop=1.0)
```

Equally spaced points with uniform probabilities (all `1/n_points`).

### NormalMixtureIIDProcess

Two-component normal mixture:
$\varepsilon \sim p_1 \, N(\mu_1, \sigma_1^2) + (1 - p_1) \, N(\mu_2, \sigma_2^2)$.

```python
NormalMixtureIIDProcess(
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

## AR(1) Processes

Processes with serial correlation. The process is
$y_t = \mu + \rho \, y_{t-1} + \varepsilon_t$. The innovation distribution depends on
the class:

- `TauchenAR1Process` and `RouwenhorstAR1Process`: $\varepsilon_t \sim N(0, \sigma^2)$
- `TauchenNormalMixtureAR1Process`:
  $\varepsilon_t \sim p_1 \, N(\mu_1, \sigma_1^2) + (1 - p_1) \, N(\mu_2, \sigma_2^2)$

### TauchenAR1Process

Discretization via @tauchen1986. Uses CDF-based transition probabilities.

```python
TauchenAR1Process(
    n_points=7,
    gauss_hermite=False,
    rho=0.9,
    sigma=0.1,
    mu=0.0,
    n_std=2.0,
)
```

- `gauss_hermite`: If `True`, use Gauss-Hermite quadrature nodes.
- `n_std`: Number of unconditional standard deviations for the grid boundary. Mutually
  exclusive with `gauss_hermite=True`.

### RouwenhorstAR1Process

Discretization via @rouwenhorst1995 / @kopecky2010. Better for highly persistent
processes ($\rho$ close to 1).

```python
RouwenhorstAR1Process(n_points=7, rho=0.95, sigma=0.1, mu=0.0)
```

### TauchenNormalMixtureAR1Process

AR(1) with mixture-of-normals innovations, discretized via Tauchen. Following
@fella2019.

```python
TauchenNormalMixtureAR1Process(
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

## Using a Continuous Stochastic Process in a Regime

A process goes in `states`. It must **not** appear in `state_transitions` — it manages
its own transition:

```python
from lcm import LinSpacedGrid, NormalIIDProcess, Regime

working = Regime(
    transition=next_regime,
    states={
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=50),
        "income_shock": NormalIIDProcess(
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

1. A process goes in `states` — it defines the values the shock can take.
1. A process must **not** appear in `state_transitions` — placing it there is a
   validation error.
1. Process parameters can be specified at construction or deferred to runtime (set to
   `None`).
1. Runtime params follow the same hierarchy as other params (see
   [Parameters](parameters.md)).

## Runtime Parameters

Set distribution parameters to `None` at construction to supply them at runtime:

```python
NormalIIDProcess(n_points=5, gauss_hermite=False, mu=None, sigma=None, n_std=None)
```

Then supply the values in the params dict, keyed by regime name:

```python
params = {
    "regime_name": {
        "mu": 0.0,
        "sigma": 1.0,
        "n_std": 2.0,
    },
}
```

`n_points` and `gauss_hermite` are structural, not distribution parameters — they must
always be given at construction.

## See Also

- [Approximating Continuous Shocks](../explanations/approximating_continuous_shocks.ipynb)
  — theory behind Tauchen, Rouwenhorst, and quadrature methods
- [Grids](grids.md) — deterministic grid types
- [Parameters](parameters.md) — how to supply runtime process parameters
