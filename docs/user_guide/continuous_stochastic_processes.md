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
1. A shock whose size depends on a discrete state is declared with `StateConditioned`,
   and is then fixed at build time (see
   [State-Conditioned Shock Size](#state-conditioned-shock-size)).

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

(state-conditioned-shock-size)=

## State-Conditioned Shock Size

The size of a shock often depends on where the subject currently is: earnings
innovations are more variable out of work than in it, returns more variable in a
high-volatility regime. Declare that with `StateConditioned` on the process.

```python
from lcm import DiscreteGrid, NormalIIDProcess, StateConditioned, categorical
from lcm.typing import ScalarInt


@categorical(ordered=False)
class EmploymentStatus:
    employed: ScalarInt
    unemployed: ScalarInt


income_shock = NormalIIDProcess(
    n_points=7,
    gauss_hermite=False,
    mu=0.0,
    n_std=3.0,
    sigma=StateConditioned(
        on="employment_status",
        by={"employed": 0.2, "unemployed": 0.5},
    ),
)
```

`on` names a `DiscreteGrid` state the regime carries, and `by` gives the innovation
standard deviation for each of its categories. The regime declares both:

```python
working = Regime(
    transition=next_regime,
    states={
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=50),
        "employment_status": DiscreteGrid(EmploymentStatus),
        "income_shock": income_shock,
    },
    state_transitions={
        "wealth": next_wealth,
        "employment_status": MarkovTransition(next_employment_status),
    },
    actions={...},
    functions={...},
)
```

### The declaration stands where the scalar would

`StateConditioned` is written in place of the parameter it conditions, so which
parameter varies is explicit and there is no way to give that parameter twice.

A discretized process has one axis in the value function, so every category has to share
one set of nodes. Those nodes are placed from the widest value in `by` — the narrowest
axis that still covers every category. The per-category values never move the nodes;
they enter only the transition probabilities. To widen the axis beyond that, raise
`n_std`.

Only `sigma` can be conditioned today, and only for the processes whose transition
probabilities carry it: the CDF-binned `NormalIIDProcess` and `TauchenAR1Process`. A
Rouwenhorst transition depends on `rho` alone, so fixing the nodes would leave a
conditioned `sigma` no channel at all, and the model refuses to build.

### The conditioning value is dated t

Writing $s_t$ for the time-$t$ value of `on` and $\sigma_{s_t}$ for `by[s_t]`, an AR(1)
process transitions as

```{math}
y_{t+1} \mid y_t, s_t \sim N(\mu + \rho y_t,\ \sigma_{s_t}^2),
```

with an IID process dropping the $\rho y_t$ term. The variance of the innovation
realized between $t$ and $t+1$ is therefore set by where the subject is at $t$ — the
employment status they are leaving, not the one they arrive in.

When a process is declared in more than one regime, the values in force are the ones
declared by the regime being *entered*, selected by the conditioning state at $t$. Two
regimes may declare different values on purpose; build the conditioning `DiscreteGrid`
from the same `@categorical` class in each of them, so the categories line up.

### Everything is fixed at build time

A state-conditioned process cannot defer any parameter to runtime, and the values in
`by` never appear in the params template. Both are rejected or absent by design, so
**these values cannot be estimated** — they are part of the model's structure, not its
parameters. Give every parameter at construction.

### Which processes support it

Conditioning rides in the transition CDF, so it is available exactly where `sigma` sits
there:

| Process                                  | Supported                                 |
| ---------------------------------------- | ----------------------------------------- |
| `NormalIIDProcess(gauss_hermite=False)`  | yes                                       |
| `TauchenAR1Process(gauss_hermite=False)` | yes                                       |
| Either with `gauss_hermite=True`         | no — the nodes scale with `sigma`         |
| `RouwenhorstAR1Process`                  | no — its transition depends on `rho` only |

Anything else raises at model build.

## See Also

- [Approximating Continuous Shocks](../explanations/approximating_continuous_shocks.ipynb)
  — theory behind Tauchen, Rouwenhorst, and quadrature methods
- [Grids](grids.md) — deterministic grid types
- [Parameters](parameters.md) — how to supply runtime process parameters
