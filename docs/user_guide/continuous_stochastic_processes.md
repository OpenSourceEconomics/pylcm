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

> **Check solver support before adding the shock.** A process can enlarge the
> continuation-node product or change stored value-array topology when `fold=True`.
> EGM-family and collective regimes support narrower combinations than `GridSearch`;
> build the intended specialized regime early and let model validation check it.

A shock drawn and used within one period can instead be declared as a named `ShockGroup`
(`Model(shocks=...)`, `Regime(shocks=...)`). In an endogenous-grid regime such a group
is solved as the IID node states described here, while its published values integrate
the nodes; see
[Named current-IID shocks in endogenous-grid regimes](../reference/solvers.md#current-iid-egm).

## Simulation draws

Continuous process simulation defaults to `simulate_draws="continuous"`: draw from its
distribution rather than from the discretization. Set `simulate_draws="nodes"` when
simulation must stay on the solve support:

```python
NormalIIDProcess(
    n_points=7, gauss_hermite=True, mu=0.0, sigma=1.0, simulate_draws="nodes"
)
```

Node mode samples the exact resolved physical solve nodes with the IID weights or
current Markov transition row. It works with fixed and runtime parameters, supplied
Tauchen nodes and ordinary folded IID declarations. It changes forward realizations; it
leaves the solve grid and expectations alone. Transition-local shock lotteries already
sample their declared support and retain that behavior in either mode.

To solve under one process law and simulate a different continuous law, declare the
process state with public `Phased`:

```python
from lcm import Phased

income = Phased(
    solve=TauchenAR1Process(
        n_points=7, gauss_hermite=False, n_std=2.0, mu=0.0, rho=0.6, sigma=0.2
    ),
    simulate=TauchenAR1Process(
        n_points=7, gauss_hermite=False, n_std=2.0, mu=0.0, rho=0.8, sigma=0.4
    ),
)
```

Put `income` in `states`, with no ordinary `state_transitions` entry. Values and
continuation decisions use the solve law; realized shocks use the simulation law.
Constructor parameters can hold separate estimates. A parameter left `None` in both
variants uses the same runtime parameter name and value. Process/non-process pairs,
folded phase pairs, and distinct simulation laws with node draws are rejected at
construction. Ordinary node simulation is a bare process declaration.

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

With `gauss_hermite=True`, positive even and odd `n_points` are supported. The examples
use odd counts, which include a node at the mean; even counts have no mean node.

### JointNormalIIDProcess

Declare correlated Gaussian shocks together under one parameter namespace:

```python
from lcm import JointNormalIIDProcess, Regime

working = Regime(
    states={
        "wages": JointNormalIIDProcess(
            states=("eps_husband", "eps_wife"),
            n_points=(7, 5),
            gauss_hermite=True,
            mean=(0.0, 0.0),
            cov=None,
        ),
    },
    functions={"utility": utility},
)
params = {"working": {"wages": {"cov": [[1.0, 0.4], [0.4, 2.0]]}}}
```

The scalar names in `states` are the physical shocks your DAG functions consume. Initial
observations and simulation dataframe columns use those names too. A scalar `n_points`
applies to every dimension; a tuple supplies counts in the named order. `mean` and `cov`
can be fixed on the declaration or supplied through its parameter namespace. The
covariance must be finite, symmetric and positive semidefinite; invalid runtime
covariance identifies the process and regime even with runtime checks disabled. Singular
covariance, including zero variance, is supported; initial observations must lie on its
support.

Quadrature forms the product of independent standard-normal rules and applies a
lower-triangular covariance factor. Simulation draws continuous normal coordinates and
applies the same factor. With `gauss_hermite=False`, supply positive `n_std`: the scalar
rules use CDF bins. This declaration belongs directly in `Regime.states`; it is not a
state-conditioned law or a replacement for `JointTransition`.

The internal Cartesian value axes are named `<group>_normal_<dimension>` and carry
standard-normal coordinates. Physical correlated support is not Cartesian, so these axes
differ from the physical names in the DAG and simulation. `fold=True` folds every
internal coordinate and inherits the scalar IID solver and timing restrictions. It
retains physical shocks in simulation output.

Initial-population recipes can inspect `model.initial_state_names(regime_name=...)` to
obtain physical scalar names, including folded and carried states. For a continuous
state,
`model.initial_state_bounds(params=..., regime_name=..., state_name=..., period=...)`
returns its marginal grid extents using the supplied runtime law. Supply the zero-based
period for specialized grids. These intervals describe the discretized support; they
neither form a Cartesian correlated grid nor truncate continuous draws. `state_names`
and `state_grid` continue to describe value-array axes.

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

Gauss-Hermite supports even counts here too. Only odd counts include the unconditional
mean. Transition probabilities still use midpoint CDF bins, rather than quadrature
weights.

### Tauchen on supplied nodes

Use `TauchenAR1Process.on_nodes` when the physical state grid is supplied by a
calibration, or the process has a unit root and cannot place a stationary grid:

```python
TauchenAR1Process.on_nodes(
    nodes=jnp.array([-1.2, -0.5, 0.0, 0.5, 1.2]),
    rho=None,
    sigma=None,
)
```

The state takes these physical values directly. Nodes must contain at least two finite,
strictly increasing values that remain distinct at the working precision. Transition
probabilities integrate the conditional normal with mean `mu + rho * y` over bins
separated by adjacent midpoints; the first and last bins include the tails. `mu`
defaults to zero. Set `rho`, `sigma`, or `mu` to `None` to supply that law parameter at
solve or simulation time. The nodes remain fixed as parameters vary, and `rho=1` is
supported. Simulation defaults to the continuous AR(1) law, as for ordinary Tauchen. Set
`simulate_draws="nodes"` to sample these physical nodes with the transition rows.

This factory does not reproduce a row-normalized density-at-nodes rule. Use an explicit
categorical transition when that discrete law is the intended model.

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

## Folding an IID shock out of stored values

An IID process used as a state may declare `fold=True`:

```python
income_shock = NormalIIDProcess(
    n_points=7,
    gauss_hermite=True,
    mu=0.0,
    sigma=1.0,
    fold=True,
)
```

### Shocks observed before the within-period decision

Folding removes the process axis from each stored value function; it does **not** remove
the shock from the economic model. The solve evaluates every shock node, evaluates the
feasible action choice at that node, and only then averages the resulting values with
the process's own quadrature weights. Consequently a folded period stores one fewer
axis, but utility and the chosen action may still depend on the realized shock.

For an eligible JIT `GridSearch` fold, pylcm evaluates the action product in bounded
blocks at each shock node before applying that same quadrature. The fold-node axis
itself is still evaluated and reduced in full: this route changes only action
evaluation, not the stochastic rule or reduction order. Runtime and peak-memory effects
require paired measurement.

Forward simulation is unchanged economically. A subject entering from another regime
draws the shock on that transition; a subject whose initial regime is the folding regime
must seed the process state in `initial_conditions`. Subsequent IID transitions redraw
it, and the realized value remains available to utility, policy, and the output table.
With the same seed, a folded model and its otherwise identical unfolded model produce
the same simulated panel; only value-function storage differs.

This is a narrow exact reduction. Model construction enforces all of the following:

- the declaration is an IID process state; persistent processes have no `fold` field;
- the regime is singleton and uses `GridSearch`;
- every distribution parameter is fixed when the process is constructed, because fold
  weights are built with the kernels rather than read from runtime params;
- continuation uses an absent or exact `LinearExpectation`, excluding nonlinear
  certainty equivalents and custom subclasses;
- the regime has no EV1 taste shocks;
- no next-state or next-regime transition, including a transitive helper dependency,
  reads the folded shock's realized node;
- no same-period value-dependent predicate or projection owned by the folding regime
  reads the folded realization;
- the folding regime itself is not named as a gated-edge target, route fallback,
  same-period reference, or gate reference. This endpoint ban is unconditional: those
  readers need its per-node value even when their projections do not name the folded
  shock.

Those rules describe a shock observed before the within-period decision and averaged
after optimizing at each realization. If that realization changes a later state, selects
a regime, or is needed by a value-dependent endpoint, retain the ordinary unfolded axis.

A folded `StateConditioned` IID shock has one further timing restriction. Within a
non-terminal folding regime, every declared law for the conditioner must be
`fixed_transition(conditioner_name)`; a terminal folding regime has no local law.
Additionally, every structurally reachable incoming source's target-local cell **toward
the folding regime** must be fixed. Such an incoming source may move the conditioner
toward a different target. The shock entering period $t$ was drawn using the category at
$t-1$, whereas the fold gathers a quadrature row along the category visible at $t$;
those rows are the same only when the conditioner cannot move into the fold. A model
that needs a moving conditioner there must use `fold=False`.

### Shocks realized after choices and used only by a transition

A shock realized after the period's choices, such as a bill taken off end-of-period
wealth, is no state: nothing chosen in the period can depend on it. Declare it as a
`ShockGroup` in `Model(shocks=...)` and list the group in the regime's
`post_choice_shocks`:

```python
from lcm import Regime
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup


def medical_bill(next_xi):
    return 0.4 * jnp.exp(0.5 * next_xi)


def next_wealth(*, wealth, consumption, medical_bill):
    return wealth - consumption - medical_bill


shocks = {
    "bill_shock": ShockGroup(
        shocks={"xi": Normal(mu=0.0, sigma=1.0)},
        nodes=Product(xi=GaussHermite(5)),
    )
}
alive = Regime(
    states={"wealth": wealth_grid},
    actions={"consumption": consumption_grid},
    state_transitions={"wealth": next_wealth},
    functions={"utility": utility, "medical_bill": medical_bill},
    post_choice_shocks=("bill_shock",),
)
```

Pass `shocks` to the model alongside its regimes. Only the regime's state transitions
read a post-choice shock, as `next_<shock>`, directly or through functions that only
they use. Utility, constraints and the regime transition are evaluated before the draw;
a read of the shock from any of them, under either spelling, is refused when the model
is built, naming the function and the shock. The regime's value carries no axis for the
shock, and no initial value is required for it.

The solve takes the expectation over the group's nodes and weights inside the
continuation, so the draw may change a next state. Nonlinear certainty equivalents apply
their transform to each realized continuation value before averaging. Simulation draws
one node per subject and period from the same weights; `shock_paths` accepts a plain
array of physical values for the shock, dated at the destination period.

A post-choice group is lowered to one fixed node set per shock. Its laws are continuous
with fixed numeric parameters, its shocks independent, and its rule deterministic and
unsigned: `Product` of scalar rules, or `UserNodes` and `Smolyak` for a one-shock group.
Runtime or state-dependent law parameters, `Discrete` laws, correlation and sampled
(`MonteCarlo`, `QuasiMonteCarlo`) rules are refused when the model is built. A group is
either current or post-choice across the whole model, never both. Only `GridSearch`
regimes declare post-choice shocks.

### Precomputing continuation values for an additive transition

When such a shock is added to a continuous state, declare the law as
`lcm.AdditiveShockTransition(base=..., shock=..., conditioners=...)` instead of a
function computing `base + shock`. The law is `next_<state> = base + shock`, so a bill
enters as its negative:

```python
def negative_medical_bill(medical_bill):
    return -medical_bill


state_transitions = {
    "wealth": lcm.AdditiveShockTransition(
        base="wealth_before_bill",
        shock="negative_medical_bill",
        conditioners={"is_insured": (False, True)},
    ),
}
```

With an exact `LinearExpectation` (including the model's default), the continuation
precomputes a shock-averaged target table once per period,
`W(z) = sum_k w_k V(z + shock_k)`, and reuses it across state–choice combinations at
`z = wealth_before_bill`. This saves repeated interpolation at every shock draw for
every source point. The additive formula makes this precomputation possible.

The averaged draws are those the transition alone makes with probabilities that read
only parameters and time. A draw the target stores, or one whose probabilities depend on
the source point, keeps its node axis. With linear interpolation the value is linear
between its grid points `a_j`, so `W` is linear between the merged points
`{a_j - shock_k}` and is stored there: the result is exact up to rounding, not an
approximation on the target's grid. The base may not read a draw, and the shock may read
nothing but its conditioners, the draws, the period or age, and parameters. Each
conditioner is a function or discrete variable of the source with the finite support
given; declared conditioners the shock does not read are ignored.

Precomputation admits the exact `LinSpacedGrid` and `IrregSpacedGrid` classes for the
target's shocked state. Their interpolation and extrapolation are linear in the physical
state between the represented knots and along the endpoint segments. Other grids,
including `LogSpacedGrid`, piecewise grids and custom subclasses, retain ordinary
outcome enumeration. This preserves their existing coordinate semantics, including
extrapolation outside the target grid.

Eligibility uses the exact `LinearExpectation` class, rather than its subclasses.
`PowerMean`, `QuasiArithmeticMean`, and custom certainty equivalents retain ordinary
outcome enumeration, including subclasses of `LinearExpectation` that override its
meaning. The same boundary applies to certainty equivalents supplied on the model or
regime and survives archiving models built with these rules. Model archives preserve
their economic closures and continuation logic; rebuild from declarations when adopting
different solver numerics. The additive declaration's dependency checks still run before
fallback; an invalid decomposition raises a construction error. On the linear averaging
route, gated, collective, entered or factored continuations are unsupported and raise a
construction error; use a plain `base + shock` function there.

The merged-point table and interpolation weights require temporary storage that grows
with the target grid, integrated shock support, conditioners and retained draws. They
can cost more memory than direct enumeration on small source grids; the runtime and
peak-memory benefit require a paired comparison of otherwise equivalent models.

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
    states={
        "wealth": LinSpacedGrid(start=0, stop=100, n_points=50),
        "employment_status": DiscreteGrid(category_class=EmploymentStatus),
        "income_shock": income_shock,
    },
    state_transitions={
        "wealth": next_wealth,
        "employment_status": StochasticTransition(func=next_employment_status),
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
