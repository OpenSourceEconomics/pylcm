---
title: Parameters
---

# Parameters

Parameters are the numerical inputs to your model — discount factors, wages, risk
aversion coefficients, and so on. pylcm discovers parameters automatically from function
signatures.

Always start from `model.get_params_template()`. Specialized solver declarations,
certainty equivalents, collective value constraints, and per-target transitions add
branches that are easy to miss in a hand-written parameter dictionary.

## Getting the Parameter Template

```python
template = model.get_params_template()
```

This returns a mutable nested dict showing every parameter the model expects, organized
as `{regime_name: {function_name: {param_name: type_name}}}`. Parameters of callables
declared in `Model(edges=...)` sit under one more root, `"edges"`; see
[Edge parameters](#edge-parameters). Use the template as a starting point to see what
values you need to provide.

For profiles indexed by age or period, use explicit temporal declarations and labelled
values; see [Periods and temporal parameters](period_time.md#temporal-parameters).

Only *free* parameters appear in the template — arguments that are states, actions, or
outputs of other functions in the DAG are resolved automatically and do not show up
here.

## Parameter Structure

Parameters follow a three-level hierarchy. You can specify a parameter at whichever
level is most convenient.

### Function level (most specific)

```python
params = {
    "working_life": {
        "utility": {"risk_aversion": 1.5},
        "earnings": {"wage": 20.0},
    },
}
```

### Regime level

A parameter specified at regime level applies to all functions within that regime that
need it:

```python
params = {
    "working_life": {"risk_aversion": 1.5},
}
```

### Model level (most general)

A parameter specified at model level applies everywhere it is needed:

```python
params = {
    "risk_aversion": 1.5,
    "discount_factor": 0.95,
}
```

### All parameters at model level

When every parameter has a single value across all regimes and functions, you can
specify everything at the top level:

```python
params = {
    "discount_factor": 0.95,
    "risk_aversion": 1.5,
    "wage": 20.0,
    "interest_rate": 0.03,
    "consumption_floor": 2.0,
    "tax_rate": 0.2,
    "disutility_of_work": 1.0,
    "last_working_age": 45,
}
```

## Mixing Levels

You can mix levels freely — just avoid ambiguity:

```python
params = {
    "discount_factor": 0.95,  # model level
    "interest_rate": 0.03,  # model level
    "working_life": {
        "utility": {
            "disutility_of_work": 1.0,  # function level
        },
        "earnings": {"wage": 20.0},  # function level
    },
}
```

## The Ambiguity Rule

A parameter cannot appear at multiple levels within the same subtree. pylcm raises an
`InvalidNameError` if a parameter value could be resolved from more than one level:

- `"risk_aversion"` at model level **and** `"working_life" -> "risk_aversion"` at regime
  level = **error** (ambiguous)
- `"risk_aversion"` at regime level **and**
  `"working_life" -> "utility" -> "risk_aversion"` at function level = **error**
  (ambiguous)
- `"risk_aversion"` in `"working_life"` at regime level **and** `"risk_aversion"` in
  `"retirement"` at regime level = **OK** (different subtrees)

(edge-parameters)=

## Edge parameters

The regime-transition law, gates, gate references and route fallbacks are declared in
`Model(edges=...)`, and their parameters live at their declaration path under
`params["edges"]` rather than under a regime:

```python
params = {
    "discount_factor": 0.95,
    "working_life": {"utility": {"disutility_of_work": 1.0}},
    "edges": {"working_life": {"last_working_age": 45}},
}
```

A law over all targets reads `params["edges"][source][arg]`; a per-target cell reads
`params["edges"][source][target][arg]`. Their values resolve from the exact path, from
`params["edges"][source][arg]` (every edge callable of that source), or from the model
level. A value under the regime, `params["working_life"]["last_working_age"]`, feeds
only that regime's own functions and never an edge callable. Per-target state laws
belong to their regime and keep their paths there. The full path table:
[Edge parameter paths](../reference/transitions.md#api-edge-parameters).

## Special Parameters

### `discount_factor`

Used by the default Koopmans aggregator
$W(\text{utility}, \text{CE}, \text{discount\_factor}) = \text{utility} + \text{discount\_factor} \cdot \text{CE}$.

Typically set at model level:

```python
params = {"discount_factor": 0.95}
```

Not needed if you supply a custom `koopmans_aggregator` that takes no `discount_factor`.

### Shock parameters

Shock grids with `None` parameters (deferred to runtime) expect their values in the
params dict. They follow the same hierarchy rules. See
[Continuous stochastic processes](continuous_stochastic_processes.md) for details.

### Fixed parameters

Parameters can be baked into the model at initialization time via `fixed_params`:

```python
model = Model(
    ...,
    fixed_params={"discount_factor": 0.95, "interest_rate": 0.03},
)
```

Fixed parameters are partialled into compiled functions and removed from the template.
You don't need to supply them at `solve()` / `simulate()` time.

**Recommended workflow for estimation:** In estimation contexts, `fixed_params` should
contain everything except the parameters you are estimating. This avoids passing a large
params dict on every likelihood evaluation and makes the estimated parameter set
explicit.

```python
# During estimation, only risk_aversion and disutility_of_work are free
model = Model(
    ...,
    fixed_params={
        "discount_factor": 0.95,
        "interest_rate": 0.03,
        "wage": 20.0,
        "consumption_floor": 2.0,
        "tax_rate": 0.2,
        "last_working_age": 45,
    },
)

# The solve/simulate calls only need the estimated parameters
params = {"risk_aversion": 1.5, "disutility_of_work": 1.0}
result = model.simulate(
    params=params,
    initial_conditions=...,
    log_level="debug",
)
```

## What Counts as a Parameter?

pylcm inspects function signatures and classifies each argument:

- **States** (from `states` dict) — not a parameter
- **Actions** (from `actions` dict) — not a parameter
- **Other functions** (from `functions` dict) — not a parameter (resolved via the
  function DAG)
- **Special names** (`continuation_value`) — not a parameter
- **Everything else** — a parameter (must appear in the params dict)

## See Also

- [Defining Models](defining_models.md) — the `Model` constructor and `fixed_params`
- [Continuous stochastic processes](continuous_stochastic_processes.md) — runtime
  process parameters
