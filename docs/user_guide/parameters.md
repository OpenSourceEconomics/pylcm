---
title: Parameters
---

# Parameters

Parameters are the numerical inputs to your model — discount factors, wages, risk
aversion coefficients, and so on. pylcm discovers parameters automatically from function
signatures.

## Getting the Parameter Template

```python
template = model.get_params_template()
```

This returns a mutable nested dict showing every parameter the model expects, organized
as `{regime_name: {function_name: {param_name: type_name}}}`. Use it as a starting point
to see what values you need to provide.

## Parameter Structure

Parameters follow a three-level hierarchy. You can specify a parameter at whichever
level is most convenient.

### Function level (most specific)

```python
params = {
    "working": {
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
    "working": {"risk_aversion": 1.5},
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

## Mixing Levels

You can mix levels freely — just avoid ambiguity:

```python
params = {
    "discount_factor": 0.95,            # model level
    "interest_rate": 0.03,              # model level
    "working": {
        "utility": {
            "disutility_of_work": 1.0,  # function level
        },
        "earnings": {"wage": 20.0},     # function level
    },
}
```

## The Ambiguity Rule

A parameter cannot appear at multiple levels within the same subtree. pylcm raises an
error if a parameter value could be resolved from more than one level:

- `"risk_aversion"` at model level **and** `"working" -> "risk_aversion"` at regime level
  = **error** (ambiguous)
- `"risk_aversion"` at regime level **and**
  `"working" -> "utility" -> "risk_aversion"` at function level = **error** (ambiguous)
- `"risk_aversion"` in `"working"` at regime level **and** `"risk_aversion"` in
  `"retired"` at regime level = **OK** (different subtrees)

## Special Parameters

### `discount_factor`

Used by the default aggregation function
$H(\text{utility}, \text{continuation\_value}, \text{discount\_factor}) = \text{utility} + \text{discount\_factor} \cdot \text{continuation\_value}$.

Typically set at model level:

```python
params = {"discount_factor": 0.95}
```

Not needed if you provide a custom `H` function in your regime's `functions` dict.

### Shock parameters

Shock grids with `None` parameters (deferred to runtime) expect their values in the
params dict. They follow the same hierarchy rules. See [Shocks](shocks.md) for details.

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

## What Counts as a Parameter?

pylcm inspects function signatures and classifies each argument:

- **States** (from `states` dict) — not a parameter
- **Actions** (from `actions` dict) — not a parameter
- **Other functions** (from `functions` dict) — not a parameter (resolved via the
  function DAG)
- **Special names** (`continuation_value`) — not a parameter
- **Everything else** — a parameter (must appear in the params dict)

## See Also

- [Parameters Workflow](../explanations/params_workflow.ipynb) — deep dive with error
  examples and edge cases
- [Defining Models](defining_models.md) — the `Model` constructor and `fixed_params`
- [Shocks](shocks.md) — runtime shock parameters
