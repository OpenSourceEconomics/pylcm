---
title: Precautionary Savings with Health
---

# Precautionary Savings with Health

A two-regime consumption-savings model with health and exercise. The agent chooses
whether to work, how much to consume, and how much to exercise. Two continuous states
(wealth, health) evolve over time. At retirement age the agent enters a terminal regime
where utility depends only on remaining wealth and health.

This model demonstrates multiple continuous states, multiple continuous actions,
auxiliary functions (wage, labor income), constraints, and regime transitions.

:::{note}
The parameterization is chosen to showcase pylcm's features, not to match any empirical
calibration.
:::

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/precautionary_savings_health.py)

## Usage

```python
import jax.numpy as jnp
from lcm_examples.precautionary_savings_health import get_model, get_params

model = get_model()
params = get_params()

result = model.simulate(
    params=params,
    initial_conditions={
        "age": jnp.full(1_000, model.ages.values[0]),
        "wealth": jnp.full(1_000, 1.0),
        "health": jnp.full(1_000, 1.0),
        "regime": jnp.full(1_000, model.regime_names_to_ids["working_life"]),
    },
    period_to_regime_to_V_arr=None,
)

df = result.to_dataframe(additional_targets="all")
```

## See Also

- [A Tiny Example](../user_guide/tiny_example.ipynb) — simpler three-period model
- [Writing Economics](../user_guide/write_economics.ipynb) — function DAGs and regime
  design
- [Grids](../user_guide/grids.md) — grid types and transitions
- [Parameters](../user_guide/parameters.md) — constructing the params dict
