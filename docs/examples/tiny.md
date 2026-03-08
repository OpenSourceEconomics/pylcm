---
title: Tiny Consumption-Savings
---

# Tiny Consumption-Savings

A minimal two-regime consumption-savings model with working life and retirement.
Three periods, discrete labor supply, log-spaced consumption grid, and a simple
tax-and-transfer system that guarantees a consumption floor.

This is the simplest complete model in `lcm_examples` and a good starting point for
learning the pylcm API.

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/tiny.py)

## Usage

```python
import jax.numpy as jnp
from lcm_examples.tiny import get_model, get_params

model = get_model()
params = get_params()

result = model.solve_and_simulate(
    params=params,
    initial_regimes=["working_life"] * 100,
    initial_states={
        "age": jnp.full(100, model.ages.values[0]),
        "wealth": jnp.linspace(1, 20, 100),
    },
)

df = result.to_dataframe(additional_targets="all")
```
