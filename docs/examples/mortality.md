---
title: Mortality
---

# Mortality

A three-regime consumption-savings model with stochastic mortality: working life,
retirement, and death. The agent chooses labor supply and consumption. Log utility
with work disutility and a borrowing constraint. Mortality is age-dependent: a vector
of per-period survival probabilities (last entry = 0.0 for certain death) governs the
transition to the dead regime.

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/mortality.py)

## Usage

```python
import jax.numpy as jnp
from lcm_examples.mortality import get_model, get_params

model = get_model(n_periods=5)
params = get_params(n_periods=5)

result = model.solve_and_simulate(
    params=params,
    initial_regimes=["working_life"] * 100,
    initial_states={
        "age": jnp.full(100, model.ages.values[0]),
        "wealth": jnp.linspace(1, 100, 100),
    },
    seed=1234,
)

df = result.to_dataframe(additional_targets="all")
```
