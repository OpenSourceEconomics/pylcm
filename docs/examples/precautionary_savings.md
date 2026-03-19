---
title: Precautionary Savings
---

# Precautionary Savings

A two-regime consumption-savings model with income shocks (alive + dead). Supports
IID shocks (Normal Gauss-Hermite) and persistent AR(1) shocks (Rouwenhorst, Tauchen).
Configurable interest rate and wealth grid type.

With FGP-calibrated parameters, this replicates the simplified benchmark of
@fella2019.

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/precautionary_savings.py)

## Usage

```python
import jax.numpy as jnp
from lcm_examples.precautionary_savings import get_model, get_params

model = get_model(n_periods=7, shock_type="rouwenhorst")
params = get_params(shock_type="rouwenhorst", sigma=0.1, rho=0.95)

result = model.simulate(
    params=params,
    initial_conditions={
        "age": jnp.full(100, model.ages.values[0]),
        "wealth": jnp.linspace(1, 10, 100),
        "income": jnp.zeros(100),
        "regime": jnp.full(100, model.regime_names_to_ids["alive"]),
    },
    V_arr_dict=None,
)

df = result.to_dataframe(additional_targets="all")
```
