---
title: Precautionary Savings
---

# Precautionary Savings

A two-regime consumption-savings model with income shocks (alive + dead). Supports IID
shocks (Normal Gauss-Hermite) and persistent AR(1) shocks (Rouwenhorst, Tauchen).
Configurable interest rate and wealth grid type.

With FGP-calibrated parameters, this replicates the simplified benchmark of @fella2019.

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
        "regime_id": jnp.full(100, model.regime_names_to_ids["alive"]),
    },
    log_level="debug",
)

df = result.to_dataframe(additional_targets="all")
```

For `shock_type="normal_gh"`, income is a named current Normal shock, observed before
consumption is chosen. Supply a known initial income explicitly with `PhysicalValues`;
subsequent periods receive fresh IID draws:

```python
from lcm.shocks import PhysicalValues

model = get_model(n_periods=7, shock_type="normal_gh")
params = get_params(shock_type="normal_gh", mu=0.0, sigma=0.1)
result = model.simulate(
    params=params,
    initial_conditions={
        "age": jnp.full(100, 20.0),
        "wealth": jnp.linspace(1, 10, 100),
        "income": PhysicalValues(values=jnp.zeros(100)),
        "regime_id": jnp.full(100, model.regime_names_to_ids["alive"]),
    },
)
```

The IID helper supplies law parameters in the model-level `income_risk` group; the
persistent variants retain their regime-local income-process parameters.
`income_n_points` sets the positive standard-normal Gauss-Hermite base rule for IID
income. The solve integrates current policy values over this rule rather than
interpolating a stored income-state policy surface, so previous finite-grid values and
seeded simulations are not numerical reference results for this route. Income still
enters the next wealth update in its current period.

The separate `validate_initial_conditions` helper currently validates the ordinary
states for this IID example, whose constraint reads only wealth and consumption. It does
not accept the tagged income entry used by `simulate`; checking the same observed income
through that helper remains an API gap.
