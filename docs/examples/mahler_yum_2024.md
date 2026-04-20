---
title: Mahler & Yum (2024)
---

# Mahler & Yum (2024)

Full replication of the lifecycle model from @mahler2024.

Two regimes (alive/dead), 8 states including health, education, productivity type, and
health type. Three actions: labor supply, saving, and health effort. Features stochastic
health and regime transitions, AR(1) productivity shocks, and discount-factor
heterogeneity. Ships with calibrated data files for survival probabilities and initial
distributions.

::::\{important} This model is computationally intensive and requires GPU acceleration.
Run it in a CUDA environment (e.g., `pixi run -e cuda13 python your_script.py`). ::::

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/mahler_yum_2024/_model.py)

## Usage

```python
import jax.numpy as jnp
from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    create_inputs,
)

n_subjects = 1_000

# Build inputs: per-subject initial states include `discount_type`
# (small/large), and `params["discount_factor"]["discount_factor_by_type"]`
# carries the two-element beta array that the `discount_factor` DAG
# function indexes with the state.
common_params, initial_states = create_inputs(
    seed=7235,
    n_simulation_subjects=n_subjects,
    **START_PARAMS,
)

result = MAHLER_YUM_MODEL.simulate(
    params={"alive": common_params},
    initial_conditions={
        **initial_states,
        "regime": jnp.full(
            n_subjects,
            MAHLER_YUM_MODEL.regime_names_to_ids["alive"],
        ),
    },
    period_to_regime_to_V_arr=None,
    seed=8295,
)
```
