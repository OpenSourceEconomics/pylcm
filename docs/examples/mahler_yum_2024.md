---
title: Mahler & Yum (2024)
---

# Mahler & Yum (2024)

Full replication of the lifecycle model from @mahler2024.

Two regimes (alive/dead), 8 states including health, education, productivity type,
and health type. Three actions: labor supply, saving, and health effort. Features
stochastic health and regime transitions, AR(1) productivity shocks, and
discount-factor heterogeneity. Ships with calibrated data files for survival
probabilities and initial distributions.

::::{important}
This model is computationally intensive and requires GPU acceleration. Run it in a CUDA
environment (e.g., `pixi run -e cuda13 python your_script.py`).
::::

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/mahler_yum_2024/_model.py)

## Usage

```python
import jax.numpy as jnp
from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    create_inputs,
)

# Build inputs (params, initial states, discount factor types)
start_params_without_beta = {k: v for k, v in START_PARAMS.items() if k != "beta"}
common_params, initial_states, discount_factor_types = create_inputs(
    seed=7235,
    n_simulation_subjects=1_000,
    **start_params_without_beta,
)

beta_mean = START_PARAMS["beta"]["mean"]
beta_std = START_PARAMS["beta"]["std"]

# Select initial states with high discount factor type
selected_ids_high = jnp.flatnonzero(discount_factor_types)
initial_states_high =   {
                        state: values[selected_ids_high] for state, values
                        in initial_states.items()
                        }

# Solve and simulate for high discount factor type
result = MAHLER_YUM_MODEL.solve_and_simulate(
    params={
        "alive": {
            "discount_factor": beta_mean + beta_std,
            **common_params,
        },
    },
    initial_states=initial_states_high,
    initial_regimes=["alive" for i in range(selected_ids_high.shape[0])],
    seed=8295,
)
```
