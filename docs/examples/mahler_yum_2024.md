---
title: Mahler & Yum (2024)
---

# Mahler & Yum (2024)

Example implementation of the lifecycle model from @mahler2024.

Two regimes (alive/dead). The alive regime has nine states: wealth, health, productivity
shock, lagged effort, adjustment cost, education, productivity type, health type, and
discount type. The dead regime carries discount type only. Three actions: labor supply,
saving, and health effort. Features stochastic health and regime transitions, AR(1)
productivity shocks, and discount-factor heterogeneity. Ships with calibrated data files
for survival probabilities and initial distributions.

::::\{important} This model is computationally intensive. A GPU is recommended; run it
in a CUDA environment (e.g., `pixi run -e cuda13 python your_script.py`). ::::

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/mahler_yum_2024/__init__.py)

## Usage

```python
from lcm_examples.mahler_yum_2024 import (
    MAHLER_YUM_MODEL,
    START_PARAMS,
    create_inputs,
)

model_params, initial_conditions = create_inputs(
    seed=7235,
    n_simulation_subjects=1_000,
    params=START_PARAMS,
)

result = MAHLER_YUM_MODEL.simulate(
    params={"alive": model_params},
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=None,
    log_level="debug",
    seed=8295,
)
```
