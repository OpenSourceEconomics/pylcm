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
import numpy as np
import pandas as pd
from lcm import initial_conditions_from_dataframe
from lcm_examples.tiny import get_model, get_params

model = get_model()
params = get_params()

initial_df = pd.DataFrame({
    "regime": "working_life",
    "age": model.ages.values[0],
    "wealth": np.linspace(1, 20, 100),
})

initial_conditions = initial_conditions_from_dataframe(initial_df, model=model)

result = model.solve_and_simulate(
    params=params,
    initial_conditions=initial_conditions,
)

df = result.to_dataframe(additional_targets="all")
```
