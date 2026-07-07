---
title: Examples
---

# Examples

Complete example models built with pylcm, ordered from simplest to most complex. All
models live in the `lcm_examples` package and can be imported directly.

1. **[Tiny Consumption-Savings](tiny.md)** — `lcm_examples.tiny` Minimal 2-regime model.
   3 periods, discrete labor, log-spaced grids, tax-and-transfer system.

1. **[Mortality](mortality.md)** — `lcm_examples.mortality` 3-regime model with death.
   Discrete labor, borrowing constraint.

1. **[Epstein–Zin Lifecycle](epstein_zin.ipynb)** — `lcm_examples.epstein_zin` 2-regime
   model with health shocks, mortality, and Epstein–Zin preferences.

1. **[Iskhakov et al. (2017)](iskhakov_et_al_2017.md)** —
   `lcm_examples.iskhakov_et_al_2017` Deterministic retirement model with a closed-form
   solution, used as the test suite's analytical oracle. Discrete-continuous choice,
   saw-tooth consumption function.

1. **[Precautionary Savings](precautionary_savings.md)** —
   `lcm_examples.precautionary_savings` 2-regime model with income shocks. IID,
   Rouwenhorst, and Tauchen shock types.

1. **[Precautionary Savings with Health](precautionary_savings_health.md)** —
   `lcm_examples.precautionary_savings_health` 2-regime model with health & exercise.
   Multiple continuous states and actions, auxiliary functions, constraints.

1. **[Mahler & Yum (2024)](mahler_yum_2024.md)** — `lcm_examples.mahler_yum_2024`
   Econometrica lifecycle model. 9 states, stochastic transitions, data files,
   discount-factor heterogeneity. GPU recommended.

## Customizing models

All models export regime objects that can be customized via `.replace()`:

```python
from lcm import LinSpacedGrid
from lcm_examples.mortality import working_life

# Use a finer wealth grid
custom_regime = working_life.replace(
    states={"wealth": LinSpacedGrid(start=1, stop=1000, n_points=500)},
)
```

See [Regimes](../user_guide/regimes.ipynb) for more on `.replace()`.
