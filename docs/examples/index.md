---
title: Examples
---

# Examples

Complete example models built with pylcm, ordered from simplest to most complex.
All models live in the `lcm_examples` package and can be imported directly.

1. **[Tiny Consumption-Savings](tiny.md)** — `lcm_examples.tiny`
   Minimal 2-regime model. 3 periods, discrete labor, log-spaced grids,
   tax-and-transfer system.

2. **[Mortality](mortality.md)** — `lcm_examples.mortality`
   3-regime model with death. Discrete labor, borrowing constraint.

3. **[Precautionary Savings](precautionary_savings.md)** — `lcm_examples.precautionary_savings`
   2-regime model with income shocks. IID, Rouwenhorst, and Tauchen shock types.

4. **[Precautionary Savings with Health](precautionary_savings_health.md)** — `lcm_examples.precautionary_savings_health`
   2-regime model with health & exercise. Multiple continuous states and actions,
   auxiliary functions, constraints.

5. **[Mahler & Yum (2024)](mahler_yum_2024.md)** — `lcm_examples.mahler_yum_2024`
   Full Econometrica replication. 8 states, stochastic transitions, data files,
   discount-factor heterogeneity.

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
