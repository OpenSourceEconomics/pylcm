---
title: Mahler & Yum (2024)
---

# Mahler & Yum (2024)

Example implementation of the lifecycle model from @mahler2024.

Three regimes represent working life, mandatory retirement from age 65, and death. The
working regime has nine states: wealth, health, productivity shock, lagged effort,
education, productivity type, health type, and discount type, plus phase-specific
derived quantities. Its continuous effort and consumption choices form a nested
EGM-family problem; the paper implementation declares working and retirement as
`NestedConsumptionSavingsRegime` objects. The model features stochastic health and
regime transitions, AR(1) productivity shocks, and discount-factor heterogeneity. It
ships with calibrated data files for survival probabilities and initial distributions.

::::\{important} This model is computationally intensive. A GPU is recommended; run it
in a CUDA environment (for example, `pixi run -e cuda13 python your_script.py`). ::::

[View paper-mode source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/mahler_yum_2024/paper.py)

## Paper-mode usage

```python
from lcm_examples.mahler_yum_2024 import START_PARAMS, create_inputs
from lcm_examples.mahler_yum_2024.paper import (
    adapt_params_to_paper_mode,
    create_mahler_yum_model,
)

model = create_mahler_yum_model(implementation="paper")
model_params, initial_conditions = create_inputs(
    seed=7235,
    n_simulation_subjects=1_000,
    params=START_PARAMS,
)
model_params = adapt_params_to_paper_mode(model_params)

result = model.simulate(
    params=model_params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=None,
    log_level="debug",
    seed=8295,
)
```

This is the principal economic implementation. It uses a continuous outer effort margin
and an inner non-convex-budget EGM solve. See
[Nested endogenous-grid methods](../methods/nested_egm.md) and
[Declared non-convex budgets](../methods/nonconvex_budgets.md) for the two layers.

## Historical grid-search baseline

`MAHLER_YUM_MODEL` remains available for labeled brute-force and historical comparisons:

```python
from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL

grid_search_baseline = MAHLER_YUM_MODEL
```

Do not substitute this baseline when the paper-mode specialized declaration is the
object being studied.
