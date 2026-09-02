---
title: Iskhakov et al. (2017)
---

# Iskhakov et al. (2017)

The deterministic retirement model of @iskhakov2017: a worker chooses consumption and
whether to keep working or retire; retirement is absorbing and death arrives
deterministically at a known age. Log utility with work disutility and a borrowing
constraint.

The paper provides a closed-form solution, which pylcm's test suite uses as an
analytical oracle (`tests/data/analytical_solution/`). The discrete retirement choice
makes the value function non-concave and produces the paper's signature saw-tooth
consumption function — see the
[discrete-continuous choice explanation](../explanations/iskhakov_et_al_2017.ipynb) for
figures and the brute-force vs DC-EGM comparison.

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/iskhakov_et_al_2017.py)

## DC-EGM usage

```python
import jax.numpy as jnp
from lcm_examples.iskhakov_et_al_2017 import get_dcegm_model, get_params

model = get_dcegm_model(n_periods=6)
params = get_params(
    n_periods=6,
    discount_factor=0.98,  # the paper's analytical-solution parametrization
    disutility_of_work=1.0,
    interest_rate=0.0,
    wage=20.0,
)

result = model.simulate(
    params=params,
    initial_conditions={
        "age": jnp.full(100, model.ages.values[0]),
        "wealth": jnp.linspace(1, 100, 100),
        "regime_id": jnp.full(100, model.regime_names_to_ids["working_life"]),
    },
    log_level="warning",
)

df = result.to_dataframe(additional_targets="all")
```

The non-terminal regimes are `ConsumptionSavingsRegime` declarations using `DCEGM`. The
solve constructs an off-grid policy, while the standard simulator currently re-decides
consumption on the declared action grid. Read
[the DCEGM simulation contract](../reference/solvers.md#api-dcegm) before comparing
solved and simulated actions or taste-shock frequencies.

## Grid-search comparison

The same economics is available as a labeled grid-search baseline:

```python
from lcm_examples.iskhakov_et_al_2017 import get_model

grid_search_model = get_model(n_periods=6)
```

Use it for reduced-grid diagnostics; it is not the principal solver path on this page.

## Structure

- **Regimes**: `working_life` (labor supply + consumption), `retirement` (absorbing,
  consumption only), `dead` (terminal).
- **States**: wealth (linear grid).
- **Actions**: consumption (continuous), labor supply (work / retire; `working_life`
  only).
- **Transitions**: deterministic — retiring moves the agent to `retirement` forever;
  everyone is `dead` from `final_age_alive + 1` on.
