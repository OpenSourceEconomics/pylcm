"""Generate the DC-EGM cross-check fixture from the `dcegm` package.

Not a test and not runnable in any pylcm environment — `dcegm` is deliberately
not a dependency. See README.md for the throwaway-venv recipe and the pinned
upstream commit.

Model: the Iskhakov et al. (2017) consumption-retirement toy model shipped with
`dcegm` (`toy_models/cons_ret_model_dcegm_paper`), parametrized with
deterministic income and EV1 taste shocks:

- choices: `0 = work`, `1 = retire`; retirement is absorbing. (The upstream
  docstrings state the opposite coding; the code — `(1 - choice) * delta`
  disutility and `(1 - lagged_choice) * income` — says 0 = work. Trust the
  code.)
- utility: `(c**(1 - rho) - 1) / (1 - rho) - delta * (1 - choice)`
- income, paid one period after working (depends on the lagged choice) and
  indexed by the receiving period's age (`age = period + min_age`):
  `exp(constant + exp * age + exp_squared * age**2)`
- budget: `wealth' = max(income(lagged_choice, age') + (1 + r) * savings,
  consumption_floor)`
- final period: consume everything; the work/retire choice still carries the
  taste shock and the work disutility (with no income benefit).

The CSV holds, per (period, lagged_choice, wealth): choice-specific values and
consumption policies and the probability of working. Rows with
`lagged_choice = 1` leave the work columns empty (infeasible choice). The
smoothed value is derivable as
`scale * logsumexp([value_work, value_retire] / scale)`.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"

import dcegm  # ty: ignore[unresolved-import]
import dcegm.toy_models as tm  # ty: ignore[unresolved-import]
import numpy as np
import pandas as pd

PARAMS = {
    "discount_factor": 0.95,
    "delta": 0.35,
    "rho": 1.95,
    "constant": 0.75,
    "exp": 0.04,
    "exp_squared": -0.0002,
    "income_shock_std": 0.0,
    "income_shock_mean": 0.0,
    "taste_shock_scale": 0.2,
    "interest_rate": 0.05,
    "consumption_floor": 0.001,
}
N_PERIODS = 10
WEALTH_POINTS = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 35.0])

model_config = {
    "n_periods": N_PERIODS,
    "choices": [0, 1],
    "continuous_states": {"assets_end_of_period": np.linspace(0.0, 50.0, 500)},
    "n_quad_points": 5,
}
model_specs = {"n_periods": N_PERIODS, "min_age": 20, "n_choices": 2}

funcs = tm.load_example_model_functions("dcegm_paper")
model = dcegm.setup_model(
    model_config=model_config,
    model_specs=model_specs,
    utility_functions=funcs["utility_functions"],
    utility_functions_final_period=funcs["utility_functions_final_period"],
    budget_constraint=funcs["budget_constraint"],
    state_space_functions=funcs["state_space_functions"],
)
solved = model.solve(PARAMS)

rows = []
for lagged_choice in (0, 1):
    for period in range(N_PERIODS - 1):
        n = WEALTH_POINTS.size
        states = {
            "period": np.full(n, period, dtype=int),
            "lagged_choice": np.full(n, lagged_choice, dtype=int),
            "assets_begin_of_period": WEALTH_POINTS,
        }
        values = np.asarray(solved.choice_values_for_states(states))
        policies = np.asarray(solved.choice_policies_for_states(states))
        probs = np.asarray(solved.choice_probabilities_for_states(states))
        for i, wealth in enumerate(WEALTH_POINTS):
            rows.append(
                {
                    "period": period,
                    "lagged_choice": lagged_choice,
                    "wealth": wealth,
                    "value_work": float(values[i, 0]),
                    "value_retire": float(values[i, 1]),
                    "policy_work": float(policies[i, 0]),
                    "policy_retire": float(policies[i, 1]),
                    "prob_work": float(probs[i, 0]),
                }
            )

df = pd.DataFrame(rows)
df.to_csv("ijrs_taste_shocks_reference.csv", index=False)
