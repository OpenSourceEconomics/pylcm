---
title: Epstein–Zin Lifecycle
---

# Epstein–Zin Lifecycle

A two-regime lifecycle savings model with health-dependent mortality and Epstein–Zin
preferences. The agent saves over 20 annual periods (ages 60–79), faces a two-state
health Markov chain, and transitions stochastically into a terminal `dead` regime whose
value is a bequest function. The model is an example consumer block in the spirit of
Atal, Fang, Karlsson & Ziebarth (2025, *JPE* 133(6), doi:10.1086/734781) — their richer
model additionally includes insurance, long-term care, and out-of-pocket medical
spending — with the expected-utility recursion replaced by Epstein–Zin.

[View source on GitHub](https://github.com/OpenSourceEconomics/pylcm/blob/main/src/lcm_examples/epstein_zin.py)

## The recursion and how it maps onto pylcm

The Epstein–Zin recursion separates the elasticity of intertemporal substitution from
risk aversion. With current consumption $c_t$ and continuation value $V_{t+1}$:

```{math}
V_t = \Bigl[(1-\beta)\,c_t^{\rho} + \beta\,\mathrm{CE}_t^{\rho}\Bigr]^{1/\rho},
\qquad
\mathrm{CE}_t = \Bigl(\mathbb{E}_t\bigl[V_{t+1}^{\,1-\gamma}\bigr]\Bigr)^{1/(1-\gamma)}
```

where $\beta$ is the discount factor, $\rho = 1 - 1/\psi$ with $\psi$ the elasticity of
intertemporal substitution, and $\gamma$ is the coefficient of relative risk aversion.
When $\gamma = \rho$ the recursion collapses to expected CRRA utility.

The mapping onto pylcm has four parts:

- **`utility`** returns $c_t$ directly — values stay in consumption units, keeping the
  power transform well-defined.
- **`H`** implements the outer Kreps–Porteus aggregator. Its `E_next_V` argument
  receives the certainty equivalent already computed by the engine, so
  `H = ((1 − β) · utility^ρ + β · E_next_V^ρ)^(1/ρ)`.
- **`certainty_equivalent=PowerCertaintyEquivalent()`** implements $\mathrm{CE}_t$ with
  the power transform $g(v) = v^{1-\gamma}$. Its runtime parameter lives at
  `params["alive"]["certainty_equivalent"]["risk_aversion"]`.
- The **expectation** runs jointly over the health Markov chain and the alive/dead
  regime transition:
  $\mathrm{CE} = g^{-1}\!\bigl(\sum_r p_r\,\mathbb{E}_w[g(V'_r)]\bigr)$ with
  $g(v) = v^{1-\gamma}$. The engine handles the two-level expectation automatically
  given the stochastic regime transition and the stochastic health transition in
  `state_transitions`.

## Pitfalls

- **Positivity.** Power transforms require $V' > 0$ everywhere. With $\gamma > 1$, a
  zero continuation yields $0^{1-\gamma} = \infty$, which propagates backward and
  corrupts the entire solution. Keep values in consumption units (`utility = c`, not
  `utility = log(c)`) and give death a strictly positive bequest — here `sqrt(wealth)`
  with wealth grid lower bound 0.0 for the dead regime and consumption/wealth lower
  bound 0.5 for the alive regime, so every reachable wealth is positive.
- **Targets without a value contribution.** A reachable target regime with no states
  contributes $0$ to the transformed sum, which equals $p_r \cdot g(0)$ only when
  $g(0) = 0$ — but $g(0) = \infty$ for $\gamma > 1$. Model death with an explicit wealth
  state and a bequest utility instead of an empty terminal regime.
- **`risk_aversion = 1` (log CE)** is not representable by `PowerCertaintyEquivalent`.
  The power transform degenerates at $\gamma = 1$; use expected utility with log utility
  instead.
- **Solver restriction.** `certainty_equivalent` is supported only with the `GridSearch`
  solver. Passing `certainty_equivalent` together with a DC-EGM solver is rejected at
  model build.

## Run it

```python
import jax.numpy as jnp
import plotly.graph_objects as go
from lcm_examples.epstein_zin import get_model, get_params, EzRegimeId, HealthStatus

model = get_model()
n_subjects = 200

# Solve and simulate for two values of risk aversion
traces = {}
for risk_aversion in [0.5, 5.0]:
    params = get_params(risk_aversion=risk_aversion)
    V_arr = model.solve(params=params, log_level="progress")
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.linspace(1.0, 20.0, n_subjects),
            "health": jnp.full(n_subjects, HealthStatus.good, dtype=jnp.int32),
            "regime_id": jnp.full(n_subjects, EzRegimeId.alive, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=V_arr,
        log_level="progress",
        seed=42,
    )
    df = result.to_dataframe()
    alive_df = df[df["regime_name"] == "alive"]
    traces[risk_aversion] = alive_df.groupby("period")["wealth"].mean()

# Compare mean wealth paths
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=traces[0.5].index,
        y=traces[0.5].values,
        name="γ = 0.5 (risk tolerant)",
        line={"color": "#999999"},
    )
)
fig.add_trace(
    go.Scatter(
        x=traces[5.0].index,
        y=traces[5.0].values,
        name="γ = 5.0 (risk averse)",
        line={"color": "#e63946"},
    )
)
fig.update_layout(
    xaxis_title="Period",
    yaxis_title="Mean wealth (alive subjects)",
    template="plotly_white",
    legend={"x": 0.01, "y": 0.99},
)
fig.show()
```

Risk-averse agents ($\gamma = 5$) accumulate more precautionary wealth early in the
lifecycle to buffer against health shocks and mortality risk; the gap relative to
risk-tolerant agents ($\gamma = 0.5$) widens in mid-life and narrows toward the end as
surviving subjects draw down assets.
